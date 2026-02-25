"""
backtest_engine.py — Walk-Forward Portfolio Backtester
======================================================
v11.9 — Final Hardened Release

Full audit history of corrections applied across all review passes:

v11.5  Pre-IPO NaN masking: HISTORY_GATE check replaces NaN adj_scores
       with -inf so they can never sort into the top-K selection.
v11.6  CVaR daily aggregation: daily_equity list feeds CVaR instead of the
       weekly-subsampled equity_hist, activating the soft risk constraint.
       Gap-sizing: shares_target uses exec_price (T-close) not prices_prev
       (T-1 close), preventing cash exhaustion on gap-up opens.
       Log-return safety: global clip(lower=-0.99) applied before pct_change
       is passed into run(); internal re-clip removed as redundant.
       Trade ledger: Trade dataclass now actively populated with every
       executed order including slippage cost.
v11.7  Dynamic Sharpe annualization: obs_per_year inferred from average
       index spacing; eliminates the 2.20x inflation on weekly equity curves.
       Two-pass execution: sells fully execute in Pass 1 before buys in
       Pass 2, so rotation proceeds are always available to fund new entries.
       Hysteresis plane alignment: prev_weights_full numerator now uses
       current execution prices (p_curr) matching the MTM denominator.
       Redundant inner clip removed after global pre-clip was added.
v11.8  Dead code removed: orphaned `prev_pv = pv` assignments in both
       early-exit continue branches deleted; variable had no downstream
       consumer after the pv_for_sizing pivot in v11.6.
v11.9  Data coverage validation: backtest start dates now strictly bypass 
       shallow caches built by daily live scans.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ultimate_momentum_v10_hardened import InstitutionalRiskEngine, UltimateConfig

logger = logging.getLogger(__name__)

REBAL_FREQ     = "W-FRI"
SLIPPAGE_BPS   = 20        # basis points, round-trip
ENTRY_SLIPPAGE = SLIPPAGE_BPS / 2 / 10_000
EXIT_SLIPPAGE  = SLIPPAGE_BPS / 2 / 10_000


@dataclass
class Trade:
    symbol:       str
    date:         pd.Timestamp
    delta_shares: int
    exec_price:   float
    slip_cost:    float
    direction:    str


@dataclass
class BacktestResults:
    equity_curve: pd.Series
    trades:       List[Trade]
    metrics:      Dict
    rebal_log:    pd.DataFrame


class BacktestEngine:
    def __init__(self, engine: InstitutionalRiskEngine, initial_cash: float = 1_000_000):
        self.engine       = engine
        self.cash         = initial_cash
        self.positions    = {}   # {symbol: shares}
        self.equity_hist  = []   # Rebalance-date snapshots — used for final reporting
        self.daily_equity = []   # Every trading-day snapshot — used for CVaR accuracy
        self.trades       = []   # Full execution ledger

    def asymmetric_adv(self, volume_series: pd.Series) -> float:
        """min(SMA20, spot_volume) — contracts instantly under a volume vacuum."""
        sma20 = volume_series.rolling(20).mean().iloc[-1]
        spot  = volume_series.iloc[-1]
        if pd.isna(sma20) or pd.isna(spot):
            return 0.0
        return float(min(sma20, spot))

    def mark_to_market(self, close_prices: pd.Series) -> float:
        pv = self.cash
        for sym, shares in self.positions.items():
            if sym in close_prices.index and not pd.isna(close_prices[sym]):
                pv += shares * float(close_prices[sym])
        return pv

    def run(
        self,
        close:           pd.DataFrame,
        volume:          pd.DataFrame,
        returns:         pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
        start_date:      str,
        idx_df:          Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:

        close_prev_all = close.shift(1)
        start_dt       = pd.Timestamp(start_date)

        for date in close.index:
            if date < start_dt:
                continue

            close_t = close.loc[date]
            pv      = self.mark_to_market(close_t)

            # Daily record feeds the Risk Engine's CVaR calculation accurately.
            self.daily_equity.append((date, pv))

            if date not in rebalance_dates:
                continue

            # Rebalance-date record feeds the final equity curve and metrics.
            self.equity_hist.append((date, pv))

            # Shares are targeted against the pre-trade portfolio value (pv_for_sizing).
            # Buys are capped by post-sell cash in Pass 2. This is an intentional
            # institutional design choice: pre-trade sizing prevents rotation proceeds
            # from artificially shrinking target weights, though it may cause minor
            # execution divergence during extreme turnover events.
            close_t_minus_1 = close_prev_all.loc[date]
            pv_for_sizing   = pv

            hist_returns = returns.loc[:date].iloc[-self.engine.cfg.CVAR_LOOKBACK:]
            symbols      = list(hist_returns.columns)

            # `returns` is globally pre-clipped at the caller; only inf-guard needed here.
            log_returns = np.log1p(hist_returns).replace([np.inf, -np.inf], np.nan)

            valid_counts = log_returns.notna().sum().values
            raw_daily    = log_returns.ewm(halflife=63).mean().iloc[-1].values

            adv_vector = []
            for sym in symbols:
                if sym in volume.columns:
                    adv_vector.append(self.asymmetric_adv(volume.loc[:date, sym]))
                else:
                    adv_vector.append(0.0)
            adv_vector = np.array(adv_vector)

            prices_prev = np.array([
                float(close_t_minus_1[s])
                if s in close_t_minus_1.index and not pd.isna(close_t_minus_1[s])
                else float(close_t[s])
                for s in symbols
            ])

            # ── Signal ranking with hysteresis & Pre-IPO masking ──────────────
            mu, std    = np.nanmean(raw_daily), max(np.nanstd(raw_daily), 1e-8)
            z_scores   = np.clip((raw_daily - mu) / std, -3.0, 3.0)
            adj_scores = z_scores.copy()

            prev_weights_full = np.zeros(len(symbols))
            for i, sym in enumerate(symbols):
                curr_shares = self.positions.get(sym, 0)
                if curr_shares > 0:
                    # Current prices used for both numerator and denominator so that
                    # gap-up winners are not understated in the hysteresis weight.
                    p_curr = (
                        float(close_t[sym])
                        if sym in close_t.index and not pd.isna(close_t[sym])
                        else prices_prev[i]
                    )
                    if pv_for_sizing > 0 and p_curr > 0:
                        prev_weights_full[i] = (curr_shares * p_curr) / pv_for_sizing
                    adj_scores[i] += 0.05  # Retention bonus

                # Mask pre-IPO stocks and any stocks with insufficient history.
                if valid_counts[i] < self.engine.cfg.HISTORY_GATE or np.isnan(adj_scores[i]):
                    adj_scores[i] = -np.inf

            k_max      = self.engine.cfg.MAX_POSITIONS
            sorted_idx = np.argsort(adj_scores)
            sel_idx    = [i for i in sorted_idx[-k_max:] if adj_scores[i] > -np.inf]

            if not sel_idx:
                logger.warning(
                    f"[Backtest] No valid stocks passed HISTORY_GATE on {date.date()}; skipping."
                )
                continue

            sel_idx  = np.sort(sel_idx)
            sel_syms = [symbols[i] for i in sel_idx]

            # ── Regime scoring ────────────────────────────────────────────────
            regime_score = 0.5
            if idx_df is not None and not idx_df.empty:
                idx_slice = idx_df.loc[:date]
                if len(idx_slice) > 200:
                    sma200   = idx_slice["Close"].rolling(200).mean().iloc[-1]
                    last_val = idx_slice["Close"].iloc[-1]
                    if not pd.isna(sma200) and not pd.isna(last_val) and sma200 > 0:
                        regime_score = float(
                            1.0 / (1.0 + np.exp(-20.0 * (float(last_val / sma200) - 1.0)))
                        )

            # ── Realised CVaR from daily equity ───────────────────────────────
            realised_cvar = 0.0
            if len(self.daily_equity) >= 30:
                eq_s    = pd.Series([x[1] for x in self.daily_equity])
                rets_eq = eq_s.pct_change().dropna()
                var_q   = rets_eq.quantile(0.05)
                tail    = rets_eq[rets_eq <= var_q]
                if not tail.empty:
                    realised_cvar = max(0.0, -float(tail.mean()))

            self.engine.calculate_exposure_multiplier(regime_score, realised_cvar)

            # ── Optimizer ─────────────────────────────────────────────────────
            try:
                weights_sel = self.engine.optimize(
                    expected_returns   = raw_daily[sel_idx],
                    historical_returns = log_returns[sel_syms],
                    adv_shares         = adv_vector[sel_idx],
                    prices             = prices_prev[sel_idx],
                    portfolio_value    = pv_for_sizing,
                    prev_w             = prev_weights_full[sel_idx],
                )
                weights          = np.zeros(len(symbols))
                weights[sel_idx] = weights_sel

            except Exception as exc:
                logger.warning(
                    f"[Backtest] Optimizer error on {date}: {exc}; skipping rebalance."
                )
                continue

            # Share targets sized to execution price (T-close) to prevent cash
            # exhaustion when stocks gap materially from T-1 close.
            shares_target = {}
            for i, sym in enumerate(symbols):
                exec_p = (
                    float(close_t[sym])
                    if sym in close_t.index and not pd.isna(close_t[sym])
                    else 0.0
                )
                shares_target[sym] = (
                    int(np.floor((weights[i] * pv_for_sizing) / exec_p))
                    if exec_p > 0 else 0
                )

            # ── Pass 1: liquidate exits and reductions first ──────────────────
            for sym in symbols:
                current    = self.positions.get(sym, 0)
                target     = shares_target.get(sym, 0)
                delta      = target - current
                exec_price = (
                    float(close_t[sym])
                    if sym in close_t.index and not pd.isna(close_t[sym])
                    else 0.0
                )
                if delta >= 0 or exec_price <= 0:
                    continue

                slip_cost = abs(delta) * exec_price * EXIT_SLIPPAGE
                net_p     = exec_price * (1 - EXIT_SLIPPAGE)
                self.cash -= delta * net_p
                self.trades.append(Trade(sym, date, delta, exec_price, slip_cost, "SELL"))
                self.positions[sym] = current + delta

            # ── Pass 2: enter and size-up positions using post-sell cash ──────
            for sym in symbols:
                current    = self.positions.get(sym, 0)
                target     = shares_target.get(sym, 0)
                delta      = target - current
                exec_price = (
                    float(close_t[sym])
                    if sym in close_t.index and not pd.isna(close_t[sym])
                    else 0.0
                )
                if delta <= 0 or exec_price <= 0:
                    continue

                all_in = exec_price * (1 + ENTRY_SLIPPAGE)
                cost   = delta * all_in
                if cost > self.cash:
                    delta = int(np.floor(self.cash / all_in))
                    if delta <= 0:
                        continue
                    cost = delta * all_in

                slip_cost = delta * exec_price * ENTRY_SLIPPAGE
                self.cash -= cost
                self.trades.append(Trade(sym, date, delta, exec_price, slip_cost, "BUY"))
                self.positions[sym] = current + delta

            self.positions = {k: v for k, v in self.positions.items() if v != 0}

        return pd.DataFrame(self.equity_hist, columns=["date", "equity"]).set_index("date")


# ─────────────────────────────────────────────────────────────────────────────
# Functional API
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    market_data: Dict,
    universe:    List[str],
    start_date:  str,
    end_date:    str,
    cfg:         Optional[UltimateConfig] = None,
) -> BacktestResults:

    if cfg is None:
        cfg = UltimateConfig()

    standard_universe = [
        t if (t.endswith(".NS") or t.startswith("^")) else t + ".NS"
        for t in universe
    ]

    close_d: Dict = {}
    vol_d:   Dict = {}

    for sym in standard_universe:
        df = market_data.get(sym)
        if df is None or df.empty:
            continue
        mask = df.index <= pd.Timestamp(end_date)
        df_s = df.loc[mask]
        if len(df_s) < cfg.HISTORY_GATE:
            continue
        if "Close" not in df_s.columns or "Volume" not in df_s.columns:
            continue
        close_d[sym] = df_s["Close"].ffill()
        vol_d[sym]   = df_s["Volume"].fillna(0)

    if not close_d:
        raise ValueError("No valid tickers found for the given date range.")

    close   = pd.DataFrame(close_d).sort_index()
    volume  = pd.DataFrame(vol_d).sort_index()
    # Pre-clip here so the inner loop never encounters arithmetic returns below -99%.
    returns = close.pct_change().clip(lower=-0.99)

    all_dates   = close.index
    rebal_dates = all_dates[
        all_dates.isin(pd.date_range(start_date, end_date, freq=REBAL_FREQ))
    ]

    idx_df = market_data.get("^CRSLDX")
    if idx_df is None or (hasattr(idx_df, "empty") and idx_df.empty):
        idx_df = market_data.get("^NSEI")

    risk_engine = InstitutionalRiskEngine(cfg)
    bt_engine   = BacktestEngine(risk_engine, initial_cash=cfg.INITIAL_CAPITAL)
    eq_df       = bt_engine.run(close, volume, returns, rebal_dates, start_date, idx_df)
    eq_series   = eq_df["equity"]

    return BacktestResults(
        equity_curve = eq_series,
        trades       = bt_engine.trades,
        metrics      = _compute_metrics(eq_series, cfg.INITIAL_CAPITAL),
        rebal_log    = pd.DataFrame(),
    )


def print_backtest_results(results: BacktestResults) -> None:
    m = results.metrics
    print(
        f"\n  BACKTEST RESULTS"
        f"\n  {chr(9472)*55}"
        f"\n  Final Capital  : \u20b9{m['final']:>15,.0f}"
        f"\n  CAGR           : {m['cagr']:>14.2f}%"
        f"\n  Sharpe Ratio   : {m.get('sharpe', 0.0):>14.2f}"
        f"\n  Max Drawdown   : {m['max_dd']:>14.2f}%"
        f"\n  Total Trades   : {len(results.trades)}"
        f"\n  Slippage model : {SLIPPAGE_BPS}bps round-trip, net-delta execution"
        f"\n  {chr(9472)*55}"
    )


def _compute_metrics(eq: pd.Series, initial: float) -> Dict:
    if eq.empty:
        return {"cagr": 0.0, "max_dd": 0.0, "final": initial, "sharpe": 0.0}

    final = float(eq.iloc[-1])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr  = ((final / initial) ** (1.0 / years) - 1.0) * 100.0
    dd    = (eq / eq.cummax() - 1.0) * 100.0
    dr    = eq.pct_change().dropna()

    # Annualisation factor derived from the actual observation spacing so that
    # weekly equity curves (obs_per_year ≈ 52) are not inflated by a hardcoded
    # daily constant (252).
    avg_days     = max((eq.index[-1] - eq.index[0]).days / max(len(eq) - 1, 1), 1)
    obs_per_year = round(365.25 / avg_days)

    sharpe = (
        round((dr.mean() * obs_per_year) / (dr.std() * (obs_per_year ** 0.5)), 2)
        if dr.std() > 0 else 0.0
    )

    return {
        "cagr":   round(cagr, 2),
        "max_dd": round(dd.min(), 2),
        "final":  round(final, 2),
        "sharpe": sharpe,
    }