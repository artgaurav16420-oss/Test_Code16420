"""
backtest_engine.py — Deterministic Walk-Forward Engine
=======================================================
Weekly rebalance cadence with full equity ledger, CVaR risk management,
and sector-diversified portfolio construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from momentum_engine import (
    InstitutionalRiskEngine,
    UltimateConfig,
    OptimizationError,
    OptimizationErrorType,
    PortfolioState,
    execute_rebalance,
    Trade,
    to_ns,
)
from signals import generate_signals, compute_regime_score, compute_single_adv

logger = logging.getLogger(__name__)


# ─── Results container ────────────────────────────────────────────────────────

@dataclass
class BacktestResults:
    equity_curve: pd.Series
    trades:       List[Trade]
    metrics:      Dict
    rebal_log:    pd.DataFrame   # one row per rebalance date


# ─── Engine ───────────────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, engine: InstitutionalRiskEngine, initial_cash: float = 1_000_000):
        self.engine              = engine
        self.state               = PortfolioState(cash=initial_cash)
        self.state.equity_hist_cap = engine.cfg.EQUITY_HIST_CAP
        self.trades:  List[Trade]  = []
        self._eq_dates: list       = []
        self._eq_vals:  list       = []
        self._rebal_rows: list     = []   # accumulated rebalance log rows

    def run(
        self,
        close:           pd.DataFrame,
        volume:          pd.DataFrame,
        returns:         pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
        start_date:      str,
        end_date:        Optional[str] = None,
        idx_df:          Optional[pd.DataFrame] = None,
        sector_map:      Optional[dict]         = None,
    ) -> pd.DataFrame:
        start_dt = pd.Timestamp(start_date)
        end_dt   = pd.Timestamp(end_date) if end_date else close.index[-1]
        symbols  = list(close.columns)

        # O(1) membership tests for rapid daily simulation
        rebal_set = set(rebalance_dates)

        for date in close.index:
            if date < start_dt or date > end_dt:
                continue

            close_t  = close.loc[date]
            prices_t = close_t.values.astype(float)

            active_idx = {sym: i for i, sym in enumerate(symbols)}
            pv = self.state.cash + sum(
                self.state.shares.get(sym, 0) * (
                    float(close_t[sym])
                    if (sym in active_idx and pd.notna(close_t[sym]))
                    else self.state.last_known_prices.get(sym, 0.0)
                )
                for sym in self.state.shares
            )

            if date in rebal_set:
                self._run_rebalance(
                    date, close, volume, returns, symbols, prices_t,
                    pv, idx_df, sector_map,
                )

            price_dict = {
                sym: prices_t[active_idx[sym]]
                for sym in symbols
                if pd.notna(close_t[sym])
            }
            self.state.record_eod(price_dict)
            post_pv = self.state.equity_hist[-1] if self.state.equity_hist else pv

            self._eq_dates.append(date)
            self._eq_vals.append(post_pv)

        return pd.DataFrame({"equity": pd.Series(self._eq_vals, index=self._eq_dates)})

    def _run_rebalance(
        self,
        date:       pd.Timestamp,
        close:      pd.DataFrame,
        volume:     pd.DataFrame,
        returns:    pd.DataFrame,
        symbols:    List[str],
        prices_t:   np.ndarray,
        pv:         float,
        idx_df,
        sector_map,
    ) -> None:
        cfg = self.engine.cfg

        # T-1 history (exclude today to prevent look-ahead).
        hist_log_rets = (
            np.log1p(returns.loc[:date].iloc[:-1])
            .replace([np.inf, -np.inf], np.nan)
        )

        adv_vector = _build_adv_vector(symbols, volume, date)

        # Use T-1 prices from state cache — NOT today's execution price —
        # to compute current weights, preventing subtle look-ahead bias.
        prev_w_dict = {
            sym: (self.state.shares.get(sym, 0) * px) / pv
            for sym in symbols
            if self.state.shares.get(sym, 0) > 0 and pv > 0
            for px in [self.state.last_known_prices.get(sym, 0.0)]
            if px is not None and np.isfinite(px) and px > 0
        }

        raw_daily, adj_scores, sel_idx = generate_signals(
            hist_log_rets,
            adv_vector,
            cfg,
            prev_weights=prev_w_dict,
        )

        _idx_ok      = idx_df is not None and not (hasattr(idx_df, "empty") and idx_df.empty)
        idx_slice    = idx_df.loc[:date].iloc[:-1] if _idx_ok else None
        regime_score = compute_regime_score(idx_slice)

        # Guard: only use realised CVaR once enough history has accumulated.
        if len(self.state.equity_hist) >= cfg.CVAR_MIN_HISTORY:
            realised_cvar = self.state.realised_cvar(min_obs=cfg.CVAR_MIN_HISTORY)
        else:
            realised_cvar = 0.0

        gross_exposure = sum(
            self.state.shares.get(sym, 0) * (
                float(close.loc[date, sym])
                if pd.notna(close.loc[date, sym])
                else self.state.last_known_prices.get(sym, 0.0)
            )
            for sym in self.state.shares
            if sym in symbols
        ) / max(pv, 1e-6)
        
        self.state.update_exposure(regime_score, realised_cvar, cfg, gross_exposure=gross_exposure)

        target_weights         = np.zeros(len(symbols))
        apply_decay            = False
        optimization_succeeded = False

        if sel_idx:
            sel_syms      = [symbols[i] for i in sel_idx]
            sector_labels = _build_sector_labels(sel_syms, sector_map)
            prev_weights  = np.array([prev_w_dict.get(sym, 0.0) for sym in symbols])

            try:
                weights_sel = self.engine.optimize(
                    expected_returns    = raw_daily[sel_idx],
                    historical_returns  = hist_log_rets[[symbols[i] for i in sel_idx]],
                    adv_shares          = adv_vector[sel_idx],
                    prices              = prices_t[sel_idx],
                    portfolio_value     = pv,
                    prev_w              = prev_weights[sel_idx],
                    exposure_multiplier = self.state.exposure_multiplier,
                    sector_labels       = sector_labels,
                )
                target_weights[sel_idx]  = weights_sel
                self.state.consecutive_failures = 0
                self.state.decay_rounds  = 0
                optimization_succeeded   = True

            except OptimizationError as oe:
                if oe.error_type != OptimizationErrorType.DATA:
                    self.state.consecutive_failures += 1
                    logger.debug(
                        "[Backtest] Solver failure #%d on %s: %s",
                        self.state.consecutive_failures, date, oe,
                    )
                    if self.state.consecutive_failures >= 2:
                        logger.debug(
                            "[Backtest] Applying %.0f%% deleverage on %s.",
                            (1 - cfg.DECAY_FACTOR) * 100, date,
                        )
                        apply_decay = True

        if optimization_succeeded or apply_decay:
            execute_rebalance(
                self.state, target_weights, prices_t, symbols, cfg,
                date_context=date, trade_log=self.trades, apply_decay=apply_decay,
            )
            self._rebal_rows.append({
                "date":               date,
                "regime_score":       round(regime_score, 4),
                "realised_cvar":      round(realised_cvar, 6),
                "exposure_multiplier":round(self.state.exposure_multiplier, 4),
                "override_active":    self.state.override_active,
                "n_positions":        len(self.state.shares),
                "apply_decay":        apply_decay,
            })


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_adv_vector(symbols: List[str], volume: pd.DataFrame, date: pd.Timestamp) -> np.ndarray:
    """
    Build the ADV (average daily volume) vector for sizing constraints.
    Applies the T-1 slice (iloc[:-1]) before delegating to compute_single_adv.
    """
    adv = []
    for sym in symbols:
        if sym in volume.columns:
            try:
                series = volume.loc[:date, sym].iloc[:-1]
                adv.append(compute_single_adv(series))
            except Exception:
                adv.append(0.0)
        else:
            adv.append(0.0)
    return np.array(adv, dtype=float)


def _build_sector_labels(sel_syms: List[str], sector_map: Optional[dict]) -> Optional[np.ndarray]:
    if not sector_map:
        return None
    unique_sectors = sorted(set(sector_map.get(s, "Unknown") for s in sel_syms))
    sec_idx        = {s: i for i, s in enumerate(unique_sectors)}
    return np.array([sec_idx[sector_map.get(sym, "Unknown")] for sym in sel_syms], dtype=int)


# ─── Public API ───────────────────────────────────────────────────────────────

def run_backtest(
    market_data: dict,
    universe:    List[str],
    start_date:  str,
    end_date:    str,
    cfg:         Optional[UltimateConfig] = None,
    sector_map:  Optional[dict]           = None,
) -> BacktestResults:
    if cfg is None:
        cfg = UltimateConfig()

    close_d  = {}
    volume_d = {}
    for sym in universe:
        if not sym:
            continue
        key = sym if sym.endswith(".NS") else sym + ".NS"
        if key not in market_data:
            continue
        close_d[sym]  = market_data[key]["Close"].ffill()
        volume_d[sym] = market_data[key]["Volume"]

    if not close_d:
        raise ValueError("No valid symbols found in market_data for the given universe.")

    close   = pd.DataFrame(close_d).sort_index()
    volume  = pd.DataFrame(volume_d).sort_index()
    returns = close.pct_change(fill_method=None).clip(lower=-0.99)

    # HIGH-INTEGRITY ENHANCEMENT: Instead of hardcoded weekly intervals that blindly 
    # skip holiday Fridays, map all target rebalance dates to the nearest preceding 
    # valid trading day within the actual market data index.
    all_target_dates = pd.date_range(start_date, end_date, freq=cfg.REBALANCE_FREQ)
    valid_rebal_dates = close.index.asof(all_target_dates).dropna().unique()
    rebal_dates = pd.DatetimeIndex(valid_rebal_dates)

    idx_df = market_data.get("^CRSLDX")
    if idx_df is None or idx_df.empty:
        idx_df = market_data.get("^NSEI")

    engine = InstitutionalRiskEngine(cfg)
    bt     = BacktestEngine(engine, initial_cash=cfg.INITIAL_CAPITAL)
    bt.run(close, volume, returns, rebal_dates, start_date, end_date=end_date, idx_df=idx_df, sector_map=sector_map)

    eq_daily  = pd.Series(bt._eq_vals, index=bt._eq_dates)
    eq_weekly = eq_daily[eq_daily.index.isin(rebal_dates)]

    if eq_weekly.empty and not eq_daily.empty:
        logger.warning(
            "[Backtest] No rebalance dates align with equity curve index. "
            "Defaulting to daily series for metrics."
        )
        eq_weekly = eq_daily

    rebal_log = (
        pd.DataFrame(bt._rebal_rows).set_index("date")
        if bt._rebal_rows
        else pd.DataFrame()
    )

    return BacktestResults(
        equity_curve = eq_weekly,
        trades       = bt.trades,
        metrics      = _compute_metrics(eq_daily, cfg.INITIAL_CAPITAL),
        rebal_log    = rebal_log,
    )


def print_backtest_results(results: BacktestResults) -> None:
    m = results.metrics
    print(f"\n  \033[1;36mBACKTEST RESULTS\033[0m")
    print(f"  \033[90m{chr(9472)*65}\033[0m")
    print(
        f"  \033[1mFinal:\033[0m \033[32m₹{m['final']:,.0f}\033[0m  "
        f"\033[1mCAGR:\033[0m {m['cagr']:.2f}%  "
        f"\033[1mSharpe:\033[0m {m['sharpe']:.2f}  "
        f"\033[1mSortino:\033[0m {m['sortino']:.2f}  "
        f"\033[1mMaxDD:\033[0m {m['max_dd']:.2f}%  "
        f"\033[1mCalmar:\033[0m {m['calmar']:.2f}"
    )
    print(f"  \033[90m{chr(9472)*65}\033[0m\n")


def _compute_metrics(eq: pd.Series, initial: float) -> Dict:
    if eq.empty:
        return {"cagr": 0.0, "max_dd": 0.0, "final": initial, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0}

    final  = float(eq.iloc[-1])
    span   = (eq.index[-1] - eq.index[0]).days
    years  = max(span / 365.25, 0.1)
    cagr   = ((final / initial) ** (1.0 / years) - 1.0) * 100.0
    dd     = (eq / eq.cummax() - 1.0) * 100.0
    max_dd = float(dd.min())

    dr = eq.pct_change(fill_method=None).dropna()
    if len(dr) > 1 and dr.std() > 0:
        avg_days_between = span / len(dr)
        periods_per_year = 365.25 / avg_days_between
        sharpe = (dr.mean() * periods_per_year) / (dr.std() * np.sqrt(periods_per_year))
        
        downside = dr[dr < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (dr.mean() * periods_per_year) / (downside.std() * np.sqrt(periods_per_year))
        else:
            sortino = sharpe  # Fallback if no downside volatility exists
    else:
        sharpe  = 0.0
        sortino = 0.0

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "cagr":    round(cagr,    2),
        "max_dd":  round(max_dd,  2),
        "final":   round(final,   2),
        "sharpe":  round(sharpe,  2),
        "sortino": round(sortino, 2),
        "calmar":  round(calmar,  2),
    }