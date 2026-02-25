"""
daily_workflow.py — Ultimate Momentum V11 Daily Workflow
=========================================================
v11.9 — Final Hardened Release

Full audit history of corrections applied across all review passes:

v11.5  Memory isolation: main_menu() uses a dictionary of independent
       PortfolioStates so switching between universes does not
       cross-contaminate or liquidate holdings from the other universe.
v11.6  PortfolioState dict migration: weights and shares moved from
       positional numpy arrays to {symbol: value} dicts, eliminating
       index-alignment drift when the universe composition shifts.
       Mark-to-market equity: current_portfolio_value derived from live
       MTM equity (shares × prices + cash) rather than a static anchor.
       Output sort: active holdings displayed in descending weight order.
v11.7  Pre-IPO masking parity: HISTORY_GATE check and NaN-to-(-inf)
       masking added to the live scan ranking block, matching the
       correction applied to the backtest engine in v11.5.
       Early guard ordering: len(sel_idx) == 0 check moved above the
       dependent sel_syms and np.sort calls, guarded via len() before
       any numpy array conversion occurs.
v11.8  SIGNAL_ANNUAL_FACTOR rename: cfg.ANNUAL_FACTOR renamed to
       cfg.SIGNAL_ANNUAL_FACTOR to distinguish signal-plane annualisation
       (252 trading days) from the dynamic Sharpe obs_per_year introduced
       in backtest_engine v11.7.
v11.9  Diagnostic UI: 'Compression Ratio' explicitly renamed to 
       'Budget Utilisation' to clarify situations where the allocated capital
       naturally scales up to the regime elasticity bounds.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from ultimate_momentum_v10_hardened import InstitutionalRiskEngine, UltimateConfig
from universe_manager import fetch_screener_universe, get_nifty500, invalidate_universe_cache
from data_cache import get_cache_summary, invalidate_cache, load_or_fetch
from backtest_engine import run_backtest, print_backtest_results

SCREENER_URL = "https://www.screener.in/screens/3506127/hello/"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


_setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    weights:     Dict[str, float] = field(default_factory=dict)
    shares:      Dict[str, int]   = field(default_factory=dict)
    equity_hist: List[float]      = field(default_factory=list)
    universe:    List[str]        = field(default_factory=list)
    cash:        float            = field(default_factory=lambda: UltimateConfig().INITIAL_CAPITAL)

    def realised_cvar(self) -> float:
        """CVaR at 95% confidence computed from the running MTM equity history."""
        if len(self.equity_hist) < 30:
            return 0.0
        rets  = pd.Series(self.equity_hist).pct_change().dropna()
        var_q = rets.quantile(0.05)
        tail  = rets[rets <= var_q]
        return max(0.0, -float(tail.mean())) if not tail.empty else 0.0


def _run_scan(universe: List[str], state: PortfolioState, label: str) -> PortfolioState:
    cfg    = UltimateConfig()
    engine = InstitutionalRiskEngine(cfg)

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")

    all_syms    = list(
        {t if t.endswith(".NS") else t + ".NS" for t in universe} | {"^NSEI", "^CRSLDX"}
    )
    market_data = load_or_fetch(all_syms, start_date, end_date)

    # ── 1. Regime detection ───────────────────────────────────────────────────
    idx_df = market_data.get("^CRSLDX")
    if idx_df is None or idx_df.empty:
        idx_df = market_data.get("^NSEI")

    regime_score = 0.5
    regime_label = "🟡 Neutral"
    if idx_df is not None and len(idx_df) > 200:
        sma200       = idx_df["Close"].rolling(200).mean().iloc[-1]
        last         = idx_df["Close"].iloc[-1]
        regime_score = float(1.0 / (1.0 + np.exp(-20.0 * (float(last / sma200) - 1.0))))
        regime_label = (
            "🟢 Bull"    if regime_score > 0.6 else
            "🔴 Bear"    if regime_score < 0.4 else
            "🟡 Neutral"
        )

    # ── 2. Data alignment and cleaning ───────────────────────────────────────
    close_d = {}
    for sym in universe:
        ns = sym if sym.endswith(".NS") else sym + ".NS"
        df = market_data.get(ns)
        if df is not None and len(df) >= cfg.HISTORY_GATE:
            close_d[sym] = df["Close"].ffill()

    if not close_d:
        print("  ❌ No valid tickers with sufficient history found.")
        return state

    close     = pd.DataFrame(close_d).sort_index()
    log_rets  = np.log1p(close.pct_change().clip(lower=-0.99)).replace(
        [np.inf, -np.inf], np.nan
    )
    hist_rets = log_rets.dropna(how="all")
    active    = list(close.columns)
    prices    = close.iloc[-1].values.astype(float)

    adv_arr = np.zeros(len(active))
    for i, sym in enumerate(active):
        ns = sym if sym.endswith(".NS") else sym + ".NS"
        df = market_data.get(ns)
        if df is not None and "Volume" in df.columns and len(df) >= 20:
            sma20 = df["Volume"].rolling(20).mean().iloc[-1]
            spot  = df["Volume"].iloc[-1]
            if not pd.isna(sma20) and not pd.isna(spot):
                adv_arr[i] = float(min(sma20, spot))

    # ── 3. Mark-to-market equity via symbol-keyed dict ────────────────────────
    # Shares are stored by ticker name, so universe composition shifts never
    # cause a positional index mismatch.
    active_idx   = {sym: i for i, sym in enumerate(active)}
    mtm_notional = sum(
        count * prices[active_idx[sym]]
        for sym, count in state.shares.items()
        if sym in active_idx
    )
    current_portfolio_value = mtm_notional + state.cash

    # ── 4. Signal ranking with hysteresis and Pre-IPO masking ─────────────────
    valid_counts = hist_rets.notna().sum().values
    raw_daily    = hist_rets.ewm(halflife=63).mean().iloc[-1].values
    raw_annual   = raw_daily * cfg.SIGNAL_ANNUAL_FACTOR

    mu, std   = np.nanmean(raw_daily), max(np.nanstd(raw_daily), 1e-8)
    exp_rets  = np.clip((raw_daily - mu) / std, -3.0, 3.0)
    adj_scores = exp_rets.copy()

    for i, sym in enumerate(active):
        if state.weights.get(sym, 0.0) > 0.001:
            adj_scores[i] += 0.05  # Retention bonus for existing holdings

        # Mask pre-IPO stocks and any stock with insufficient return history.
        if valid_counts[i] < cfg.HISTORY_GATE or np.isnan(adj_scores[i]):
            adj_scores[i] = -np.inf

    k          = cfg.MAX_POSITIONS
    sorted_idx = np.argsort(adj_scores)
    sel_idx    = [i for i in sorted_idx[-k:] if adj_scores[i] > -np.inf]

    # Guard checked via len() while sel_idx is still a plain list.
    if len(sel_idx) == 0:
        logger.warning("No valid tickers survived ranking criteria.")
        return state

    sel_idx  = np.sort(sel_idx)
    sel_syms = [active[i] for i in sel_idx]

    # ── 5. Optimizer execution ────────────────────────────────────────────────
    prev_w_slice = np.array([state.weights.get(sym, 0.0) for sym in sel_syms])
    engine.calculate_exposure_multiplier(regime_score, state.realised_cvar())

    weights_sel = engine.optimize(
        expected_returns   = raw_daily[sel_idx],
        historical_returns = hist_rets[sel_syms],
        adv_shares         = adv_arr[sel_idx],
        prices             = prices[sel_idx],
        portfolio_value    = current_portfolio_value,
        prev_w             = prev_w_slice,
    )

    weights          = np.zeros(len(active))
    weights[sel_idx] = weights_sel

    # ── 6. Remap results to symbol-keyed dicts ────────────────────────────────
    new_weights     = {}
    new_shares      = {}
    actual_notional = 0.0

    for i in range(len(active)):
        if weights[i] > 0:
            sym = active[i]
            s   = int(np.floor(weights[i] * current_portfolio_value / max(prices[i], 1e-6)))
            if s > 0:
                new_weights[sym]  = weights[i]
                new_shares[sym]   = s
                actual_notional  += s * prices[i]

    residual_cash = current_portfolio_value - actual_notional
    total_equity  = actual_notional + residual_cash

    state.weights     = new_weights
    state.shares      = new_shares
    state.universe    = active
    state.cash        = residual_cash
    state.equity_hist.append(total_equity)

    # ── 7. Formatted output ───────────────────────────────────────────────────
    diag = engine.last_diag
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    print(f"  Regime: {regime_score:.2f} {regime_label} | CVaR: {state.realised_cvar():.2%}")
    if diag:
        print(f"\n  GOVERNANCE & RISK AUDIT")
        print(f"  {'─'*30}")
        print(f"  Solver Status     : {diag.status}")
        print(f"  Regime Intent (γ) : {diag.gamma_intent:.2f}")
        print(f"  Actual Exposure   : {diag.actual_weight:.3f}")
        print(f"  Budget Utilisation: {diag.budget_utilisation:.1%}")
        print(f"  ADV Saturation    : {diag.adv_binding_count} stocks at hard limit")

    print(f"\n  {'Stock (Price)':<22} {'Weight':>7}  {'Shares':>8}  {'Notional':>12}  {'Exp Ret'}")
    print(f"  {'─'*68}")

    display_rows = []
    for sym, count in state.shares.items():
        i            = active_idx[sym]
        display_name = f"{sym} ({prices[i]:,.1f})"
        notional     = count * prices[i]
        exp_ret      = raw_annual[i] * 100
        display_rows.append((state.weights[sym], display_name, count, notional, exp_ret))

    display_rows.sort(key=lambda x: x[0], reverse=True)

    for w_val, display_name, s_val, notional, exp_ret in display_rows:
        print(
            f"  {display_name:<22} {w_val:>7.2%}  {s_val:>8,}  "
            f"₹{notional:>10,.0f}  {exp_ret:>7.1f}%"
        )

    print(f"  {'─'*68}")
    print(f"  OPTIMAL TOTAL INVESTMENT : ₹{actual_notional:,.0f}")
    print(f"  RESIDUAL CASH            : ₹{residual_cash:,.0f}")
    print(f"  TOTAL MTM EQUITY         : ₹{total_equity:,.0f}")
    print(f"  (Reference Anchor: ₹{cfg.INITIAL_CAPITAL:,.0f})")
    print(f"{'='*72}\n")
    return state


def main_menu():
    # Independent state objects prevent cross-contamination between universes.
    states = {
        "screener": PortfolioState(),
        "nifty":    PortfolioState(),
    }

    while True:
        print(f"\n{'═'*62}")
        print(f"  ULTIMATE MOMENTUM V11 — DAILY WORKFLOW")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'═'*62}")
        print("  [1] Screener Scan  [2] Nifty 500 Scan  [3] Backtest  [4] Status  [5] Clear  [q] Quit")

        c = input("\n  Choice: ").strip().lower()

        if c == "1":
            states["screener"] = _run_scan(
                fetch_screener_universe(SCREENER_URL), states["screener"], "SCREENER.IN SCAN"
            )
        elif c == "2":
            states["nifty"] = _run_scan(
                get_nifty500(), states["nifty"], "NIFTY 500 SCAN"
            )
        elif c == "3":
            universe = get_nifty500()
            start    = input("  Start (YYYY-MM-DD): ")
            end      = datetime.today().strftime("%Y-%m-%d")
            data     = load_or_fetch(universe + ["^NSEI", "^CRSLDX"], start, end)
            print_backtest_results(run_backtest(data, universe, start, end))
        elif c == "4":
            summary = get_cache_summary()
            print(summary.to_string(index=False) if not summary.empty else "  Cache is empty.")
        elif c == "5":
            invalidate_cache()
            invalidate_universe_cache()
            states = {"screener": PortfolioState(), "nifty": PortfolioState()}
            print("  ✅ Cache, Universe, and Portfolio States purged.")
        elif c == "q":
            break


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n  👋 Goodbye!")
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")