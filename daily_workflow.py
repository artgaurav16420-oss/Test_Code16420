"""
daily_workflow.py — Ultimate Momentum — Daily Workflow
=======================================================
Interactive CLI for live scanning, status display, and backtesting.
Features robust capital management and direct Screener.in web scraping.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import numpy as np
import pandas as pd

from momentum_engine import (
    InstitutionalRiskEngine,
    UltimateConfig,
    OptimizationError,
    OptimizationErrorType,
    PortfolioState,
    execute_rebalance,
    to_ns,
    to_bare,
    Trade,
)
from universe_manager import (
    fetch_nse_equity_universe,
    get_nifty500,
    get_sector_map,
    invalidate_universe_cache,
)
from data_cache import get_cache_summary, invalidate_cache, load_or_fetch
from backtest_engine import run_backtest, print_backtest_results
from signals import generate_signals, compute_adv, compute_regime_score

__version__ = "11.42"

# ─── ANSI colour palette ─────────────────────────────────────────────────────

class C:
    BLU   = "\033[34m"
    CYN   = "\033[36m"
    GRN   = "\033[32m"
    YLW   = "\033[33m"
    RED   = "\033[31m"
    GRY   = "\033[90m"
    RST   = "\033[0m"
    BLD   = "\033[1m"
    B_CYN = "\033[1;36m"
    B_GRN = "\033[1;32m"
    B_RED = "\033[1;31m"


logger = logging.getLogger(__name__)


def _render_meter(label: str, progress: float, width: int = 30) -> str:
    """Build a professional text meter to show long-running stage progress."""
    clipped = max(0.0, min(1.0, progress))
    filled = int(round(width * clipped))
    bar = f"{'█' * filled}{'░' * (width - filled)}"
    pct = f"{clipped * 100:5.1f}%"
    return f"  {C.CYN}{label:<18}{C.RST} [{bar}] {C.BLD}{pct}{C.RST}"


def _print_stage_status(label: str, progress: float, detail: str) -> None:
    """Print stage meter and contextual status text for user visibility."""
    print(_render_meter(label, progress))
    print(f"  {C.GRY}{detail}{C.RST}")


# ─── Screener.in Scraper & Prompters ─────────────────────────────────────────

def _scrape_screener(base_url: str) -> List[str]:
    """Handles pagination to scrape all tickers from a public Screener.in URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
        import re
    except ImportError:
        print(f"\n  {C.RED}[!] Missing dependencies for web scraping.{C.RST}")
        print(f"  {C.GRY}Please run: pip install requests beautifulsoup4{C.RST}\n")
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    symbols = set()
    page = 1
    
    # HIGH-INTEGRITY FIX: Use proper URL parsing to remove only 'page' and preserve other screener filters
    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs.pop('page', None)
    clean_url = urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))
    
    while True:
        sep = "&" if "?" in clean_url else "?"
        url = f"{clean_url}{sep}page={page}"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
        except requests.RequestException as e:
            logger.error("[Screener] Network error while reaching Screener.in: %s", e)
            break

        if resp.status_code in (401, 403):
            print(f"\n  {C.RED}[!] Screener.in denied access (HTTP {resp.status_code}).{C.RST}")
            print(f"  {C.GRY}Verify the screen is marked Public at:{C.RST}")
            print(f"  {C.GRY}screener.in → Your Screen → Edit → Visibility: Public{C.RST}\n")
            break
        elif resp.status_code != 200:
            break
            
        soup = BeautifulSoup(resp.text, 'html.parser')
        links = soup.find_all('a', href=re.compile(r'^/company/[^/]+/(?:consolidated/)?$'))
        
        page_symbols = 0
        for link in links:
            match = re.search(r'/company/([^/]+)/', link['href'])
            if match:
                sym = match.group(1).upper()
                symbols.add(sym)
                page_symbols += 1
                
        # Break loop if we hit a page with no new symbols
        if page_symbols == 0:
            break
            
        page += 1
        
    return list(symbols)


def _get_custom_universe() -> List[str]:
    """Automatically gets universe from Screener.in URL or local fallback."""
    DEFAULT_URL = "https://www.screener.in/screens/3506127/hello/"
    url_file = "data/screener_url.txt"
    
    saved_url = DEFAULT_URL
    if os.path.exists(url_file):
        with open(url_file, "r") as f:
            content = f.read().strip()
            if content:
                saved_url = content
    else:
        # Save the default URL so it can be manually edited later if needed
        os.makedirs("data", exist_ok=True)
        with open(url_file, "w") as f:
            f.write(DEFAULT_URL)
            
    print(f"\n  {C.B_CYN}── Custom Screener Integration ──{C.RST}")
    logger.info("[Screener] Fetching universe from: %s", saved_url)
    
    tickers = _scrape_screener(saved_url)
    if tickers:
        return tickers
        
    logger.warning("[Screener] Scraping failed or returned 0 tickers. Attempting local file fallback...")

    # Fallback to local files
    files = ["custom_screener.csv", "custom_screener.txt"]
    for f in files:
        if os.path.exists(f):
            try:
                with open(f, "r") as file:
                    content = file.read().replace(",", "\n")
                    tickers = [line.strip().upper() for line in content.split("\n") if line.strip()]
                    tickers = [t for t in tickers if t not in ("SYMBOL", "TICKER", "")]
                    return list(set(tickers))
            except Exception as e:
                logger.error("[Screener] Failed to read %s: %s", f, e)
    return []


def _check_and_prompt_initial_capital(state: PortfolioState, label: str, name: str) -> None:
    """Prompts for real-world capital if the portfolio is brand new."""
    if not state.shares and not state.equity_hist and abs(state.cash - 1_000_000.0) < 1.0:
        print(f"\n  {C.YLW}⚡ New portfolio detected for {label}{C.RST}")
        try:
            raw_cap = input(f"  {C.CYN}Enter your starting capital (₹) [Default 10,00,000]: {C.RST}").replace(",", "").strip()
            if raw_cap:
                cap = float(raw_cap)
                if cap > 0:
                    state.cash = cap
                    save_portfolio_state(state, name)
                    print(f"  {C.GRN}[+] Initial capital set to ₹{cap:,.2f}{C.RST}\n")
        except ValueError:
            print(f"  {C.RED}Invalid input. Using default ₹10,00,000.{C.RST}\n")


# ─── Corporate action / split detection ──────────────────────────────────────

_SPLIT_RATIOS = [2, 5, 10, 3, 4, 20, 0.5, 0.2]
_SPLIT_TOLERANCE = 0.04   


def detect_and_apply_splits(state: PortfolioState, market_data: dict) -> List[str]:
    adjusted: List[str] = []
    for sym in list(state.shares.keys()):
        ns = to_ns(sym)
        if ns not in market_data or market_data[ns].empty:
            continue
        current_price = float(market_data[ns]["Close"].iloc[-1])
        if not np.isfinite(current_price) or current_price <= 0:
            continue
        last_price = state.last_known_prices.get(sym)
        if last_price is None or last_price <= 0:
            continue

        ratio = last_price / current_price

        for r in _SPLIT_RATIOS:
            if abs(ratio - r) / r <= _SPLIT_TOLERANCE:
                old_shares     = state.shares[sym]
                theoretical_new_shares = old_shares * r
                # Exchanges do not round split allotments upward.
                # Any fractional entitlement is cash-settled by the broker/registrar.
                new_shares     = int(np.floor(theoretical_new_shares + 1e-12))
                old_entry      = state.entry_prices.get(sym, current_price * r)
                new_entry      = old_entry / r

                # HIGH-INTEGRITY FIX: Fractional post-split shares are settled in cash.
                fractional_shares = max(0.0, theoretical_new_shares - new_shares)
                fractional_value = fractional_shares * current_price
                state.cash = round(state.cash + fractional_value, 10)

                logger.warning(
                    "SPLIT DETECTED: %s  ratio=%.3f (≈%gx)  "
                    "shares %d→%d  entry_price ₹%.2f→₹%.2f",
                    sym, ratio, r, old_shares, new_shares, old_entry, new_entry,
                )
                state.shares[sym]       = new_shares
                state.entry_prices[sym] = round(new_entry, 4)
                state.last_known_prices[sym] = current_price
                adjusted.append(sym)
                break

    return adjusted


# ─── State persistence ────────────────────────────────────────────────────────

def save_portfolio_state(state: PortfolioState, name: str) -> None:
    os.makedirs("data", exist_ok=True)
    state_file = f"data/portfolio_state_{name}.json"
    tmp_file   = f"{state_file}.tmp"
    try:
        for i in range(1, -1, -1):
            src, dst = f"{state_file}.bak.{i}", f"{state_file}.bak.{i+1}"
            if os.path.exists(src):
                shutil.copy2(src, dst)
        if os.path.exists(state_file):
            shutil.copy2(state_file, f"{state_file}.bak.0")
        with open(tmp_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, state_file)
        if os.name == "posix":
            dir_fd = os.open("data", os.O_DIRECTORY)
            os.fsync(dir_fd)
            os.close(dir_fd)
    except Exception as exc:
        logger.error("Durable save failed for '%s': %s", name, exc)


def load_portfolio_state(name: str) -> PortfolioState:
    state_file = f"data/portfolio_state_{name}.json"
    backups    = [state_file] + [f"{state_file}.bak.{i}" for i in range(3)]
    for path in backups:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return PortfolioState.from_dict(json.load(f))
            except Exception as exc:
                logger.warning("Corrupted state at %s: %s", path, exc)
    return PortfolioState()


# ─── Core scan logic ──────────────────────────────────────────────────────────

def _run_scan(
    universe: List[str],
    state:    PortfolioState,
    label:    str,
    cfg_override: Optional[UltimateConfig] = None,
) -> tuple[PortfolioState, dict]:
    scan_started_at = time.perf_counter()
    _print_stage_status("Download", 0.05, f"Preparing {len(universe):,} symbols for {label}...")

    cfg    = cfg_override if cfg_override else UltimateConfig()
    engine = InstitutionalRiskEngine(cfg)
    state.equity_hist_cap = cfg.EQUITY_HIST_CAP

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")

    all_syms   = list({to_ns(t) for t in universe} | {"^NSEI", "^CRSLDX"})
    _print_stage_status(
        "Download",
        0.35,
        f"Fetching/caching OHLCV data ({start_date} → {end_date}) for {len(all_syms):,} instruments...",
    )
    market_data = load_or_fetch(all_syms, start_date, end_date, cfg=cfg)
    _print_stage_status("Download", 1.0, f"Data ready. Starting iteration and signal analysis for {label}.")
    _print_stage_status("Analysis", 0.10, "Normalizing market snapshots and benchmark regime inputs...")

    idx_df = market_data.get("^CRSLDX")
    if idx_df is None or idx_df.empty:
        idx_df = market_data.get("^NSEI")
        
    idx_slice   = idx_df.iloc[:-1] if idx_df is not None and not idx_df.empty else None
    regime_score = compute_regime_score(idx_slice)

    close_d: Dict[str, pd.Series] = {}
    for sym in universe:
        ns = to_ns(sym)
        if ns in market_data:
            close_d[to_bare(ns)] = market_data[ns]["Close"].ffill()

    if not close_d:
        logger.warning("[Scan] No data available for any universe symbol.")
        return state, market_data

    _print_stage_status("Analysis", 0.35, f"Built close-price matrix for {len(close_d):,} active symbols.")

    split_syms = detect_and_apply_splits(state, market_data)
    if split_syms:
        logger.warning("[Scan] Applied split adjustments for: %s", split_syms)

    close    = pd.DataFrame(close_d).sort_index()
    active   = list(close.columns)          
    prices   = close.iloc[-1].values.astype(float)
    active_idx = {sym: i for i, sym in enumerate(active)}

    mtm_notional = sum(
        state.shares.get(sym, 0) * prices[active_idx[sym]]
        for sym in state.shares
        if sym in active_idx
    )
    pv = mtm_notional + state.cash

    close_hist    = close.iloc[:-1]   
    log_rets      = np.log1p(close_hist.pct_change(fill_method=None).clip(lower=-0.99)).replace([np.inf, -np.inf], np.nan)
    adv_arr       = compute_adv(market_data, active)  
    prev_w_arr    = np.array([state.weights.get(sym, 0.0) for sym in active])
    _print_stage_status("Analysis", 0.55, "Running momentum iterations, liquidity filters, and risk gates...")

    gross_exposure = mtm_notional / pv if pv > 0 else 1.0
    state.update_exposure(regime_score, state.realised_cvar(), cfg, gross_exposure=gross_exposure)

    weights              = np.zeros(len(active))
    apply_decay          = False
    optimization_succeeded = False
    total_slippage       = 0.0
    trade_log: List[Trade] = []

    try:
        raw_daily, adj_scores, sel_idx = generate_signals(
            log_rets, adv_arr, cfg.HISTORY_GATE, cfg.MAX_POSITIONS,
            prev_weights=state.weights,   
            halflife_fast=cfg.HALFLIFE_FAST,
            halflife_slow=cfg.HALFLIFE_SLOW,
        )
        if not sel_idx:
            raise OptimizationError("No valid universe candidates.", OptimizationErrorType.DATA)

        sel_syms      = [active[i] for i in sel_idx]
        sector_map    = get_sector_map(sel_syms, cfg=cfg)
        unique_sectors = sorted(set(sector_map.values()))
        sec_idx        = {s: i for i, s in enumerate(unique_sectors)}
        sector_labels  = np.array([sec_idx[sector_map[sym]] for sym in sel_syms], dtype=int)

        weights_sel = engine.optimize(
            expected_returns    = raw_daily[sel_idx],
            historical_returns  = log_rets[[active[i] for i in sel_idx]],
            adv_shares          = adv_arr[sel_idx],
            prices              = prices[sel_idx],
            portfolio_value     = pv,
            prev_w              = prev_w_arr[sel_idx],
            exposure_multiplier = state.exposure_multiplier,
            sector_labels       = sector_labels,
        )
        weights[sel_idx]               = weights_sel
        state.consecutive_failures     = 0
        state.decay_rounds             = 0   
        optimization_succeeded         = True

    except OptimizationError as exc:
        if exc.error_type != OptimizationErrorType.DATA:
            state.consecutive_failures += 1
            logger.error("Solver failure #%d: %s. Freezing state.", state.consecutive_failures, exc)
            if state.consecutive_failures >= 2:
                logger.warning(
                    "Regime decay: forcing -%.0f%% exposure reduction.",
                    (1 - cfg.DECAY_FACTOR) * 100,
                )
                apply_decay = True
        else:
            logger.error("Data error (not escalated): %s. Freezing state.", exc)

    if optimization_succeeded or apply_decay:
        total_slippage = execute_rebalance(
            state, weights, prices, active, cfg,
            date_context=pd.Timestamp(end_date), trade_log=trade_log, apply_decay=apply_decay,
        )

    _print_stage_status("Analysis", 0.85, "Applying rebalance decisions and updating portfolio marks...")

    price_dict = {sym: prices[active_idx[sym]] for sym in active}
    state.record_eod(price_dict)
    final_pv = state.equity_hist[-1] if state.equity_hist else pv

    logger.info(
        "%s%s%s | Regime: %.2f | CVaR: %.2f%% | Failures: %d | "
        "Equity: %s₹%s%s | Slippage: %s₹%s%s",
        C.BLU, label, C.RST,
        regime_score,
        state.realised_cvar() * 100,
        state.consecutive_failures,
        C.GRN, f"{final_pv:,.0f}", C.RST,
        C.RED, f"{total_slippage:,.0f}", C.RST,
    )

    elapsed = time.perf_counter() - scan_started_at
    _print_stage_status("Analysis", 1.0, f"{label} completed in {elapsed:.1f}s.")
    
    # ── Print Action Sheet sorted by weight for manual execution ──
    if trade_log:
        print(f"\n  {C.B_CYN}EXECUTION ACTION SHEET (Manual Targets){C.RST}")
        print(f"  {C.GRY}{'─' * 66}{C.RST}")
        
        sorted_trades = sorted(trade_log, key=lambda t: state.weights.get(t.symbol, 0.0), reverse=True)
        
        for t in sorted_trades:
            action_color = C.B_GRN if t.direction == "BUY" else C.B_RED
            target_weight = state.weights.get(t.symbol, 0.0)
            print(
                f"  {action_color}{t.direction:<4}{C.RST} | {C.BLD}{t.symbol:<12}{C.RST} | "
                f"{abs(t.delta_shares):>6,d} shares @ ≈ ₹{t.exec_price:>9,.2f} | Tgt: {C.CYN}{target_weight:>5.1%}{C.RST}"
            )
        print(f"  {C.GRY}{'─' * 66}{C.RST}\n")

    return state, market_data


# ─── Status display ───────────────────────────────────────────────────────────

def _print_status(state: PortfolioState, label: str, market_data: dict) -> None:
    print(f"\n  {C.GRY}╭{'─' * 88}╮{C.RST}")
    print(f"  {C.GRY}│{C.BLD}  STATUS — {label}  {C.RST}{C.GRY}{' ' * (75 - len(label))}│{C.RST}")
    print(f"  {C.GRY}╰{'─' * 88}╯{C.RST}")

    if not state.shares:
        print(f"  {C.GRY}No open positions.{C.RST}\n")
        return

    active     = list(state.shares.keys())
    prices_now = {}
    for sym in active:
        ns = to_ns(sym)
        if ns in market_data and not market_data[ns].empty:
            prices_now[sym] = float(market_data[ns]["Close"].iloc[-1])

    mtm = sum(state.shares[s] * prices_now.get(s, 0.0) for s in active)
    pv  = mtm + state.cash

    rows      = []
    total_pnl = 0.0
    for sym in active:
        shares   = state.shares[sym]
        price    = prices_now.get(sym, float("nan"))
        entry    = state.entry_prices.get(sym, float("nan"))
        notional = shares * price if np.isfinite(price) else 0.0
        weight   = notional / pv if pv > 0 else 0.0
        pnl      = (price - entry) * shares if (np.isfinite(price) and np.isfinite(entry)) else float("nan")
        if np.isfinite(pnl):
            total_pnl += pnl
        rows.append({
            "sym": sym, "shares": shares, "price": price,
            "entry": entry, "weight": weight, "notional": notional, "pnl": pnl,
        })

    rows.sort(key=lambda x: x["weight"], reverse=True)
    c_pipe = f"{C.GRY}│{C.RST}"

    print(f"  {C.GRY}┌──────────────┬─────────┬───────────┬───────────┬────────┬─────────────┬─────────────┐{C.RST}")
    print(
        f"  {c_pipe} {C.B_CYN}{'Symbol':<12}{C.RST} {c_pipe} {C.B_CYN}{'Shares':>7}{C.RST} "
        f"{c_pipe} {C.B_CYN}{'Price':>9}{C.RST} {c_pipe} {C.B_CYN}{'Entry':>9}{C.RST} "
        f"{c_pipe} {C.B_CYN}{'Weight':>6}{C.RST} {c_pipe} {C.B_CYN}{'Notional':>11}{C.RST} "
        f"{c_pipe} {C.B_CYN}{'Unreal P&L':>11}{C.RST} {c_pipe}"
    )
    print(f"  {C.GRY}├──────────────┼─────────┼───────────┼───────────┼────────┼─────────────┼─────────────┤{C.RST}")

    for r in rows:
        pnl_raw   = f"₹{r['pnl']:+,.0f}" if np.isfinite(r["pnl"]) else "n/a"
        pnl_color = C.B_GRN if r["pnl"] > 0 else (C.B_RED if r["pnl"] < 0 else C.RST)
        print(
            f"  {c_pipe} {C.BLD}{r['sym']:<12}{C.RST} {c_pipe} {r['shares']:>7,d} "
            f"{c_pipe} {r['price']:>9,.2f} {c_pipe} {r['entry']:>9,.2f} "
            f"{c_pipe} {C.CYN}{r['weight']:>6.1%}{C.RST} {c_pipe} {r['notional']:>11,.0f} "
            f"{c_pipe} {pnl_color}{pnl_raw:>11}{C.RST} {c_pipe}"
        )

    print(f"  {C.GRY}├──────────────┼─────────┼───────────┼───────────┼────────┼─────────────┼─────────────┤{C.RST}")
    print(
        f"  {c_pipe} {C.BLD}{'Cash':<12}{C.RST} {c_pipe} {'':>7} {c_pipe} {'':>9} {c_pipe} {'':>9} "
        f"{c_pipe} {C.CYN}{state.cash/pv:>6.1%}{C.RST} {c_pipe} {state.cash:>11,.0f} {c_pipe} {'':>11} {c_pipe}"
    )
    print(f"  {C.GRY}├──────────────┼─────────┼───────────┼───────────┼────────┼─────────────┼─────────────┤{C.RST}")
    tot_color = C.B_GRN if total_pnl > 0 else (C.B_RED if total_pnl < 0 else C.RST)
    print(
        f"  {c_pipe} {C.BLD}{'TOTAL':<12}{C.RST} {c_pipe} {'':>7} {c_pipe} {'':>9} {c_pipe} {'':>9} "
        f"{c_pipe} {C.BLD}{1.0:>6.1%}{C.RST} {c_pipe} {C.BLD}{pv:>11,.0f}{C.RST} "
        f"{c_pipe} {tot_color}{'₹'+f'{total_pnl:+,.0f}':>11}{C.RST} {c_pipe}"
    )
    print(f"  {C.GRY}└──────────────┴─────────┴───────────┴───────────┴────────┴─────────────┴─────────────┘{C.RST}")

    cvar        = state.realised_cvar()
    cvar_color  = C.RED if cvar > 0.12 else C.GRN
    print(f"\n  {C.BLD}Portfolio Diagnostics:{C.RST}")
    print(f"  {C.YLW}⚡{C.RST} Exposure Multiplier : {C.BLD}{state.exposure_multiplier:.3f}{C.RST}")
    print(f"  {C.RED}🛡️ {C.RST} Override Active     : {C.BLD}{state.override_active}{C.RST}  {C.GRY}(Cooldown: {state.override_cooldown}){C.RST}")
    print(f"  {C.CYN}📉{C.RST} CVaR (realised)     : {cvar_color}{cvar:.2%}{C.RST}")
    print(f"  {C.RED}⚠️ {C.RST} Consec. Failures    : {C.BLD}{state.consecutive_failures}{C.RST}")
    print(f"  {C.BLU}📊{C.RST} Equity History Pts  : {C.BLD}{len(state.equity_hist)}{C.RST}\n")


def _portfolio_activity_badge(state: PortfolioState) -> str:
    """Compact portfolio activity badge for menu cards."""
    has_activity = bool(state.shares or state.equity_hist or abs(state.cash - 1_000_000.0) >= 1.0)
    if not has_activity:
        return f"{C.GRY}Idle{C.RST}"
    positions = len(state.shares)
    return f"{C.B_GRN}Active{C.RST} {C.GRY}({positions} pos | Cash ₹{state.cash:,.0f}){C.RST}"


def _render_main_menu(states: Dict[str, PortfolioState]) -> None:
    """Render a richer command palette for daily operations."""
    box_width = 78

    def _menu_box_line(text: str = "") -> str:
        trimmed = text[:box_width]
        return f"{C.BLU}  │{C.RST}{trimmed:<{box_width}}{C.BLU}│{C.RST}"

    now = datetime.now().strftime("%d %b %Y, %I:%M %p")
    title = f"ULTIMATE MOMENTUM V{__version__} — DAILY WORKFLOW"
    snapshot = f"Snapshot: {now}    Tip: Run status after each scan."

    print(f"\n{C.BLU}  ╭{'─' * box_width}╮{C.RST}")
    print(_menu_box_line(f"{title:^{box_width}}"))
    print(_menu_box_line(f"  {snapshot}"))
    print(f"{C.BLU}  ╰{'─' * box_width}╯{C.RST}")

    print(f"  {C.B_CYN}Scans & Research{C.RST}")
    print(f"    {C.BLD}[1]{C.RST} NSE Total Scan      {C.GRY}Run full-market rebalance preview.{C.RST}")
    print(f"    {C.BLD}[2]{C.RST} Nifty 500 Scan      {C.GRY}Focused large-cap and liquid basket.{C.RST}")
    print(f"    {C.BLD}[3]{C.RST} Custom Screener     {C.GRY}Use Screener.in or local custom list.{C.RST}")
    print(f"    {C.BLD}[4]{C.RST} Backtest            {C.GRY}Replay strategy performance by date.{C.RST}")

    print(f"  {C.B_CYN}Portfolio Operations{C.RST}")
    print(f"    {C.BLD}[5]{C.RST} Status              {C.GRY}Holdings table + risk diagnostics.{C.RST}")
    print(f"    {C.BLD}[6]{C.RST} Manage Cash         {C.GRY}Deposit/withdraw portfolio cash.{C.RST}")
    print(f"    {C.BLD}[7]{C.RST} Clear States        {C.GRY}Reset local state and cache files.{C.RST}")
    print(f"    {C.BLD}[q]{C.RST} Quit\n")

    print(f"  {C.BLD}Portfolio Health:{C.RST}")
    print(f"    NSE Total       → {_portfolio_activity_badge(states['nse_total'])}")
    print(f"    Nifty 500       → {_portfolio_activity_badge(states['nifty'])}")
    print(f"    Custom Screener → {_portfolio_activity_badge(states['custom'])}")


def _prompt_menu_choice(prompt: str, valid: List[str], default: Optional[str] = None) -> str:
    """Prompt for menu input with validation and optional default."""
    raw = input(prompt).strip().lower()
    if not raw and default is not None:
        return default
    if raw not in valid:
        print(f"  {C.RED}Invalid choice. Valid options: {', '.join(valid)}{C.RST}")
        return ""
    return raw


def _normalise_start_date(raw: str, default: str = "2020-01-01") -> str:
    """Return validated ISO date (YYYY-MM-DD), falling back to default for blank input."""
    candidate = raw.strip() or default
    try:
        datetime.strptime(candidate, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date '{candidate}'. Expected format YYYY-MM-DD.") from exc
    return candidate


# ─── Main menu ────────────────────────────────────────────────────────────────

def main_menu() -> None:
    states    = {
        "nse_total": load_portfolio_state("nse_total"),
        "nifty":     load_portfolio_state("nifty"),
        "custom":    load_portfolio_state("custom"),
    }
    mkt_cache: dict = {"nse_total": {}, "nifty": {}, "custom": {}}

    while True:
        _render_main_menu(states)

        c = _prompt_menu_choice(f"\n  {C.CYN}Choice: {C.RST}", ["1", "2", "3", "4", "5", "6", "7", "q"])
        if not c:
            continue

        if c == "1":
            _check_and_prompt_initial_capital(states["nse_total"], "NSE TOTAL", "nse_total")
            cfg          = UltimateConfig()
            preview      = copy.deepcopy(states["nse_total"])
            preview, mkt = _run_scan(fetch_nse_equity_universe(cfg=cfg), preview, "NSE TOTAL MKT SCAN", cfg)
            mkt_cache["nse_total"] = mkt
            _print_status(preview, "PREVIEW — NSE TOTAL", mkt)
            if input(f"  {C.YLW}Save these changes? (y/n): {C.RST}").strip().lower() == "y":
                states["nse_total"] = preview
                save_portfolio_state(preview, "nse_total")
                print(f"  {C.GRN}[+] Saved permanently.{C.RST}")
            else:
                print(f"  {C.GRY}[-] Discarded.{C.RST}")

        elif c == "2":
            _check_and_prompt_initial_capital(states["nifty"], "NIFTY 500", "nifty")
            cfg          = UltimateConfig()
            preview      = copy.deepcopy(states["nifty"])
            preview, mkt = _run_scan(get_nifty500(), preview, "NIFTY 500 SCAN", cfg)
            mkt_cache["nifty"] = mkt
            _print_status(preview, "PREVIEW — NIFTY 500", mkt)
            if input(f"  {C.YLW}Save these changes? (y/n): {C.RST}").strip().lower() == "y":
                states["nifty"] = preview
                save_portfolio_state(preview, "nifty")
                print(f"  {C.GRN}[+] Saved permanently.{C.RST}")
            else:
                print(f"  {C.GRY}[-] Discarded.{C.RST}")

        elif c == "3":
            universe = _get_custom_universe()
            if not universe:
                print(f"  {C.RED}[!] No custom universe found.{C.RST}")
                print(f"  {C.GRY}Please verify the Screener.in URL or provide a local file and try again.{C.RST}")
                continue
                
            logger.info("[Universe] Loaded %d symbols from custom screener.", len(universe))
            _check_and_prompt_initial_capital(states["custom"], "CUSTOM SCREENER", "custom")
            
            # Structurally tighten limits for small universes to preserve math feasibility
            custom_cfg = UltimateConfig()
            if len(universe) < 100:
                custom_cfg.MAX_POSITIONS = 8
                
            preview      = copy.deepcopy(states["custom"])
            preview, mkt = _run_scan(universe, preview, "CUSTOM SCREENER", custom_cfg)
            mkt_cache["custom"] = mkt
            _print_status(preview, "PREVIEW — CUSTOM SCREENER", mkt)
            if input(f"  {C.YLW}Save these changes? (y/n): {C.RST}").strip().lower() == "y":
                states["custom"] = preview
                save_portfolio_state(preview, "custom")
                print(f"  {C.GRN}[+] Saved permanently.{C.RST}")
            else:
                print(f"  {C.GRY}[-] Discarded.{C.RST}")

        elif c == "4":
            print(f"\n  {C.CYN}Backtest — Select Universe:{C.RST}")
            print(f"  [1] NSE Total  [2] Nifty 500  [3] Custom Screener")
            bt_c = _prompt_menu_choice(f"  {C.CYN}Choice [Default 2]: {C.RST}", ["1", "2", "3"], default="2")
            if not bt_c:
                continue
            
            # HIGH-INTEGRITY UX FIX: Add a default fallback for the Start Date
            raw_start = input(f"  {C.CYN}Start (YYYY-MM-DD) [Default 2020-01-01]: {C.RST}")
            try:
                start = _normalise_start_date(raw_start)
            except ValueError as exc:
                print(f"  {C.RED}{exc}{C.RST}")
                continue
            
            if bt_c == "1":
                universe = fetch_nse_equity_universe()
            elif bt_c == "3":
                universe = _get_custom_universe()
            else:
                universe = get_nifty500()

            end        = datetime.today().strftime("%Y-%m-%d")
            data       = load_or_fetch(universe + ["^NSEI", "^CRSLDX"], start, end)
            sector_map = get_sector_map(universe)
            print_backtest_results(run_backtest(data, universe, start, end, sector_map=sector_map))

        elif c == "5":
            for name, label in [("nse_total", "NSE TOTAL"), ("nifty", "NIFTY 500"), ("custom", "CUSTOM SCREENER")]:
                has_activity = states[name].shares or states[name].equity_hist or abs(states[name].cash - 1_000_000.0) >= 1.0
                if has_activity:
                    mkt = mkt_cache.get(name) or {}
                    if not mkt and states[name].shares:
                        syms = list({to_ns(s) for s in states[name].shares})
                        end  = datetime.today().strftime("%Y-%m-%d")
                        # High-Integrity Fix: Fetch 22 days to guarantee valid T-1 ADV signals immediately after checking status
                        mkt  = load_or_fetch(
                            syms,
                            (datetime.today() - timedelta(days=22)).strftime("%Y-%m-%d"),
                            end,
                        )
                        mkt_cache[name] = mkt
                    _print_status(states[name], label, mkt)
            if not any((states[n].shares or states[n].equity_hist or abs(states[n].cash - 1_000_000.0) >= 1.0) for n in states):
                print(f"  {C.GRY}All portfolios are empty.{C.RST}")

        elif c == "6":
            print(f"\n  {C.CYN}Manage Cash — Select Portfolio:{C.RST}")
            print(f"  [1] NSE Total  [2] Nifty 500  [3] Custom Screener")
            p_c = _prompt_menu_choice(f"  {C.CYN}Choice: {C.RST}", ["1", "2", "3"])
            if not p_c:
                continue
            p_map = {"1": "nse_total", "2": "nifty", "3": "custom"}
            if p_c in p_map:
                name = p_map[p_c]
                state = states[name]
                print(f"\n  {C.BLD}Current Cash: {C.GRN}₹{state.cash:,.2f}{C.RST}")
                print(f"  {C.GRY}Use positive number to deposit, negative to withdraw.{C.RST}")
                try:
                    amt_str = input(f"  {C.CYN}Amount (₹): {C.RST}").replace(",", "").strip()
                    amt = float(amt_str)
                    state.cash = max(0.0, state.cash + amt)
                    save_portfolio_state(state, name)
                    action = "Deposited" if amt >= 0 else "Withdrew"
                    print(f"  {C.GRN}[+] {action} ₹{abs(amt):,.2f}. New Cash: ₹{state.cash:,.2f}{C.RST}")
                except ValueError:
                    print(f"  {C.RED}Invalid amount.{C.RST}")
            else:
                print(f"  {C.RED}Invalid choice.{C.RST}")

        elif c == "7":
            print(f"\n  {C.B_RED}WARNING: This will permanently erase ALL portfolio states and caches.{C.RST}")
            confirm = input(f"  {C.CYN}Type 'YES' to confirm: {C.RST}").strip()
            # HIGH-INTEGRITY FIX: Make confirmation case-insensitive
            if confirm.upper() == "YES":
                invalidate_cache()
                invalidate_universe_cache()
                for n in ["nse_total", "nifty", "custom"]:
                    p = f"data/portfolio_state_{n}.json"
                    for suffix in ["", ".bak.0", ".bak.1", ".bak.2"]:
                        target = p + suffix
                        if os.path.exists(target):
                            os.remove(target)
                states    = {"nse_total": PortfolioState(), "nifty": PortfolioState(), "custom": PortfolioState()}
                mkt_cache = {"nse_total": {}, "nifty": {}, "custom": {}}
                print(f"  {C.GRN}[+] All states and caches cleared.{C.RST}")
            else:
                print(f"  {C.GRY}Cancelled.{C.RST}")

        elif c == "q":
            print(f"  {C.GRY}Goodbye!{C.RST}\n")
            break


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=f"{C.GRY}[%(asctime)s]{C.RST} %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main_menu()
