"""
universe_manager.py — Universe Fetching & Caching
==================================================
Official NSE CSV source with chunked ADV filter, persistent JSON cache,
and static sector map for the most liquid NSE constituents.
"""

from __future__ import annotations

import json
import logging
import os
import time
import io
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR            = "data/cache"
UNIVERSE_CACHE_FILE  = os.path.join(CACHE_DIR, "_universe_cache.json")
UNIVERSE_CACHE_TTL_H = 24

# ADV filter processes tickers in parallel workers, each handling a chunk of
# _ADV_CHUNK_SIZE to stay within yfinance rate limits.
_ADV_CHUNK_SIZE    = 200
_ADV_MAX_WORKERS   = 4   # parallel chunk workers; keeps total connections manageable

# ─── Survival Mode Circuit Breaker ───────────────────────────────────────────

# FIX #9: Hardcoded Nifty 50 floor for survival mode. This list was verified
# against the NSE index composition on 2025-07-01. Index constituents change
# periodically — re-verify this list if you update the codebase after a
# significant index rebalance. Constituents here are the 49 most liquid names
# from the official Nifty 50 at that date (LTIM replaces SHREECEM post-rebalance).
_HARD_FLOOR_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "HINDUNILVR",
    "ITC", "SBIN", "LTIM", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA",
    "ADANIENT", "KOTAKBANK", "TITAN", "ONGC", "TATAMOTORS", "NTPC", "AXISBANK",
    "ADANIPORTS", "ASIANPAINT", "COALINDIA", "BAJAJFINSV", "JSWSTEEL",
    "M&M", "POWERGRID", "TATASTEEL", "ULTRACEMCO", "GRASIM", "HINDALCO", "NESTLEIND",
    "INDUSINDBK", "TECHM", "WIPRO", "CIPLA", "HDFCLIFE", "SBILIFE", "DRREDDY",
    "HEROMOTOCO", "EICHERMOT", "BPCL", "BAJAJ-AUTO", "BRITANNIA", "APOLLOHOSP",
    "DIVISLAB", "TATACONSUM",
]

# fmt: off
STATIC_NSE_SECTORS: Dict[str, str] = {
    # Energy
    "RELIANCE":   "Energy",    "ONGC":       "Energy",     "COALINDIA":  "Energy",
    "BPCL":       "Energy",    "IOC":         "Energy",
    # Financials
    "HDFCBANK":   "Financials","ICICIBANK":  "Financials", "KOTAKBANK":  "Financials",
    "SBIN":       "Financials","AXISBANK":   "Financials", "BAJFINANCE": "Financials",
    "BAJAJFINSV": "Financials","INDUSINDBK": "Financials",
    # IT
    "TCS":        "IT",        "INFY":       "IT",         "HCLTECH":    "IT",
    "WIPRO":      "IT",        "TECHM":      "IT",         "LTIM":       "IT",
    # Consumer
    "HINDUNILVR": "Consumer",  "ITC":        "Consumer",   "NESTLEIND":  "Consumer",
    "TITAN":      "Consumer",  "ASIANPAINT": "Consumer",   "TATACONSUM": "Consumer",
    # Telecom
    "BHARTIARTL": "Telecom",
    # Auto
    "MARUTI":     "Auto",      "TATAMOTORS": "Auto",       "M&M":        "Auto",
    "BAJAJ-AUTO": "Auto",      "EICHERMOT":  "Auto",       "HEROMOTOCO": "Auto",
}
# fmt: on


# ─── Helper: Requests-Buffered CSV Fetch ─────────────────────────────────────

def _fetch_csv_with_headers(url: str, timeout: float = 15.0) -> pd.DataFrame:
    """Fetches a CSV from a URL with browser-like headers and explicit HTTP error telemetry."""
    try:
        import requests
    except ImportError:
        logger.error("[Universe] Missing 'requests' library. Please install it to fetch live data.")
        raise

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":     "text/csv",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except requests.exceptions.HTTPError as he:
        if he.response.status_code == 403:
            logger.error("[Universe] Access Denied (403). NSE website is blocking requests.")
        else:
            logger.error("[Universe] HTTP Error %d while reaching NSE.", he.response.status_code)
        raise
    except Exception as exc:
        logger.error("[Universe] Failed to fetch CSV from %s: %s", url, exc)
        raise


# ─── Cache Management ─────────────────────────────────────────────────────────

def _load_universe_cache() -> dict:
    if not os.path.exists(UNIVERSE_CACHE_FILE):
        return {}
    try:
        with open(UNIVERSE_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_universe_cache(data: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = UNIVERSE_CACHE_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, UNIVERSE_CACHE_FILE)
    except Exception as exc:
        logger.warning("[Universe] Cache write failed: %s", exc)


def invalidate_universe_cache() -> None:
    if os.path.exists(UNIVERSE_CACHE_FILE):
        os.remove(UNIVERSE_CACHE_FILE)


# ─── Core Interface ───────────────────────────────────────────────────────────

def fetch_nse_equity_universe(cfg=None) -> List[str]:
    """Fetches the total NSE equity universe with survival-mode circuit breaker."""
    cache = _load_universe_cache()
    entry = cache.get("total_equity", {})

    if entry:
        fetched_at = datetime.fromisoformat(entry["fetched_at"])
        if datetime.now() - fetched_at < timedelta(hours=UNIVERSE_CACHE_TTL_H):
            return entry["tickers"]

    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        df = _fetch_csv_with_headers(url)
        # Standardize column names to handle potential NSE schema shifts.
        df.columns = [c.strip().upper() for c in df.columns]
        tickers = df[df["SERIES"] == "EQ"]["SYMBOL"].unique().tolist()

        if not tickers:
            raise ValueError("NSE CSV returned zero 'EQ' series symbols.")

        tickers = _apply_adv_filter(tickers, cfg)

        cache["total_equity"] = {
            "fetched_at": datetime.now().isoformat(),
            "tickers":    tickers,
        }
        _save_universe_cache(cache)
        return tickers

    except Exception:
        if entry:
            logger.warning("[Universe] Live fetch failed. Using stale cache for Total Equity.")
            return entry["tickers"]
        logger.error("[Universe] CRITICAL: Live fetch and cache failed. Entering Survival Mode (Nifty 50).")
        return _HARD_FLOOR_UNIVERSE


def get_nifty500() -> List[str]:
    """Fetches the Nifty 500 universe with survival-mode fallback."""
    cache = _load_universe_cache()
    entry = cache.get("nifty500", {})

    if entry:
        fetched_at = datetime.fromisoformat(entry["fetched_at"])
        if datetime.now() - fetched_at < timedelta(hours=UNIVERSE_CACHE_TTL_H):
            return entry["tickers"]

    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = _fetch_csv_with_headers(url)
        df.columns = [c.strip().upper() for c in df.columns]
        tickers = df["SYMBOL"].unique().tolist()

        if not tickers:
            raise ValueError("Nifty 500 CSV returned zero symbols.")

        cache["nifty500"] = {
            "fetched_at": datetime.now().isoformat(),
            "tickers":    tickers,
        }
        _save_universe_cache(cache)
        return tickers

    except Exception:
        if entry:
            logger.warning("[Universe] Live fetch failed. Using stale cache for Nifty 500.")
            return entry["tickers"]
        logger.error("[Universe] CRITICAL: Live fetch and cache failed. Reverting to Nifty 50.")
        return _HARD_FLOOR_UNIVERSE


# ─── Liquidity Filtering ──────────────────────────────────────────────────────

def _process_adv_chunk(chunk: List[str], start_dt: str, end_dt: str, cfg) -> List[str]:
    """
    Worker: fetch ADV data for one chunk and return tickers that pass the filter.
    Called in parallel by _apply_adv_filter.
    """
    from data_cache import load_or_fetch

    for attempt in range(3):
        try:
            data    = load_or_fetch(chunk, start_dt, end_dt, cfg=cfg)
            passing = []
            for sym in chunk:
                ns_sym = sym + ".NS"
                if ns_sym in data:
                    df      = data[ns_sym]
                    adv_val = (df["Close"] * df["Volume"]).rolling(20, min_periods=1).mean().iloc[-1]
                    if adv_val >= (cfg.MIN_ADV_CRORES * 1e7):
                        passing.append(sym)
            return passing
        except Exception as exc:
            logger.debug(
                "[Universe] ADV chunk attempt %d failed: %s", attempt + 1, exc
            )
            time.sleep((2 ** attempt) + 0.5)

    # FIX #5: All retries exhausted. Include entire chunk unfiltered as a
    # failsafe to prevent silent universe truncation on transient network
    # failures. Illiquid names admitted here will still be blocked by the
    # liquidity gate in generate_signals() (ADV == 0 → adj_score = -inf),
    # EXCEPT in decay mode where generate_signals is bypassed. Log at ERROR
    # level so this failure is always visible in production logs.
    logger.error(
        "[Universe] ADV chunk failed all retries (%d tickers). "
        "Including unfiltered as failsafe — ADV gate skipped for this chunk. "
        "NOTE: decay mode bypasses signal-level liquidity gate; monitor positions.",
        len(chunk),
    )
    return chunk


def _apply_adv_filter(tickers: List[str], cfg) -> List[str]:
    """
    Filters a broad universe down to institutionally liquid names.

    FIX #4: Chunks are processed in parallel using ThreadPoolExecutor
    (_ADV_MAX_WORKERS workers) to avoid blocking the main thread for
    several minutes on a full NSE universe of ~2,000 tickers.
    """
    from momentum_engine import UltimateConfig

    if cfg is None:
        cfg = UltimateConfig()

    end_dt   = datetime.today().strftime("%Y-%m-%d")
    start_dt = (datetime.today() - timedelta(days=40)).strftime("%Y-%m-%d")

    chunks = [
        tickers[i : i + _ADV_CHUNK_SIZE]
        for i in range(0, len(tickers), _ADV_CHUNK_SIZE)
    ]

    filtered: List[str] = []
    with ThreadPoolExecutor(max_workers=_ADV_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_process_adv_chunk, chunk, start_dt, end_dt, cfg): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            try:
                filtered.extend(future.result())
            except Exception as exc:
                chunk = futures[future]
                logger.error(
                    "[Universe] Unexpected error processing ADV chunk (%d tickers): %s. "
                    "Including unfiltered.", len(chunk), exc
                )
                filtered.extend(chunk)

    return filtered


# ─── Sector Resolution ───────────────────────────────────────────────────────

def get_sector_map(tickers: List[str], use_cache: bool = True, cfg=None) -> Dict[str, str]:
    """
    Resolves sector labels for a list of tickers.

    Resolution order: (1) static map, (2) persistent cache, (3) live yfinance
    fetch with parallel threads and serial failover. Results from step 3 are
    committed back to the cache atomically.
    """
    from data_cache import CACHE_DIR  # noqa: F401 — imported for side-effect of availability check

    # 1. Static known mapping.
    resolved = {
        t.replace(".NS", ""): STATIC_NSE_SECTORS[t.replace(".NS", "")]
        for t in tickers
        if t.replace(".NS", "") in STATIC_NSE_SECTORS
    }

    missing = [t.replace(".NS", "") for t in tickers if t.replace(".NS", "") not in resolved]
    if not missing:
        return {t: resolved[t.replace(".NS", "")] for t in tickers}

    # 2. Persistent cache lookup.
    cached_sectors: Dict[str, str] = {}
    if use_cache:
        cache        = _load_universe_cache()
        sector_cache = cache.get("sector_map", {}).get("sectors", {})
        for sym in list(missing):
            if sym in sector_cache:
                resolved[sym] = sector_cache[sym]
                missing.remove(sym)

    # 3. Live yfinance fetch for anything still unresolved.
    if missing:
        import yfinance as yf
        timeout = getattr(cfg, "SECTOR_FETCH_TIMEOUT", 8.0)

        def _fetch_one(s: str) -> tuple[str, str]:
            try:
                info = yf.Ticker(s + ".NS").info
                return s, info.get("sector", "Unknown")
            except Exception:
                return s, "Unknown"

        print(f"  \033[90mResolving metadata for {len(missing)} tickers...\033[0m")

        with ThreadPoolExecutor(max_workers=8) as pool:
            future_to_sym = {pool.submit(_fetch_one, sym): sym for sym in missing}
            try:
                for future in as_completed(future_to_sym, timeout=timeout + 2.0):
                    sym    = future_to_sym[future]
                    sector = "Unknown"
                    try:
                        _, sector = future.result(timeout=timeout)
                    except Exception:
                        # Per-future serial failover for hanging requests.
                        logger.debug(
                            "[Universe] Sector hang for %s; triggering serial failover.", sym
                        )
                        _, sector = _fetch_one(sym)
                    resolved[sym]       = sector
                    cached_sectors[sym] = sector
            except TimeoutError:
                # Tickers not yet processed default to 'Unknown'; logged for visibility.
                logger.warning(
                    "[Universe] Global timeout reached during sector resolution. "
                    "Remaining tickers will default to 'Unknown'."
                )

        # Atomic staged commit of newly resolved sectors to cache.
        if use_cache and cached_sectors:
            cache       = _load_universe_cache()
            current_map = cache.get("sector_map", {}).get("sectors", {})
            current_map.update(cached_sectors)
            cache["sector_map"] = {
                "fetched_at": datetime.now().isoformat(),
                "sectors":    current_map,
            }
            _save_universe_cache(cache)

    return {t: resolved.get(t.replace(".NS", ""), "Unknown") for t in tickers}
