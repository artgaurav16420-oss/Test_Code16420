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
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR            = "data/cache"
UNIVERSE_CACHE_FILE  = os.path.join(CACHE_DIR, "_universe_cache.json")
UNIVERSE_CACHE_TTL_H = 24

# ADV filter is run in chunks to avoid yfinance rate-limiting on large universes.
_ADV_CHUNK_SIZE = 200

# fmt: off
STATIC_NSE_SECTORS: Dict[str, str] = {
    # Energy
    "RELIANCE":    "Energy",
    "ONGC":        "Energy",
    "COALINDIA":   "Energy",
    "BPCL":        "Energy",
    "IOC":         "Energy",
    # Financials
    "HDFCBANK":    "Financials",
    "ICICIBANK":   "Financials",
    "KOTAKBANK":   "Financials",
    "SBIN":        "Financials",
    "AXISBANK":    "Financials",
    "BAJFINANCE":  "Financials",
    "BAJAJFINSV":  "Financials",
    "INDUSINDBK":  "Financials",
    "HDFCLIFE":    "Financials",
    "CHOLAFIN":    "Financials",
    # Information Technology
    "TCS":         "Information Technology",
    "INFY":        "Information Technology",
    "HCLTECH":     "Information Technology",
    "WIPRO":       "Information Technology",
    "TECHM":       "Information Technology",
    "LTIM":        "Information Technology",
    # Consumer Staples
    "HINDUNILVR":  "Consumer Staples",
    "ITC":         "Consumer Staples",
    "BRITANNIA":   "Consumer Staples",
    "NESTLEIND":   "Consumer Staples",
    # Consumer Discretionary
    "MARUTI":      "Consumer Discretionary",
    "TATAMOTORS":  "Consumer Discretionary",
    "M&M":         "Consumer Discretionary",
    "BAJAJ-AUTO":  "Consumer Discretionary",
    "HEROMOTOCO":  "Consumer Discretionary",
    "EICHERMOT":   "Consumer Discretionary",
    "TVSMOTOR":    "Consumer Discretionary",
    "TRENT":       "Consumer Discretionary",
    # Health Care
    "SUNPHARMA":   "Health Care",
    "DRREDDY":     "Health Care",
    "CIPLA":       "Health Care",
    "APOLLOHOSP":  "Health Care",
    "DIVISLAB":    "Health Care",
    # Industrials
    "LT":          "Industrials",
    "SIEMENS":     "Industrials",
    "ABB":         "Industrials",
    # Materials
    "TATASTEEL":   "Materials",
    "JSWSTEEL":    "Materials",
    "HINDALCO":    "Materials",
    "GRASIM":      "Materials",
    "ULTRACEMCO":  "Materials",
    "ASIANPAINT":  "Materials",
    # Communication Services
    "BHARTIARTL":  "Communication Services",
    "ZOMATO":      "Communication Services",
    # Real Estate
    "DLF":         "Real Estate",
    "GODREJPROP":  "Real Estate",
    # Utilities
    "ADANIENT":    "Utilities",
    "TORNTPOWER":  "Utilities",
    "NTPC":        "Utilities",
    "POWERGRID":   "Utilities",
    # Diversified / Other
    "TITAN":       "Consumer Discretionary",
    "PNB":         "Financials",
}
# fmt: on

# Deduplicated Nifty 500 fallback (no duplicate symbols).
_NIFTY500_FALLBACK: List[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "LT", "HCLTECH", "AXISBANK", "BAJFINANCE", "MARUTI",
    "WIPRO", "NTPC", "TITAN", "SUNPHARMA", "ULTRACEMCO", "ASIANPAINT", "ONGC",
    "TATASTEEL", "ADANIENT", "POWERGRID", "BAJAJFINSV", "M&M", "JSWSTEEL", "HDFCLIFE",
    "COALINDIA", "GRASIM", "TECHM", "BRITANNIA", "BAJAJ-AUTO", "HINDALCO",
    "INDUSINDBK", "CIPLA", "APOLLOHOSP", "DRREDDY", "HEROMOTOCO", "EICHERMOT", "DIVISLAB",
    "TRENT", "BPCL", "LTIM", "ZOMATO", "TVSMOTOR", "CHOLAFIN", "PNB",
]


# ─── Universe cache helpers ───────────────────────────────────────────────────

def _load_universe_cache(source_key: str) -> Optional[List[str]]:
    if not os.path.exists(UNIVERSE_CACHE_FILE):
        return None
    try:
        with open(UNIVERSE_CACHE_FILE) as f:
            data = json.load(f)
        entry      = data.get(source_key, {})
        fetched_at = datetime.fromisoformat(entry.get("fetched_at", "2000-01-01"))
        if datetime.now() - fetched_at < timedelta(hours=UNIVERSE_CACHE_TTL_H):
            tickers = [t for t in entry.get("tickers", []) if not t.isdigit()]
            if tickers:
                logger.info("[Universe] Cache hit (%s): %d tickers.", source_key, len(tickers))
                return tickers
    except Exception as exc:
        logger.warning("[Universe] Cache read failed: %s", exc)
    return None


def _save_universe_cache(source_key: str, tickers: List[str]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        existing: dict = {}
        if os.path.exists(UNIVERSE_CACHE_FILE):
            with open(UNIVERSE_CACHE_FILE) as f:
                existing = json.load(f)
        existing[source_key] = {"fetched_at": datetime.now().isoformat(), "tickers": tickers}
        tmp = UNIVERSE_CACHE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp, UNIVERSE_CACHE_FILE)
    except Exception as exc:
        logger.warning("[Universe] Cache write failed: %s", exc)


# ─── ADV filter ───────────────────────────────────────────────────────────────

def _apply_adv_filter_chunk(tickers: List[str], min_adv_crores: float, timeout: float) -> List[str]:
    """Filter one chunk of tickers by 20-day average daily value."""
    import yfinance as yf

    ns_tickers = [t if t.endswith(".NS") else t + ".NS" for t in tickers]

    def _do() -> pd.DataFrame:
        for attempt in range(3):
            try:
                return yf.download(ns_tickers, period="1mo", progress=False, auto_adjust=True)
            except Exception as exc:
                if attempt == 2:
                    raise exc
                time.sleep((2 ** attempt) + random.uniform(0, 1))

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            raw = pool.submit(_do).result(timeout=timeout)
    except Exception as exc:
        logger.warning("[Universe] ADV chunk failed (%s); bypassing filter for chunk.", exc)
        return tickers

    if raw is None or raw.empty:
        return tickers

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    if len(ns_tickers) == 1:
        close  = raw[["Close"]].rename(columns={"Close": ns_tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": ns_tickers[0]})
    elif is_multi:
        close  = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        close  = raw[["Close"]]
        volume = raw[["Volume"]]

    adv_cr = (close * volume).rolling(20).mean().iloc[-1] / 1e7
    passed = []
    for sym, ns in zip(tickers, ns_tickers):
        val = adv_cr.get(ns, np.nan)
        if isinstance(val, (float, int, np.floating)) and np.isfinite(val) and val >= min_adv_crores:
            passed.append(sym)
        else:
            # SAFETY FIX: `val or 0` raises ValueError if val is a pandas Series.
            # Use an explicit scalar-safe conversion before calling np.isfinite.
            _val_scalar = float(val) if isinstance(val, (float, int, np.floating)) else 0.0
            logger.debug("[Universe] Dropping %s: ADV ₹%.1f cr < ₹%.0f cr threshold.", sym, _val_scalar if np.isfinite(_val_scalar) else 0.0, min_adv_crores)
    return passed


def _apply_adv_filter(tickers: List[str], min_adv_crores: float, cfg=None) -> List[str]:
    """
    Run ADV filter in chunks of _ADV_CHUNK_SIZE to avoid yfinance rate-limiting.
    Falls back gracefully: if a chunk download fails, that chunk passes unfiltered.
    """
    if min_adv_crores <= 0:
        return tickers

    timeout = getattr(cfg, "YF_ADV_TIMEOUT", 60.0) if cfg else 60.0
    passed: List[str] = []

    for i in range(0, len(tickers), _ADV_CHUNK_SIZE):
        chunk = tickers[i: i + _ADV_CHUNK_SIZE]
        logger.info(
            "[Universe] ADV filtering chunk %d/%d (%d tickers)...",
            i // _ADV_CHUNK_SIZE + 1,
            (len(tickers) - 1) // _ADV_CHUNK_SIZE + 1,
            len(chunk),
        )
        passed.extend(_apply_adv_filter_chunk(chunk, min_adv_crores, timeout))

    logger.info("[Universe] ADV filter: %d → %d tickers.", len(tickers), len(passed))
    return passed


# ─── Universe sources ─────────────────────────────────────────────────────────

def fetch_nse_equity_universe(
    use_cache: bool = True,
    apply_adv: bool = True,
    cfg=None,
) -> List[str]:
    """
    Fetch the complete NSE equity universe from the official EQUITY_L CSV.
    Falls back to Nifty 500 if the download fails.
    """
    cache_key = "nse_equity_all"
    if use_cache:
        cached = _load_universe_cache(cache_key)
        if cached:
            return cached

    logger.info("[Universe] Fetching official NSE Equity list (EQUITY_L)...")
    try:
        df = pd.read_csv("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
        symbols = [
            s for s in df["SYMBOL"].astype(str).tolist()
            if not s.isdigit() and " " not in s
        ]
        logger.info("[Universe] NSE Equity fetch complete: %d symbols.", len(symbols))

        if apply_adv and symbols:
            min_adv = getattr(cfg, "MIN_ADV_CRORES", 50.0) if cfg else 50.0
            symbols = _apply_adv_filter(symbols, min_adv, cfg)

        if symbols:
            _save_universe_cache(cache_key, symbols)
        return symbols

    except Exception as exc:
        logger.error("[Universe] NSE Equity fetch failed: %s. Falling back to Nifty 500.", exc)
        return get_nifty500(use_cache)


def get_nifty500(use_cache: bool = True) -> List[str]:
    """Fetch the NSE Nifty 500 constituent list, with a static fallback."""
    cache_key = "nifty500"
    if use_cache:
        cached = _load_universe_cache(cache_key)
        if cached:
            return cached
    try:
        df  = pd.read_csv("https://archives.nseindia.com/content/indices/ind_nifty500list.csv")
        col = next((c for c in ["Symbol", "SYMBOL"] if c in df.columns), None)
        tickers = (
            df[col].astype(str).tolist() if col
            else df.iloc[:, 0].astype(str).tolist()
        )
        # Deduplicate while preserving order.
        seen: set = set()
        tickers   = [t for t in tickers if not (t in seen or seen.add(t))]
        _save_universe_cache(cache_key, tickers)
        return tickers
    except Exception as exc:
        logger.warning("[Universe] Nifty 500 fetch failed: %s. Using static fallback.", exc)
        return list(_NIFTY500_FALLBACK)  # return a copy


def invalidate_universe_cache() -> None:
    if os.path.exists(UNIVERSE_CACHE_FILE):
        os.remove(UNIVERSE_CACHE_FILE)


# ─── Sector map ───────────────────────────────────────────────────────────────

def get_sector_map(
    tickers:             List[str],
    use_cache:           bool  = True,
    ttl_days:            int   = 7,
    max_workers:         int   = 8,
    per_ticker_timeout:  float = 8.0,
    cfg=None,
) -> Dict[str, str]:
    """
    Return {ticker → sector} for every ticker in the list.

    Resolution order:
    1. STATIC_NSE_SECTORS (instant, no network)
    2. Persistent sector cache (JSON, ttl_days TTL)
    3. yfinance live fetch (parallel, with serial fallback)

    All lookups use *bare* symbol keys (no .NS suffix).
    """
    import yfinance as yf

    timeout = getattr(cfg, "SECTOR_FETCH_TIMEOUT", per_ticker_timeout) if cfg else per_ticker_timeout
    bare    = [t.replace(".NS", "") for t in tickers]
    resolved: Dict[str, str] = {s: STATIC_NSE_SECTORS[s] for s in bare if s in STATIC_NSE_SECTORS}

    # Load persisted cache.
    cached_sectors: Dict[str, str] = {}
    if use_cache and os.path.exists(UNIVERSE_CACHE_FILE):
        try:
            with open(UNIVERSE_CACHE_FILE) as f:
                data = json.load(f)
            entry      = data.get("sector_map", {})
            fetched_at = datetime.fromisoformat(entry.get("fetched_at", "2000-01-01"))
            if datetime.now() - fetched_at < timedelta(days=ttl_days):
                cached_sectors = entry.get("sectors", {})
        except Exception as exc:
            logger.warning("[Universe] Sector cache read failed: %s", exc)

    for sym in bare:
        if sym not in resolved and sym in cached_sectors:
            resolved[sym] = cached_sectors[sym]

    still_missing = [s for s in bare if s not in resolved]

    if still_missing:
        logger.info("[Universe] Fetching sector data for %d tickers...", len(still_missing))

        def _fetch_one(sym: str) -> tuple:
            for attempt in range(3):
                try:
                    info   = yf.Ticker(sym + ".NS").info
                    sector = info.get("sector") or info.get("sectorDisp") or "Unknown"
                    return sym, sector
                except Exception:
                    if attempt == 2:
                        return sym, "Unknown"
                    time.sleep((2 ** attempt) + random.uniform(0, 1))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, sym): sym for sym in still_missing}
            for fut, sym in futures.items():
                sector = "Unknown"
                try:
                    _, sector = fut.result(timeout=timeout)
                except Exception:
                    logger.debug("[Universe] Sector fetch timeout for %s; falling back to serial.", sym)
                    try:
                        _, sector = _fetch_one(sym)
                    except Exception:
                        pass
                resolved[sym]       = sector
                cached_sectors[sym] = sector

        if use_cache:
            try:
                existing: dict = {}
                if os.path.exists(UNIVERSE_CACHE_FILE):
                    with open(UNIVERSE_CACHE_FILE) as f:
                        existing = json.load(f)
                existing["sector_map"] = {
                    "fetched_at": datetime.now().isoformat(),
                    "sectors":    {
                        **existing.get("sector_map", {}).get("sectors", {}),
                        **cached_sectors,
                    },
                }
                tmp = UNIVERSE_CACHE_FILE + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(existing, f, indent=2)
                os.replace(tmp, UNIVERSE_CACHE_FILE)
            except Exception as exc:
                logger.warning("[Universe] Sector cache write failed: %s", exc)

    return {t: resolved.get(t.replace(".NS", ""), "Unknown") for t in tickers}
