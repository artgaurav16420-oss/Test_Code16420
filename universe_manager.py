"""
universe_manager.py — Universe Fetching & Caching
==================================================
v11.9 — Final Hardened Release

Full audit history of corrections applied across all review passes:

v11.1  _apply_adv_filter now handles both single-ticker flat DataFrames
       and multi-ticker MultiIndex DataFrames returned by yfinance.
       Previously, .get(ns_sym) against a tuple-keyed MultiIndex always
       returned NaN, silently dropping the entire universe.
v11.6  Network resiliency: transient yfinance failures in _apply_adv_filter
       now return the unfiltered ticker list (fail-open) rather than an
       empty list (fail-closed), preventing a 5-second network hiccup from
       zeroing the investment universe.
       Atomic universe cache writes via os.replace() to prevent partial
       JSON files on sudden process termination.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR            = "data/cache"
UNIVERSE_CACHE_FILE  = os.path.join(CACHE_DIR, "_universe_cache.json")
UNIVERSE_CACHE_TTL_H = 24
MIN_ADV_CRORES       = 5.0
SCREENER_HEADERS     = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
SCREENER_TIMEOUT     = 12
MAX_PAGES            = 10


def _load_universe_cache(source_key: str) -> Optional[List[str]]:
    if not os.path.exists(UNIVERSE_CACHE_FILE):
        return None
    try:
        with open(UNIVERSE_CACHE_FILE) as f:
            data = json.load(f)
        entry      = data.get(source_key, {})
        fetched_at = datetime.fromisoformat(entry.get("fetched_at", "2000-01-01"))
        if datetime.now() - fetched_at < timedelta(hours=UNIVERSE_CACHE_TTL_H):
            tickers = entry.get("tickers", [])
            if tickers:
                clean = [t for t in tickers if not t.isdigit()]
                if clean:
                    logger.info(f"[Universe] Cache hit ({source_key}): {len(clean)} tickers")
                    return clean
    except Exception as e:
        logger.warning(f"[Universe] Cache read failed: {e}")
    return None


def _save_universe_cache(source_key: str, tickers: List[str]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        existing = {}
        if os.path.exists(UNIVERSE_CACHE_FILE):
            with open(UNIVERSE_CACHE_FILE) as f:
                existing = json.load(f)
        existing[source_key] = {
            "fetched_at": datetime.now().isoformat(),
            "tickers":    tickers,
        }
        tmp_path = UNIVERSE_CACHE_FILE + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp_path, UNIVERSE_CACHE_FILE)
    except Exception as e:
        logger.warning(f"[Universe] Cache write failed: {e}")


def _apply_adv_filter(tickers: List[str], min_adv_crores: float) -> List[str]:
    if min_adv_crores <= 0:
        return tickers

    import yfinance as yf

    ns = [t if t.endswith(".NS") else t + ".NS" for t in tickers]

    try:
        raw = yf.download(ns, period="1mo", progress=False, auto_adjust=True)
        if raw.empty:
            logger.warning(
                "[Universe] ADV filter: yfinance returned empty DataFrame. "
                "Bypassing filter to preserve universe."
            )
            return tickers

        if len(ns) == 1:
            # Single-ticker: yfinance returns a flat DataFrame — rename to ticker.
            close  = raw[["Close"]].rename(columns={"Close": ns[0]})
            volume = raw[["Volume"]].rename(columns={"Volume": ns[0]})
        else:
            close  = raw["Close"].copy()
            volume = raw["Volume"].copy()
            # Flatten MultiIndex columns if present (e.g. ("RELIANCE.NS", "") tuples).
            if isinstance(close.columns, pd.MultiIndex):
                close.columns  = close.columns.get_level_values(0)
                volume.columns = volume.columns.get_level_values(0)

        adv_cr = (close * volume).rolling(20).mean().iloc[-1] / 1e7

        passed = []
        for sym, ns_sym in zip(tickers, ns):
            adv_val = adv_cr.get(ns_sym, np.nan)
            if isinstance(adv_val, (float, int, np.floating)) and np.isfinite(adv_val):
                if adv_val >= min_adv_crores:
                    passed.append(sym)
            else:
                logger.debug(
                    f"[Universe] Dropping {sym}: invalid or missing ADV ({adv_val})."
                )

        logger.info(f"[Universe] ADV filter: {len(tickers)} → {len(passed)} tickers")
        return passed

    except Exception as e:
        # Fail-open: a transient network error must not zero the investment universe.
        logger.warning(
            f"[Universe] ADV filter failed ({e}); bypassing filter."
        )
        return tickers


def fetch_screener_universe(
    url:            str,
    use_cache:      bool  = True,
    apply_adv:      bool  = True,
    min_adv_crores: float = MIN_ADV_CRORES,
) -> List[str]:
    import requests
    from bs4 import BeautifulSoup

    cache_key = f"screener_{url.strip('/').split('/')[-1]}"
    if use_cache:
        cached = _load_universe_cache(cache_key)
        if cached:
            return cached

    logger.info(f"[Universe] Fetching Screener.in: {url}")
    seen, symbols = set(), []
    try:
        for page in range(1, MAX_PAGES + 1):
            sep      = "&" if "?" in url else "?"
            page_url = f"{url}{sep}page={page}"
            resp     = requests.get(
                page_url, headers=SCREENER_HEADERS, timeout=SCREENER_TIMEOUT
            )
            if resp.status_code != 200:
                break

            soup      = BeautifulSoup(resp.text, "html.parser")
            page_syms = []
            for link in soup.select('a[href^="/company/"]'):
                parts = link.get("href", "").strip("/").split("/")
                if len(parts) >= 2 and parts[0] == "company":
                    sym = parts[1].upper().strip()
                    if sym and not sym.isdigit():
                        page_syms.append(sym)

            new_found = 0
            for s in page_syms:
                if s not in seen:
                    seen.add(s)
                    symbols.append(s)
                    new_found += 1
            if new_found == 0:
                break
            time.sleep(0.5)

    except Exception as e:
        logger.warning(f"[Universe] Screener fetch failed: {e}")

    if apply_adv and symbols:
        symbols = _apply_adv_filter(symbols, min_adv_crores)

    if symbols:
        _save_universe_cache(cache_key, symbols)
    return symbols


def get_nifty500(use_cache: bool = True) -> List[str]:
    cache_key = "nifty500"
    if use_cache:
        cached = _load_universe_cache(cache_key)
        if cached:
            return cached
    try:
        df  = pd.read_csv(
            "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        )
        col     = next((c for c in ["Symbol", "SYMBOL"] if c in df.columns), None)
        tickers = df[col].tolist() if col else df.iloc[:, 0].astype(str).tolist()
        _save_universe_cache(cache_key, tickers)
        return tickers
    except Exception as e:
        logger.warning(f"[Universe] Nifty 500 fetch failed: {e}")
        return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]


def invalidate_universe_cache() -> None:
    if os.path.exists(UNIVERSE_CACHE_FILE):
        os.remove(UNIVERSE_CACHE_FILE)