"""
data_cache.py — Persistent Atomic Downloader
=============================================
Parquet-backed cache with atomic writes, manifest tracking,
and exponential-backoff retry on yfinance failures.
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

import pandas as pd
import yfinance as yf

logger        = logging.getLogger(__name__)
CACHE_DIR     = "data/cache"
MANIFEST_FILE = os.path.join(CACHE_DIR, "_manifest.json")


# ─── Manifest helpers ─────────────────────────────────────────────────────────

def _load_manifest() -> dict:
    if not os.path.exists(MANIFEST_FILE):
        return {}
    try:
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_manifest(m: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = MANIFEST_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2)
    os.replace(tmp, MANIFEST_FILE)


# ─── yfinance extraction helpers ─────────────────────────────────────────────

def _extract_ticker_df(raw: pd.DataFrame, ticker: str, is_multi: bool) -> Optional[pd.DataFrame]:
    if not is_multi:
        return raw.copy()
    try:
        levels = raw.columns.get_level_values
        if ticker in levels(0):
            return raw[ticker].copy()
        if ticker in levels(1):
            return raw.xs(ticker, axis=1, level=1).copy()
    except Exception as exc:
        logger.debug("[Cache] Extraction error for %s: %s", ticker, exc)
    return None


# ─── Public interface ─────────────────────────────────────────────────────────

def load_or_fetch(
    tickers:        List[str],
    required_start: str,
    required_end:   str,
    force_refresh:  bool = False,
    cfg=None,
) -> Dict[str, pd.DataFrame]:
    """
    Return a {ticker: OHLCV DataFrame} mapping.

    Tickers are normalised to .NS suffix before caching.  All files are
    written atomically (tmp → replace) so a crash mid-write never leaves a
    corrupt parquet on disk.
    """
    yf_timeout = getattr(cfg, "YF_BATCH_TIMEOUT", 120.0)

    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest = _load_manifest()

    standard_tickers = list({
        t if (t.endswith(".NS") or t.startswith("^")) else t + ".NS"
        for t in tickers
    })

    # HIGH-INTEGRITY FIX: Backend defensive fallback to prevent NaT (Not a Time) crashes
    # if an empty string ever bypasses the UI layer.
    if not required_start or str(required_start).strip() == "":
        required_start = "2020-01-01"

    fetch_start = (pd.Timestamp(required_start) - timedelta(days=400)).strftime("%Y-%m-%d")
    
    # Calculate the last valid business day to ensure we don't treat data from Friday 
    # as fresh on a Monday morning when Monday's data is actually missing.
    today_bday = (pd.Timestamp.today().normalize() - pd.offsets.BDay(1)).strftime("%Y-%m-%d")

    to_download: List[str]         = []
    market_data: Dict[str, pd.DataFrame] = {}

    for t in standard_tickers:
        entry         = manifest.get(t, {})
        fetched_at    = entry.get("fetched_at", "2000-01-01")
        covered_start = entry.get("covered_start", entry.get("first_date", "2099-01-01"))
        last_date     = entry.get("last_date", "2000-01-01")
        
        stale_time    = (datetime.now() - datetime.fromisoformat(fetched_at)) > timedelta(hours=20)
        stale_bday    = last_date < today_bday
        
        parquet_path  = os.path.join(CACHE_DIR, f"{t}.parquet")

        needs_download = (
            force_refresh
            or stale_time
            or stale_bday
            or not os.path.exists(parquet_path)
            or pd.Timestamp(covered_start) > pd.Timestamp(fetch_start)
        )
        
        if needs_download:
            to_download.append(t)
        else:
            try:
                market_data[t] = pd.read_parquet(parquet_path)
            except Exception:
                to_download.append(t)

    if to_download:
        logger.info(
            "[Cache] Batch downloading %d tickers (%s → %s) ...",
            len(to_download), fetch_start, required_end,
        )
        raw = _download_with_retry(to_download, fetch_start, required_end, yf_timeout)
        _ingest_raw(raw, to_download, manifest, market_data, fetch_start)

    _save_manifest(manifest)

    missing = set(standard_tickers) - set(market_data.keys())
    if missing:
        logger.warning("[Cache] %d ticker(s) unavailable: %s", len(missing), sorted(missing))

    return market_data


def _download_with_retry(
    tickers:     List[str],
    start:       str,
    end:         str,
    timeout:     float,
) -> pd.DataFrame:
    def _do() -> pd.DataFrame:
        for attempt in range(3):
            try:
                return yf.download(
                    tickers, start=start, end=end,
                    group_by="ticker", progress=False, auto_adjust=True,
                )
            except Exception as exc:
                if attempt == 2:
                    raise exc
                logger.debug("[Cache] Download attempt %d failed; retrying.", attempt + 1)
                time.sleep((2 ** attempt) + random.uniform(0, 1))

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_do).result(timeout=timeout)
    except Exception as exc:
        logger.error("[Cache] Download failed or timed out: %s. Using cached data.", exc)
        return pd.DataFrame()


def _ingest_raw(
    raw:         pd.DataFrame,
    to_download: List[str],
    manifest:    dict,
    market_data: dict,
    fetch_start: str,
) -> None:
    if raw.empty:
        logger.warning("[Cache] Download returned empty DataFrame.")
        return

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    now      = datetime.now().isoformat()

    for t in to_download:
        try:
            df = _extract_ticker_df(raw, t, is_multi)
            if df is None or df.empty or len(df) < 5:
                continue

            if getattr(df.index, "tz", None):
                df.index = df.index.tz_convert(None)
            df = df.dropna(how="all")

            path     = os.path.join(CACHE_DIR, f"{t}.parquet")
            tmp_path = path + ".tmp"
            df.to_parquet(tmp_path)
            os.replace(tmp_path, path)

            market_data[t] = df
            manifest[t]    = {
                "fetched_at":    now,
                "rows":          len(df),
                "first_date":    df.index[0].strftime("%Y-%m-%d"),
                "last_date":     df.index[-1].strftime("%Y-%m-%d"),
                "covered_start": fetch_start,
            }
        except Exception as exc:
            logger.warning("[Cache] Failed to process %s: %s", t, exc)


# ─── Utilities ────────────────────────────────────────────────────────────────

def get_cache_summary() -> pd.DataFrame:
    m = _load_manifest()
    if not m:
        return pd.DataFrame(columns=["ticker", "fetched_at", "rows", "last_date"])
    return pd.DataFrame([{"ticker": k, **v} for k, v in m.items()])


def invalidate_cache() -> None:
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)