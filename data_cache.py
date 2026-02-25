"""
data_cache.py — Persistent Atomic Downloader
============================================
v11.9 — Final Hardened Release

Full audit history of corrections applied across all review passes:

v11.2  Dynamic MultiIndex level parsing: _extract_ticker_df checks both
       Level 0 and Level 1 for the ticker symbol, resolving a silent drop
       bug where yfinance's group_by="ticker" placed tickers at Level 0
       but the previous hardcoded extractor only checked Level 1.
v11.6  Atomic parquet writes: each ticker file is written to a .tmp path
       first and then renamed atomically via os.replace(), preventing
       silent cache corruption if the process is interrupted mid-write.
       Warning logged when the returned universe is smaller than requested
       so callers are aware of a reduced universe.
v11.9  Data coverage validation: Cache now specifically tracks `covered_start`
       so historical backtests correctly force a re-download if the existing
       cache was populated by a shallow daily live scan request.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import yfinance as yf

logger        = logging.getLogger(__name__)
CACHE_DIR     = "data/cache"
MANIFEST_FILE = os.path.join(CACHE_DIR, "_manifest.json")


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


def _extract_ticker_df(
    raw: pd.DataFrame, ticker: str, is_multi: bool
) -> pd.DataFrame:
    """
    Robustly extract a single ticker's OHLCV DataFrame from a yfinance
    batch download result, dynamically locating the ticker in either
    MultiIndex level.
    """
    if not is_multi:
        return raw.copy()

    try:
        if ticker in raw.columns.get_level_values(0):
            return raw[ticker].copy()
        elif ticker in raw.columns.get_level_values(1):
            return raw.xs(ticker, axis=1, level=1).copy()
    except Exception as e:
        logger.debug(f"[Cache] Extraction error for {ticker}: {e}")

    return None


def load_or_fetch(
    tickers:        List[str],
    required_start: str,
    required_end:   str,
    force_refresh:  bool = False,
) -> Dict[str, pd.DataFrame]:

    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest = _load_manifest()

    # Normalise ticker symbols: indices keep '^', equities get '.NS'.
    standard_tickers = list({
        t if (t.endswith(".NS") or t.startswith("^")) else t + ".NS"
        for t in tickers
    })

    fetch_start = (
        pd.Timestamp(required_start) - timedelta(days=400)
    ).strftime("%Y-%m-%d")

    to_download = []
    market_data = {}

    for t in standard_tickers:
        entry         = manifest.get(t, {})
        fetched_at    = entry.get("fetched_at", "2000-01-01")
        # `covered_start` tracks the depth we previously asked yfinance for. 
        # Resolves the bug where cache hits falsely satisfy deep historical backtest requests.
        covered_start = entry.get("covered_start", entry.get("first_date", "2099-01-01"))
        
        stale = (
            datetime.now() - datetime.fromisoformat(fetched_at) > timedelta(hours=20)
        )
        parquet_path = os.path.join(CACHE_DIR, f"{t}.parquet")

        if force_refresh or stale or not os.path.exists(parquet_path):
            to_download.append(t)
        elif pd.Timestamp(covered_start) > pd.Timestamp(fetch_start):
            # Cache is fresh but too shallow to satisfy the historical request.
            to_download.append(t)
        else:
            try:
                market_data[t] = pd.read_parquet(parquet_path)
            except Exception:
                to_download.append(t)

    if to_download:
        logger.info(
            f"[Cache] Batch downloading {len(to_download)} tickers "
            f"({fetch_start} → {required_end}) ..."
        )
        raw = yf.download(
            to_download,
            start=fetch_start,
            end=required_end,
            group_by="ticker",
            progress=True,
            auto_adjust=True,
        )

        is_multi = isinstance(raw.columns, pd.MultiIndex)
        now      = datetime.now().isoformat()

        for t in to_download:
            try:
                df = _extract_ticker_df(raw, t, is_multi)

                if df is None or df.empty or len(df) < 5:
                    logger.debug(f"[Cache] Skipping {t}: no usable data returned.")
                    continue

                if getattr(df.index, "tz", None):
                    df.index = df.index.tz_convert(None)

                df = df.dropna(how="all")

                # Atomic write: commit to .tmp then rename so a crash never
                # leaves a partial file that would be read as valid on restart.
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
                    "covered_start": fetch_start, # Validates future depth checks 
                }

            except Exception as e:
                logger.warning(f"[Cache] Failed to process {t}: {e}")

    _save_manifest(manifest)

    missing = set(standard_tickers) - set(market_data.keys())
    if missing:
        logger.warning(
            f"[Cache] {len(missing)} ticker(s) unavailable — portfolio will run "
            f"with reduced universe. Missing: {sorted(missing)}"
        )

    return market_data


def get_cache_summary() -> pd.DataFrame:
    m = _load_manifest()
    if not m:
        return pd.DataFrame(columns=["ticker", "fetched_at", "rows", "last_date"])
    return pd.DataFrame([{"ticker": k, **v} for k, v in m.items()])


def invalidate_cache() -> None:
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)