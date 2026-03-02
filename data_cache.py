"""
data_cache.py — Persistent Atomic Downloader
=============================================
Parquet-backed cache with atomic writes, manifest tracking,
and ThreadPoolExecutor isolation for multiprocessing safety on all platforms.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger         = logging.getLogger(__name__)
CACHE_DIR      = "data/cache"
MANIFEST_FILE  = os.path.join(CACHE_DIR, "_manifest.json")
SCHEMA_VERSION = 1


# ─── Worker ───────────────────────────────────────────────────────────────────

def _yf_fetch_worker(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Downloads OHLCV data via yfinance with exponential-backoff retry on
    transient network failures. Raises on the third consecutive failure.
    """
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
    return pd.DataFrame()


# ─── Manifest helpers ─────────────────────────────────────────────────────────

def _load_manifest() -> dict:
    """Loads the manifest with backward compatibility for flat legacy structures."""
    if not os.path.exists(MANIFEST_FILE):
        return {"schema_version": SCHEMA_VERSION, "entries": {}}
    try:
        with open(MANIFEST_FILE) as f:
            m = json.load(f)
            # Legacy migration: convert flat dict to versioned 'entries' structure.
            if "schema_version" not in m:
                return {"schema_version": SCHEMA_VERSION, "entries": m}
            return m
    except Exception:
        return {"schema_version": SCHEMA_VERSION, "entries": {}}


def _save_manifest(m: dict) -> None:
    """Atomic write for the JSON manifest using a temp-file swap."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = MANIFEST_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2)
    os.replace(tmp, MANIFEST_FILE)


# ─── Data Validation Gate ────────────────────────────────────────────────────

def _is_valid_dataframe(df: Optional[pd.DataFrame]) -> bool:
    """
    Strict validation gate blocking ingestion of corrupted OHLCV parquets.
    Checks for index continuity, price variance, and presence of close data.
    """
    if df is None or df.empty or len(df) < 5:
        return False

    # 1. Uniqueness & Monotonicity
    if not (df.index.is_unique and df.index.is_monotonic_increasing):
        return False

    # 2. Continuity: reject if any gap exceeds 30 calendar days.
    #    Suspended stocks remain excluded until the cache is manually invalidated.
    if len(df) > 1:
        max_gap = df.index.to_series().diff().dt.days.max()
        if max_gap > 30:
            return False

    # 3. Presence & Variance
    if "Close" not in df.columns or df["Close"].isnull().all():
        return False
    if df["Close"].nunique() <= 1:
        return False

    return True


# ─── Downloader Implementation ───────────────────────────────────────────────

def _download_with_timeout(tickers: List[str], start: str, end: str, timeout: float) -> pd.DataFrame:
    """
    Executes the yfinance download in a background thread with a hard timeout.

    Uses ThreadPoolExecutor rather than ProcessPoolExecutor: yfinance releases
    the GIL during network I/O so threading is efficient, and it avoids the
    Windows multiprocessing 'spawn' requirement that causes recursive import
    errors when called outside a __main__ guard.

    Note: if the underlying socket blocks at the OS level, the background thread
    may continue running after TimeoutError is raised. This is a known limitation
    of Python threads — they cannot be forcefully killed. In practice yfinance
    respects socket timeouts and this scenario is rare.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_yf_fetch_worker, tickers, start, end)
            return future.result(timeout=timeout)
    except TimeoutError:
        logger.error(
            "[Cache] Thread pool fetch timed out after %.1fs. "
            "Thread may still be running in background.",
            timeout,
        )
        return pd.DataFrame()
    except Exception as exc:
        logger.error("[Cache] Thread pool fetch failed: %s", exc)
        return pd.DataFrame()


def _extract_ticker_df(raw: pd.DataFrame, ticker: str, is_multi: bool) -> Optional[pd.DataFrame]:
    """Extracts a single ticker's OHLCV DataFrame from a yf.download multi-index result."""
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


def _ingest_raw(raw: pd.DataFrame, to_download: List[str], fetch_start: str) -> dict:
    """
    Validates each ticker's data and writes individual parquets atomically.
    Returns a sub-manifest of successfully written tickers (staged commit).
    Only tickers that pass validation and are written to disk appear in the
    returned dict — callers use this to update the master manifest.
    """
    sub_manifest = {}
    if raw is None or raw.empty:
        return sub_manifest

    is_multi = isinstance(raw.columns, pd.MultiIndex)
    now      = datetime.now().isoformat()

    for t in to_download:
        try:
            df = _extract_ticker_df(raw, t, is_multi)

            if not _is_valid_dataframe(df):
                logger.debug("[Cache] Validation failed for %s.", t)
                continue

            if getattr(df.index, "tz", None):
                df.index = df.index.tz_convert(None)
            df = df.dropna(how="all")

            path     = os.path.join(CACHE_DIR, f"{t}.parquet")
            tmp_path = path + ".tmp"
            df.to_parquet(tmp_path)
            os.replace(tmp_path, path)

            sub_manifest[t] = {
                "fetched_at":    now,
                "rows":          len(df),
                "first_date":    df.index[0].strftime("%Y-%m-%d"),
                "last_date":     df.index[-1].strftime("%Y-%m-%d"),
                "covered_start": fetch_start,
            }
        except Exception as exc:
            logger.warning("[Cache] Failed to process %s: %s", t, exc)

    return sub_manifest


# ─── Public Interface ────────────────────────────────────────────────────────

def load_or_fetch(
    tickers:        List[str],
    required_start: str,
    required_end:   str,
    force_refresh:  bool = False,
    cfg=None,
) -> Dict[str, pd.DataFrame]:
    """
    Main entry point for retrieving market data.

    Checks the manifest for each ticker to determine staleness, loads fresh
    data from disk where available, and downloads only what is missing or stale.
    Implements a staged commit: the manifest is only updated for tickers that
    were successfully validated and written to disk.
    """
    yf_timeout = getattr(cfg, "YF_BATCH_TIMEOUT", 120.0)
    os.makedirs(CACHE_DIR, exist_ok=True)

    full_manifest    = _load_manifest()
    manifest_entries = full_manifest["entries"]

    standard_tickers = list({
        t if (t.endswith(".NS") or t.startswith("^")) else t + ".NS"
        for t in tickers
    })

    if not required_start or str(required_start).strip() == "":
        required_start = "2020-01-01"

    # Request extra history ahead of required_start to warm up indicators.
    fetch_start = (pd.Timestamp(required_start) - timedelta(days=400)).strftime("%Y-%m-%d")
    today_bday  = (pd.Timestamp.today().normalize() - pd.offsets.BDay(1)).strftime("%Y-%m-%d")

    to_download: List[str]               = []
    market_data: Dict[str, pd.DataFrame] = {}

    for t in standard_tickers:
        entry         = manifest_entries.get(t, {})
        fetched_at    = entry.get("fetched_at", "2000-01-01")
        covered_start = entry.get("covered_start", entry.get("first_date", "2099-01-01"))
        last_date     = entry.get("last_date", "2000-01-01")

        stale_time   = (datetime.now() - datetime.fromisoformat(fetched_at)) > timedelta(hours=20)
        stale_bday   = last_date < today_bday
        parquet_path = os.path.join(CACHE_DIR, f"{t}.parquet")

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
        raw = _download_with_timeout(to_download, fetch_start, required_end, yf_timeout)

        # Staged commit: only update manifest for tickers successfully written to disk.
        success_sub_manifest = _ingest_raw(raw, to_download, fetch_start)

        successful_tickers = list(success_sub_manifest.keys())
        success_rate       = len(successful_tickers) / len(to_download) if to_download else 1.0
        missing_tickers    = [t for t in to_download if t not in successful_tickers]

        if missing_tickers:
            if success_rate >= 0.8:
                if len(missing_tickers) <= 10:
                    logger.warning("[Cache] %d missed. Surgical retry...", len(missing_tickers))
                    for mt in missing_tickers:
                        raw_single = _download_with_timeout(
                            [mt], fetch_start, required_end,
                            getattr(cfg, "YF_ADV_TIMEOUT", 60.0),
                        )
                        success_sub_manifest.update(_ingest_raw(raw_single, [mt], fetch_start))
                else:
                    logger.warning(
                        "[Cache] %d missed tickers exceeds surgical cap. Skipping.",
                        len(missing_tickers),
                    )
            else:
                logger.error(
                    "[Cache] Batch success %.1f%% < 80%%. Aborting retries.",
                    success_rate * 100,
                )

        # Final atomic commit to the manifest.
        if success_sub_manifest:
            manifest_entries.update(success_sub_manifest)
            _save_manifest(full_manifest)

            for t in success_sub_manifest:
                try:
                    market_data[t] = pd.read_parquet(os.path.join(CACHE_DIR, f"{t}.parquet"))
                except Exception:
                    pass

    return market_data


def get_cache_summary() -> pd.DataFrame:
    """Returns a DataFrame summarising cached tickers for diagnostic purposes."""
    m       = _load_manifest()
    entries = m.get("entries", {})
    if not entries:
        return pd.DataFrame(columns=["ticker", "fetched_at", "rows", "last_date"])
    return pd.DataFrame([{"ticker": k, **v} for k, v in entries.items()])


def invalidate_cache() -> None:
    """Removes all cached parquet files and the manifest."""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)