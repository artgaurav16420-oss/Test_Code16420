"""
signals.py — Deterministic Regime & Momentum Kernel
=====================================================
Multi-timeframe EWMA momentum with volatility-adjusted regime scoring.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_regime_score(idx_hist: Optional[pd.DataFrame]) -> float:
    """
    Deterministic regime score in [0, 1].

    Combines a logistic transform of price-vs-SMA200 with a 20-day realised
    volatility overlay that shaves 10 points when annualised vol exceeds 18%.

    Caller must pass T-1 sliced history (``iloc[:-1]``) to prevent look-ahead.
    Returns 0.5 (neutral) on any data quality issue rather than raising.
    """
    if idx_hist is None or len(idx_hist) < 200:
        return 0.5

    if "Close" not in idx_hist.columns:
        return 0.5

    if not idx_hist.index.is_monotonic_increasing:
        logger.warning(
            "compute_regime_score: index not monotonic (duplicate dates from yfinance). "
            "Deduplicating and continuing."
        )
        idx_hist = idx_hist[~idx_hist.index.duplicated(keep="last")]
        if len(idx_hist) < 200:
            return 0.5

    close  = idx_hist["Close"]
    sma200 = float(close.rolling(200).mean().iloc[-1])
    last   = float(close.iloc[-1])

    if sma200 <= 0 or not np.isfinite(sma200) or not np.isfinite(last):
        return 0.5

    score = 1.0 / (1.0 + np.exp(-20.0 * (last / sma200 - 1.0)))

    # Volatility overlay: deleverage when short-term realised vol is elevated.
    rets_20 = close.pct_change(fill_method=None).tail(20)
    if len(rets_20) == 20:
        vol_20 = float(rets_20.std() * np.sqrt(252))
        if vol_20 > 0.18:
            score = max(0.0, score - 0.10)

    return round(float(score), 10)


def compute_adv(market_data: dict, active: List[str]) -> np.ndarray:
    """
    Compute 20-day average daily volume (in shares) for each symbol.

    Uses the *minimum* of the rolling 20-day mean and the most recent single day,
    which is the conservative choice for liquidity sizing.
    """
    from momentum_engine import to_ns

    adv = []
    for sym in active:
        ns = to_ns(sym)
        try:
            vol  = market_data[ns]["Volume"]
            ma20 = float(vol.rolling(20).mean().iloc[-1])
            last = float(vol.iloc[-1])
            val  = min(ma20, last)
            adv.append(val if np.isfinite(val) else 0.0)
        except Exception:
            adv.append(0.0)
    return np.array(adv, dtype=float)


def generate_signals(
    log_rets:      pd.DataFrame,
    adv_arr:       np.ndarray,
    history_gate:  int,
    max_positions: int,
    prev_weights:  Optional[Dict[str, float]] = None,
    halflife_fast: int = 21,
    halflife_slow: int = 63,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Score assets by blended multi-timeframe EWMA momentum.

    Parameters
    ----------
    log_rets:      T × N log-return DataFrame; caller must pass T-1 slice.
    adv_arr:       N-length array of average daily volume in shares.
    history_gate:  Minimum non-NaN observations required to be selectable.
    max_positions: Maximum number of assets to return in sel_idx.
    prev_weights:  Dict[symbol → current weight] for continuity bonus.
    halflife_fast: EWMA half-life for the fast momentum component (trading days).
    halflife_slow: EWMA half-life for the slow momentum component (trading days).

    Returns
    -------
    raw_daily:  Raw blended score per asset.
    adj_scores: Cross-sectionally z-scored, clipped, bonus-adjusted scores.
    sel_idx:    Indices of top-ranked assets (sorted ascending, so slice [-n:]).
    """
    if prev_weights is None:
        prev_weights = {}

    active = list(log_rets.columns)

    fast = log_rets.ewm(halflife=halflife_fast).mean().iloc[-1].values.astype(float)
    slow = log_rets.ewm(halflife=halflife_slow).mean().iloc[-1].values.astype(float)
    raw_daily = 0.5 * fast + 0.5 * slow

    mu  = float(np.nanmean(raw_daily))
    std = float(max(np.nanstd(raw_daily), 1e-8))
    adj_scores = np.clip((raw_daily - mu) / std, -3.0, 3.0)

    for i, sym in enumerate(active):
        if prev_weights.get(sym, 0.0) > 0.001:
            adj_scores[i] += 0.05                         # continuity bonus
        if int(log_rets[sym].notna().sum()) < history_gate:
            adj_scores[i] = -np.inf                       # history gate

    # CRITICAL: map any remaining NaN (e.g. from EWMA on all-NaN columns) to -inf
    # *before* argsort.  NumPy sorts NaN as if it were +inf, so without this,
    # NaN assets consume the top-K selection slots and the subsequent isfinite
    # filter silently truncates the portfolio.
    adj_scores = np.where(np.isfinite(adj_scores), adj_scores, -np.inf)

    sel_idx = [
        i for i in np.argsort(adj_scores)[-max_positions:]
        if adj_scores[i] > -np.inf
    ]

    return raw_daily, adj_scores, sel_idx