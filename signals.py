"""
signals.py — Deterministic Regime & Momentum Kernel
=====================================================
Multi-timeframe EWMA momentum with volatility-adjusted regime scoring.
"""

from __future__ import annotations

# NOTE: UltimateConfig is imported under TYPE_CHECKING to avoid a circular
# import (momentum_engine → signals → momentum_engine). The `from __future__
# import annotations` directive at the top of this file makes all annotations
# lazy strings at runtime, so UltimateConfig is never evaluated at import time.
# DO NOT remove `from __future__ import annotations` without updating this block.
import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from momentum_engine import UltimateConfig

logger = logging.getLogger(__name__)


def compute_regime_score(idx_hist: Optional[pd.DataFrame]) -> float:
    """
    Deterministic regime score in [0, 1].

    Combines a logistic transform of price-vs-SMA200 with a 20-day realised
    volatility overlay that shaves points when annualised vol exceeds 18%.
    Caller must pass T-1 sliced history to prevent look-ahead.
    Returns 0.5 (neutral) on any data quality issue rather than raising.
    """
    if idx_hist is None or len(idx_hist) < 200:
        return 0.5
    if "Close" not in idx_hist.columns:
        return 0.5

    if not idx_hist.index.is_monotonic_increasing:
        logger.warning(
            "compute_regime_score: index not monotonic — deduplicating and continuing."
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

    # Multiplicative decay preserves proportional effect across the full [0, 1]
    # range, avoiding the boundary distortion of absolute subtraction.
    rets_20 = close.pct_change(fill_method=None).tail(20)
    if len(rets_20) == 20:
        vol_20 = float(rets_20.std() * np.sqrt(252))
        if vol_20 > 0.18:
            score *= 0.85

    return round(float(score), 10)


def compute_single_adv(series: pd.Series) -> float:
    """
    Core ADV calculation logic shared by both the daily engine and backtester
    to strictly enforce mathematical parity.

    Note: For live use, pass the full volume series directly. For backtesting,
    the caller (backtest_engine._build_adv_vector) is responsible for applying
    the T-1 slice *before* calling this function — this function is intentionally
    slice-agnostic to remain composable.
    """
    try:
        clean_series = series.replace(0, np.nan).ffill().fillna(0)
        if clean_series.empty:
            return 0.0
        ma20 = float(clean_series.rolling(20, min_periods=1).mean().iloc[-1])
        last = float(clean_series.iloc[-1])
        val  = min(ma20, last)
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def compute_adv(market_data: dict, active: List[str]) -> np.ndarray:
    """
    Compute 20-day average daily volume (in shares) for each symbol.

    Live use only — no T-1 slice is applied here. For backtesting, use
    backtest_engine._build_adv_vector, which performs the T-1 slice before
    delegating to compute_single_adv.
    """
    from momentum_engine import to_ns

    adv = []
    for sym in active:
        ns = to_ns(sym)
        if ns in market_data:
            adv.append(compute_single_adv(market_data[ns]["Volume"]))
        else:
            adv.append(0.0)
    return np.array(adv, dtype=float)


def generate_signals(
    log_rets:     pd.DataFrame,
    adv_arr:      np.ndarray,
    cfg:          "UltimateConfig",
    prev_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Score assets by blended multi-timeframe EWMA momentum.

    Parameters
    ----------
    log_rets     : T × N log-return DataFrame; caller must pass T-1 slice.
    adv_arr      : N-length array of average daily volume in shares (T-1 slice).
                   Must be the same length as log_rets.columns. Assets with
                   ADV == 0 are disqualified before ranking. This acts as a
                   first-pass liquidity gate before the optimizer's position-size
                   constraint, which is critical in decay/fallback mode where the
                   optimizer is bypassed and the last valid weights are scaled —
                   an illiquid stock that slips past this gate can remain in the
                   portfolio indefinitely.
    cfg          : UltimateConfig containing all tunable signal parameters.
    prev_weights : Dict[symbol → current weight] for continuity bonus.

    Returns
    -------
    raw_daily  : Raw blended score per asset (N-length float array).
    adj_scores : Cross-sectionally z-scored, clipped, bonus-adjusted scores.
    sel_idx    : Indices of top-ranked assets (after all gates applied).
    """
    if prev_weights is None:
        prev_weights = {}

    # HIGH-INTEGRITY FIX: Validation guard to prevent hard math crashes 
    # if upstream filters remove all valid trading data
    if log_rets.empty or log_rets.isna().all().all():
        raise ValueError("generate_signals: log_rets contains no valid data.")

    active = list(log_rets.columns)

    if len(adv_arr) != len(active):
        raise ValueError(
            f"generate_signals: adv_arr length {len(adv_arr)} != "
            f"log_rets columns {len(active)}. Caller must align arrays before calling."
        )

    # min_periods enforced to block ghost signals from newly listed stocks
    # that have a sparse array of NaNs inflating the EWMA result.
    fast = log_rets.ewm(
        halflife=cfg.HALFLIFE_FAST, min_periods=max(1, cfg.HALFLIFE_FAST // 2)
    ).mean().iloc[-1].values.astype(float)

    slow = log_rets.ewm(
        halflife=cfg.HALFLIFE_SLOW, min_periods=max(1, cfg.HALFLIFE_SLOW // 2)
    ).mean().iloc[-1].values.astype(float)

    raw_daily = 0.5 * fast + 0.5 * slow

    mu  = float(np.nanmean(raw_daily))
    std = float(max(np.nanstd(raw_daily), 1e-8))
    adj_scores = np.clip((raw_daily - mu) / std, -cfg.Z_SCORE_CLIP, cfg.Z_SCORE_CLIP)

    # ── Per-asset gates ───────────────────────────────────────────────────────

    for i, sym in enumerate(active):
        # Continuity bonus: reward existing positions to reduce unnecessary churn.
        if prev_weights.get(sym, 0.0) > 0.001:
            adj_scores[i] += cfg.CONTINUITY_BONUS
        # History gate: require sufficient non-NaN observations.
        if int(log_rets[sym].notna().sum()) < cfg.HISTORY_GATE:
            adj_scores[i] = -np.inf

    # LIQUIDITY GATE: disqualify any stock with zero or missing ADV.
    # Operates independently of the optimizer's position-size constraint so
    # that illiquid stocks are blocked even when the optimizer is in decay mode.
    for i, adv_val in enumerate(adv_arr):
        if not np.isfinite(adv_val) or adv_val <= 0:
            adj_scores[i] = -np.inf
            logger.debug(
                "Liquidity gate: %s disqualified (ADV=%.0f).",
                active[i],
                adv_val if np.isfinite(adv_val) else float("nan"),
            )

    # FALLING KNIFE GATE: disqualify stocks down more than KNIFE_THRESHOLD
    # over the last KNIFE_WINDOW trading days. Enforces a light absolute-
    # momentum check on top of the cross-sectional z-score ranking.
    if len(log_rets) >= cfg.KNIFE_WINDOW:
        ret_nd = log_rets.iloc[-cfg.KNIFE_WINDOW:].sum().values
        for i, rn in enumerate(ret_nd):
            if np.isfinite(rn) and rn < cfg.KNIFE_THRESHOLD:
                adj_scores[i] = -np.inf
                logger.debug(
                    "Falling knife gate: %s disqualified (%dd log-ret=%.2f%%).",
                    active[i], cfg.KNIFE_WINDOW, rn * 100,
                )

    # Map any remaining NaN to -inf before argsort. NumPy treats NaN as +inf
    # during sort, which would cause NaN assets to silently consume top-K slots.
    adj_scores = np.where(np.isfinite(adj_scores), adj_scores, -np.inf)

    sel_idx = [
        i for i in np.argsort(adj_scores)[-cfg.MAX_POSITIONS:]
        if adj_scores[i] > -np.inf
    ]

    return raw_daily, adj_scores, sel_idx