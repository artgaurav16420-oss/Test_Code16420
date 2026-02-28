"""
momentum_engine.py — Institutional Risk Engine
===============================================
CVaR-constrained Mean-Variance Optimizer with full Transaction Cost formulation.

Architecture
------------
InstitutionalRiskEngine is a *stateless* optimizer.  All persistent portfolio
state — including exposure_multiplier, override flags, and equity history —
lives exclusively in PortfolioState.  No manual sync boilerplate at call sites.

Corporate actions
-----------------
yfinance ``auto_adjust=True`` back-adjusts all historical prices for splits and
dividends.  The backtest operates on this adjusted series chronologically, so a
pre-split purchase books proportionally more shares at the (lower adjusted) price,
exactly replicating the post-split share count.  No separate split ledger is needed.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import osqp
import scipy.sparse as sp
from sklearn.covariance import LedoitWolf

# HIGH-INTEGRITY FIX: Scope the warning filter so we don't silence critical OSQP/NumPy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logger = logging.getLogger(__name__)


# ─── Symbol helpers ───────────────────────────────────────────────────────────

def to_ns(sym: str) -> str:
    """Return ticker with .NS suffix; pass-through for indices (^) and already-suffixed."""
    if sym.startswith("^") or sym.endswith(".NS"):
        return sym
    return sym + ".NS"


def to_bare(sym: str) -> str:
    """Strip .NS suffix."""
    return sym[:-3] if sym.endswith(".NS") else sym


# ─── Enumerations & exceptions ────────────────────────────────────────────────

class OptimizationErrorType(Enum):
    NUMERICAL  = auto()
    INFEASIBLE = auto()
    DATA       = auto()


class OptimizationError(Exception):
    def __init__(self, message: str, error_type: OptimizationErrorType = OptimizationErrorType.NUMERICAL):
        super().__init__(message)
        self.error_type = error_type


# ─── Value objects ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:       str
    date:         pd.Timestamp
    delta_shares: int
    exec_price:   float
    slip_cost:    float
    direction:    str  # "BUY" | "SELL"


@dataclass
class SolverDiagnostics:
    status:            str
    gamma_intent:      float
    actual_weight:     float
    l_gamma:           float
    u_gamma:           float
    cvar_value:        float
    slack_value:       float
    sum_adv_limit:     float
    adv_binding_count: int      # number of weights at their ADV cap
    ridge_applied:     float
    cond_number:       float
    t_cvar:            int

    @property
    def budget_utilisation(self) -> float:
        return self.actual_weight / self.u_gamma if self.u_gamma > 0 else 0.0


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class UltimateConfig:
    # Portfolio construction
    INITIAL_CAPITAL:          float = 1_000_000.0
    MAX_POSITIONS:            int   = 15
    MAX_PORTFOLIO_RISK_PCT:   float = 0.12
    MAX_ADV_PCT:              float = 0.05
    IMPACT_COEFF:             float = 5e-4
    SIGNAL_ANNUAL_FACTOR:     int   = 252

    # CVaR
    CVAR_DAILY_LIMIT:         float = 0.022
    CVAR_ALPHA:               float = 0.95
    CVAR_LOOKBACK:            int   = 200
    CVAR_SENTINEL_MULTIPLIER: float = 2.5   

    # Exposure management
    DELEVERAGING_LIMIT:       float = 0.10
    MIN_EXPOSURE_FLOOR:       float = 0.25
    CAPITAL_ELASTICITY:       float = 0.15

    # Signal / optimizer
    HISTORY_GATE:             int   = 90
    HALFLIFE_FAST:            int   = 21
    HALFLIFE_SLOW:            int   = 63
    RISK_AVERSION:            float = 5.0
    SLACK_PENALTY:            float = 10.0
    DIMENSIONALITY_MULTIPLIER:int   = 3
    MAX_SECTOR_WEIGHT:        float = 0.30

    # Execution
    SLIPPAGE_BPS:             float = 20.0
    DECAY_FACTOR:             float = 0.85
    MIN_ADV_CRORES:           float = 50.0

    # Ghost position / data-glitch protection
    MAX_ABSENT_PERIODS:       int   = 3
    MAX_DECAY_ROUNDS:         int   = 3

    # Network / data
    YF_BATCH_TIMEOUT:         float = 120.0
    YF_ADV_TIMEOUT:           float = 60.0
    SECTOR_FETCH_TIMEOUT:     float = 8.0

    @property
    def EQUITY_HIST_CAP(self) -> int:
        return self.CVAR_LOOKBACK + 50


# ─── Portfolio state ──────────────────────────────────────────────────────────

@dataclass
class PortfolioState:
    weights:              Dict[str, float] = field(default_factory=dict)
    shares:               Dict[str, int]   = field(default_factory=dict)
    entry_prices:         Dict[str, float] = field(default_factory=dict)
    equity_hist:          List[float]      = field(default_factory=list)
    universe:             List[str]        = field(default_factory=list)
    cash:                 float            = 1_000_000.0
    exposure_multiplier:  float            = 1.0
    override_active:      bool             = False
    override_cooldown:    int              = 0
    consecutive_failures: int              = 0
    equity_hist_cap:      int              = 250
    absent_periods:       Dict[str, int]   = field(default_factory=dict)
    last_known_prices:    Dict[str, float] = field(default_factory=dict)
    decay_rounds:         int              = 0

    def update_exposure(
        self,
        regime_score:   float,
        realized_cvar:  float,
        cfg:            UltimateConfig,
        gross_exposure: float = 1.0,
    ) -> None:
        # Smooth step
        target = 1.0 / (1.0 + np.exp(-10.0 * (regime_score - 0.5)))
        new_mult = self.exposure_multiplier + float(np.clip(target - self.exposure_multiplier, -cfg.DELEVERAGING_LIMIT, cfg.DELEVERAGING_LIMIT))

        # HIGH-INTEGRITY FIX: Prevent double-deleveraging on transient CVaR spikes
        normalised_cvar = realized_cvar / max(float(gross_exposure), 0.05)
        breach = normalised_cvar > cfg.MAX_PORTFOLIO_RISK_PCT * 1.5

        if self.override_cooldown > 0:
            self.override_cooldown -= 1

        if breach and not self.override_active and self.override_cooldown == 0:
            override_mult = max(cfg.MIN_EXPOSURE_FLOOR, self.exposure_multiplier * 0.5)
            self.exposure_multiplier = min(new_mult, override_mult)
            self.override_active     = True
            self.override_cooldown   = 4
        else:
            self.exposure_multiplier = new_mult
            if not breach:
                self.override_active = False

        self.exposure_multiplier = float(np.clip(self.exposure_multiplier, cfg.MIN_EXPOSURE_FLOOR, 1.0))

    def realised_cvar(self, min_obs: int = 30) -> float:
        n = len(self.equity_hist)
        if n < min_obs:
            if n > 1:
                logger.debug(
                    "CVaR computed on only %d observations (min %d for stability); "
                    "returning 0.0 during warm-up.", n, min_obs
                )
            return 0.0

        rets  = np.log1p(pd.Series(self.equity_hist).pct_change(fill_method=None).clip(lower=-0.99)).dropna()
        if rets.empty:
            return 0.0
        var_q = rets.quantile(0.05)
        tail  = rets[rets <= var_q]
        return round(max(0.0, -float(tail.mean())), 10) if not tail.empty else 0.0

    def record_eod(self, prices: Dict[str, float]) -> None:
        pv = self.cash
        for sym, n_shares in self.shares.items():
            px = prices.get(sym)
            if px is not None:
                self.last_known_prices[sym] = float(px)
            else:
                px = self.last_known_prices.get(sym)
                if px is None:
                    logger.warning("record_eod: no price for %s and no last known price; treating as ₹0.", sym)
                else:
                    logger.debug("record_eod: carrying %s at last known price ₹%.2f.", sym, px)
            pv += n_shares * float(px or 0.0)

        pv_rounded = round(float(pv), 10)
        
        # HIGH-INTEGRITY FIX: Removed the flat-day deduplication (`if eq[-1] == pv_rounded: return`). 
        # Stripping identical consecutive values mathematically removes zero-return days, artificially inflating CVaR.
        self.equity_hist.append(pv_rounded)
        if len(self.equity_hist) > self.equity_hist_cap:
            self.equity_hist = self.equity_hist[-self.equity_hist_cap:]

    def to_dict(self) -> dict:
        def _r(v):
            if isinstance(v, float): return round(v, 10)
            if isinstance(v, dict):  return {k: _r(val) for k, val in sorted(v.items())}
            if isinstance(v, list):  return [_r(x) for x in v]
            return v
        return {
            "weights":              _r(self.weights),
            "shares":               dict(sorted(self.shares.items())),
            "entry_prices":         _r(self.entry_prices),
            "equity_hist":          _r(self.equity_hist),
            "universe":             self.universe,
            "cash":                 _r(self.cash),
            "exposure_multiplier":  _r(self.exposure_multiplier),
            "override_active":      self.override_active,
            "override_cooldown":    self.override_cooldown,
            "consecutive_failures": self.consecutive_failures,
            "equity_hist_cap":      self.equity_hist_cap,
            "absent_periods":       dict(sorted(self.absent_periods.items())),
            "last_known_prices":    _r(self.last_known_prices),
            "decay_rounds":         self.decay_rounds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioState":
        ps     = cls()
        errors: List[str] = []

        def _as_bool(value) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v = value.strip().lower()
                if v in {"true", "1", "yes", "y", "on"}:
                    return True
                if v in {"false", "0", "no", "n", "off"}:
                    return False
                raise ValueError(f"cannot parse bool from '{value}'")
            if isinstance(value, int):
                if value in (0, 1):
                    return bool(value)
                raise ValueError(f"integer bool flag must be 0/1, got {value}")
            raise TypeError(f"unsupported bool type: {type(value).__name__}")

        def _get(key, converter, default):
            try:
                return converter(d[key]) if key in d else default
            except Exception as exc:
                errors.append(f"{key}: {exc}")
                return default

        ps.weights              = _get("weights",              lambda v: {k: float(x) for k, x in v.items()}, {})
        ps.shares               = _get("shares",               lambda v: {k: int(x)   for k, x in v.items()}, {})
        ps.entry_prices         = _get("entry_prices",         lambda v: {k: float(x) for k, x in v.items()}, {})
        ps.equity_hist          = _get("equity_hist",          lambda v: [float(x) for x in v],                [])
        ps.universe             = _get("universe",             list,                                            [])
        ps.cash                 = _get("cash",                 float,                                           ps.cash)
        ps.exposure_multiplier  = _get("exposure_multiplier",  float,                                           1.0)
        # bool("False") is True; parse defensively for manual/legacy JSON states.
        ps.override_active      = _get("override_active",      _as_bool,                                        False)
        ps.override_cooldown    = _get("override_cooldown",    int,                                             0)
        ps.consecutive_failures = _get("consecutive_failures", int,                                             0)
        ps.equity_hist_cap      = _get("equity_hist_cap",      int,                                             250)
        ps.absent_periods       = _get("absent_periods",       lambda v: {k: int(x)   for k, x in v.items()}, {})
        ps.last_known_prices    = _get("last_known_prices",    lambda v: {k: float(x) for k, x in v.items()}, {})
        ps.decay_rounds         = _get("decay_rounds",         int,                                             0)

        if errors:
            logger.error("PortfolioState.from_dict: %d field(s) reset to defaults: %s", len(errors), errors)
        return ps


# ─── Execution ────────────────────────────────────────────────────────────────

def execute_rebalance(
    state:          PortfolioState,
    target_weights: np.ndarray,
    prices:         np.ndarray,
    active_symbols: List[str],
    cfg:            UltimateConfig,
    date_context=None,
    trade_log:      Optional[List[Trade]] = None,
    apply_decay:    bool = False,
) -> float:
    entry_slip = exit_slip = (cfg.SLIPPAGE_BPS / 2) / 10_000
    active_idx = {sym: i for i, sym in enumerate(active_symbols)}

    for sym, i in active_idx.items():
        state.last_known_prices[sym] = float(prices[i])
        state.absent_periods.pop(sym, None)   

    symbols_to_force_close: List[str] = []
    for sym in list(state.shares.keys()):
        if sym not in active_idx:
            count = state.absent_periods.get(sym, 0) + 1
            state.absent_periods[sym] = count
            if count >= cfg.MAX_ABSENT_PERIODS:
                logger.warning(
                    "execute_rebalance: %s absent for %d consecutive periods "
                    "(≥ MAX_ABSENT_PERIODS=%d) — treating as delisted; closing position.",
                    sym, count, cfg.MAX_ABSENT_PERIODS,
                )
                symbols_to_force_close.append(sym)
            else:
                logger.info(
                    "execute_rebalance: %s absent from data feed (period %d/%d); "
                    "carrying position at last known price ₹%.2f.",
                    sym, count, cfg.MAX_ABSENT_PERIODS,
                    state.last_known_prices.get(sym, 0.0),
                )

    pv = state.cash
    for sym, n_shares in state.shares.items():
        if sym in active_idx:
            pv += n_shares * float(prices[active_idx[sym]])
        elif sym not in symbols_to_force_close:
            carry_price = state.last_known_prices.get(sym, 0.0)
            pv += n_shares * carry_price

    hold_flat = False   
    if apply_decay:
        if state.decay_rounds >= cfg.MAX_DECAY_ROUNDS:
            logger.warning(
                "execute_rebalance: MAX_DECAY_ROUNDS=%d reached — holding positions flat "
                "instead of applying further %.0f%% decay.",
                cfg.MAX_DECAY_ROUNDS, (1 - cfg.DECAY_FACTOR) * 100,
            )
            hold_flat  = True
            apply_decay = False   
        else:
            state.decay_rounds += 1
            logger.info(
                "execute_rebalance: decay round %d/%d (factor=%.2f).",
                state.decay_rounds, cfg.MAX_DECAY_ROUNDS, cfg.DECAY_FACTOR,
            )

    new_weights:      Dict[str, float] = {}
    new_shares:       Dict[str, int]   = {}
    new_entry_prices: Dict[str, float] = dict(state.entry_prices)
    total_slippage = actual_notional = 0.0

    for i, sym in enumerate(active_symbols):
        if hold_flat:
            w = round(state.weights.get(sym, 0.0), 10)    
        elif apply_decay:
            w = round(state.weights.get(sym, 0.0) * cfg.DECAY_FACTOR, 10)
        else:
            w = round(float(target_weights[i]), 10)
        price = max(float(prices[i]), 1e-6)
        old_s = state.shares.get(sym, 0)
        s     = int(np.floor(w * pv / price)) if w > 0.001 else 0

        if s > 0:
            delta            = s - old_s
            slip             = abs(delta) * price * (entry_slip if delta > 0 else exit_slip)
            total_slippage  += slip
            actual_notional += s * price
            new_weights[sym] = w
            new_shares[sym]  = s

            if delta > 0:
                if old_s == 0:
                    new_entry_prices[sym] = price * (1.0 + entry_slip)
                else:
                    old_basis             = new_entry_prices.get(sym, price)
                    new_entry_prices[sym] = (old_basis * old_s + price * (1.0 + entry_slip) * delta) / s

            if delta != 0 and trade_log is not None and date_context is not None:
                trade_log.append(Trade(sym, date_context, delta, price, slip, "BUY" if delta > 0 else "SELL"))

        elif old_s > 0:
            slip            = old_s * price * exit_slip
            total_slippage += slip
            if trade_log is not None and date_context is not None:
                trade_log.append(Trade(sym, date_context, -old_s, price, slip, "SELL"))

    for sym in state.shares:
        if sym not in active_idx and sym not in symbols_to_force_close:
            new_shares[sym]       = state.shares[sym]
            new_weights[sym]      = state.weights.get(sym, 0.0)
            new_entry_prices[sym] = state.entry_prices.get(sym, 0.0)
            actual_notional      += new_shares[sym] * state.last_known_prices.get(sym, 0.0)

    for sym in symbols_to_force_close:
        close_price = state.last_known_prices.get(sym, 0.0)
        n_shares    = state.shares.get(sym, 0)
        if n_shares > 0 and close_price > 0:
            slip            = n_shares * close_price * exit_slip
            total_slippage += slip
            pv             += n_shares * close_price   
            if trade_log is not None and date_context is not None:
                trade_log.append(Trade(sym, date_context, -n_shares, close_price, slip, "SELL"))
        state.absent_periods.pop(sym, None)

    for sym in list(new_entry_prices):
        if sym not in new_shares:
            del new_entry_prices[sym]

    state.weights      = new_weights
    state.shares       = new_shares
    state.entry_prices = new_entry_prices
    # HIGH-INTEGRITY FIX: Prevent silent floating-point cash inversion below ₹0
    state.cash         = max(0.0, round(pv - actual_notional - total_slippage, 10))
    return round(total_slippage, 10)


# ─── Optimizer ────────────────────────────────────────────────────────────────

class InstitutionalRiskEngine:
    def __init__(self, cfg: UltimateConfig):
        self.cfg:       UltimateConfig              = cfg
        self.last_diag: Optional[SolverDiagnostics] = None

    def optimize(
        self,
        expected_returns:    np.ndarray,
        historical_returns:  pd.DataFrame,
        adv_shares:          np.ndarray,
        prices:              np.ndarray,
        portfolio_value:     float,
        prev_w:              Optional[np.ndarray] = None,
        exposure_multiplier: float                = 1.0,
        sector_labels:       Optional[np.ndarray] = None,
    ) -> np.ndarray:
        m = len(expected_returns)
        if m == 0:
            return np.array([])

        if len(prices) != m or len(adv_shares) != m:
            raise OptimizationError(
                "Input length mismatch across expected_returns/prices/adv_shares.",
                OptimizationErrorType.DATA,
            )

        if prev_w is not None and len(prev_w) != m:
            raise OptimizationError(
                "prev_w length must match expected_returns length.",
                OptimizationErrorType.DATA,
            )

        if not np.all(np.isfinite(expected_returns)):
            raise OptimizationError("expected_returns contains non-finite values.", OptimizationErrorType.DATA)
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise OptimizationError("prices must be finite and strictly positive.", OptimizationErrorType.DATA)
        if not np.all(np.isfinite(adv_shares)) or np.any(adv_shares < 0):
            raise OptimizationError("adv_shares must be finite and non-negative.", OptimizationErrorType.DATA)
        if not np.isfinite(portfolio_value) or portfolio_value <= 0:
            raise OptimizationError("portfolio_value must be finite and strictly positive.", OptimizationErrorType.DATA)

        clean_rets = historical_returns.replace([np.inf, -np.inf], np.nan).ffill().dropna()
        T = len(clean_rets)
        min_rows = self.cfg.DIMENSIONALITY_MULTIPLIER * m
        if T < min_rows:
            raise OptimizationError(f"Insufficient history: {T} rows for {m} assets.", OptimizationErrorType.DATA)

        # HIGH-INTEGRITY FIX: Revert sentinel to check the Equally-Weighted (EW) basket CVaR. 
        # Checking the worst single asset is mathematically too strict for momentum universes, 
        # as it forces the engine to abort if even ONE selected stock is highly volatile. 
        # The OSQP solver's mathematical purpose is to dynamically size down volatile assets.
        ew_rets  = clean_rets.mean(axis=1)
        var_95   = ew_rets.quantile(1 - self.cfg.CVAR_ALPHA)
        ew_cvar  = -float(ew_rets[ew_rets <= var_95].mean()) if not ew_rets.empty else 0.0
        
        sentinel = self.cfg.CVAR_DAILY_LIMIT * self.cfg.CVAR_SENTINEL_MULTIPLIER
        if ew_cvar > sentinel:
            raise OptimizationError(f"Selection EW-CVaR {ew_cvar:.2%} exceeds sentinel {sentinel:.2%}.", OptimizationErrorType.INFEASIBLE)

        # HIGH-INTEGRITY FIX: LedoitWolf is explicitly stateful (caches `.covariance_`). 
        # Moving instantiation inside `optimize()` guarantees thread-safe, purely stateless runs.
        lw = LedoitWolf()
        lw.fit(clean_rets)
        Sigma     = lw.covariance_
        ridge     = 1e-8 * float(np.trace(Sigma))
        Sigma_reg = Sigma + ridge * np.eye(m)

        gamma     = float(np.clip(exposure_multiplier, self.cfg.MIN_EXPOSURE_FLOOR, 1.0))
        adv_limit = np.clip((adv_shares * prices * self.cfg.MAX_ADV_PCT) / portfolio_value, 1e-9, 0.40)
        
        l_gamma = max(self.cfg.MIN_EXPOSURE_FLOOR, gamma * (1.0 - self.cfg.CAPITAL_ELASTICITY))
        u_gamma = min(1.0, gamma * (1.0 + self.cfg.CAPITAL_ELASTICITY))
        
        max_possible_weight = 0.0
        if sector_labels is not None:
            labels = np.asarray(sector_labels, dtype=int)
            for sec_id in np.unique(labels):
                mask = labels == sec_id
                sec_max = float(np.sum(adv_limit[mask]))
                if mask.sum() >= 2: 
                    sec_max = min(sec_max, self.cfg.MAX_SECTOR_WEIGHT)
                max_possible_weight += sec_max
        else:
            max_possible_weight = float(np.sum(adv_limit))

        u_gamma = min(u_gamma, max_possible_weight)
        if max_possible_weight < l_gamma:
            l_gamma = max_possible_weight * 0.99
            
        if np.sum(adv_limit) < l_gamma:
            l_gamma = np.sum(adv_limit) * 0.99
            
        u_gamma = max(u_gamma, l_gamma) 

        impact = np.clip(self.cfg.IMPACT_COEFF * portfolio_value / (np.maximum(prices, 1.0) * np.maximum(adv_shares, 1.0)), 0.0, 1e4)
        T_cvar     = min(T, self.cfg.CVAR_LOOKBACK)
        losses     = -clean_rets.iloc[-T_cvar:].values      
        n_vars     = 2 * m + 1 + T_cvar + 1
        prev_w_arr = prev_w if prev_w is not None else np.zeros(m)

        P_w = 2.0 * (self.cfg.RISK_AVERSION * Sigma_reg + np.diag(impact))
        P_aux = sp.eye(n_vars - m, format="csc") * 1e-6
        P   = sp.block_diag([sp.csc_matrix(P_w), P_aux], format="csc")
        
        q        = np.zeros(n_vars)
        q[:m]    = -expected_returns - 2.0 * impact * prev_w_arr
        q[m:2*m] = self.cfg.SLIPPAGE_BPS / 10_000.0   
        q[-1]    = self.cfg.SLACK_PENALTY

        A_parts: list = []
        l_parts: list = []
        u_parts: list = []

        budget_data = np.ones(m)
        budget_row  = sp.csc_matrix((budget_data, (np.zeros(m, int), np.arange(m))), shape=(1, n_vars))
        A_parts.append(budget_row)
        l_parts.append([l_gamma]); u_parts.append([u_gamma])

        A_scen = sp.lil_matrix((T_cvar, n_vars))
        A_scen[:, :m]  = losses
        A_scen[:, 2*m] = -1.0
        for i in range(T_cvar):
            A_scen[i, 2*m + 1 + i] = -1.0
        A_parts.append(A_scen.tocsc())
        l_parts.append([-np.inf] * T_cvar); u_parts.append([0.0] * T_cvar)

        scen_c = 1.0 / (T_cvar * (1.0 - self.cfg.CVAR_ALPHA))
        lim    = sp.lil_matrix((1, n_vars))
        lim[0, 2*m]                       = 1.0
        lim[0, 2*m+1:2*m+1+T_cvar]        = scen_c
        lim[0, -1]                         = -1.0
        A_parts.append(lim.tocsc())
        l_parts.append([-np.inf]); u_parts.append([self.cfg.CVAR_DAILY_LIMIT])

        lb = np.full(n_vars, -np.inf)
        ub = np.full(n_vars,  np.inf)
        lb[:m]                   = 0.0;       ub[:m]                   = adv_limit
        lb[m:2*m]                = 0.0        
        lb[2*m+1:2*m+1+T_cvar]  = 0.0        
        lb[-1]                   = 0.0        
        A_parts.append(sp.eye(n_vars, format="csc"))
        l_parts.append(lb.tolist()); u_parts.append(ub.tolist())

        if sector_labels is not None:
            labels = np.asarray(sector_labels, dtype=int)
            for sec_id in np.unique(labels):
                mask = labels == sec_id
                if mask.sum() < 2: continue
                sec_row = sp.lil_matrix((1, n_vars))
                sec_row[0, np.where(mask)[0]] = 1.0
                A_parts.append(sec_row.tocsc())
                l_parts.append([0.0]); u_parts.append([self.cfg.MAX_SECTOR_WEIGHT])

        tc = sp.lil_matrix((2 * m, n_vars))
        for i in range(m):
            tc[2*i,   i] =  1.0; tc[2*i,   m+i] = -1.0
            tc[2*i+1, i] = -1.0; tc[2*i+1, m+i] = -1.0
        A_parts.append(tc.tocsc())
        tc_u = []
        for p in prev_w_arr:
            tc_u.extend([p, -p])
        l_parts.append([-np.inf] * (2 * m)); u_parts.append(tc_u)

        A = sp.vstack(A_parts, format="csc")
        l = np.array([v for block in l_parts for v in block], dtype=float)
        u = np.array([v for block in u_parts for v in block], dtype=float)

        prob = osqp.OSQP()
        prob.setup(
            P, q, A, l, u,
            verbose=False, eps_abs=1e-4, eps_rel=1e-4, 
            polish=True, adaptive_rho=True, max_iter=50000        
        )
        res = prob.solve()

        if res.info.status not in ("solved", "solved inaccurate", "solved_inaccurate"):
            raise OptimizationError(f"OSQP status: {res.info.status}", OptimizationErrorType.NUMERICAL)

        w_opt  = np.maximum(res.x[:m], 0.0)
        eta    = res.x[2*m]
        z_vec  = res.x[2*m+1: 2*m+1+T_cvar]
        achieved_cvar     = float(eta + np.sum(z_vec) / (T_cvar * (1.0 - self.cfg.CVAR_ALPHA)))
        adv_binding_count = int(np.sum(w_opt >= adv_limit - 1e-6))

        self.last_diag = SolverDiagnostics(
            status=res.info.status,
            gamma_intent=gamma,
            actual_weight=float(np.sum(w_opt)),
            l_gamma=l_gamma,
            u_gamma=u_gamma,
            cvar_value=achieved_cvar,
            slack_value=float(res.x[-1]),
            sum_adv_limit=float(np.sum(adv_limit)),
            adv_binding_count=adv_binding_count,
            ridge_applied=ridge,
            cond_number=float(np.linalg.cond(Sigma_reg)),
            t_cvar=T_cvar,
        )
        return w_opt
