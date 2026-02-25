"""
ultimate_momentum_v10_hardened.py — Institutional Risk Engine
=============================================================
v11.9 — Final Hardened Release

Full audit history of corrections applied across all review passes:

v11.4  Missing t_cvar argument added to the early-exit _set_diag() calls
       (no_valid_cols and dimensionality_guard) to prevent TypeError
       exceptions during backtest abort sequences.
v11.6  ADV dual-variable offset bug: binding constraint tally was previously
       read from res.y with an incorrect index offset. Now deduced directly
       from the primal solution: a stock is binding when
       w_opt[i] >= adv_limit[i] - 1e-4.
v11.8  ANNUAL_FACTOR renamed to SIGNAL_ANNUAL_FACTOR throughout UltimateConfig
       to distinguish signal-plane annualisation (252 trading days) from the
       dynamic Sharpe obs_per_year introduced in backtest_engine v11.7.
v11.9  Diagnostic reframing: Compression Ratio > 100% caused semantic confusion.
       Replaced with Budget Utilisation to explicitly reflect actual allocated 
       capital as a percentage of the mathematically allowable budget cap.
       Log noise: CVaR window warm-up warnings safely downgraded to debug.
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import osqp
import scipy.sparse as sp
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DimensionalityError(Exception):
    pass


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
    adv_binding_count: int
    ridge_applied:     float
    cond_number:       float
    t_cvar:            int

    @property
    def budget_utilisation(self) -> float:
        """Percentage of allowable budget ceiling successfully deployed."""
        return self.actual_weight / self.u_gamma if self.u_gamma > 0 else 0.0


@dataclass
class UltimateConfig:
    INITIAL_CAPITAL:           float = 1_000_000.0
    MAX_POSITIONS:             int   = 10
    MAX_PORTFOLIO_RISK_PCT:    float = 0.12
    MAX_ADV_PCT:               float = 0.05
    IMPACT_COEFF:              float = 1e-4
    # Used only for display-plane annualisation of expected return signals.
    # Distinct from the dynamic obs_per_year used for Sharpe in _compute_metrics.
    SIGNAL_ANNUAL_FACTOR:      int   = 252
    CVAR_DAILY_LIMIT:          float = 0.035
    CVAR_ALPHA:                float = 0.95
    CVAR_LOOKBACK:             int   = 200
    DELEVERAGING_LIMIT:        float = 0.10
    MIN_EXPOSURE_FLOOR:        float = 0.25
    CAPITAL_ELASTICITY:        float = 0.15
    HISTORY_GATE:              int   = 60
    RISK_AVERSION:             float = 5.0
    SLACK_PENALTY:             float = 10.0
    DIMENSIONALITY_MULTIPLIER: int   = 3


class InstitutionalRiskEngine:

    def __init__(self, cfg: UltimateConfig):
        self.cfg                = cfg
        self.lw_estimator       = LedoitWolf()
        self.exposure_multiplier: float           = 1.0
        self._override_active:   bool             = False
        self._override_cooldown: int              = 0
        self.last_diag:          SolverDiagnostics = None

    # ── Regime-adaptive exposure multiplier ──────────────────────────────────

    def calculate_exposure_multiplier(self, regime_score: float, realized_cvar: float) -> float:
        target = 1.0 / (1.0 + np.exp(-10.0 * (regime_score - 0.5)))
        change = target - self.exposure_multiplier
        self.exposure_multiplier += float(
            np.clip(change, -self.cfg.DELEVERAGING_LIMIT, self.cfg.DELEVERAGING_LIMIT)
        )
        if self._override_cooldown > 0:
            self._override_cooldown -= 1

        breach = realized_cvar > (self.cfg.MAX_PORTFOLIO_RISK_PCT * 1.5)
        if breach and not self._override_active and self._override_cooldown == 0:
            self.exposure_multiplier = max(
                self.cfg.MIN_EXPOSURE_FLOOR, self.exposure_multiplier * 0.5
            )
            self._override_active   = True
            self._override_cooldown = 4
        elif not breach:
            self._override_active = False

        self.exposure_multiplier = float(
            np.clip(self.exposure_multiplier, self.cfg.MIN_EXPOSURE_FLOOR, 1.0)
        )
        return self.exposure_multiplier

    # ── Portfolio optimiser ───────────────────────────────────────────────────

    def optimize(
        self,
        expected_returns:   np.ndarray,
        historical_returns: "pd.DataFrame",
        adv_shares:         np.ndarray,
        prices:             np.ndarray,
        portfolio_value:    float,
        prev_w:             np.ndarray = None,
    ) -> np.ndarray:

        N = len(expected_returns)
        if N == 0:
            return np.array([])
        if prev_w is None or len(prev_w) != N:
            prev_w = np.zeros(N)

        gamma = float(np.clip(self.exposure_multiplier, self.cfg.MIN_EXPOSURE_FLOOR, 1.0))

        # ── History gate ─────────────────────────────────────────────────────
        valid_cols = [
            c for c in historical_returns.columns
            if historical_returns[c].notna().sum() >= self.cfg.HISTORY_GATE
        ]
        if not valid_cols:
            logger.warning("[Optimizer] No columns pass HISTORY_GATE; floor allocation.")
            self._set_diag("no_valid_cols", gamma, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0)
            w      = np.zeros(N)
            w[:]   = gamma / N
            return w

        valid_idx  = [list(historical_returns.columns).index(c) for c in valid_cols]
        m          = len(valid_idx)
        clean_rets = historical_returns[valid_cols].ffill().dropna()
        T          = len(clean_rets)

        # ── Dimensionality guard ─────────────────────────────────────────────
        if T < self.cfg.DIMENSIONALITY_MULTIPLIER * m:
            logger.warning(
                f"[Optimizer] T={T} < {self.cfg.DIMENSIONALITY_MULTIPLIER}×m={m}. "
                "Dimensionality guard triggered; floor allocation."
            )
            self._set_diag("dimensionality_guard", gamma, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0)
            w_f             = np.zeros(N)
            w_f[valid_idx]  = gamma / m
            return w_f

        # ── Ledoit-Wolf covariance (daily plane) ─────────────────────────────
        self.lw_estimator.fit(clean_rets)
        Sigma     = self.lw_estimator.covariance_
        cond      = np.linalg.cond(Sigma)
        ridge     = max(1e-6, cond * 1e-12)
        Sigma_reg = Sigma + ridge * np.eye(m)

        # ── ADV dollar-volume cap ─────────────────────────────────────────────
        adv_dollar  = (
            np.maximum(adv_shares[valid_idx], 0.0) *
            np.maximum(prices[valid_idx], 1.0)
        )
        adv_limit   = np.clip(
            (adv_dollar * self.cfg.MAX_ADV_PCT) / portfolio_value, 1e-9, 0.40
        )
        sum_adv_lim = float(np.sum(adv_limit))

        # ── Regime-adaptive budget bounds ────────────────────────────────────
        l_gamma = max(self.cfg.MIN_EXPOSURE_FLOOR, gamma * (1.0 - self.cfg.CAPITAL_ELASTICITY))
        u_gamma = min(1.0, gamma * (1.0 + self.cfg.CAPITAL_ELASTICITY))
        if sum_adv_lim < l_gamma:
            l_gamma = sum_adv_lim * 0.999
            u_gamma = min(u_gamma, sum_adv_lim)

        # ── Market-impact / turnover penalty ─────────────────────────────────
        impact = np.clip(
            self.cfg.IMPACT_COEFF * (portfolio_value ** 2) /
            (
                np.maximum(prices[valid_idx], 1.0) *
                np.maximum(adv_shares[valid_idx], 1.0) ** 2
            ),
            0.0, 1e4,
        )

        # ── CVaR scenario matrix ─────────────────────────────────────────────
        T_cvar = min(T, self.cfg.CVAR_LOOKBACK)
        if T_cvar < self.cfg.CVAR_LOOKBACK:
            # Relegated to debug to prevent 50+ lines of log noise during expected warm-ups
            logger.debug(f"[Optimizer] CVaR window {T_cvar} < {self.cfg.CVAR_LOOKBACK}.")
            
        cvar_rets = clean_rets.iloc[-T_cvar:].values
        losses    = -cvar_rets
        alpha     = self.cfg.CVAR_ALPHA

        # ── Variable layout ──────────────────────────────────────────────────
        # Indices: w[0:m], eta[m], z[m+1 : m+1+T_cvar], slack[m+1+T_cvar]
        i_eta  = m
        i_zs   = m + 1
        i_ze   = m + 1 + T_cvar   # exclusive end of z-block (also == i_slk)
        i_slk  = m + 1 + T_cvar   # slack variable slot
        n_vars = m + 1 + T_cvar + 1

        # ── QP objective ─────────────────────────────────────────────────────
        P_ww = 2.0 * (self.cfg.RISK_AVERSION * Sigma_reg + np.diag(impact))
        P    = sp.block_diag(
            [sp.csc_matrix(P_ww), sp.csc_matrix((1 + T_cvar + 1, 1 + T_cvar + 1))],
            format="csc",
        )
        q         = np.zeros(n_vars)
        q[:m]     = -expected_returns[valid_idx] - 2.0 * impact * prev_w[valid_idx]
        q[i_slk]  = self.cfg.SLACK_PENALTY

        # ── Constraints ──────────────────────────────────────────────────────
        rows_A, rows_l, rows_u = [], [], []

        # (a) Budget
        r      = np.zeros(n_vars); r[:m] = 1.0
        rows_A.append(r); rows_l.append(l_gamma); rows_u.append(u_gamma)

        # (b) CVaR auxiliary: L·w - η - z_t ≤ 0
        for i in range(T_cvar):
            r             = np.zeros(n_vars)
            r[:m]         = losses[i]
            r[i_eta]      = -1.0
            r[i_zs + i]   = -1.0
            rows_A.append(r); rows_l.append(-np.inf); rows_u.append(0.0)

        # (c) CVaR soft bound: η + Σz/(T·(1-α)) - slack ≤ CVAR_DAILY_LIMIT
        r              = np.zeros(n_vars)
        r[i_eta]       = 1.0
        r[i_zs:i_ze]   = 1.0 / (T_cvar * (1.0 - alpha))
        r[i_slk]       = -1.0
        rows_A.append(r); rows_l.append(-np.inf); rows_u.append(self.cfg.CVAR_DAILY_LIMIT)

        # (d) Variable bounds encoded as explicit constraint rows
        lb = np.full(n_vars, -np.inf)
        ub = np.full(n_vars,  np.inf)
        lb[:m]        = 0.0
        ub[:m]        = adv_limit
        lb[i_zs:i_ze] = 0.0
        lb[i_slk]     = 0.0

        eye = np.eye(n_vars)
        for i in range(n_vars):
            rows_A.append(eye[i]); rows_l.append(lb[i]); rows_u.append(ub[i])

        A_sp = sp.csc_matrix(np.vstack(rows_A))
        l_np = np.array(rows_l, dtype=float)
        u_np = np.array(rows_u, dtype=float)

        # ── Solve ─────────────────────────────────────────────────────────────
        prob = osqp.OSQP()
        prob.setup(
            P, q, A_sp, l_np, u_np,
            verbose=False, eps_abs=1e-6, eps_rel=1e-6,
            max_iter=100_000, scaling=True, adaptive_rho=True,
        )
        res = prob.solve()

        # ── Solver failure ────────────────────────────────────────────────────
        if res.info.status not in ("solved", "solved_inaccurate"):
            logger.warning(f"[Optimizer] OSQP: {res.info.status}; floor allocation.")
            self._set_diag(
                res.info.status, gamma, 0.0, l_gamma, u_gamma,
                0.0, 0, sum_adv_lim, ridge, cond, T_cvar,
            )
            w_f            = np.zeros(N)
            w_f[valid_idx] = l_gamma / m
            return w_f

        x      = res.x
        w_opt  = x[:m]
        eta_v  = float(x[i_eta])
        z_vals = x[i_zs:i_ze]
        slack  = float(x[i_slk])
        cvar_r = eta_v + float(np.sum(z_vals)) / (T_cvar * (1.0 - alpha))

        # ── Degeneracy guard ──────────────────────────────────────────────────
        exposure = float(np.sum(np.abs(w_opt)))
        if exposure < self.cfg.MIN_EXPOSURE_FLOOR:
            logger.info("[Optimizer] Degenerate solution; defaulting to cash.")
            self._set_diag(
                "cash_default", gamma, exposure, l_gamma, u_gamma,
                cvar_r, 0, sum_adv_lim, ridge, cond, T_cvar, slack,
            )
            return np.zeros(N)

        # ADV binding count read from primal solution to avoid dual-index ambiguity.
        adv_binding_cnt = int(np.sum(w_opt >= adv_limit - 1e-4))

        self._set_diag(
            res.info.status, gamma, float(np.sum(w_opt)),
            l_gamma, u_gamma, cvar_r, adv_binding_cnt,
            sum_adv_lim, ridge, cond, T_cvar, slack,
        )

        w_full = np.zeros(N)
        for i, idx in enumerate(valid_idx):
            w_full[idx] = max(0.0, w_opt[i])
        return w_full

    def _set_diag(
        self, status, gamma, actual_w, l_g, u_g, cvar_v,
        adv_cnt, sum_adv, ridge, cond, t_cvar, slack=0.0,
    ):
        self.last_diag = SolverDiagnostics(
            status            = status,
            gamma_intent      = gamma,
            actual_weight     = actual_w,
            l_gamma           = l_g,
            u_gamma           = u_g,
            cvar_value        = cvar_v,
            slack_value       = slack,
            sum_adv_limit     = sum_adv,
            adv_binding_count = adv_cnt,
            ridge_applied     = ridge,
            cond_number       = cond,
            t_cvar            = t_cvar,
        )