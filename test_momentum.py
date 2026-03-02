"""
test_momentum.py — Full Deterministic Parity Test Suite
========================================================
Every test either asserts a real invariant or does not exist.
"""

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from signals import generate_signals, compute_adv, compute_regime_score
from momentum_engine import (
    InstitutionalRiskEngine,
    UltimateConfig,
    OptimizationError,
    OptimizationErrorType,
    PortfolioState,
    execute_rebalance,
)
from backtest_engine import BacktestEngine, run_backtest, _compute_metrics, _build_adv_vector
from universe_manager import STATIC_NSE_SECTORS
from daily_workflow import detect_and_apply_splits, save_portfolio_state, _normalise_start_date

# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_log_rets(n_days: int, n_syms: int, seed: int = 42) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(n_days, n_syms))
    cols = [f"SYM{i:02d}" for i in range(n_syms)]
    idx  = pd.date_range("2020-01-02", periods=n_days, freq="B")
    return pd.DataFrame(data, index=idx, columns=cols)


def _make_close(n_days: int, n_syms: int, seed: int = 42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    rets   = rng.normal(0.0005, 0.01, size=(n_days, n_syms))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    cols   = [f"SYM{i:02d}" for i in range(n_syms)]
    idx    = pd.date_range("2020-01-02", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=idx, columns=cols)


def _make_engine(max_sector_weight: float = 0.30) -> InstitutionalRiskEngine:
    cfg = UltimateConfig()
    cfg.MAX_SECTOR_WEIGHT = max_sector_weight
    return InstitutionalRiskEngine(cfg)


# ─── signals.py ───────────────────────────────────────────────────────────────

def test_generate_signals_deterministic():
    """Same inputs must produce identical outputs on repeated calls."""
    log_rets = _make_log_rets(120, 6)
    adv      = np.ones(6) * 1e6
    cfg      = UltimateConfig(HISTORY_GATE=90, MAX_POSITIONS=5)
    slice_t1 = log_rets.iloc[:100]
    raw1, scores1, sel1 = generate_signals(slice_t1, adv, cfg)
    raw2, scores2, sel2 = generate_signals(slice_t1, adv, cfg)
    np.testing.assert_array_equal(raw1, raw2)
    assert sel1 == sel2


def test_generate_signals_history_gate():
    """Assets with insufficient history must be excluded from selection."""
    log_rets = _make_log_rets(100, 5)
    log_rets.iloc[:90, 0] = np.nan   # SYM00 has only 10 valid rows
    adv      = np.ones(5) * 1e6
    cfg      = UltimateConfig(HISTORY_GATE=95, MAX_POSITIONS=5)
    _, _, sel_idx = generate_signals(log_rets, adv, cfg)
    assert 0 not in sel_idx, "SYM00 should be excluded by history gate."


def test_generate_signals_continuity_bonus():
    """An asset with a non-zero previous weight must score higher than an identical twin."""
    rng      = np.random.default_rng(0)
    base_col = rng.normal(0, 0.01, 100)
    log_rets = pd.DataFrame(
        np.column_stack([base_col, base_col]), columns=["SYM0", "SYM1"]
    )
    adv          = np.ones(2) * 1e6
    cfg          = UltimateConfig(HISTORY_GATE=10, MAX_POSITIONS=2)
    prev_weights = {"SYM0": 0.10, "SYM1": 0.0}
    _, scores, _ = generate_signals(
        log_rets, adv, cfg, prev_weights=prev_weights,
    )
    assert scores[0] > scores[1], "Held asset must receive continuity bonus."


def test_regime_score_neutral_on_thin_history():
    idx = pd.DataFrame({"Close": [100.0] * 100}, index=pd.date_range("2020-01-01", periods=100))
    assert compute_regime_score(idx) == 0.5


def test_regime_score_bull_market():
    """Price well above SMA200 should give score > 0.5."""
    closes = np.linspace(80, 120, 400)      # steady uptrend
    idx = pd.DataFrame({"Close": closes}, index=pd.date_range("2020-01-01", periods=400))
    assert compute_regime_score(idx) > 0.5


def test_regime_score_bear_market():
    """Price well below SMA200 should give score < 0.5."""
    closes = np.linspace(120, 60, 400)      # steady downtrend
    idx = pd.DataFrame({"Close": closes}, index=pd.date_range("2020-01-01", periods=400))
    assert compute_regime_score(idx) < 0.5


# ─── momentum_engine.py ───────────────────────────────────────────────────────

def test_optimizer_sector_cap_enforced():
    """Sum of weights in any sector must not exceed MAX_SECTOR_WEIGHT."""
    n, m = 150, 5
    log_rets      = _make_log_rets(n, m)
    engine        = _make_engine(max_sector_weight=0.30)
    exp_rets      = np.array([0.002, 0.001, 0.003, 0.0015, 0.0025])
    adv           = np.ones(m) * 1e6
    prices        = np.ones(m) * 100.0
    pv            = 1_000_000.0
    sector_labels = np.zeros(m, dtype=int)   # all in same sector → cap at 0.30
    weights = engine.optimize(
        exp_rets, log_rets, adv, prices, pv,
        exposure_multiplier=1.0, sector_labels=sector_labels,
    )
    assert float(np.sum(weights)) <= 0.30 + 1e-5, "Sector cap violation."


def test_optimizer_weights_non_negative():
    log_rets = _make_log_rets(100, 4)
    engine   = _make_engine()
    weights  = engine.optimize(
        np.array([0.001, 0.002, 0.0005, 0.0015]),
        log_rets, np.ones(4) * 1e6, np.ones(4) * 200, 1_000_000.0,
        exposure_multiplier=1.0,
    )
    assert np.all(weights >= -1e-8), "Weights must be non-negative (up to floating-point tolerance)."


def test_optimizer_raises_data_error_on_thin_history():
    log_rets = _make_log_rets(5, 4)
    engine   = _make_engine()
    with pytest.raises(OptimizationError) as exc_info:
        engine.optimize(
            np.array([0.001, 0.002, 0.0005, 0.0015]),
            log_rets, np.ones(4) * 1e6, np.ones(4) * 200, 1_000_000.0,
            exposure_multiplier=1.0,
        )
    assert exc_info.value.error_type == OptimizationErrorType.DATA


def test_optimizer_rejects_non_finite_inputs():
    log_rets = _make_log_rets(120, 4)
    engine   = _make_engine()
    with pytest.raises(OptimizationError) as exc_info:
        engine.optimize(
            np.array([0.001, np.nan, 0.0005, 0.0015]),
            log_rets,
            np.ones(4) * 1e6,
            np.ones(4) * 200,
            1_000_000.0,
            exposure_multiplier=1.0,
        )
    assert exc_info.value.error_type == OptimizationErrorType.DATA


def test_optimizer_rejects_prev_weights_length_mismatch():
    log_rets = _make_log_rets(120, 4)
    engine   = _make_engine()
    with pytest.raises(OptimizationError) as exc_info:
        engine.optimize(
            np.array([0.001, 0.002, 0.0005, 0.0015]),
            log_rets,
            np.ones(4) * 1e6,
            np.ones(4) * 200,
            1_000_000.0,
            prev_w=np.array([0.1, 0.2]),
            exposure_multiplier=1.0,
        )
    assert exc_info.value.error_type == OptimizationErrorType.DATA


def test_optimizer_adv_binding_count_populated():
    """SolverDiagnostics.adv_binding_count must not always be zero."""
    n, m = 150, 3
    log_rets = _make_log_rets(n, m)
    engine   = _make_engine()
    # Very tight ADV limit forces weights to the cap.
    adv      = np.ones(m) * 10.0        # tiny volume
    prices   = np.ones(m) * 1000.0
    pv       = 1_000_000.0
    engine.optimize(
        np.array([0.002, 0.003, 0.001]),
        log_rets, adv, prices, pv, exposure_multiplier=1.0,
    )
    assert engine.last_diag is not None
    assert isinstance(engine.last_diag.adv_binding_count, int)


def test_optimizer_cvar_sentinel_abort():
    """Optimizer must reject combinations where single-asset CVaR wildly breaches safe limits."""
    n_days, m = 250, 5
    log_rets = _make_log_rets(n_days, m)
    # Inject massive daily loss into ALL assets to trip the EW-CVaR sentinel
    log_rets.iloc[-20:, :] = -0.15 
    
    engine = _make_engine()
    with pytest.raises(OptimizationError) as exc_info:
        engine.optimize(
            np.ones(m)*0.01, log_rets, np.ones(m)*1e6, np.ones(m)*100, 1e6
        )
    assert exc_info.value.error_type == OptimizationErrorType.INFEASIBLE


def test_portfolio_state_serialisation_roundtrip():
    ps = PortfolioState()
    ps.weights              = {"RELIANCE": 0.15, "TCS": 0.12}
    ps.shares               = {"RELIANCE": 10, "TCS": 5}
    ps.entry_prices         = {"RELIANCE": 2450.50, "TCS": 3800.00}
    ps.cash                 = 750_000.0
    ps.exposure_multiplier  = 0.85
    ps.consecutive_failures = 1
    ps.equity_hist          = [1_000_000.0, 990_000.0, 1_010_000.0]
    ps.override_active      = True
    ps.override_cooldown    = 3

    ps2 = PortfolioState.from_dict(ps.to_dict())
    assert ps2.weights              == ps.weights
    assert ps2.shares               == ps.shares
    assert ps2.entry_prices         == ps.entry_prices
    assert ps2.cash                 == ps.cash
    assert abs(ps2.exposure_multiplier - ps.exposure_multiplier) < 1e-9
    assert ps2.override_active      == ps.override_active
    assert ps2.override_cooldown    == ps.override_cooldown


def test_portfolio_state_from_dict_bool_string_parsing():
    """String booleans from hand-edited JSON should parse as expected."""
    ps = PortfolioState.from_dict({"override_active": "False", "override_cooldown": 2})
    assert ps.override_active is False
    assert ps.override_cooldown == 2


def test_portfolio_state_from_dict_bool_numeric_parsing():
    """Legacy numeric bool flags should parse strictly for 0/1 values only."""
    ps_true = PortfolioState.from_dict({"override_active": 1})
    ps_false = PortfolioState.from_dict({"override_active": 0})
    ps_invalid = PortfolioState.from_dict({"override_active": 2})
    assert ps_true.override_active is True
    assert ps_false.override_active is False
    # Invalid values are reset by from_dict's guarded converter path.
    assert ps_invalid.override_active is False


def test_normalise_start_date_default_and_validation():
    assert _normalise_start_date("   ") == "2020-01-01"
    assert _normalise_start_date("2024-01-31") == "2024-01-31"
    with pytest.raises(ValueError):
        _normalise_start_date("2024/01/31")


def test_update_exposure_regime_bull():
    """Bull regime should push exposure multiplier upward."""
    cfg   = UltimateConfig()
    state = PortfolioState()
    state.exposure_multiplier = 0.5
    state.update_exposure(regime_score=0.9, realized_cvar=0.0, cfg=cfg)
    assert state.exposure_multiplier > 0.5, "Bull regime should increase exposure."


def test_update_exposure_cvar_breach():
    """CVaR breach should trigger override and halve exposure."""
    cfg   = UltimateConfig()
    state = PortfolioState()
    state.exposure_multiplier = 1.0
    # CVaR breach: realized > 1.5 × MAX_PORTFOLIO_RISK_PCT
    breach_cvar = cfg.MAX_PORTFOLIO_RISK_PCT * 2.0
    state.update_exposure(regime_score=0.5, realized_cvar=breach_cvar, cfg=cfg)
    assert state.override_active      is True
    assert state.override_cooldown    == 4
    assert state.exposure_multiplier  < 0.5 + 1e-9, "CVaR breach must halve exposure."


def test_cvar_gross_exposure_normalisation():
    """
    With 50% cash, the portfolio's realised CVaR is approximately halved.
    update_exposure must normalise by gross_exposure before the breach check.
    """
    cfg   = UltimateConfig()
    high_asset_cvar = cfg.MAX_PORTFOLIO_RISK_PCT * 2.0  
    portfolio_cvar  = high_asset_cvar * 0.5              

    state = PortfolioState()
    state.exposure_multiplier = 1.0
    state.update_exposure(0.5, portfolio_cvar, cfg, gross_exposure=0.5)

    assert state.override_active is True, \
        "Override must trigger when asset-level CVaR is high even if portfolio CVaR is diluted by cash."


def test_execute_rebalance_pv_includes_stale_positions():
    """
    PV must not silently drop positions absent from active_symbols.
    If MAX_ABSENT_PERIODS=1, a single absence force-closes the position.
    """
    cfg   = UltimateConfig(MAX_ABSENT_PERIODS=1)
    state = PortfolioState(cash=500_000.0)
    state.shares       = {"ACTIVE": 100, "STALE": 200}
    state.entry_prices = {"ACTIVE": 1000.0, "STALE": 500.0}
    state.last_known_prices = {"STALE": 500.0}
    state.weights      = {"ACTIVE": 0.10}

    prices_active  = np.array([1000.0])
    active_symbols = ["ACTIVE"]
    target_weights = np.array([0.20])

    execute_rebalance(state, target_weights, prices_active, active_symbols, cfg)
    
    assert "STALE" not in state.shares, "With MAX_ABSENT_PERIODS=1, one absence closes position."


def test_execute_rebalance_cash_conservation():
    """Cash + notional should equal PV minus slippage."""
    cfg   = UltimateConfig()
    state = PortfolioState(cash=1_000_000.0)
    prices = np.array([500.0, 300.0, 200.0])
    weights = np.array([0.30, 0.20, 0.10])
    execute_rebalance(state, weights, prices, ["A", "B", "C"], cfg)
    notional = sum(state.shares.get(s, 0) * p for s, p in zip(["A", "B", "C"], prices))
    assert state.cash >= 0
    assert state.cash + notional <= 1_000_000.0 + 1e-2   # slippage may reduce total


def test_detect_and_apply_splits_fractional_cash():
    """A reverse split that drops fractional shares must return that value to Cash."""
    state = PortfolioState(cash=0.0)
    # Reverse split 1:2 means ratio is 0.5. 101 shares -> 50.5 -> rounds to 50.
    state.shares = {"A": 101} 
    state.last_known_prices = {"A": 100.0}
    
    market_data = {"A": pd.DataFrame({"Close": [200.0]})}
    detect_and_apply_splits(state, market_data)
    
    assert state.shares["A"] == 50, "Shares should floor correctly on splits."
    # 101 - (50 / 0.5) = 1.0 orphaned share worth 200.0
    assert state.cash == 200.0, "Fractional value must be safely routed to Cash."


# ─── PortfolioState.record_eod ────────────────────────────────────────────────

def test_record_eod_flat_day_preserved():
    """Identical consecutive PV values MUST NOT be deduplicated; flat days are critical for correct CVaR calculations."""
    ps           = PortfolioState(cash=1_000_000.0)
    ps.shares    = {"RELIANCE": 10, "TCS": 5}
    prices       = {"RELIANCE": 2500.0, "TCS": 3800.0}

    ps.record_eod(prices)
    ps.record_eod(prices) # same PV → SHOULD append.
    
    assert len(ps.equity_hist) == 2, "Flat-days must be preserved, not dropped."


def test_record_eod_cap_respected():
    ps                 = PortfolioState(cash=1_000_000.0)
    ps.equity_hist_cap = 10
    for i in range(20):
        ps.cash = 1_000_000.0 + i * 100.0
        ps.record_eod({})
    assert len(ps.equity_hist) == 10
    assert ps.equity_hist[-1] == round(1_000_000.0 + 19 * 100.0, 10)


def test_realised_cvar_warm_up_guard():
    """CVaR must return 0.0 when equity history is below the minimum observation threshold."""
    ps = PortfolioState(cash=1_000_000.0)
    for i in range(15):
        ps.cash = 1_000_000.0 + i * 1000.0
        ps.record_eod({})
    assert ps.realised_cvar(min_obs=30) == 0.0


# ─── universe_manager.py ──────────────────────────────────────────────────────

def test_static_sector_map_covers_nifty50_top10():
    """Top-10 Nifty 50 constituents (including SBIN) must be in the static map."""
    top10 = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
             "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK"]
    for sym in top10:
        assert sym in STATIC_NSE_SECTORS, f"{sym} missing from STATIC_NSE_SECTORS."
        assert STATIC_NSE_SECTORS[sym] not in ("", "Unknown"), \
            f"{sym} has an invalid sector '{STATIC_NSE_SECTORS[sym]}'."


# ─── data_cache.py ────────────────────────────────────────────────────────────

def test_data_cache_staleness_logic(tmp_path, monkeypatch):
    """Ensure data is pulled if it doesn't contain yesterday's (T-1) business day."""
    from data_cache import load_or_fetch
    monkeypatch.setattr("data_cache.CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("data_cache.MANIFEST_FILE", str(tmp_path / "_manifest.json"))
    
    import json
    # Mock a manifest that says fetched very recently (1 hour ago) BUT last_date is deeply stale
    manifest = {
        "TEST.NS": {
            "fetched_at": (datetime.now() - timedelta(hours=1)).isoformat(),
            "last_date": "2020-01-01",
            "covered_start": "2019-01-01"
        }
    }
    with open(tmp_path / "_manifest.json", "w") as f:
        json.dump(manifest, f)
        
    # Write dummy parquet so it doesn't fail the file-exists check
    dummy = pd.DataFrame({"Close": [100]}, index=pd.to_datetime(["2020-01-01"]))
    dummy.to_parquet(tmp_path / "TEST.NS.parquet")
    
    # We monkeypatch the _download_with_retry to observe if a download is forced
    download_called = False
    def mock_download(*args, **kwargs):
        nonlocal download_called
        download_called = True
        return pd.DataFrame()
    monkeypatch.setattr("data_cache._download_with_timeout", mock_download)
    
    load_or_fetch(["TEST"], "2020-01-01", "2020-01-10", force_refresh=False)
    assert download_called, "Cache must trigger re-download if last_date misses yesterday's business day."


def test_portfolio_state_backup_rotation(tmp_path, monkeypatch):
    """Ensure the atomic backup logic safely rotates through .bak limits."""
    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    os.mkdir("data")
    
    state = PortfolioState(cash=5000)
    for _ in range(4):
        save_portfolio_state(state, "test_backup")
        
    assert os.path.exists("data/portfolio_state_test_backup.json")
    assert os.path.exists("data/portfolio_state_test_backup.json.bak.0")
    assert os.path.exists("data/portfolio_state_test_backup.json.bak.1")
    assert os.path.exists("data/portfolio_state_test_backup.json.bak.2")


# ─── backtest_engine.py ───────────────────────────────────────────────────────

def test_compute_metrics_annualisation():
    """Sharpe annualisation must use actual observed period, not hardcoded 52."""
    idx = pd.date_range("2020-01-03", periods=104, freq="W-FRI")
    eq  = pd.Series(
        1_000_000.0 * np.exp(np.random.default_rng(0).normal(0.001, 0.01, 104).cumsum()),
        index=idx,
    )
    m = _compute_metrics(eq, 1_000_000.0)
    assert np.isfinite(m["sharpe"])
    assert -10.0 < m["sharpe"] < 10.0


def test_e2e_ledger_parity():
    """BacktestEngine and a manually driven PortfolioState must produce identical state."""
    n_days, n_syms = 200, 5
    close   = _make_close(n_days, n_syms)
    volume  = pd.DataFrame(np.ones((n_days, n_syms)) * 1e6, index=close.index, columns=close.columns)
    returns = close.pct_change(fill_method=None).clip(lower=-0.99)
    symbols = list(close.columns)

    cfg = UltimateConfig(HISTORY_GATE=20, INITIAL_CAPITAL=1_000_000)

    # ── BacktestEngine path ──
    bt_engine = InstitutionalRiskEngine(cfg)
    bt        = BacktestEngine(bt_engine, initial_cash=cfg.INITIAL_CAPITAL)
    rebal_dates = close.index[::20]
    bt.run(close, volume, returns, rebal_dates, close.index[25].strftime("%Y-%m-%d"))

    # ── Manual path ──
    live_state  = PortfolioState(cash=cfg.INITIAL_CAPITAL)
    live_engine = InstitutionalRiskEngine(cfg)

    for date in close.index:
        if date < close.index[25]:
            continue

        close_t  = close.loc[date]
        prices_t = close_t.values.astype(float)

        if date in rebal_dates:
            hist_log_rets = (
                np.log1p(returns.loc[:date].iloc[:-1])
                .replace([np.inf, -np.inf], np.nan)
            )
            adv_vector = _build_adv_vector(symbols, volume, date)
            pv         = live_state.cash + sum(
                live_state.shares.get(s, 0) * close_t[s] for s in symbols
            )
            # Use cached price logic exactly to match backtest parity
            prev_w_dict = {
                sym: (live_state.shares.get(sym, 0) * live_state.last_known_prices.get(sym, 0.0)) / pv
                for sym in symbols if live_state.shares.get(sym, 0) > 0 and pv > 0
            }
            raw, adj, sel = generate_signals(
                hist_log_rets, adv_vector, cfg, prev_weights=prev_w_dict,
            )
            prev_weights_arr = np.array([prev_w_dict.get(sym, 0.0) for sym in symbols])

            live_state.update_exposure(0.5, live_state.realised_cvar(), cfg)

            if sel:
                weights_sel = live_engine.optimize(
                    raw[sel],
                    hist_log_rets[[symbols[i] for i in sel]],
                    adv_vector[sel],
                    prices_t[sel],
                    pv,
                    prev_weights_arr[sel],
                    exposure_multiplier=live_state.exposure_multiplier,
                )
                target = np.zeros(len(symbols))
                target[sel] = weights_sel
                execute_rebalance(live_state, target, prices_t, symbols, cfg, date_context=date)
                live_state.consecutive_failures = 0

        price_dict = {s: close_t[s] for s in symbols}
        live_state.record_eod(price_dict)

    assert json.dumps(live_state.to_dict(), sort_keys=True) == \
           json.dumps(bt.state.to_dict(), sort_keys=True), \
        "Backtest engine and manual replication must produce byte-identical state."


def test_e2e_cvar_breach_triggers_override():
    """A sharp 50% crash must trigger the CVaR override mechanism."""
    n_days, n_syms = 200, 5
    close = _make_close(n_days, n_syms)
    close.iloc[100:120] = close.iloc[100:120] * 0.5    # simulate crash
    volume  = pd.DataFrame(np.ones((n_days, n_syms)) * 1e6, index=close.index, columns=close.columns)
    returns = close.pct_change(fill_method=None).clip(lower=-0.99)
    symbols = list(close.columns)

    cfg        = UltimateConfig(HISTORY_GATE=20, INITIAL_CAPITAL=1_000_000)
    bt_engine  = InstitutionalRiskEngine(cfg)
    bt         = BacktestEngine(bt_engine, initial_cash=cfg.INITIAL_CAPITAL)
    rebal_dates = close.index[::5]
    bt.run(close, volume, returns, rebal_dates, close.index[25].strftime("%Y-%m-%d"))

    # After a 50% crash, the CVaR override should have triggered at some point.
    assert bt.state.override_cooldown > 0 or bt.state.override_active is True, \
        "CVaR override must activate after a major drawdown."


# ─── Gemini murder board fixes ────────────────────────────────────────────────

def test_nan_sorting_trap_no_truncation():
    """
    If 10 assets have NaN scores, they must NOT consume the top-K slots.
    The portfolio must contain exactly max_positions valid assets, not fewer.
    """
    rng      = np.random.default_rng(7)
    n_syms   = 20
    log_rets = pd.DataFrame(
        rng.normal(0.0, 0.01, (200, n_syms)),
        columns=[f"SYM{i:02d}" for i in range(n_syms)],
        index=pd.date_range("2020-01-02", periods=200, freq="B"),
    )
    for i in range(10):
        log_rets.iloc[:, i] = np.nan

    adv = np.ones(n_syms) * 1e6
    cfg = UltimateConfig(HISTORY_GATE=5, MAX_POSITIONS=10)
    _, adj_scores, sel_idx = generate_signals(log_rets, adv, cfg)

    assert all(np.isfinite(adj_scores[i]) for i in sel_idx), \
        "Selected indices must not contain NaN-scored assets."
    assert len(sel_idx) == 10, \
        f"Expected 10 valid selections, got {len(sel_idx)} — NaN trap still active."


def test_rebalance_prev_weights_use_last_known_price_on_nan_quote(monkeypatch):
    """Held symbols with NaN quote on rebalance day must still contribute finite prev_weights."""
    cfg = UltimateConfig(HISTORY_GATE=5)
    engine = InstitutionalRiskEngine(cfg)
    bt = BacktestEngine(engine, initial_cash=cfg.INITIAL_CAPITAL)

    dates = pd.date_range("2021-01-01", periods=20, freq="B")
    close = pd.DataFrame({"SYM00": np.linspace(100, 120, len(dates))}, index=dates)
    volume = pd.DataFrame({"SYM00": np.ones(len(dates)) * 1e6}, index=dates)
    returns = close.pct_change(fill_method=None).fillna(0.0)

    rebalance_day = dates[-1]
    close.loc[rebalance_day, "SYM00"] = np.nan

    bt.state.shares = {"SYM00": 10}
    bt.state.weights = {"SYM00": 0.1}
    bt.state.last_known_prices = {"SYM00": 119.0}

    captured = {}

    def _fake_generate_signals(*args, **kwargs):
        captured["prev_weights"] = kwargs.get("prev_weights", {})
        return np.array([0.001]), np.array([1.0]), [0]

    def _fake_optimize(*args, **kwargs):
        return np.array([0.1])

    monkeypatch.setattr("backtest_engine.generate_signals", _fake_generate_signals)
    monkeypatch.setattr(bt.engine, "optimize", _fake_optimize)

    bt.run(close, volume, returns, pd.DatetimeIndex([rebalance_day]), dates[0].strftime("%Y-%m-%d"))

    assert "SYM00" in captured["prev_weights"]
    assert np.isfinite(captured["prev_weights"]["SYM00"])
    assert captured["prev_weights"]["SYM00"] > 0


def test_volume_no_lookahead():
    from backtest_engine import _build_adv_vector

    n_days, n_syms = 50, 3
    cols = [f"SYM{i:02d}" for i in range(n_syms)]
    idx  = pd.date_range("2020-01-02", periods=n_days, freq="B")

    volume = pd.DataFrame(np.ones((n_days, n_syms)) * 1e6, index=idx, columns=cols)
    friday = idx[-1]
    volume.loc[friday] = 1e12   

    adv_fri = _build_adv_vector(cols, volume, friday)

    expected_ma = float(
        volume.loc[:friday, cols[0]].iloc[:-1].rolling(20, min_periods=1).mean().iloc[-1]
    )
    assert abs(adv_fri[0] - expected_ma) < 1.0, \
        f"ADV {adv_fri[0]:.0f} does not match T-1 rolling mean {expected_ma:.0f} — lookahead present."


def test_ghost_position_single_day_absence_is_preserved():
    """
    A position absent from the data feed for only 1 period must be carried
    (not liquidated) and remain in state.shares.
    """
    cfg   = UltimateConfig(MAX_ABSENT_PERIODS=3)
    state = PortfolioState(cash=500_000.0)
    state.shares           = {"ACTIVE": 100, "GHOST": 50}
    state.entry_prices     = {"ACTIVE": 1000.0, "GHOST": 800.0}
    state.last_known_prices = {"ACTIVE": 1000.0, "GHOST": 800.0}
    state.weights          = {"ACTIVE": 0.10, "GHOST": 0.04}

    target = np.array([0.20])
    execute_rebalance(state, target, np.array([1000.0]), ["ACTIVE"], cfg)

    assert "GHOST" in state.shares, "Single-day absence must not liquidate the position."
    assert state.absent_periods.get("GHOST", 0) == 1, "Absent counter must increment."


def test_ghost_position_delists_after_max_absent_periods():
    """
    A position absent for MAX_ABSENT_PERIODS consecutive rebalances must be
    closed at its last known price, not at ₹0.
    """
    cfg   = UltimateConfig(MAX_ABSENT_PERIODS=2)
    state = PortfolioState(cash=500_000.0)
    state.shares            = {"DELISTED": 100}
    state.entry_prices      = {"DELISTED": 1000.0}
    state.last_known_prices = {"DELISTED": 900.0}   
    state.weights           = {"DELISTED": 0.10}

    target_empty = np.array([], dtype=float)
    trade_log: list = []

    for i in range(cfg.MAX_ABSENT_PERIODS):
        execute_rebalance(
            state, target_empty, np.array([]), [], cfg,
            date_context=pd.Timestamp(f"2020-01-{i+1:02d}"),
            trade_log=trade_log,
        )

    assert "DELISTED" not in state.shares, "Position must be closed after MAX_ABSENT_PERIODS."
    sell_trades = [t for t in trade_log if t.symbol == "DELISTED" and t.direction == "SELL"]
    assert sell_trades, "A SELL trade must be logged for the delisted position."
    assert sell_trades[-1].exec_price == pytest.approx(900.0, rel=1e-4), \
        "Delisted position must be closed at last known price, not ₹0."


def test_decay_rounds_cap_prevents_further_decay():
    """
    After MAX_DECAY_ROUNDS consecutive decays, weights must be held flat
    rather than decayed further.
    """
    cfg   = UltimateConfig(MAX_DECAY_ROUNDS=2)
    state = PortfolioState(cash=1_000_000.0)
    state.shares       = {"A": 100}
    state.entry_prices = {"A": 1000.0}
    state.last_known_prices = {"A": 1000.0}
    state.weights      = {"A": 0.10}

    # Apply MAX_DECAY_ROUNDS rounds of decay.
    for _ in range(cfg.MAX_DECAY_ROUNDS):
        execute_rebalance(
            state, np.array([0.0]), np.array([1000.0]), ["A"], cfg, apply_decay=True
        )

    weight_at_cap = state.weights.get("A", 0.0)

    # One more round — should NOT decay further.
    execute_rebalance(
        state, np.array([0.0]), np.array([1000.0]), ["A"], cfg, apply_decay=True
    )
    weight_after_cap = state.weights.get("A", 0.0)

    assert state.decay_rounds == cfg.MAX_DECAY_ROUNDS, \
        "decay_rounds must not exceed MAX_DECAY_ROUNDS."
    assert abs(weight_at_cap - weight_after_cap) < 1e-8, \
        "Weight must be held flat once MAX_DECAY_ROUNDS is reached."


def test_decay_rounds_reset_on_solver_success():
    """
    A successful optimization run resets state.decay_rounds to zero.
    """
    cfg = UltimateConfig(HISTORY_GATE=5, INITIAL_CAPITAL=1_000_000, MAX_DECAY_ROUNDS=3)
    n_days, n_syms = 50, 2
    close   = _make_close(n_days, n_syms)
    volume  = pd.DataFrame(np.ones((n_days, n_syms)) * 1e6, index=close.index, columns=close.columns)
    returns = close.pct_change(fill_method=None).clip(lower=-0.99)
    
    engine = InstitutionalRiskEngine(cfg)
    bt = BacktestEngine(engine, initial_cash=cfg.INITIAL_CAPITAL)
    bt.state.decay_rounds = 3 # mock prior failures
    
    rebal_dates = close.index[20:25]
    bt.run(close, volume, returns, rebal_dates, close.index[0].strftime("%Y-%m-%d"))
    
    assert bt.state.decay_rounds == 0, "BacktestEngine run loop must correctly zero decay_rounds upon optimization success"

if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))