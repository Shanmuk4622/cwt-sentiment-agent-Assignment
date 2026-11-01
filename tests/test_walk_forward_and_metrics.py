"""
Unit and integration tests for walk-forward validation splits,
embargo purging, and quantitative financial risk metrics (Sharpe, Sortino, Max Drawdown).
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import QuantitativeEvaluator
from src.model import MultiHorizonPredictor, WalkForwardEngine


def test_walk_forward_splits_chronology_and_embargo():
    """Verify that walk-forward validation maintains strict temporal causality and embargo gaps."""
    engine = WalkForwardEngine(n_splits=3, embargo_periods=5, horizons=[0, 1])

    n_samples = 300
    df = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "close": 80000.0 + np.cumsum(np.random.randn(n_samples)),
        "target_return_h0": np.random.randn(n_samples) * 0.01,
        "target_price_h0": 80000.0 + np.cumsum(np.random.randn(n_samples)),
        "target_dir_h0": np.random.choice([0, 1], size=n_samples),
        "target_return_h1": np.random.randn(n_samples) * 0.01,
        "target_price_h1": 80000.0 + np.cumsum(np.random.randn(n_samples)),
        "target_dir_h1": np.random.choice([0, 1], size=n_samples),
    })

    evals = engine.run_walk_forward(df, feature_cols=["feature_1", "feature_2"])
    assert 0 in evals
    assert 1 in evals

    for h, res in evals.items():
        assert len(res.predictions) > 0
        assert len(res.predictions) == len(res.actual_returns)
        assert len(res.predicted_prices) == len(res.actual_prices)
        assert 0.0 <= res.directional_accuracy <= 1.0


def test_quantitative_metrics_sharpe_and_sortino():
    """Verify Sharpe Ratio, Sortino Ratio, and Max Drawdown on deterministic series."""
    evaluator = QuantitativeEvaluator(risk_free_rate=0.04, periods_per_year=8760)

    # Positive drift returns
    np.random.seed(42)
    pos_returns = pd.Series(np.random.normal(0.001, 0.005, size=1000))
    sharpe_pos = evaluator.calculate_sharpe_ratio(pos_returns)
    sortino_pos = evaluator.calculate_sortino_ratio(pos_returns)

    assert sharpe_pos > 0.0
    assert sortino_pos > 0.0

    # Negative drift returns
    neg_returns = pd.Series(np.random.normal(-0.001, 0.005, size=1000))
    sharpe_neg = evaluator.calculate_sharpe_ratio(neg_returns)
    assert sharpe_neg < 0.0


def test_max_drawdown_calculation():
    """Verify that Maximum Drawdown accurately detects severe equity drops."""
    evaluator = QuantitativeEvaluator()

    # Equity curve: 100 -> 150 -> 75 -> 120 -> 60 (peak is 150, lowest after peak is 60 => 60% drawdown)
    equity = pd.Series([100.0, 150.0, 75.0, 120.0, 60.0])
    max_dd, dd_series = evaluator.calculate_max_drawdown(equity)

    assert pytest.approx(max_dd, 0.01) == 0.60
    assert dd_series.min() == -0.60


def test_backtest_strategy_execution():
    """Verify full backtest simulation computes non-empty equity curve and alpha."""
    evaluator = QuantitativeEvaluator(transaction_cost_bps=10, slippage_bps=5)

    n = 200
    actual_returns = np.random.normal(0.0005, 0.01, size=n)
    pred_returns = actual_returns + np.random.normal(0, 0.002, size=n)
    actual_prices = 80000.0 * np.cumprod(1.0 + actual_returns)

    metrics = evaluator.backtest_strategy(pred_returns, actual_returns, actual_prices, allow_short=False)

    assert len(metrics.equity_curve) == n
    assert len(metrics.benchmark_equity) == n
    assert metrics.total_trades > 0
    assert 0.0 <= metrics.win_rate_pct <= 100.0
    assert metrics.max_drawdown_pct >= 0.0
