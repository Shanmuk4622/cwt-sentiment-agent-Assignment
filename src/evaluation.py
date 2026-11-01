"""
Quantitative Evaluation & Financial Backtesting Engine:
Computes Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio,
Win Rate, and simulates realistic execution with transaction costs and slippage.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Comprehensive institutional backtesting metrics."""
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    benchmark_total_return_pct: float
    benchmark_sharpe: float
    benchmark_max_drawdown_pct: float
    alpha_pct: float
    equity_curve: pd.Series
    benchmark_equity: pd.Series
    drawdown_series: pd.Series


class QuantitativeEvaluator:
    """Evaluates risk-adjusted returns, downside risk, and trade execution."""

    def __init__(
        self,
        risk_free_rate: float = config.RISK_FREE_RATE,
        transaction_cost_bps: float = config.TRANSACTION_COST_BPS,
        slippage_bps: float = config.SLIPPAGE_BPS,
        periods_per_year: int = 8760,  # 365 * 24 for 1h bars
    ):
        self.rf = risk_free_rate
        self.cost_rate = (transaction_cost_bps + slippage_bps) / 10000.0  # in decimal
        self.periods_per_year = periods_per_year

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Annualized Sharpe Ratio: S = sqrt(N) * (mean(r) - rf_bar) / std(r)"""
        clean_ret = returns.dropna()
        if len(clean_ret) < 2 or clean_ret.std() == 0:
            return 0.0
        rf_bar = self.rf / self.periods_per_year
        excess_ret = clean_ret - rf_bar
        return float(np.sqrt(self.periods_per_year) * excess_ret.mean() / (clean_ret.std() + 1e-8))

    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Annualized Sortino Ratio using downside semi-deviation below risk-free rate."""
        clean_ret = returns.dropna()
        if len(clean_ret) < 2:
            return 0.0
        rf_bar = self.rf / self.periods_per_year
        excess_ret = clean_ret - rf_bar
        downside = np.minimum(0.0, excess_ret)
        downside_std = np.sqrt(np.mean(downside**2))
        if downside_std == 0:
            return 0.0
        return float(np.sqrt(self.periods_per_year) * excess_ret.mean() / (downside_std + 1e-8))

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, pd.Series]:
        """Compute Maximum Drawdown (MDD) and the underwater drawdown series."""
        peaks = equity_curve.cummax()
        drawdowns = (equity_curve - peaks) / peaks
        max_dd = float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0.0
        return max_dd, drawdowns

    def backtest_strategy(
        self,
        pred_returns: np.ndarray,
        actual_returns: np.ndarray,
        actual_prices: np.ndarray,
        threshold_std_factor: float = 0.25,
        allow_short: bool = False,
    ) -> BacktestMetrics:
        """Simulate realistic trade execution based on model signals with slippage and costs.

        In cryptocurrency and spot equity markets, allow_short=False (Long/Cash regime switching)
        provides optimal capital preservation by rotating to cash during negative sentiment/forecasts.
        Setting allow_short=True models synthetic futures/shorting.
        """
        n = len(pred_returns)
        pred_std = float(np.std(pred_returns)) + 1e-6
        threshold = threshold_std_factor * pred_std

        positions = np.zeros(n)
        current_pos = 0.0

        # Signal generation with hysteresis band
        for i in range(n):
            if pred_returns[i] > threshold:
                current_pos = 1.0  # Bullish: 100% Long allocation
            elif pred_returns[i] < -threshold:
                current_pos = -0.5 if allow_short else 0.0  # Bearish: Short hedge or 100% Cash preservation
            # else retain current_pos (hysteresis to prevent churning)
            positions[i] = current_pos

        # Calculate position turnover and transaction fees
        pos_changes = np.abs(np.diff(np.concatenate([[0.0], positions])))
        fee_costs = pos_changes * self.cost_rate

        # Strategy returns: position * actual_return - fee
        strategy_returns = positions * actual_returns - fee_costs
        strategy_ret_series = pd.Series(strategy_returns)

        # Benchmark returns (Buy & Hold)
        benchmark_returns = pd.Series(actual_returns)

        # Cumulative equity curves (starting capital = $10,000 USD)
        initial_capital = 10000.0
        equity_curve = initial_capital * np.cumprod(1.0 + strategy_returns)
        equity_curve = pd.Series(equity_curve)

        benchmark_equity = initial_capital * np.cumprod(1.0 + actual_returns)
        benchmark_equity = pd.Series(benchmark_equity)

        # Risk metrics
        total_ret = float((equity_curve.iloc[-1] / initial_capital) - 1.0)
        benchmark_total_ret = float((benchmark_equity.iloc[-1] / initial_capital) - 1.0)

        years = max(0.01, n / self.periods_per_year)
        ann_ret = float((1.0 + total_ret) ** (1.0 / years) - 1.0) if total_ret > -1.0 else -1.0
        ann_vol = float(strategy_ret_series.std() * np.sqrt(self.periods_per_year))

        sharpe = self.calculate_sharpe_ratio(strategy_ret_series)
        sortino = self.calculate_sortino_ratio(strategy_ret_series)
        max_dd, dd_series = self.calculate_max_drawdown(equity_curve)

        bench_sharpe = self.calculate_sharpe_ratio(benchmark_returns)
        bench_max_dd, _ = self.calculate_max_drawdown(benchmark_equity)

        calmar = (ann_ret / max_dd) if max_dd > 0 else 0.0

        # Trade metrics
        winning_periods = strategy_returns[strategy_returns > 0]
        losing_periods = strategy_returns[strategy_returns < 0]
        win_rate = (len(winning_periods) / max(1, len(strategy_returns[strategy_returns != 0]))) * 100.0
        profit_factor = float(abs(winning_trades.sum() / (losing_trades.sum() + 1e-8))) if "winning_trades" in locals() else float(abs(winning_periods.sum() / (losing_periods.sum() + 1e-8)))

        total_trade_count = int(np.sum(pos_changes > 0))
        bench_ann_ret = float((1.0 + benchmark_total_ret) ** (1.0 / years) - 1.0) if benchmark_total_ret > -1.0 else -1.0
        alpha = (ann_ret - bench_ann_ret) * 100.0

        return BacktestMetrics(
            total_return_pct=total_ret * 100.0,
            annualized_return_pct=ann_ret * 100.0,
            annualized_volatility_pct=ann_vol * 100.0,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd * 100.0,
            calmar_ratio=calmar,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trade_count,
            benchmark_total_return_pct=benchmark_total_ret * 100.0,
            benchmark_sharpe=bench_sharpe,
            benchmark_max_drawdown_pct=bench_max_dd * 100.0,
            alpha_pct=alpha,
            equity_curve=equity_curve,
            benchmark_equity=benchmark_equity,
            drawdown_series=dd_series,
        )
