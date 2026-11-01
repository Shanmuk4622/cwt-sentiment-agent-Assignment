"""
Visual Matrix Output Generator & Multi-Horizon Trajectory Visualizer:
Produces publication-grade visual charts illustrating:
1. X sentiment change against price trajectories as time evolves.
2. Cross-correlation lead-lag dynamics between sentiment and asset returns.
3. Multi-horizon (n+0 to n+5) price forecast trajectories.
4. Walk-forward cumulative equity curves and drawdown analysis.
5. Feature importance breakdown across sentiment, price, and interaction features.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import config
from src.evaluation import BacktestMetrics
from src.model import HorizonEvaluation

logger = logging.getLogger(__name__)

# Configure publication-grade styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class VisualMatrixGenerator:
    """Generates visual analytics and sentiment-vs-price trajectory charts."""

    def __init__(self, figures_dir: Optional[Path] = None):
        self.figures_dir = figures_dir or config.FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def generate_sentiment_vs_price_matrix(
        self, df: pd.DataFrame, filename: str = "sentiment_vs_price_matrix.png"
    ) -> Path:
        """Create a 4-panel visual matrix showing X sentiment dynamics vs price trajectory."""
        logger.info("Generating Sentiment vs Price Matrix Chart...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
        plt.subplots_adjust(hspace=0.28, wspace=0.22)

        plot_df = df.tail(min(500, len(df))).copy()
        plot_df["dt"] = pd.to_datetime(plot_df["timestamp_utc"])

        # -------------------------------------------------------------
        # Panel A: Price Trajectory & Engagement-Weighted Sentiment
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        color_price = "#1f77b4"
        color_sent = "#2ca02c"

        ax1.plot(plot_df["dt"], plot_df["close"], color=color_price, linewidth=1.8, label=f"{config.TARGET_ASSET} Price")
        ax1.set_ylabel(f"Price (USD)", color=color_price, fontsize=11, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=color_price)
        ax1.set_title("Panel A: Asset Price vs. Composite X Sentiment Trajectory", fontsize=12, fontweight="bold", pad=10)

        # Dual axis for sentiment
        ax1_twin = ax1.twinx()
        ax1_twin.plot(
            plot_df["dt"],
            plot_df["sentiment_composite_smoothed"],
            color=color_sent,
            linewidth=1.4,
            alpha=0.85,
            label="Smoothed Sentiment",
        )
        ax1_twin.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax1_twin.set_ylabel("Engagement-Weighted Sentiment Score", color=color_sent, fontsize=11, fontweight="bold")
        ax1_twin.tick_params(axis="y", labelcolor=color_sent)
        ax1_twin.grid(False)

        # -------------------------------------------------------------
        # Panel B: Sentiment-Price Cross-Correlation (Lead-Lag Structure)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        lags = range(-12, 13)
        corrs = []
        sent_series = df["sentiment_composite"]
        ret_series = df["return_1"]

        for lag in lags:
            if lag < 0:
                # Sentiment leads return (sentiment_t vs return_{t+|lag|})
                c = sent_series.corr(ret_series.shift(-lag))
            elif lag > 0:
                # Return leads sentiment (return_t vs sentiment_{t+lag})
                c = ret_series.corr(sent_series.shift(-lag))
            else:
                c = sent_series.corr(ret_series)
            corrs.append(c if not np.isnan(c) else 0.0)

        bar_colors = ["#2ca02c" if lag < 0 else ("#d62728" if lag > 0 else "#1f77b4") for lag in lags]
        ax2.bar(lags, corrs, color=bar_colors, alpha=0.85, edgecolor="#333333", width=0.7)
        ax2.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax2.axhline(0, color="black", linestyle="-", linewidth=0.6)
        ax2.set_xlabel("Lag (Hours): Negative = Sentiment Leads Price | Positive = Price Leads", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Pearson Correlation", fontsize=10, fontweight="bold")
        ax2.set_title("Panel B: Lead-Lag Cross-Correlation (Sentiment vs Return)", fontsize=12, fontweight="bold", pad=10)

        # -------------------------------------------------------------
        # Panel C: Forward Return by Sentiment Quintiles
        # -------------------------------------------------------------
        ax3 = axes[1, 0]
        df_valid = df.dropna(subset=["sentiment_composite", "target_return_h1"]).copy()
        bins = [-np.inf, -0.15, -0.02, 0.02, 0.15, np.inf]
        labels = ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"]
        df_valid["sentiment_quintile"] = pd.cut(df_valid["sentiment_composite"], bins=bins, labels=labels)
        avg_ret_by_q = df_valid.groupby("sentiment_quintile", observed=False)["target_return_h1"].mean() * 100.0
        q_colors = ["#d62728", "#ff7f0e", "#7f7f7f", "#8c564b", "#2ca02c"]

        bars = ax3.bar(
            avg_ret_by_q.index, avg_ret_by_q.values, color=q_colors, alpha=0.85, edgecolor="#333333", width=0.6
        )
        ax3.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax3.set_ylabel("Average 1-Step Forward Return (%)", fontsize=10, fontweight="bold")
        ax3.set_title("Panel C: Forward Asset Return by Sentiment Quintile", fontsize=12, fontweight="bold", pad=10)

        for b in bars:
            h = b.get_height()
            offset = 0.005 if h >= 0 else -0.015
            ax3.annotate(
                f"{h:.3f}%",
                (b.get_x() + b.get_width() / 2, h + offset),
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )

        # -------------------------------------------------------------
        # Panel D: Sentiment vs Subsequent Return Scatter & Regression
        # -------------------------------------------------------------
        ax4 = axes[1, 1]
        sample_pts = df_valid.sample(min(1200, len(df_valid)), random_state=42)
        sns.regplot(
            data=sample_pts,
            x="sentiment_composite",
            y="target_return_h1",
            ax=ax4,
            scatter_kws={"alpha": 0.35, "color": "#1f77b4", "s": 18},
            line_kws={"color": "#d62728", "linewidth": 2.0},
        )
        ax4.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax4.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax4.set_xlabel("Engagement-Weighted Sentiment Composite", fontsize=10, fontweight="bold")
        ax4.set_ylabel("Actual Forward Return (h=1)", fontsize=10, fontweight="bold")
        ax4.set_title("Panel D: Sentiment Impact Function with Robust OLS Fit", fontsize=12, fontweight="bold", pad=10)

        out_path = self.figures_dir / filename
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info("Saved Sentiment vs Price Matrix Chart to: %s", out_path)
        return out_path

    def generate_multi_horizon_trajectories(
        self,
        horizon_evals: Dict[int, HorizonEvaluation],
        filename: str = "multi_horizon_trajectories.png",
    ) -> Path:
        """Create multi-horizon trajectory comparison (n+0 to n+5) against actual price."""
        logger.info("Generating Multi-Horizon Trajectory Chart...")
        fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=300)
        plt.subplots_adjust(hspace=0.32, wspace=0.25)
        axes_flat = axes.flatten()

        for h, eval_res in horizon_evals.items():
            ax = axes_flat[h]
            n_pts = min(120, len(eval_res.actual_prices))
            x_range = np.arange(n_pts)

            actual = eval_res.actual_prices[-n_pts:]
            predicted = eval_res.predicted_prices[-n_pts:]

            ax.plot(x_range, actual, label="Actual Price", color="#1f77b4", linewidth=2.0)
            ax.plot(
                x_range,
                predicted,
                label=f"Predicted (n+{h})",
                color="#e377c2" if h == 0 else "#2ca02c",
                linestyle="--",
                linewidth=1.8,
            )

            ax.set_title(
                f"Horizon n+{h} Forecast | Dir Acc: {eval_res.directional_accuracy*100:.1f}% | AUC: {eval_res.auc_roc:.3f}",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Out-of-Sample Evaluation Bars (Hours)", fontsize=9)
            ax.set_ylabel("Price Level (USD)", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)

        out_path = self.figures_dir / filename
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info("Saved Multi-Horizon Trajectory Chart to: %s", out_path)
        return out_path

    def generate_walk_forward_equity_chart(
        self,
        metrics: BacktestMetrics,
        filename: str = "walk_forward_equity_curve.png",
    ) -> Path:
        """Plot Walk-Forward cumulative equity curve vs benchmark and underwater drawdown."""
        logger.info("Generating Walk-Forward Cumulative Equity Chart...")
        fig, (ax_equity, ax_dd) = plt.subplots(2, 1, figsize=(15, 9), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.5, 1.0]})
        plt.subplots_adjust(hspace=0.1)

        x_vals = np.arange(len(metrics.equity_curve))

        # Top Panel: Cumulative Equity
        ax_equity.plot(x_vals, metrics.equity_curve, label=f"CWT Sentiment Agent (Sharpe: {metrics.sharpe_ratio:.2f})", color="#2ca02c", linewidth=2.2)
        ax_equity.plot(x_vals, metrics.benchmark_equity, label=f"Buy & Hold Benchmark (Sharpe: {metrics.benchmark_sharpe:.2f})", color="#7f7f7f", linestyle="--", linewidth=1.6)
        ax_equity.set_ylabel("Portfolio Value ($USD)", fontsize=11, fontweight="bold")
        ax_equity.set_title(
            f"Walk-Forward Out-of-Sample Performance | Strategy Return: {metrics.total_return_pct:+.2f}% vs Benchmark: {metrics.benchmark_total_return_pct:+.2f}%",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax_equity.legend(loc="upper left", fontsize=10, frameon=True)

        # Bottom Panel: Underwater Drawdown
        ax_dd.fill_between(x_vals, metrics.drawdown_series * 100.0, 0, color="#d62728", alpha=0.45, label=f"Max DD: {metrics.max_drawdown_pct:.1f}%")
        ax_dd.plot(x_vals, metrics.drawdown_series * 100.0, color="#d62728", linewidth=1.2)
        ax_dd.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
        ax_dd.set_xlabel("Out-of-Sample Execution Bars (1h)", fontsize=10, fontweight="bold")
        ax_dd.legend(loc="lower left", fontsize=9)

        out_path = self.figures_dir / filename
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info("Saved Walk-Forward Equity Chart to: %s", out_path)
        return out_path

    def generate_feature_importance_chart(
        self,
        importances: pd.Series,
        filename: str = "feature_importance_matrix.png",
        top_n: int = 15,
    ) -> Path:
        """Plot horizontal bar chart of top predictive features."""
        logger.info("Generating Feature Importance Chart...")
        plt.figure(figsize=(12, 7), dpi=300)

        top_feats = importances.head(top_n).iloc[::-1]

        # Categorize features by color
        colors = []
        for name in top_feats.index:
            if "sentiment" in name or "tweet" in name or "bull_bear" in name or "engagement" in name:
                colors.append("#2ca02c")  # Sentiment: Green
            elif "vol_scaled" in name or "mult" in name or "divergence" in name or "mom" in name and "vol" in name:
                colors.append("#ff7f0e")  # Interaction: Orange
            else:
                colors.append("#1f77b4")  # Price/Technicals: Blue

        bars = plt.barh(top_feats.index, top_feats.values, color=colors, alpha=0.85, edgecolor="#333333")
        plt.xlabel("Ensemble Relative Feature Importance", fontsize=10, fontweight="bold")
        plt.title("Top Feature Importances (Sentiment, Technicals & Cross-Interactions)", fontsize=12, fontweight="bold", pad=10)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ca02c", edgecolor="#333333", label="X Sentiment & Engagement"),
            Patch(facecolor="#ff7f0e", edgecolor="#333333", label="Sentiment-Price Interaction"),
            Patch(facecolor="#1f77b4", edgecolor="#333333", label="Price Momentum & Volatility"),
        ]
        plt.legend(handles=legend_elements, loc="lower right", fontsize=9)

        out_path = self.figures_dir / filename
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info("Saved Feature Importance Chart to: %s", out_path)
        return out_path
