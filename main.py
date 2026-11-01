"""
CrowdWisdomTrading: X Sentiment Price Prediction Agent
End-to-End CLI Orchestrator and Pipeline Runner.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from src.config import config
from src.database import RelationalDatabase, populate_database_from_ingestion
from src.evaluation import QuantitativeEvaluator
from src.features import FeatureEngineer
from src.ingestion import run_full_ingestion
from src.model import MultiHorizonPredictor, WalkForwardEngine
from src.reporting import ReportGenerator
from src.visuals import VisualMatrixGenerator

logger = logging.getLogger("cwt_main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def run_pipeline(step: str = "all", force_scrape: bool = False) -> None:
    """Execute end-to-end or modular pipeline steps."""
    logger.info("================================================================================")
    logger.info("   CROWDWISDOMTRADING: X SENTIMENT PRICE PREDICTION AGENT (HORIZONS n+0..n+5)   ")
    logger.info("================================================================================")
    logger.info("Target Asset: %s | Days: %d | Granularity: %s | Device: %s", config.TARGET_ASSET, config.HISTORY_DAYS, config.BAR_INTERVAL, config.DEVICE)

    # 1. Ingestion Step
    if step in ("all", "ingest"):
        logger.info("\n>>> STEP 1: INGESTION (Apify X Scraper, Google Doc Baseline, Tavily/Exa, Price Feeds)")
        ingest_res = run_full_ingestion()
        logger.info("Ingestion completed: %d price bars, %d tweets, %d news items.", len(ingest_res["prices"]), len(ingest_res["tweets"]), len(ingest_res["news"]))

    # 2. Database Step
    if step in ("all", "db"):
        logger.info("\n>>> STEP 2: RELATIONAL DATABASE & UTC/EST TIMEZONE ALIGNMENT")
        db = populate_database_from_ingestion()
        stats = db.get_stats()
        logger.info("Database synchronized: %s", stats)
    else:
        db = RelationalDatabase()

    # 3. Features Step
    if step in ("all", "features", "train", "evaluate", "visuals", "report"):
        logger.info("\n>>> STEP 3: QUANTITATIVE FEATURE ENGINEERING (Sentiment + Price ONLY)")
        engineer = FeatureEngineer(db=db)
        features_df = engineer.build_feature_dataset()
        feature_cols = engineer.get_feature_column_names(features_df)
        logger.info("Engineered %d features across %d aligned observations.", len(feature_cols), len(features_df))

    # 4. Modeling & Walk-Forward Validation
    if step in ("all", "train", "evaluate", "visuals", "report"):
        logger.info("\n>>> STEP 4: MULTI-HORIZON PREDICTOR & WALK-FORWARD VALIDATION (n+0 to n+5)")
        wf_engine = WalkForwardEngine(n_splits=config.WALK_FORWARD_SPLITS, embargo_periods=config.EMBARGO_PERIODS)
        horizon_evals = wf_engine.run_walk_forward(features_df, feature_cols)

    # 5. Financial Evaluation (Sharpe, Sortino, Max Drawdown)
    if step in ("all", "evaluate", "report"):
        logger.info("\n>>> STEP 5: RISK-ADJUSTED FINANCIAL EVALUATION & BACKTESTING")
        evaluator = QuantitativeEvaluator()
        # Backtest primary horizon (n+1)
        h1_eval = horizon_evals[1]
        backtest_metrics = evaluator.backtest_strategy(
            pred_returns=h1_eval.predictions,
            actual_returns=h1_eval.actual_returns,
            actual_prices=h1_eval.actual_prices,
            allow_short=False,
        )
        logger.info(
            "Backtest Results: Sharpe=%.2f | Sortino=%.2f | Max Drawdown=%.2f%% | Alpha=%.2f%%",
            backtest_metrics.sharpe_ratio,
            backtest_metrics.sortino_ratio,
            backtest_metrics.max_drawdown_pct,
            backtest_metrics.alpha_pct,
        )

    # 6. Visual Matrix Generation
    if step in ("all", "visuals", "report"):
        logger.info("\n>>> STEP 6: VISUAL MATRIX OUTPUT GENERATOR")
        visuals = VisualMatrixGenerator()
        p1 = visuals.generate_sentiment_vs_price_matrix(features_df)
        p2 = visuals.generate_multi_horizon_trajectories(horizon_evals)
        p3 = visuals.generate_walk_forward_equity_chart(backtest_metrics)

        # Fit single predictor for feature importances
        ref_predictor = MultiHorizonPredictor(horizon=1)
        split_idx = int(len(features_df) * 0.7)
        ref_predictor.fit(
            features_df.iloc[:split_idx][feature_cols],
            features_df.iloc[:split_idx]["target_return_h1"],
            features_df.iloc[:split_idx]["target_dir_h1"],
        )
        importances = ref_predictor.get_feature_importances()
        p4 = visuals.generate_feature_importance_chart(importances)
        logger.info("All visual matrix charts successfully saved in: %s", config.FIGURES_DIR)

    # 7. Automated Reporting
    if step in ("all", "report"):
        logger.info("\n>>> STEP 7: AUTOMATED REPORT GENERATION")
        reporter = ReportGenerator()
        from src.ingestion import ApifyXScraper, NewsIngestionAgent
        x_agent = ApifyXScraper()
        news_agent = NewsIngestionAgent()

        latest_tweets = x_agent.fetch_tweets(days=2, max_items=10)
        latest_news = news_agent.fetch_news(days=2)
        current_price = float(features_df["close"].iloc[-1])

        p_outlook = reporter.generate_sentiment_outlook_1page(
            df=features_df,
            horizon_evals=horizon_evals,
            current_price=current_price,
            latest_tweets=latest_tweets,
            latest_news=latest_news,
        )

        p_eval = reporter.generate_evaluation_report(
            horizon_evals=horizon_evals,
            backtest_metrics=backtest_metrics,
            db_stats=db.get_stats(),
            importances=importances,
        )

        logger.info("Reports successfully generated:")
        logger.info("  1-Page Sentiment Outlook: %s", p_outlook)
        logger.info("  Full Evaluation Report:   %s", p_eval)

    logger.info("================================================================================")
    logger.info("   PIPELINE COMPLETED SUCCESSFULLY. ALL ARTIFACTS READY FOR REVIEW.   ")
    logger.info("================================================================================")


def main():
    parser = argparse.ArgumentParser(
        description="CrowdWisdomTrading: X Sentiment Multi-Horizon Price Prediction Agent"
    )
    parser.add_argument(
        "--step",
        choices=["all", "ingest", "db", "features", "train", "evaluate", "visuals", "report"],
        default="all",
        help="Pipeline step to execute (default: all)",
    )
    parser.add_argument(
        "--force-scrape",
        action="store_true",
        help="Force live scraping via Apify / Tavily instead of reading cache",
    )
    args = parser.parse_args()
    run_pipeline(step=args.step, force_scrape=args.force_scrape)


if __name__ == "__main__":
    main()
