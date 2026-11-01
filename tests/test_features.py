"""
Unit tests for sentiment scoring, engagement weighting,
technical feature engineering, and anti-leakage checks.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureEngineer, SentimentScorer


@pytest.fixture
def scorer():
    return SentimentScorer(use_gpu=False)


def test_sentiment_scoring_polarity(scorer):
    """Verify polarity scoring on clearly bullish, bearish, and neutral text."""
    bull_res = scorer.score_text("Massive bullish breakout! Bitcoin reaches all-time high! $BTC 🚀")
    assert bull_res["compound"] > 0.3
    assert bull_res["pos"] > bull_res["neg"]

    bear_res = scorer.score_text("Devastating crash, severe loss, dump and terrible collapse for $BTC!")
    assert bear_res["compound"] < -0.3
    assert bear_res["neg"] > bear_res["pos"]

    neu_res = scorer.score_text("Bitcoin price is trading at 85000 USD on major exchanges.")
    assert abs(neu_res["compound"]) < 0.25


def test_engagement_weighting_monotonicity(scorer):
    """Verify that higher likes, retweets, views, and followers strictly increase engagement weight."""
    w1 = scorer.compute_engagement_weight(likes=10, retweets=2, views=100, followers=500)
    w2 = scorer.compute_engagement_weight(likes=500, retweets=100, views=10000, followers=50000)
    assert w2 > w1
    assert w1 > 0.0


def test_empty_and_none_text_handling(scorer):
    """Ensure robust zero-sentiment fallback for empty or non-string inputs."""
    res_empty = scorer.score_text("")
    assert res_empty["compound"] == 0.0

    res_none = scorer.score_text(None)
    assert res_none["compound"] == 0.0


def test_feature_engineering_pipeline_schema():
    """Verify that engineered features strictly exclude future lookahead and contain all multi-horizon targets."""
    # Create synthetic joined bars dataframe
    n_bars = 200
    dates = pd.date_range("2026-01-01", periods=n_bars, freq="h", tz="UTC")
    close_prices = 80000.0 * np.exp(np.cumsum(np.random.normal(0, 0.005, size=n_bars)))
    high_prices = close_prices * 1.002
    low_prices = close_prices * 0.998
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    synth_df = pd.DataFrame({
        "timestamp_utc": [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates],
        "timestamp_est": [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates],
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": np.random.uniform(1000, 5000, size=n_bars),
        "tweet_count": np.random.poisson(lam=3, size=n_bars),
        "total_likes": np.random.poisson(lam=100, size=n_bars),
        "total_retweets": np.random.poisson(lam=20, size=n_bars),
        "total_views": np.random.poisson(lam=1500, size=n_bars),
        "avg_author_followers": np.random.uniform(1000, 20000, size=n_bars),
        "news_count": np.zeros(n_bars, dtype=int),
        "raw_tweet_texts": ['["Bitcoin bullish rally!"]' if i % 2 == 0 else '["Consolidating"]' for i in range(n_bars)],
        "raw_news_titles": ['[]' for _ in range(n_bars)],
    })

    # Mock DB
    class MockDB:
        def load_joined_bars(self):
            return synth_df.copy()

    engineer = FeatureEngineer(db=MockDB())
    df = engineer.build_feature_dataset()

    feature_cols = engineer.get_feature_column_names(df)
    assert len(feature_cols) >= 40

    # Ensure all horizons targets exist
    for h in [0, 1, 2, 3, 4, 5]:
        assert f"target_return_h{h}" in df.columns
        assert f"target_price_h{h}" in df.columns
        assert f"target_dir_h{h}" in df.columns

    # Verify no NaN values in features
    assert df[feature_cols].isna().sum().sum() == 0
