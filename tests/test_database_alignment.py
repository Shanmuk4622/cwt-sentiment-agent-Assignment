"""
Unit and integration tests for SQLite database schema, UTC/EST alignment,
and bar bucketing joins.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import pytz

from src.database import RelationalDatabase


@pytest.fixture
def temp_db():
    """Create a temporary isolated SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    db = RelationalDatabase(db_path=db_path)
    db.init_db()
    yield db
    if db_path.exists():
        try:
            os.remove(db_path)
        except Exception:
            pass


def test_database_initialization(temp_db):
    """Verify that all relational tables are created properly."""
    stats = temp_db.get_stats()
    assert "prices" in stats
    assert "tweets" in stats
    assert "news" in stats
    assert "joined_bars" in stats
    assert all(count == 0 for count in stats.values())


def test_price_insertion_and_timezone_alignment(temp_db):
    """Test insertion of price bars with UTC and EST timestamp validation."""
    now_utc = datetime(2026, 10, 15, 14, 0, 0, tzinfo=timezone.utc)
    est_tz = pytz.timezone("US/Eastern")
    now_est = now_utc.astimezone(est_tz)

    sample_prices = pd.DataFrame([
        {
            "timestamp_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_est": now_est.strftime("%Y-%m-%d %H:%M:%S"),
            "open": 85000.0,
            "high": 85500.0,
            "low": 84800.0,
            "close": 85200.0,
            "volume": 1250000.0,
        }
    ])

    rows = temp_db.insert_prices(sample_prices)
    assert rows == 1
    stats = temp_db.get_stats()
    assert stats["prices"] == 1


def test_joined_bars_bucketing_and_no_lookahead(temp_db):
    """Test that tweets are bucketed into prior/current price bar intervals without lookahead."""
    dt_bar1 = datetime(2026, 10, 15, 10, 0, 0, tzinfo=timezone.utc)
    dt_bar2 = datetime(2026, 10, 15, 11, 0, 0, tzinfo=timezone.utc)
    est_tz = pytz.timezone("US/Eastern")

    prices = pd.DataFrame([
        {
            "timestamp_utc": dt_bar1.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_est": dt_bar1.astimezone(est_tz).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 80000.0, "high": 80200.0, "low": 79900.0, "close": 80100.0, "volume": 1000.0,
        },
        {
            "timestamp_utc": dt_bar2.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_est": dt_bar2.astimezone(est_tz).strftime("%Y-%m-%d %H:%M:%S"),
            "open": 80100.0, "high": 80400.0, "low": 80050.0, "close": 80350.0, "volume": 1200.0,
        },
    ])
    temp_db.insert_prices(prices)

    # Tweet 1 occurs during bar 1 (10:15 UTC)
    tweet_time1 = datetime(2026, 10, 15, 10, 15, 0, tzinfo=timezone.utc)
    # Tweet 2 occurs during bar 1 (10:45 UTC)
    tweet_time2 = datetime(2026, 10, 15, 10, 45, 0, tzinfo=timezone.utc)

    tweets = [
        {
            "tweet_id": "tw_01",
            "timestamp_utc": tweet_time1.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_est": tweet_time1.astimezone(est_tz).strftime("%Y-%m-%d %H:%M:%S"),
            "text": "Bullish Bitcoin rally! $BTC",
            "likes": 100,
            "retweets": 20,
            "views": 1500,
            "author_followers": 5000,
            "source": "test",
        },
        {
            "tweet_id": "tw_02",
            "timestamp_utc": tweet_time2.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_est": tweet_time2.astimezone(est_tz).strftime("%Y-%m-%d %H:%M:%S"),
            "text": "Breaking out! $BTC",
            "likes": 50,
            "retweets": 10,
            "views": 800,
            "author_followers": 2000,
            "source": "test",
        },
    ]
    temp_db.insert_tweets(tweets)

    # Build joined bars
    joined_df = temp_db.build_joined_bars(interval="1h")
    assert len(joined_df) == 2

    # Verify Bar 1 has 2 tweets, total_likes = 150, total_views = 2300
    bar1 = joined_df.iloc[0]
    assert bar1["tweet_count"] == 2
    assert bar1["total_likes"] == 150
    assert bar1["total_views"] == 2300

    # Verify Bar 2 has 0 tweets (no lookahead or leakage)
    bar2 = joined_df.iloc[1]
    assert bar2["tweet_count"] == 0
    assert bar2["total_likes"] == 0
