"""
Relational Database Layer: SQLite schema, UTC/EST timezone alignment,
and relational join operations uniting X posts, news, and OHLCV price bars.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pytz
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)

from src.config import config

logger = logging.getLogger(__name__)


class RelationalDatabase:
    """Manages the SQLite relational database layer uniting X sentiment, news, and prices."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path.as_posix()}", echo=False)
        self.metadata = MetaData()
        self._define_schema()

    def _define_schema(self) -> None:
        """Define relational tables with UTC/EST timestamps and optimized indices."""
        self.prices_table = Table(
            "prices",
            self.metadata,
            Column("timestamp_utc", String, primary_key=True),
            Column("timestamp_est", String, index=True),
            Column("open", Float, nullable=False),
            Column("high", Float, nullable=False),
            Column("low", Float, nullable=False),
            Column("close", Float, nullable=False),
            Column("volume", Float, nullable=False),
        )

        self.tweets_table = Table(
            "tweets",
            self.metadata,
            Column("tweet_id", String, primary_key=True),
            Column("timestamp_utc", String, index=True),
            Column("timestamp_est", String, index=True),
            Column("text", Text, nullable=False),
            Column("likes", Integer, default=0),
            Column("retweets", Integer, default=0),
            Column("views", Integer, default=0),
            Column("author_followers", Integer, default=0),
            Column("source", String, default="apify"),
        )

        self.news_table = Table(
            "news",
            self.metadata,
            Column("article_id", String, primary_key=True),
            Column("timestamp_utc", String, index=True),
            Column("timestamp_est", String, index=True),
            Column("title", Text, nullable=False),
            Column("content", Text, default=""),
            Column("source", String, default="tavily"),
            Column("url", String, default=""),
        )

        self.joined_bars_table = Table(
            "joined_bars",
            self.metadata,
            Column("timestamp_utc", String, primary_key=True),
            Column("timestamp_est", String, index=True),
            Column("open", Float, nullable=False),
            Column("high", Float, nullable=False),
            Column("low", Float, nullable=False),
            Column("close", Float, nullable=False),
            Column("volume", Float, nullable=False),
            Column("tweet_count", Integer, default=0),
            Column("total_likes", Integer, default=0),
            Column("total_retweets", Integer, default=0),
            Column("total_views", Integer, default=0),
            Column("avg_author_followers", Float, default=0.0),
            Column("news_count", Integer, default=0),
            Column("raw_tweet_texts", Text, default="[]"),
            Column("raw_news_titles", Text, default="[]"),
        )

    def init_db(self) -> None:
        """Create tables and verify database integrity."""
        self.metadata.create_all(self.engine)
        logger.info("Initialized SQLite database schema at: %s", self.db_path)

    def insert_prices(self, price_df: pd.DataFrame) -> int:
        """Insert or replace OHLCV price records."""
        if price_df.empty:
            return 0
        df = price_df.copy()
        df["timestamp_utc"] = df["timestamp_utc"].astype(str)
        df["timestamp_est"] = df["timestamp_est"].astype(str)

        with self.engine.begin() as conn:
            df.to_sql("prices", conn, if_exists="replace", index=False)
        logger.info("Inserted %d price bars into 'prices' table.", len(df))
        return len(df)

    def insert_tweets(self, tweets: List[Dict[str, Any]]) -> int:
        """Insert or replace tweet records."""
        if not tweets:
            return 0
        df = pd.DataFrame(tweets)
        with self.engine.begin() as conn:
            df.to_sql("tweets", conn, if_exists="replace", index=False)
        logger.info("Inserted %d tweets into 'tweets' table.", len(df))
        return len(df)

    def insert_news(self, news: List[Dict[str, Any]]) -> int:
        """Insert or replace news records."""
        if not news:
            return 0
        df = pd.DataFrame(news)
        with self.engine.begin() as conn:
            df.to_sql("news", conn, if_exists="replace", index=False)
        logger.info("Inserted %d news items into 'news' table.", len(df))
        return len(df)

    def build_joined_bars(self, interval: str = "1h") -> pd.DataFrame:
        """Relational join: buckets tweets and news strictly into prior price bar intervals.

        Ensures zero lookahead bias by grouping tweets/news published in [T_bar - Delta, T_bar].
        """
        logger.info("Building joined_bars relation aligning X sentiment and news to %s price bars...", interval)
        with self.engine.connect() as conn:
            prices_df = pd.read_sql_table("prices", conn)
            tweets_df = pd.read_sql_table("tweets", conn)
            news_df = pd.read_sql_table("news", conn)

        if prices_df.empty:
            raise ValueError("Cannot build joined_bars: 'prices' table is empty.")

        prices_df["dt_utc"] = pd.to_datetime(prices_df["timestamp_utc"])
        prices_df.sort_values("dt_utc", inplace=True)
        prices_df.reset_index(drop=True, inplace=True)

        # Parse tweets timestamps
        if not tweets_df.empty:
            tweets_df["dt_utc"] = pd.to_datetime(tweets_df["timestamp_utc"])
            if interval == "1h":
                tweets_df["bar_bucket"] = tweets_df["dt_utc"].dt.floor("h")
            else:
                tweets_df["bar_bucket"] = tweets_df["dt_utc"].dt.floor("d")
        else:
            tweets_df["bar_bucket"] = pd.Series(dtype="datetime64[ns]")

        # Parse news timestamps
        if not news_df.empty:
            news_df["dt_utc"] = pd.to_datetime(news_df["timestamp_utc"])
            if interval == "1h":
                news_df["bar_bucket"] = news_df["dt_utc"].dt.floor("h")
            else:
                news_df["bar_bucket"] = news_df["dt_utc"].dt.floor("d")
        else:
            news_df["bar_bucket"] = pd.Series(dtype="datetime64[ns]")

        # Aggregate tweets per bucket
        tweet_agg = {}
        if not tweets_df.empty:
            grouped = tweets_df.groupby("bar_bucket")
            for bucket, group in grouped:
                tweet_agg[bucket] = {
                    "tweet_count": len(group),
                    "total_likes": int(group["likes"].sum()),
                    "total_retweets": int(group["retweets"].sum()),
                    "total_views": int(group["views"].sum()),
                    "avg_author_followers": float(group["author_followers"].mean()),
                    "raw_tweet_texts": json.dumps(group["text"].tolist()[:15]),  # Keep top sample
                    "tweets_raw_list": group[["text", "likes", "retweets", "views", "author_followers"]].to_dict(orient="records"),
                }

        # Aggregate news per bucket
        news_agg = {}
        if not news_df.empty:
            grouped_news = news_df.groupby("bar_bucket")
            for bucket, group in grouped_news:
                news_agg[bucket] = {
                    "news_count": len(group),
                    "raw_news_titles": json.dumps(group["title"].tolist()),
                }

        # Merge with price bars
        joined_rows = []
        est_tz = pytz.timezone(config.TIMEZONE_DISPLAY)

        for _, row in prices_df.iterrows():
            dt_bar = row["dt_utc"]
            bucket_key = dt_bar.floor("h") if interval == "1h" else dt_bar.floor("d")

            t_info = tweet_agg.get(bucket_key, {
                "tweet_count": 0,
                "total_likes": 0,
                "total_retweets": 0,
                "total_views": 0,
                "avg_author_followers": 0.0,
                "raw_tweet_texts": "[]",
                "tweets_raw_list": [],
            })

            n_info = news_agg.get(bucket_key, {
                "news_count": 0,
                "raw_news_titles": "[]",
            })

            joined_rows.append({
                "timestamp_utc": row["timestamp_utc"],
                "timestamp_est": row["timestamp_est"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "tweet_count": t_info["tweet_count"],
                "total_likes": t_info["total_likes"],
                "total_retweets": t_info["total_retweets"],
                "total_views": t_info["total_views"],
                "avg_author_followers": t_info["avg_author_followers"],
                "news_count": n_info["news_count"],
                "raw_tweet_texts": t_info["raw_tweet_texts"],
                "raw_news_titles": n_info["raw_news_titles"],
            })

        joined_df = pd.DataFrame(joined_rows)
        with self.engine.begin() as conn:
            joined_df.to_sql("joined_bars", conn, if_exists="replace", index=False)

        logger.info(
            "Successfully created 'joined_bars' with %d aligned bars (%d bars with active tweets).",
            len(joined_df),
            (joined_df["tweet_count"] > 0).sum(),
        )
        return joined_df

    def load_joined_bars(self) -> pd.DataFrame:
        """Load the joined bars relation."""
        with self.engine.connect() as conn:
            return pd.read_sql_table("joined_bars", conn)

    def get_stats(self) -> Dict[str, int]:
        """Return row counts for all tables."""
        stats = {}
        with self.engine.connect() as conn:
            for table_name in ["prices", "tweets", "news", "joined_bars"]:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                    stats[table_name] = int(result)
                except Exception:
                    stats[table_name] = 0
        return stats


def populate_database_from_ingestion() -> RelationalDatabase:
    """End-to-end database pipeline: ingest or load raw cached files and build joined relation."""
    from src.ingestion import ApifyXScraper, NewsIngestionAgent, PriceIngestion

    db = RelationalDatabase()
    db.init_db()

    # Load / ingest price data
    price_agent = PriceIngestion()
    price_df = price_agent.fetch_price_data(days=config.HISTORY_DAYS, interval=config.BAR_INTERVAL)
    db.insert_prices(price_df)

    # Load / ingest tweets
    x_agent = ApifyXScraper()
    tweets = x_agent.fetch_tweets(days=config.HISTORY_DAYS, max_items=config.APIFY_MAX_ITEMS)
    db.insert_tweets(tweets)

    # Load / ingest news
    news_agent = NewsIngestionAgent()
    news = news_agent.fetch_news(days=config.HISTORY_DAYS)
    db.insert_news(news)

    # Build aligned relational join
    db.build_joined_bars(interval=config.BAR_INTERVAL)
    stats = db.get_stats()
    logger.info("Database populated successfully. Stats: %s", stats)
    return db


if __name__ == "__main__":
    populate_database_from_ingestion()
