"""
Quantitative Feature Engineering Layer:
Computes engagement-weighted sentiment, sentiment momentum/dispersion,
price momentum/volatility, cross-interaction features, and multi-horizon targets (n+0 to n+5).
Strictly adheres to: ONLY sentiment and price data.
"""

from datetime import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd

from src.config import config
from src.database import RelationalDatabase

logger = logging.getLogger(__name__)


class SentimentScorer:
    """Engagement-weighted sentiment scoring engine using VADER."""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception:
            nltk.download("vader_lexicon", quiet=True)
            self.sia = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> Dict[str, float]:
        """Compute compound, positive, negative, and neutral sentiment scores."""
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        return self.sia.polarity_scores(text)

    def compute_engagement_weight(
        self, likes: float, retweets: float, views: float, followers: float
    ) -> float:
        """Engagement weighting formula:

        w = ln(1 + w_l*likes + w_r*retweets + w_v*views + w_f*followers)
        """
        w_cfg = config.ENGAGEMENT_WEIGHTS
        raw_score = (
            w_cfg.get("likes", 1.0) * max(0.0, likes)
            + w_cfg.get("retweets", 2.0) * max(0.0, retweets)
            + w_cfg.get("views", 0.1) * max(0.0, views)
            + w_cfg.get("followers", 0.01) * max(0.0, followers)
        )
        return float(np.log1p(raw_score))


class FeatureEngineer:
    """Quantitative feature engineer generating sentiment, price, and interaction features."""

    def __init__(self, db: Optional[RelationalDatabase] = None):
        self.db = db or RelationalDatabase()
        self.scorer = SentimentScorer(use_gpu=config.DEVICE == "cuda")

    def build_feature_dataset(self) -> pd.DataFrame:
        """Load joined bars and compute all quantitative features and multi-horizon targets."""
        logger.info("Loading joined_bars from relational database...")
        df = self.db.load_joined_bars()
        if df.empty:
            raise ValueError("Database table 'joined_bars' is empty. Ingest data first.")

        df["dt_utc"] = pd.to_datetime(df["timestamp_utc"])
        df.sort_values("dt_utc", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info("Computing engagement-weighted sentiment scores across %d bars...", len(df))
        df = self._add_sentiment_scores(df)

        logger.info("Generating price-based quantitative technical features...")
        df = self._add_price_features(df)

        logger.info("Engineering sentiment-price interaction features...")
        df = self._add_interaction_features(df)

        logger.info("Generating multi-horizon forward targets (n+0 to n+5)...")
        df = self._add_multi_horizon_targets(df)

        # Drop initial warmup bars where rolling statistics contain NaNs
        initial_len = len(df)
        warmup_period = max(config.ROLLING_WINDOWS) + 5
        clean_df = df.iloc[warmup_period:].copy()
        clean_df.dropna(subset=[f"target_return_h{h}" for h in config.HORIZONS], inplace=True)
        clean_df.reset_index(drop=True, inplace=True)

        logger.info("Feature engineering complete: %d usable bars retained (warmup removed: %d).", len(clean_df), initial_len - len(clean_df))
        return clean_df

    def _add_sentiment_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse tweet texts & compute engagement-weighted sentiment and rolling metrics."""
        tweet_compound_list = []
        tweet_pos_list = []
        tweet_neg_list = []
        weighted_sentiment_list = []
        news_sentiment_list = []

        for _, row in df.iterrows():
            texts_raw = row["raw_tweet_texts"]
            news_raw = row["raw_news_titles"]

            # Parse tweets
            try:
                texts = json.loads(texts_raw) if isinstance(texts_raw, str) else []
            except Exception:
                texts = []

            # Parse news
            try:
                news_titles = json.loads(news_raw) if isinstance(news_raw, str) else []
            except Exception:
                news_titles = []

            # Compute news sentiment
            if news_titles:
                news_compounds = [self.scorer.score_text(t)["compound"] for t in news_titles]
                n_score = float(np.mean(news_compounds))
            else:
                n_score = 0.0
            news_sentiment_list.append(n_score)

            if not texts:
                tweet_compound_list.append(0.0)
                tweet_pos_list.append(0.0)
                tweet_neg_list.append(0.0)
                weighted_sentiment_list.append(0.0)
                continue

            scores = [self.scorer.score_text(t) for t in texts]
            compounds = np.array([s["compound"] for s in scores])
            poses = np.array([s["pos"] for s in scores])
            negs = np.array([s["neg"] for s in scores])

            # Engagement weighting proxy per tweet in bar
            base_w = self.scorer.compute_engagement_weight(
                likes=row["total_likes"] / max(1, len(texts)),
                retweets=row["total_retweets"] / max(1, len(texts)),
                views=row["total_views"] / max(1, len(texts)),
                followers=row["avg_author_followers"],
            )
            w = max(0.1, base_w)

            tweet_compound_list.append(float(np.mean(compounds)))
            tweet_pos_list.append(float(np.mean(poses)))
            tweet_neg_list.append(float(np.mean(negs)))
            weighted_sentiment_list.append(float(np.mean(compounds) * (1.0 + 0.1 * np.log1p(w))))

        df["sentiment_raw"] = tweet_compound_list
        df["sentiment_pos"] = tweet_pos_list
        df["sentiment_neg"] = tweet_neg_list
        df["sentiment_weighted"] = weighted_sentiment_list
        df["sentiment_news"] = news_sentiment_list

        # Combined Sentiment (80% X Sentiment, 20% News Sentiment)
        df["sentiment_composite"] = 0.8 * df["sentiment_weighted"] + 0.2 * df["sentiment_news"]

        # Exponential moving decay for bars without tweets to carry forward sentiment inertia
        df["sentiment_composite_smoothed"] = df["sentiment_composite"].ewm(span=6, min_periods=1).mean()

        # Sentiment volume dynamics
        df["tweet_volume_log"] = np.log1p(df["tweet_count"])
        df["engagement_total_log"] = np.log1p(df["total_likes"] + df["total_retweets"] + df["total_views"])

        # Rolling sentiment metrics across configured windows
        for w in config.ROLLING_WINDOWS:
            df[f"sentiment_roll_mean_{w}"] = df["sentiment_composite"].rolling(w).mean()
            df[f"sentiment_roll_std_{w}"] = df["sentiment_composite"].rolling(w).std().fillna(0.0)
            # Sentiment z-score
            mean = df[f"sentiment_roll_mean_{w}"]
            std = df[f"sentiment_roll_std_{w}"] + 1e-6
            df[f"sentiment_zscore_{w}"] = (df["sentiment_composite"] - mean) / std

        # Sentiment momentum & acceleration
        df["sentiment_velocity_3"] = df["sentiment_composite"] - df["sentiment_composite"].shift(3)
        df["sentiment_velocity_6"] = df["sentiment_composite"] - df["sentiment_composite"].shift(6)
        df["sentiment_acceleration"] = df["sentiment_velocity_3"] - df["sentiment_velocity_3"].shift(3)

        # Sentiment dispersion / bull-bear spread
        df["bull_bear_spread"] = df["sentiment_pos"] - df["sentiment_neg"]

        # Abnormal tweet volume (relative to 24h rolling average)
        vol_24 = df["tweet_count"].rolling(24).mean().fillna(1.0) + 1.0
        df["sentiment_volume_abnormal"] = df["tweet_count"] / vol_24

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute strictly price-based technical and quantitative features."""
        # Log returns
        df["return_1"] = np.log(df["close"] / df["close"].shift(1))
        df["return_bar"] = np.log(df["close"] / df["open"])

        # Lagged returns
        for lag in [1, 2, 3, 5, 10, 24]:
            df[f"lag_return_{lag}"] = df["return_1"].shift(lag)

        # High-Low spread (Parkinson volatility proxy)
        df["hl_spread"] = np.log(df["high"] / (df["low"] + 1e-6))

        # Realized volatility across windows
        for w in config.ROLLING_WINDOWS:
            df[f"realized_vol_{w}"] = df["return_1"].rolling(w).std().fillna(0.0)

        # Price momentum (percentage change over horizons)
        for w in [3, 6, 12, 24, 72]:
            df[f"price_momentum_{w}"] = df["close"] / df["close"].shift(w) - 1.0

        # Relative Strength Index (RSI 14)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-6)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # MACD (12, 26, 9)
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd_line"] = ema_12 - ema_26
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]

        # Bollinger Bands %B (20 period, 2 std)
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std().fillna(1.0)
        upper_bb = sma_20 + 2 * std_20
        lower_bb = sma_20 - 2 * std_20
        df["bb_percent_b"] = (df["close"] - lower_bb) / (upper_bb - lower_bb + 1e-6)

        # Volume z-score
        vol_mean = df["volume"].rolling(24).mean()
        vol_std = df["volume"].rolling(24).std().fillna(1.0) + 1.0
        df["volume_zscore_24"] = (df["volume"] - vol_mean) / vol_std

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-features between sentiment and price dynamics."""
        # Sentiment-Return interaction
        df["sentiment_return_mult"] = df["sentiment_composite"] * df["return_1"]

        # Sentiment-Price Divergence: contrarian signal when sentiment direction opposes price movement
        sentiment_sign = np.sign(df["sentiment_composite"])
        return_sign = np.sign(df["return_1"].fillna(0))
        df["sentiment_price_divergence"] = (sentiment_sign != return_sign).astype(float)

        # Volatility-scaled sentiment
        df["vol_scaled_sentiment"] = df["sentiment_composite"] / (df["realized_vol_24"] + 1e-4)

        # Volume-weighted sentiment momentum
        df["vol_weighted_sentiment_mom"] = df["sentiment_velocity_6"] * np.log1p(df["volume_zscore_24"].clip(lower=0))

        return df

    def _add_multi_horizon_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create forward return and price targets for horizons n+0 to n+5.

        n+0: current bar return (close vs open)
        n+1 to n+5: cumulative forward return ln(close_{t+k} / close_t)
        """
        # Horizon n+0: Intraday bar return from current open to close
        df["target_return_h0"] = np.log(df["close"] / df["open"])
        df["target_price_h0"] = df["close"]
        df["target_dir_h0"] = (df["target_return_h0"] > 0).astype(int)

        # Horizons n+1 to n+5: Forward returns
        for h in [1, 2, 3, 4, 5]:
            forward_close = df["close"].shift(-h)
            df[f"target_return_h{h}"] = np.log(forward_close / df["close"])
            df[f"target_price_h{h}"] = forward_close
            df[f"target_dir_h{h}"] = (df[f"target_return_h{h}"] > 0).astype(int)

        return df

    def get_feature_column_names(self, df: pd.DataFrame) -> List[str]:
        """Return list of strictly valid input feature column names."""
        exclude_prefixes = ("target_", "timestamp_", "dt_", "raw_")
        exclude_exact = {"open", "high", "low", "close", "volume"}

        feature_cols = [
            c for c in df.columns
            if not any(c.startswith(p) for p in exclude_prefixes)
            and c not in exclude_exact
        ]
        return feature_cols


def run_feature_pipeline() -> pd.DataFrame:
    """Build and return feature-engineered dataset."""
    engineer = FeatureEngineer()
    df = engineer.build_feature_dataset()
    features = engineer.get_feature_column_names(df)
    logger.info("Engineered %d quantitative features across %d rows.", len(features), len(df))
    logger.info("Sample features: %s", features[:10])
    return df


if __name__ == "__main__":
    run_feature_pipeline()
