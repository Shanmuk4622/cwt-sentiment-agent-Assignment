"""
Configuration loader, environment variables, and constants.
"""

import os
from pathlib import Path
from typing import Any, Dict, List
import yaml
from dotenv import load_dotenv

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
load_dotenv(ROOT_DIR / ".env")


def load_yaml_config(config_path: Path = ROOT_DIR / "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration with defaults fallback."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_raw_config = load_yaml_config()


class AppConfig:
    """Centralized configuration object for the CWT Sentiment Agent."""

    # Paths
    PROJECT_ROOT: Path = ROOT_DIR
    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DATA_DIR: Path = ROOT_DIR / "data" / "raw"
    REPORTS_DIR: Path = ROOT_DIR / "reports"
    FIGURES_DIR: Path = ROOT_DIR / "reports" / "figures"
    DB_PATH: Path = ROOT_DIR / _raw_config.get("database", {}).get("path", "data/cwt_sentiment.db")

    # API Keys & Credentials
    APIFY_API_TOKEN: str = os.getenv("APIFY_API_TOKEN", "")
    APIFY_ACTOR_ID: str = os.getenv(
        "APIFY_TWITTER_ACTOR_ID",
        _raw_config.get("ingestion", {}).get("apify", {}).get("actor_id", "apidojo/tweet-scraper")
    )
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")

    # Asset Settings
    TARGET_ASSET: str = os.getenv(
        "TARGET_ASSET",
        _raw_config.get("asset", {}).get("ticker", "BTC-USD")
    )
    ASSET_NAME: str = _raw_config.get("asset", {}).get("name", "Bitcoin")
    CASHTAGS: List[str] = _raw_config.get("asset", {}).get("cashtags", ["$BTC", "#BTC", "BTC", "Bitcoin"])
    HISTORY_DAYS: int = int(_raw_config.get("asset", {}).get("history_days", 180))
    BAR_INTERVAL: str = _raw_config.get("asset", {}).get("bar_interval", "1h")
    TIMEZONE_STORAGE: str = _raw_config.get("asset", {}).get("timezone_storage", "UTC")
    TIMEZONE_DISPLAY: str = _raw_config.get("asset", {}).get("timezone_display", "US/Eastern")

    # Ingestion Settings
    APIFY_QUERIES: List[str] = _raw_config.get("ingestion", {}).get("apify", {}).get("queries", ["$BTC", "Bitcoin price"])
    APIFY_MAX_ITEMS: int = int(_raw_config.get("ingestion", {}).get("apify", {}).get("max_items", 5000))
    GOOGLE_DOC_URL: str = _raw_config.get("ingestion", {}).get("google_doc", {}).get(
        "url", "https://docs.google.com/document/d/1w-Y86OjkiOjlIIth0VgTAohql9zcGi3uoDNYTaTsZuw/edit?tab=t.0"
    )
    GOOGLE_DOC_EXPORT_URL: str = _raw_config.get("ingestion", {}).get("google_doc", {}).get(
        "export_url", "https://docs.google.com/document/d/1w-Y86OjkiOjlIIth0VgTAohql9zcGi3uoDNYTaTsZuw/export?format=txt"
    )
    TAVILY_MAX_RESULTS: int = int(_raw_config.get("ingestion", {}).get("tavily", {}).get("max_results", 100))
    EXA_NUM_RESULTS: int = int(_raw_config.get("ingestion", {}).get("exa", {}).get("num_results", 50))

    # Features & Horizons
    HORIZONS: List[int] = _raw_config.get("modeling", {}).get("horizons", [0, 1, 2, 3, 4, 5])
    ROLLING_WINDOWS: List[int] = _raw_config.get("features", {}).get("rolling_windows", [3, 6, 12, 24, 72])
    ENGAGEMENT_WEIGHTS: Dict[str, float] = _raw_config.get("features", {}).get("engagement_weights", {
        "likes": 1.0,
        "retweets": 2.0,
        "views": 0.1,
        "followers": 0.01,
    })

    # Modeling & Walk-Forward
    WALK_FORWARD_SPLITS: int = int(_raw_config.get("modeling", {}).get("walk_forward", {}).get("n_splits", 5))
    EMBARGO_PERIODS: int = int(_raw_config.get("modeling", {}).get("walk_forward", {}).get("embargo_periods", 5))
    TEST_SIZE_RATIO: float = float(_raw_config.get("modeling", {}).get("walk_forward", {}).get("test_size_ratio", 0.15))

    # Device
    DEVICE: str = "cuda" if os.getenv("DEVICE", "cuda") == "cuda" else "cpu"

    # Evaluation
    RISK_FREE_RATE: float = float(_raw_config.get("evaluation", {}).get("risk_free_rate", 0.04))
    TRANSACTION_COST_BPS: float = float(_raw_config.get("evaluation", {}).get("transaction_cost_bps", 10))
    SLIPPAGE_BPS: float = float(_raw_config.get("evaluation", {}).get("slippage_bps", 5))

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary project directories if they do not exist."""
        for path in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.REPORTS_DIR, cls.FIGURES_DIR]:
            path.mkdir(parents=True, exist_ok=True)


config = AppConfig()
config.ensure_directories()
