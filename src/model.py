"""
Multi-Horizon Predictive Model & Walk-Forward Validation Engine:
Trains ensembled models (XGBoost, CatBoost, and PyTorch GPU Neural Networks)
to forecast asset returns and price levels across horizons n+0 to n+5.
Implements leak-free walk-forward time-series validation with strict embargos.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier

from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class PyTorchGPUMLP(nn.Module):
    """Deep PyTorch Neural Network leveraging CUDA for return & directional prediction."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.return_head = nn.Linear(prev_dim, 1)          # Predicts continuous log return
        self.dir_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()                                   # Predicts directional probability [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        ret_pred = self.return_head(features)
        dir_pred = self.dir_head(features)
        return ret_pred.squeeze(-1), dir_pred.squeeze(-1)


@dataclass
class HorizonEvaluation:
    """Evaluation metrics for a specific horizon."""
    horizon: int
    mse: float
    mae: float
    r2: float
    directional_accuracy: float
    auc_roc: float
    predictions: np.ndarray
    predicted_prices: np.ndarray
    actual_returns: np.ndarray
    actual_prices: np.ndarray


class MultiHorizonPredictor:
    """Ensemble forecaster for a specific prediction horizon."""

    def __init__(self, horizon: int, device: Optional[str] = None):
        self.horizon = horizon
        self.device = torch.device(
            "cuda" if (device or config.DEVICE) == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.scaler = StandardScaler()
        self.xgb_model: Optional[xgb.XGBRegressor] = None
        self.cat_model: Optional[CatBoostRegressor] = None
        self.nn_model: Optional[PyTorchGPUMLP] = None
        self.feature_names: List[str] = []

    def fit(self, X_train: pd.DataFrame, y_train_ret: pd.Series, y_train_dir: pd.Series) -> "MultiHorizonPredictor":
        """Train XGBoost, CatBoost, and PyTorch GPU models."""
        self.feature_names = list(X_train.columns)
        X_scaled = self.scaler.fit_transform(X_train.fillna(0.0))

        # 1. Train XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + self.horizon,
            verbosity=0,
            n_jobs=-1,
        )
        self.xgb_model.fit(X_train.fillna(0.0), y_train_ret)

        # 2. Train CatBoost
        self.cat_model = CatBoostRegressor(
            iterations=180,
            depth=4,
            learning_rate=0.03,
            random_seed=42 + self.horizon,
            verbose=0,
        )
        self.cat_model.fit(X_train.fillna(0.0), y_train_ret)

        # 3. Train PyTorch Deep MLP on GPU
        self.nn_model = PyTorchGPUMLP(input_dim=X_scaled.shape[1]).to(self.device)
        self._train_torch_model(X_scaled, y_train_ret.values, y_train_dir.values)

        return self

    def _train_torch_model(self, X: np.ndarray, y_ret: np.ndarray, y_dir: np.ndarray, epochs: int = 25) -> None:
        """Fit PyTorch neural net with AdamW and dual loss."""
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_ret = torch.tensor(y_ret, dtype=torch.float32)
        tensor_dir = torch.tensor(y_dir, dtype=torch.float32)

        dataset = TensorDataset(tensor_x, tensor_ret, tensor_dir)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.AdamW(self.nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        mse_crit = nn.MSELoss()
        bce_crit = nn.BCELoss()

        self.nn_model.train()
        for epoch in range(epochs):
            for bx, by_ret, by_dir in loader:
                bx = bx.to(self.device)
                by_ret = by_ret.to(self.device)
                by_dir = by_dir.to(self.device)

                optimizer.zero_grad()
                pred_ret, pred_dir = self.nn_model(bx)
                loss = mse_crit(pred_ret, by_ret) + 0.5 * bce_crit(pred_dir, by_dir)
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame, current_prices: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate ensembled return predictions, directional probs, and expected price levels."""
        X_clean = X.fillna(0.0)
        X_scaled = self.scaler.transform(X_clean)

        # 1. XGBoost predictions
        xgb_preds = self.xgb_model.predict(X_clean)

        # 2. CatBoost predictions
        cat_preds = self.cat_model.predict(X_clean)

        # 3. PyTorch GPU predictions
        self.nn_model.eval()
        with torch.no_grad():
            t_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            nn_ret, nn_dir = self.nn_model(t_x)
            torch_ret_preds = nn_ret.cpu().numpy()
            torch_dir_preds = nn_dir.cpu().numpy()

        # Weighted Ensemble: 40% XGBoost, 35% CatBoost, 25% PyTorch GPU
        ensembled_ret = 0.40 * xgb_preds + 0.35 * cat_preds + 0.25 * torch_ret_preds

        # Combined directional probability: sigmoid blend
        xgb_dir = 1.0 / (1.0 + np.exp(-15.0 * xgb_preds))
        ensembled_prob = 0.40 * xgb_dir + 0.35 * (1.0 / (1.0 + np.exp(-15.0 * cat_preds))) + 0.25 * torch_dir_preds

        # Target Price Level: P_{t+h} = P_t * exp(r_{t+h})
        predicted_prices = current_prices.values * np.exp(ensembled_ret)

        return ensembled_ret, ensembled_prob, predicted_prices

    def get_feature_importances(self) -> pd.Series:
        """Return relative feature importance from tree ensemble."""
        if self.xgb_model and hasattr(self.xgb_model, "feature_importances_"):
            return pd.Series(self.xgb_model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        return pd.Series()


class WalkForwardEngine:
    """Purged and embargoed walk-forward validation engine across horizons n+0 to n+5."""

    def __init__(
        self,
        n_splits: int = config.WALK_FORWARD_SPLITS,
        embargo_periods: int = config.EMBARGO_PERIODS,
        horizons: List[int] = config.HORIZONS,
    ):
        self.n_splits = n_splits
        self.embargo_periods = embargo_periods
        self.horizons = horizons

    def run_walk_forward(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict[int, HorizonEvaluation]:
        """Execute expanding-window walk-forward validation with strict out-of-sample tracking."""
        total_bars = len(df)
        logger.info(
            "Initiating Walk-Forward Validation (%d splits, %d embargo bars) on %d observations...",
            self.n_splits,
            self.embargo_periods,
            total_bars,
        )

        test_fold_size = int(total_bars * 0.12)
        min_train_size = total_bars - (self.n_splits * test_fold_size)

        if min_train_size < 500:
            test_fold_size = int(total_bars / (self.n_splits + 2))
            min_train_size = total_bars - (self.n_splits * test_fold_size)

        horizon_evaluations: Dict[int, HorizonEvaluation] = {}

        # Run multi-horizon validation
        for h in self.horizons:
            logger.info("--- Validating Horizon n+%d ---", h)
            all_oos_preds = []
            all_oos_probs = []
            all_oos_pred_prices = []
            all_oos_actual_ret = []
            all_oos_actual_prices = []

            target_ret_col = f"target_return_h{h}"
            target_price_col = f"target_price_h{h}"
            target_dir_col = f"target_dir_h{h}"

            for fold in range(self.n_splits):
                train_end = min_train_size + fold * test_fold_size
                # Embargo: purge the gap between train and test to prevent multi-step forward leakage
                purge_train_end = max(100, train_end - self.embargo_periods)
                test_start = train_end
                test_end = min(total_bars, test_start + test_fold_size)

                train_df = df.iloc[:purge_train_end]
                test_df = df.iloc[test_start:test_end]

                if len(test_df) == 0:
                    continue

                X_train = train_df[feature_cols]
                y_train_ret = train_df[target_ret_col]
                y_train_dir = train_df[target_dir_col]

                X_test = test_df[feature_cols]
                y_test_ret = test_df[target_ret_col]
                current_prices = test_df["close"]

                # Fit horizon predictor
                predictor = MultiHorizonPredictor(horizon=h)
                predictor.fit(X_train, y_train_ret, y_train_dir)

                # Out-of-sample inference
                pred_ret, pred_prob, pred_prices = predictor.predict(X_test, current_prices)

                all_oos_preds.extend(pred_ret)
                all_oos_probs.extend(pred_prob)
                all_oos_pred_prices.extend(pred_prices)
                all_oos_actual_ret.extend(y_test_ret.values)
                all_oos_actual_prices.extend(test_df[target_price_col].values)

            # Convert to arrays and evaluate
            oos_pred_ret = np.array(all_oos_preds)
            oos_actual_ret = np.array(all_oos_actual_ret)
            oos_pred_prices = np.array(all_oos_pred_prices)
            oos_actual_prices = np.array(all_oos_actual_prices)
            oos_probs = np.array(all_oos_probs)

            # Regression & Classification Metrics
            mse = float(mean_squared_error(oos_actual_ret, oos_pred_ret))
            mae = float(mean_absolute_error(oos_actual_ret, oos_pred_ret))
            r2 = float(r2_score(oos_actual_ret, oos_pred_ret))
            actual_dir = (oos_actual_ret > 0).astype(int)
            pred_dir = (oos_pred_ret > 0).astype(int)
            acc = float(accuracy_score(actual_dir, pred_dir))

            try:
                auc = float(roc_auc_score(actual_dir, oos_probs))
            except Exception:
                auc = 0.50

            logger.info(
                "Horizon n+%d Results: OOS Accuracy=%.2f%% | AUC=%.3f | MAE=%.5f | R2=%.4f",
                h, acc * 100.0, auc, mae, r2
            )

            horizon_evaluations[h] = HorizonEvaluation(
                horizon=h,
                mse=mse,
                mae=mae,
                r2=r2,
                directional_accuracy=acc,
                auc_roc=auc,
                predictions=oos_pred_ret,
                predicted_prices=oos_pred_prices,
                actual_returns=oos_actual_ret,
                actual_prices=oos_actual_prices,
            )

        return horizon_evaluations


def run_model_pipeline() -> Tuple[Dict[int, HorizonEvaluation], pd.DataFrame]:
    """Execute end-to-end feature extraction and walk-forward validation."""
    from src.features import FeatureEngineer

    engineer = FeatureEngineer()
    df = engineer.build_feature_dataset()
    features = engineer.get_feature_column_names(df)

    engine = WalkForwardEngine()
    results = engine.run_walk_forward(df, features)
    return results, df


if __name__ == "__main__":
    run_model_pipeline()
