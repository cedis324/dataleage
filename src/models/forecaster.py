"""Forecasting models for follower items."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.features.feature_builder import build_pair_features


@dataclass
class PairForecastResult:
    leader: str
    follower: str
    lead: int
    model_path: Path
    validation_mae: float


class PairForecaster:
    """Train and persist follower forecasting models."""

    def __init__(
        self,
        leader_lags: int,
        follower_lags: int,
        model_dir: Path,
        random_state: int,
        model_params: Dict[str, object],
    ) -> None:
        self.leader_lags = leader_lags
        self.follower_lags = follower_lags
        self.model_dir = model_dir
        self.random_state = random_state
        self.model_params = {**model_params, "random_state": random_state}

    def _model_filename(self, leader: str, follower: str) -> Path:
        safe_leader = leader.replace("/", "-")
        safe_follower = follower.replace("/", "-")
        return self.model_dir / f"model_{safe_leader}__{safe_follower}.joblib"

    def fit(
        self,
        leader_series: pd.Series,
        follower_series: pd.Series,
        lead: int,
    ) -> Tuple[GradientBoostingRegressor, pd.DataFrame, pd.Series]:
        X, y = build_pair_features(
            leader_series,
            follower_series,
            lead=lead,
            leader_lags=self.leader_lags,
            follower_lags=self.follower_lags,
        )
        if X.empty:
            raise ValueError("Not enough observations to train the model for this pair.")
        model = GradientBoostingRegressor()
        model.set_params(**self.model_params)
        model.fit(X, y)
        return model, X, y

    def fit_with_validation(
        self,
        leader_series: pd.Series,
        follower_series: pd.Series,
        lead: int,
        validation_start: pd.Timestamp,
    ) -> Tuple[GradientBoostingRegressor, float]:
        model, X, y = self.fit(leader_series, follower_series, lead)
        val_mask = X.index >= validation_start
        if val_mask.sum() == 0:
            return model, float("nan")
        y_true = y.loc[val_mask]
        y_pred = model.predict(X.loc[val_mask])
        mae = mean_absolute_error(y_true, y_pred)
        return model, float(mae)

    def save_model(self, model: GradientBoostingRegressor, leader: str, follower: str) -> Path:
        path = self._model_filename(leader, follower)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return path

    def load_model(self, leader: str, follower: str) -> GradientBoostingRegressor:
        path = self._model_filename(leader, follower)
        if not path.exists():
            raise FileNotFoundError(f"Model weight file not found: {path}")
        return joblib.load(path)

    def predict_next_month(
        self,
        leader_series: pd.Series,
        follower_series: pd.Series,
        lead: int,
        model: GradientBoostingRegressor,
    ) -> float:
        feature_row: Dict[str, float] = {}
        for lag in range(1, self.follower_lags + 1):
            value = follower_series.shift(lag).iloc[-1]
            if pd.isna(value):
                raise ValueError("Missing follower lag values for prediction.")
            feature_row[f"follower_lag_{lag}"] = float(value)
        for lag in range(lead, lead + self.leader_lags):
            value = leader_series.shift(lag).iloc[-1]
            if pd.isna(value):
                raise ValueError("Missing leader lag values for prediction.")
            feature_row[f"leader_lag_{lag}"] = float(value)
        X = pd.DataFrame([feature_row])
        prediction = model.predict(X)[0]
        return float(prediction)
