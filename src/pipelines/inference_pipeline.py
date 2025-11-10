"""Inference pipeline for generating competition submissions."""
from __future__ import annotations

import json
import logging
from typing import List

import pandas as pd

from src.config import InferenceConfig, ensure_directories, resolve_path
from src.data.loaders import load_submission_template, load_trade_data
from src.data.preprocessing import pivot_trade_series
from src.models.forecaster import PairForecaster
from src.utils.logging import configure_logging


def run_inference(config: InferenceConfig) -> pd.DataFrame:
    """Run inference and generate the submission file."""

    ensure_directories(config)
    log_path = resolve_path(config.data.processed_dir / "inference.log")
    configure_logging(log_path)
    logger = logging.getLogger(__name__)

    raw_data = load_trade_data(config.data.trade_data_path)
    series = pivot_trade_series(raw_data)
    prediction_date = series.index.max() + pd.offsets.MonthBegin(1)
    logger.info("Running inference for %s", prediction_date.date())

    pairs_path = resolve_path(config.model_registry_dir / "pair_candidates.json")
    if not pairs_path.exists():
        raise FileNotFoundError(
            "Pair candidates file not found. Please run training before inference."
        )
    with pairs_path.open("r", encoding="utf-8") as f:
        pair_records = json.load(f)

    training_config_path = resolve_path(config.model_registry_dir / "training_config.json")
    if training_config_path.exists():
        with training_config_path.open("r", encoding="utf-8") as f:
            training_config_dict = json.load(f)
        forecast_cfg = training_config_dict.get("forecast", {})
        leader_lags = int(forecast_cfg.get("leader_lags", 6))
        follower_lags = int(forecast_cfg.get("follower_lags", 6))
        random_state = int(forecast_cfg.get("random_state", 42))
        model_params = forecast_cfg.get("gradient_boosting_params", {})
    else:
        logger.warning("Training configuration not found. Falling back to default inference hyper-parameters.")
        leader_lags = follower_lags = 6
        random_state = 42
        model_params = {}

    forecaster = PairForecaster(
        leader_lags=leader_lags,
        follower_lags=follower_lags,
        model_dir=resolve_path(config.model_registry_dir),
        random_state=random_state,
        model_params=model_params,
    )

    predictions: List[dict] = []
    for record in pair_records:
        leader = record["leader"]
        follower = record["follower"]
        lead = int(record["lead"])
        try:
            model = forecaster.load_model(leader, follower)
            pred = forecaster.predict_next_month(
                leader_series=series[leader],
                follower_series=series[follower],
                lead=lead,
                model=model,
            )
            predictions.append({
                "leader": leader,
                "follower": follower,
                "prediction_date": prediction_date,
                "value": pred,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to score pair %s -> %s: %s", leader, follower, exc)

    prediction_df = pd.DataFrame(predictions)
    if prediction_df.empty:
        raise RuntimeError("No predictions were generated. Check the training artifacts.")

    submission_template = load_submission_template(config.data.submission_template_path)
    if submission_template is not None and {"leader", "follower"}.issubset(submission_template.columns):
        merged = submission_template.merge(
            prediction_df,
            on=["leader", "follower"],
            how="left",
        )
        merged["value"] = merged["value"].fillna(0.0)
        output_df = merged[["leader", "follower", "value"]]
    else:
        output_df = prediction_df[["leader", "follower", "value"]]

    output_path = resolve_path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info("Saved inference results to %s", output_path)

    return output_df
