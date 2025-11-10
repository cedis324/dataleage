"""Training pipeline for the AI trade competition."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List

import pandas as pd

from src.config import TrainingConfig, ensure_directories, resolve_path
from src.data.loaders import load_trade_data
from src.data.preprocessing import pivot_trade_series
from src.models.forecaster import PairForecaster, PairForecastResult
from src.models.pair_selector import ComovementPairSelector
from src.utils.logging import configure_logging


def run_training(config: TrainingConfig) -> List[PairForecastResult]:
    """Execute the training flow and persist artifacts."""

    ensure_directories(config)
    log_path = resolve_path(config.data.processed_dir / "training.log")
    configure_logging(log_path)
    logger = logging.getLogger(__name__)

    logger.info("Loading trade data from %s", config.data.trade_data_path)
    raw_data = load_trade_data(config.data.trade_data_path)
    logger.info("Raw data shape: %s", raw_data.shape)

    series = pivot_trade_series(raw_data)
    logger.info("Pivoted series shape: %s", series.shape)

    pair_selector = ComovementPairSelector(
        min_correlation=config.pair_selection.min_correlation,
        max_lead_months=config.pair_selection.max_lead_months,
        rolling_window=config.pair_selection.rolling_window,
        top_k_pairs=config.pair_selection.top_k_pairs,
    )
    candidates = pair_selector.fit_transform(series)
    logger.info("Identified %d candidate pairs", len(candidates))

    config_path = resolve_path(config.model_registry_dir / "training_config.json")
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2, default=str)
    logger.info("Saved training configuration to %s", config_path)

    pairs_path = resolve_path(config.model_registry_dir / "pair_candidates.json")
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("w", encoding="utf-8") as f:
        json.dump(ComovementPairSelector.to_records(candidates), f, ensure_ascii=False, indent=2)

    results: List[PairForecastResult] = []
    if len(series.index) <= config.forecast.validation_months:
        raise ValueError("Not enough historical observations for the requested validation split.")

    validation_start = series.index[-config.forecast.validation_months]
    forecaster = PairForecaster(
        leader_lags=config.forecast.leader_lags,
        follower_lags=config.forecast.follower_lags,
        model_dir=resolve_path(config.model_registry_dir),
        random_state=config.forecast.random_state,
        model_params=config.forecast.gradient_boosting_params,
    )

    for candidate in candidates:
        leader_series = series[candidate.leader]
        follower_series = series[candidate.follower]
        try:
            model, val_mae = forecaster.fit_with_validation(
                leader_series,
                follower_series,
                lead=candidate.lead,
                validation_start=validation_start,
            )
        except ValueError as exc:
            logger.warning("Skipping pair %s -> %s: %s", candidate.leader, candidate.follower, exc)
            continue
        model_path = forecaster.save_model(model, candidate.leader, candidate.follower)
        results.append(
            PairForecastResult(
                leader=candidate.leader,
                follower=candidate.follower,
                lead=candidate.lead,
                model_path=model_path,
                validation_mae=val_mae,
            )
        )
        logger.info(
            "Trained model for %s -> %s (lead=%d) | validation MAE: %.4f",
            candidate.leader,
            candidate.follower,
            candidate.lead,
            val_mae,
        )

    summary_df = pd.DataFrame(
        [
            {
                "leader": result.leader,
                "follower": result.follower,
                "lead": result.lead,
                "model_path": str(result.model_path.relative_to(resolve_path(Path(".")))),
                "validation_mae": result.validation_mae,
            }
            for result in results
        ]
    )
    summary_path = resolve_path(config.model_registry_dir / "forecast_results.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved forecast summary to %s", summary_path)

    return results
