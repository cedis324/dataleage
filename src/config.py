"""Configuration dataclasses and utilities for the AI trade forecasting project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    """Paths to the raw trade data files."""

    trade_data_path: Path = Path("data/raw/train.csv")
    metadata_path: Optional[Path] = None
    submission_template_path: Path = Path("data/raw/sample_submission.csv")
    processed_dir: Path = Path("data/processed")

    def __post_init__(self) -> None:
        self.trade_data_path = Path(self.trade_data_path)
        if self.metadata_path is not None:
            self.metadata_path = Path(self.metadata_path)
        self.submission_template_path = Path(self.submission_template_path)
        self.processed_dir = Path(self.processed_dir)


@dataclass
class PairSelectionConfig:
    """Hyper-parameters governing the comovement pair selection process."""

    min_correlation: float = 0.6
    max_lead_months: int = 3
    rolling_window: int = 6
    top_k_pairs: int = 200


@dataclass
class ForecastConfig:
    """Hyper-parameters for the follower forecasting models."""

    follower_lags: int = 6
    leader_lags: int = 6
    validation_months: int = 4
    random_state: int = 42
    gradient_boosting_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "loss": "huber",
        }
    )


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    pair_selection: PairSelectionConfig = field(default_factory=PairSelectionConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    model_registry_dir: Path = Path("models")

    def __post_init__(self) -> None:
        if not isinstance(self.data, DataConfig):
            self.data = DataConfig(**self.data)
        if not isinstance(self.pair_selection, PairSelectionConfig):
            self.pair_selection = PairSelectionConfig(**self.pair_selection)
        if not isinstance(self.forecast, ForecastConfig):
            self.forecast = ForecastConfig(**self.forecast)
        self.model_registry_dir = Path(self.model_registry_dir)


@dataclass
class InferenceConfig:
    """Configuration for the inference pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    model_registry_dir: Path = Path("models")
    output_path: Path = Path("data/processed/submission.csv")

    def __post_init__(self) -> None:
        if not isinstance(self.data, DataConfig):
            self.data = DataConfig(**self.data)
        self.model_registry_dir = Path(self.model_registry_dir)
        self.output_path = Path(self.output_path)


def resolve_path(path: Path) -> Path:
    """Resolve a path relative to the project root."""

    return (Path.cwd() / path).resolve()


def ensure_directories(config: TrainingConfig | InferenceConfig) -> None:
    """Ensure that all directory paths referenced in a configuration exist."""

    dirs: List[Path] = []
    if isinstance(config, TrainingConfig):
        dirs.extend(
            [
                config.model_registry_dir,
                config.data.processed_dir,
            ]
        )
    else:
        dirs.extend([config.model_registry_dir, config.data.processed_dir])

    for directory in dirs:
        resolve_path(directory).mkdir(parents=True, exist_ok=True)
