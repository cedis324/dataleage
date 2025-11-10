"""Command-line entry point for model training."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from src.config import TrainingConfig
from src.pipelines.train_pipeline import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models for the AI trade competition.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> TrainingConfig:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)
    training_section = raw_cfg.get("training", raw_cfg)
    return TrainingConfig(**training_section)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
