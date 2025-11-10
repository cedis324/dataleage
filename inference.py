"""Command-line entry point for inference."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from src.config import InferenceConfig
from src.pipelines.inference_pipeline import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the AI trade competition.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> InferenceConfig:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)
    inference_section = raw_cfg.get("inference", raw_cfg)
    return InferenceConfig(**inference_section)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_inference(config)


if __name__ == "__main__":
    main()
