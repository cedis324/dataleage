"""Data loading utilities for the AI trade forecasting competition."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import resolve_path


def load_trade_data(path: Path) -> pd.DataFrame:
    """Load the raw trade data CSV.

    Args:
        path: Relative path to the raw trade CSV.

    Returns:
        DataFrame with at least ["item_id", "date", "value"] columns.
    """

    absolute_path = resolve_path(path)
    if not absolute_path.exists():
        raise FileNotFoundError(
            f"Trade data file not found at {absolute_path}. Please place the raw data "
            "according to the README instructions."
        )

    data = pd.read_csv(absolute_path)
    if "date" not in data.columns:
        date_cols = [col for col in data.columns if col.lower() in {"yyyymm", "ym", "month"}]
        if not date_cols:
            raise ValueError(
                "The raw data must contain a 'date' column or a YYYYMM-style column such as 'yyyymm'."
            )
        primary = date_cols[0]
        data["date"] = pd.to_datetime(data[primary].astype(str))
    else:
        data["date"] = pd.to_datetime(data["date"])

    if "value" not in data.columns:
        raise ValueError("The raw data must contain a 'value' column.")

    if "item_id" not in data.columns:
        candidate_cols = [col for col in data.columns if col.lower() in {"item", "product_id", "hs_code"}]
        if not candidate_cols:
            raise ValueError("The raw data must contain an 'item_id' (or similar) column.")
        data = data.rename(columns={candidate_cols[0]: "item_id"})

    return data[["item_id", "date", "value"]].sort_values(["item_id", "date"]).reset_index(drop=True)


def load_submission_template(path: Path) -> Optional[pd.DataFrame]:
    """Load the submission template if present."""

    absolute_path = resolve_path(path)
    if not absolute_path.exists():
        return None
    return pd.read_csv(absolute_path)
