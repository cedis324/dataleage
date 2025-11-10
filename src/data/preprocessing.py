"""Pre-processing helpers for the trade dataset."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def pivot_trade_series(data: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format trade data into an item-by-date matrix."""

    pivot = data.pivot_table(index="date", columns="item_id", values="value")
    pivot = pivot.sort_index()
    pivot = pivot.interpolate(method="linear", limit_direction="both")
    pivot = pivot.fillna(method="bfill").fillna(method="ffill")
    return pivot


def compute_growth_rates(series: pd.DataFrame) -> pd.DataFrame:
    """Compute month-over-month growth rates for each item."""

    growth = series.pct_change().replace([np.inf, -np.inf], np.nan)
    return growth.fillna(0.0)


def train_validation_split(series: pd.DataFrame, validation_months: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the series data into train and validation segments."""

    if validation_months <= 0:
        raise ValueError("validation_months must be positive")
    if validation_months >= len(series):
        raise ValueError("validation_months must be smaller than the number of time steps")

    train = series.iloc[:-validation_months].copy()
    valid = series.iloc[-validation_months:].copy()
    return train, valid
