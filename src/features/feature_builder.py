"""Feature generation utilities for comovement detection and forecasting."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling z-score for anomaly-resistant scaling."""

    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    z = (series - rolling_mean) / rolling_std
    return z.fillna(0.0)


def build_pair_features(
    leader: pd.Series,
    follower: pd.Series,
    lead: int,
    leader_lags: int,
    follower_lags: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct supervised learning features for a leader-follower pair."""

    df = pd.DataFrame({"target": follower.shift(-1)})

    for lag in range(1, follower_lags + 1):
        df[f"follower_lag_{lag}"] = follower.shift(lag)

    for lag in range(lead, lead + leader_lags):
        df[f"leader_lag_{lag}"] = leader.shift(lag)

    df = df.dropna()
    y = df.pop("target")
    return df, y


def summarize_pairs(pairs: Iterable[Dict[str, object]]) -> pd.DataFrame:
    """Convert a list of dictionaries describing pairs into a dataframe."""

    return pd.DataFrame(list(pairs))
