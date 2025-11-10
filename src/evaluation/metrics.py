"""Competition-specific metrics."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


EPSILON = 1e-6


def f1_score(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    return 2 * precision * recall / (precision + recall + EPSILON)


def normalized_mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Compute the NMAE with FP/FN penalties as defined by the 대회 규칙."""

    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)

    if y_true_arr.size != y_pred_arr.size:
        raise ValueError("y_true and y_pred must have the same length")

    errors = np.empty_like(y_true_arr, dtype=float)

    valid_mask = ~np.isnan(y_true_arr) & ~np.isnan(y_pred_arr)
    denom = np.abs(y_true_arr[valid_mask]) + EPSILON
    errors[valid_mask] = np.minimum(1.0, np.abs(y_true_arr[valid_mask] - y_pred_arr[valid_mask]) / denom)

    errors[~valid_mask] = 1.0

    if errors.size == 0:
        return 1.0

    return float(errors.mean())


def competition_score(tp: int, fp: int, fn: int, y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    f1 = f1_score(tp, fp, fn)
    nmae = normalized_mae(y_true, y_pred)
    return 0.6 * f1 + 0.4 * (1 - nmae)


def evaluate_submission(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> Tuple[int, int, int, float]:
    merged = predictions.merge(
        ground_truth,
        on=["leader", "follower"],
        how="outer",
        suffixes=("_pred", "_true"),
    )

    tp_mask = merged["value_pred"].notna() & merged["value_true"].notna()
    fp_mask = merged["value_pred"].notna() & merged["value_true"].isna()
    fn_mask = merged["value_pred"].isna() & merged["value_true"].notna()

    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())

    nmae = normalized_mae(
        merged["value_true"].to_numpy(dtype=float),
        merged["value_pred"].to_numpy(dtype=float),
    )
    score = 0.6 * f1_score(tp, fp, fn) + 0.4 * (1 - nmae)
    return tp, fp, fn, score
