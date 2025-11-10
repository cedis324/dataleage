"""Pair selection logic using cross-correlation and rolling statistics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.features.feature_builder import rolling_zscore


@dataclass
class PairCandidate:
    leader: str
    follower: str
    lead: int
    correlation: float


class ComovementPairSelector:
    """Identify leader-follower pairs exhibiting comovement."""

    def __init__(
        self,
        min_correlation: float,
        max_lead_months: int,
        rolling_window: int,
        top_k_pairs: int,
    ) -> None:
        self.min_correlation = min_correlation
        self.max_lead_months = max_lead_months
        self.rolling_window = rolling_window
        self.top_k_pairs = top_k_pairs

    def _prepare_series(self, series: pd.Series) -> pd.Series:
        scaled = rolling_zscore(series, window=self.rolling_window)
        return scaled.fillna(0.0)

    def _compute_lead_correlation(self, leader: pd.Series, follower: pd.Series, lead: int) -> float:
        if lead <= 0:
            raise ValueError("lead must be positive")
        shifted_leader = leader.shift(lead)
        valid = follower.notna() & shifted_leader.notna()
        if valid.sum() < self.rolling_window:
            return 0.0
        return float(follower[valid].corr(shifted_leader[valid]))

    def fit_transform(self, series: pd.DataFrame) -> List[PairCandidate]:
        """Find the top-K comovement pairs."""

        items = series.columns.tolist()
        prepared = {item: self._prepare_series(series[item]) for item in items}
        candidates: List[PairCandidate] = []

        for leader in items:
            for follower in items:
                if leader == follower:
                    continue
                best_corr = 0.0
                best_lead = 0
                for lead in range(1, self.max_lead_months + 1):
                    corr = self._compute_lead_correlation(prepared[leader], prepared[follower], lead)
                    if np.isnan(corr):
                        continue
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lead = lead
                if abs(best_corr) >= self.min_correlation and best_lead > 0:
                    candidates.append(
                        PairCandidate(leader=leader, follower=follower, lead=best_lead, correlation=best_corr)
                    )

        candidates.sort(key=lambda c: abs(c.correlation), reverse=True)
        return candidates[: self.top_k_pairs]

    @staticmethod
    def to_records(candidates: Iterable[PairCandidate]) -> List[Dict[str, object]]:
        """Serialize the candidates to dictionaries for persistence."""

        return [candidate.__dict__ for candidate in candidates]

    @staticmethod
    def from_records(records: Iterable[Dict[str, object]]) -> List[PairCandidate]:
        """Deserialize pair candidates from dictionaries."""

        return [PairCandidate(**record) for record in records]
