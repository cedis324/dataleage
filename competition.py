"""국민대학교 AI 빅데이터 경진대회용 단일 파일 파이프라인."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------------------------
# 설정 관련 dataclass
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """입출력 경로 설정."""

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
    """공행성 탐지를 위한 하이퍼파라미터."""

    min_correlation: float = 0.6
    max_lead_months: int = 3
    rolling_window: int = 6
    top_k_pairs: int = 200


@dataclass
class ForecastConfig:
    """후행 품목 예측 모델 설정."""

    follower_lags: int = 6
    leader_lags: int = 6
    validation_months: int = 4
    random_state: int = 42
    gradient_boosting_params: Dict[str, object] = field(
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
    data: DataConfig = field(default_factory=DataConfig)
    model_registry_dir: Path = Path("models")
    output_path: Path = Path("data/processed/submission.csv")

    def __post_init__(self) -> None:
        if not isinstance(self.data, DataConfig):
            self.data = DataConfig(**self.data)
        self.model_registry_dir = Path(self.model_registry_dir)
        self.output_path = Path(self.output_path)


# ---------------------------------------------------------------------------
# 공통 유틸리티
# ---------------------------------------------------------------------------


def resolve_path(path: Path) -> Path:
    """상대 경로를 현재 작업 디렉터리 기준 절대 경로로 변환."""

    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def ensure_directories(config: TrainingConfig | InferenceConfig) -> None:
    """필요한 출력 디렉터리를 생성."""

    dirs: List[Path] = []
    if isinstance(config, TrainingConfig):
        dirs.extend([config.model_registry_dir, config.data.processed_dir])
    else:
        dirs.extend([config.model_registry_dir, config.data.processed_dir])

    for directory in dirs:
        resolve_path(directory).mkdir(parents=True, exist_ok=True)


def configure_logging(log_path: Optional[Path] = None) -> None:
    """콘솔 및 선택적 파일 로그 설정."""

    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# 데이터 로딩 및 전처리
# ---------------------------------------------------------------------------


def load_trade_data(path: Path) -> pd.DataFrame:
    absolute_path = resolve_path(path)
    if not absolute_path.exists():
        raise FileNotFoundError(
            f"무역 데이터 파일을 찾을 수 없습니다: {absolute_path}. README 지침에 따라 데이터를 배치해 주세요."
        )

    data = pd.read_csv(absolute_path)
    if "date" not in data.columns:
        date_cols = [col for col in data.columns if col.lower() in {"yyyymm", "ym", "month"}]
        if not date_cols:
            raise ValueError("'date' 또는 YYYYMM 형식의 열이 필요합니다.")
        primary = date_cols[0]
        data["date"] = pd.to_datetime(data[primary].astype(str))
    else:
        data["date"] = pd.to_datetime(data["date"])

    if "value" not in data.columns:
        raise ValueError("무역 데이터에는 반드시 'value' 열이 포함되어야 합니다.")

    if "item_id" not in data.columns:
        candidate_cols = [col for col in data.columns if col.lower() in {"item", "product_id", "hs_code"}]
        if not candidate_cols:
            raise ValueError("무역 데이터에는 'item_id' 열이 필요합니다.")
        data = data.rename(columns={candidate_cols[0]: "item_id"})

    return data[["item_id", "date", "value"]].sort_values(["item_id", "date"]).reset_index(drop=True)


def load_submission_template(path: Path) -> Optional[pd.DataFrame]:
    absolute_path = resolve_path(path)
    if not absolute_path.exists():
        return None
    return pd.read_csv(absolute_path)


def pivot_trade_series(data: pd.DataFrame) -> pd.DataFrame:
    pivot = data.pivot_table(index="date", columns="item_id", values="value")
    pivot = pivot.sort_index()
    pivot = pivot.interpolate(method="linear", limit_direction="both")
    pivot = pivot.fillna(method="bfill").fillna(method="ffill")
    return pivot


# ---------------------------------------------------------------------------
# 특성 생성 및 모델 구성 요소
# ---------------------------------------------------------------------------


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
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
    df = pd.DataFrame({"target": follower.shift(-1)})

    for lag in range(1, follower_lags + 1):
        df[f"follower_lag_{lag}"] = follower.shift(lag)

    for lag in range(lead, lead + leader_lags):
        df[f"leader_lag_{lag}"] = leader.shift(lag)

    df = df.dropna()
    y = df.pop("target")
    return df, y


@dataclass
class PairCandidate:
    leader: str
    follower: str
    lead: int
    correlation: float


class ComovementPairSelector:
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
            raise ValueError("lead 값은 양수여야 합니다.")
        shifted_leader = leader.shift(lead)
        valid = follower.notna() & shifted_leader.notna()
        if valid.sum() < self.rolling_window:
            return 0.0
        return float(follower[valid].corr(shifted_leader[valid]))

    def fit_transform(self, series: pd.DataFrame) -> List[PairCandidate]:
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
                    candidates.append(PairCandidate(leader=leader, follower=follower, lead=best_lead, correlation=best_corr))

        candidates.sort(key=lambda c: abs(c.correlation), reverse=True)
        return candidates[: self.top_k_pairs]

    @staticmethod
    def to_records(candidates: Iterable[PairCandidate]) -> List[Dict[str, object]]:
        return [candidate.__dict__ for candidate in candidates]

    @staticmethod
    def from_records(records: Iterable[Dict[str, object]]) -> List[PairCandidate]:
        return [PairCandidate(**record) for record in records]


@dataclass
class PairForecastResult:
    leader: str
    follower: str
    lead: int
    model_path: Path
    validation_mae: float


class PairForecaster:
    def __init__(
        self,
        leader_lags: int,
        follower_lags: int,
        model_dir: Path,
        random_state: int,
        model_params: Dict[str, object],
    ) -> None:
        self.leader_lags = leader_lags
        self.follower_lags = follower_lags
        self.model_dir = model_dir
        self.random_state = random_state
        self.model_params = {**model_params, "random_state": random_state}

    def _model_filename(self, leader: str, follower: str) -> Path:
        safe_leader = leader.replace("/", "-")
        safe_follower = follower.replace("/", "-")
        return self.model_dir / f"model_{safe_leader}__{safe_follower}.joblib"

    def fit(
        self,
        leader_series: pd.Series,
        follower_series: pd.Series,
        lead: int,
    ) -> Tuple[GradientBoostingRegressor, pd.DataFrame, pd.Series]:
        X, y = build_pair_features(
            leader_series,
            follower_series,
            lead=lead,
            leader_lags=self.leader_lags,
            follower_lags=self.follower_lags,
        )
        if X.empty:
            raise ValueError("학습에 필요한 관측치가 부족합니다.")
        model = GradientBoostingRegressor()
        model.set_params(**self.model_params)
        model.fit(X, y)
        return model, X, y

    def fit_with_validation(
        self,
        leader_series: pd.Series,
        follower_series: pd.Series,
        lead: int,
        validation_start: pd.Timestamp,
    ) -> Tuple[GradientBoostingRegressor, float]:
        model, X, y = self.fit(leader_series, follower_series, lead)
        val_mask = X.index >= validation_start
        if val_mask.sum() == 0:
            return model, float("nan")
        y_true = y.loc[val_mask]
        y_pred = model.predict(X.loc[val_mask])
        mae = mean_absolute_error(y_true, y_pred)
        return model, float(mae)

    def save_model(self, model: GradientBoostingRegressor, leader: str, follower: str) -> Path:
        path = self._model_filename(leader, follower)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return path

    def load_model(self, leader: str, follower: str) -> GradientBoostingRegressor:
        path = self._model_filename(leader, follower)
        if not path.exists():
            raise FileNotFoundError(f"모델 가중치 파일이 없습니다: {path}")
        return joblib.load(path)

    def predict_next_month(
        self,
        leader_series: pd.Series,
        follower_series: pd.Series,
        lead: int,
        model: GradientBoostingRegressor,
    ) -> float:
        feature_row: Dict[str, float] = {}
        for lag in range(1, self.follower_lags + 1):
            value = follower_series.shift(lag).iloc[-1]
            if pd.isna(value):
                raise ValueError("추론에 필요한 후행 품목 시차 값이 부족합니다.")
            feature_row[f"follower_lag_{lag}"] = float(value)
        for lag in range(lead, lead + self.leader_lags):
            value = leader_series.shift(lag).iloc[-1]
            if pd.isna(value):
                raise ValueError("추론에 필요한 선행 품목 시차 값이 부족합니다.")
            feature_row[f"leader_lag_{lag}"] = float(value)
        X = pd.DataFrame([feature_row])
        prediction = model.predict(X)[0]
        return float(prediction)


# ---------------------------------------------------------------------------
# 대회 평가 지표
# ---------------------------------------------------------------------------


EPSILON = 1e-6


def f1_score(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    return 2 * precision * recall / (precision + recall + EPSILON)


def normalized_mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)

    if y_true_arr.size != y_pred_arr.size:
        raise ValueError("y_true와 y_pred의 길이가 일치해야 합니다.")

    errors = np.empty_like(y_true_arr, dtype=float)

    valid_mask = ~np.isnan(y_true_arr) & ~np.isnan(y_pred_arr)
    denom = np.abs(y_true_arr[valid_mask]) + EPSILON
    errors[valid_mask] = np.minimum(1.0, np.abs(y_true_arr[valid_mask] - y_pred_arr[valid_mask]) / denom)

    errors[~valid_mask] = 1.0

    if errors.size == 0:
        return 1.0

    return float(errors.mean())


def competition_score(
    tp: int,
    fp: int,
    fn: int,
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> float:
    f1 = f1_score(tp, fp, fn)
    nmae = normalized_mae(y_true, y_pred)
    return 0.6 * f1 + 0.4 * (1 - nmae)


# ---------------------------------------------------------------------------
# 학습 & 추론 파이프라인
# ---------------------------------------------------------------------------


def run_training(config: TrainingConfig) -> List[PairForecastResult]:
    ensure_directories(config)
    log_path = resolve_path(config.data.processed_dir / "training.log")
    configure_logging(log_path)
    logger = logging.getLogger("training")

    logger.info("무역 데이터를 로드합니다: %s", config.data.trade_data_path)
    raw_data = load_trade_data(config.data.trade_data_path)
    logger.info("원본 데이터 형태: %s", raw_data.shape)

    series = pivot_trade_series(raw_data)
    logger.info("피벗 테이블 형태: %s", series.shape)

    selector = ComovementPairSelector(
        min_correlation=config.pair_selection.min_correlation,
        max_lead_months=config.pair_selection.max_lead_months,
        rolling_window=config.pair_selection.rolling_window,
        top_k_pairs=config.pair_selection.top_k_pairs,
    )
    candidates = selector.fit_transform(series)
    logger.info("선정된 공행성 후보 수: %d", len(candidates))

    registry = resolve_path(config.model_registry_dir)
    registry.mkdir(parents=True, exist_ok=True)

    config_path = registry / "training_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2, default=str)
    logger.info("학습 설정을 저장했습니다: %s", config_path)

    pairs_path = registry / "pair_candidates.json"
    with pairs_path.open("w", encoding="utf-8") as f:
        json.dump(ComovementPairSelector.to_records(candidates), f, ensure_ascii=False, indent=2)

    if len(series.index) <= config.forecast.validation_months:
        raise ValueError("검증 구간이 학습 기간보다 길어 학습할 수 없습니다.")

    validation_start = series.index[-config.forecast.validation_months]
    forecaster = PairForecaster(
        leader_lags=config.forecast.leader_lags,
        follower_lags=config.forecast.follower_lags,
        model_dir=registry,
        random_state=config.forecast.random_state,
        model_params=config.forecast.gradient_boosting_params,
    )

    results: List[PairForecastResult] = []
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
            logger.warning("%s -> %s 쌍을 건너뜁니다: %s", candidate.leader, candidate.follower, exc)
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
            "모델 학습 완료: %s -> %s (lead=%d), 검증 MAE=%.4f",
            candidate.leader,
            candidate.follower,
            candidate.lead,
            val_mae,
        )

    summary_records = []
    project_root = resolve_path(Path("."))
    for result in results:
        try:
            relative_path = result.model_path.relative_to(project_root)
        except ValueError:
            relative_path = result.model_path
        summary_records.append(
            {
                "leader": result.leader,
                "follower": result.follower,
                "lead": result.lead,
                "model_path": str(relative_path),
                "validation_mae": result.validation_mae,
            }
        )

    summary_df = pd.DataFrame(summary_records)
    summary_path = registry / "forecast_results.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("모델 요약을 저장했습니다: %s", summary_path)

    return results


def run_inference(config: InferenceConfig) -> pd.DataFrame:
    ensure_directories(config)
    log_path = resolve_path(config.data.processed_dir / "inference.log")
    configure_logging(log_path)
    logger = logging.getLogger("inference")

    raw_data = load_trade_data(config.data.trade_data_path)
    series = pivot_trade_series(raw_data)
    prediction_date = series.index.max() + pd.offsets.MonthBegin(1)
    logger.info("%s 날짜 예측을 수행합니다.", prediction_date.date())

    registry = resolve_path(config.model_registry_dir)
    pairs_path = registry / "pair_candidates.json"
    if not pairs_path.exists():
        raise FileNotFoundError("pair_candidates.json이 없습니다. 먼저 학습을 수행해 주세요.")
    with pairs_path.open("r", encoding="utf-8") as f:
        pair_records = json.load(f)

    training_config_path = registry / "training_config.json"
    if training_config_path.exists():
        with training_config_path.open("r", encoding="utf-8") as f:
            training_config_dict = json.load(f)
        forecast_cfg = training_config_dict.get("forecast", {})
        leader_lags = int(forecast_cfg.get("leader_lags", 6))
        follower_lags = int(forecast_cfg.get("follower_lags", 6))
        random_state = int(forecast_cfg.get("random_state", 42))
        model_params = forecast_cfg.get("gradient_boosting_params", {})
    else:
        logger.warning("training_config.json을 찾을 수 없어 기본 추론 설정을 사용합니다.")
        leader_lags = follower_lags = 6
        random_state = 42
        model_params = {}

    forecaster = PairForecaster(
        leader_lags=leader_lags,
        follower_lags=follower_lags,
        model_dir=registry,
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
            predictions.append(
                {
                    "leader": leader,
                    "follower": follower,
                    "prediction_date": prediction_date,
                    "value": pred,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s -> %s 예측에 실패했습니다: %s", leader, follower, exc)

    prediction_df = pd.DataFrame(predictions)
    if prediction_df.empty:
        raise RuntimeError("생성된 예측이 없습니다. 학습 산출물을 확인하세요.")

    submission_template = load_submission_template(config.data.submission_template_path)
    if submission_template is not None and {"leader", "follower"}.issubset(submission_template.columns):
        merged = submission_template.merge(prediction_df, on=["leader", "follower"], how="left")
        merged["value"] = merged["value"].fillna(0.0)
        output_df = merged[["leader", "follower", "value"]]
    else:
        output_df = prediction_df[["leader", "follower", "value"]]

    output_path = resolve_path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info("추론 결과를 저장했습니다: %s", output_path)

    return output_df


# ---------------------------------------------------------------------------
# 설정 로딩 및 CLI
# ---------------------------------------------------------------------------


def load_training_config(path: Optional[Path]) -> TrainingConfig:
    if path is None:
        return TrainingConfig()
    actual_path = Path(path)
    if not actual_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {actual_path}")
    with actual_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    training_section = raw_cfg.get("training", raw_cfg)
    return TrainingConfig(**training_section)


def load_inference_config(path: Optional[Path]) -> InferenceConfig:
    if path is None:
        return InferenceConfig()
    actual_path = Path(path)
    if not actual_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {actual_path}")
    with actual_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    inference_section = raw_cfg.get("inference", raw_cfg)
    return InferenceConfig(**inference_section)


def print_repo_path() -> None:
    print(resolve_path(Path(".")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="국민대 AI 빅데이터 경진대회 단일 파일 파이프라인")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="모델 학습")
    train_parser.add_argument("--config", type=Path, default=None, help="학습 설정 YAML 경로")

    infer_parser = subparsers.add_parser("infer", help="추론 실행")
    infer_parser.add_argument("--config", type=Path, default=None, help="추론 설정 YAML 경로")

    subparsers.add_parser("show-path", help="저장소 절대 경로 출력")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        config = load_training_config(args.config)
        run_training(config)
    elif args.command == "infer":
        config = load_inference_config(args.config)
        run_inference(config)
    elif args.command == "show-path":
        print_repo_path()
    else:  # pragma: no cover - argparse가 보장함
        raise ValueError(f"알 수 없는 명령입니다: {args.command}")


if __name__ == "__main__":
    main()
