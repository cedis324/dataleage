# 국민대학교 AI 빅데이터 분석 경진대회 단일 파일 템플릿

이 저장소는 "제3회 국민대학교 AI 빅데이터 분석 경진대회" 참가자를 위해 **단일 파일**(`competition.py`)만으로 학습과 추론을 모두 수행할 수 있도록 구성되었습니다. 대회 규칙(상대 경로 사용, 재현 가능한 환경, 추론 스크립트 분리 등)을 그대로 반영하면서도, 많은 모듈을 관리해야 하는 번거로움을 줄이는 것이 목표입니다.

## 주요 기능

- 100개 품목의 월별 무역 데이터를 로드하고 전처리합니다.
- 시계열 공행성(comovement)을 탐지하여 선행-후행 품목 쌍을 자동으로 선택합니다.
- Gradient Boosting 회귀 모델을 활용해 후행 품목의 다음 달 무역량을 예측하고, 관련 메타데이터를 저장합니다.
- 학습과 추론 모두 `competition.py` 하나로 실행되며, 필요 시 YAML 설정 파일을 통해 하이퍼파라미터를 조정할 수 있습니다.
- `show-path` 서브커맨드를 제공해 컨테이너/로컬에서 저장소의 절대 경로를 즉시 확인할 수 있습니다.

## 디렉터리 구성

```
.
├── competition.py        # 학습/추론/경로 출력이 모두 포함된 단일 파일
├── data/                 # (사용자 생성) 원시 데이터 및 산출물 저장소
├── models/               # (자동 생성) 학습된 모델과 메타데이터
├── README.md             # 현재 문서
├── requirements.txt      # 의존성 목록
└── LICENSE               # MIT 라이선스
```

> `data/`와 `models/` 디렉터리는 학습 및 추론 시 자동으로 생성됩니다. 대회 제공 CSV는 `data/raw/` 아래에 직접 복사해야 합니다.

## 환경 설정

1. Python 3.10 이상을 권장합니다.
2. (선택) 가상환경을 생성합니다.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. 의존성을 설치합니다.
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. 저장소 절대 경로가 필요하다면 다음 명령으로 확인합니다.
   ```bash
   python competition.py show-path
   ```
   기본 제공 컨테이너라면 `/workspace/dataleage`가 출력됩니다.
5. 데이터 배치
   - `data/raw/train.csv` : 2022.01 ~ 2025.07 기간의 월별 수입 데이터 (컬럼 예시: `item_id`, `date`, `value`).
   - `data/raw/sample_submission.csv` : 제출 양식(선택 사항).
   - 필요 시 `data/raw/` 디렉터리를 직접 생성한 뒤 CSV 파일을 복사합니다.

## 학습 실행

기본 설정으로 학습하려면 다음 명령을 실행합니다.

```bash
python competition.py train
```

- 학습 산출물은 `models/` 디렉터리에 저장됩니다.
  - `pair_candidates.json` : 탐지된 선행-후행 쌍 목록
  - `model_*.joblib` : 각 쌍에 대한 Gradient Boosting 모델 가중치
  - `forecast_results.csv` : 검증 MAE 요약
  - `training_config.json` : 사용된 설정 값
- 로그는 `data/processed/training.log`에 기록됩니다.

## 추론 실행

학습이 끝난 후 추론을 수행하려면 다음을 실행합니다.

```bash
python competition.py infer
```

- 추론 결과는 `data/processed/submission.csv`에 저장됩니다.
- 로그는 `data/processed/inference.log`에서 확인할 수 있습니다.
- 학습 산출물이 존재하지 않으면 추론이 실패합니다.

## 설정 커스터마이징

`competition.py`는 YAML 파일을 통해 하이퍼파라미터를 오버라이드할 수 있습니다. 예시:

```yaml
# custom_config.yaml
training:
  data:
    trade_data_path: data/raw/train.csv
  pair_selection:
    min_correlation: 0.7
    top_k_pairs: 150
  forecast:
    follower_lags: 8
    leader_lags: 8
    validation_months: 6
inference:
  data:
    trade_data_path: data/raw/train.csv
  output_path: data/processed/my_submission.csv
```

사용법은 다음과 같습니다.

```bash
python competition.py train --config custom_config.yaml
python competition.py infer --config custom_config.yaml
```

설정을 지정하지 않으면 스크립트에 내장된 기본값이 사용됩니다.

## 유의 사항

- 모든 경로는 상대 경로로 유지해 재현 가능한 환경을 보장합니다.
- 외부 데이터 사용은 대회 규칙에 따라 금지되어 있으며, 제공된 데이터 내에서만 모델을 학습해야 합니다.
- 사전 학습 모델을 사용할 경우 2025-11-09 이전에 공개된 가중치이며 상업적 이용이 허용된 라이선스임을 확인해야 합니다.
- API 형태의 외부 모델 호출(OpenAI API 등)은 허용되지 않습니다.
- 추론 전에 반드시 학습을 완료하고 `models/` 디렉터리가 생성되었는지 확인하세요.

## 라이선스

이 템플릿은 MIT 라이선스로 배포됩니다. 대회 규칙을 준수하여 자유롭게 수정 및 활용하실 수 있습니다.
