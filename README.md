# 국민대학교 AI 빅데이터 분석 경진대회 솔루션 템플릿

이 저장소는 국민대학교 AI빅데이터/디지털마케팅전공과 경영대학이 주관하는 "제3회 AI빅데이터 분석 경진대회" 참가자를 위한 학습·추론 코드 베이스를 제공합니다. 대회 규칙(상대 경로 사용, 재현 가능한 환경, 추론 스크립트 분리 등)을 준수하도록 설계되었습니다.

## 주요 기능

- 100개 품목의 월별 무역 데이터를 로드하고 전처리합니다.
- 시계열 공행성(comovement)을 탐지하기 위해 선행-후행 품목 쌍을 자동으로 선택합니다.
- 선행 품목의 흐름을 활용한 후행 품목 다음 달 무역량 예측 모델(Gradient Boosting)을 학습합니다.
- 훈련된 모델과 하이퍼파라미터 설정을 저장하고, 동일한 설정으로 추론을 재현합니다.
- DACON 제출 형식(csv)을 출력하는 독립적인 `inference.py` 스크립트를 제공합니다.

## 디렉터리 구조

```
.
├── configs/                # YAML 설정 파일
├── data/
│   ├── raw/                # 대회에서 제공된 원시 데이터 (직접 추가)
│   └── processed/          # 전처리 및 제출 산출물 저장 위치
├── models/                 # 학습된 모델 가중치 및 메타데이터
├── notebooks/              # 탐색적 분석(선택 사항)
├── src/
│   ├── config.py           # 데이터/모델 설정 dataclass
│   ├── data/               # 로딩 및 전처리 모듈
│   ├── features/           # 특성 생성 유틸리티
│   ├── models/             # 공행성 탐지 및 예측 모델
│   ├── pipelines/          # 학습/추론 파이프라인
│   └── utils/              # 로깅 설정
├── train.py                # 학습 엔트리 포인트
├── inference.py            # 추론 엔트리 포인트
└── requirements.txt        # 재현 가능한 환경을 위한 라이브러리 버전
```

## 환경 설정

1. Python 3.10 이상을 권장합니다.
2. (선택) 가상환경을 생성하고 활성화합니다.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows의 경우 .venv\Scripts\activate
   ```
3. 의존성을 설치합니다.
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. 컨테이너에서 프로젝트 위치를 확인하려면 다음 명령 중 하나를 실행합니다.
   ```bash
   pwd
   ```
   또는
   ```bash
   python scripts/show_repo_path.py
   ```
   두 명령 모두 현재 저장소의 절대 경로를 출력합니다. 기본 제공 환경이라면 `/workspace/dataleage`가 표시됩니다.
4. 데이터 배치:
   - `data/raw/train.csv` : 2022.01 ~ 2025.07 기간의 월별 수입 데이터 (컬럼 예시: `item_id`, `date`, `value`).
   - `data/raw/sample_submission.csv` : 제출 양식(선택 사항).
   - 필요 시 `data/raw/` 디렉터리를 직접 생성한 후 CSV 파일을 복사합니다.

## 학습 실행

```bash
python train.py --config configs/default.yaml
```

- 학습이 완료되면 `models/` 디렉터리에 선후행 쌍(`pair_candidates.json`), 가중치(`model_*.joblib`), 검증 지표(`forecast_results.csv`), 학습 설정(`training_config.json`)이 생성됩니다.
- 로그는 `data/processed/training.log`에 저장됩니다.
- 여러 설정을 테스트하려면 YAML 파일을 복사한 뒤 `--config` 인자로 전달하면 됩니다.

## 추론 실행

```bash
python inference.py --config configs/default.yaml
```

- 추론 시 학습 단계에서 저장한 설정과 모델을 재사용하여 다음 달(예: 2025년 8월) 무역량을 예측합니다.
- 결과는 `data/processed/submission.csv`로 출력됩니다.
- 로그는 `data/processed/inference.log`에서 확인할 수 있습니다.
- 학습된 모델이 없다면 `python train.py ...`를 먼저 실행해야 합니다.

## 빠른 시작 예시

```bash
git clone https://github.com/<YOUR_ID>/dataleage.git
cd dataleage
# 원격 저장소에 `work` 브랜치가 기본으로 올라가 있다면 아래 명령으로 확인 후 체크아웃하세요.
git fetch --all
git switch work  # 또는 git checkout work
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data/raw
cp /path/to/train.csv data/raw/
cp /path/to/sample_submission.csv data/raw/  # 선택
python train.py --config configs/default.yaml
python inference.py --config configs/default.yaml
```

위 명령을 순서대로 실행하면 로컬 환경에서 학습과 추론을 재현할 수 있으며, 생성된 `data/processed/submission.csv` 파일을 제출용으로 활용할 수 있습니다.

## 코드가 보이지 않을 때 점검 사항

1. **브랜치 확인**: 현재 체크아웃된 브랜치가 `work`인지 `git status` 또는 `git branch`로 확인합니다. `git switch work` 명령으로 전환할 수 있습니다.
2. **최신 커밋 동기화**: 원격 저장소를 사용 중이라면 `git pull` 또는 `git fetch --all` 후 `git merge`/`git switch`로 최신 커밋을 받아옵니다.
3. **서브모듈 여부 확인**: 본 템플릿은 서브모듈을 사용하지 않지만, 만약 디렉터리가 비어 있다면 `git submodule status`로 체크합니다.
4. **다운로드 폴더 재확인**: GitHub 웹 UI에서 `work` 브랜치를 선택하거나, 로컬 클론 위치가 맞는지 확인합니다.

## 커스터마이징 가이드

- `configs/default.yaml`의 `training` 섹션을 수정하여 공행성 탐지 기준(상관계수, 최대 리드 기간 등)과 예측 모델 하이퍼파라미터를 변경할 수 있습니다.
- `inference` 섹션은 추론 출력 경로 및 입력 데이터를 제어합니다.
- 필요 시 `src/models/forecaster.py`를 수정하여 다른 회귀 모델(예: LightGBM, XGBoost 등)로 교체할 수 있으며, 대회 규칙상 외부 API 호출은 허용되지 않음을 유의하세요.

## 주의 사항

- 모든 입·출력 경로는 상대 경로로 유지되어 재현 가능한 환경을 보장합니다.
- 외부 데이터 사용은 대회 규칙에 따라 금지되어 있으며, 제공된 데이터 내에서만 모델을 학습해야 합니다.
- `inference.py`는 반드시 학습 후 실행해야 하며, 모델 가중치가 존재하지 않을 경우 예외가 발생합니다.

## 라이선스

이 템플릿은 MIT 라이선스로 배포됩니다. 단, 대회 규칙을 준수하여 사용하십시오.
