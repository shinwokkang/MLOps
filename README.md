# BAF 사기 탐지 MLOps 미니 프로젝트

[![tests](https://github.com/shinwokkang/MLOps/actions/workflows/tests.yml/badge.svg)](https://github.com/shinwokkang/MLOps/actions/workflows/tests.yml)

은행 계좌 개설 신청 데이터(Bank Account Fraud Dataset, NeurIPS 2022)로 사기 여부를 예측하는 모델을 만들고, 재현 가능한 파이프라인(전처리 → 학습 → 실험 기록)으로 관리하는 프로젝트입니다.

자세한 문제 정의, 평가 지표, 아키텍처는 [`docs/presentation.md`](docs/presentation.md)를 참고하세요. 발표 대본은 [`docs/script.md`](docs/script.md)에 있습니다.

## 프로젝트 구조

```
MLOps/
├── conf/config.yaml       # 경로·전처리 규칙·하이퍼파라미터·시간분할 설정
├── src/
│   ├── config.py           # conf/config.yaml 로더
│   ├── preprocess.py        # CSV -> 결측 처리 -> category 변환 -> 시간 분할 -> parquet
│   ├── train.py              # LightGBM 학습 + MLflow 기록
│   └── evaluate.py           # TPR@FPR5%, AUPRC 계산
├── tests/
│   ├── test_preprocess.py    # 전처리 함수 단위 테스트 (가짜 미니 데이터)
│   ├── test_data_quality.py  # 실제 전처리 결과물 값 범위 검증
│   └── test_eda_patterns.py  # EDA에서 발견한 패턴이 유지되는지 검증
├── notebooks/eda.ipynb     # 탐색적 데이터 분석
├── data/
│   ├── raw/                 # 원본 CSV (DVC 관리, git에는 포인터만)
│   └── processed/            # 전처리 결과 parquet (DVC 관리)
├── Dockerfile
├── docker-compose.yml       # train 컨테이너 + mlflow-server 컨테이너
└── requirements.txt
```

## 처음 시작하는 법

### 1. 환경 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LightGBM은 시스템에 OpenMP 런타임이 필요합니다.

```bash
# macOS
brew install libomp
```

### 2. 데이터 받기

원본 CSV는 git에 없고 DVC로 관리됩니다. 이 저장소를 처음 받았다면:

```bash
dvc pull
```

(DVC remote가 로컬 경로로 설정되어 있어, 저장소를 만든 사람의 컴퓨터가 아니면 이 단계는 실패합니다.
그 경우 [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)에서
`Base.csv`를 직접 받아 `data/raw/`에 넣고 아래 3번부터 진행하세요.)

### 3. 파이프라인 실행

```bash
python src/preprocess.py   # data/processed/{train,valid,test}.parquet 생성
python src/train.py         # 학습 + MLflow에 run 기록
```

### 4. 결과 확인

```bash
mlflow ui --port 5001 --backend-store-uri ./mlruns
# http://127.0.0.1:5001 접속
```

### 5. 테스트 실행

```bash
pytest tests/ -v
```

## Docker로 실행

로컬 환경 설정 없이 컨테이너 안에서 전체 파이프라인을 실행할 수 있습니다.

```bash
docker build -t baf-fraud-baseline .
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/mlruns:/app/mlruns" \
  baf-fraud-baseline
```

## Docker Compose로 실행 (학습 컨테이너 + MLflow 서버 분리)

```bash
docker compose up --build
# http://localhost:5001 에서 MLflow UI 확인
docker compose down   # 종료
```

## EDA 노트북

```bash
python -m ipykernel install --user --name mlops-venv --display-name "Python 3 (.venv)"
jupyter lab notebooks/eda.ipynb
```

## 평가 지표

- **TPR@FPR5%** (주지표): 정상 고객 오탐률을 5% 이내로 제한했을 때의 사기 탐지율
- **AUPRC** (보조지표)
- accuracy는 사용하지 않습니다 (사기 비율 1.1%라 무의미).

베이스라인(LightGBM 기본값, `random_state=42`) 결과: TPR@FPR5% = 0.5034, AUPRC = 0.1541

## 시간 분할

무작위 분할이 아니라 `month` 기준 시간 순서로 분할합니다 (데이터 누수 방지).

| 구간 | 기간 | 용도 |
|---|---|---|
| train | month 0~5 | 학습 |
| valid | month 6 | 검증 |
| test | month 7 | 미사용 (3주차 드리프트 실험용) |
