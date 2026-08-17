# BAF 사기 탐지 프로젝트 — 2차 발표 (EDA 심화 + 재현성 인프라)

1차 발표(`docs/presentation.md`)는 필수 구현(전처리/학습/평가, Docker, MLflow)까지 다뤘습니다. 이번 발표는 그 이후 진행한 **EDA 심화 분석**과 **재현성·검증 인프라(DVC, README, CI)**를 다룹니다.

---

## 1. EDA 심화 분석

기본 EDA(클래스 불균형, 피처 분포, 월별 추이, 결측-사기 관계, 범주형 사기율)에 이어, 네 가지 관점을 추가로 분석했습니다.

### 1-1. 어떤 피처가 사기와 가장 관련 깊은가 (상관관계)

수치형 피처 전체를 `fraud_bool`과의 상관계수로 정렬한 결과, 가장 강한 피처(`credit_risk_score`)조차 **0.068**로 전반적으로 매우 약했습니다.

| 피처 | 상관계수 |
|---|---|
| credit_risk_score | 0.068 |
| proposed_credit_limit | 0.067 |
| customer_age | 0.063 |
| keep_alive_session | -0.049 |

**해석**: 상관계수는 "직선적인 관계"만 잡아낸다. LightGBM 같은 나무 기반 모델은 피처 여러 개의 **조합**으로 판단하기 때문에, 개별 피처의 약한 상관관계가 곧 "쓸모없는 피처"를 의미하지 않는다.

### 1-2. 나이대별 사기율 — 가장 뚜렷한 단일 패턴

`customer_age`를 10살 단위로 구간화(binning)해서 본 결과:

| 나이대 | 사기율 | 표본 수 |
|---|---|---|
| 10대 | 0.34% | 18,057 |
| 20대 | 0.47% | 224,832 |
| 40대 | 1.16% | 210,852 |
| 60대 | 3.15% | 32,269 |
| **70대+** | **4.01%** | 7,376 |

**나이가 많을수록 사기율이 거의 일직선으로 증가하며, 10대와 70대+는 약 12배 차이가 난다.** 상관계수(0.063)로는 이 강도가 잘 드러나지 않았는데, 관계가 직선이 아니라 완만한 곡선 형태이기 때문 — 상관계수의 한계를 보여주는 사례이기도 하다.

### 1-3. 피처 조합(교차) — 나이대 × 기기(OS)

나이대와 기기 종류를 동시에 교차한 히트맵:

| 나이대 | linux | macintosh | windows |
|---|---|---|---|
| 10대 | 0.17% | 0.14% | 0.74% |
| 40대 | 0.54% | 1.73% | 2.45% |
| 70대+ | 2.01% | **6.94%** | **6.71%** |

**개별 피처를 따로 볼 때보다 훨씬 강한 신호가 조합에서 나온다.** 10대+linux(0.17%)와 70대+macintosh(6.94%)는 약 **40배** 차이 — 나이와 기기가 곱하듯 상호작용하는 조합 효과(interaction)를 확인했다.

### 1-4. 행동 패턴 — 신청 몰림(velocity)과 사기의 반직관적 관계

`velocity_6h/24h/4w`(각각 6시간/24시간/4주 동안의 신청 몰림 정도)를 5분위 구간으로 나눠 비교:

| 구간 | velocity_6h | velocity_24h | velocity_4w |
|---|---|---|---|
| 1 (몰림 낮음) | **1.33%** | **1.17%** | **1.33%** |
| 5 (몰림 높음) | 0.85% | 0.86% | 1.01% |

**"신청이 몰릴수록 사기일 것"이라는 상식적 가정과 정반대로, 세 시간 창 모두에서 몰림이 적을 때 사기율이 더 높았다.** 하나의 피처에서만 나온 우연이 아니라 세 가지 시간 규모에서 일관되게 나타나 신뢰도가 높은 패턴이다. 검증 없이 가정만으로 피처를 설계했다면 잘못된 방향으로 갈 뻔한 사례.

### EDA 검증을 pytest로 고정

위 발견들이 데이터가 바뀌어도 유지되는지 자동으로 확인하는 테스트 7개를 추가했다 (`tests/test_eda_patterns.py`): 결측-사기율 관계, 나이-사기율 증가 추세, velocity 저/고 구간 사기율 비교. 이 테스트들은 향후 드리프트 감지(7주차)의 기초 아이디어이기도 하다 — "알고 있던 패턴이 깨지는 순간이 곧 데이터가 달라졌다는 신호".

---

## 2. 재현성·검증 인프라

### 2-1. DVC (데이터 버전 관리)

- `dvc init` 후 로컬 폴더(`~/dvc-storage/mlops-baf`)를 remote로 등록
- `data/raw/Base.csv`(204MB), `data/processed/`(48MB, 3개 parquet)를 DVC 추적 대상으로 등록
- git에는 실제 데이터 대신 **143바이트짜리 포인터 파일**(`.dvc`)만 커밋, 실 데이터는 `dvc push`로 remote에 별도 저장
- 미사용 Variant I~V CSV(각 200MB)는 실수로 git에 잡힐 뻔한 걸 발견해 `.gitignore`로 제외

### 2-2. GitHub 저장소 + README

- `git init` → `github.com/shinwokkang/MLOps`에 최초 push
- `README.md`에 환경 설정, 데이터 준비(`dvc pull`), 파이프라인 실행, Docker/Compose 실행, 테스트 실행 방법을 정리

### 2-3. GitHub Actions CI

- `.github/workflows/tests.yml`: push/PR마다 자동으로 pytest 21개 실행
- **첫 시도는 실패했다** — CI 서버 파이썬을 3.11로 설정했는데 `requirements.txt`는 로컬(3.9)에서 얼려둔 목록이라 일부 패키지 설치가 깨짐
- CI 서버 파이썬을 3.9로 맞춰서 재실행 → 성공. 이 실패-수정 과정 자체가 "로컬에서만 되던 게 다른 환경에선 깨질 수 있다"는 걸 실제로 보여준 사례
- data/processed 등 DVC로만 관리되는 데이터는 CI 서버에 없으므로, 관련 테스트는 `skipif`로 자동 건너뜀 (설계상 의도된 동작)

---

## 3. FastAPI 서빙 (6주차 착수)

### 3-1. 왜 필요한가

지금까지는 `model.pkl`이 디스크에 저장만 되어 있고, 실제로 쓰려면 파이썬 스크립트를 직접 실행해야 했다. 서빙은 이 모델을 **항상 켜져 있는 서버**에 올려두고, 외부에서 HTTP 요청으로 "이 신청 사기인가요?"라고 물으면 즉시 확률로 답해주는 구조를 만드는 작업이다.

### 3-2. 구현 내용

- `train.py`가 학습이 끝나면 MLflow run 폴더뿐 아니라 **고정 경로(`models/model.pkl`)에도 모델을 추가로 저장**하도록 수정 — MLflow run 경로는 실행마다 이름이 바뀌어 서빙 서버가 참조하기 불편하기 때문
- `src/serve.py`: FastAPI로 `/health`, `/predict` 두 엔드포인트 구현
  - `pydantic.BaseModel`로 요청 스키마(피처 29개) 정의, 타입이 안 맞으면 자동으로 에러 응답
  - 서버 시작 시 모델을 한 번만 메모리에 로드 (요청마다 파일을 다시 읽지 않음)
  - 요청 1건을 pandas 1행 표로 변환 후, 학습 때와 동일하게 범주형 컬럼을 `category` dtype으로 맞춰서 예측

### 3-3. 실행 중 발견한 문제와 해결

1건짜리 요청에 결측치(`null`)가 섞여 있으면, pandas가 그 컬럼의 타입을 숫자가 아니라 애매한 타입(`object`)으로 잘못 추론해서 LightGBM이 에러를 냈다(`pandas dtypes must be int, float or bool`). `pd.to_numeric(..., errors="coerce")`로 범주형이 아닌 모든 컬럼을 명시적으로 숫자로 변환해서 해결.

### 3-4. 실제 요청으로 검증

| 케이스 | 실제 정답 | 모델 예측 확률 |
|---|---|---|
| 정상 건 | fraud_bool=0 | 0.0023 (0.23%) |
| 사기 건 | fraud_bool=1 | 0.0344 (3.44%) |

사기 건이 정상 건보다 약 15배 높은 확률로 나와 방향은 일치했지만, 절대값 자체는 "사기"로 확정 짓기엔 낮다. 이는 버그가 아니라 **베이스라인 모델의 실제 성능 한계(TPR@FPR5%=0.50, 사기의 절반은 놓침)가 개별 예측에서도 그대로 드러난 것** — 모델을 개선하지 않는 한 서빙을 아무리 잘 만들어도 이 한계는 그대로 남는다.

### 3-5. Docker로 서빙 컨테이너화

`Dockerfile.serve`를 별도로 작성했다. 학습용 `Dockerfile`과 앞부분(베이스 이미지·시스템 라이브러리·패키지 설치)은 동일하지만, 마지막에 "전처리+학습 실행 후 종료" 대신 **계속 떠서 요청을 기다리는 uvicorn 서버**를 실행한다는 점이 다르다. `models/model.pkl`은 이미지에 넣지 않고, `data/`를 볼륨으로 연결했던 것과 같은 이유로 실행 시점에 볼륨(`-v ./models:/app/models`)으로 연결했다.

**1차 빌드는 실패했다.** `typer`(DVC의 의존 패키지)가 파이썬 3.10 이상에서 더 높은 버전의 `click`을 요구하는데, `requirements.txt`는 로컬(파이썬 3.9)에서 얼려둔 목록이라 낮은 버전의 `click`이 고정되어 있어 충돌(`ResolutionImpossible`)이 났다. GitHub Actions CI에서 겪었던 것과 **완전히 같은 유형의 문제** — 파이썬 버전이 안 맞으면 requirements.txt 안의 패키지들이 다르게 반응한다는 걸 세 번째로 확인한 사례. 학습용 `Dockerfile`도 같은 위험을 안고 있었다는 걸 이번에 같이 발견해서, 두 Dockerfile의 베이스 이미지를 `python:3.11-slim` → `python:3.9-slim`으로 통일해서 해결했다.

**검증 결과**: 컨테이너 안에서 실행한 `/predict` 응답이 로컬에서 직접 실행했을 때와 **소수점까지 완전히 동일**(0.0023413776876002874 / 0.03442823830441845)했다. 학습 파이프라인 때 확인했던 재현성 패턴이 서빙에서도 그대로 성립함을 확인.

### 3-6. 임계값 기반 사기/정상 이분법 응답 추가

지금까지는 확률만 반환했는데, 실무에서 바로 쓰려면 "그래서 사기냐 아니냐"가 필요하다. 통상적인 0.5 기준은 우리 데이터(사기 1.1%)에서는 사실상 전부 "정상"으로 판정돼버려 쓸 수 없다. 대신 `evaluate.py`에 `threshold_at_fpr` 함수를 추가해, TPR@FPR5% 계산 때 쓴 것과 같은 지점의 **실제 확률 임계값**(0.0413)을 계산해서 판정 기준으로 사용했다.

### 3-7. mlflow.pyfunc로 모델+임계값 통합

**문제의식**: 모델(`model.pkl`)과 임계값(`threshold.json`)이 파일 두 개로 따로 있으면, 모델만 재학습하고 임계값 갱신을 깜빡해서 서로 어긋날 위험이 있다.

**해결**: `mlflow.pyfunc.PythonModel`을 상속한 `FraudPyfuncModel` 클래스를 만들어, "모델 로드"와 "확률 계산 + 임계값 비교 판정"을 하나의 배포 단위로 묶었다.

```python
class FraudPyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, threshold): ...
    def load_context(self, context): ...   # LightGBM 모델 로드
    def predict(self, context, model_input, params=None):
        # 확률 계산 + threshold 비교 -> fraud_probability, is_fraud 반환
```

`train.py`는 이제 `models/pyfunc_model/` 폴더 하나에 모델+임계값을 통째로 저장하고, `serve.py`는 `mlflow.pyfunc.load_model()`로 이 패키지를 통째로 불러와 `.predict()`만 호출한다.

**실행 중 발견한 문제**: 서버 실행 방식(`uvicorn src.serve:app`)과 학습 실행 방식(`python src/train.py`)이 파이썬 모듈을 찾는 경로가 서로 달라서, cloudpickle로 저장된 `fraud_model` 모듈을 서버가 못 찾는 에러(`ModuleNotFoundError`)가 났다. `serve.py`에서 `src/` 폴더를 모듈 검색 경로에 직접 추가해서 해결.

**검증**: 통합 전/후 동일한 입력에 대해 확률값이 소수점까지 완전히 일치(`0.0023413776876002874`, `0.06256071991078879`) — 구조만 바뀌고 동작은 그대로임을 확인.

### 3-8. 다단계 위험도 분류 (이분법 → 3단계)

**문제의식**: 확률 0.042(임계값 바로 위)와 0.9(거의 확실한 사기)를 똑같이 "사기"로 취급하면, 애매한 케이스까지 전부 자동 차단되어 정상 고객이 억울하게 막힐 수 있다.

**구현**: 임계값을 하나가 아니라 두 개로 나눴다.
- `block_threshold`(엄격, FPR 5% 지점, 기존 임계값): 이 이상이면 즉시 차단. 실수하면 정상 고객을 막는 손해가 크므로 엄격하게 잡는다.
- `review_threshold`(느슨, FPR 20% 지점): 이 이상이면 사람이 재검토. 사람이 다시 확인하니 더 널찍하게 잡아도 된다.

```python
if probability >= self.block_threshold:   # 0.0413
    return "block"       # 즉시 차단
if probability >= self.review_threshold:  # 0.0098
    return "review"       # 사람이 수동으로 재검토
return "low"               # 자동 승인
```

`FraudPyfuncModel`이 `fraud_probability`뿐 아니라 `risk_level`(low/review/block)과 `recommended_action`(구간별 후속 조치 문구)까지 함께 반환하도록 확장했다.

**검증**: 세 구간 각각 실제 데이터로 확인.

| 구간 | 확률 | risk_level | recommended_action |
|---|---|---|---|
| low | 0.0023 | low | 자동 승인 |
| review | 0.0150 | review | 사람이 수동으로 재검토 |
| block | 0.0626 | block | 즉시 차단 |

### 3-9. 확률 보정(Calibration) 검토

**질문**: 모델이 "확률 0.06"이라고 예측했을 때, 실제로 그런 예측을 받은 사람들 중 6%가 진짜 사기일까?

**검증 방법**: valid셋을 확률 구간 10개로 나눠(`sklearn.calibration.calibration_curve`), 각 구간의 "모델 평균 예측 확률"과 "실제 사기 비율"을 비교했다.

| 예측확률 평균 | 실제사기비율 | 차이 |
|---|---|---|
| 0.0150 | 0.0218 | 0.0068 (가장 큰 차이) |
| 0.0865 | 0.0867 | 0.0002 |
| (그 외 8개 구간) | | 0.0001~0.0005 |

차이가 전반적으로 작았다. `CalibratedClassifierCV(method="isotonic")`로 실제 보정까지 적용해서 Brier score(낮을수록 정확, 0=완벽)로 비교:

| | Brier score |
|---|---|
| 보정 전 | 0.012391 |
| 보정 후 | 0.012076 |

개선폭이 약 2.5%로 미미했다 (게다가 이 수치는 valid셋을 보정 학습과 평가에 동시에 써서 다소 낙관적으로 나왔을 가능성도 있음).

**결론: 지금 모델(LightGBM, log-loss 목적함수)은 이미 확률이 꽤 잘 보정되어 있어 calibration을 도입하지 않기로 판단했다.** 목록에 있는 기법을 무조건 다 적용하는 게 아니라, 데이터로 직접 검증하고 "지금은 필요 없다"고 판단하는 것도 정당한 엔지니어링 결정이라는 걸 확인한 사례.

### 3-10. 전용 모델 서빙 프레임워크 검토 (BentoML / Seldon Core / KServe / SageMaker)

| 도구 | 핵심 특징 |
|---|---|
| BentoML | 모델을 "Bento" 배포 단위로 패키징, Docker 이미지·API 문서·배치 예측 자동 지원 |
| Seldon Core / KServe | 쿠버네티스 기반, 자동 스케일링·A/B 테스트를 설정만으로 지원 |
| AWS SageMaker | 완전관리형 클라우드 서비스, 서버 관리 불필요 (비용 발생) |

**판단**: 이 도구들은 트래픽이 많고 여러 모델을 팀이 함께 운영하는 상황을 위해 만들어졌다. 지금은 혼자 진행하는 학습 프로젝트에 실제 트래픽도 없어서, 도입 시 얻는 이득보다 새로 배워야 할 개념(쿠버네티스 등)이 더 많다. **지금은 도입하지 않고, FastAPI로 직접 짜서 "아래에서 무슨 일이 일어나는지" 이해하는 것을 우선**하기로 했다. 트래픽이 늘거나 모델을 여러 개 운영해야 하는 상황이 오면 그때 BentoML부터 검토.

### 3-11. 잘못된 입력값 처리 (범주형 + 숫자 범위)

**범주형**: `payment_type: str`처럼 아무 문자열이나 받던 걸, 학습 데이터에 실제 존재했던 값만 허용하는 `Literal["AA","AB","AC","AD","AE"]`로 좁혔다 (`employment_status`, `housing_status`, `source`, `device_os` 5개 컬럼 전부 동일 적용). 학습 때 없던 값(`"ZZ"` 등)을 보내면 예측 시도 없이 `422`로 즉시 거부되고, "AA, AB, AC, AD, AE 중 하나여야 합니다" 같은 구체적 에러 메시지가 자동으로 붙는다.

**숫자 범위**: `tests/test_data_quality.py`에서 학습 데이터 값 범위를 검증했던 것과 같은 아이디어를, 이번엔 **들어오는 요청**에도 적용했다. 24개 숫자 컬럼 전부에 `pydantic.Field(ge=..., le=...)`로 학습 데이터의 실제 min~max를 그대로 하한/상한으로 걸었다. 단, `velocity_6h`·`credit_risk_score`는 음수가 정상값이라는 걸 알고 있으므로(`preprocess.py`의 규칙과 동일) 하한 없이 상한만 걸었다.

```python
customer_age: int = Field(ge=10, le=90)       # 학습 데이터 실제 범위
velocity_6h: float = Field(le=16715.57)        # 음수 정상값이라 하한 없음
```

**검증**: `customer_age=-50`, `income=999`는 `422`로 거부되고, `velocity_6h=-50`(음수)은 의도대로 정상 통과(`200`) 확인.

**한계(의도적 트레이드오프)**: 이 범위는 학습 데이터 기준이라, 미래에 정상적으로 범위를 벗어나는 값(예: 91세 고객)이 생겨도 똑같이 거부된다. "모델이 한 번도 못 본 값에 대한 신뢰 못할 예측"보다 "일단 막고 사람이 확인"을 선택한 것이며, 재학습 시 범위도 함께 갱신해야 한다.

### 3-12. 서빙 코드용 pytest 테스트

`fastapi.testclient.TestClient`로 서버를 실제로 켜지 않고도 요청/응답을 검증하는 테스트를 작성했다 (`tests/test_serve.py`). `models/pyfunc_model`이 없는 환경(CI 등)에서는 자동으로 skip된다.

| 검증 항목 | 개수 |
|---|---|
| `/health` 응답 | 1 |
| `/predict` 정상 응답 (200, 필드 형태) | 2 |
| 잘못된 범주형 값 거부 (422) | 1 |
| 필수 필드 누락 거부 (422) | 1 |
| 결측 허용 필드 null 처리 | 1 |
| 숫자 범위 초과 거부 (422, 4가지 케이스) | 4 |
| 음수 허용 컬럼 정상 통과 | 1 |

총 11개. 전체 pytest 스위트는 21(기존) → 27(서빙 기본) → **32개**로 늘었고 전부 통과. CI에도 자동 포함된다.

### 3-13. 로깅

**문제의식**: `preprocess.py`/`train.py`는 `print()`로만 결과를 출력했고, `serve.py`는 요청이 들어와도 콘솔에 아무 기록이 안 남았다. "언제 이상한 요청이 왔는지", "예측이 왜 실패했는지"를 나중에 추적할 방법이 없었다.

**구현**: `src/logging_config.py`에 모든 파일이 공유하는 로깅 설정을 하나로 통일해서 만들고, `preprocess.py`/`train.py`의 `print()`를 `logger.info()`로 교체, `serve.py`에는 세 종류의 로그를 추가했다.

- 서버 시작 시(`INFO`): 모델 경로, 임계값
- 예측 성공 시(`INFO`): 주요 입력값 + 확률 + risk_level
- 잘못된 요청 거부 시(`WARNING`): `RequestValidationError`에 대한 커스텀 exception handler로 포착
- 예측 자체가 실패하는 예상 못 한 상황(`ERROR`): `logger.exception`으로 전체 스택 트레이스까지 기록

```
2026-08-16 23:36:32 [INFO] serve: 모델 로드 완료: models/pyfunc_model (review_threshold=0.0098, block_threshold=0.0413)
2026-08-16 23:36:41 [INFO] serve: 예측 완료: age=50, payment_type=AA, device_os=other -> probability=0.0023, risk_level=low
2026-08-16 23:36:41 [WARNING] serve: 잘못된 요청 거부됨 (422): [...]
```

**오버스펙 경계**: 로그 파일 회전(rotation), 외부 로그 수집 시스템 연동(ELK 등), 요청별 고유 ID는 의도적으로 넣지 않았다. 지금은 콘솔 로그만으로 충분하고, 실제 트래픽 규모가 생기면 그때 추가하는 게 맞다고 판단.

**검증**: 정상 요청과 거부된 요청을 실제로 보내서 각각 `INFO`/`WARNING`으로 정확히 분리되어 기록되는 것을 확인. pytest 32개는 로깅 추가 후에도 전부 그대로 통과 (로깅은 응답 내용에 영향을 주지 않는 부가 기능이므로).

### 3-14. 예측 로그를 파일로 영속화 (7주차 드리프트 감지 준비)

**문제의식**: 3-13에서 만든 콘솔 로그는 사람이 읽기엔 좋지만, 서버를 끄면 사라지고 프로그램이 다시 읽어서 분석하기엔 불편하다. 7주차 드리프트 감지를 위해서는 "운영 중 실제로 들어온 데이터"가 파일로 남아있어야 학습 데이터 분포와 비교할 수 있다.

**구현**: `logging_config.py`에 `log_prediction_record()` 함수를 추가해, 예측 성공 시 입력 피처 전체 + 결과를 **JSONL(한 줄에 JSON 객체 하나)** 형식으로 `logs/predictions.jsonl`에 이어붙이도록 했다. 거부된 요청(422)은 포함하지 않아서 "드리프트 비교용 피처 데이터"라는 목적을 흐리지 않게 범위를 한정했다.

```python
def log_prediction_record(record: dict) -> None:
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with open(PREDICTION_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
```

**검증**: 요청 2건을 보내 파일에 정확히 2줄이 쌓이는 것을 확인했고, `pd.read_json(..., lines=True)` + `pd.json_normalize`로 다시 표(DataFrame)로 복원해서 `data/processed/train.parquet`과 나란히 피처 평균을 비교할 수 있는 것까지 확인했다 — 7주차에 하게 될 작업의 축소판.

### 3-15. Docker Compose에 `api` 서비스 통합 + 서빙 테스트를 CI에서도 항상 실행되게 개선

**문제의식**: 지금 `docker-compose.yml`엔 `train`, `mlflow-server`만 있고 `api`(서빙)는 따로 `docker run`으로만 띄워봤다. 학습→기록→서빙이 하나의 명령으로 이어지지 않았다. 또한 `tests/test_serve.py`는 `models/pyfunc_model`(train.py 실행 결과물)이 있어야 도는데, GitHub Actions CI 서버는 DVC remote(로컬 경로)에 접근할 수 없어서 원본 데이터를 못 받고, 그래서 `train.py`를 실행할 수 없다 — 즉 CI에서는 이 테스트가 **항상 skip**되고 있었다.

처음엔 "CI에서 preprocess.py+train.py를 먼저 돌리자"는 계획이었지만, 위 이유로 **실행 자체가 불가능한 계획**이라는 걸 뒤늦게 발견했다. 방향을 바꿔서, `tests/test_preprocess.py`가 100만 행 대신 손으로 만든 미니 데이터를 쓰는 것과 같은 원리로, **테스트 안에서 작은 가짜 학습 데이터로 진짜와 똑같은 구조의 모델을 그 자리에서 만들어 쓰도록** `test_serve.py`를 다시 짰다.

**구현**:
- `fraud_model.py`에 `save_fraud_pyfunc_model()` 공용 함수 추가 — `train.py`(진짜 모델)와 `test_serve.py`(가짜 모델)가 "모델+임계값을 pyfunc로 저장"하는 절차를 중복 작성하지 않도록 함
- `serve.py`의 모델 경로를 환경변수(`PYFUNC_MODEL_PATH`)로 덮어쓸 수 있게 수정 — 테스트가 자체 생성한 가짜 모델 경로를 가리키게 하기 위함
- `test_serve.py`가 300행짜리 가짜 학습 데이터로 작은 LightGBM을 학습하고, 그걸로 만든 pyfunc 모델을 테스트에 사용하도록 재작성. `pytest.mark.skipif` 제거 — 이제 실제 학습 여부와 무관하게 **항상 실행**됨
- `docker-compose.yml`에 `api` 서비스 추가 (`Dockerfile.serve`로 빌드, `train` 완료 후 시작하도록 `depends_on: condition: service_completed_successfully`, `models/`·`logs/` 볼륨 공유)

**검증**:
- `models/` 폴더를 통째로 지운 상태(=CI 환경 재현)에서 `test_serve.py` 11개가 여전히 전부 통과하는 것을 직접 확인
- `docker compose up`으로 mlflow-server → train → api 순서로 자동 기동되는 것 확인. `train`이 만든 실제 임계값(review=0.0098, block=0.0413)을 `api`가 그대로 읽어서 서빙하는 것, 그리고 예측 결과가 로컬 실행 때와 동일한 것 확인
- 컨테이너 안에서 남긴 `logs/predictions.jsonl`이 볼륨을 통해 호스트에도 그대로 남는 것 확인

### 3-16. 아직 남은 것

여기서부터는 지금 프로젝트 규모(1인, 무트래픽)에는 오버스펙으로 판단해 보류: 인증/인가, rate limiting, 로그 수집 시스템, 메트릭/알림, 오토스케일링, A/B 테스트, MLflow 모델 레지스트리 정식 승격 절차. (자세한 목록과 판단 근거는 발표 시 구두로 설명)

---

## 4. 다음 단계

7주차: 드리프트 감지, 자동 재학습 루프. `logs/predictions.jsonl`을 운영 데이터 삼아 학습 데이터 분포와 비교하는 것부터 시작.
