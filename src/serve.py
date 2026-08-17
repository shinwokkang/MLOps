"""
모델+임계값이 하나로 묶인 MLflow pyfunc 모델(models/pyfunc_model)을 FastAPI로 서빙한다.

실행:
    uvicorn src.serve:app --reload --port 8000   (프로젝트 루트에서 실행)

문서(자동 생성된 API 화면) 확인:
    http://127.0.0.1:8000/docs
"""

import os
import sys
from pathlib import Path
from typing import Literal, Optional

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# train.py가 모델을 저장할 때(`python src/train.py`) src/ 폴더가 파이썬의
# 모듈 검색 경로에 자동으로 들어가서, fraud_model 모듈이 "최상위 모듈"이라는
# 이름으로 함께 저장된다(cloudpickle 내부에 이 모듈 이름이 그대로 박힘).
# uvicorn으로 서버를 실행할 때도 같은 이름으로 fraud_model을 찾을 수 있도록,
# src/ 폴더를 직접 검색 경로에 추가해준다.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_config import get_logger, log_prediction_record

logger = get_logger("serve")

# 환경변수로 덮어쓸 수 있게 해서, 테스트(tests/test_serve.py)가 진짜 학습 없이도
# 자체적으로 만든 작은 가짜 모델을 가리키게 할 수 있다.
PYFUNC_MODEL_PATH = Path(os.environ.get("PYFUNC_MODEL_PATH", "models/pyfunc_model"))

# fraud_model.py의 CATEGORICAL_COLS와 동일한 목록.
# (여기서도 필요한 이유: FastAPI가 받은 값의 dtype을 미리 손질해야 pyfunc 모델에
#  정확히 넘길 수 있다 - 실제 category 변환 자체는 FraudPyfuncModel.predict 안에서 한다)
CATEGORICAL_COLS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

app = FastAPI(title="BAF Fraud Detection API")

# 서버가 켜질 때 딱 한 번만 모델(+임계값)을 읽어서 메모리에 올려둔다.
loaded_model = mlflow.pyfunc.load_model(str(PYFUNC_MODEL_PATH))

# 응답에 "어떤 임계값을 썼는지" 보여주기 위해, 감싸진 원본 파이썬 객체에서 값을 꺼내온다.
_wrapped = loaded_model.unwrap_python_model()
REVIEW_THRESHOLD = _wrapped.review_threshold
BLOCK_THRESHOLD = _wrapped.block_threshold

logger.info(
    "모델 로드 완료: %s (review_threshold=%.4f, block_threshold=%.4f)",
    PYFUNC_MODEL_PATH,
    REVIEW_THRESHOLD,
    BLOCK_THRESHOLD,
)


class FraudCheckRequest(BaseModel):
    # 숫자 컬럼들의 ge(이상)/le(이하)는 학습 데이터(train.parquet)에 실제로
    # 존재했던 최소~최대값 그대로다. 모델이 한 번도 본 적 없는 범위의 값이
    # 들어오면 거부해서, 결과를 신뢰할 수 없는 외삽(extrapolation)을 막는다.
    # 주의: 미래에 정상적으로 이 범위를 벗어나는 값이 생길 수도 있다 -
    #       그 경우엔 재학습 시 범위를 다시 계산해서 갱신해야 한다.
    income: float = Field(ge=0.1, le=0.9)
    name_email_similarity: float = Field(ge=0.0, le=1.0)
    prev_address_months_count: Optional[float] = Field(default=None, ge=0, le=377)
    current_address_months_count: Optional[float] = Field(default=None, ge=0, le=425)
    customer_age: int = Field(ge=10, le=90)
    days_since_request: float = Field(ge=0, le=76.59)
    intended_balcon_amount: Optional[float] = Field(default=None, ge=0, le=113)
    # str 대신 Literal을 쓰면, 학습 데이터에 실제로 존재했던 값 외의 것이
    # 들어오면 FastAPI가 요청 자체를 거부한다.
    payment_type: Literal["AA", "AB", "AC", "AD", "AE"]
    zip_count_4w: int = Field(ge=0, le=6700)
    # velocity_6h는 음수가 정상값이라 하한을 두지 않는다 (preprocess.py 규칙과 동일)
    velocity_6h: float = Field(le=16715.57)
    velocity_24h: float = Field(ge=0, le=9506.9)
    velocity_4w: float = Field(ge=0, le=6994.77)
    bank_branch_count_8w: int = Field(ge=0, le=2381)
    date_of_birth_distinct_emails_4w: int = Field(ge=0, le=39)
    employment_status: Literal["CA", "CB", "CC", "CD", "CE", "CF", "CG"]
    # credit_risk_score도 음수가 정상값이라 하한을 두지 않는다
    credit_risk_score: int = Field(le=389)
    email_is_free: Literal[0, 1]
    housing_status: Literal["BA", "BB", "BC", "BD", "BE", "BF", "BG"]
    phone_home_valid: Literal[0, 1]
    phone_mobile_valid: Literal[0, 1]
    bank_months_count: Optional[float] = Field(default=None, ge=0, le=32)
    has_other_cards: Literal[0, 1]
    proposed_credit_limit: float = Field(ge=0, le=2100)
    foreign_request: Literal[0, 1]
    source: Literal["INTERNET", "TELEAPP"]
    session_length_in_minutes: Optional[float] = Field(default=None, ge=0, le=85.9)
    device_os: Literal["linux", "macintosh", "other", "windows", "x11"]
    keep_alive_session: Literal[0, 1]
    device_distinct_emails_8w: Optional[float] = Field(default=None, ge=0, le=2)


class FraudCheckResponse(BaseModel):
    fraud_probability: float
    risk_level: str          # "low" | "review" | "block"
    recommended_action: str  # 위험도 구간별 후속 조치 (fraud_model.py의 RECOMMENDED_ACTION)
    review_threshold: float
    block_threshold: float


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # pydantic이 요청을 거부한 경우(잘못된 카테고리 값, 범위 초과 등)를 별도로 기록한다.
    # 이게 없으면 "누가 어떤 이상한 값을 보내다 막혔는지"가 로그에 전혀 안 남는다.
    logger.warning("잘못된 요청 거부됨 (422): %s", exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=FraudCheckResponse)
def predict(request: FraudCheckRequest):
    # 요청 1건 -> pandas DataFrame 1행으로 변환
    row = pd.DataFrame([request.model_dump()])

    # 행이 1개뿐이면 값이 null(None)인 컬럼의 dtype을 pandas가 숫자로
    # 제대로 추론 못 하고 object로 남겨버릴 수 있다. 범주형이 아닌 컬럼은
    # 전부 명시적으로 숫자로 변환해서 이 문제를 방지한다.
    for col in row.columns:
        if col not in CATEGORICAL_COLS:
            row[col] = pd.to_numeric(row[col], errors="coerce")

    try:
        # FraudPyfuncModel.predict가 fraud_probability, risk_level, recommended_action을 반환한다
        result = loaded_model.predict(row)
    except Exception:
        # 검증(pydantic)은 통과했지만 예측 자체가 실패한, 예상 못 한 상황.
        # logger.exception은 에러 메시지뿐 아니라 전체 스택 트레이스까지 로그에 남긴다.
        logger.exception(
            "예측 중 예상치 못한 에러 발생 (age=%s, payment_type=%s)",
            request.customer_age,
            request.payment_type,
        )
        raise

    fraud_probability = float(result["fraud_probability"].iloc[0])
    risk_level = str(result["risk_level"].iloc[0])

    logger.info(
        "예측 완료: age=%s, payment_type=%s, device_os=%s -> probability=%.4f, risk_level=%s",
        request.customer_age,
        request.payment_type,
        request.device_os,
        fraud_probability,
        risk_level,
    )

    # 사람이 읽는 콘솔 로그와 별개로, 나중에 학습 데이터 분포와 비교할 수 있도록
    # 입력 피처 전체 + 예측 결과를 logs/predictions.jsonl에 구조화된 형태로 남긴다
    # (7주차 드리프트 감지에서 "운영 데이터"로 재사용할 예정)
    log_prediction_record(
        {
            "input": request.model_dump(),
            "fraud_probability": fraud_probability,
            "risk_level": risk_level,
        }
    )

    return FraudCheckResponse(
        fraud_probability=fraud_probability,
        risk_level=risk_level,
        recommended_action=str(result["recommended_action"].iloc[0]),
        review_threshold=REVIEW_THRESHOLD,
        block_threshold=BLOCK_THRESHOLD,
    )
