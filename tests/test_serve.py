"""
FastAPI 서빙 코드(src/serve.py)를 검증하는 테스트.

TestClient는 실제로 서버를 켜지 않고(uvicorn 없이), 코드 안에서
요청/응답을 그대로 흉내 내서 빠르게 테스트할 수 있게 해준다.

주의: 이 테스트는 진짜 학습된 모델(models/pyfunc_model)에 의존하지 않는다.
CI 서버는 DVC remote(로컬 경로)에 접근할 수 없어서 원본 데이터를 못 받고,
따라서 train.py를 실행할 수 없다 - 그래서 여기서는 preprocess.py의
convert_missing_markers 등을 미니 데이터로 검증했던 것과 같은 원리로,
작은 가짜 학습 데이터로 진짜와 똑같은 구조의 모델을 그 자리에서 만들어 쓴다.
"""

import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import lightgbm as lgb
from evaluate import threshold_at_fpr
from fraud_model import CATEGORICAL_COLS, save_fraud_pyfunc_model


def _make_synthetic_training_data(n: int = 300, seed: int = 0):
    """FraudCheckRequest와 같은 29개 피처 스키마를 가진 작은 가짜 데이터."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "income": rng.uniform(0.1, 0.9, n),
            "name_email_similarity": rng.uniform(0, 1, n),
            "prev_address_months_count": rng.uniform(0, 300, n),
            "current_address_months_count": rng.uniform(0, 300, n),
            "customer_age": rng.integers(10, 90, n),
            "days_since_request": rng.uniform(0, 50, n),
            "intended_balcon_amount": rng.uniform(0, 100, n),
            "payment_type": rng.choice(["AA", "AB", "AC", "AD", "AE"], n),
            "zip_count_4w": rng.integers(0, 5000, n),
            "velocity_6h": rng.uniform(-100, 10000, n),
            "velocity_24h": rng.uniform(0, 8000, n),
            "velocity_4w": rng.uniform(0, 6000, n),
            "bank_branch_count_8w": rng.integers(0, 2000, n),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 30, n),
            "employment_status": rng.choice(["CA", "CB", "CC", "CD", "CE", "CF", "CG"], n),
            "credit_risk_score": rng.integers(-100, 300, n),
            "email_is_free": rng.integers(0, 2, n),
            "housing_status": rng.choice(["BA", "BB", "BC", "BD", "BE", "BF", "BG"], n),
            "phone_home_valid": rng.integers(0, 2, n),
            "phone_mobile_valid": rng.integers(0, 2, n),
            "bank_months_count": rng.uniform(0, 30, n),
            "has_other_cards": rng.integers(0, 2, n),
            "proposed_credit_limit": rng.uniform(100, 2000, n),
            "foreign_request": rng.integers(0, 2, n),
            "source": rng.choice(["INTERNET", "TELEAPP"], n),
            "session_length_in_minutes": rng.uniform(0, 60, n),
            "device_os": rng.choice(["linux", "macintosh", "other", "windows", "x11"], n),
            "keep_alive_session": rng.integers(0, 2, n),
            "device_distinct_emails_8w": rng.uniform(0, 2, n),
        }
    )
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    fraud_bool = pd.Series(rng.choice([0, 1], size=n, p=[0.9, 0.1]))
    return df, fraud_bool


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # 1) 가짜 데이터로 작은 LightGBM 모델을 학습
    X, y = _make_synthetic_training_data()
    model = lgb.LGBMClassifier(n_estimators=10, random_state=0)
    model.fit(X, y, categorical_feature="auto")

    scores = model.predict_proba(X)[:, 1]
    block_threshold = threshold_at_fpr(y, scores, target_fpr=0.05)
    review_threshold = threshold_at_fpr(y, scores, target_fpr=0.20)

    # 2) 진짜 train.py와 똑같은 방식(fraud_model.py의 공용 함수)으로 pyfunc 모델 저장
    model_dir = tmp_path_factory.mktemp("pyfunc_model")
    save_fraud_pyfunc_model(model, review_threshold, block_threshold, model_dir)

    # 3) serve.py가 이 가짜 모델을 읽도록 환경변수로 경로를 지정한 뒤 import
    #    (serve.py는 모듈이 처음 import될 때 딱 한 번 모델을 로드하므로, 이 시점 이전에
    #     환경변수가 설정되어 있어야 한다)
    os.environ["PYFUNC_MODEL_PATH"] = str(model_dir)
    from serve import app

    return TestClient(app)


@pytest.fixture(scope="module")
def valid_request_body():
    """모델이 기대하는 모든 피처를 담은, 값 자체는 임의로 만든 정상 요청."""
    return {
        "income": 0.5,
        "name_email_similarity": 0.5,
        "prev_address_months_count": None,
        "current_address_months_count": 10.0,
        "customer_age": 30,
        "days_since_request": 0.01,
        "intended_balcon_amount": 10.0,
        "payment_type": "AA",
        "zip_count_4w": 100,
        "velocity_6h": 1000.0,
        "velocity_24h": 1000.0,
        "velocity_4w": 1000.0,
        "bank_branch_count_8w": 1,
        "date_of_birth_distinct_emails_4w": 1,
        "employment_status": "CA",
        "credit_risk_score": 100,
        "email_is_free": 0,
        "housing_status": "BA",
        "phone_home_valid": 1,
        "phone_mobile_valid": 1,
        "bank_months_count": 10.0,
        "has_other_cards": 0,
        "proposed_credit_limit": 500.0,
        "foreign_request": 0,
        "source": "INTERNET",
        "session_length_in_minutes": 5.0,
        "device_os": "linux",
        "keep_alive_session": 1,
        "device_distinct_emails_8w": 1.0,
    }


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_with_valid_request_returns_200(client, valid_request_body):
    response = client.post("/predict", json=valid_request_body)
    assert response.status_code == 200


def test_predict_response_has_expected_fields(client, valid_request_body):
    response = client.post("/predict", json=valid_request_body)
    body = response.json()
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"low", "review", "block"}
    assert isinstance(body["recommended_action"], str)


def test_predict_rejects_unknown_payment_type(client, valid_request_body):
    """학습 때 없던 카테고리 값(payment_type='ZZ')은 422로 거부되어야 한다."""
    bad_body = {**valid_request_body, "payment_type": "ZZ"}
    response = client.post("/predict", json=bad_body)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("customer_age", -50),        # 학습 데이터 최소값(10)보다 작음
        ("customer_age", 200),        # 학습 데이터 최대값(90)보다 큼
        ("income", 999),              # 학습 데이터 최대값(0.9)보다 큼
        ("zip_count_4w", -1),         # 음수가 될 수 없는 컬럼
    ],
)
def test_predict_rejects_out_of_range_numeric_values(client, valid_request_body, field, bad_value):
    """학습 데이터에 없던 범위의 숫자값은 422로 거부되어야 한다."""
    bad_body = {**valid_request_body, field: bad_value}
    response = client.post("/predict", json=bad_body)
    assert response.status_code == 422


def test_predict_allows_negative_velocity_6h(client, valid_request_body):
    """velocity_6h는 음수가 정상값이므로 거부되면 안 된다 (preprocess.py 규칙과 동일)."""
    body = {**valid_request_body, "velocity_6h": -50.0}
    response = client.post("/predict", json=body)
    assert response.status_code == 200


def test_predict_rejects_missing_required_field(client, valid_request_body):
    """필수 필드(customer_age)가 아예 빠지면 422로 거부되어야 한다."""
    bad_body = {k: v for k, v in valid_request_body.items() if k != "customer_age"}
    response = client.post("/predict", json=bad_body)
    assert response.status_code == 422


def test_predict_allows_null_for_optional_fields(client, valid_request_body):
    """결측치 허용 컬럼(prev_address_months_count 등)은 null이어도 정상 처리되어야 한다."""
    body = {**valid_request_body, "prev_address_months_count": None}
    response = client.post("/predict", json=body)
    assert response.status_code == 200
