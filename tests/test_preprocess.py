"""
preprocess.py의 각 함수가 규약대로 동작하는지 검증하는 단위 테스트.

실제 100만 행 CSV 대신, 각 규칙을 검증하기 딱 맞는 크기의
손으로 만든 미니 DataFrame을 사용한다.
"""

import pandas as pd
import pytest

from preprocess import (
    NEG1_AS_MISSING_COLS,
    CATEGORICAL_COLS,
    drop_uninformative_columns,
    convert_missing_markers,
    cast_categoricals,
    split_by_month,
)


def make_sample_df() -> pd.DataFrame:
    """preprocess.py의 모든 함수가 다루는 컬럼을 갖춘 미니 데이터셋."""
    return pd.DataFrame(
        {
            "fraud_bool": [0, 1, 0, 0, 1],
            "prev_address_months_count": [-1, 12, -1, 5, 8],
            "bank_months_count": [-1, -1, 3, 7, 2],
            "current_address_months_count": [10, -1, 4, 2, 6],
            "session_length_in_minutes": [5.0, -1, 3.2, 1.1, 2.5],
            "device_distinct_emails_8w": [1, -1, 2, 1, 1],
            "intended_balcon_amount": [-5.3, 10.2, -0.1, 20.0, -3.0],
            "credit_risk_score": [-10, 50, -5, 30, 12],  # 음수가 정상값인 컬럼
            "velocity_6h": [-100.0, 200.0, -50.0, 10.0, 5.0],  # 음수가 정상값인 컬럼
            "payment_type": ["AA", "AB", "AA", "AC", "AB"],
            "employment_status": ["CA", "CB", "CA", "CA", "CB"],
            "housing_status": ["BA", "BB", "BA", "BC", "BA"],
            "source": ["INTERNET", "INTERNET", "TELEAPP", "INTERNET", "INTERNET"],
            "device_os": ["linux", "windows", "other", "linux", "macintosh"],
            "device_fraud_count": [0, 0, 0, 0, 0],
            # 0~7월을 골고루 포함 + 경계값인 5월을 반드시 포함
            "month": [0, 3, 5, 6, 7],
        }
    )


# ---------- drop_uninformative_columns ----------

def test_drop_uninformative_columns_removes_device_fraud_count():
    df = make_sample_df()
    result = drop_uninformative_columns(df)
    assert "device_fraud_count" not in result.columns
    # 다른 컬럼은 그대로 남아있어야 함
    assert "fraud_bool" in result.columns


# ---------- convert_missing_markers ----------

def test_neg1_columns_become_nan():
    df = make_sample_df()
    result = convert_missing_markers(df)
    for col in NEG1_AS_MISSING_COLS:
        # 원본에서 -1이었던 자리는 NaN이어야 함
        was_neg1 = df[col] == -1
        assert result.loc[was_neg1, col].isna().all(), f"{col}에서 -1이 NaN으로 안 바뀜"
        # -1이 아니었던 값은 그대로 보존되어야 함
        was_not_neg1 = df[col] != -1
        assert (result.loc[was_not_neg1, col] == df.loc[was_not_neg1, col]).all()


def test_intended_balcon_amount_negative_becomes_nan():
    df = make_sample_df()
    result = convert_missing_markers(df)
    was_negative = df["intended_balcon_amount"] < 0
    assert result.loc[was_negative, "intended_balcon_amount"].isna().all()
    was_not_negative = df["intended_balcon_amount"] >= 0
    assert (
        result.loc[was_not_negative, "intended_balcon_amount"]
        == df.loc[was_not_negative, "intended_balcon_amount"]
    ).all()


def test_credit_risk_score_and_velocity_6h_negatives_preserved():
    """음수가 정상값인 컬럼은 절대 건드리면 안 된다."""
    df = make_sample_df()
    result = convert_missing_markers(df)
    assert (result["credit_risk_score"] == df["credit_risk_score"]).all()
    assert (result["velocity_6h"] == df["velocity_6h"]).all()


# ---------- cast_categoricals ----------

def test_categorical_columns_get_category_dtype():
    df = make_sample_df()
    result = cast_categoricals(df)
    for col in CATEGORICAL_COLS:
        assert str(result[col].dtype) == "category", f"{col}이 category 타입이 아님"


# ---------- split_by_month ----------

def test_split_by_month_no_overlap_and_full_coverage():
    df = make_sample_df()
    train, valid, test = split_by_month(df)

    # 겹치는 행이 없어야 함 (인덱스 기준 교집합이 비어있어야 함)
    assert set(train.index) & set(valid.index) == set()
    assert set(train.index) & set(test.index) == set()
    assert set(valid.index) & set(test.index) == set()

    # 전체 행 수를 합치면 원본과 같아야 함 (빠지는 행이 없어야 함)
    assert len(train) + len(valid) + len(test) == len(df)


def test_split_by_month_boundaries_are_correct():
    df = make_sample_df()  # month: 0, 3, 6, 7
    train, valid, test = split_by_month(df)

    assert set(train["month"]) <= set(range(0, 6))
    assert set(valid["month"]) == {6}
    assert set(test["month"]) == {7}
