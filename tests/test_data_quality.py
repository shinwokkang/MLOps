"""
실제로 전처리된 data/processed/train.parquet을 열어서
컬럼별 값 범위/허용값 집합이 상식과 어긋나지 않는지 확인한다.

가짜 데이터가 아니라 진짜 파이프라인 결과물을 검사하므로,
preprocess.py를 먼저 실행해서 parquet이 만들어져 있어야 한다.
(CI 등 데이터가 없는 환경에서는 이 파일 전체를 건너뛴다.)
"""

import pandas as pd
import pytest

TRAIN_PATH = "data/processed/train.parquet"

pytestmark = pytest.mark.skipif(
    not __import__("os").path.exists(TRAIN_PATH),
    reason="data/processed/train.parquet 없음 - 먼저 preprocess.py를 실행해야 함",
)


@pytest.fixture(scope="module")
def train_df():
    return pd.read_parquet(TRAIN_PATH)


def test_fraud_bool_is_binary(train_df):
    assert set(train_df["fraud_bool"].unique()) <= {0, 1}


def test_income_is_normalized_0_to_1(train_df):
    assert train_df["income"].min() >= 0
    assert train_df["income"].max() <= 1


def test_month_within_train_range(train_df):
    assert set(train_df["month"].unique()) <= set(range(0, 6))


def test_categorical_columns_have_expected_dtype(train_df):
    cat_cols = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
    for col in cat_cols:
        assert str(train_df[col].dtype) == "category", f"{col}이 category 타입이 아님"


def test_device_fraud_count_dropped(train_df):
    assert "device_fraud_count" not in train_df.columns


def test_credit_risk_score_can_be_negative(train_df):
    """이 컬럼은 음수가 정상값이므로, 음수가 존재해야 오히려 정상이다."""
    assert (train_df["credit_risk_score"] < 0).any()


def test_neg1_missing_markers_are_gone(train_df):
    """전처리 후에는 -1이 결측치(NaN)로 바뀌어 있어야 하므로, -1 값 자체가 없어야 한다."""
    cols = [
        "prev_address_months_count",
        "bank_months_count",
        "current_address_months_count",
        "session_length_in_minutes",
        "device_distinct_emails_8w",
    ]
    for col in cols:
        assert (train_df[col] == -1).sum() == 0, f"{col}에 여전히 -1 값이 남아있음"
