"""
EDA(notebooks/eda.ipynb)에서 발견한 패턴들이 데이터에 실제로 존재하는지 검증한다.

이 테스트들은 "코드가 버그 없이 동작하는가"가 아니라
"데이터가 우리가 알고 있는 특성을 여전히 유지하고 있는가"를 확인한다.
나중에 새 데이터(예: month 7, 또는 운영 중 유입되는 데이터)가 들어왔을 때
같은 검사를 돌리면, 이 패턴이 깨진 순간이 곧 "데이터가 달라졌다(드리프트)"는 신호가 된다.
(7주차 드리프트 감지 로직의 기초가 되는 아이디어)

train.py/preprocess.py를 먼저 실행해서 parquet이 만들어져 있어야 한다.
"""

import os

import pandas as pd
import pytest

TRAIN_PATH = "data/processed/train.parquet"
VALID_PATH = "data/processed/valid.parquet"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(TRAIN_PATH) and os.path.exists(VALID_PATH)),
    reason="전처리된 parquet 없음 - 먼저 preprocess.py를 실행해야 함",
)


@pytest.fixture(scope="module")
def df():
    train = pd.read_parquet(TRAIN_PATH)
    valid = pd.read_parquet(VALID_PATH)
    return pd.concat([train, valid], ignore_index=True)


def _fraud_rate_by_missing(df, col):
    is_missing = df[col].isna()
    rate_missing = df.loc[is_missing, "fraud_bool"].mean()
    rate_present = df.loc[~is_missing, "fraud_bool"].mean()
    return rate_missing, rate_present


@pytest.mark.parametrize(
    "col",
    ["prev_address_months_count", "bank_months_count", "intended_balcon_amount"],
)
def test_missing_value_has_higher_fraud_rate(df, col):
    """EDA 발견: '정보가 없다'는 사실 자체가 사기 신호였다 (결측일 때 사기율이 3~5배 높음)."""
    rate_missing, rate_present = _fraud_rate_by_missing(df, col)
    assert rate_missing > rate_present, (
        f"{col}: 결측일 때 사기율({rate_missing:.4f})이 "
        f"값 있을 때({rate_present:.4f})보다 높아야 하는데 아님"
    )


def test_fraud_rate_increases_with_age(df):
    """EDA 발견: 나이대가 높을수록 사기율이 뚜렷하게 증가 (10대 대비 70대+가 약 12배)."""
    bins = [10, 20, 30, 40, 50, 60, 70, 90]
    age_group = pd.cut(df["customer_age"], bins=bins, right=False, labels=False)
    rate_by_age = df.groupby(age_group, observed=True)["fraud_bool"].mean()

    youngest_rate = rate_by_age.iloc[0]
    oldest_rate = rate_by_age.iloc[-1]
    assert oldest_rate > youngest_rate * 2, (
        f"최고령 구간 사기율({oldest_rate:.4f})이 최연소 구간({youngest_rate:.4f})의 "
        "2배를 넘어야 하는데 아님 - 나이-사기율 관계가 예전과 달라졌을 수 있음"
    )


@pytest.mark.parametrize("col", ["velocity_6h", "velocity_24h", "velocity_4w"])
def test_low_velocity_bucket_has_higher_fraud_rate_than_high(df, col):
    """EDA 발견(반직관적): 신청이 몰릴수록(velocity가 높을수록) 오히려 사기율이 낮았다."""
    bucket = pd.qcut(df[col], q=5, labels=[1, 2, 3, 4, 5])
    rate_by_bucket = df.groupby(bucket, observed=True)["fraud_bool"].mean()
    assert rate_by_bucket.loc[1] > rate_by_bucket.loc[5], (
        f"{col}: 가장 낮은 구간 사기율({rate_by_bucket.loc[1]:.4f})이 "
        f"가장 높은 구간({rate_by_bucket.loc[5]:.4f})보다 커야 하는데 아님"
    )
