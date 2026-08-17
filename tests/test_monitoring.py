"""
train/test parquet(학습 파이프라인)과 logs/predictions.jsonl(운영 요청)이
드리프트 비교에 쓰일 수 있는 정합된 스키마를 가지는지 검증한다.

세 데이터 소스의 역할:
- train.parquet (month 0~5): 기준 분포(baseline) - 모델이 학습한 데이터
- test.parquet (month 7)   : 시간이 지난 "미래" 데이터 - 드리프트 실험용으로 예약된 것
- logs/predictions.jsonl    : 실제 서빙 API에 들어온 운영 요청 기록

여기서는 "드리프트가 있는가"(통계 검정)가 아니라, 세 소스를 나란히 놓고
비교할 수 있는 상태인지(같은 피처 컬럼, 같은 dtype)만 확인한다.
"""

import os

import pandas as pd
import pytest

from fraud_model import CATEGORICAL_COLS
from monitoring import load_prediction_logs

TRAIN_PATH = "data/processed/train.parquet"
TEST_PATH = "data/processed/test.parquet"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)),
    reason="전처리된 parquet 없음 - 먼저 preprocess.py를 실행해야 함",
)


def _feature_columns(df: pd.DataFrame, exclude) -> set:
    return set(df.columns) - set(exclude)


@pytest.fixture(scope="module")
def train_df():
    return pd.read_parquet(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_parquet(TEST_PATH)


@pytest.fixture(scope="module")
def logs_df():
    logs = load_prediction_logs()
    if logs.empty:
        pytest.skip(
            "logs/predictions.jsonl이 비어있음 - "
            "scripts/simulate_requests.py 등으로 트래픽을 먼저 만들어야 함"
        )
    return logs


def test_train_and_test_have_same_feature_columns(train_df, test_df):
    """train과 test(month7)는 fraud_bool/month를 제외하면 피처 구성이 같아야 한다."""
    train_features = _feature_columns(train_df, ["fraud_bool", "month"])
    test_features = _feature_columns(test_df, ["fraud_bool", "month"])
    assert train_features == test_features


def test_logs_feature_columns_match_train(train_df, logs_df):
    """운영 로그의 피처 구성이 학습 데이터와 정확히 일치해야 비교가 가능하다."""
    train_features = _feature_columns(train_df, ["fraud_bool", "month"])
    logs_features = _feature_columns(logs_df, ["timestamp", "fraud_probability", "risk_level"])
    assert train_features == logs_features


@pytest.mark.parametrize("col", CATEGORICAL_COLS)
def test_categorical_dtype_matches_across_sources(train_df, test_df, logs_df, col):
    """세 소스 모두 범주형 컬럼이 category dtype이어야 groupby 등 비교 연산이 일관되게 동작한다."""
    assert str(train_df[col].dtype) == "category"
    assert str(test_df[col].dtype) == "category"
    assert str(logs_df[col].dtype) == "category"


@pytest.mark.parametrize("col", CATEGORICAL_COLS)
def test_logs_categorical_values_are_known_from_training(train_df, logs_df, col):
    """serve.py가 Literal 타입으로 검증하므로, 로그에 학습 때 없던 카테고리 값이 있으면 안 된다."""
    known_categories = set(train_df[col].cat.categories)
    seen_categories = set(logs_df[col].dropna().unique())
    assert seen_categories <= known_categories
