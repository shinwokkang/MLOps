"""
BAF(Base.csv) 원본 CSV를 읽어서
1) 불필요한 컬럼 제거
2) -1 / 음수로 표시된 결측치를 실제 NaN으로 변환
3) 범주형 컬럼을 category 타입으로 변환
4) month 기준으로 train / valid / test 분할
5) data/processed/ 에 parquet으로 저장

경로, 컬럼 목록, 시간 분할 기준 같은 "설정값"은 코드에 직접 적지 않고
conf/config.yaml에서 읽어온다 (config.py 참고).
"""

import pandas as pd

from config import load_config
from logging_config import get_logger

logger = get_logger("preprocess")

_cfg = load_config()

RAW_PATH = _cfg["paths"]["raw_csv"]
PROCESSED_DIR = _cfg["paths"]["processed_dir"]

DROP_COLUMNS = _cfg["preprocessing"]["drop_columns"]

# -1이 결측치를 의미하는 컬럼들 (요구사항 문서 + 실제 데이터로 검증 완료)
NEG1_AS_MISSING_COLS = _cfg["preprocessing"]["neg1_as_missing_columns"]

# 음수 전체가 결측치를 의미하는 컬럼들 (예: intended_balcon_amount)
NEGATIVE_AS_MISSING_COLS = _cfg["preprocessing"]["negative_as_missing_columns"]

# 범주형 컬럼 (LightGBM 네이티브 처리를 위해 category dtype으로 변환)
CATEGORICAL_COLS = _cfg["preprocessing"]["categorical_columns"]

_TRAIN_MONTH_START, _TRAIN_MONTH_END = _cfg["time_split"]["train_months"]
_VALID_MONTH = _cfg["time_split"]["valid_month"]
_TEST_MONTH = _cfg["time_split"]["test_month"]


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    # device_fraud_count는 전체 값이 0 하나뿐이라 모델이 배울 정보가 없음
    return df.drop(columns=DROP_COLUMNS)


def convert_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # -1을 진짜 결측(NaN)으로 표시
    for col in NEG1_AS_MISSING_COLS:
        df.loc[df[col] == -1, col] = pd.NA

    # 이 컬럼들은 -1이 아니라 "음수 전체"가 결측 표시 규칙
    for col in NEGATIVE_AS_MISSING_COLS:
        df.loc[df[col] < 0, col] = pd.NA

    # credit_risk_score, velocity_6h의 음수는 정상값이므로 건드리지 않음
    return df


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def split_by_month(df: pd.DataFrame):
    train = df[df["month"].between(_TRAIN_MONTH_START, _TRAIN_MONTH_END)]
    valid = df[df["month"] == _VALID_MONTH]
    test = df[df["month"] == _TEST_MONTH]
    return train, valid, test


def main():
    logger.info("전처리 시작: %s", RAW_PATH)

    df = load_raw(RAW_PATH)
    logger.info("원본 로드 완료: %d행 %d열", *df.shape)

    df = drop_uninformative_columns(df)
    df = convert_missing_markers(df)
    df = cast_categoricals(df)

    train, valid, test = split_by_month(df)

    train.to_parquet(f"{PROCESSED_DIR}/train.parquet", index=False)
    valid.to_parquet(f"{PROCESSED_DIR}/valid.parquet", index=False)
    test.to_parquet(f"{PROCESSED_DIR}/test.parquet", index=False)

    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        n_fraud = int(part["fraud_bool"].sum())
        logger.info(
            "%s: %d행 / 사기 %d건 / %.3f%%", name, len(part), n_fraud, n_fraud / len(part) * 100
        )

    logger.info("전처리 종료: %s 에 parquet 저장 완료", PROCESSED_DIR)


if __name__ == "__main__":
    main()
