"""
전처리된 parquet(train/valid)을 읽어 LightGBM 베이스라인을 학습하고,
MLflow run 1건에 params + metrics + 모델 아티팩트를 기록한다.
"""

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

from config import load_config
from evaluate import evaluate_model

_cfg = load_config()

PROCESSED_DIR = _cfg["paths"]["processed_dir"]
TARGET_COL = _cfg["target_col"]
# month는 시간 그 자체를 나타내는 컬럼이라 피처로 넣으면 시간 분할의 의미가 흐려짐 -> 제외
DROP_COLS = [TARGET_COL, "month"]

RANDOM_STATE = _cfg["model"]["random_state"]

# LightGBM 하이퍼파라미터: random_state + yaml에 추가로 적어둔 값(있다면)
MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
    **_cfg["model"]["params"],
}

_train_start, _train_end = _cfg["time_split"]["train_months"]
TIME_SPLIT_CONFIG = {
    "train_months": f"{_train_start}-{_train_end}",
    "valid_month": _cfg["time_split"]["valid_month"],
    "test_month": _cfg["time_split"]["test_month"],
}


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(f"{PROCESSED_DIR}/{name}.parquet")


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]
    return X, y


def main():
    train_df = load_split("train")
    valid_df = load_split("valid")

    X_train, y_train = split_features_target(train_df)
    X_valid, y_valid = split_features_target(valid_df)

    mlflow.set_experiment("baf-fraud-baseline")

    with mlflow.start_run(run_name="lightgbm-baseline"):
        # 고정 params 기록: 하이퍼파라미터, random_state, 시간 분할 설정
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_params(TIME_SPLIT_CONFIG)

        model = lgb.LGBMClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train, categorical_feature="auto")

        metrics = evaluate_model(model, X_valid, y_valid)
        mlflow.log_metrics(metrics)

        # signature/input_example은 시도하지 않음: category dtype 컬럼과
        # MLflow의 스키마 검증기가 서로 맞지 않아 생기는 문제라, 실제 입력
        # 스키마 설계는 서빙 단계(6주차)에서 다루는 것으로 미룬다.
        mlflow.lightgbm.log_model(model, name="model")

        print("검증 성능:", metrics)


if __name__ == "__main__":
    main()
