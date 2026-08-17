"""
전처리된 parquet(train/valid)을 읽어 LightGBM 베이스라인을 학습하고,
MLflow run 1건에 params + metrics + 모델 아티팩트를 기록한다.
"""

import pickle
import tempfile
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import pandas as pd

from config import load_config
from evaluate import evaluate_model, threshold_at_fpr
from fraud_model import FraudPyfuncModel, save_fraud_pyfunc_model
from logging_config import get_logger

logger = get_logger("train")

# 서빙(serve.py)이 항상 같은 위치에서 최신 모델(모델+임계값 통합본)을
# 찾을 수 있도록 고정 경로에도 저장한다.
# (MLflow run 경로는 실행할 때마다 이름이 바뀌어서 서빙 서버가 참조하기 불편함)
PYFUNC_EXPORT_PATH = Path("models/pyfunc_model")

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
    logger.info("학습 시작")

    train_df = load_split("train")
    valid_df = load_split("valid")
    logger.info("데이터 로드 완료: train=%d행, valid=%d행", len(train_df), len(valid_df))

    X_train, y_train = split_features_target(train_df)
    X_valid, y_valid = split_features_target(valid_df)

    mlflow.set_experiment("baf-fraud-baseline")

    with mlflow.start_run(run_name="lightgbm-baseline"):
        # 고정 params 기록: 하이퍼파라미터, random_state, 시간 분할 설정
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_params(TIME_SPLIT_CONFIG)

        model = lgb.LGBMClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train, categorical_feature="auto")
        logger.info("모델 학습 완료 (random_state=%s)", RANDOM_STATE)

        metrics = evaluate_model(model, X_valid, y_valid)
        mlflow.log_metrics(metrics)
        logger.info("검증 성능: %s", metrics)

        # 위험도 구간 2개(block/review)에 해당하는 실제 확률 임계값 계산
        # - block(엄격, FPR 5%): 즉시 차단. 실수 시 정상 고객을 막는 손해가 크므로 엄격하게
        # - review(느슨, FPR 20%): 사람 재검토 대상. 사람이 다시 확인하니 널찍하게 잡아도 됨
        y_score_valid = model.predict_proba(X_valid)[:, 1]
        block_threshold = threshold_at_fpr(y_valid, y_score_valid, target_fpr=0.05)
        review_threshold = threshold_at_fpr(y_valid, y_score_valid, target_fpr=0.20)
        mlflow.log_metric("block_threshold_at_fpr5", block_threshold)
        mlflow.log_metric("review_threshold_at_fpr20", review_threshold)

        # signature/input_example은 시도하지 않음: category dtype 컬럼과
        # MLflow의 스키마 검증기가 서로 맞지 않아 생기는 문제라, 실제 입력
        # 스키마 설계는 서빙 단계(6주차)에서 다루는 것으로 미룬다.
        # (원본 LightGBM 모델도 실험 비교용으로 그대로 기록해둔다)
        mlflow.lightgbm.log_model(model, name="model")

        # LightGBM 모델을 pyfunc 아티팩트로 넘기려면 먼저 파일로 존재해야 한다
        # (mlflow.pyfunc.log_model의 artifacts는 "파일 경로"를 받는 방식이라서)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_path = Path(tmp_dir) / "lgbm_model.pkl"
            with open(tmp_model_path, "wb") as f:
                pickle.dump(model, f)

            # 모델 + 임계값(2개)을 하나로 묶은 배포 단위를 MLflow run에 기록
            mlflow.pyfunc.log_model(
                name="fraud_pyfunc_model",
                python_model=FraudPyfuncModel(
                    review_threshold=review_threshold, block_threshold=block_threshold
                ),
                artifacts={"lgbm_model": str(tmp_model_path)},
            )

        # 서빙용 고정 경로에도 동일하게 저장 (fraud_model.py의 공용 함수 재사용)
        save_fraud_pyfunc_model(model, review_threshold, block_threshold, PYFUNC_EXPORT_PATH)

        logger.info(
            "임계값 계산 완료: block=%.4f(FPR5%%), review=%.4f(FPR20%%)",
            block_threshold,
            review_threshold,
        )
        logger.info("서빙용 모델(모델+임계값 통합) 저장 완료: %s", PYFUNC_EXPORT_PATH)
        logger.info("학습 종료")


if __name__ == "__main__":
    main()
