"""
LightGBM 모델과 '위험도 구간 임계값'을 하나로 묶은 MLflow pyfunc 모델.

기존에는 model.pkl(모델)과 threshold.json(임계값)이 파일 두 개로 따로
저장되어 있어서, 모델만 새로 학습하고 임계값 갱신을 깜빡하면 서로 어긋날
위험이 있었다. 이 클래스는 둘을 하나의 "배포 단위"로 묶어서 그 위험을 없앤다.

임계값은 하나(사기/정상 이분법)가 아니라 두 개를 써서 세 구간으로 나눈다:
    확률 < review_threshold          -> low    (자동 승인)
    review_threshold <= 확률 < block_threshold -> review (사람 검토)
    확률 >= block_threshold          -> block  (즉시 차단)

block_threshold(엄격, FPR 5%)는 실수 시 손해가 큰 "즉시 차단"에 쓰고,
review_threshold(느슨, FPR 20%)는 어차피 사람이 다시 확인하는 "검토 대상"이라
더 널찍하게 잡는다.
"""

import pickle
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import mlflow.pyfunc

# preprocess.py에서 category dtype으로 바꿨던 컬럼과 동일한 목록.
CATEGORICAL_COLS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

# 위험도 구간별 후속 조치 (실무에서는 이 부분이 실제 운영 정책과 연결된다)
RECOMMENDED_ACTION = {
    "low": "자동 승인",
    "review": "사람이 수동으로 재검토",
    "block": "즉시 차단",
}


class FraudPyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, review_threshold: float, block_threshold: float):
        self.review_threshold = review_threshold
        self.block_threshold = block_threshold

    def load_context(self, context):
        # 모델 아티팩트("lgbm_model")를 실제로 불러오는 시점.
        # log_model/save_model에서 artifacts={"lgbm_model": ...}로 넘긴 경로를 읽는다.
        with open(context.artifacts["lgbm_model"], "rb") as f:
            self.model = pickle.load(f)

    def _risk_level(self, probability: float) -> str:
        if probability >= self.block_threshold:
            return "block"
        if probability >= self.review_threshold:
            return "review"
        return "low"

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        row = model_input.copy()
        for col in CATEGORICAL_COLS:
            if col in row.columns:
                row[col] = row[col].astype("category")

        probabilities = self.model.predict_proba(row)[:, 1]
        risk_levels = [self._risk_level(p) for p in probabilities]
        actions = [RECOMMENDED_ACTION[level] for level in risk_levels]

        return pd.DataFrame(
            {
                "fraud_probability": probabilities,
                "risk_level": risk_levels,
                "recommended_action": actions,
            }
        )


def save_fraud_pyfunc_model(model, review_threshold: float, block_threshold: float, path) -> None:
    """LightGBM 모델 + 임계값 2개를 하나의 pyfunc 배포 단위로 만들어 path에 저장한다.

    train.py(진짜 모델 저장)와 tests/test_serve.py(테스트용 가짜 모델 저장)가
    똑같은 절차(임시 pickle -> mlflow.pyfunc.save_model)를 반복하지 않도록 공용 함수로 뺐다.
    """
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_model_path = Path(tmp_dir) / "lgbm_model.pkl"
        with open(tmp_model_path, "wb") as f:
            pickle.dump(model, f)

        if path.exists():
            shutil.rmtree(path)
        mlflow.pyfunc.save_model(
            path=str(path),
            python_model=FraudPyfuncModel(
                review_threshold=review_threshold, block_threshold=block_threshold
            ),
            artifacts={"lgbm_model": str(tmp_model_path)},
        )
