"""
7주차 드리프트 감지를 위한 예측 로그 로딩 유틸리티.

logs/predictions.jsonl(serve.py가 남기는 JSONL)은 JSON을 거치면서
pandas의 category dtype 정보가 사라진다(전부 object로 로드됨).
train/valid/test parquet과 나란히 비교하려면 같은 dtype이어야 하므로,
이 정합화를 이 파일 한 곳에서만 담당한다 - 드리프트 비교 코드는
앞으로 pd.read_json을 직접 쓰지 않고 항상 load_prediction_logs()를 통해서만
로그에 접근한다.
"""

from pathlib import Path

import pandas as pd

from fraud_model import CATEGORICAL_COLS

PREDICTION_LOG_PATH = Path("logs/predictions.jsonl")


def load_prediction_logs(path=PREDICTION_LOG_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    logs = pd.read_json(path, lines=True)

    # "input" 컬럼 하나에 29개 피처가 딕셔너리로 뭉쳐 들어있던 걸 표로 펼침
    inputs = pd.json_normalize(logs["input"])

    # train/valid/test parquet과 동일하게 범주형 컬럼을 category dtype으로 복원
    for col in CATEGORICAL_COLS:
        if col in inputs.columns:
            inputs[col] = inputs[col].astype("category")

    # 피처 + 예측 결과(확률/위험도)를 한 표로 합쳐서 반환
    return pd.concat(
        [inputs, logs[["timestamp", "fraud_probability", "risk_level"]]], axis=1
    )
