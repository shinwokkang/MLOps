"""
프로젝트 전체가 공유하는 로깅 설정.

여러 파일(preprocess.py, train.py, serve.py)에서 각자 다른 형식으로
로깅을 설정하면 로그 모양이 제각각이 되므로, 이 함수 하나로 통일한다.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# 사람이 읽는 콘솔 로그와는 별개로, 나중에 pandas로 다시 읽어서
# 학습 데이터 분포와 비교할 수 있도록(드리프트 감지 준비) 예측 결과를
# 한 줄짜리 JSON(JSONL 형식)으로 파일에 계속 이어붙인다.
PREDICTION_LOG_PATH = Path("logs/predictions.jsonl")


def log_prediction_record(record: dict) -> None:
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with open(PREDICTION_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
