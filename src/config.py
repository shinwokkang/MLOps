"""
conf/config.yaml을 읽어 딕셔너리로 반환하는 공용 함수.

conf/config.yaml의 위치는 "이 파일(config.py)의 위치" 기준으로 계산한다.
(data/처럼 실행 시점의 현재 폴더에 의존하는 상대경로가 아니라,
 코드 파일 위치 기준 고정 경로 - 어느 폴더에서 스크립트를 실행하든 항상 같은 config를 찾는다)
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "conf" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)
