"""
pytest가 테스트를 실행하기 전에 가장 먼저 읽는 설정 파일.
src/ 폴더를 파이썬이 import할 수 있는 경로에 추가해서,
tests/ 안에서 `from preprocess import ...` 처럼 바로 불러올 수 있게 한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
