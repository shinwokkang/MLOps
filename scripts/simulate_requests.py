"""
드리프트 감지 실습을 위한 '운영 데이터' 시뮬레이션 스크립트.

완전 무작위 값을 만들면 그 자체가 학습 데이터 분포와 달라서 가짜 드리프트
신호가 생긴다. 그래서 valid.parquet(month 6, 모델이 학습 때 안 쓴 데이터)에서
실제 행을 그대로 뽑아 API 요청 형태로만 바꿔 보낸다 - 지금은 "드리프트 없는
정상 기준선"을 만드는 용도. 나중에 --shift 옵션 등으로 특정 컬럼을 일부러
비틀면 드리프트를 의도적으로 재현하는 데도 이 스크립트를 재사용할 수 있다.

사용법:
    1) 다른 터미널에서 서버를 켜둔다:
       uvicorn src.serve:app --port 8000 --app-dir .
    2) 이 스크립트를 실행한다:
       python scripts/simulate_requests.py --n 50
"""

import argparse
import time

import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"
VALID_PATH = "data/processed/valid.parquet"


def load_sample_requests(n: int, seed: int) -> list[dict]:
    df = pd.read_parquet(VALID_PATH)
    sample = df.drop(columns=["fraud_bool", "month"]).sample(n=n, random_state=seed)

    # NaN은 JSON으로 못 보내므로 None으로 바꾸고, 나머지는 파이썬 기본 타입으로 변환
    records = sample.astype(object).where(pd.notna(sample), None).to_dict(orient="records")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="보낼 요청 개수")
    parser.add_argument("--seed", type=int, default=None, help="샘플링 랜덤 시드")
    parser.add_argument("--delay", type=float, default=0.0, help="요청 사이 대기 시간(초)")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())
    records = load_sample_requests(args.n, seed)

    risk_counts = {"low": 0, "review": 0, "block": 0}
    errors = 0

    for record in records:
        response = requests.post(API_URL, json=record, timeout=10)
        if response.status_code == 200:
            risk_counts[response.json()["risk_level"]] += 1
        else:
            errors += 1
            print(f"  실패 ({response.status_code}): {response.text[:200]}")
        if args.delay:
            time.sleep(args.delay)

    print(f"\n총 {len(records)}건 요청 완료 (seed={seed})")
    print(f"위험도 분포: {risk_counts}")
    print(f"실패: {errors}건")


if __name__ == "__main__":
    main()
