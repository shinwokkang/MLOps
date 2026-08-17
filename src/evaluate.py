"""
불균형 이진분류 평가 지표 계산 함수 모음.

accuracy는 쓰지 않는다 - 사기 비율이 1% 수준이라 전부 정상으로
예측해도 99%에 가까운 accuracy가 나오기 때문에 의미가 없다.
대신 TPR@FPR5%(주지표)와 AUPRC(보조지표)를 사용한다.
"""

import numpy as np
from sklearn.metrics import roc_curve, average_precision_score


def tpr_at_fpr(y_true, y_score, target_fpr: float = 0.05) -> float:
    """FPR을 target_fpr(기본 5%) 이하로 제한했을 때 얻을 수 있는 최대 TPR.

    "정상 고객을 사기로 잘못 판단하는 비율을 5%로 묶어뒀을 때,
    실제 사기 중 몇 %를 잡아낼 수 있는가"를 의미한다.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # fpr은 오름차순으로 정렬되어 있음 -> target_fpr을 넘지 않는 마지막 지점을 찾는다
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def threshold_at_fpr(y_true, y_score, target_fpr: float = 0.05) -> float:
    """FPR을 target_fpr 이하로 제한하는 지점의 실제 확률 임계값(threshold).

    tpr_at_fpr가 "그 지점에서 TPR이 얼마인지"를 반환한다면,
    이 함수는 "그 지점이 확률 몇 이상부터 시작되는지"를 반환한다.
    서빙 시 이 값 이상을 '사기'로 판정하는 데 쓴다.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(thresholds[idx])


def auprc(y_true, y_score) -> float:
    """Precision-Recall 곡선 아래 면적. 불균형 데이터에서 AUROC보다 신뢰도가 높다."""
    return float(average_precision_score(y_true, y_score))


def evaluate_model(model, X, y_true) -> dict:
    """학습된 모델과 검증/테스트 데이터로 지표 dict를 계산해서 반환."""
    y_score = model.predict_proba(X)[:, 1]
    return {
        "tpr_at_fpr5": tpr_at_fpr(y_true, y_score, target_fpr=0.05),
        "auprc": auprc(y_true, y_score),
    }
