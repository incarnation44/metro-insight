"""모델 평가 지표.

같은 Test 데이터에서 A~E 모델의 예측값과 실제 승차 인원을 비교한다.

* MAE: 평균(|실제값 - 예측값|). 평균적으로 몇 명 차이 나는지
* RMSE: √평균((실제값 - 예측값)²). 큰 오차에 더 큰 패널티
"""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelScore:
    """한 모델의 Test 데이터 평가 결과 (단위: 명)."""

    mae: float
    rmse: float


def _errors(actual: Sequence[float], predicted: Sequence[float]) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"실제값과 예측값의 개수가 다르다: {actual.shape} != {predicted.shape}")
    if actual.size == 0:
        raise ValueError("평가할 데이터가 없다")
    return actual - predicted


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return float(np.mean(np.abs(_errors(actual, predicted))))


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return float(np.sqrt(np.mean(_errors(actual, predicted) ** 2)))


def evaluate(actual: Sequence[float], predicted: Sequence[float]) -> ModelScore:
    return ModelScore(mae=mae(actual, predicted), rmse=rmse(actual, predicted))
