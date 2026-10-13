import math

import pytest

from metro_insight.model.evaluation import ModelScore, evaluate, mae, rmse


def test_mae_is_mean_absolute_error():
    assert mae([100, 200, 300], [110, 190, 300]) == pytest.approx(20 / 3)


def test_rmse_penalizes_large_errors():
    actual = [100, 100, 100, 100]
    even = [110, 90, 110, 90]  # 오차 10 네 번
    spiky = [100, 100, 100, 140]  # 오차 40 한 번
    assert mae(actual, even) == mae(actual, spiky) == 10
    assert rmse(actual, even) == 10
    assert rmse(actual, spiky) == 20


def test_evaluate():
    assert evaluate([0, 0], [3, -4]) == ModelScore(mae=3.5, rmse=math.sqrt(12.5))
    assert evaluate([5, 5], [5, 5]) == ModelScore(mae=0, rmse=0)


def test_invalid_input():
    with pytest.raises(ValueError):
        mae([1, 2], [1])
    with pytest.raises(ValueError):
        rmse([], [])
