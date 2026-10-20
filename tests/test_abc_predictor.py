"""ABCPredictor 단위 테스트. models/·data/processed 없이 임시 CSV로만 돌린다."""
from pathlib import Path

import pandas as pd
import pytest

from metro_insight.model.predictor import (
    TIME_CODE_BY_HOUR,
    ABCPredictor,
    PredictionInput,
    PredictionUnavailable,
)


def values(station: str = "잠실", date: str = "2026-06-08", hour: int = 8,
           temperature: float | None = 10.0) -> PredictionInput:
    import datetime
    d = datetime.date.fromisoformat(date)
    return PredictionInput(
        date=date, line="2호선", station=station, hour=hour,
        weekday=d.weekday(), is_weekend=d.weekday() >= 5, is_holiday=False,
        temperature=temperature, rainfall=False, snowfall=False,
    )


def write_history(tmp_path: Path, days: int = 8,
                  station: str = "잠실(송파구청)", code: int = 222,
                  start: str = "2026-06-01") -> None:
    dates = pd.date_range(start, periods=days).strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates,
        "line": "2호선",
        "station": station,
        "station_code": code,
        "line_code": 2,
        "hour_code": 3,  # 08시 버킷
        "boarding": range(100, 100 + days),
    })
    df.to_csv(tmp_path / "train.csv.gz", index=False, compression="gzip")


def make_predictor(tmp_path: Path) -> ABCPredictor:
    return ABCPredictor(
        models_dir=tmp_path, data_dir=tmp_path, metrics_path=tmp_path / "metrics.json",
    )


def test_station_key_strips_parenthetical():
    assert ABCPredictor._station_key("잠실(송파구청)") == "잠실"
    assert ABCPredictor._station_key("수유(강북구청)") == "수유"
    assert ABCPredictor._station_key("총신대입구(이수)") == "총신대입구"
    assert ABCPredictor._station_key("강남") == "강남"


@pytest.mark.parametrize("hour,code", [(0, 19), (3, 19), (4, 0), (5, 0), (6, 1),
                                       (8, 3), (23, 18), (24, 19)])
def test_hour_mapping(hour, code):
    assert TIME_CODE_BY_HOUR[hour] == code


def test_alias_resolves(tmp_path):
    write_history(tmp_path)
    p = make_predictor(tmp_path)
    assert p._station_code[("2호선", "잠실(송파구청)")] == 222
    assert p._station_code[("2호선", "잠실")] == 222  # 별칭
    assert p.date_range() == ("2026-06-01", "2026-06-08")


def test_unknown_station_raises(tmp_path):
    write_history(tmp_path)
    p = make_predictor(tmp_path)
    with pytest.raises(PredictionUnavailable, match="학습 데이터에 없"):
        p.predict(values(station="가능"))


def test_missing_history_raises(tmp_path):
    write_history(tmp_path, days=3)  # 7일 이력이 못 되는 데이터
    p = make_predictor(tmp_path)
    with pytest.raises(PredictionUnavailable, match="직전 7일"):
        p.predict(values(date="2026-06-03"))


def test_mean_prev7d_is_rounded_like_training(tmp_path):
    """학습표와 같은 0.1자리 반올림이어야 화면과 모델 직접 호출 값이 같아진다."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-06-01", periods=8).strftime("%Y-%m-%d"),
        "line": "2호선",
        "station": "잠실(송파구청)",
        "station_code": 222,
        "line_code": 2,
        "hour_code": 3,
        "boarding": [1, 2, 4, 8, 16, 32, 64, 128],  # 7일 평균 127/7 = 18.1428...
    })
    df.to_csv(tmp_path / "train.csv.gz", index=False, compression="gzip")

    seen = {}

    class Capture:
        def predict(self, X):
            seen.update(X.iloc[0].to_dict())
            return [0.0]

    p = make_predictor(tmp_path)
    p._models = {"A": Capture()}
    assert p.predict(values()) == {"A": 0.0}
    assert seen["mean_prev7d"] == 18.1  # round(18.142857..., 1)


def test_predict_with_full_history(tmp_path):
    write_history(tmp_path, days=8)  # 2026-06-08 의 직전 7일이 전부 있다

    class Dummy:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return [self.value]

    p = make_predictor(tmp_path)
    p._models = {"A": Dummy(1.0), "B": Dummy(2.0), "C": Dummy(3.0)}
    assert p.predict(values()) == {"A": 1.0, "B": 2.0, "C": 3.0}
    # 기온이 없으면 날씨 모델 C는 뺀다
    assert p.predict(values(temperature=None)) == {"A": 1.0, "B": 2.0}


class BrokenModel:
    """계산에 실패하는 모델 (화면이 이유를 보여 주도록)."""

    def predict(self, X):
        raise ValueError("테스트용 실패")


def test_seasonal_fallback_for_dates_outside_data(tmp_path):
    """학습 데이터 밖 날짜는 '작년 같은 요일' 값으로 예측한다 (오차가 커진다는 뜻)."""
    write_history(tmp_path, days=15, start="2025-10-01")
    p = make_predictor(tmp_path)
    sc = p._station_code[("2호선", "잠실(송파구청)")]
    # 2026-10-09 는 데이터 밖 → 364일 전(2025-10-09) 값으로 대체되어야 한다
    lag1, lag7, mean7 = p._lags(sc, 3, "2026-10-09")
    assert lag1 is not None and mean7 is not None
    assert p.seasonal_used is True
    # 데이터 안 날짜(직전 7일이 전부 있는 경우)는 그대로 (전년치 사용 안 함)
    p._seasonal_used = False
    p._lags(sc, 3, "2025-10-15")
    assert p.seasonal_used is False


def test_model_failure_reason_is_kept(tmp_path):
    write_history(tmp_path, days=8)

    p = make_predictor(tmp_path)
    p._models = {"A": BrokenModel()}
    out = p.predict(values())
    assert out == {}  # 실패한 모델은 숫자를 내지 않는다
    assert "ValueError" in p.last_errors["A"]
    assert "테스트용 실패" in p.last_errors["A"]


def test_stale_cache_is_ignored(tmp_path):
    """원본과 맞지 않는 캐시(버전·시각)는 버리고 CSV 를 읽는다."""
    import numpy as np

    write_history(tmp_path, days=8)
    # 옛 형식(버전 1)·엉뚱한 역 코드를 가진 캐시를 심어 둔다
    np.savez(
        tmp_path / "boarding_history.npz",
        boarding=np.zeros((1, 20, 1), dtype="float32"),
        station_codes=np.asarray([999]),
        dates=np.asarray(["2026-06-01"]),
        pair_lines=np.asarray(["2호선"]),
        pair_stations=np.asarray(["잠실(송파구청)"]),
        pair_codes=np.asarray([999], dtype="int32"),
        line_names=np.asarray(["2호선"]),
        line_codes=np.asarray([1], dtype="int32"),
        version=np.asarray(1, dtype="int64"),
        csv_mtime=np.asarray(0, dtype="int64"),
        source_rows=np.asarray(0, dtype="int64"),
    )
    p = make_predictor(tmp_path)
    assert p._station_code[("2호선", "잠실(송파구청)")] == 222  # CSV 값 (캐시 999 는 무시)
