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
                  station: str = "잠실(송파구청)", code: int = 222) -> None:
    dates = pd.date_range("2026-06-01", periods=days).strftime("%Y-%m-%d")
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
