"""ABCPredictor 단위 테스트. models/·data/processed 없이 임시 CSV·npz만 돌린다."""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metro_insight.model.predictor import (
    FEATS,
    TIME_CODE_BY_HOUR,
    VERIFIED_HORIZON_BUCKETS,
    ABCPredictor,
    PredictionInput,
    PredictionUnavailable,
    climatology_temperature,
    horizon_bucket,
)

MONTH_ROWS = list[tuple[int, int, int, float, int]]
WEEKDAY_ROWS = list[tuple[int, int, int, float, int]]


def values(station: str = "잠실", date: str = "2025-10-10", hour: int = 8,
           temperature: float | None = 10.0, rainfall: bool | None = False,
           snowfall: bool | None = False) -> PredictionInput:
    import datetime
    d = datetime.date.fromisoformat(date)
    return PredictionInput(
        date=date, line="2호선", station=station, hour=hour,
        weekday=d.weekday(), is_weekend=d.weekday() >= 5, is_holiday=False,
        temperature=temperature, rainfall=rainfall, snowfall=snowfall,
    )


def write_history(tmp_path: Path, days: int = 20,
                  station: str = "잠실(송파구청)", code: int = 222,
                  start: str = "2025-10-01") -> None:
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


def write_profiles(tmp_path: Path, month_rows: MONTH_ROWS = (),
                   weekday_rows: WEEKDAY_ROWS = ()) -> None:
    """cal_profiles.npz: (역, 시간대, 월) 과 (역, 시간대, 요일) 프로파일."""
    month = np.asarray(month_rows, dtype="float64") if month_rows else np.empty((0, 5))
    weekday = np.asarray(weekday_rows, dtype="float64") if weekday_rows else np.empty((0, 5))
    np.savez(
        tmp_path / "cal_profiles.npz",
        station_code=month[:, 0].astype("int64"),
        hour_code=month[:, 1].astype("int64"),
        month=month[:, 2].astype("int64"),
        mean=month[:, 3],
        count=month[:, 4].astype("int64"),
        station_code_wd=weekday[:, 0].astype("int64"),
        hour_code_wd=weekday[:, 1].astype("int64"),
        weekday_wd=weekday[:, 2].astype("int64"),
        mean_wd=weekday[:, 3],
        count_wd=weekday[:, 4].astype("int64"),
    )


def make_predictor(tmp_path: Path, metrics_path: Path | None = None) -> ABCPredictor:
    return ABCPredictor(
        models_dir=tmp_path, data_dir=tmp_path,
        metrics_path=metrics_path or tmp_path / "metrics.json",
    )


class Capture:
    """마지막으로 받은 입력 한 행을 기억하고 고정값을 돌려주는 가짜 모델."""

    def __init__(self, value: float = 1.0):
        self.value = value
        self.seen: dict = {}

    def predict(self, X):
        self.seen = X.iloc[0].to_dict()
        return [self.value]


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


def test_date_range_is_serving_window(tmp_path):
    """데이터가 없어도 서빙 기간은 오늘부터 6개월(182일)이다."""
    p = make_predictor(tmp_path)
    expected_end = (date.today() + timedelta(days=182)).isoformat()
    assert p.date_range() == ("2024-01-08", expected_end)


def test_date_beyond_serving_end_raises(tmp_path):
    """오늘+182일을 넘는 날짜는 안내 메시지와 함께 거부한다."""
    write_history(tmp_path)
    p = make_predictor(tmp_path)
    p._models = {"B": Capture()}
    beyond = (date.today() + timedelta(days=183)).isoformat()
    with pytest.raises(ValueError, match="예측 가능 범위는"):
        p.predict(values(date=beyond))
    with pytest.raises(ValueError, match="오늘부터 6개월"):
        p.predict(values(date=beyond))


def test_unknown_station_raises(tmp_path):
    write_history(tmp_path)
    p = make_predictor(tmp_path)
    with pytest.raises(PredictionUnavailable, match="학습 데이터에 없"):
        p.predict(values(station="가능"))


def test_without_calendar_models_raises(tmp_path):
    write_history(tmp_path)
    p = make_predictor(tmp_path)  # cal_*.joblib 없음
    with pytest.raises(PredictionUnavailable, match="달력 모델"):
        p.predict(values())


def test_past_date_uses_asof_profiles(tmp_path):
    """데이터 안 날짜는 그 날짜 이전 자료만으로 월·요일 프로파일을 다시 만든다."""
    write_history(tmp_path, days=20)  # 2025-10-01 ~ 10-20, 승차 100~119
    p = make_predictor(tmp_path)
    capture = Capture()
    p._models = {"B": capture}

    assert p.predict(values(date="2025-10-10"))["B"] == 1.0
    # 월 프로파일: 10-01 ~ 10-09 (100~108) 만 쓴다 (그날 값 109 는 들어가면 안 된다)
    assert capture.seen["profile_month_mean"] == pytest.approx(104.0)
    assert capture.seen["profile_n"] == 9

    # 요일 프로파일: 그날 이전 + 같은 요일만
    dates = pd.date_range("2025-10-01", periods=20)
    same = (dates < pd.Timestamp("2025-10-10")) & (dates.weekday == pd.Timestamp("2025-10-10").weekday())
    expected = np.arange(100, 120)[same]
    assert len(expected) < 9  # 월 프로파일보다 좁게 잡힌다
    assert capture.seen["profile_weekday_n"] == len(expected)
    assert capture.seen["profile_weekday_mean"] == pytest.approx(float(expected.mean()))


def test_past_date_has_no_horizon(tmp_path):
    """과거 날짜는 실측이 있으므로 지평 라벨·지평 오차를 붙이지 않는다."""
    write_history(tmp_path)
    p = make_predictor(tmp_path)
    p._models = {"B": Capture()}

    out = p.predict(values(date="2025-10-10"))
    assert out["horizon_bucket"] is None
    assert out["horizon_scores"] == {}
    assert p.horizon_scores("2025-10-10") == {}


def test_future_date_uses_calendar_profiles(tmp_path):
    """데이터 끝 이후 날짜는 cal_profiles.npz 의 전체 이력 프로파일을 쓴다."""
    write_history(tmp_path)
    write_profiles(tmp_path, month_rows=[(222, 3, 3, 1234.5, 90)],
                   weekday_rows=[(222, 3, 2, 987.5, 13)])  # 2027-03-03 은 수요일
    p = make_predictor(tmp_path)
    capture = Capture()
    p._models = {"B": capture}

    out = p.predict(values(date="2027-03-03"))
    assert capture.seen["profile_month_mean"] == pytest.approx(1234.5)
    assert capture.seen["profile_n"] == 90
    assert capture.seen["profile_weekday_mean"] == pytest.approx(987.5)
    assert capture.seen["profile_weekday_n"] == 13
    assert out["horizon_bucket"] == "182-366"  # 2026-06-30 + 246일


def test_future_date_without_profiles_raises(tmp_path):
    write_history(tmp_path)
    p = make_predictor(tmp_path)
    p._models = {"B": Capture()}
    with pytest.raises(PredictionUnavailable, match="프로파일"):
        p.predict(values(date="2027-03-03"))


def test_missing_profile_key_gives_nan_and_zero(tmp_path):
    """프로파일 파일은 있지만 그 키가 없으면 학습과 같은 결측 규칙(NaN, 0)으로 넘긴다."""
    write_history(tmp_path)
    write_profiles(tmp_path, month_rows=[(222, 3, 1, 500.0, 4)])  # 3월·그 요일 키는 없다
    p = make_predictor(tmp_path)
    capture = Capture()
    p._models = {"B": capture}

    p.predict(values(date="2027-03-03"))
    assert np.isnan(capture.seen["profile_month_mean"])
    assert capture.seen["profile_n"] == 0
    assert np.isnan(capture.seen["profile_weekday_mean"])
    assert capture.seen["profile_weekday_n"] == 0


def test_missing_temperature_uses_monthly_normal(tmp_path):
    """기온이 없으면 월별 평년값, 비·눈은 '안 옴'(0)으로 두고 C 도 계산한다."""
    write_history(tmp_path)
    write_profiles(tmp_path, month_rows=[(222, 3, 3, 900.0, 30)],
                   weekday_rows=[(222, 3, 2, 800.0, 12)])
    p = make_predictor(tmp_path)
    capture = Capture(3.0)
    p._models = {"C": capture}

    out = p.predict(values(date="2027-03-03", temperature=None, rainfall=None, snowfall=None))
    assert out["C"] == 3.0  # C 를 건너뛰지 않는다
    assert capture.seen["temperature"] == pytest.approx(climatology_temperature(3))
    assert capture.seen["rainfall"] == 0
    assert capture.seen["snowfall"] == 0


def test_input_features_follow_model_order(tmp_path):
    """모델이 학습한 feature_names_in_ 순서대로 한 행을 만든다."""
    write_history(tmp_path)
    p = make_predictor(tmp_path)

    class Ordered(Capture):
        feature_names_in_ = np.asarray(["temperature", "station_code", "profile_weekday_n"])

    model = Ordered()
    p._models = {"C": model}
    p.predict(values(temperature=12.5))
    assert list(model.seen) == ["temperature", "station_code", "profile_weekday_n"]
    assert model.seen["temperature"] == pytest.approx(12.5)


def test_feature_contract_matches_training():
    # A = 역·시간대·요일·월, B = A + 공휴일, C = B + 날씨
    assert FEATS["A"] == ["profile_month_mean", "profile_n", "station_code", "line_code",
                          "profile_weekday_mean", "profile_weekday_n", "hour_code", "weekday"]
    assert FEATS["B"] == FEATS["A"] + ["is_holiday"]
    assert FEATS["C"] == FEATS["B"] + ["temperature", "rainfall", "snowfall"]
    assert "is_weekend" not in FEATS["C"]  # weekday 에서 유도되므로 넣지 않는다


class BrokenModel:
    """계산에 실패하는 모델 (화면이 이유를 보여 주도록)."""

    def predict(self, X):
        raise ValueError("테스트용 실패")


def test_model_failure_reason_is_kept(tmp_path):
    write_history(tmp_path)
    p = make_predictor(tmp_path)
    p._models = {"A": BrokenModel()}
    out = p.predict(values())
    assert "A" not in out  # 실패한 모델은 숫자를 내지 않는다
    assert "ValueError" in p.last_errors["A"]
    assert "테스트용 실패" in p.last_errors["A"]


@pytest.mark.parametrize("day,bucket", [
    ("2026-06-30", None),  # 데이터 안 → 지평 없음
    ("2026-07-01", "1-31"),
    ("2026-07-31", "1-31"),
    ("2026-08-01", "32-91"),
    ("2026-12-29", "182-366"),
    ("2027-07-01", "182-366"),
    ("2027-07-02", "367+"),
    ("2029-12-31", "367+"),
])
def test_horizon_bucket(day, bucket):
    assert horizon_bucket(day) == bucket


def test_verified_horizon_stops_at_181_days():
    """Test 로 오차를 재 본 구간은 181일까지다. 182일 이후는 미검증 구간이다."""
    assert horizon_bucket("2026-12-28") == "92-181"  # 데이터 끝 + 181일
    assert horizon_bucket("2026-12-29") == "182-366"  # + 182일
    assert VERIFIED_HORIZON_BUCKETS == ("1-31", "32-91", "92-181")
    assert "182-366" not in VERIFIED_HORIZON_BUCKETS
    assert "367+" not in VERIFIED_HORIZON_BUCKETS


def test_result_carries_horizon_scores(tmp_path):
    """구간 점수도 test 우선 — 화면 전체 ±MAE(scores())와 같은 기준을 쓴다."""
    write_history(tmp_path)
    write_profiles(tmp_path, month_rows=[(222, 3, 7, 900.0, 30)],
                   weekday_rows=[(222, 3, 4, 800.0, 12)])
    metrics = tmp_path / "metrics_calendar.json"
    # evaluate_calendar.py 는 test.{모델}.horizon.{구간}, train_calendar.py 는 valid.{모델}.horizon.{구간}
    metrics.write_text(json.dumps({
        "features": FEATS,
        "train_end": "2026-02-14",
        "valid": {
            "A": {"mae": 800.0, "rmse": 1100.0, "n": 100,
                  "horizon": {"1-31": {"mae": 750.0, "rmse": 1000.0, "n": 60}}},
        },
        "test": {
            "A": {"mae": 820.0, "rmse": 1050.0, "n": 40,
                  "horizon": {"1-31": {"mae": 700.0, "rmse": 950.0, "n": 20}}},
        },
    }), encoding="utf-8")
    p = make_predictor(tmp_path, metrics_path=metrics)
    p._models = {"A": Capture()}

    out = p.predict(values(date="2026-07-05"))
    assert out["horizon_bucket"] == "1-31"
    assert out["horizon_scores"]["A"].mae == pytest.approx(700.0)  # test 가 valid 를 이긴다
    assert p.scores()["A"].mae == pytest.approx(820.0)
    assert p.horizon_scores("2026-07-05")["A"].rmse == pytest.approx(950.0)


def test_horizon_falls_back_to_valid_without_test(tmp_path):
    """test 블록이 없으면 valid 구간 점수를 그대로 쓴다."""
    write_history(tmp_path)
    write_profiles(tmp_path, month_rows=[(222, 3, 7, 900.0, 30)],
                   weekday_rows=[(222, 3, 4, 800.0, 12)])
    metrics = tmp_path / "metrics_calendar.json"
    metrics.write_text(json.dumps({
        "features": FEATS,
        "train_end": "2026-02-14",
        "valid": {
            "A": {"mae": 800.0, "rmse": 1100.0, "n": 100,
                  "horizon": {"1-31": {"mae": 750.0, "rmse": 1000.0, "n": 60}}},
        },
    }), encoding="utf-8")
    p = make_predictor(tmp_path, metrics_path=metrics)
    p._models = {"A": Capture()}

    out = p.predict(values(date="2026-07-05"))
    assert out["horizon_scores"]["A"].mae == pytest.approx(750.0)
    assert p.scores()["A"].mae == pytest.approx(800.0)
    assert p.horizon_scores("2026-07-05")["A"].rmse == pytest.approx(1000.0)


def test_backtest_horizon_block_is_read(tmp_path):
    """백테스트가 넣는 최상위 horizon 블록(182-366 근사값)도 읽는다."""
    write_history(tmp_path)
    write_profiles(tmp_path, month_rows=[(222, 3, 12, 900.0, 30)],
                   weekday_rows=[(222, 3, 1, 800.0, 12)])
    metrics = tmp_path / "metrics_calendar.json"
    metrics.write_text(json.dumps({
        "features": FEATS,
        "horizon": {"182-366": {"A": {"mae": 1200.0, "rmse": 1800.0, "n": 5}}},
        "valid": {},
    }), encoding="utf-8")
    p = make_predictor(tmp_path, metrics_path=metrics)
    p._models = {"A": Capture()}

    out = p.predict(values(date="2026-12-29"))  # 데이터 끝 + 182일
    assert out["horizon_bucket"] == "182-366"
    assert out["horizon_scores"]["A"].mae == pytest.approx(1200.0)
    assert out["horizon_scores"]["A"].rmse == pytest.approx(1800.0)


def test_stale_cache_is_ignored(tmp_path):
    """원본과 맞지 않는 캐시(버전·시각)는 버리고 CSV 를 읽는다."""
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
