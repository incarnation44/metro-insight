"""집계·이상치 테스트: 가짜 예측기(배열 3개)로 계산 규칙만 검증한다.

models/ 아티팩트를 읽지 않는다. NaN 제외·커버리지·필터·경계가 계약이다.
"""
import numpy as np
import pytest

from metro_insight.model.analytics import Analytics

HOURS_PER_DAY = 20
DATES = ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")  # 월~목


class FakePredictor:
    """예측기의 이력 배열 인터페이스만 흉내 내는 최소 객체."""

    def __init__(self, history, dates, codes, pairs, lines=None):
        self._history = history
        self._dates = list(dates)
        self._station_index = {code: i for i, code in enumerate(codes)}
        self._station_code = dict(pairs)
        self._line_code = dict(lines or {})

    def has_data(self, line: str, station: str) -> bool:
        return (line, station) in self._station_code

    def _lookup(self, line: str, station: str):
        return self._station_code.get((line, station)), self._line_code.get(line)


@pytest.fixture()
def fake():
    history = np.full((2, HOURS_PER_DAY, len(DATES)), np.nan, dtype=np.float32)
    history[0, 0, :] = [10, 20, 30, 40]  # 잠실 · hour_code 0 (04~05시)
    history[0, 1, :] = [1, 2, 3, 4]  # 잠실 · hour_code 1 (06시)
    history[1, 0, :] = [100, 200, 300, 400]  # 강남 · hour_code 0
    pairs = {("2호선", "잠실"): 1, ("2호선", "강남"): 2}
    return FakePredictor(history, DATES, [1, 2], pairs, {"2호선": 2})


def test_series_hour_slice_and_daily_total(fake):
    analytics = Analytics(fake)
    dates, values = analytics.series("2호선", "잠실", hour=4)
    assert list(dates) == list(DATES)
    assert values.tolist() == [10.0, 20.0, 30.0, 40.0]

    _, totals = analytics.series("2호선", "잠실")
    assert totals.tolist() == [11.0, 22.0, 33.0, 44.0]


def test_aggregate_sum_mean_max(fake):
    analytics = Analytics(fake)
    kwargs = {"line": "2호선", "station": "잠실", "date_from": "2026-01-05", "date_to": "2026-01-08"}

    total = analytics.aggregate(hour=4, metric="sum", **kwargs)
    assert total == {"value": 100.0, "n": 4, "coverage": 1.0}

    mean = analytics.aggregate(metric="mean", **kwargs)
    assert mean["value"] == pytest.approx(13.75)  # (100 + 10) / 8
    assert mean["n"] == 8
    assert mean["coverage"] == pytest.approx(8 / 80)

    biggest = analytics.aggregate(metric="max", **kwargs)
    assert biggest["value"] == 40.0


def test_hour_filter(fake):
    analytics = Analytics(fake)
    result = analytics.aggregate(
        line="2호선",
        station="잠실",
        hour=6,
        date_from="2026-01-05",
        date_to="2026-01-08",
    )
    assert result["value"] == 10.0
    assert result["n"] == 4
    assert result["coverage"] == 1.0


def test_weekday_filter(fake):
    analytics = Analytics(fake)
    result = analytics.aggregate(
        line="2호선",
        station="잠실",
        hour=4,
        weekday=0,  # 2026-01-05 월요일만
        date_from="2026-01-05",
        date_to="2026-01-08",
    )
    assert result == {"value": 10.0, "n": 1, "coverage": 1.0}


def test_coverage_counts_missing_cells(fake):
    analytics = Analytics(fake)
    # 2행 × 1시간대 × 4일 = 8칸 기대, 강남 hour_code 1 은 전부 NaN → 4칸만 유효
    result = analytics.aggregate(
        line="2호선", hour=6, date_from="2026-01-05", date_to="2026-01-08"
    )
    assert result["value"] == 10.0
    assert result["n"] == 4
    assert result["coverage"] == pytest.approx(0.5)


def test_nan_is_excluded_not_zero(fake):
    analytics = Analytics(fake)
    _, values = analytics.series("2호선", "강남")
    assert values.tolist() == [100.0, 200.0, 300.0, 400.0]
    # 값이 없는 시간대만 고르면 0 이 아니라 NaN
    _, empty = analytics.series("2호선", "강남", hour=6)
    assert np.isnan(empty).all()


def test_series_preserves_nan():
    dates = ("2026-05-04", "2026-05-05", "2026-05-06")
    history = np.full((1, HOURS_PER_DAY, len(dates)), np.nan, dtype=np.float32)
    history[0, 0, :] = [10.0, np.nan, 30.0]
    predictor = FakePredictor(history, dates, [1], {("2호선", "잠실"): 1}, {"2호선": 2})
    analytics = Analytics(predictor)

    _, values = analytics.series("2호선", "잠실", hour=4)
    assert values.tolist()[0] == 10.0
    assert np.isnan(values[1])  # 결측을 0 으로 채우지 않는다
    assert values.tolist()[2] == 30.0

    _, daily = analytics.daily_series(line="2호선", hour=4)
    assert np.isnan(daily[1])
    assert int(np.count_nonzero(np.isnan(daily))) == 1


def test_empty_window_reports_nan_and_zero_coverage(fake):
    analytics = Analytics(fake)
    result = analytics.aggregate(
        line="2호선", station="잠실", date_from="2027-01-01", date_to="2027-01-02"
    )
    assert np.isnan(result["value"])
    assert result["n"] == 0
    assert result["coverage"] == 0.0


def test_daily_series_metrics(fake):
    analytics = Analytics(fake)
    dates, totals = analytics.daily_series(line="2호선", hour=4)
    assert list(dates) == list(DATES)
    assert totals.tolist() == [110.0, 220.0, 330.0, 440.0]

    _, means = analytics.daily_series(line="2호선", hour=4, metric="mean")
    assert means.tolist() == [55.0, 110.0, 165.0, 220.0]

    _, biggest = analytics.daily_series(line="2호선", hour=4, metric="max")
    assert biggest.tolist() == [100.0, 200.0, 300.0, 400.0]


def test_unknown_station_and_bad_metric_raise(fake):
    analytics = Analytics(fake)
    with pytest.raises(ValueError):
        analytics.series("2호선", "없는역")
    with pytest.raises(ValueError):
        analytics.aggregate(
            line="2호선", date_from="2026-01-05", date_to="2026-01-08", metric="median"
        )
    with pytest.raises(ValueError):
        analytics.aggregate(line="2호선", date_from="2026-01-08", date_to="2026-01-05")
    with pytest.raises(ValueError):
        analytics.series("2호선", "잠실", hour=25)


def test_outliers_z_boundary_is_strict():
    dates = [f"2026-03-{day:02d}" for day in range(1, 11)]
    history = np.full((1, HOURS_PER_DAY, len(dates)), np.nan, dtype=np.float32)
    history[0, 0, :] = [100.0] * 9 + [1000.0]
    predictor = FakePredictor(history, dates, [1], {("2호선", "잠실"): 1}, {"2호선": 2})
    analytics = Analytics(predictor)

    values = np.array([100.0] * 9 + [1000.0])
    top = float(np.abs((values - values.mean()) / values.std()).max())
    assert top == pytest.approx(3.0)

    common = {"line": "2호선", "date_from": dates[0], "date_to": dates[-1]}
    # 경계(정확히 top)는 이상치가 아니다
    assert analytics.outliers(z=top, **common) == []
    found = analytics.outliers(z=np.nextafter(top, 0.0), **common)
    assert len(found) == 1
    assert found[0]["date"] == "2026-03-10"
    assert found[0]["value"] == 1000.0
    assert found[0]["z"] == pytest.approx(top)


def test_outliers_skips_nan_and_short_windows():
    dates = [f"2026-04-{day:02d}" for day in range(1, 9)]
    history = np.full((1, HOURS_PER_DAY, len(dates)), np.nan, dtype=np.float32)
    history[0, 0, 0] = 50.0
    predictor = FakePredictor(history, dates, [1], {("2호선", "잠실"): 1}, {"2호선": 2})
    analytics = Analytics(predictor)
    assert analytics.outliers(date_from=dates[0], date_to=dates[-1]) == []