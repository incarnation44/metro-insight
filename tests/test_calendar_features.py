"""달력 피처 테스트 — 음력 표·연휴 맥락·기존 함수 시그니처 보존.

실제 data/processed 는 열지 않는다. 날짜만 넣고 표와 규칙을 검증한다.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT / "notebooks", ROOT / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import calendar_features as cf  # noqa: E402


def _ctx(*days: str) -> pd.DataFrame:
    return cf.holiday_context(pd.Series(pd.to_datetime(list(days))))


# ───────────────────────── 음력 표 (수용 테스트) ─────────────────────────
def test_lunar_table_covers_2024_2027() -> None:
    """연도 4개, 해마다 설과 추석이 각각 있어야 한다."""
    assert sorted(cf.LUNAR_HOLIDAYS) == [2024, 2025, 2026, 2027]
    for year, entries in cf.LUNAR_HOLIDAYS.items():
        assert len(entries) >= 2, f"{year} 항목이 {len(entries)}개다"
        stamps = [pd.Timestamp(text) for text in entries]
        assert all(stamp.year == year for stamp in stamps)
        seollal = [s for s in stamps if s.month in (1, 2)]
        chuseok = [s for s in stamps if s.month in (9, 10)]
        assert seollal, f"{year} 설날이 없다"
        assert chuseok, f"{year} 추석이 없다"
        assert stamps == sorted(stamps), f"{year} 항목은 설날·추석 순서여야 한다"


# ───────────────────────── 맥락 피처 ─────────────────────────
def test_holiday_context_columns_and_index() -> None:
    dates = pd.Series(pd.to_datetime(["2024-02-10", "2025-07-15"]), index=[7, 9])
    table = cf.holiday_context(dates)
    expected = ["run_len", "pos_in_run", "seollal_window", "chuseok_window",
                "substitute", "vacation", "exam_day", "fixed_day", "ctx_is_holiday"]
    assert list(table.columns) == expected
    assert list(table.index) == [7, 9]
    for column in expected:
        assert pd.api.types.is_integer_dtype(table[column]), column


def test_run_length_and_position() -> None:
    """2024 설 연휴(2/9~2/12, 4일)의 길이와 위치."""
    table = _ctx("2024-02-08", "2024-02-09", "2024-02-10", "2024-02-12", "2024-02-13")
    assert table["run_len"].tolist() == [0, 4, 4, 4, 0]
    assert table["pos_in_run"].tolist() == [-1, 0, 0, 0, 1]
    assert table["seollal_window"].tolist() == [1, 1, 1, 1, 1]
    assert table["ctx_is_holiday"].tolist() == [0, 1, 1, 1, 0]
    outside = _ctx("2024-03-20")
    assert int(outside["pos_in_run"].iloc[0]) == 3  # 3일을 넘으면 ±3 으로 포화(부호는 공휴일 이후 +)


def test_run_length_is_batch_invariant() -> None:
    """한 날짜만 넣어도 여러 날짜를 넣어도 같은 연휴 길이가 나와야 한다."""
    single = _ctx("2024-02-12")["run_len"].iloc[0]
    batch = _ctx("2024-01-10", "2024-02-12", "2024-06-01")["run_len"].iloc[0:3]
    assert int(single) == 4
    assert batch.tolist() == [0, 4, 0]


def test_substitute_holiday_rule() -> None:
    """주말에 걸친 공휴일 다음 평일을 대체공휴일로 근사한다(2024-05-06)."""
    table = _ctx("2024-05-05", "2024-05-06", "2024-05-07")
    assert table["fixed_day"].tolist() == [1, 0, 0]
    assert table["substitute"].tolist() == [0, 1, 0]
    assert table["ctx_is_holiday"].tolist() == [1, 1, 0]
    # 2025-05-06 은 부처님오신날과 겹친 실제 대체공휴일이라 규칙으로 잡히지 않는다(표에 없는 날)
    assert int(_ctx("2025-05-06")["substitute"].iloc[0]) == 0


def test_lunar_windows_do_not_cross_years() -> None:
    table = _ctx("2024-09-17", "2025-10-06", "2026-02-17")
    assert table["chuseok_window"].tolist() == [1, 1, 0]
    assert table["seollal_window"].tolist() == [0, 0, 1]


def test_vacation_exam_and_fixed_days() -> None:
    table = _ctx("2025-01-15", "2025-07-15", "2025-11-15", "2024-11-14", "2025-12-25")
    assert table["vacation"].tolist() == [1, 1, 0, 0, 0]
    assert table["exam_day"].tolist() == [0, 0, 0, 1, 0]
    assert table["fixed_day"].tolist() == [0, 0, 0, 0, 1]


def test_calendar_feature_frame_contract() -> None:
    dates = pd.Series(pd.to_datetime(["2024-02-10", "2024-05-06"]))
    frame = cf.calendar_feature_frame(dates)
    assert list(frame.columns) == cf.CALENDAR_FEATURE_COLS
    assert frame.notna().all().all()
    assert frame.index.equals(dates.index)
    for column in cf.CALENDAR_FEATURE_COLS:
        assert pd.api.types.is_integer_dtype(frame[column]), column


# ───────────────────────── 공휴일 탄력성 (as-of) ─────────────────────────
def _elastic_frame(
    n_days: int = 42,
    station_scale: dict[int, float] | None = None,
    station_base: dict[int, float] | None = None,
    holiday_weekday: int = 3,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """(역, 시간대 0) 하루 1행짜리 합성 프레임. 공휴일은 7일마다 하루."""
    scale = {1: 0.25, 2: 0.90} if station_scale is None else station_scale
    base = {s: 100.0 * s for s in scale} if station_base is None else station_base
    dates = pd.date_range(start, periods=n_days, freq="D")
    records = []
    for i, day in enumerate(dates):
        is_holiday = int(i % 7 == holiday_weekday)
        for station, factor in scale.items():
            value = base[station] * (factor if is_holiday else 1.0)
            records.append((day, station, 0, value, is_holiday))
    return pd.DataFrame(records, columns=["date", "station_code", "hour_code", "boarding", "is_holiday"])


def test_asof_holiday_ratio_columns_and_contract() -> None:
    df = _elastic_frame()
    out = cf.asof_holiday_elasticity(df)
    assert list(out.columns) == cf.PROFILE_COLS_HOLIDAY
    assert out.index.equals(df.index)
    assert out["profile_holiday_ratio"].dtype == np.float32
    assert out["profile_holiday_n"].dtype == np.int32
    assert np.isfinite(out["profile_holiday_ratio"]).all()  # NaN 대신 1.0 폴백
    assert (out["profile_holiday_n"].to_numpy() >= 0).all()


def test_asof_holiday_ratio_has_no_same_day_or_future_leak() -> None:
    """당일·미래 행의 boarding 을 100배로 바꿔도 그 날짜까지의 배율은 그대로여야 한다."""
    df = _elastic_frame()
    cutoff = pd.Timestamp("2024-01-20")
    base = cf.asof_holiday_elasticity(df)["profile_holiday_ratio"]

    touched = df.copy()
    future = (touched["date"] >= cutoff).to_numpy()
    assert future.any()
    touched.loc[future, "boarding"] = touched.loc[future, "boarding"] * 100.0
    after = cf.asof_holiday_elasticity(touched)["profile_holiday_ratio"]

    keep = (df["date"] <= cutoff).to_numpy()  # 경계 당일 포함
    pd.testing.assert_series_equal(base[keep], after[keep])

    # 반대로 과거를 바꾸면 이후 배율은 바뀌어야 한다(누수 검사가 비어 있지 않다는 확인)
    past = df.copy()
    changed = (past["date"] < cutoff).to_numpy()
    past.loc[changed, "boarding"] = past.loc[changed, "boarding"] * 0.5
    moved = cf.asof_holiday_elasticity(past)["profile_holiday_ratio"]
    assert not np.allclose(base[~keep].to_numpy(), moved[~keep].to_numpy())


def test_asof_holiday_ratio_separates_station_elasticity() -> None:
    """공휴일 하락이 큰 역(0.25배)과 작은 역(0.90배)이 배율로 구분돼야 한다."""
    df = _elastic_frame()
    out = cf.asof_holiday_elasticity(df)
    last_date = df["date"].max()
    last = out[(df["date"] == last_date).to_numpy()]
    ratio = last["profile_holiday_ratio"].to_numpy()
    assert ratio[0] < 0.55, ratio            # 업무지구(0.25배) — 사전분포 쪽으로 당겨져도 낮다
    assert ratio[1] > 0.70, ratio            # 관광지(0.90배)
    assert ratio[1] - ratio[0] > 0.15, ratio
    assert (last["profile_holiday_n"].to_numpy() > 0).all()


def test_asof_holiday_ratio_prior_and_zero_division() -> None:
    """hol_n == 0 이면 사전분포(global_ratio)가 나오고, nonhol_mean == 0 은 0 나눗셈을 막는다."""
    dates = pd.date_range("2024-03-01", periods=10, freq="D")
    rows = []
    for i, day in enumerate(dates):
        is_holiday = int(i in (3, 7))
        rows.append((day, 1, 0, 50.0 if is_holiday else 100.0, is_holiday))
        rows.append((day, 2, 0, 200.0, 0))  # 공휴일 표본이 없는 역
    df = pd.DataFrame(rows, columns=["date", "station_code", "hour_code", "boarding", "is_holiday"])

    # 7일째(=두 번째 공휴일) 이전: 역1 공휴일 1건(50), 비공휴일 6건(100×6) + 역2 비공휴일 7건(200×7)
    global_ratio = 50.0 / ((6 * 100.0 + 7 * 200.0) / 13.0)
    target = (df["date"] == dates[7]).to_numpy()

    for prior_weight in (1.0, 5.0):
        out = cf.asof_holiday_elasticity(df, prior_weight=prior_weight)
        got = out.loc[target, "profile_holiday_ratio"].to_numpy()
        expected_s1 = (1 * 0.5 + prior_weight * global_ratio) / (1 + prior_weight)
        assert got[0] == pytest.approx(expected_s1, rel=1e-5)     # 표본 1건 + 사전분포 평활
        assert got[1] == pytest.approx(global_ratio, rel=1e-5)    # hol_n == 0 → 사전분포 그대로
        n = out.loc[target, "profile_holiday_n"].to_numpy()
        assert n.tolist() == [1, 0]

    # 첫 공휴일은 이전 관측이 없으므로 n=0, 배율 1.0(모른다)
    first = cf.asof_holiday_elasticity(df)
    first_row = (df["date"] == dates[3]).to_numpy()
    assert first.loc[first_row, "profile_holiday_n"].tolist() == [0, 0]
    assert first.loc[first_row, "profile_holiday_ratio"].tolist() == [1.0, 1.0]

    # nonhol_mean == 0: 역3 은 비공휴일 0, 공휴일 5 → raw 는 무한대가 아니라 사전분포로 대체
    zero = pd.DataFrame(
        [(day, 3, 0, 5.0 if i in (3, 7) else 0.0, int(i in (3, 7))) for i, day in enumerate(dates)]
        + rows,
        columns=["date", "station_code", "hour_code", "boarding", "is_holiday"],
    )
    out = cf.asof_holiday_elasticity(zero)
    ratio = out["profile_holiday_ratio"].to_numpy()
    assert np.isfinite(ratio).all(), ratio
    g2 = (55.0 / 2.0) / ((6 * 100.0 + 7 * 200.0 + 0.0) / 19.0)  # 역3 의 0 값 6건 포함 글로벌
    same_day = (zero["station_code"] == 3).to_numpy() & (zero["date"] == dates[7]).to_numpy()
    assert out.loc[same_day, "profile_holiday_ratio"].iloc[0] == pytest.approx(g2, rel=1e-5)


def test_frozen_holiday_elasticity_uses_only_before_cut() -> None:
    df = _elastic_frame()
    cut = pd.Timestamp("2024-01-25")
    table = cf.frozen_holiday_elasticity(df, cut)
    assert list(table.columns) == [*cf.KEY_SH, *cf.PROFILE_COLS_HOLIDAY]

    at_or_after = (df["date"] >= cut).to_numpy()
    assert at_or_after.any()
    later = df.copy()
    later.loc[at_or_after, "boarding"] = later.loc[at_or_after, "boarding"] * 100.0
    pd.testing.assert_frame_equal(table, cf.frozen_holiday_elasticity(later, cut))

    earlier = df.copy()
    before = (df["date"] < cut).to_numpy()
    past_holidays = before & (earlier["is_holiday"] == 1).to_numpy()
    assert past_holidays.any()
    earlier.loc[past_holidays, "boarding"] = earlier.loc[past_holidays, "boarding"] * 3.0  # 배율만 흔든다
    assert not np.allclose(table["profile_holiday_ratio"].to_numpy(),
                           cf.frozen_holiday_elasticity(earlier, cut)["profile_holiday_ratio"].to_numpy())

    # cut 당일의 공휴일은 표본에 들어가지 않는다
    only_cut_day = df.copy()
    only_cut_day["is_holiday"] = (only_cut_day["date"] == cut).astype(int)
    only_cut_day.loc[only_cut_day["date"] == cut, "boarding"] = 1.0
    cut_table = cf.frozen_holiday_elasticity(only_cut_day, cut)
    assert cut_table["profile_holiday_n"].sum() == 0
    assert (cut_table["profile_holiday_ratio"] == np.float32(1.0)).all()

    # target 매핑: 표에 없는 키는 글로벌 폴백, n 은 0
    target = df[df["date"] >= cut].head(3).copy()
    mapped = cf.frozen_holiday_elasticity(df, cut, target=target)
    assert list(mapped.columns) == cf.PROFILE_COLS_HOLIDAY
    assert mapped.index.equals(target.index)
    assert np.isfinite(mapped["profile_holiday_ratio"]).all()

    unknown = target.copy()
    unknown.loc[unknown.index[0], "station_code"] = 999
    unknown_mapped = cf.frozen_holiday_elasticity(df, cut, target=unknown)
    assert unknown_mapped.loc[unknown.index[0], "profile_holiday_n"] == 0
    assert np.isfinite(unknown_mapped.loc[unknown.index[0], "profile_holiday_ratio"])


def test_asof_profile_frame_holiday_is_opt_in() -> None:
    """기본 6컬럼 계약은 그대로, include_holiday=True 일 때만 2컬럼이 붙는다."""
    df = _elastic_frame()
    df = cf.add_month(df)
    df["weekday"] = df["date"].dt.weekday.astype("int8")
    plain = cf.asof_profile_frame(df)
    assert len(plain.columns) == 6
    assert list(plain.columns) == [*cf.PROFILE_COLS_SH, *cf.PROFILE_COLS_MONTH,
                                   *cf.PROFILE_COLS_WEEKDAY]

    extended = cf.asof_profile_frame(df, include_holiday=True)
    assert list(extended.columns) == [*plain.columns, *cf.PROFILE_COLS_HOLIDAY]
    pd.testing.assert_frame_equal(extended[plain.columns], plain)


# ───────────────────────── 기존 계약 보존 ─────────────────────────
def test_existing_signatures_unchanged() -> None:
    """기존 함수의 매개변수 이름·순서가 그대로여야 한다(호출부·서빙 계약 보존)."""
    expected = {
        "load_processed": ["path", "columns"],
        "add_month": ["df"],
        "asof_profile": ["df", "key_cols", "value", "mean_col", "n_col", "recency"],
        "asof_profile_frame": ["df", "value", "recency", "include_holiday"],
        "frozen_profile": ["df", "cut", "key_cols", "value", "mean_col", "n_col", "recency"],
        "frozen_profile_frame": ["df_fit", "df_target", "cut", "value", "recency"],
        "lookup": ["profile", "df", "key_cols", "col"],
        "horizon_bucket": ["days"],
        "save_profile_npz": ["path", "df", "value", "recency"],
    }
    for name, params in expected.items():
        got = list(inspect.signature(getattr(cf, name)).parameters)
        assert got == params, f"{name} 매개변수가 바뀌었다: {got}"
    assert cf.HORIZON_BUCKETS == [(1, 31), (32, 91), (92, 181), (182, 366)]
    assert cf.PROFILE_COLS_MONTH == ["profile_month_mean", "profile_n"]
    assert cf.PROFILE_COLS_SH == ["profile_sh_mean", "profile_sh_n"]


def test_horizon_bucket_labels_unchanged() -> None:
    assert cf.horizon_bucket(np.array([1, 31, 32, 366, 400])).tolist() == [
        "1-31", "1-31", "32-91", "182-366", "other",
    ]


def test_empty_input_returns_empty_frame() -> None:
    table = cf.holiday_context(pd.Series([], dtype="datetime64[ns]"))
    assert table.empty
    assert "ctx_is_holiday" in table.columns


def test_unknown_year_falls_back_to_fixed_days() -> None:
    """표에 없는 연도(2029)도 양력 고정 공휴일은 계산한다(음력은 비어 있음)."""
    table = _ctx("2029-12-25", "2029-02-10")
    assert table["fixed_day"].tolist() == [1, 0]
    assert table["seollal_window"].tolist() == [0, 0]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))