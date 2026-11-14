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


# ───────────────────────── 기존 계약 보존 ─────────────────────────
def test_existing_signatures_unchanged() -> None:
    """기존 함수의 매개변수 이름·순서가 그대로여야 한다(호출부·서빙 계약 보존)."""
    expected = {
        "load_processed": ["path", "columns"],
        "add_month": ["df"],
        "asof_profile": ["df", "key_cols", "value", "mean_col", "n_col", "recency"],
        "asof_profile_frame": ["df", "value", "recency"],
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