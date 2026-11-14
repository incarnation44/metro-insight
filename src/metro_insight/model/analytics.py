"""집계·이상치 계산: 예측기가 들고 있는 이력 배열을 **읽기만** 해서 요약한다.

예측기(:class:`metro_insight.model.predictor.ABCPredictor`)는 학습 데이터를
``(역, 시간대 20, 날짜)`` 3차원 배열로 들고 있다. 이 모듈은 그 배열을 그대로
재사용하고, 필요한 행·시간대·날짜만 잘라 계산한다 (전체 스캔 회피).

정직성 규칙
* ``aggregate`` 는 NaN(관측 없음)을 집계에서 **빼고**, 유효값 수 ``n`` 과 커버리지
  ``coverage`` 로 얼마나 비었는지 함께 보고한다.
* ``series``·``daily_series`` 는 진단 필드를 돌려주지 않는다. 전부 비어 있는 날짜는
  NaN 으로 남기고(0 으로 채우지 않는다), 일부 셀만 비면 있는 값으로 계산한다.
  결측 비율이 필요하면 ``aggregate`` 의 ``n``·``coverage`` 를 쓴다.
* 유효값이 하나도 없으면 값을 NaN 으로 돌려준다 (0 으로 속이지 않는다).
* 역·호선 해석은 예측기의 기존 별칭 규칙을 재사용한다 (새 별칭 표를 만들지 않는다).
"""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from datetime import date as _date
from typing import Literal

import numpy as np

from metro_insight.model.predictor import HOURS_PER_DAY, TIME_CODE_BY_HOUR

Metric = Literal["sum", "mean", "max"]

_METRICS = ("sum", "mean", "max")
_PARENS = re.compile(r"\(.*?\)")


def _base_name(name: str) -> str:
    """'잠실(송파구청)' → '잠실'. 예측기 별칭 규칙과 같은 정규화(표만 새로 만들지 않는다)."""
    return _PARENS.sub("", str(name)).strip()


def _iso(value: str) -> str:
    """'2026-01-05' 형태로 정규화한다. 형식이 아니면 ValueError."""
    try:
        return _date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"날짜 형식이 아닙니다: {value}") from exc


class Analytics:
    """예측기의 이력 배열 위에서 시계열·집계·이상치를 계산한다."""

    def __init__(self, predictor) -> None:
        self._predictor = predictor

    # ---------- 내부: 배열 접근 ----------

    @property
    def _history(self) -> np.ndarray | None:
        return getattr(self._predictor, "_history", None)

    @property
    def _dates(self) -> list[str]:
        return list(getattr(self._predictor, "_dates", []) or [])

    def _lookup(self, line: str, station: str) -> int | None:
        """(호선, 역) → station_code. 예측기의 ``_lookup``·별칭을 그대로 쓴다."""
        lookup = getattr(self._predictor, "_lookup", None)
        if callable(lookup):
            try:
                code, _ = lookup(str(line), str(station))
            except (KeyError, TypeError, ValueError):
                return None
            return int(code) if code is not None else None
        pairs: dict = getattr(self._predictor, "_station_code", {}) or {}
        code = pairs.get((str(line), str(station)))
        if code is None:
            wanted = _base_name(station)
            for (pair_line, pair_station), pair_code in pairs.items():
                if pair_line == line and _base_name(pair_station) == wanted:
                    code = pair_code
                    break
        return int(code) if code is not None else None

    def _rows(self, line: str | None, station: str | None) -> np.ndarray:
        """조건에 맞는 이력 행 번호. 호선만·역만·둘 다·전체를 모두 받는다."""
        index: dict = getattr(self._predictor, "_station_index", {}) or {}
        pairs: dict = getattr(self._predictor, "_station_code", {}) or {}
        codes: set[int] = set()

        if line is not None and station is not None:
            code = self._lookup(str(line), str(station))
            if code is not None:
                codes.add(code)
        elif station is not None:
            wanted = _base_name(station)
            for (_pair_line, pair_station), pair_code in pairs.items():
                if _base_name(pair_station) == wanted:
                    codes.add(int(pair_code))
        elif line is not None:
            for (pair_line, _pair_station), pair_code in pairs.items():
                if pair_line == line:
                    codes.add(int(pair_code))
        else:
            codes.update(int(c) for c in index)

        rows = sorted(index[code] for code in codes if code in index)
        if not rows and (line is not None or station is not None):
            who = f"{line or ''} {station or ''}".strip()
            raise ValueError(f"학습 데이터에 없는 역입니다: {who}")
        return np.asarray(rows, dtype=np.int64)

    def _hour_buckets(self, hour: int | None) -> np.ndarray:
        """시간(0~24) → 이력 시간대 열 번호. None 이면 20개 전부."""
        if hour is None:
            return np.arange(HOURS_PER_DAY, dtype=np.int64)
        try:
            value = int(hour)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"hour 는 정수여야 합니다: {hour}") from exc
        if not 0 <= value <= 24:
            raise ValueError(f"hour 는 0~24 사이여야 합니다: {hour}")
        return np.asarray([TIME_CODE_BY_HOUR[value]], dtype=np.int64)

    def _window(self, date_from: str, date_to: str) -> tuple[list[str], np.ndarray]:
        """[date_from, date_to] 날짜 슬라이스. 이력에 있는 날짜만 돌려준다."""
        start, end = _iso(date_from), _iso(date_to)
        if start > end:
            raise ValueError(f"date_from({start}) 이 date_to({end}) 보다 늦습니다.")
        dates = self._dates
        lo = bisect_left(dates, start)
        hi = bisect_right(dates, end)
        return dates[lo:hi], np.arange(lo, hi, dtype=np.int64)

    def _weekday_filter(self, dates: list[str], cols: np.ndarray, weekday: int | None) -> np.ndarray:
        """요일(0=월) 조건으로 날짜 열을 거른다. None 이면 그대로."""
        if weekday is None:
            return cols
        try:
            wanted = int(weekday)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weekday 는 정수여야 합니다: {weekday}") from exc
        if not 0 <= wanted <= 6:
            raise ValueError(f"weekday 는 0~6 사이여야 합니다: {weekday}")
        keep = [
            col
            for day, col in zip(dates, cols, strict=True)
            if _date.fromisoformat(day[:10]).weekday() == wanted
        ]
        return np.asarray(keep, dtype=np.int64)

    def _slice(self, rows: np.ndarray, hours: np.ndarray, cols: np.ndarray) -> np.ndarray | None:
        """필요한 (행, 시간대, 날짜) 블록만 잘라 float64 로 돌려준다. 이력이 없으면 None."""
        history = self._history
        if history is None:
            return None
        shape = (int(rows.size), int(hours.size), int(cols.size))
        if 0 in shape:
            return np.empty(shape, dtype=np.float64)
        block = history[np.ix_(rows, hours, cols)]
        return np.asarray(block, dtype=np.float64)

    @staticmethod
    def _check_metric(metric: str) -> None:
        if metric not in _METRICS:
            raise ValueError(f"metric 은 {_METRICS} 중 하나여야 합니다: {metric}")

    def _reduce(self, block: np.ndarray, metric: str) -> np.ndarray:
        """(행, 시간대, 날짜) → 날짜별 1차원. NaN 은 빼고, 전부 NaN 인 날은 NaN."""
        n_dates = block.shape[-1]
        flat = block.reshape(-1, n_dates) if block.size else np.empty((0, n_dates), dtype=np.float64)
        counts = np.count_nonzero(~np.isnan(flat), axis=0)
        if metric == "sum":
            values = np.nansum(flat, axis=0) if flat.size else np.zeros(n_dates)
        elif metric == "mean":
            totals = np.nansum(flat, axis=0) if flat.size else np.zeros(n_dates)
            values = np.zeros(n_dates, dtype=np.float64)
            np.divide(totals, counts, out=values, where=counts > 0)
        else:
            filled = np.where(np.isnan(flat), -np.inf, flat) if flat.size else flat
            values = filled.max(axis=0) if flat.size else np.zeros(n_dates)
        return np.where(counts > 0, values, np.nan).astype(np.float64, copy=False)

    # ---------- 공개 API ----------

    def series(
        self, line: str, station: str, hour: int | None = None
    ) -> tuple[list[str], np.ndarray]:
        """한 역의 날짜별 값. ``hour`` 가 None 이면 하루 합계, 지정하면 그 시간대 값.

        원자료 시계열을 그대로 돌려준다 — 전부 비어 있는 날짜는 NaN 으로 남기고
        0 으로 채우지 않는다. 일부 셀만 비면 있는 값으로 계산하므로 **결측 비율**은
        돌려주지 않는다. 결측 비율이 필요하면 ``aggregate`` 의 ``n``·``coverage`` 를 쓴다.
        """
        rows = self._rows(str(line), str(station))
        hours = self._hour_buckets(hour)
        dates = self._dates
        cols = np.arange(len(dates), dtype=np.int64)
        block = self._slice(rows, hours, cols)
        if block is None:
            return dates, np.full(len(dates), np.nan, dtype=np.float64)
        return dates, self._reduce(block, "sum")

    def aggregate(
        self,
        *,
        line: str | None = None,
        station: str | None = None,
        hour: int | None = None,
        weekday: int | None = None,
        date_from: str,
        date_to: str,
        metric: Metric = "sum",
    ) -> dict:
        """조건에 맞는 셀들의 합계·평균·최대 + 유효값 수 + 커버리지."""
        self._check_metric(metric)
        rows = self._rows(line, station)
        hours = self._hour_buckets(hour)
        dates, cols = self._window(date_from, date_to)
        cols = self._weekday_filter(dates, cols, weekday)
        expected = int(rows.size) * int(hours.size) * int(cols.size)
        block = self._slice(rows, hours, cols)
        if block is None or expected == 0:
            return {"value": float("nan"), "n": 0, "coverage": 0.0}
        valid = int(np.count_nonzero(~np.isnan(block)))
        if valid == 0:
            value = float("nan")
        elif metric == "sum":
            value = float(np.nansum(block))
        elif metric == "mean":
            value = float(np.nanmean(block))
        else:
            value = float(np.nanmax(block))
        return {"value": value, "n": valid, "coverage": valid / expected}

    def daily_series(
        self,
        *,
        line: str | None = None,
        hour: int | None = None,
        metric: Metric = "sum",
    ) -> tuple[list[str], np.ndarray]:
        """전체 기간의 날짜별 집계 (행·시간대를 접는다).

        ``series`` 와 같은 규칙이다 — 전부 비어 있는 날짜는 NaN 으로 남기고, 일부 셀만
        비면 있는 값으로 계산한다. 결측 비율은 ``aggregate`` 의 ``n``·``coverage`` 로 본다.
        """
        self._check_metric(metric)
        rows = self._rows(line, None)
        hours = self._hour_buckets(hour)
        dates = self._dates
        cols = np.arange(len(dates), dtype=np.int64)
        block = self._slice(rows, hours, cols)
        if block is None:
            return dates, np.full(len(dates), np.nan, dtype=np.float64)
        return dates, self._reduce(block, metric)

    def outliers(
        self,
        *,
        line: str | None = None,
        station: str | None = None,
        hour: int | None = None,
        date_from: str,
        date_to: str,
        z: float = 3.0,
    ) -> list[dict]:
        """날짜별 합계가 평균에서 표준편차 ``z`` 배를 넘는 날. 경계(==z)는 넣지 않는다."""
        rows = self._rows(line, station)
        hours = self._hour_buckets(hour)
        dates, cols = self._window(date_from, date_to)
        block = self._slice(rows, hours, cols)
        if block is None or not dates:
            return []
        values = self._reduce(block, "sum")
        sample = values[~np.isnan(values)]
        if sample.size < 2:
            return []
        mean = float(sample.mean())
        std = float(sample.std())
        if not np.isfinite(std) or std == 0.0:
            return []
        threshold = float(z)
        found: list[dict] = []
        for day, value in zip(dates, values, strict=True):
            if np.isnan(value):
                continue
            score = (float(value) - mean) / std
            if abs(score) > threshold:
                found.append(
                    {"date": day, "value": float(value), "z": score, "mean": mean, "std": std}
                )
        return found