"""평가 프로토콜 — 롤링 8개 기준일(cut) · 고정 피처·파라미터 · 산출물 명세(manifest) 동결.

실행 (프로젝트 루트에서)
    python notebooks/eval_protocol.py --protocol rolling8-v1 --model B \\
        --feature-set base --out models/_exp/20261114-P0-R0-01
    python notebooks/eval_protocol.py --protocol rolling8-v1 --model B \\
        --feature-set base --out models/_exp/20261114-P0-R0-01 --cuts 2024-11-01,2025-04-01
    python notebooks/eval_protocol.py --protocol rolling8-v1 --model B --feature-set base \\
        --out models/_exp/20261114-P0-R0-02 --approval-id APPROVAL-2026-11-14-P0-legacy-B \\
        --legacy-baseline

원칙
    * Test(test_dev·test_final)는 DataGuard 승인 경로에서만 연다. 기본 실행은 train+valid 만 쓴다.
    * 피처·프로파일은 notebooks/calendar_features.py 를 가져와 재사용한다(복사 금지).
      학습 행 프로파일 = 그 날짜 이전만(as-of), 평가 행 프로파일 = cut 시점 고정(frozen, 서빙과 같은 계약).
    * 파라미터·난수 초기값(seed) 고정, 튜닝 없음. 모든 숫자는 이 스크립트가 만든다.
    * 산출물: models/_exp/<run_id>/ 에 산출물 명세(manifest) · 지표(metrics) · 일별 오차 · 예측 · 설정(config)
    * 실행 식별자(run_id)는 디렉터리 하나에 하나다. 산출물이 있는 디렉터리를 다시 쓰면 거부하고,
      같은 날짜·회차에서 비어 있는 순번을 안내한다(기록이 겹치면 어느 실행의 수치인지 알 수 없다).

parquet 주의
    이 환경에는 pyarrow/fastparquet 가 없다(신규 의존성 금지). parquet 엔진이 없으면 같은 내용을
    ``*.csv.gz`` 로 쓰고 artifacts.format 에 그 사실을 남긴다 — 조용히 다른 형식으로 바꾸지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT / "notebooks", ROOT / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import calendar_features as cf  # noqa: E402

from metro_insight.data.guard import DataGuard  # noqa: E402

PROTOCOL_ID = "rolling8-v1"
MANIFEST_SCHEMA = "metro-manifest/1"
RUN_ID_RE = re.compile(r"^[0-9]{8}-P[0-9]+-R[0-9]+-[0-9]{2}$")
SEED = 42
BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 10_000

# HistGradientBoosting(HGB) 고정 파라미터 — 튜닝 금지(train_abc.py · train_calendar.py 와 같은 값)
PARAMS: dict[str, float | int] = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "random_state": SEED,
}

USE_COLS = [
    "date", "station_code", "line_code", "hour_code", "weekday",
    "is_weekend", "is_holiday", "temperature", "rainfall", "snowfall", "boarding",
]

# B 모델의 9피처(A 8개 + is_holiday). 출처: src/metro_insight/model/predictor.py:56-66
BASE_FEATURES = [
    "profile_month_mean", "profile_n", "station_code", "line_code",
    "profile_weekday_mean", "profile_weekday_n", "hour_code", "weekday", "is_holiday",
]
# matched = base + 원본 boarding 에서 as_of 규칙으로 **다시 계산한** lag 3개
MATCHED_EXTRA = ["lag_1d", "mean_prev7d", "lag_7d"]
HORIZON_EXTRA = ["h", "log1p_h", "h_x_is_holiday", "h_x_weekday"]

# ── 다중 지평 밴드 · 범주형 피처 ──
# 밴드 4개가 기본 설계다. 장기 cut(182~366일)의 채점 지평은 91~181 밴드를 넘으므로 5번째 밴드를
# 덧붙인다(기본 4밴드 밖 확장 — manifest.bands 에 이유를 남긴다).
HORIZON_BANDS: tuple[tuple[int, int], ...] = ((1, 7), (8, 28), (29, 91), (92, 181), (182, 366))
SPEC_BANDS: tuple[tuple[int, int], ...] = HORIZON_BANDS[:4]
LONG_BAND: tuple[int, int] = HORIZON_BANDS[4]
BAND_RULE = (
    "밴드별 모델 하나씩. 학습 행은 '목표일 d(=행 날짜) + 지평 h∈밴드'로 만들고, 프로파일은 "
    "t=d−h 시점에 고정한 값(as-of at t)을 쓴다. h 는 난수 초기값(seed) 42 의 순열(permutation)로 "
    "밴드 안에 고르게 배분한다. "
    "평가 행은 as_of 고정 프로파일 + h 를 받고, 예측(predict)은 행의 h 로 밴드 모델을 고른다. "
    "182~366 밴드는 4밴드 설계 밖 확장이다(장기 cut 의 지평이 181일을 넘는다)."
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("line_code", "hour_code")
STATION_CODE = "station_code"
# HistGradientBoosting 의 범주 피처 카디널리티 상한은 max_bins(≤255) 와 같다.
HGB_MAX_CARDINALITY = 255
LAG_COLS: tuple[str, ...] = ("lag_1d", "lag_7d", "mean_prev7d")
LAG_RULE = (
    "lag_1d · lag_7d · mean_prev7d 를 원본 boarding 에서 다시 계산한다(CSV 의 기존 lag 열은 쓰지 않는다). "
    "학습 행(기준=자기 날짜): lag_1d=boarding(d−1), lag_7d=boarding(d−7), mean_prev7d=평균(d−7..d−1). "
    "평가 행(기준=as_of): 같은 정의를 기준일 as_of 에 적용해 as_of 시점에 볼 수 있는 값만 쓴다. "
    "따라서 어떤 행도 as_of 이후 값을 쓰지 않는다."
)

AS_OF_RULE = (
    "학습 행(date <= as_of): 그 날짜 이전 실측만 쓴 as-of 프로파일. "
    "평가 행(date > as_of): as_of 시점에 고정한(frozen) 프로파일(서빙과 같은 계약). "
    "평가 행의 boarding 은 프로파일 계산에 쓰지 않는다."
)

@dataclass(frozen=True)
class Cut:
    """롤링 오리진 한 칸: 기준일 · 학습 일수 · 평가 지평(일) · 종류."""

    cut_date: str
    train_days: int
    horizon: tuple[int, int]
    kind: Literal["long", "short"]


# 8개 고정 (cut_date 오름차순). train_days 는 2024-01-08 기준 경과일.
CUTS: list[Cut] = [
    Cut("2024-11-01", 298, (182, 366), "long"),
    Cut("2024-12-15", 342, (182, 365), "long"),
    Cut("2025-01-01", 359, (182, 366), "long"),
    Cut("2025-02-13", 402, (182, 364), "long"),
    Cut("2025-04-01", 449, (1, 181), "short"),
    Cut("2025-07-01", 540, (1, 181), "short"),
    Cut("2025-10-01", 632, (1, 181), "short"),
    Cut("2026-01-01", 724, (1, 180), "short"),
]

# 이전 기준선(legacy) 재현(승인 1회)용 고정 cut — 2026-02-14 프로파일로 Test 136일 채점
LEGACY_CUT = Cut("2026-02-14", 768, (1, 136), "short")
LEGACY_APPROVAL_ID = "APPROVAL-2026-11-14-P0-legacy-B"
LEGACY_TARGET_MAE = 92.72
LEGACY_TOLERANCE = 0.5


# ───────────────────────── 공통 ─────────────────────────
def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_order_hash(names: list[str]) -> str:
    """피처 순서 지문 — sha256('sha256-newline-join-v1\\n' + '\\n'.join(names)). 순서가 바뀌면 값이 바뀐다."""
    payload = "sha256-newline-join-v1\n" + "\n".join(names)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_names(df: pd.DataFrame) -> list[str]:
    names = df.attrs.get("feature_names")
    if not names:
        raise ValueError("피처 이름이 없다 — build_features() 결과를 그대로 넘겨야 한다")
    return list(names)


def _inherit(df: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    """슬라이스·샘플 뒤에도 attrs(feature_names 등)를 유지한다."""
    df.attrs.update(src.attrs)
    return df


def _git_state() -> dict:
    def run(*cmd: str) -> str:
        return subprocess.run(
            ["git", *cmd], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


# ───────────────────────── 피처 ─────────────────────────
@dataclass
class SeriesGrid:
    """(역·시간대) 시리즈 × 날짜 격자. 결측일은 NaN 으로 남긴다.

    lag 재계산과 밴드 학습 표본 생성이 같은 격자를 쓴다. 날짜가 빠진 곳에서 엉뚱한 날 값을
    가져오는 단순 밀기(shift) 오류를 막는 것이 목적이다.
    """

    values: np.ndarray            # (n_series, n_dates) 값
    dates: pd.DatetimeIndex
    series_pos: np.ndarray        # 행 → 시리즈 위치
    date_pos: np.ndarray          # 행 → 날짜 위치


def series_grid(df: pd.DataFrame, *, value: str = "boarding") -> SeriesGrid:
    """프레임을 (시리즈, 날짜) 격자로 펼친다. 같은 (역·시간대·날짜)는 하나만 있어야 한다."""
    keys = list(cf.KEY_SH)
    dates = pd.DatetimeIndex(np.sort(pd.to_datetime(df["date"]).unique()))
    series = df[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    order = pd.MultiIndex.from_frame(df[[*keys, "date"]])
    if not order.is_unique:
        raise ValueError(f"(역·시간대·날짜) 조합이 겹친다 — {value} 격자를 만들 수 없다")
    series_pos = pd.MultiIndex.from_frame(series).get_indexer(pd.MultiIndex.from_frame(df[keys]))
    date_pos = dates.get_indexer(pd.to_datetime(df["date"]))
    if (series_pos < 0).any() or (date_pos < 0).any():
        raise ValueError("격자 위치를 찾지 못한 행이 있다")
    flat = np.full((len(series), len(dates)), np.nan)
    flat[series_pos, date_pos] = df[value].to_numpy(dtype="float64")
    return SeriesGrid(values=flat, dates=dates, series_pos=series_pos, date_pos=date_pos)


def lag_columns(grid: SeriesGrid, anchor_pos: np.ndarray) -> dict[str, np.ndarray]:
    """기준 위치(anchor)마다 lag_1d · lag_7d · mean_prev7d 를 계산한다.

    기준 위치 행 자체와 그 이후 값은 쓰지 않는다(한 칸 밀기 의미론). 기준 위치가 0 이면 전부 NaN 이다.
    """
    values = grid.values
    n_series, n_dates = values.shape
    pad = 8
    padded = np.full((n_series, n_dates + pad), np.nan)
    padded[:, pad:] = values
    filled = np.nan_to_num(padded, nan=0.0)
    valid = (~np.isnan(padded)).astype("float64")
    cum_sum = np.zeros((n_series, padded.shape[1] + 1))
    cum_sum[:, 1:] = np.cumsum(filled, axis=1)
    cum_cnt = np.zeros((n_series, padded.shape[1] + 1))
    cum_cnt[:, 1:] = np.cumsum(valid, axis=1)

    rows = grid.series_pos
    anchor = np.asarray(anchor_pos, dtype="int64")
    lag_1d = padded[rows, anchor + (pad - 1)]
    lag_7d = padded[rows, anchor + (pad - 7)]
    lo = anchor + (pad - 7)
    hi = anchor + pad
    window_sum = cum_sum[rows, hi] - cum_sum[rows, lo]
    window_cnt = cum_cnt[rows, hi] - cum_cnt[rows, lo]
    mean_prev7d = np.where(
        window_cnt == 7.0,
        window_sum / np.where(window_cnt == 0.0, 1.0, window_cnt),
        np.nan,
    )
    return {"lag_1d": lag_1d, "lag_7d": lag_7d, "mean_prev7d": mean_prev7d}


# ── 임의 기준일의 고정 프로파일 조회 ──
_ASOF_STRIDE = 1_000_000


@dataclass
class AsOfProfileTables:
    """(그룹키, 기준일) 로 조회하는 as-of 평균 표 3종 — 역·시간대 / 월 / 요일.

    평가 행이 받는 고정 프로파일과 **같은 키**(목표 월·목표 요일)를 쓰면서 기준일만 행마다
    지정할 수 있다. 밴드 학습 표본은 기준일 t=d−h 로 조회하므로 t 이후 값이 피처에 들어가지 않는다.
    """

    parts: dict[str, tuple[np.ndarray, np.ndarray]]  # 이름 → (정렬 합성키, 누적합)


def asof_profile_tables(df: pd.DataFrame, *, value: str = "boarding") -> AsOfProfileTables:
    """정렬된 (키, 날짜) 누적합 표를 만든다. 조회는 :func:`asof_mean_at`."""
    parts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    layout = (("sh", cf.KEY_SH), ("month", cf.KEY_SH_MONTH), ("weekday", cf.KEY_SH_WEEKDAY))
    for name, key_cols in layout:
        order = df.sort_values([*key_cols, "date"], kind="mergesort")
        keys = cf._keys(order, key_cols).astype("int64")
        days = order["date"].to_numpy(dtype="datetime64[D]").astype("int64")
        composite = keys * _ASOF_STRIDE + days
        values = order[value].to_numpy(dtype="float64")
        prefix = np.concatenate(([0.0], np.cumsum(values)))
        parts[name] = (composite, prefix)
    return AsOfProfileTables(parts=parts)


def asof_mean_at(tables: AsOfProfileTables, name: str, key_frame: pd.DataFrame,
                 key_cols: list[str], days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """각 행의 (키, 기준일) 까지의 평균과 표본수를 돌려준다. 기준일 당일을 포함한다."""
    composite, prefix = tables.parts[name]
    keys = cf._keys(key_frame, key_cols).astype("int64")
    probe = keys * _ASOF_STRIDE + np.asarray(days, dtype="int64")
    pos = np.searchsorted(composite, probe, side="right")
    start = np.searchsorted(composite, keys * _ASOF_STRIDE, side="left")
    n = (pos - start).astype("float64")
    total = prefix[pos] - prefix[start]
    mean = np.divide(total, n, out=np.full(len(n), np.nan, dtype="float64"), where=n > 0)
    return mean, n


def month_profile_fallback(frame: pd.DataFrame) -> pd.DataFrame:
    """월 프로파일이 없는 행을 (역·시간대) 평균·표본수로 채운다.

    첫 관측 연도에는 목표 월의 이력이 없어 월 프로파일이 비는 구간이 생긴다. 그대로 두면
    학습 표본에서 열 전체가 비어 트리 학습이 실패하므로, 값이 있는 가장 가까운 수준으로 대체한다.
    """
    n = frame["profile_n"].to_numpy(dtype="float64")
    mean = frame["profile_month_mean"].to_numpy(dtype="float64")
    sh_mean = frame["profile_sh_mean"].to_numpy(dtype="float64")
    sh_n = frame["profile_sh_n"].to_numpy(dtype="float64")
    frame["profile_month_mean"] = np.where(n > 0, mean, sh_mean)
    frame["profile_n"] = np.where(n > 0, n, sh_n)
    return frame


def build_features(
    df: pd.DataFrame,
    *,
    as_of: str,
    feature_set: Literal["base", "matched", "horizon"],
) -> pd.DataFrame:
    """as_of 앞뒤로 프로파일을 다르게 붙여 피처를 만든다.

    * date <= as_of 행: 그 날짜 이전만 쓴 as-of 프로파일
    * date >  as_of 행: as_of 시점에 고정한(frozen) 프로파일
    * feature_set="base"    : 기본 9피처
    * feature_set="matched" : base + as_of 규칙으로 다시 계산한 lag_1d · mean_prev7d · lag_7d
    * feature_set="horizon" : base + h · log1p(h) · h×is_holiday · h×weekday
    """
    if feature_set not in ("base", "matched", "horizon"):
        raise ValueError(f"모르는 feature_set: {feature_set}")
    as_of_ts = pd.Timestamp(as_of)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in [*cf.KEY_SH, "date", "boarding"]:
        if col not in out.columns:
            raise KeyError(f"컬럼이 없다: {col}")
    out = cf.add_month(out)

    is_fit = out["date"] <= as_of_ts
    fit_part = out.loc[is_fit]
    eval_part = out.loc[~is_fit]
    if fit_part.empty:
        raise ValueError(f"as_of({as_of}) 이전 행이 없다 — 프로파일을 만들 수 없다")

    prof_fit = cf.asof_profile_frame(fit_part)
    prof_cols = [*cf.PROFILE_COLS_SH, *cf.PROFILE_COLS_MONTH, *cf.PROFILE_COLS_WEEKDAY]
    if eval_part.empty:
        prof_eval = pd.DataFrame(index=eval_part.index, columns=prof_cols)
    else:
        prof_eval = cf.frozen_profile_frame(fit_part, eval_part, as_of_ts)
    profile = pd.concat([prof_fit, prof_eval]).reindex(out.index)
    out = pd.concat([out, profile[prof_cols]], axis=1)

    names = list(BASE_FEATURES)
    missing: list[str] = []
    if feature_set == "matched":
        # CSV 의 lag 열은 as_of 기준이 아니다(평가 행이면 미래 값이 섞인다) → boarding 에서 다시 계산한다.
        grid = series_grid(out[["date", *cf.KEY_SH, "boarding"]])
        anchor = grid.date_pos.copy()
        is_eval = out["date"].to_numpy() > np.datetime64(as_of_ts)
        if is_eval.any():
            as_of_pos = int(grid.dates.searchsorted(as_of_ts, side="right")) - 1
            if as_of_pos < 0:
                raise ValueError(f"as_of({as_of}) 가 격자 시작보다 이르다 — lag 기준점을 잡을 수 없다")
            anchor[is_eval] = as_of_pos
        for col, values in lag_columns(grid, anchor).items():
            out[col] = values
        del grid, anchor
        names += list(MATCHED_EXTRA)
    elif feature_set == "horizon":
        days = (out["date"] - as_of_ts).dt.days
        h = days.clip(lower=0).astype("int32")
        out["h"] = h
        out["log1p_h"] = np.log1p(h.to_numpy(dtype="float64"))
        out["h_x_is_holiday"] = (h * out["is_holiday"].astype("int64")).astype("int32")
        out["h_x_weekday"] = (h * out["weekday"].astype("int64")).astype("int32")
        out = month_profile_fallback(out)
        names += HORIZON_EXTRA

    out.attrs["feature_names"] = names
    out.attrs["feature_set"] = feature_set
    out.attrs["as_of"] = as_of_ts.strftime("%Y-%m-%d")
    out.attrs["missing_features"] = missing
    out.attrs["categorical_candidates"] = list(CATEGORICAL_FEATURES)
    out.attrs["bands"] = [tuple(b) for b in HORIZON_BANDS] if feature_set == "horizon" else []
    out.attrs["lag_rule"] = LAG_RULE if feature_set == "matched" else None
    return out


# ───────────────────────── 학습·채점 ─────────────────────────
class CategoricalRegressor:
    """범주 피처를 HistGradientBoosting 에 넘기는 얇은 래퍼.

    HistGradientBoosting 은 범주 피처 카디널리티가 max_bins(≤255) 이하일 때만 받는다.
    station_code 는 273개라 그대로는 거부되므로 (몫, 나머지) 두 조각으로 나눠 무손실로 인코딩한다.
    조각은 순서 정보가 없는 이름표이므로 수치 투입과 다르다.
    """

    def __init__(self, params: dict, seed: int, categorical: list[str]) -> None:
        self.params = dict(params)
        self.seed = int(seed)
        self.categorical = [c for c in categorical]
        self.feature_names_in_: np.ndarray | None = None
        self.encoder: dict[str, str] = {}
        self.cardinality: dict[str, int] = {}
        self.split: dict[str, bool] = {}
        self._extra: list[str] = []
        self._model: HistGradientBoostingRegressor | None = None

    # ── 인코딩 ──
    def _encode(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        """학습 때 정한 분해 여부를 그대로 재사용한다(예측 쪽 카디널리티가 달라도 열이 어긋나지 않게)."""
        out = frame.copy()
        extra: list[str] = []
        for col in self.categorical:
            if col not in out.columns:
                raise KeyError(f"범주 피처 컬럼이 없다: {col}")
            values = out[col].to_numpy()
            if fit:
                self.cardinality[col] = int(pd.Series(values).nunique(dropna=True))
                self.split[col] = self.cardinality[col] > HGB_MAX_CARDINALITY
            if not self.split.get(col, False):
                out[col] = pd.Series(values, index=out.index).astype("int64")
                self.encoder[col] = f"categorical({self.cardinality.get(col, 0)}종)"
                continue
            high, low = np.divmod(values.astype("int64"), HGB_MAX_CARDINALITY)
            out[f"{col}_hi"] = high.astype("int64")
            out[f"{col}_lo"] = low.astype("int64")
            extra += [f"{col}_hi", f"{col}_lo"]
            self.encoder[col] = (
                f"categorical({self.cardinality[col]}종) → {col}_hi + {col}_lo 분해 "
                f"(카디널리티 상한 {HGB_MAX_CARDINALITY})"
            )
        self._extra = extra
        return out

    def _columns(self, names: list[str]) -> list[str]:
        """분해된 원본 열은 빼고, 분해 조각을 뒤에 붙인다."""
        base = [n for n in names if not self.split.get(n, False)]
        return [*base, *self._extra]

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> CategoricalRegressor:
        names = list(frame.columns)
        self.feature_names_in_ = np.asarray(names, dtype=object)
        encoded = self._encode(frame, fit=True)
        columns = self._columns(names)
        mask = [name in self.categorical for name in columns]
        kwargs = dict(self.params)
        kwargs.setdefault("random_state", self.seed)
        model = HistGradientBoostingRegressor(**kwargs, categorical_features=mask)
        model.fit(encoded[columns], np.asarray(y, dtype="float64"))
        self._model = model
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit 하지 않은 모델이다")
        names = [str(n) for n in self.feature_names_in_]
        encoded = self._encode(frame.loc[:, names], fit=False)
        return np.asarray(self._model.predict(encoded[self._columns(names)]), dtype="float64")


class LightGBMRegressorWrapper:
    """LightGBM 비교용 래퍼. 범주 피처를 라이브러리 기본 처리에 맡긴다(카디널리티 상한 없음)."""

    def __init__(self, params: dict, seed: int, categorical: list[str]) -> None:
        self.params = dict(params)
        self.seed = int(seed)
        self.categorical = list(categorical)
        self.feature_names_in_: np.ndarray | None = None
        self.categories: dict[str, list] = {}
        self._model = None

    def _prepare(self, frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        out = frame.copy()
        for col in self.categorical:
            if col not in out.columns:
                raise KeyError(f"범주 피처 컬럼이 없다: {col}")
            if fit:
                cats = sorted(pd.Series(out[col]).dropna().unique().tolist())
                self.categories[col] = cats
            cats = self.categories.get(col)
            out[col] = pd.Categorical(out[col], categories=cats)
        return out

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> LightGBMRegressorWrapper:
        import lightgbm as lgb

        names = list(frame.columns)
        self.feature_names_in_ = np.asarray(names, dtype=object)
        prepared = self._prepare(frame, fit=True)
        model = lgb.LGBMRegressor(
            n_estimators=int(self.params["max_iter"]),
            num_leaves=int(self.params["max_leaf_nodes"]),
            learning_rate=float(self.params["learning_rate"]),
            colsample_bytree=0.9,
            random_state=self.seed,
            verbose=-1,
        )
        model.fit(prepared[names], np.asarray(y, dtype="float64"),
                  categorical_feature=list(self.categorical))
        self._model = model
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit 하지 않은 모델이다")
        names = [str(n) for n in self.feature_names_in_]
        prepared = self._prepare(frame.loc[:, names], fit=False)
        return np.asarray(self._model.predict(prepared[names]), dtype="float64")


class BandedModel:
    """지평 밴드별 모델 묶음. 예측(predict)은 행의 h 값으로 밴드를 고른다."""

    def __init__(self, models: dict[tuple[int, int], object], names: list[str]) -> None:
        if not models:
            raise ValueError("밴드 모델이 비었다")
        self.models = dict(models)
        self.bands = sorted(self.models)
        self.feature_names_in_ = np.asarray(names, dtype=object)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if "h" not in frame.columns:
            raise KeyError("밴드 모델은 h 컬럼이 필요하다 — feature_set='horizon' 결과를 넘겨야 한다")
        horizon = frame["h"].to_numpy(dtype="int64")
        out = np.full(len(frame), np.nan)
        covered = np.zeros(len(frame), dtype=bool)
        for band, model in self.models.items():
            mask = (horizon >= band[0]) & (horizon <= band[1])
            if not mask.any():
                continue
            out[mask] = np.asarray(model.predict(frame[mask]), dtype="float64")
            covered |= mask
        if not covered.all():
            unknown = sorted(set(horizon[~covered].tolist()))[:5]
            raise ValueError(f"어느 밴드에도 속하지 않는 지평이 있다: {unknown}")
        return out


def _fit_single(
    frame: pd.DataFrame,
    *,
    params: dict,
    seed: int,
    categorical: list[str] | None,
    backend: str,
) -> object:
    names = feature_names(frame)
    missing = [c for c in names if c not in frame.columns]
    if missing:
        raise KeyError(f"피처 컬럼이 없다: {missing}")
    if backend not in ("hgb", "lightgbm"):
        raise ValueError(f"모르는 backend: {backend}")
    if backend == "lightgbm":
        model: object = LightGBMRegressorWrapper(params, seed, categorical or [])
        return model.fit(frame[names], frame["boarding"].to_numpy(dtype="float64"))
    if categorical:
        wrapper = CategoricalRegressor(params, seed, list(categorical))
        return wrapper.fit(frame[names], frame["boarding"].to_numpy(dtype="float64"))
    kwargs = dict(params)
    kwargs.setdefault("random_state", seed)
    model = HistGradientBoostingRegressor(**kwargs)
    model.fit(frame[names], frame["boarding"].to_numpy(dtype="float64"))
    return model


def fit_model(
    df: pd.DataFrame,
    *,
    params: dict,
    seed: int = 42,
    categorical: list[str] | None = None,
    backend: str = "hgb",
    bands: list[tuple[int, int]] | None = None,
):
    """고정 파라미터로 모델을 학습한다(튜닝 없음).

    bands 를 주면 밴드별 모델을 학습해 BandedModel 로 묶는다. 주지 않으면 단일 모델이다.
    """
    names = feature_names(df)
    if not bands:
        return _fit_single(df, params=params, seed=seed, categorical=categorical, backend=backend)
    models: dict[tuple[int, int], object] = {}
    as_of_ts = pd.Timestamp(df.attrs["as_of"])
    tables = asof_profile_tables(df.loc[df["date"] <= as_of_ts])
    for band in bands:
        frame = band_training_frame(df, band=tuple(band), seed=seed, tables=tables)
        models[tuple(band)] = _fit_single(
            frame, params=params, seed=seed, categorical=categorical, backend=backend
        )
        del frame
    del tables
    return BandedModel(models, names)


def band_training_frame(
    features: pd.DataFrame,
    *,
    band: tuple[int, int],
    as_of: str | None = None,
    seed: int = 42,
    max_rows: int | None = None,
    tables: AsOfProfileTables | None = None,
) -> pd.DataFrame:
    """한 밴드의 학습 표본: 목표일 d, 지평 h∈밴드, 프로파일은 t=d−h 시점 고정값.

    프로파일 키는 평가 행과 같다(목표 월·목표 요일). 값은 t 까지의 관측만 쓴다.
    h 는 순열(permutation)로 밴드 안에 고르게 배분한다(결정적).
    """
    names = feature_names(features)
    lo, hi = band
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(features.attrs["as_of"])
    frame = features.loc[features["date"] <= as_of_ts]
    if frame.empty:
        raise ValueError(f"as_of({as_of_ts.date()}) 이전 행이 없어 밴드 {band} 학습 표본을 만들 수 없다")
    width = hi - lo + 1
    rng = np.random.default_rng(seed)
    horizon = lo + (rng.permutation(len(frame)) % width).astype("int64")

    tables = tables if tables is not None else asof_profile_tables(frame)
    target_days = frame["date"].to_numpy(dtype="datetime64[D]").astype("int64")
    # 기준일 t=d−h 에 최소 하루의 이력이 있어야 표본이 된다 → h 상한을 날짜별로 자른다.
    max_h = target_days - int(target_days.min()) - 1
    usable = (max_h >= lo) & (horizon <= max_h)
    if not usable.any():
        raise ValueError(
            f"밴드 {band}: 학습 이력이 짧아 표본을 만들 수 없다(가장 긴 지평 {int(max_h.max())}일)"
        )
    anchor_days = target_days - horizon
    sh_mean, sh_n = asof_mean_at(tables, "sh", frame, cf.KEY_SH, anchor_days)
    mo_mean, mo_n = asof_mean_at(tables, "month", frame, cf.KEY_SH_MONTH, anchor_days)
    wd_mean, wd_n = asof_mean_at(tables, "weekday", frame, cf.KEY_SH_WEEKDAY, anchor_days)
    usable &= sh_n > 0  # (역·시간대) 이력이 아예 없는 행은 피처가 전부 비어 있다

    out = frame.loc[usable].copy()
    out[cf.PROFILE_COLS_SH[0]] = sh_mean[usable]
    out[cf.PROFILE_COLS_SH[1]] = sh_n[usable]
    out[cf.PROFILE_COLS_MONTH[0]] = mo_mean[usable]
    out[cf.PROFILE_COLS_MONTH[1]] = mo_n[usable]
    out[cf.PROFILE_COLS_WEEKDAY[0]] = wd_mean[usable]
    out[cf.PROFILE_COLS_WEEKDAY[1]] = wd_n[usable]
    out = month_profile_fallback(out)
    kept_h = horizon[usable]
    out["h"] = kept_h
    out["log1p_h"] = np.log1p(kept_h.astype("float64"))
    out["h_x_is_holiday"] = (kept_h * out["is_holiday"].astype("int64")).astype("int64")
    out["h_x_weekday"] = (kept_h * out["weekday"].astype("int64")).astype("int64")
    if max_rows and len(out) > int(max_rows):
        out = out.sample(n=int(max_rows), random_state=seed)
    out.attrs.update(features.attrs)
    out.attrs["feature_names"] = list(names)
    out.attrs["band"] = (int(lo), int(hi))
    return out


def band_training_rows(features: pd.DataFrame, *, band: tuple[int, int], as_of: str,
                       tables: AsOfProfileTables | None = None) -> int:
    """밴드 학습 표본 수(manifest 기록용)."""
    return int(len(band_training_frame(features, band=band, as_of=as_of, tables=tables)))


def score(y: np.ndarray, pred: np.ndarray, horizon_days: np.ndarray) -> dict:
    """평균절대오차(MAE) · 제곱근평균제곱오차(RMSE) · 가중절대백분율오차(WAPE) · 표본수(n) ·
    편향(bias) · 음수 예측 · 지평 구간 슬라이스."""
    y = np.asarray(y, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    if y.shape != pred.shape:
        raise ValueError(f"길이가 다르다: y={y.shape} pred={pred.shape}")
    err = y - pred
    abs_err = np.abs(err)
    n = int(len(y))
    y_abs_sum = float(np.abs(y).sum())
    out: dict = {
        "n": n,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "wape": float(abs_err.sum() / y_abs_sum) if y_abs_sum > 0 else float("nan"),
        "bias": float(err.mean()),
        "negative_predictions": int((pred < 0).sum()),
        "abs_err_sum": float(abs_err.sum()),
        "sq_err_sum": float((err ** 2).sum()),
        "signed_err_sum": float(err.sum()),
        "y_abs_sum": y_abs_sum,
        "horizon_bucket": {},
    }
    bucket = cf.horizon_bucket(np.asarray(horizon_days))
    for lo, hi in cf.HORIZON_BUCKETS:
        label = f"{lo}-{hi}"
        mask = bucket == label
        if mask.any():
            e = err[mask]
            y_sum = float(np.abs(y[mask]).sum())
            out["horizon_bucket"][label] = {
                "mae": float(np.abs(e).mean()),
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "wape": float(np.abs(e).sum() / y_sum) if y_sum > 0 else float("nan"),
                "n": int(mask.sum()),
            }
    return out


def naive_prediction(df_eval: pd.DataFrame) -> np.ndarray:
    """기준선: 월 프로필 평균, 없으면 (역·시간) 평균. 프로파일은 as_of 시점 고정값이다."""
    month = df_eval[cf.PROFILE_COLS_MONTH[0]].to_numpy(dtype="float64")
    sh = df_eval[cf.PROFILE_COLS_SH[0]].to_numpy(dtype="float64")
    return np.where(np.isnan(month), np.nan_to_num(sh, nan=0.0), month)


def daily_table(dates, abs_err, model: str, cut: str) -> pd.DataFrame:
    """날짜별 평균 절대오차 1행 — 부트스트랩 입력."""
    frame = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(dates)),
        "abs_err": np.asarray(abs_err, dtype="float64"),
    })
    grouped = frame.groupby("date", observed=True)["abs_err"].agg(["sum", "count"]).reset_index()
    grouped["abs_err"] = grouped["sum"] / grouped["count"]
    return pd.DataFrame({
        "date": grouped["date"],
        "model": model,
        "abs_err": grouped["abs_err"].astype("float64"),
        "n": grouped["count"].astype("int64"),
        "cut": cut,
    })


def evaluate_cut(model, df_eval: pd.DataFrame, *, cut: Cut) -> dict:
    """평가 창(cut+horizon[0] ~ cut+horizon[1])에 실제값이 있는 행만 채점한다."""
    names = list(getattr(model, "feature_names_in_", [])) or feature_names(df_eval)
    model_name = df_eval.attrs.get("model", "?")
    cut_ts = pd.Timestamp(cut.cut_date)
    lo, hi = cut.horizon
    frame = df_eval.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    horizon = (frame["date"] - cut_ts).dt.days.to_numpy(dtype="int64")
    keep = (horizon >= lo) & (horizon <= hi)
    if not keep.any():
        raise ValueError(f"cut {cut.cut_date}: 평가 창에 실제값이 있는 행이 없다")
    frame = frame.loc[keep]
    horizon = horizon[keep]

    y = frame["boarding"].to_numpy(dtype="float64")
    pred = np.asarray(model.predict(frame[names]), dtype="float64")
    out = score(y, pred, horizon)
    out.update({
        "cut_date": cut.cut_date,
        "kind": cut.kind,
        "horizon": [int(lo), int(hi)],
        "train_days": int(cut.train_days),
        "model": model_name,
        "days": int(frame["date"].nunique()),
        "date_min": frame["date"].min().strftime("%Y-%m-%d"),
        "date_max": frame["date"].max().strftime("%Y-%m-%d"),
        "horizon_min": int(horizon.min()),
        "horizon_max": int(horizon.max()),
    })
    naive = naive_prediction(frame)
    out["naive"] = {**score(y, naive, horizon), "cut_date": cut.cut_date, "model": "naive_profile_mean"}
    out["daily"] = daily_table(frame["date"], np.abs(y - pred), out["model"], cut.cut_date).to_dict("records")
    out["daily_naive"] = daily_table(
        frame["date"], np.abs(y - naive), "naive_profile_mean", cut.cut_date
    ).to_dict("records")
    out["_arrays"] = {
        "date": frame["date"].to_numpy(),
        "station_code": frame["station_code"].to_numpy(dtype="int32"),
        "line_code": frame["line_code"].to_numpy(dtype="int16"),
        "hour_code": frame["hour_code"].to_numpy(dtype="int16"),
        "weekday": frame["weekday"].to_numpy(dtype="int8"),
        "is_holiday": frame["is_holiday"].to_numpy(dtype="int8"),
        "is_weekend": frame["is_weekend"].to_numpy(dtype="int8"),
        "y": y,
        "pred": pred,
        "horizon": horizon.astype("int16"),
    }
    out["horizon_band"] = band_slices(y, pred, horizon)
    return out


HOUR_GROUPS: tuple[tuple[int, int], ...] = ((0, 4), (5, 9), (10, 14), (15, 19))


def band_slices(y: np.ndarray, pred: np.ndarray, horizon: np.ndarray) -> dict[str, dict]:
    """지평 밴드별 MAE. 밴드 밖 지평은 세지 않는다(라벨을 만들지 않는다)."""
    y = np.asarray(y, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    horizon = np.asarray(horizon)
    out: dict[str, dict] = {}
    for lo, hi in HORIZON_BANDS:
        mask = (horizon >= lo) & (horizon <= hi)
        if not mask.any():
            continue
        err = y[mask] - pred[mask]
        y_abs = float(np.abs(y[mask]).sum())
        out[f"{lo}-{hi}"] = {
            "n": int(mask.sum()),
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "wape": float(np.abs(err).sum() / y_abs) if y_abs > 0 else float("nan"),
            "bias": float(err.mean()),
        }
    return out


def _slice_labels(arrays: dict) -> dict[str, np.ndarray]:
    horizon = np.asarray(arrays["horizon"])
    labels: dict[str, np.ndarray] = {}
    band = np.full(len(horizon), "밖", dtype=object)
    for lo, hi in HORIZON_BANDS:
        band[(horizon >= lo) & (horizon <= hi)] = f"{lo}-{hi}"
    labels["horizon_band"] = band
    labels["holiday"] = np.where(np.asarray(arrays["is_holiday"]) > 0, "holiday", "weekday")
    labels["weekday_group"] = np.where(np.asarray(arrays["weekday"]) >= 5, "weekend", "weekday")
    hour = np.asarray(arrays["hour_code"])
    hour_group = np.full(len(hour), "밖", dtype=object)
    for lo, hi in HOUR_GROUPS:
        hour_group[(hour >= lo) & (hour <= hi)] = f"{lo}-{hi}"
    labels["hour_group"] = hour_group
    labels["line"] = np.asarray(arrays["line_code"]).astype(str)
    return labels


def slice_sums(arrays: dict) -> dict[str, dict[str, dict]]:
    """슬라이스별 합계를 낸다. cut 을 합칠 때 n 가중이 유지되도록 합계로 넘긴다."""
    y = np.asarray(arrays["y"], dtype="float64")
    pred = np.asarray(arrays["pred"], dtype="float64")
    err = y - pred
    abs_err = np.abs(err)
    acc: dict[str, dict[str, dict]] = {}
    for group, labels in _slice_labels(arrays).items():
        bucket: dict[str, dict] = {}
        for label in np.unique(labels):
            mask = labels == label
            bucket[str(label)] = {
                "n": int(mask.sum()),
                "abs_err_sum": float(abs_err[mask].sum()),
                "sq_err_sum": float((err[mask] ** 2).sum()),
                "signed_err_sum": float(err[mask].sum()),
                "y_abs_sum": float(np.abs(y[mask]).sum()),
            }
        acc[group] = bucket
    return acc


def combine_slice_sums(parts: list[dict[str, dict[str, dict]]]) -> dict[str, dict[str, dict]]:
    """여러 cut 의 슬라이스 합계를 더해 지표로 바꾼다."""
    total: dict[str, dict[str, dict]] = {}
    for part in parts:
        for group, bucket in part.items():
            target = total.setdefault(group, {})
            for label, sums in bucket.items():
                slot = target.setdefault(
                    label, {"n": 0, "abs_err_sum": 0.0, "sq_err_sum": 0.0,
                            "signed_err_sum": 0.0, "y_abs_sum": 0.0}
                )
                for key, value in sums.items():
                    slot[key] += value
    out: dict[str, dict[str, dict]] = {}
    for group, bucket in total.items():
        out[group] = {}
        for label, sums in bucket.items():
            n = sums["n"]
            out[group][label] = {
                "n": int(n),
                "mae": sums["abs_err_sum"] / n if n else float("nan"),
                "rmse": float(np.sqrt(sums["sq_err_sum"] / n)) if n else float("nan"),
                "bias": sums["signed_err_sum"] / n if n else float("nan"),
                "wape": sums["abs_err_sum"] / sums["y_abs_sum"] if sums["y_abs_sum"] else float("nan"),
            }
    return out


def pooled(per_cut: list[dict]) -> dict:
    """cut 들을 n 가중으로 합친다(단순 평균 금지)."""
    if not per_cut:
        raise ValueError("per_cut 이 비었다")

    def combine(entries: list[dict]) -> dict:
        n = sum(int(e["n"]) for e in entries)
        abs_sum = sum(float(e["abs_err_sum"]) for e in entries)
        sq_sum = sum(float(e["sq_err_sum"]) for e in entries)
        signed = sum(float(e["signed_err_sum"]) for e in entries)
        y_sum = sum(float(e["y_abs_sum"]) for e in entries)
        return {
            "n": n,
            "mae": abs_sum / n,
            "rmse": float(np.sqrt(sq_sum / n)),
            "wape": abs_sum / y_sum if y_sum > 0 else float("nan"),
            "bias": signed / n,
            "negative_predictions": sum(int(e["negative_predictions"]) for e in entries),
            "cuts": len(entries),
            "cut_dates": [e["cut_date"] for e in entries],
            "per_cut_mae": {e["cut_date"]: e["mae"] for e in entries},
        }

    out = combine(per_cut)
    naive = [c["naive"] for c in per_cut if "naive" in c]
    if naive and len(naive) == len(per_cut):
        out["naive"] = combine(naive)
    return out


# ───────────────────────── 부트스트랩 ─────────────────────────
def _daily_frame(daily) -> pd.DataFrame:
    """일별 표를 블록(날짜 또는 cut+날짜) 단위로 정리한다. 여러 줄이면 n 가중 평균."""
    frame = daily if isinstance(daily, pd.DataFrame) else pd.DataFrame(list(daily))
    for col in ("date", "abs_err"):
        if col not in frame.columns:
            raise KeyError(f"일별 표에 컬럼이 없다: {col}")
    date = pd.to_datetime(frame["date"])
    n = frame["n"].astype("float64") if "n" in frame.columns else np.ones(len(frame))
    work = pd.DataFrame({"date": date, "abs_err": frame["abs_err"].astype("float64"), "n": n})
    if "cut" in frame.columns:
        work["block"] = frame["cut"].astype(str).to_numpy() + "|" + work["date"].dt.strftime("%Y-%m-%d")
    else:
        work["block"] = work["date"].dt.strftime("%Y-%m-%d")
    work["_w"] = work["abs_err"] * work["n"]
    grouped = work.groupby("block", observed=True).agg(_w=("_w", "sum"), n=("n", "sum")).reset_index()
    grouped["abs_err"] = grouped["_w"] / grouped["n"]
    return grouped.loc[:, ["block", "n", "abs_err"]]


def day_block_bootstrap(daily_a, daily_b, *, n: int = 10_000, seed: int = 42) -> dict:
    """날짜(cut+날짜) 덩어리 단위 재표집 — 행 단위 재표집 금지.

    같은 날의 여러 행은 한 덩어리로 함께 뽑히므로 두 모델의 짝이 깨지지 않는다.
    """
    a = _daily_frame(daily_a)
    b = _daily_frame(daily_b)
    merged = a.merge(b, on="block", suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        raise ValueError("두 일별 표에 겹치는 블록이 없다")
    blocks = len(merged)
    # 블록 키가 'cut|date' 형태면 cut 이 섞인 표다 → 라벨과 의존성 경고를 정확히 남긴다.
    has_cut = bool(merged["block"].str.contains("|", regex=False).any())
    overlapping = bool(merged["block"].str.split("|", n=1).str[-1].duplicated().any()) if has_cut else False
    n = int(n)
    wa = merged["n_a"].to_numpy(dtype="float64")
    wb = merged["n_b"].to_numpy(dtype="float64")
    ea = merged["abs_err_a"].to_numpy(dtype="float64")
    eb = merged["abs_err_b"].to_numpy(dtype="float64")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n, dtype="float64")
    batch = 500  # 메모리 상한(블록이 1천 개를 넘어도 안전)
    for start in range(0, n, batch):
        size = min(batch, n - start)
        idx = rng.integers(0, blocks, size=(size, blocks))
        mae_a = (ea[idx] * wa[idx]).sum(axis=1) / wa[idx].sum(axis=1)
        mae_b = (eb[idx] * wb[idx]).sum(axis=1) / wb[idx].sum(axis=1)
        diffs[start:start + size] = mae_b - mae_a
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    result = {
        "unit": "cut+date 블록(날짜 단위 재표집)" if has_cut else "date 블록(날짜 단위 재표집)",
        "n_boot": n,
        "blocks": int(blocks),
        "seed": int(seed),
        "mae_a": float((ea * wa).sum() / wa.sum()),
        "mae_b": float((eb * wb).sum() / wb.sum()),
        "mean_diff_b_minus_a": float(diffs.mean()),
        "ci95": [float(lo), float(hi)],
        "p_b_better": float((diffs < 0).mean()),
    }
    if has_cut:
        result["blocks_are_independent"] = not overlapping
        if overlapping:
            result["note"] = (
                "cut 창이 겹쳐 같은 날짜가 여러 블록에 들어간다. 블록 독립 가정이 약하므로 "
                "이 CI 폭은 하한(과소)으로 읽는다."
            )
    return result


# ───────────────────────── 산출물 ─────────────────────────
def _dump_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def _write_table(df: pd.DataFrame, run_dir: Path, stem: str) -> dict:
    """parquet 우선, 엔진이 없으면 csv.gz (형식을 조용히 바꾸지 않고 artifacts 에 남긴다)."""
    parquet = run_dir / f"{stem}.parquet"
    try:
        df.to_parquet(parquet, index=False)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:  # parquet 엔진 없음
        csv_path = run_dir / f"{stem}.csv.gz"
        df.to_csv(csv_path, index=False, compression="gzip")
        return {
            "path": csv_path.name,
            "format": "csv.gz",
            "degraded_from": "parquet",
            "reason": f"parquet 엔진 없음({type(exc).__name__}) — 신규 의존성 금지",
            "rows": int(len(df)),
            "sha256": _sha256_file(csv_path),
        }
    return {"path": parquet.name, "format": "parquet", "rows": int(len(df)),
            "sha256": _sha256_file(parquet)}


def _validate_manifest(doc: dict) -> None:
    problems: list[str] = []
    if doc.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"schema != {MANIFEST_SCHEMA}")
    run_id = doc.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        problems.append("run_id 형식 오류(YYYYMMDD-Pn-Rn-NN)")
    if not isinstance(doc.get("created_at"), str) or not doc.get("created_at"):
        problems.append("created_at 없음")
    git = doc.get("git")
    if not isinstance(git, dict) or "commit" not in git or "dirty" not in git:
        problems.append("git{commit,dirty} 없음")
    if doc.get("task") not in ("T1", "T2"):
        problems.append("task 는 T1 또는 T2")
    if not isinstance(doc.get("horizon"), dict):
        problems.append("horizon 없음")
    data = doc.get("data")
    if not isinstance(data, dict) or not data:
        problems.append("data 없음")
    else:
        for split, meta in data.items():
            if not isinstance(meta, dict):
                problems.append(f"data.{split} 형식 오류")
                continue
            for key in ("path", "sha256", "rows", "date_min", "date_max"):
                if key not in meta:
                    problems.append(f"data.{split}.{key} 없음")
    features = doc.get("features")
    if not isinstance(features, dict) or not isinstance(features.get("names"), list) or not features["names"]:
        problems.append("features.names 없음")
    elif features.get("order_hash") != feature_order_hash(features["names"]):
        problems.append("features.order_hash 불일치(순서 지문 재계산 실패)")
    elif not features.get("as_of_rule"):
        problems.append("features.as_of_rule 없음")
    if not isinstance(doc.get("params"), dict) or not doc.get("params"):
        problems.append("params 없음")
    if not isinstance(doc.get("seeds"), dict):
        problems.append("seeds 없음")
    if doc.get("protocol") != PROTOCOL_ID:
        problems.append(f"protocol != {PROTOCOL_ID}")
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict):
        problems.append("metrics 없음")
    else:
        if not isinstance(metrics.get("per_cut"), list) or not metrics["per_cut"]:
            problems.append("metrics.per_cut 없음")
        if not isinstance(metrics.get("pooled"), dict):
            problems.append("metrics.pooled 없음")
        if not isinstance(metrics.get("ci"), dict):
            problems.append("metrics.ci 없음")
    if not isinstance(doc.get("verdict"), dict) or not doc["verdict"].get("result"):
        problems.append("verdict.result 없음")
    if not isinstance(doc.get("artifacts"), dict):
        problems.append("artifacts 없음")
    if problems:
        raise ValueError("manifest 검증 실패: " + "; ".join(problems))


def write_manifest(run_dir: Path, payload: dict) -> Path:
    """manifest.json 을 쓰고 경로를 돌려준다. 필수 필드가 없으면 ValueError."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = dict(payload)
    doc.setdefault("schema", MANIFEST_SCHEMA)
    doc.setdefault("created_at", _now())
    _validate_manifest(doc)
    path = _dump_json(run_dir / "manifest.json", doc)
    return path


def _data_meta(guard: DataGuard, splits: list[str], frames: dict[str, pd.DataFrame]) -> dict:
    meta = {}
    for name in splits:
        frame = frames[name]
        dates = pd.to_datetime(frame["date"])
        rel = {
            "train": "data/processed/train.csv.gz",
            "valid": "data/processed/valid.csv.gz",
            "test_dev": "data/processed/test.csv.gz",
            "test_final": "data/processed/test.csv.gz",
        }[name]
        path = guard.root / rel
        meta[name] = {
            "path": rel,
            "sha256": _sha256_file(path) if path.exists() else None,
            "rows": int(len(frame)),
            "date_min": dates.min().strftime("%Y-%m-%d"),
            "date_max": dates.max().strftime("%Y-%m-%d"),
        }
    return meta


def _predictions_frame(arrays: dict, model_name: str, cut: Cut) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(arrays["date"]),
        "cut": cut.cut_date,
        "model": model_name,
        "station_code": arrays["station_code"],
        "line_code": arrays["line_code"],
        "hour_code": arrays["hour_code"],
        "weekday": arrays["weekday"],
        "y": arrays["y"].astype("float32"),
        "pred": arrays["pred"].astype("float32"),
    })


def _public_cut(res: dict) -> dict:
    """metrics.json 에 넣을 형태(내부 배열·일별 원자료는 뺀다)."""
    out = {k: v for k, v in res.items() if k not in ("_arrays", "daily", "daily_naive")}
    return out


# ───────────────────────── 실행 ─────────────────────────
def _cut_frames(base: pd.DataFrame, cut: Cut) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(cut.cut_date)
    lo = as_of + pd.Timedelta(days=cut.horizon[0])
    hi = as_of + pd.Timedelta(days=cut.horizon[1])
    fit = base.loc[base["date"] <= as_of]
    ev = base.loc[(base["date"] >= lo) & (base["date"] <= hi)]
    return fit, ev


def select_cuts(spec: str | None) -> list[Cut]:
    if not spec:
        return list(CUTS)
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    by_date = {c.cut_date for c in CUTS}
    unknown = sorted(wanted - by_date)
    if unknown:
        raise SystemExit(f"CUTS 에 없는 cut: {unknown}")
    return [c for c in CUTS if c.cut_date in wanted]


def load_splits(guard: DataGuard, purpose_prefix: str = PROTOCOL_ID) -> dict[str, pd.DataFrame]:
    """train+valid 만 읽는다(test 는 승인 경로 밖에서 열지 않는다)."""
    splits = {
        "train": guard.open_split("train", purpose=f"{purpose_prefix} — 학습", columns=USE_COLS),
        "valid": guard.open_split("valid", purpose=f"{purpose_prefix} — 학습/채점", columns=USE_COLS),
    }
    for frame in splits.values():
        frame["date"] = pd.to_datetime(frame["date"])
    return splits


def _run_protocol(args, out_dir: Path, run_id: str) -> int:
    cuts = select_cuts(args.cuts)
    guard = DataGuard(
        ROOT,
        lane=args.lane,
        approval_id=args.approval_id,
        allow_test=False,
    )
    splits = load_splits(guard)
    base = pd.concat(list(splits.values()), ignore_index=True)
    print(f"[{run_id}] 학습+검증(train+valid) {len(base):,}행 · "
          f"{base['date'].min().date()} ~ {base['date'].max().date()} · cut {len(cuts)}개")

    per_cut: list[dict] = []
    daily_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    slice_parts: list[dict] = []
    band_rows_by_cut: dict[str, dict[str, int]] = {}
    names: list[str] = []
    missing: list[str] = []
    categorical = _categorical_choice(args)
    bands_on = _bands_choice(args)
    for cut in cuts:
        fit, ev = _cut_frames(base, cut)
        if fit.empty or ev.empty:
            raise SystemExit(f"cut {cut.cut_date}: 학습 {len(fit):,}행 · 평가 {len(ev):,}행 — 진행 불가")
        features = build_features(
            pd.concat([fit, ev], ignore_index=True), as_of=cut.cut_date, feature_set=args.feature_set
        )
        as_of_ts = pd.Timestamp(cut.cut_date)
        fit_f = _inherit(features.loc[features["date"] <= as_of_ts], features)
        ev_f = _inherit(features.loc[features["date"] > as_of_ts], features)
        ev_f.attrs["model"] = args.model
        if not names:
            names = feature_names(features)
            missing = list(features.attrs.get("missing_features", []))

        if bands_on:
            bands = [b for b in HORIZON_BANDS if b[1] >= cut.horizon[0] and b[0] <= cut.horizon[1]]
            models: dict[tuple[int, int], object] = {}
            rows: dict[str, int] = {}
            tables = asof_profile_tables(fit_f)
            for band in bands:
                band_frame = band_training_frame(
                    features, as_of=cut.cut_date, band=band, seed=SEED,
                    max_rows=args.band_rows, tables=tables
                )
                models[band] = _fit_single(
                    band_frame, params=PARAMS, seed=SEED, categorical=categorical, backend=args.backend
                )
                rows[f"{band[0]}-{band[1]}"] = int(len(band_frame))
                del band_frame
            del tables
            model = BandedModel(models, names)
            band_rows_by_cut[cut.cut_date] = rows
            train_note = " + ".join(f"{k}:{v:,}" for k, v in rows.items())
        else:
            if args.max_rows and len(fit_f) > args.max_rows:
                fit_f = _inherit(fit_f.sample(n=int(args.max_rows), random_state=SEED), features)
            model = fit_model(fit_f, params=PARAMS, seed=SEED,
                              categorical=categorical, backend=args.backend)
            train_note = f"{len(fit_f):,}행"

        res = evaluate_cut(model, ev_f, cut=cut)
        arrays = res.pop("_arrays")
        daily_frames.append(pd.DataFrame(res["daily"]))
        daily_frames.append(pd.DataFrame(res["daily_naive"]))
        pred_frames.append(_predictions_frame(arrays, args.model, cut))
        slice_parts.append(slice_sums(arrays))
        per_cut.append(_public_cut(res))
        print(f"  cut {cut.cut_date} ({cut.kind}, {cut.horizon[0]}-{cut.horizon[1]}일): "
              f"n={res['n']:,} 평균절대오차={res['mae']:.2f} 제곱근평균제곱오차={res['rmse']:.2f} "
              f"가중절대백분율오차={res['wape']:.4f} · 학습 {train_note} · "
              f"{res['date_min']}~{res['date_max']}",
              flush=True)
        del features, fit_f, ev_f, model, arrays

    agg = pooled(per_cut)
    daily = pd.concat(daily_frames, ignore_index=True)
    daily["model"] = daily["model"].astype(str)
    ci = day_block_bootstrap(
        daily[daily["model"] == args.model],
        daily[daily["model"] == "naive_profile_mean"],
        n=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
    )
    baseline_daily, baseline_metrics = _baseline_run(args.baseline_run)
    ci_vs_baseline = _bootstrap_vs_baseline(daily, args.model, baseline_daily)
    slices = combine_slice_sums(slice_parts)
    metrics = {"per_cut": per_cut, "pooled": agg, "ci": ci,
               "ci_vs_baseline": ci_vs_baseline, "slices": slices}
    verdict = _gate_verdict(agg, per_cut, baseline_metrics)

    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    config = {
        "protocol": PROTOCOL_ID,
        "run_id": run_id,
        "model": args.model,
        "feature_set": args.feature_set,
        "params": PARAMS,
        "seeds": {"model": SEED, "bootstrap": BOOTSTRAP_SEED},
        "cuts": [c.cut_date for c in cuts],
        "max_rows": args.max_rows,
        "lane": args.lane,
        "approval_id": args.approval_id,
        "test_accessed": False,
        "data_rows": int(len(base)),
        "parquet_engine": _parquet_engine(),
        "backend": args.backend,
        "categorical": categorical or [],
        "bands": bands_on,
        "band_rows": args.band_rows,
        "band_training_rows": band_rows_by_cut,
        "baseline_run": args.baseline_run,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(out_dir / "config.json", config)
    _dump_json(out_dir / "metrics.json", metrics)
    artifacts = {
        "config": {"path": "config.json", "sha256": _sha256_file(out_dir / "config.json")},
        "metrics": {"path": "metrics.json", "sha256": _sha256_file(out_dir / "metrics.json")},
        "daily_errors": _write_table(daily, out_dir, "daily_errors"),
    }
    if len(predictions):
        artifacts["predictions"] = _write_table(predictions, out_dir, "predictions")
    artifacts["manifest"] = {"path": "manifest.json"}

    horizon_min = min(c.horizon[0] for c in cuts)
    horizon_max = max(c.horizon[1] for c in cuts)
    payload = {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "created_at": _now(),
        "git": _git_state(),
        "source": {
            "eval_protocol.py": _sha256_file(Path(__file__).resolve()),
            "note": "산출물을 만든 코드 지문. 재현하려면 이 값과 일치하는 코드를 써야 한다.",
        },
        "task": args.task or ("T1" if any(c.kind == "long" for c in cuts) else "T2"),
        "task_scope": {
            "long_cuts": sum(1 for c in cuts if c.kind == "long"),
            "short_cuts": sum(1 for c in cuts if c.kind == "short"),
            "note": "T1=장기(182~366일), T2=단기(1~181일). "
                    "혼합 실행은 긴 지평이 있으면 T1 로 적고 여기에 분포를 남긴다.",
        },
        "horizon": {"min": horizon_min, "max": horizon_max},
        "data": _data_meta(guard, ["train", "valid"], splits),
        "features": {
            "names": names,
            "order_hash": feature_order_hash(names),
            "order_hash_rule": "sha256-newline-join-v1",
            "as_of_rule": AS_OF_RULE,
            "feature_set": args.feature_set,
            "missing_features": missing,
            "categorical": categorical or [],
            "encoder": _encoder_note(args.backend, categorical),
            "lag_rule": LAG_RULE if args.feature_set == "matched" else None,
        },
        "bands": {
            "bands": [list(b) for b in HORIZON_BANDS],
            "spec_bands": [list(b) for b in SPEC_BANDS],
            "extension": list(LONG_BAND) if bands_on else None,
            "rule": BAND_RULE,
            "training_rows": band_rows_by_cut,
            "models_per_cut": {cut.cut_date: sorted(band_rows_by_cut.get(cut.cut_date, {}))
                               for cut in cuts},
        },
        "params": PARAMS,
        "seeds": {"model": SEED, "bootstrap": BOOTSTRAP_SEED},
        "protocol": PROTOCOL_ID,
        "cuts": per_cut,
        "metrics": metrics,
        "verdict": verdict,
        "artifacts": artifacts,
    }
    path = write_manifest(out_dir, payload)
    _append_run_log(out_dir, {
        "run_id": run_id,
        "protocol": PROTOCOL_ID,
        "feature_set": args.feature_set,
        "backend": args.backend,
        "bands": bands_on,
        "pooled_mae": agg["mae"],
        "baseline_mae": BASELINE_POOLED_MAE,
        "gate_mae": GATE_POOLED_MAE,
        "delta_pct": verdict["gate"].get("delta_pct"),
        "verdict": verdict["result"],
        "reason": verdict.get("reason"),
        "created_at": _now(),
    })
    guard.assert_unchanged()

    print(f"\n통합(pooled) n={agg['n']:,} 평균절대오차(MAE)={agg['mae']:.2f} "
          f"제곱근평균제곱오차(RMSE)={agg['rmse']:.2f} "
          f"가중절대백분율오차(WAPE)={agg['wape']:.4f} 편향(bias)={agg['bias']:+.2f} "
          f"음수예측={agg['negative_predictions']:,}")
    if "naive" in agg:
        print(f"  단순평균(naive) MAE={agg['naive']['mae']:.2f} · "
              f"차이 {ci['mean_diff_b_minus_a']:+.2f} "
              f"(95% 신뢰구간 {ci['ci95'][0]:+.2f}~{ci['ci95'][1]:+.2f}, "
              f"p(모델 우세)={ci['p_b_better']:.3f})")
    if ci_vs_baseline.get("available"):
        print(f"  기준선({BASELINE_POOLED_MAE:.4f}) 대비 {ci_vs_baseline['mean_diff_b_minus_a']:+.2f} "
              f"(95% 신뢰구간 {ci_vs_baseline['ci95'][0]:+.2f}~{ci_vs_baseline['ci95'][1]:+.2f}, "
              f"p(개선)={ci_vs_baseline['p_b_better']:.3f})")
    else:
        print(f"  기준선 대비 신뢰구간: {ci_vs_baseline.get('note')}")
    print(f"  판정 {verdict['result']}: {verdict.get('reason')}")
    print(f"저장: {path.parent} (산출물 명세(manifest) · 지표(metrics) · "
          f"{artifacts['daily_errors']['format']} · "
          f"{artifacts.get('predictions', {}).get('format', '예측 파일 없음')})")
    return 0 if verdict["result"] != "rejected" else 2


BASELINE_POOLED_MAE = 93.3281
GATE_POOLED_MAE = 88.66
GATE_RULE = (
    "개발 구간 통합 평균절대오차(pooled dev MAE) ≤ 88.66 (기준선 93.3281 대비 −5%). "
    "두 실행의 차이가 평균절대오차 2 미만이면 동률로 본다. "
    "미달이면 수치를 고치지 않고 음성 결과로 남긴다."
)
DEFAULT_BASELINE_RUN = "models/_exp/20261114-P0-R0-01"


def _categorical_choice(args) -> list[str] | None:
    if args.categorical == "off":
        return None
    if args.categorical == "on" or args.feature_set in ("horizon", "matched"):
        return [STATION_CODE, *CATEGORICAL_FEATURES]
    return None


def _bands_choice(args) -> bool:
    if args.bands == "on":
        return True
    if args.bands == "off":
        return False
    return args.feature_set == "horizon"


def _encoder_note(backend: str, categorical: list[str] | None) -> dict:
    if not categorical:
        return {"backend": backend, "categorical": "없음(수치 그대로)"}
    if backend == "lightgbm":
        return {"backend": backend, "categorical": "LightGBM 기본 범주 처리",
                "columns": list(categorical)}
    return {
        "backend": backend,
        "categorical": "HistGradientBoosting 범주 피처 지정(범주 열 이름 전달)",
        "columns": list(categorical),
        "station_code": (
            f"범주 가짓수가 상한(max_bins, 최대 {HGB_MAX_CARDINALITY})을 넘으면 "
            "(몫, 나머지) 두 범주 열로 무손실 분해한다"
        ),
    }


def _baseline_run(rel: str | None) -> tuple[pd.DataFrame | None, dict | None]:
    """기준선 실행의 일별 오차표와 지표를 읽는다(없으면 None)."""
    if not rel:
        return None, None
    base = Path(rel)
    if not base.is_absolute():
        base = ROOT / base
    daily = None
    metrics = None
    daily_path = base / "daily_errors.csv.gz"
    if daily_path.exists():
        daily = pd.read_csv(daily_path, compression="gzip")
        daily["date"] = pd.to_datetime(daily["date"])
    metrics_path = base / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return daily, metrics


def _bootstrap_vs_baseline(daily: pd.DataFrame, model_name: str,
                           baseline_daily: pd.DataFrame | None) -> dict:
    if baseline_daily is None or baseline_daily.empty:
        return {"available": False, "note": "기준선 일별 오차표가 없어 구간 비교를 하지 않았다"}
    ours = daily[daily["model"] == model_name]
    theirs = baseline_daily[baseline_daily["model"] == model_name]
    if ours.empty or theirs.empty:
        return {"available": False, "note": "두 실행의 일별 표에 같은 모델 이름이 없다"}
    result = day_block_bootstrap(theirs, ours, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    result["available"] = True
    result["note"] = "a=기준선 · b=이 실행. mean_diff_b_minus_a 가 음수면 이 실행이 낮다(좋다)."
    return result


def _gate_verdict(agg: dict, per_cut: list[dict], baseline_metrics: dict | None) -> dict:
    mae = float(agg["mae"])
    delta = mae - BASELINE_POOLED_MAE
    delta_pct = delta / BASELINE_POOLED_MAE * 100.0
    passed = mae <= GATE_POOLED_MAE
    gate = {
        "rule": GATE_RULE,
        "baseline_mae": BASELINE_POOLED_MAE,
        "gate_mae": GATE_POOLED_MAE,
        "observed_mae": mae,
        "delta": delta,
        "delta_pct": delta_pct,
        "tie_band_mae": 2.0,
        "passed": bool(passed),
    }
    per_cut_delta = {}
    if baseline_metrics:
        base_by_cut = {c["cut_date"]: float(c["mae"]) for c in baseline_metrics.get("per_cut", [])}
        for res in per_cut:
            base = base_by_cut.get(res["cut_date"])
            if base is None:
                continue
            per_cut_delta[res["cut_date"]] = {
                "baseline_mae": base,
                "observed_mae": float(res["mae"]),
                "delta": float(res["mae"]) - base,
                "tie": abs(float(res["mae"]) - base) < 2.0,
            }
    gate["per_cut"] = per_cut_delta
    bands = {}
    for res in per_cut:
        for label, stats in (res.get("horizon_band") or {}).items():
            bands.setdefault(label, {"n": 0, "abs_err_sum": 0.0})
            bands[label]["n"] += int(stats["n"])
            bands[label]["abs_err_sum"] += float(stats["mae"]) * int(stats["n"])
    gate["bands"] = {label: {"n": v["n"], "mae": v["abs_err_sum"] / v["n"]}
                     for label, v in bands.items()}
    if not passed:
        reason = (
            f"통합 평균절대오차(pooled MAE) {mae:.4f} 가 임계 {GATE_POOLED_MAE:.2f} 를 넘었다 "
            f"(기준선 {BASELINE_POOLED_MAE:.4f} 대비 {delta:+.4f}, {delta_pct:+.2f}%). "
            "지평 밴드별 악화 여부는 기준선 밴드 지표가 없어 이 판정에 넣지 않았다."
        )
        return {"result": "rejected", "reason": reason, "gate": gate}
    reason = (f"통합 평균절대오차(pooled MAE) {mae:.4f} ≤ {GATE_POOLED_MAE:.2f} "
              f"(기준선 대비 {delta:+.4f}, {delta_pct:+.2f}%)")
    return {"result": "adopt", "reason": reason, "gate": gate}


def _log_has_run(path: Path, run_id: str) -> bool:
    """로그에 같은 실행 식별자의 기록이 이미 있는지 확인한다."""
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("run_id") == run_id:
            return True
    return False


def _append_run_log(run_dir: Path, record: dict) -> Path:
    """run 디렉터리에 판정 한 줄을 덧붙인다(덮어쓰지 않는다).

    같은 실행 식별자의 기록이 이미 있으면 덧붙이지 않는다 — 재판정이 같은 run_id 로 다시
    돌아도 판정 줄이 두 개로 늘지 않게 한다.
    """
    path = Path(run_dir) / "log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = record.get("run_id")
    if run_id is not None and _log_has_run(path, str(run_id)):
        print(f"  판정 줄: 이미 기록된 run_id({run_id}) — 덧붙이지 않았다")
        return path
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


RUN_SEQ_RE = re.compile(r"^(?P<head>.*-)(?P<seq>[0-9]{2})$")
RUN_ARTIFACT_NAMES = {"manifest.json", "metrics.json", "config.json", "log.jsonl"}


def existing_artifacts(run_dir: Path) -> list[str]:
    """산출물이 이미 있는 run 디렉터리인지 확인한다(같은 식별자 재사용을 막는 근거)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    found = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name in RUN_ARTIFACT_NAMES or name.startswith("daily_errors.") \
                or name.startswith("predictions."):
            found.append(name)
    return found


def next_free_run_id(run_dir: Path) -> str:
    """같은 접두사에서 산출물이 없는 순번을 찾는다(충돌 시 안내용)."""
    run_dir = Path(run_dir)
    match = RUN_SEQ_RE.match(run_dir.name)
    if not match:
        return run_dir.name
    head = match.group("head")
    seq = int(match.group("seq"))
    for candidate in range(seq + 1, 100):
        candidate_id = f"{head}{candidate:02d}"
        if not existing_artifacts(run_dir.parent / candidate_id):
            return candidate_id
    raise ValueError(f"순번이 99를 넘었다 — 다른 날짜·회차 접두사를 쓴다: {run_dir.parent}")


def _parquet_engine() -> str | None:
    for name in ("pyarrow", "fastparquet"):
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    return None


def write_legacy_evidence(run_dir: Path, approval_id: str, *, note: str = "") -> Path:
    """승인 실행의 산출물(manifest·metrics)에서 증거 파일을 만든다 — 추가 데이터 접근이 없다."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    per_cut = metrics["per_cut"][0]
    try:
        manifest_rel = str(run_dir.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        manifest_rel = str(run_dir)
    evidence = {
        "approval_id": approval_id,
        "run_id": manifest["run_id"],
        "created_at": _now(),
        "protocol": manifest["protocol"],
        "cut": {
            "cut_date": per_cut["cut_date"],
            "horizon": per_cut["horizon"],
            "days": per_cut["days"],
            "n": per_cut["n"],
        },
        "observed_mae": per_cut["mae"],
        "observed_rmse": per_cut["rmse"],
        "observed_wape": per_cut["wape"],
        "target_mae": LEGACY_TARGET_MAE,
        "tolerance": LEGACY_TOLERANCE,
        "delta": per_cut["mae"] - LEGACY_TARGET_MAE,
        "verdict": manifest["verdict"],
        "slices": metrics.get("legacy_slices", {}),
        "horizon_bucket": per_cut["horizon_bucket"],
        "access_log": "models/_exp/access.jsonl",
        "manifest": manifest_rel,
        "source": "manifest.json + metrics.json (재실행 없이 기록)",
    }
    if note:
        evidence["note"] = note
    return _dump_json(ROOT / "models" / "_exp" / "legacy_b_baseline.json", evidence)


def _run_legacy(args, out_dir: Path, run_id: str) -> int:
    """승인 1회 — train+valid 학습 · 2026-02-14 프로파일 고정 · Test 136일 채점(B base)."""
    approval = args.approval_id or LEGACY_APPROVAL_ID
    guard = DataGuard(ROOT, lane=args.lane, approval_id=approval, allow_test=True)
    learn_note = "이전 기준선(legacy) B 재현 — B 평가(Test) 평균절대오차 92.72 확인(학습)"
    test_note = "이전 기준선(legacy) B 재현 — B 평가(Test) 평균절대오차 92.72 확인(채점, 승인 1회)"
    frames = {
        "train": guard.open_split("train", purpose=learn_note, columns=USE_COLS),
        "valid": guard.open_split("valid", purpose=learn_note, columns=USE_COLS),
        "test_dev": guard.open_split("test_dev", purpose=test_note, columns=USE_COLS),
        "test_final": guard.open_split("test_final", purpose=test_note, columns=USE_COLS),
    }
    base = pd.concat([frames[k] for k in ("train", "valid", "test_dev", "test_final")], ignore_index=True)
    base["date"] = pd.to_datetime(base["date"])
    cut = LEGACY_CUT
    print(f"[{run_id}] 이전 기준선(legacy) B 재현 — 학습 "
          f"{len(frames['train']) + len(frames['valid']):,}행 · "
          f"평가(test) {len(frames['test_dev']) + len(frames['test_final']):,}행")

    features = pd.concat(
        [base.loc[base["date"] <= pd.Timestamp(cut.cut_date)],
         base.loc[base["date"] > pd.Timestamp(cut.cut_date)]],
        ignore_index=True,
    )
    features = build_features(features, as_of=cut.cut_date, feature_set="base")
    as_of_ts = pd.Timestamp(cut.cut_date)
    fit_f = _inherit(features.loc[features["date"] <= as_of_ts], features)
    ev_f = _inherit(features.loc[features["date"] > as_of_ts], features)
    ev_f.attrs["model"] = "B"

    model = fit_model(fit_f, params=PARAMS, seed=SEED)
    res = evaluate_cut(model, ev_f, cut=cut)
    arrays = res.pop("_arrays")
    daily = pd.DataFrame(res["daily"])
    daily_naive = pd.DataFrame(res["daily_naive"])
    preds = _predictions_frame(arrays, "B", cut)

    horizon = arrays["horizon"]
    dev_mask = horizon <= 75
    slices = {
        "test_dev": score(arrays["y"][dev_mask], arrays["pred"][dev_mask], horizon[dev_mask]),
        "test_final": score(arrays["y"][~dev_mask], arrays["pred"][~dev_mask], horizon[~dev_mask]),
    }
    mae = float(res["mae"])
    delta = mae - LEGACY_TARGET_MAE
    verdict = {
        "result": "pass" if abs(delta) <= LEGACY_TOLERANCE else "fail",
        "gate": "이전 기준선(legacy) B 평가 평균절대오차 재현",
        "target_mae": LEGACY_TARGET_MAE,
        "tolerance": LEGACY_TOLERANCE,
        "observed_mae": mae,
        "delta": delta,
    }
    metrics = {"per_cut": [_public_cut(res)], "pooled": pooled([_public_cut(res)]),
               "ci": day_block_bootstrap(daily, daily_naive, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED),
               "legacy_slices": slices}
    config = {
        "protocol": PROTOCOL_ID,
        "run_id": run_id,
        "mode": "legacy-baseline",
        "model": "B",
        "feature_set": "base",
        "params": PARAMS,
        "seeds": {"model": SEED, "bootstrap": BOOTSTRAP_SEED},
        "approval_id": approval,
        "test_accessed": True,
        "parquet_engine": _parquet_engine(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(out_dir / "config.json", config)
    _dump_json(out_dir / "metrics.json", metrics)
    artifacts = {
        "config": {"path": "config.json", "sha256": _sha256_file(out_dir / "config.json")},
        "metrics": {"path": "metrics.json", "sha256": _sha256_file(out_dir / "metrics.json")},
        "daily_errors": _write_table(daily, out_dir, "daily_errors"),
        "predictions": _write_table(preds, out_dir, "predictions"),
    }
    artifacts["manifest"] = {"path": "manifest.json"}
    payload = {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "created_at": _now(),
        "git": _git_state(),
        "source": {
            "eval_protocol.py": _sha256_file(Path(__file__).resolve()),
            "note": "산출물을 만든 코드 지문. 재현하려면 이 값과 일치하는 코드를 써야 한다.",
        },
        "task": "T2",
        "horizon": {"min": cut.horizon[0], "max": cut.horizon[1]},
        "data": _data_meta(guard, ["train", "valid", "test_dev", "test_final"], frames),
        "features": {
            "names": feature_names(features),
            "order_hash": feature_order_hash(feature_names(features)),
            "order_hash_rule": "sha256-newline-join-v1",
            "as_of_rule": AS_OF_RULE,
            "feature_set": "base",
            "missing_features": [],
        },
        "params": PARAMS,
        "seeds": {"model": SEED, "bootstrap": BOOTSTRAP_SEED},
        "protocol": PROTOCOL_ID,
        "cuts": [_public_cut(res)],
        "metrics": metrics,
        "verdict": verdict,
        "artifacts": artifacts,
    }
    path = write_manifest(out_dir, payload)
    write_legacy_evidence(out_dir, approval)
    guard.assert_unchanged()

    print(f"  B 평가(Test) 평균절대오차={mae:.2f} (목표 {LEGACY_TARGET_MAE} ±{LEGACY_TOLERANCE}, "
          f"차이 {delta:+.2f}) → {verdict['result'].upper()}")
    print(f"  개발 확인(test_dev) 평균절대오차={slices['test_dev']['mae']:.2f} · "
          f"최종 확인(test_final) 평균절대오차={slices['test_final']['mae']:.2f}")
    print(f"저장: {path.parent} · 증거 models/_exp/legacy_b_baseline.json")
    return 0 if verdict["result"] == "pass" else 3


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="평가 프로토콜 (rolling8-v1)")
    parser.add_argument("--protocol", default=PROTOCOL_ID)
    parser.add_argument("--model", default="B")
    parser.add_argument("--feature-set", default="base", choices=["base", "matched", "horizon"])
    parser.add_argument("--out", required=True, help="models/_exp/<run_id>")
    parser.add_argument("--approval-id", default=None)
    parser.add_argument("--cuts", default=None, help="쉼표로 구분한 cut_date 목록(기본 8개 전부)")
    parser.add_argument("--max-rows", type=int, default=None, help="cut 별 학습 행 상한(표본)")
    parser.add_argument("--lane", default=os.environ.get("MACHINE_ID", "unknown"))
    parser.add_argument("--task", default=None, choices=["T1", "T2"])
    parser.add_argument("--legacy-baseline", action="store_true", help="승인 1회 이전 기준선(legacy) B 재현")
    parser.add_argument("--backend", default="hgb", choices=["hgb", "lightgbm"],
                        help="비교용 백엔드(기본 hgb)")
    parser.add_argument("--categorical", default="auto", choices=["auto", "on", "off"],
                        help="범주 피처 처리(auto: horizon·matched 에서 켬)")
    parser.add_argument("--bands", default="auto", choices=["auto", "on", "off"],
                        help="지평 밴드별 모델(auto: horizon 에서 켬)")
    parser.add_argument("--band-rows", type=int, default=None,
                        help="밴드별 학습 행 상한(기본 없음 = 전부)")
    parser.add_argument("--baseline-run", default=DEFAULT_BASELINE_RUN,
                        help="비교 기준선 실행 디렉터리(일별 오차표·지표)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    run_id = out_dir.name
    if not RUN_ID_RE.match(run_id):
        raise SystemExit(f"run_id 형식이 틀렸다: {run_id!r} — 예: 20261114-P0-R0-01")
    if args.protocol != PROTOCOL_ID:
        raise SystemExit(f"모르는 protocol: {args.protocol} ({PROTOCOL_ID} 만 지원)")
    if args.model != "B":
        raise SystemExit("이 스크립트는 B 피처 계약만 쓴다 — 피처를 바꾸지 않는다")
    used = existing_artifacts(out_dir)
    if used:
        raise SystemExit(
            f"run_id 충돌: {out_dir} 에 이미 산출물이 있다({', '.join(used)}) — 덮어쓰지 않는다. "
            f"비어 있는 경로: {out_dir.parent / next_free_run_id(out_dir)}"
        )
    if args.legacy_baseline:
        return _run_legacy(args, out_dir, run_id)
    return _run_protocol(args, out_dir, run_id)


if __name__ == "__main__":
    raise SystemExit(main())