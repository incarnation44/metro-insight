"""불확실성 — 분할 등각(split conformal) 예측구간 · 조건부 커버리지 · VaR 백테스트.

설계
    * 점예측(B) 위에 얹는 후처리다. 모델을 바꾸지 않는다(모델 비의존).
    * 비순응도 점수 s = |잔차| / scale,  scale = ``profile_month_mean`` (규모 정규화).
    * 그룹 = 지평밴드 4개 × 규모 tertile 3개 = 12개.
      지평밴드는 (1-31, 32-91, 92-181, 182-366) 이며 ``calendar_features.HORIZON_BUCKETS`` 와 같은 값이다.
    * 밴드마다 보정 원점을 여러 개 둬서 잔차 구간 길이를 ``CAL_WINDOW_DAYS``(90일)로 맞춘다.
      원점 하나만 두면 1-31 밴드의 창이 한 달뿐이라 그 달의 날씨·수준 이탈이 분위수를 지배한다
      (실측: 같은 그룹에서 보정 잔차가 평가 잔차의 3.3배까지 벌어졌다).
    * 분위수는 유한표본 등각 분위수 ceil((n+1)·nominal)/n 을 쓴다(표본이 작을수록 보수적).
      그룹 표본이 ``MIN_GROUP_N`` 미만이면 밴드 단위 → 전체(pooled) 순으로 폴백한다.
    * 보정 잔차에는 나이 가중치 w = 2^(−age/half_life) 를 준다(age = 평가 기준 시점 − 잔차 발생일).
      창 길이가 같아도 최근 잔차가 더 무겁다. 가중 분위수는 점수를 안정 정렬(mergesort)한 뒤
      누적 가중 비율이 유한표본 수준을 처음 넘는 점수로 정의해 결정론을 지킨다.
      가중치를 주지 않으면 종전 비가중 경로와 결과가 같다.
    * 밴드 사이의 분위수 단조 제약은 강제하지 않는다. 밴드마다 보정 원점이 달라 분위수는
      지평뿐 아니라 그 창의 잔차 수준을 반영한다. 강제하면 짧은 지평 밴드의 값이 긴 지평
      밴드로 전파되어 구간이 과도하게 넓어진다(실측: 80% 구간 포함률 0.86 → 0.95).
    * 음수 보정: 점예측을 0 에서 클램프하고, 하한이 0 미만이면 폭을 유지한 채 구간을 오른쪽으로 민다.
    * 분위 교차가 생기는 경로(분위수 HGB 대조군 등)는 :func:`monotone_quantiles` 로 반드시 정렬한다.

표기 원칙
    * 여기서 만드는 것은 **명목(nominal) 포함률을 가진 등각 예측구간**이다. "신뢰구간"이라 부르지 않는다.
      신뢰구간은 모수에 대한 것이고, 이 구간은 미래 관측값에 대한 것이다.
    * 실측 포함률(PICP)은 :func:`conditional_coverage` · :func:`var_backtest` 로 따로 보고한다.
      명목값과 실측값이 다른 것을 숨기지 않는다.
    * 포함률 보장은 교환가능성 가정의 결과다. 계절·개통·제도 변화가 있는 이 데이터에서는 근사로만 읽는다.

의존성: numpy · scipy(기존 의존성)만 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[3]

# ───────────────────────── 설계 동결값 ─────────────────────────
NOMINAL = 0.8
CAL_WINDOW_DAYS = 90
HORIZON_BANDS: tuple[tuple[int, int], ...] = ((1, 31), (32, 91), (92, 181), (182, 366))
TERTILE_NAMES: tuple[str, str, str] = ("small", "medium", "large")
MIN_GROUP_N = 50
SCORE_KIND = "abs_err_over_scale"
CALIBRATOR_SCHEMA = "conformal-calibrator/1"
FLOOR = 0.0
DECAY_HALF_LIFE_DAYS = 30.0
WEIGHTING_TIME_DECAY = "time_decay"
WEIGHTING_NONE = "none"
WEIGHTING_DEFAULT = WEIGHTING_TIME_DECAY


# ───────────────────────── 구간 ─────────────────────────
@dataclass(frozen=True)
class Interval:
    """한 점예측의 예측구간. nominal 은 명목 포함률(0.8 = 80%), group 은 그룹 라벨."""

    lo: float
    hi: float
    nominal: float
    group: str

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


# ───────────────────────── 분위수 정렬 ─────────────────────────
def monotone_quantiles(q: np.ndarray) -> np.ndarray:
    """분위수 배열의 마지막 축을 오름차순으로 정렬해 교차(crossing)를 없앤다.

    분위수 회귀는 상위 분위가 하위 분위보다 작아지는 교차가 흔하다(문헌 1.6~2.1%).
    구간을 만들기 전에 정렬하지 않으면 lo > hi 인 구간이 나온다. 입력은 바꾸지 않는다.
    """
    arr = np.asarray(q, dtype="float64")
    if arr.ndim == 0:
        raise ValueError("분위수 배열은 1차원 이상이어야 한다")
    return np.sort(arr, axis=-1)


def count_crossings(q: np.ndarray) -> int:
    """마지막 축에서 바로 옆 분위수가 역전된 쌍의 수(정렬 전 진단용)."""
    arr = np.asarray(q, dtype="float64")
    if arr.shape[-1] < 2:
        return 0
    return int((np.diff(arr, axis=-1) < 0).sum())


# ───────────────────────── 등각 보정기 ─────────────────────────
class ConformalCalibrator:
    """그룹별 비순응도 분위수 표. fit 으로 만들고 interval 로 구간을 낸다.

    groups 는 "지평밴드|규모tertile" 형태의 문자열 라벨을 전제하지만, 다른 라벨도 동작한다
    (밴드 단위 폴백은 '|' 앞부분을 밴드 이름으로 본다).
    """

    def __init__(self, *, nominal: float = NOMINAL, min_group_n: int = MIN_GROUP_N) -> None:
        if not 0.0 < nominal < 1.0:
            raise ValueError(f"nominal 은 0과 1 사이여야 한다: {nominal}")
        self.nominal = float(nominal)
        self.min_group_n = int(min_group_n)
        self.quantiles: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.band_quantiles: dict[str, float] = {}
        self.band_counts: dict[str, int] = {}
        self.pooled_q: float = float("nan")
        self.pooled_n: int = 0
        self.scale_floor: float = float("nan")
        self.weighting: str = WEIGHTING_NONE
        self.meta: dict[str, Any] = {}

    # ── 적합 ──
    def fit(
        self,
        residuals: np.ndarray,
        scales: np.ndarray,
        groups: np.ndarray,
        *,
        nominal: float = NOMINAL,
        weights: np.ndarray | None = None,
    ) -> ConformalCalibrator:
        """잔차 · 규모 · 그룹 라벨로 그룹별 등각 분위수를 만든다.

        scores = |residual| / scale. scale 이 0 이하이거나 NaN 인 행은 규모 중앙값으로 대체한다.
        weights 를 주면 각 잔차의 무게를 반영한 가중 분위수를 쓴다(주지 않으면 비가중).
        """
        if not 0.0 < nominal < 1.0:
            raise ValueError(f"nominal 은 0과 1 사이여야 한다: {nominal}")
        self.nominal = float(nominal)
        res = np.asarray(residuals, dtype="float64").ravel()
        sc = np.asarray(scales, dtype="float64").ravel()
        gr = np.asarray(groups).ravel().astype(str)
        if not (len(res) == len(sc) == len(gr)):
            raise ValueError(f"길이가 다르다: residuals={len(res)} scales={len(sc)} groups={len(gr)}")
        if len(res) == 0:
            raise ValueError("보정 표본이 비었다")
        if weights is None:
            self.weighting = WEIGHTING_NONE
            wt = None
        else:
            wt = np.asarray(weights, dtype="float64").ravel()
            if len(wt) != len(res):
                raise ValueError(f"가중치 길이가 다르다: weights={len(wt)} residuals={len(res)}")
            wt = np.where(np.isfinite(wt) & (wt > 0), wt, 0.0)
            if float(wt.sum()) <= 0.0:
                raise ValueError("가중치 합이 0 이다")
            self.weighting = WEIGHTING_TIME_DECAY

        self.scale_floor = _median_positive(sc)
        scores = np.abs(res) / _sanitize_scale(sc, self.scale_floor)

        self.pooled_n = int(len(scores))
        self.pooled_q = (
            _conformal_quantile(scores, self.nominal)
            if wt is None
            else _weighted_conformal_quantile(scores, wt, self.nominal)
        )

        # 그룹별
        order = np.argsort(gr, kind="mergesort")
        gr_sorted = gr[order]
        sc_sorted = scores[order]
        wt_sorted = None if wt is None else wt[order]
        uniq, starts = np.unique(gr_sorted, return_index=True)
        bounds = list(starts) + [len(gr_sorted)]
        for i, name in enumerate(uniq):
            part = sc_sorted[bounds[i]:bounds[i + 1]]
            self.counts[str(name)] = int(len(part))
            self.quantiles[str(name)] = (
                _conformal_quantile(part, self.nominal)
                if wt_sorted is None
                else _weighted_conformal_quantile(
                    part, wt_sorted[bounds[i]:bounds[i + 1]], self.nominal
                )
            )

        # 밴드 단위 폴백용 (라벨의 '|' 앞부분)
        bands = np.array([g.split("|")[0] for g in gr], dtype=str)
        order_b = np.argsort(bands, kind="mergesort")
        bands_sorted = bands[order_b]
        sc_b = scores[order_b]
        wt_b = None if wt is None else wt[order_b]
        uniq_b, starts_b = np.unique(bands_sorted, return_index=True)
        bounds_b = list(starts_b) + [len(bands_sorted)]
        for i, name in enumerate(uniq_b):
            part = sc_b[bounds_b[i]:bounds_b[i + 1]]
            self.band_counts[str(name)] = int(len(part))
            self.band_quantiles[str(name)] = (
                _conformal_quantile(part, self.nominal)
                if wt_b is None
                else _weighted_conformal_quantile(
                    part, wt_b[bounds_b[i]:bounds_b[i + 1]], self.nominal
                )
            )
        return self

    # ── 조회 ──
    def quantile(self, group: str, *, band: str | None = None) -> float:
        """그룹 분위수. 표본 부족이면 밴드 → 전체 순으로 폴백한다."""
        key = str(group)
        if self.counts.get(key, 0) >= self.min_group_n:
            return self.quantiles[key]
        band_name = band if band is not None else key.split("|")[0]
        if self.band_counts.get(band_name, 0) >= self.min_group_n:
            return self.band_quantiles[band_name]
        return self.pooled_q

    def source(self, group: str) -> str:
        """이 그룹이 실제로 쓴 분위수의 출처: group · band · pooled."""
        key = str(group)
        if self.counts.get(key, 0) >= self.min_group_n:
            return "group"
        if self.band_counts.get(key.split("|")[0], 0) >= self.min_group_n:
            return "band"
        return "pooled"

    def interval(self, point: float, scale: float, group: str) -> Interval:
        """점예측 · 규모 · 그룹 → 예측구간. 하한은 0 미만으로 내려가지 않는다."""
        q = self.quantile(group)
        width = q * _sanitize_scale(np.array([scale], dtype="float64"), self.scale_floor)[0]
        centre = max(float(point), FLOOR)
        lo = centre - width
        hi = centre + width
        if lo < FLOOR:
            hi += FLOOR - lo  # 폭을 유지한 채 오른쪽으로 민다
            lo = FLOOR
        return Interval(lo=float(lo), hi=float(hi), nominal=self.nominal, group=str(group))

    def interval_arrays(
        self, points: np.ndarray, scales: np.ndarray, groups: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """interval 과 같은 규칙의 벡터화 판(수백만 행용)."""
        pts = np.asarray(points, dtype="float64").ravel()
        sc = _sanitize_scale(np.asarray(scales, dtype="float64").ravel(), self.scale_floor)
        gr = np.asarray(groups).ravel().astype(str)
        if not (len(pts) == len(sc) == len(gr)):
            raise ValueError("points · scales · groups 길이가 다르다")
        uniq, inv = np.unique(gr, return_inverse=True)
        q_by_group = np.array([self.quantile(u) for u in uniq], dtype="float64")
        width = q_by_group[inv] * sc
        centre = np.maximum(pts, FLOOR)
        lo = centre - width
        hi = centre + width
        shift = np.maximum(FLOOR - lo, 0.0)
        return np.maximum(lo + shift, FLOOR), hi + shift

    # ── 직렬화 ──
    def to_json(self) -> dict:
        return {
            "schema": CALIBRATOR_SCHEMA,
            "nominal": self.nominal,
            "score": SCORE_KIND,
            "weighting": self.weighting,
            "min_group_n": self.min_group_n,
            "scale_floor": _finite_or_none(self.scale_floor),
            "pooled": {"q": _finite_or_none(self.pooled_q), "n": self.pooled_n},
            "bands": {
                name: {"q": self.band_quantiles[name], "n": self.band_counts[name]}
                for name in sorted(self.band_quantiles)
            },
            "groups": {
                name: {
                    "q": self.quantiles[name],
                    "n": self.counts[name],
                    "source": self.source(name),
                }
                for name in sorted(self.quantiles)
            },
            "meta": _jsonable(self.meta),
        }

    @classmethod
    def from_json(cls, payload: dict) -> ConformalCalibrator:
        if not isinstance(payload, dict):
            raise TypeError("calibrator payload 는 dict 여야 한다")
        schema = payload.get("schema", CALIBRATOR_SCHEMA)
        if schema != CALIBRATOR_SCHEMA:
            raise ValueError(f"모르는 schema: {schema}")
        cal = cls(
            nominal=float(payload.get("nominal", NOMINAL)),
            min_group_n=int(payload.get("min_group_n", MIN_GROUP_N)),
        )
        cal.weighting = str(payload.get("weighting", WEIGHTING_NONE))
        floor = payload.get("scale_floor")
        cal.scale_floor = float(floor) if floor is not None else float("nan")
        pooled = payload.get("pooled") or {}
        cal.pooled_q = float(pooled.get("q", float("nan")))
        cal.pooled_n = int(pooled.get("n", 0))
        for name, body in (payload.get("bands") or {}).items():
            cal.band_quantiles[str(name)] = float(body["q"])
            cal.band_counts[str(name)] = int(body["n"])
        for name, body in (payload.get("groups") or {}).items():
            cal.quantiles[str(name)] = float(body["q"])
            cal.counts[str(name)] = int(body["n"])
        cal.meta = dict(payload.get("meta") or {})
        return cal


# ───────────────────────── 커버리지 ─────────────────────────
def conditional_coverage(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    slices: dict[str, np.ndarray],
) -> dict:
    """전체 + 슬라이스별 실측 포함률(PICP)과 구간폭 중앙값.

    slices 는 이름 → 불리언 마스크. 반환은 평평한 dict 이며 키 "all" 은 예약어다.
    평균 포함률은 위반을 숨기므로(심야·소규모·공휴일) 슬라이스를 반드시 함께 본다.
    """
    yv = np.asarray(y, dtype="float64").ravel()
    lov = np.asarray(lo, dtype="float64").ravel()
    hiv = np.asarray(hi, dtype="float64").ravel()
    if not (len(yv) == len(lov) == len(hiv)):
        raise ValueError(f"길이가 다르다: y={len(yv)} lo={len(lov)} hi={len(hiv)}")
    inside = (yv >= lov) & (yv <= hiv)
    width = hiv - lov
    out: dict[str, dict] = {"all": _coverage_stats(inside, width)}
    for name, mask in slices.items():
        if name == "all":
            raise ValueError("슬라이스 이름 'all' 은 예약어다")
        m = np.asarray(mask, dtype=bool).ravel()
        if len(m) != len(yv):
            raise ValueError(f"슬라이스 {name} 길이가 y 와 다르다: {len(m)} vs {len(yv)}")
        out[str(name)] = _coverage_stats(inside, width, mask=m)
    return out


# ───────────────────────── VaR 백테스트 ─────────────────────────
def var_backtest(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, nominal: float) -> dict:
    """구간 이탈을 VaR 위반으로 보고 Kupiec POF · Christoffersen 독립성 검정을 한다.

    명목 포함률 nominal 이면 기대 이탈률은 1-nominal 이다. p_value 가 작으면 "이탈률이 명목과
    다르다"(Kupiec) 또는 "이탈이 몰려 있다"(Christoffersen)는 뜻이다. 평균 포함률만 보면
    후자를 놓친다.
    """
    if not 0.0 < nominal < 1.0:
        raise ValueError(f"nominal 은 0과 1 사이여야 한다: {nominal}")
    yv = np.asarray(y, dtype="float64").ravel()
    lov = np.asarray(lo, dtype="float64").ravel()
    hiv = np.asarray(hi, dtype="float64").ravel()
    if not (len(yv) == len(lov) == len(hiv)):
        raise ValueError("y · lo · hi 길이가 다르다")
    viol = ((yv < lov) | (yv > hiv)).astype("int8")
    n = int(len(viol))
    x = int(viol.sum())
    rate = x / n if n else float("nan")
    p0 = 1.0 - nominal

    kupiec = _kupiec_pof(x, n, p0)
    christ = _christoffersen_independence(viol, p0)
    reject_pof = bool(kupiec["p_value"] < 0.05) if math.isfinite(kupiec["p_value"]) else None
    reject_ind = bool(christ["p_ind"] < 0.05) if math.isfinite(christ["p_ind"]) else None
    passed = None
    if reject_pof is not None and reject_ind is not None:
        passed = not reject_pof and not reject_ind
    return {
        "n": n,
        "violations": x,
        "rate": rate,
        "expected_rate": p0,
        "violations_low": int((yv < lov).sum()),
        "violations_high": int((yv > hiv).sum()),
        "kupiec": kupiec,
        "christoffersen": christ,
        "reject_pof": reject_pof,
        "reject_independence": reject_ind,
        "pass": passed,
    }


# ───────────────────────── 내부 유틸 ─────────────────────────
def _conformal_quantile(scores: np.ndarray, nominal: float) -> float:
    """유한표본 등각 분위수: level = ceil((n+1)·nominal)/n (1 을 넘으면 1)."""
    v = np.asarray(scores, dtype="float64").ravel()
    n = len(v)
    if n == 0:
        return float("nan")
    level = math.ceil((n + 1) * nominal) / n
    level = min(1.0, max(level, 1.0 / n))
    return float(np.quantile(v, level, method="higher"))


def _median_positive(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype="float64").ravel()
    ok = arr[np.isfinite(arr) & (arr > 0)]
    if len(ok) == 0:
        return 1.0
    return float(np.median(ok))


def decay_weights(ages_days: np.ndarray, half_life_days: float = DECAY_HALF_LIFE_DAYS) -> np.ndarray:
    """잔차 나이(일) → 가중치 2^(−age/half_life).

    age 는 평가 기준 시점에서 그 잔차가 발생한 날까지의 일수다. 반감기가 30일이면 하루 전
    잔차는 0.977, 60일 전은 0.25, 90일 전은 0.125 무게를 갖는다. 나이가 음수이거나(평가
    기준 시점보다 뒤의 잔차) 유한하지 않은 값은 무게 0 이다 — 기준 시점 뒤의 정보를 보정에
    실을 수 없다. 나이 0 은 1.0 이다.
    """
    age = np.asarray(ages_days, dtype="float64").ravel()
    if not math.isfinite(float(half_life_days)) or half_life_days <= 0:
        raise ValueError(f"반감기는 0보다 큰 유한값이어야 한다: {half_life_days}")
    usable = np.isfinite(age) & (age >= 0.0)
    w = np.zeros_like(age)
    w[usable] = np.exp2(-age[usable] / float(half_life_days))
    return w


def _weighted_conformal_quantile(
    scores: np.ndarray, weights: np.ndarray, nominal: float
) -> float:
    """가중 유한표본 등각 분위수.

    수준은 비가중 경로와 같은 ceil((n+1)·nominal)/n 을 쓰고, 그 수준을 넘는 첫 점수를 누적
    가중 비율에서 찾는다. 이 유한표본 보정은 **근사**다 — 수준을 행 수 n 으로 계산하는데
    가중 유효표본(ESS = (Σw)² / Σw²)은 무게가 기울면 0.4n 수준까지 작아진다. 그만큼 명목에서
    밀릴 수 있으나 실측 포함률 영향은 1e-5 수준이다.
    점수는 안정 정렬(mergesort)하므로 같은 점수의 순서가 뒤집혀도 결과가 달라지지 않는다.
    가중치 합이 0 이면 비가중으로 되돌린다.
    같은 무게를 주면 순서 통계량 기준으로 비가중 경로와 한 칸(1/n) 차이가 날 수 있다
    (비가중은 numpy 의 higher 규약, 가중은 누적 비율의 역함수 규약).
    """
    v = np.asarray(scores, dtype="float64").ravel()
    w = np.asarray(weights, dtype="float64").ravel()
    if len(v) != len(w):
        raise ValueError(f"점수·가중치 길이가 다르다: {len(v)} vs {len(w)}")
    n = len(v)
    if n == 0:
        return float("nan")
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total = float(w.sum())
    if total <= 0.0:
        return _conformal_quantile(v, nominal)
    level = math.ceil((n + 1) * nominal) / n  # 근사 — 유효표본이 아니라 행 수 n 기준이다
    level = min(1.0, max(level, 1.0 / n))
    order = np.argsort(v, kind="mergesort")
    cum = np.cumsum(w[order]) / total
    idx = int(np.searchsorted(cum, level, side="left"))
    idx = min(idx, n - 1)
    return float(v[order][idx])


def _sanitize_scale(scales: np.ndarray, floor: float) -> np.ndarray:
    arr = np.asarray(scales, dtype="float64").ravel()
    bad = ~np.isfinite(arr) | (arr <= 0)
    if not bad.any():
        return arr
    out = arr.copy()
    out[bad] = floor if math.isfinite(floor) and floor > 0 else 1.0
    return out


def _coverage_stats(inside: np.ndarray, width: np.ndarray, *, mask: np.ndarray | None = None) -> dict:
    sel = np.ones(len(inside), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    n = int(sel.sum())
    if n == 0:
        return {"n": 0, "picp": float("nan"), "width_median": float("nan")}
    return {
        "n": n,
        "picp": float(inside[sel].mean()),
        "width_median": float(np.median(width[sel])),
    }


def _kupiec_pof(x: int, n: int, p0: float) -> dict:
    """Kupiec 비조건부 포함률 검정(이항 우도비)."""
    if n == 0:
        return {"lr": float("nan"), "df": 1, "p_value": float("nan")}
    p_hat = x / n
    lr = 0.0
    if x > 0:
        lr += 2.0 * x * math.log(p_hat / p0)
    if n - x > 0:
        lr += 2.0 * (n - x) * math.log((1.0 - p_hat) / (1.0 - p0))
    lr = max(lr, 0.0)
    return {"lr": float(lr), "df": 1, "p_value": float(chi2.sf(lr, 1))}


def _christoffersen_independence(viol: np.ndarray, p0: float) -> dict:
    """Christoffersen 독립성 검정(Markov 전이 우도비) + 조건부 포함률 결합 통계량."""
    v = np.asarray(viol, dtype="int8").ravel()
    if len(v) < 2:
        nan = float("nan")
        return {"lr_ind": nan, "df": 1, "p_ind": nan, "lr_cc": nan, "df_cc": 2, "p_cc": nan}
    prev, cur = v[:-1], v[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total if total else float("nan")
    pi01 = n01 / (n00 + n01) if n00 + n01 else float("nan")
    pi11 = n11 / (n10 + n11) if n10 + n11 else float("nan")
    # L0: 이탈이 독립일 때의 대수우도, L1: 1차 마르코프일 때
    ll0 = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    ll1 = (
        _xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01)
        + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11)
    )
    lr_ind = max(2.0 * (ll1 - ll0), 0.0)
    pof = _kupiec_pof(n01 + n11, total, p0)
    lr_cc = lr_ind + (pof["lr"] if math.isfinite(pof["lr"]) else 0.0)
    return {
        "lr_ind": float(lr_ind),
        "df": 1,
        "p_ind": float(chi2.sf(lr_ind, 1)),
        "lr_cc": float(lr_cc),
        "df_cc": 2,
        "p_cc": float(chi2.sf(lr_cc, 2)),
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
    }


def _xlogy(count: int, prob: float) -> float:
    """count·ln(prob). count 가 0이면 0, 확률이 0 이하이면 아주 작은 값으로 대체."""
    if count == 0:
        return 0.0
    return float(count * math.log(max(prob, 1e-300)))


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _jsonable(obj: Any) -> Any:
    """numpy 스칼라·배열을 JSON 값으로 바꾼다."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return _finite_or_none(float(obj)) if isinstance(obj, np.floating) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ───────────────────────── 그룹 라벨 ─────────────────────────
def horizon_band(days: np.ndarray) -> np.ndarray:
    """cut 으로부터 경과일 → 지평밴드 라벨('1-31' 등). 범위 밖은 'other'."""
    arr = np.asarray(days).ravel()
    out = np.full(arr.shape, "other", dtype=object)
    for lo, hi in HORIZON_BANDS:
        out[(arr >= lo) & (arr <= hi)] = f"{lo}-{hi}"
    return out.astype(str)


def tertile_edges(scales: np.ndarray) -> tuple[float, float]:
    """규모 tertile 경계(1/3·2/3 분위). 보정창 규모로만 계산해 미래 정보를 쓰지 않는다."""
    arr = np.asarray(scales, dtype="float64").ravel()
    ok = arr[np.isfinite(arr) & (arr > 0)]
    if len(ok) < 3:
        return (float("inf"), float("inf"))
    lo, hi = np.quantile(ok, [1.0 / 3.0, 2.0 / 3.0])
    return (float(lo), float(hi))


def scale_tertile(scales: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    """규모 → small/medium/large. 경계는 tertile_edges 결과를 그대로 쓴다."""
    arr = np.asarray(scales, dtype="float64").ravel()
    e1, e2 = edges
    out = np.full(arr.shape, "large", dtype=object)
    out[arr <= e1] = "small"
    out[(arr > e1) & (arr <= e2)] = "medium"
    out[~np.isfinite(arr)] = "small"  # 규모를 모르면 보수적으로 소규모 취급
    return out.astype(str)


def group_label(bands: np.ndarray, tertiles: np.ndarray) -> np.ndarray:
    """밴드 · tertile → 'band|tertile' 라벨."""
    b = np.asarray(bands).ravel().astype(str)
    t = np.asarray(tertiles).ravel().astype(str)
    if len(b) != len(t):
        raise ValueError("bands · tertiles 길이가 다르다")
    return np.char.add(np.char.add(b, "|"), t)


# ───────────────────────── 실험 실행 ─────────────────────────
RUN_ID = "20261115-P3-R3-01"
RUN_DIR = ROOT / "models" / "_exp" / RUN_ID
BASE_RUN_DIR = ROOT / "models" / "_exp" / "20261114-P0-R0-01"
PICP_BAND = (0.77, 0.83)
NIGHT_HOURS = (0, 19)  # 04:00~06:00 · 23:00~24:00 (원자료에 24:00~04:00 행이 없다)
MIN_TRAIN_DAYS = 90
HOLIDAY_GAIN_TARGET = 0.15
NONHOLIDAY_TOLERANCE = 0.01


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dump_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def _append_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _band_quantile_inversions(cal: ConformalCalibrator) -> int:
    """지평밴드 순서로 놓았을 때 분위수가 역전된 횟수(진단값).

    구간은 그룹별 분위수 하나로 lo·hi 를 만들므로 교차 구간이 생기지 않는다. 이 값은
    밴드별 분위수 추정이 지평 순서와 어긋난 정도를 남겨 두는 용도다.
    """
    band_order = [f"{lo}-{hi}" for lo, hi in HORIZON_BANDS]
    tiers: dict[str, dict[str, str]] = {}
    for key in cal.quantiles:
        head, _, tail = key.partition("|")
        tiers.setdefault(tail, {})[head] = key
    inversions = 0
    for mapping in tiers.values():
        keys = [mapping[b] for b in band_order if b in mapping]
        if len(keys) < 2:
            continue
        values = np.array([cal.quantiles[k] for k in keys], dtype="float64")
        inversions += int((np.diff(values) < 0).sum())
    band_values = np.array([cal.band_quantiles.get(b, np.nan) for b in band_order], dtype="float64")
    if np.isfinite(band_values).all():
        inversions += int((np.diff(band_values) < 0).sum())
    return inversions


def _project_modules():
    """notebooks 의 피처·프로토콜 구현을 그대로 재사용한다(복사 금지)."""
    for extra in (ROOT / "notebooks", ROOT / "src"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    import calendar_features as cf
    import eval_protocol as ep

    from metro_insight.data.guard import DataGuard

    return cf, ep, DataGuard


def _frozen_scale(cf, df_fit, df_target, as_of) -> np.ndarray:
    """as_of 시점에 고정한 월 프로필 평균(규모 정규화 분모)."""
    profile = cf.frozen_profile(
        df_fit, as_of, cf.KEY_SH_MONTH,
        mean_col=cf.PROFILE_COLS_MONTH[0], n_col=cf.PROFILE_COLS_MONTH[1],
    )
    return cf.lookup(profile, df_target, cf.KEY_SH_MONTH, cf.PROFILE_COLS_MONTH[0])


def _scale_values(month_mean: np.ndarray, sh_mean: np.ndarray) -> np.ndarray:
    """규모 분모: 월 프로필 → (역, 시간대) 프로필 순으로 채운다.

    지평이 길면 cut 시점 프로필에 그 달 키가 없어 월 평균이 비어 있다(12월·먼 달).
    그 행은 계절을 지운 (역, 시간대) 평균으로 대체한다. 둘 다 없으면 NaN 으로 두고
    보정기에서 전체 중앙값으로 채운다.
    """
    month = np.asarray(month_mean, dtype="float64")
    sh = np.asarray(sh_mean, dtype="float64")
    use_month = np.isfinite(month) & (month > 0)
    return np.where(use_month, month, sh)


def _calibration_specs(cut_ts) -> list[dict]:
    """밴드별 보정 원점 목록(창 길이를 CAL_WINDOW_DAYS 로 맞춘다).

    잔차 구간 길이는 밴드 폭으로 정해진다. 원점을 밴드 폭만큼 띄워 여러 개 두면 잔차
    구간이 겹치지 않고 이어 붙는다(1-31 은 폭 31일이라 원점 3개로 93일).
    · 1-31   : 원점 cut-31 · cut-62 · cut-93
    · 32-91  : 원점 cut-91 · cut-151
    · 92-181 : 원점 cut-181
    · 182-366: 원점 cut-366 (창은 최대 90일 — 가장 먼 지평 구간)
    """
    specs: list[dict] = []
    for lo, hi in HORIZON_BANDS:
        width = hi - lo + 1
        n_origins = max(1, math.ceil(CAL_WINDOW_DAYS / width))
        for k in range(n_origins):
            specs.append({
                "bands": (f"{lo}-{hi}",),
                "back": hi + k * width,
                "hi": hi,
            })
    return specs


def _pd():
    import pandas as pd

    return pd


def _with_attrs(df, src):
    """슬라이스 뒤에도 feature_names 등 attrs 를 유지한다."""
    out = df.copy()
    out.attrs.update(src.attrs)
    return out


def _calibration_rows(cf, ep, base, cut, *, min_train_days: int = MIN_TRAIN_DAYS) -> dict:
    """한 cut 의 보정 표본(잔차·규모·밴드)을 밴드별로 모은다.

    밴드마다 원점을 여러 개 두고(``_calibration_specs``) 잔차 구간을 이어 붙여 창 길이를
    ``CAL_WINDOW_DAYS`` 로 맞춘다. 각 원점은 그 밴드의 지평만큼 앞선 시점에서 학습하므로
    평가 모델과 지평이 같아지고, 원점이 여러 개라 한 달치 날씨·수준 이탈이 분위수를
    지배하지 않는다.
    """
    pd = _pd()
    cut_ts = pd.Timestamp(cut.cut_date)
    data_start = pd.Timestamp(base["date"].min())
    res_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    band_parts: list[np.ndarray] = []
    age_parts: list[np.ndarray] = []
    info: list[dict] = []
    for spec in _calibration_specs(cut_ts):
        origin = cut_ts - pd.Timedelta(days=spec["back"])
        floor = data_start + pd.Timedelta(days=min_train_days)
        clipped = bool(origin < floor)
        if clipped:
            origin = floor
        if origin >= cut_ts:
            continue
        window_end = min(cut_ts, origin + pd.Timedelta(days=spec["hi"]))
        frame = base.loc[base["date"] <= window_end]
        feat = ep.build_features(frame, as_of=origin, feature_set="base")
        names = ep.feature_names(feat)
        fit_part = _with_attrs(feat.loc[feat["date"] <= origin], feat)
        cal_part = _with_attrs(feat.loc[feat["date"] > origin], feat)
        if len(fit_part) < 1000 or cal_part.empty:
            del frame, feat
            continue
        model = ep.fit_model(fit_part, params=ep.PARAMS)
        pred = model.predict(cal_part[names])
        y = cal_part["boarding"].to_numpy(dtype="float64")
        leads = (cal_part["date"] - origin).dt.days.to_numpy(dtype="int64")
        bands = horizon_band(leads)
        keep = np.isin(bands, spec["bands"])
        if spec["bands"] == ("182-366",):
            max_lead = int(leads.max())
            keep &= leads >= max(HORIZON_BANDS[3][0], max_lead - (CAL_WINDOW_DAYS - 1))
        n_keep = int(keep.sum())
        if n_keep:
            scale_all = _scale_values(
                cal_part[cf.PROFILE_COLS_MONTH[0]].to_numpy(dtype="float64"),
                cal_part[cf.PROFILE_COLS_SH[0]].to_numpy(dtype="float64"),
            )
            res_parts.append(y[keep] - np.maximum(pred[keep], FLOOR))
            scale_parts.append(scale_all[keep])
            band_parts.append(bands[keep])
            age_parts.append(
                (cut_ts - cal_part["date"]).dt.days.to_numpy(dtype="float64")[keep]
            )
        info.append({
            "bands": list(spec["bands"]),
            "origin": origin.strftime("%Y-%m-%d"),
            "origin_clipped": clipped,
            "window_end": window_end.strftime("%Y-%m-%d"),
            "lead_min": int(leads[keep].min()) if n_keep else None,
            "lead_max": int(leads[keep].max()) if n_keep else None,
            "n_rows": n_keep,
            "n_train_rows": int(len(fit_part)),
        })
        del frame, feat, fit_part, cal_part, model, pred
    if not res_parts:
        raise RuntimeError(f"cut {cut.cut_date}: 보정 표본을 만들지 못했다")
    return {
        "residuals": np.concatenate(res_parts),
        "scales": np.concatenate(scale_parts),
        "bands": np.concatenate(band_parts),
        "ages": np.concatenate(age_parts),
        "origins": info,
    }


def _frozen_scale_frame(cf, base, cut, preds) -> dict:
    """평가 행의 규모(월 프로필 평균)와 지평을 만든다. 프로필은 cut 시점에 고정."""
    pd = _pd()
    cut_ts = pd.Timestamp(cut.cut_date)
    fit = base.loc[base["date"] <= cut_ts]
    ev = preds.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev["month"] = ev["date"].dt.month.astype("int8")
    month_mean = _frozen_scale(cf, fit, ev, cut_ts)
    profile_sh = cf.frozen_profile(
        fit, cut_ts, cf.KEY_SH,
        mean_col=cf.PROFILE_COLS_SH[0], n_col=cf.PROFILE_COLS_SH[1],
    )
    sh_mean = cf.lookup(profile_sh, ev, cf.KEY_SH, cf.PROFILE_COLS_SH[0])
    scale = _scale_values(month_mean, sh_mean)
    horizon = (ev["date"] - cut_ts).dt.days.to_numpy(dtype="int64")
    return {
        "date": ev["date"].to_numpy(),
        "hour_code": ev["hour_code"].to_numpy(dtype="int64"),
        "line_code": ev["line_code"].to_numpy(dtype="int64"),
        "y": ev["y"].to_numpy(dtype="float64"),
        "pred": ev["pred"].to_numpy(dtype="float64"),
        "scale": scale.astype("float64"),
        "horizon": horizon,
        "bands": horizon_band(horizon),
    }


def _coverage_rows(cut_label: str, cov: dict, definitions: dict[str, str]) -> list[dict]:
    rows = []
    for name, body in cov.items():
        note = definitions.get(name)
        if note is None:
            if name.startswith("horizon_band::"):
                note = f"지평 밴드 {name.split('::', 1)[1]}일 구간"
            elif name.startswith("line::"):
                note = f"line_code {name.split('::', 1)[1]} 평가 행"
            else:
                note = ""
        rows.append({
            "cut": cut_label,
            "slice": name,
            "n": body["n"],
            "picp": body["picp"],
            "width_median": body["width_median"],
            "definition": note,
        })
    return rows


SLICE_DEFINITIONS: dict[str, str] = {
    "all": "전체 평가 행",
    "hour_bucket::24_01": (
        "심야 버킷 = 23:00-24:00 + 04:00-06:00 "
        "(원자료 업무일자가 04:00 에 시작해 24:00-04:00 행이 없다)"
    ),
    "scale_tertile::small": "규모 tertile 소 (profile_month_mean 하위 1/3, 경계는 보정창에서 계산)",
    "scale_tertile::medium": "규모 tertile 중",
    "scale_tertile::large": "규모 tertile 대",
    "day_type::holiday": "데이터 is_holiday=1 인 날",
    "day_type::nonholiday": "데이터 is_holiday=0 인 날",
}


def _holiday_mask(dates, holiday_by_date) -> np.ndarray:
    pd = _pd()
    flag = pd.Series(pd.to_datetime(dates)).map(holiday_by_date).fillna(0).to_numpy(dtype="int64")
    return flag.astype(bool)


def _eval_context(cf, base, cut, preds, holiday_by_date, edges) -> dict:
    """한 cut 의 평가 행 공통 정보(규모·지평·그룹). variant 와 무관하게 한 번만 만든다."""
    ev = _frozen_scale_frame(cf, base, cut, preds)
    tert = scale_tertile(ev["scale"], edges)
    is_hol = _holiday_mask(ev["date"], holiday_by_date)
    point = np.maximum(ev["pred"], FLOOR)
    slices = {
        "hour_bucket::24_01": np.isin(ev["hour_code"], NIGHT_HOURS),
        "scale_tertile::small": tert == "small",
        "scale_tertile::medium": tert == "medium",
        "scale_tertile::large": tert == "large",
        "day_type::holiday": is_hol,
        "day_type::nonholiday": ~is_hol,
    }
    for code in np.unique(ev["line_code"]):
        slices[f"line::{int(code)}"] = ev["line_code"] == code
    for lo_b, hi_b in HORIZON_BANDS:
        name = f"{lo_b}-{hi_b}"
        slices[f"horizon_band::{name}"] = ev["bands"] == name
    return {
        "y": ev["y"],
        "point": point,
        "scale": ev["scale"],
        "bands": ev["bands"],
        "tertiles": tert,
        "hour_code": ev["hour_code"],
        "line_code": ev["line_code"],
        "holiday": is_hol,
        "slices": slices,
        "n": int(len(ev["y"])),
        "mae_before": float(np.abs(ev["y"] - ev["pred"]).mean()),
        "mae_after": float(np.abs(ev["y"] - point).mean()),
        "negative_predictions_before": int((ev["pred"] < FLOOR).sum()),
    }


def _self_coverage(
    rows: dict,
    groups: np.ndarray,
    cal: ConformalCalibrator,
    weights: np.ndarray | None = None,
) -> float:
    """보정 표본에 자기 분위수를 적용한 포함률.

    유한표본 등각 분위수를 제대로 적용했다면 이 값은 명목(0.8)과 같아야 한다. 이 값이
    명목과 맞는데 평가 포함률이 어긋나면, 분위수 수준이 아니라 보정 잔차와 평가 잔차의
    분포가 다르다는 뜻이다(교환가능성 위반).

    weights 를 주면 가중 포함률을 낸다. 가중 분위수는 무게가 큰 최근 잔차 쪽에 맞춰지므로
    비가중 포함률은 명목에서 벗어날 수 있다. 그 차이 자체가 진단값이다.
    """
    scores = np.abs(np.asarray(rows["residuals"], dtype="float64")) / _sanitize_scale(
        np.asarray(rows["scales"], dtype="float64"), cal.scale_floor
    )
    q = np.array([cal.quantile(g) for g in groups], dtype="float64")
    inside = scores <= q
    if weights is None:
        return float(inside.mean())
    wt = np.asarray(weights, dtype="float64").ravel()
    total = float(wt.sum())
    if total <= 0.0:
        return float(inside.mean())
    return float((wt * inside).sum() / total)


def _eval_with_calibration(ctx: dict, cal: ConformalCalibrator, variant: str) -> dict:
    """평가 행 구간을 만들고 커버리지·백테스트를 계산한다."""
    groups = _variant_groups(ctx["bands"], ctx["tertiles"], variant)
    lo, hi = cal.interval_arrays(ctx["point"], ctx["scale"], groups)
    cov = conditional_coverage(ctx["y"], lo, hi, slices=ctx["slices"])
    bt = var_backtest(ctx["y"], lo, hi, nominal=cal.nominal)
    return {
        "coverage": cov,
        "backtest": bt,
        "lo": lo,
        "hi": hi,
        "width_median": float(np.median(hi - lo)),
        "negative_lower_bounds": int((lo < FLOOR).sum()),
    }


def _calendar_cut_errors(cf, ep, base, cut) -> dict:
    """달력 피처를 더한 모델의 공휴일·비공휴일 절대오차 합(같은 cut·같은 평가 행)."""
    pd = _pd()
    cut_ts = pd.Timestamp(cut.cut_date)
    lo = cut_ts + pd.Timedelta(days=cut.horizon[0])
    hi = cut_ts + pd.Timedelta(days=cut.horizon[1])
    fit = base.loc[base["date"] <= cut_ts]
    ev = base.loc[(base["date"] >= lo) & (base["date"] <= hi)]
    if fit.empty or ev.empty:
        raise RuntimeError(f"cut {cut.cut_date}: 학습·평가 행이 비었다")
    feat = ep.build_features(pd.concat([fit, ev], ignore_index=True), as_of=cut_ts,
                             feature_set="base")
    ctx = cf.calendar_feature_frame(feat["date"])
    for col in cf.CALENDAR_FEATURE_COLS:
        feat[col] = ctx[col].to_numpy(dtype="int8")
    feat.attrs["feature_names"] = [*ep.feature_names(feat), *cf.CALENDAR_FEATURE_COLS]
    fit_f = _with_attrs(feat.loc[feat["date"] <= cut_ts], feat)
    ev_f = _with_attrs(feat.loc[feat["date"] > cut_ts], feat)
    model = ep.fit_model(fit_f, params=ep.PARAMS)
    pred = model.predict(ev_f[ep.feature_names(feat)])
    y = ev_f["boarding"].to_numpy(dtype="float64")
    hol = ev_f["is_holiday"].to_numpy(dtype="int64").astype(bool)
    err = np.abs(y - np.maximum(pred, FLOOR))
    out = {
        "cut": cut.cut_date,
        "n": int(len(y)),
        "holiday": {"n": int(hol.sum()), "abs_sum": float(err[hol].sum())},
        "nonholiday": {"n": int((~hol).sum()), "abs_sum": float(err[~hol].sum())},
    }
    del feat, fit_f, ev_f, model, pred
    return out


def _base_holiday_errors(base, cut, preds, holiday_by_date) -> dict:
    """기존 B 예측의 공휴일·비공휴일 절대오차 합(비교 기준)."""
    hol = _holiday_mask(preds["date"], holiday_by_date)
    y = preds["y"].to_numpy(dtype="float64")
    pred = preds["pred"].to_numpy(dtype="float64")
    err = np.abs(y - np.maximum(pred, FLOOR))
    return {
        "cut": cut.cut_date,
        "n": int(len(y)),
        "holiday": {"n": int(hol.sum()), "abs_sum": float(err[hol].sum())},
        "nonholiday": {"n": int((~hol).sum()), "abs_sum": float(err[~hol].sum())},
    }


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _load_base_predictions(path: Path, cuts: list) -> dict:
    """기존 B 실행의 예측을 cut 별로 재사용한다(재학습 없이 잔차를 얻는 경로)."""
    pd = _pd()
    if not path.exists():
        raise FileNotFoundError(f"기존 예측 파일이 없다: {path}")
    cols = ["date", "cut", "station_code", "hour_code", "line_code", "y", "pred"]
    df = pd.read_csv(path, compression="gzip", usecols=cols,
                     dtype={"station_code": "int32", "hour_code": "int8",
                            "line_code": "int8", "y": "float32", "pred": "float32"})
    df["date"] = pd.to_datetime(df["date"])
    wanted = {c.cut_date for c in cuts}
    out = {}
    for name, part in df.groupby("cut", sort=False):
        if name in wanted:
            out[str(name)] = part.reset_index(drop=True)
    missing = wanted - set(out)
    if missing:
        raise RuntimeError(f"기존 예측에 없는 cut: {sorted(missing)}")
    return out


VARIANT_PRIMARY = "band_x_tertile"
VARIANT_FALLBACK = "band_only"


def _variant_groups(bands: np.ndarray, tertiles: np.ndarray, variant: str) -> np.ndarray:
    if variant == VARIANT_PRIMARY:
        return group_label(bands, tertiles)
    if variant == VARIANT_FALLBACK:
        return np.asarray(bands).ravel().astype(str)
    raise ValueError(f"모르는 variant: {variant}")


class _PooledAccumulator:
    """선택된 variant 의 전체 커버리지를 한 번에 계산하기 위한 모음."""

    def __init__(self) -> None:
        self.parts: dict[str, list[np.ndarray]] = {}

    def add(self, *, y, lo, hi, hour_code, line_code, tertile, band, holiday) -> None:
        tert_code = np.where(tertile == "small", 0, np.where(tertile == "medium", 1, 2))
        band_names = [f"{lo_b}-{hi_b}" for lo_b, hi_b in HORIZON_BANDS]
        band_code = np.full(len(band), -1, dtype="int8")
        for i, name in enumerate(band_names):
            band_code[band == name] = i
        body = {
            "y": y.astype("float32"),
            "lo": lo.astype("float32"),
            "hi": hi.astype("float32"),
            "hour_code": hour_code.astype("int8"),
            "line_code": line_code.astype("int8"),
            "tertile": tert_code.astype("int8"),
            "band": band_code,
            "holiday": holiday.astype(bool),
        }
        for key, value in body.items():
            self.parts.setdefault(key, []).append(value)

    def coverage(self) -> dict:
        if not self.parts:
            return {}
        arr = {k: np.concatenate(v) for k, v in self.parts.items()}
        slices = {
            "hour_bucket::24_01": np.isin(arr["hour_code"], NIGHT_HOURS),
            "scale_tertile::small": arr["tertile"] == 0,
            "scale_tertile::medium": arr["tertile"] == 1,
            "scale_tertile::large": arr["tertile"] == 2,
            "day_type::holiday": arr["holiday"],
            "day_type::nonholiday": ~arr["holiday"],
        }
        for i, (lo_b, hi_b) in enumerate(HORIZON_BANDS):
            slices[f"horizon_band::{lo_b}-{hi_b}"] = arr["band"] == i
        for code in np.unique(arr["line_code"]):
            slices[f"line::{int(code)}"] = arr["line_code"] == code
        return conditional_coverage(arr["y"], arr["lo"], arr["hi"], slices=slices)


def _conditional_gate(cov_rows: list[dict], cuts: list[str]) -> dict:
    """슬라이스 존재 게이트 — 필수 6종 + 지평밴드 4종 + 호선 슬라이스."""
    required = [
        "hour_bucket::24_01",
        "scale_tertile::small",
        "scale_tertile::medium",
        "scale_tertile::large",
        "day_type::holiday",
        "day_type::nonholiday",
        *[f"horizon_band::{lo}-{hi}" for lo, hi in HORIZON_BANDS],
    ]
    present = {(row["cut"], row["slice"]) for row in cov_rows}
    missing = [f"{cut}:{name}" for cut in cuts for name in required if (cut, name) not in present]
    line_slices = sorted({name for _, name in present if name.startswith("line::")})
    if not line_slices:
        missing.append("line::<code>")
    return {"required": [*required, "line::<code>"], "line_slices": line_slices,
            "missing": missing, "ok": not missing}


def _write_metrics(
    path: Path,
    *,
    run_id: str,
    protocol: str,
    selected: str,
    attempts: int,
    rejected: bool,
    cut_status: dict[str, dict],
    cov_rows: list[dict],
    crossings: dict[str, int],
    holiday: dict,
    run_config: dict | None = None,
) -> dict:
    """게이트 판정표를 metrics.json 으로 남긴다."""
    def _picp(rec: dict) -> float:
        return rec["picp"] if selected == VARIANT_PRIMARY else rec.get("picp_fallback", rec["picp"])

    cuts = list(cut_status)
    out_of_band = [c for c in cuts if not PICP_BAND[0] <= _picp(cut_status[c]) <= PICP_BAND[1]]
    neg_before = sum(cut_status[c]["negative_predictions_before"] for c in cuts)
    neg_after = sum(cut_status[c]["negative_lower_bounds"] for c in cuts)
    backtest_fail = [c for c in cuts if not cut_status[c]["backtest"]["pass"]]
    gates = {
        "nominal": NOMINAL,
        "picp_band": list(PICP_BAND),
        "picp_all_cuts_in_band": not out_of_band,
        "picp_out_of_band_cuts": out_of_band,
        "conditional_slices": _conditional_gate(cov_rows, cuts),
        "interval_crossings": 0,
        "interval_crossings_note": (
            "구간은 그룹별 분위수 하나로 lo·hi 를 만들므로 교차가 구조적으로 0 이다. "
            "밴드 분위수 역전은 band_quantile_inversions 에 진단값으로 남긴다."
        ),
        "band_quantile_inversions": int(crossings["before"]),
        "negative_predictions_before": int(neg_before),
        "negative_predictions_after": int(neg_after),
        "variance_backtest_failed_cuts": backtest_fail,
        "calendar": holiday,
    }
    per_cut = {
        cut: {
            "n": rec["n"],
            "picp": _picp(rec),
            "picp_primary": rec["picp"],
            "picp_fallback": rec.get("picp_fallback"),
            "calibration_self_coverage": rec.get("calibration_self_coverage"),
            "calibration_self_coverage_weighted": rec.get("calibration_self_coverage_weighted"),
            "in_band": bool(PICP_BAND[0] <= _picp(rec) <= PICP_BAND[1]),
            "width_median": rec["width_median"],
            "mae_before": rec["mae_before"],
            "mae_after": rec["mae_after"],
            "negative_predictions_before": rec["negative_predictions_before"],
            "negative_lower_bounds": rec["negative_lower_bounds"],
            "backtest_share_in_interval": rec["backtest"]["rate"],
            "backtest_pass": rec["backtest"]["pass"],
        }
        for cut, rec in cut_status.items()
    }
    payload = {
        "schema": "conformal-metrics/1",
        "run_id": run_id,
        "protocol": protocol,
        "selected_variant": selected,
        "regroup_attempts": attempts,
        "picp_rejected": rejected,
        "gates": gates,
        "config": dict(run_config or {}),
        "per_cut": per_cut,
        "test_accessed": False,
    }
    _dump_json(path, payload)
    return payload


def _write_manifest(
    path: Path,
    *,
    run_id: str,
    protocol: str,
    base_run: str,
    base_rows: int,
    date_min: str,
    date_max: str,
    cuts: list[dict],
    artifacts: list[str],
    weighting: str = WEIGHTING_NONE,
    half_life_days: float | None = None,
    log_name: str = "log.jsonl",
) -> dict:
    """산출물 목록과 해시 · 실행 설정을 manifest.json 으로 남긴다."""
    entries = []
    for name in artifacts:
        target = path.parent / name
        entries.append({"path": name, "bytes": int(target.stat().st_size), "sha256": _sha256(target)})
    payload = {
        "schema": "conformal-manifest/1",
        "run_id": run_id,
        "protocol": protocol,
        "created_at": _now(),
        "data": {
            "splits": ["train", "valid"],
            "rows": int(base_rows),
            "date_min": date_min,
            "date_max": date_max,
            "test_accessed": False,
        },
        "base_run": base_run,
        "config": {
            "nominal": NOMINAL,
            "cal_window_days": CAL_WINDOW_DAYS,
            "weighting": weighting,
            "half_life_days": half_life_days,
            "min_group_n": MIN_GROUP_N,
            "horizon_bands": [list(b) for b in HORIZON_BANDS],
            "tertile_names": list(TERTILE_NAMES),
            "score": SCORE_KIND,
            "negative_clamp": "point=max(point,0); 하한 0 미만이면 폭 유지 후 우측 이동",
            "calibration_origins": _calibration_specs(None),
        },
        "cuts": cuts,
        "artifacts": entries,
        "log": {"path": log_name, "note": "실행 중 덧붙여 쓰므로 해시 대상에서 뺀다"},
    }
    _dump_json(path, payload)
    return payload


def _log_path(run_dir: Path) -> Path:
    """이번 실행의 로그 경로. 같은 디렉터리에 로그가 있으면 번호를 올려 새 파일에 쓴다.

    이전 로그를 지우면 그 실행의 근거가 사라진다. 그래서 truncate 대신
    log.jsonl → log-2.jsonl → log-3.jsonl 순으로 비어 있는 이름을 고른다.
    """
    candidate = run_dir / "log.jsonl"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = run_dir / f"log-{n}.jsonl"
        if not candidate.exists():
            return candidate
        n += 1


# ───────────────────────── 재실행 차단 ─────────────────────────
RUN_ARTIFACTS: tuple[str, ...] = (
    "metrics.json", "calibrator.json", "conditional_coverage.csv", "manifest.json",
)
_RUN_SEQ_RE = re.compile(r"^(?P<head>.*-)(?P<seq>[0-9]{2})$")


def existing_artifacts(run_dir: Path) -> list[str]:
    """이미 있는 산출물 이름 — 하나라도 있으면 재실행을 거부한다."""
    run_dir = Path(run_dir)
    return [name for name in RUN_ARTIFACTS if (run_dir / name).exists()]


def next_free_run_name(run_dir: Path) -> str:
    """같은 접두사에서 산출물이 없는 다음 디렉터리 이름(접미 번호가 없으면 '-2')."""
    run_dir = Path(run_dir)
    match = _RUN_SEQ_RE.match(run_dir.name)
    if not match:
        return f"{run_dir.name}-2"
    head, seq = match.group("head"), int(match.group("seq"))
    for n in range(seq + 1, 100):
        if not existing_artifacts(run_dir.parent / f"{head}{n:02d}"):
            return f"{head}{n:02d}"
    return f"{run_dir.name}-2"


def _refuse_dirty_run_dir(run_dir: Path) -> None:
    """산출물이 있는 디렉터리에는 쓰지 않는다 — 이전 실행의 근거를 덮어쓰지 않기 위해서다."""
    used = existing_artifacts(run_dir)
    if not used:
        return
    raise SystemExit(
        f"산출물 충돌: {run_dir} 에 이미 산출물이 있다({', '.join(used)}) — 덮어쓰지 않는다. "
        f"비어 있는 이름: {run_dir.parent / next_free_run_name(run_dir)}"
    )


def run(
    *,
    run_dir: Path = RUN_DIR,
    cuts: str | None = None,
    lane: str = "desktop",
    base_run: Path = BASE_RUN_DIR,
    skip_calendar: bool = False,
    weighting: str = WEIGHTING_DEFAULT,
    half_life_days: float = DECAY_HALF_LIFE_DAYS,
) -> dict:
    """8 cut 분할 등각 보정 → calibrator JSON · 조건부 커버리지 표 · 로그."""
    cf, ep, DataGuard = _project_modules()
    pd = _pd()
    if weighting not in (WEIGHTING_TIME_DECAY, WEIGHTING_NONE):
        raise ValueError(f"모르는 가중 방식: {weighting}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _refuse_dirty_run_dir(run_dir)
    log_path = _log_path(run_dir)
    selected_cuts = ep.select_cuts(cuts)
    started = time.time()
    _append_log(log_path, {
        "kind": "run_start",
        "run_id": run_dir.name,
        "protocol": ep.PROTOCOL_ID,
        "nominal": NOMINAL,
        "cal_window_days": CAL_WINDOW_DAYS,
        "weighting": weighting,
        "half_life_days": half_life_days if weighting == WEIGHTING_TIME_DECAY else None,
        "cuts": [c.cut_date for c in selected_cuts],
        "base_run": base_run.name,
        "score": SCORE_KIND,
        "started_at": _now(),
    })

    guard = DataGuard(ROOT, lane=lane)
    splits = ep.load_splits(guard)
    base = pd.concat(list(splits.values()), ignore_index=True)
    base = base.sort_values("date", kind="mergesort").reset_index(drop=True)
    base = cf.add_month(base)
    holiday_by_date = base.groupby("date", observed=True)["is_holiday"].max()
    print(f"[{run_dir.name}] train+valid {len(base):,}행 · "
          f"{base['date'].min().date()} ~ {base['date'].max().date()}")
    preds = _load_base_predictions(base_run / "predictions.csv.gz", selected_cuts)

    cal_json: dict[str, dict] = {}
    cal_json_fb: dict[str, dict] = {}
    cov_rows: list[dict] = []
    cov_rows_fb: list[dict] = []
    pooled = _PooledAccumulator()
    pooled_fb = _PooledAccumulator()
    cut_status: dict[str, dict] = {}
    holiday_base_err: list[dict] = []
    holiday_cal_err: list[dict] = []
    crossings: dict[str, dict[str, int]] = {
        "primary": {"before": 0},
        "fallback": {"before": 0},
    }

    for cut in selected_cuts:
        t0 = time.time()
        rows = _calibration_rows(cf, ep, base, cut)
        tert_edges = tertile_edges(rows["scales"])
        ctx = _eval_context(cf, base, cut, preds[cut.cut_date], holiday_by_date, tert_edges)
        weights = (
            decay_weights(rows["ages"], half_life_days)
            if weighting == WEIGHTING_TIME_DECAY
            else None
        )
        for variant, target_json, target_rows, accum in (
            (VARIANT_PRIMARY, cal_json, cov_rows, pooled),
            (VARIANT_FALLBACK, cal_json_fb, cov_rows_fb, pooled_fb),
        ):
            groups = _variant_groups(rows["bands"], scale_tertile(rows["scales"], tert_edges), variant)
            cal = ConformalCalibrator(min_group_n=MIN_GROUP_N).fit(
                rows["residuals"], rows["scales"], groups, nominal=NOMINAL, weights=weights
            )
            cal.meta = {
                "cut": cut.cut_date,
                "variant": variant,
                "cal_window_days": CAL_WINDOW_DAYS,
                "weighting": weighting,
                "half_life_days": half_life_days if weights is not None else None,
                "age_days": {
                    "min": float(rows["ages"].min()),
                    "median": float(np.median(rows["ages"])),
                    "max": float(rows["ages"].max()),
                },
                "weight_sum": float(weights.sum()) if weights is not None else None,
                "tertile_edges": list(tert_edges),
                "n_rows": int(len(rows["residuals"])),
                "origins": rows["origins"],
                "grouping": "지평밴드 x 규모 tertile(profile_month_mean)"
                if variant == VARIANT_PRIMARY else "지평밴드(규모 tertile 통합)",
                "negative_clamp": "point=max(point,0); 하한 0 미만이면 폭 유지 후 우측 이동",
            }
            inversions = _band_quantile_inversions(cal)
            bucket = crossings["primary" if variant == VARIANT_PRIMARY else "fallback"]
            bucket["before"] += inversions
            cal.meta["band_quantile_inversions"] = inversions
            res = _eval_with_calibration(ctx, cal, variant)
            picp = res["coverage"]["all"]["picp"]
            band_ok = PICP_BAND[0] <= picp <= PICP_BAND[1]
            target_json[cut.cut_date] = cal.to_json()
            target_rows.extend(_coverage_rows(cut.cut_date, res["coverage"], SLICE_DEFINITIONS))
            accum.add(y=ctx["y"], lo=res["lo"], hi=res["hi"], hour_code=ctx["hour_code"],
                      line_code=ctx["line_code"], tertile=ctx["tertiles"], band=ctx["bands"],
                      holiday=ctx["holiday"])
            if variant == VARIANT_PRIMARY:
                cut_status[cut.cut_date] = {
                    "n": ctx["n"],
                    "picp": picp,
                    "in_band": bool(band_ok),
                    "calibration_self_coverage": _self_coverage(rows, groups, cal),
                    "calibration_self_coverage_weighted": _self_coverage(
                        rows, groups, cal, weights=weights
                    ),
                    "width_median": res["width_median"],
                    "mae_before": ctx["mae_before"],
                    "mae_after": ctx["mae_after"],
                    "negative_predictions_before": ctx["negative_predictions_before"],
                    "negative_lower_bounds": res["negative_lower_bounds"],
                    "backtest": res["backtest"],
                    "group_sources": {
                        name: cal.source(name) for name in sorted(cal.quantiles)
                    },
                }
            else:
                cut_status[cut.cut_date]["picp_fallback"] = picp
                cut_status[cut.cut_date]["in_band_fallback"] = bool(band_ok)
            del res
        _append_log(log_path, {"kind": "cut", **cut_status[cut.cut_date], "cut": cut.cut_date,
                               "elapsed_s": round(time.time() - t0, 1)})
        print(f"  cut {cut.cut_date}: PICP80 {cut_status[cut.cut_date]['picp']:.4f} "
              f"(fallback {cut_status[cut.cut_date]['picp_fallback']:.4f}) · "
              f"폭중앙 {cut_status[cut.cut_date]['width_median']:.1f} · "
              f"MAE {cut_status[cut.cut_date]['mae_before']:.2f}→"
              f"{cut_status[cut.cut_date]['mae_after']:.2f} · "
              f"음수 {cut_status[cut.cut_date]['negative_predictions_before']:,}→0 · "
              f"{round(time.time() - t0, 1)}s")
        del rows, ctx

    # 조건부 커버리지(전체) + 슬라이스 존재 확인
    pooled_cov = pooled.coverage()
    pooled_cov_fb = pooled_fb.coverage()
    if pooled_cov:
        cov_rows.extend(_coverage_rows("pooled", pooled_cov, SLICE_DEFINITIONS))
    if pooled_cov_fb:
        cov_rows_fb.extend(_coverage_rows("pooled", pooled_cov_fb, SLICE_DEFINITIONS))

    # variant 선택: 첫 시도가 밴드를 벗어난 cut 이 있으면 1회 재설계(band_only)로 교체
    out_of_band = [c for c, s in cut_status.items() if not s["in_band"]]
    fb_out = [c for c, s in cut_status.items() if not s["in_band_fallback"]]
    if not out_of_band:
        selected, attempts, rejected = VARIANT_PRIMARY, 1, False
    elif not fb_out:
        selected, attempts, rejected = VARIANT_FALLBACK, 2, False
        _append_log(log_path, {
            "kind": "regroup",
            "attempt": 2,
            "from": VARIANT_PRIMARY,
            "to": VARIANT_FALLBACK,
            "out_of_band_cuts": out_of_band,
            "reason": (
                "그룹 표본 부족·밴드 불일치로 12그룹 등각 분위수가 흔들린 cut 이 있어 "
                "밴드 단위로 재설계"
            ),
        })
    else:
        selected, attempts, rejected = VARIANT_PRIMARY, 2, True
        _append_log(log_path, {
            "kind": "picp_rejected",
            "attempts": 2,
            "out_of_band_primary": out_of_band,
            "out_of_band_fallback": fb_out,
            "note": "명목 80% 구간이 두 설계 모두에서 8 cut 밴드를 만족하지 못했다. 수치를 그대로 남긴다.",
        })

    # 공휴일 달력 피처 비교(공휴일 MAE −15% 이상 또는 rejected 기록)
    holiday_summary: dict = {
        "skipped": bool(skip_calendar),
        "base_holiday_mae": None,
        "calendar_holiday_mae": None,
        "holiday_gain": None,
        "holiday_gain_target": HOLIDAY_GAIN_TARGET,
        "base_nonholiday_mae": None,
        "calendar_nonholiday_mae": None,
        "nonholiday_change": None,
        "nonholiday_tolerance": NONHOLIDAY_TOLERANCE,
        "decision": "skipped" if skip_calendar else "rejected",
        "cuts": 0,
    }
    if not skip_calendar:
        for cut in selected_cuts:
            base_err = _base_holiday_errors(base, cut, preds[cut.cut_date], holiday_by_date)
            try:
                cal_err = _calendar_cut_errors(cf, ep, base, cut)
            except (RuntimeError, ValueError, KeyError) as exc:
                _append_log(log_path, {"kind": "calendar_error", "cut": cut.cut_date,
                                       "error": str(exc)})
                continue
            holiday_base_err.append(base_err)
            holiday_cal_err.append(cal_err)
            _append_log(log_path, {
                "kind": "calendar_cut",
                "cut": cut.cut_date,
                "base_holiday_mae": _finite_or_none(_ratio(base_err["holiday"]["abs_sum"],
                                                         base_err["holiday"]["n"])),
                "calendar_holiday_mae": _finite_or_none(_ratio(cal_err["holiday"]["abs_sum"],
                                                              cal_err["holiday"]["n"])),
                "base_nonholiday_mae": _finite_or_none(_ratio(base_err["nonholiday"]["abs_sum"],
                                                              base_err["nonholiday"]["n"])),
                "calendar_nonholiday_mae": _finite_or_none(_ratio(
                    cal_err["nonholiday"]["abs_sum"], cal_err["nonholiday"]["n"]
                )),
            })
            print(f"  달력 cut {cut.cut_date}: 공휴일 MAE "
                  f"{_ratio(base_err['holiday']['abs_sum'], base_err['holiday']['n']):.2f} → "
                  f"{_ratio(cal_err['holiday']['abs_sum'], cal_err['holiday']['n']):.2f}")
        if holiday_base_err:
            bh = _ratio(sum(e["holiday"]["abs_sum"] for e in holiday_base_err),
                        sum(e["holiday"]["n"] for e in holiday_base_err))
            ch = _ratio(sum(e["holiday"]["abs_sum"] for e in holiday_cal_err),
                        sum(e["holiday"]["n"] for e in holiday_cal_err))
            bn = _ratio(sum(e["nonholiday"]["abs_sum"] for e in holiday_base_err),
                        sum(e["nonholiday"]["n"] for e in holiday_base_err))
            cn = _ratio(sum(e["nonholiday"]["abs_sum"] for e in holiday_cal_err),
                        sum(e["nonholiday"]["n"] for e in holiday_cal_err))
            gain = _ratio(bh - ch, bh)
            nonhol_deg = _ratio(cn - bn, bn)
            decision = "accepted" if gain >= HOLIDAY_GAIN_TARGET else "rejected"
            holiday_summary.update({
                "base_holiday_mae": _finite_or_none(bh),
                "calendar_holiday_mae": _finite_or_none(ch),
                "holiday_gain": _finite_or_none(gain),
                "base_nonholiday_mae": _finite_or_none(bn),
                "calendar_nonholiday_mae": _finite_or_none(cn),
                "nonholiday_change": _finite_or_none(nonhol_deg),
                "decision": decision,
                "cuts": len(holiday_base_err),
            })
            _append_log(log_path, {
                "kind": "calendar_compare",
                "base_holiday_mae": bh,
                "calendar_holiday_mae": ch,
                "holiday_gain": gain,
                "holiday_gain_target": HOLIDAY_GAIN_TARGET,
                "base_nonholiday_mae": bn,
                "calendar_nonholiday_mae": cn,
                "nonholiday_change": nonhol_deg,
                "nonholiday_tolerance": NONHOLIDAY_TOLERANCE,
                "cuts": len(holiday_base_err),
                "decision": decision,
            })
            print(f"  달력 피처: 공휴일 MAE {bh:.2f} → {ch:.2f} ({gain:+.2%}) · "
                  f"비공휴일 {bn:.2f} → {cn:.2f} ({nonhol_deg:+.2%}) → {decision}")

    # 산출물
    _dump_json(run_dir / "calibrator.json", {
        "schema": "conformal-run/1",
        "run_id": run_dir.name,
        "protocol": ep.PROTOCOL_ID,
        "nominal": NOMINAL,
        "feature_set": "base",
        "score": SCORE_KIND,
        "cal_window_days": CAL_WINDOW_DAYS,
        "weighting": weighting,
        "half_life_days": half_life_days if weighting == WEIGHTING_TIME_DECAY else None,
        "horizon_bands": [list(b) for b in HORIZON_BANDS],
        "tertile_names": list(TERTILE_NAMES),
        "min_group_n": MIN_GROUP_N,
        "negative_clamp": "point=max(point,0); 하한 0 미만이면 폭 유지 후 우측 이동",
        "selected_variant": selected,
        "regroup_attempts": attempts,
        "selected_by_cut": {c: selected for c in cut_status},
        "by_cut": cal_json if selected == VARIANT_PRIMARY else cal_json_fb,
        "alternate_by_cut": cal_json_fb if selected == VARIANT_PRIMARY else cal_json,
        "latest_cut": selected_cuts[-1].cut_date,
    })
    table = pd.DataFrame(
        [{"variant": VARIANT_PRIMARY, **row} for row in cov_rows]
        + [{"variant": VARIANT_FALLBACK, **row} for row in cov_rows_fb]
    )
    if not table.empty:
        table = table[["variant", "cut", "slice", "n", "picp", "width_median", "definition"]]
        table.to_csv(run_dir / "conditional_coverage.csv", index=False, encoding="utf-8")
    metrics = _write_metrics(
        run_dir / "metrics.json",
        run_id=run_dir.name,
        protocol=ep.PROTOCOL_ID,
        selected=selected,
        attempts=attempts,
        rejected=rejected,
        cut_status=cut_status,
        cov_rows=cov_rows if selected == VARIANT_PRIMARY else cov_rows_fb,
        crossings=crossings["primary" if selected == VARIANT_PRIMARY else "fallback"],
        holiday=holiday_summary,
        run_config={
            "weighting": weighting,
            "half_life_days": half_life_days if weighting == WEIGHTING_TIME_DECAY else None,
            "cal_window_days": CAL_WINDOW_DAYS,
            "nominal": NOMINAL,
            "score": SCORE_KIND,
        },
    )
    _append_log(log_path, {
        "kind": "run_end",
        "selected_variant": selected,
        "regroup_attempts": attempts,
        "picp_rejected": rejected,
        "pooled_picp": pooled_cov.get("all", {}).get("picp") if pooled_cov else None,
        "elapsed_s": round(time.time() - started, 1),
        "test_accessed": False,
        "ended_at": _now(),
    })
    guard.assert_unchanged()
    _append_log(log_path, {"kind": "guard", "assert_unchanged": "pass", "ts": _now()})
    _write_manifest(
        run_dir / "manifest.json",
        run_id=run_dir.name,
        protocol=ep.PROTOCOL_ID,
        base_run=base_run.name,
        base_rows=len(base),
        date_min=base["date"].min().strftime("%Y-%m-%d"),
        date_max=base["date"].max().strftime("%Y-%m-%d"),
        cuts=[
            {"cut": cut.cut_date, "kind": cut.kind, "train_days": int(cut.train_days),
             "horizon": list(cut.horizon)}
            for cut in selected_cuts
        ],
        artifacts=["calibrator.json", "conditional_coverage.csv", "metrics.json"],
        weighting=weighting,
        half_life_days=half_life_days if weighting == WEIGHTING_TIME_DECAY else None,
        log_name=log_path.name,
    )
    print(f"[{run_dir.name}] 완료 · variant {selected} · {round(time.time() - started, 1)}s")
    return {"run_dir": str(run_dir), "variant": selected, "rejected": rejected,
            "cuts": cut_status, "gates": metrics["gates"]}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="분할 등각 보정 실행(8 cut 롤링)")
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    parser.add_argument("--cuts", default=None, help="쉼표 구분 cut 날짜(기본 8개 전부)")
    parser.add_argument("--lane", default="desktop")
    parser.add_argument("--base-run", default=str(BASE_RUN_DIR))
    parser.add_argument("--skip-calendar", action="store_true")
    parser.add_argument("--weighting", default=WEIGHTING_DEFAULT,
                        choices=[WEIGHTING_TIME_DECAY, WEIGHTING_NONE])
    parser.add_argument("--half-life-days", type=float, default=DECAY_HALF_LIFE_DAYS)
    args = parser.parse_args(argv)
    out = run(run_dir=Path(args.run_dir), cuts=args.cuts, lane=args.lane,
              base_run=Path(args.base_run), skip_calendar=args.skip_calendar,
              weighting=args.weighting, half_life_days=args.half_life_days)
    print(json.dumps({k: v for k, v in out.items() if k != "cuts"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())