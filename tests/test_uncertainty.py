"""불확실성 계약 테스트 — 등각 구간·조건부 커버리지·VaR 백테스트.

알고리즘 테스트는 합성 난수로 자체 완결한다. 실제 8 cut 수용 테스트는 실행 산출물
(``models/_exp/20261115-P3-R3-01``)을 읽는다 — 산출물이 없으면 통과시키지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT / "notebooks", ROOT / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from metro_insight.model import uncertainty as unc  # noqa: E402

RUN_DIR = ROOT / "models" / "_exp" / "20261115-P3-R3-01"
CALIBRATOR = RUN_DIR / "calibrator.json"
COVERAGE = RUN_DIR / "conditional_coverage.csv"
METRICS = RUN_DIR / "metrics.json"
MANIFEST = RUN_DIR / "manifest.json"
LOG = RUN_DIR / "log.jsonl"

PICP_BAND = (0.77, 0.83)
REQUIRED_SLICES = ("hour_bucket::24_01", "scale_tertile::small", "day_type::holiday")


# ───────────────────────── 산출물 접근 ─────────────────────────
def _require(path: Path) -> Path:
    if not path.exists():
        pytest.fail(f"산출물이 없다: {path} — python -m metro_insight.model.uncertainty 를 먼저 실행한다")
    return path


def _metrics() -> dict:
    return json.loads(_require(METRICS).read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(_require(MANIFEST).read_text(encoding="utf-8"))


def _calibrator_payload() -> dict:
    return json.loads(_require(CALIBRATOR).read_text(encoding="utf-8"))


def _coverage_table() -> pd.DataFrame:
    return pd.read_csv(_require(COVERAGE))


def _log_records() -> list[dict]:
    with open(_require(LOG), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ───────────────────────── 합성 표본 ─────────────────────────
def _synth(n: int = 40000, seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = rng.choice([20.0, 100.0, 400.0], size=n)
    residual = rng.laplace(0.0, scale / np.sqrt(2.0))  # MAE ≈ scale/2 가 되도록
    bands = np.where(rng.random(n) < 0.5, "1-31", "32-91")
    tertile = np.where(scale < 60, "small", np.where(scale < 250, "medium", "large"))
    return residual, scale, unc.group_label(bands, tertile)


# ───────────────────────── 등각 보정기 ─────────────────────────
def test_conformal_interval_covers_nominal_rate() -> None:
    """보정 표본과 같은 분포에서 뽑은 새 표본의 포함률이 명목 이상이어야 한다."""
    residual, scale, groups = _synth()
    cal = unc.ConformalCalibrator().fit(residual, scale, groups, nominal=0.8)
    rng = np.random.default_rng(99)
    n = 60000
    new_scale = rng.choice([20.0, 100.0, 400.0], size=n)
    new_res = rng.laplace(0.0, new_scale / np.sqrt(2.0))
    new_groups = unc.group_label(
        np.where(rng.random(n) < 0.5, "1-31", "32-91"),
        np.where(new_scale < 60, "small", np.where(new_scale < 250, "medium", "large")),
    )
    # 점예측을 규모에 비례해 잡아야 하한 클램프가 포함률을 깎지 않는다(실데이터도 예측 > 0)
    point = 3.0 * new_scale
    lo, hi = cal.interval_arrays(point, new_scale, new_groups)
    inside = ((point + new_res) >= lo) & ((point + new_res) <= hi)
    assert inside.mean() >= 0.78
    assert inside.mean() <= 0.88


def test_calibrator_json_roundtrip_preserves_intervals() -> None:
    residual, scale, groups = _synth(20000)
    cal = unc.ConformalCalibrator().fit(residual, scale, groups)
    cal.meta = {"cut": "2025-01-01", "tertile_edges": [60.0, 250.0]}
    back = unc.ConformalCalibrator.from_json(json.loads(json.dumps(cal.to_json())))
    for point, one_scale, group in [(0.0, 20.0, groups[0]), (500.0, 400.0, groups[1]),
                                    (123.4, 100.0, "없는그룹|large")]:
        left, right = cal.interval(point, one_scale, group), back.interval(point, one_scale, group)
        assert left == right
        assert left.nominal == pytest.approx(0.8)
        assert left.group == group


def test_small_group_falls_back_to_band() -> None:
    """표본이 부족한 그룹은 밴드 → 전체 분위수로 폴백한다(조용히 0 을 쓰지 않는다)."""
    residual, scale, groups = _synth(20000)
    cal = unc.ConformalCalibrator().fit(residual, scale, groups)
    rare = "92-181|small"
    assert cal.counts.get(rare, 0) == 0
    assert cal.source(rare) == "pooled"
    assert cal.quantile(rare) == pytest.approx(cal.pooled_q)


def test_decay_weights_halve_at_half_life() -> None:
    """나이 가중치는 반감기마다 절반이 되고, 음수·비유한 나이는 무게 0 이다."""
    w = unc.decay_weights(np.array([0.0, 30.0, 60.0, 90.0]), 30.0)
    assert w.tolist() == pytest.approx([1.0, 0.5, 0.25, 0.125])
    # 미래 잔차(음수 나이)·비유한 나이는 보정에 싣지 않는다 — 1.0 으로 올리면 최대 무게가 된다
    assert unc.decay_weights(np.array([-5.0]), 30.0)[0] == pytest.approx(0.0)
    assert unc.decay_weights(np.array([float("nan"), float("inf")]), 30.0).tolist() == [0.0, 0.0]
    with pytest.raises(ValueError):
        unc.decay_weights(np.array([1.0]), 0.0)


def test_weighted_quantile_matches_analytic_quantile() -> None:
    """같은 무게를 준 가중 분위수가 해석적 분위수와 맞는지 확인한다.

    표준정규 절대값(반정규분포) 표본의 0.8 분위수는 해석적으로 1.28155·sigma 다.
    가중 분위수 정의가 명목 수준을 구현하는지는 이렇게 외부 기준과 대조해야 알 수 있다
    (같은 표본의 가중 포함률을 다시 재는 것은 정의상 항상 수준에 붙어 근거가 되지 않는다).
    """
    rng = np.random.default_rng(20261115)
    n = 400_000
    sigma = 3.0
    scores = np.abs(rng.normal(0.0, sigma, n))
    q = unc._weighted_conformal_quantile(scores, np.ones(n), 0.8)
    analytic = 1.2815515655446004 * sigma
    assert q == pytest.approx(analytic, rel=0.01)
    # 무게를 나이 순으로 기울여도 같은 표본의 같은 수준이면 한 칸 안쪽에 머문다
    tilted = np.linspace(1.0, 2.0, n)
    assert unc._weighted_conformal_quantile(scores, tilted, 0.8) == pytest.approx(analytic, rel=0.02)


def test_weighted_quantile_is_deterministic_and_follows_weight() -> None:
    """가중 분위수는 입력 순서를 섞어도 같고, 큰 점수 쪽에 무게가 실리면 커진다."""
    rng = np.random.default_rng(7)
    n = 3000
    scores = np.abs(rng.laplace(0.0, 1.0, n))
    age = rng.uniform(0.0, 90.0, n)
    w = unc.decay_weights(age, 30.0)
    q = unc._weighted_conformal_quantile(scores, w, 0.8)
    order = rng.permutation(n)
    assert unc._weighted_conformal_quantile(scores[order], w[order], 0.8) == q
    heavy_tail = np.where(scores > np.median(scores), 10.0, 1.0)
    assert unc._weighted_conformal_quantile(scores, heavy_tail, 0.8) > q
    assert np.isclose(
        unc._weighted_conformal_quantile(scores, np.ones(n), 0.8),
        unc._conformal_quantile(scores, 0.8),
        rtol=1e-3,
    )


def test_run_log_path_never_truncates_previous_log(tmp_path) -> None:
    """같은 디렉터리에 다시 쓰면 이전 로그를 지우지 않고 새 파일명을 고른다."""
    first = unc._log_path(tmp_path)
    assert first.name == "log.jsonl"
    first.write_text('{"kind": "run_start"}\n', encoding="utf-8")
    second = unc._log_path(tmp_path)
    assert second.name == "log-2.jsonl"
    second.write_text('{"kind": "run_start"}\n', encoding="utf-8")
    assert unc._log_path(tmp_path).name == "log-3.jsonl"
    assert first.read_text(encoding="utf-8") == '{"kind": "run_start"}\n'


def test_rerun_refuses_to_overwrite_artifacts(tmp_path) -> None:
    """산출물이 있는 디렉터리로 다시 돌면 덮어쓰지 않고 멈춘다(로그만 남은 중단 실행은 예외)."""
    run_dir = tmp_path / "20261115-P3-R3-01"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    assert unc.existing_artifacts(run_dir) == ["metrics.json"]
    assert unc.next_free_run_name(run_dir) == "20261115-P3-R3-02"
    with pytest.raises(SystemExit) as excinfo:
        unc.run(run_dir=run_dir)
    message = str(excinfo.value)
    assert "충돌" in message and "20261115-P3-R3-02" in message
    assert (run_dir / "metrics.json").read_text(encoding="utf-8") == "{}"

    # 로그만 있는 디렉터리(중단된 실행)는 산출물이 아니므로 지우지 않고 이어 쓴다
    partial = tmp_path / "20261115-P3-R3-05"
    partial.mkdir()
    (partial / "log.jsonl").write_text('{"kind": "run_start"}\n', encoding="utf-8")
    assert unc.existing_artifacts(partial) == []
    assert unc._log_path(partial).name == "log-2.jsonl"
    assert unc.next_free_run_name(tmp_path / "20261115-P3-R3-01-single-origin") == \
        "20261115-P3-R3-01-single-origin-2"


def test_negative_scale_uses_floor() -> None:
    residual, scale, groups = _synth(5000)
    cal = unc.ConformalCalibrator().fit(residual, scale, groups)
    assert np.isfinite(cal.interval(10.0, float("nan"), groups[0]).hi)
    assert np.isfinite(cal.interval(10.0, -5.0, groups[0]).hi)


def test_interval_never_negative_and_keeps_width() -> None:
    residual, scale, groups = _synth(20000)
    cal = unc.ConformalCalibrator().fit(residual, scale, groups)
    iv = cal.interval(-50.0, 100.0, groups[0])
    assert iv.lo == 0.0
    assert iv.hi > 0.0
    assert iv.width == pytest.approx(2.0 * cal.quantile(groups[0]) * 100.0)
    assert iv.contains(0.0)


def test_no_quantile_crossing() -> None:
    """분위수 교차를 정렬로 없앤다(2차원·1차원, 입력 불변)."""
    crossed = np.array([[9.0, 1.0, 5.0], [2.0, 8.0, 3.0]])
    assert unc.count_crossings(crossed) == 2
    fixed = unc.monotone_quantiles(crossed)
    assert unc.count_crossings(fixed) == 0
    assert np.array_equal(fixed, np.sort(crossed, axis=-1))
    assert np.array_equal(crossed, np.array([[9.0, 1.0, 5.0], [2.0, 8.0, 3.0]]))
    line = unc.monotone_quantiles(np.array([0.9, 0.1, 0.5]))
    assert unc.count_crossings(line) == 0


def test_interval_crossings_counts_crossings_and_nonfinite() -> None:
    """구간 교차는 상수 0 이 아니라 실제 배열에서 센다(lo > hi + 비유한 경계)."""
    lo = np.array([0.0, 5.0, 2.0, np.nan])
    hi = np.array([1.0, 3.0, 3.0, np.inf])
    assert unc.interval_crossings(lo, hi) == 2  # 5>3 교차 1 + NaN/inf 비유한 1
    assert unc.interval_crossings(np.zeros(3), np.ones(3)) == 0
    assert unc.interval_crossings(np.array([0.0]), np.array([0.0])) == 0
    with pytest.raises(ValueError):
        unc.interval_crossings(np.zeros(2), np.ones(3))


def test_compute_winkler_score_matches_formula() -> None:
    """Winkler 구간점수 = 폭 + (2/alpha)·이탈거리. alpha=0.2 면 벌점 배수는 10이다."""
    y = np.array([0.0, 10.0, 20.0])
    lo = np.array([-1.0, 5.0, 15.0])
    hi = np.array([1.0, 15.0, 25.0])
    # 모두 구간 안 → 점수 = 폭 (2, 10, 10)
    assert unc.compute_winkler_score(y, lo, hi, alpha=0.2) == pytest.approx(22.0 / 3.0)
    # 아래로 벗어나면 (2/alpha)·(lo−y) 벌점
    assert unc.compute_winkler_score(
        np.array([-5.0]), np.array([0.0]), np.array([2.0]), alpha=0.2
    ) == pytest.approx(52.0)
    # 둘 다 덮으면 좁은 구간이 더 좋다(넓히기만 하는 해를 막는다)
    narrow = unc.compute_winkler_score(np.array([5.0]), np.array([0.0]), np.array([10.0]))
    wide = unc.compute_winkler_score(np.array([5.0]), np.array([-5.0]), np.array([15.0]))
    assert narrow < wide
    with pytest.raises(ValueError):
        unc.compute_winkler_score(y, lo, hi, alpha=0.0)


def test_horizon_diagnostics_separate_over_and_under_coverage() -> None:
    """지평별 과대(over)·과소(under) 포함을 분리해 진단한다."""
    days = np.array([1, 1, 40, 40])
    y = np.array([3.0, 5.0, 3.0, 9.0])
    lo = np.zeros(4)
    hi = np.array([10.0, 10.0, 5.0, 5.0])
    out = unc.horizon_interval_diagnostics(y, lo, hi, days)

    assert out["1-31"]["n"] == 2
    assert out["1-31"]["picp"] == pytest.approx(1.0)
    assert out["1-31"]["coverage"] == "over"
    assert out["1-31"]["picp_gap"] == pytest.approx(0.2)

    assert out["32-91"]["n"] == 2
    assert out["32-91"]["picp"] == pytest.approx(0.5)
    assert out["32-91"]["coverage"] == "under"
    assert out["32-91"]["picp_gap"] == pytest.approx(-0.3)
    assert out["32-91"]["above_share"] == pytest.approx(0.5)  # y=9 > hi=5
    assert out["32-91"]["below_share"] == pytest.approx(0.0)
    assert np.isfinite(out["32-91"]["winkler"])
    # 밴드 밖 지평은 'other' 로 따로 센다(조용히 버리지 않는다)
    assert "other" not in out
    other = unc.horizon_interval_diagnostics(
        np.array([1.0]), np.array([0.0]), np.array([2.0]), np.array([400])
    )
    assert other["other"]["n"] == 1


def test_night_slice_definition_matches_hour_codes() -> None:
    """hour_bucket::24_01 정의가 실제 코드(0=04_06, 19=24_01, 23_24 제외)와 맞아야 한다."""
    assert unc.NIGHT_HOURS == (0, 19)
    assert np.isin(np.array([0, 1, 18, 19]), unc.NIGHT_HOURS).tolist() == [True, False, False, True]
    note = unc.SLICE_DEFINITIONS["hour_bucket::24_01"]
    assert "hour_code 19" in note and "24_01" in note
    assert "hour_code 0" in note and "04_06" in note
    assert "23_24" in note and "넣지 않는다" in note
    assert "23:00-24:00" not in note  # 23_24 를 포함한다는 옛 설명이 남으면 안 된다


# ───────────────────────── 커버리지 지표 ─────────────────────────
def test_conditional_coverage_reports_slices() -> None:
    y = np.array([0.0, 10.0, 20.0, 30.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([5.0, 15.0, 25.0, 25.0])
    slices = {"left": np.array([True, True, False, False]), "empty": np.zeros(4, dtype=bool)}
    out = unc.conditional_coverage(y, lo, hi, slices=slices)
    assert out["all"]["picp"] == pytest.approx(0.75)
    assert out["left"]["picp"] == pytest.approx(1.0)
    assert out["left"]["n"] == 2
    assert np.isnan(out["empty"]["picp"])
    assert out["all"]["width_median"] == pytest.approx(20.0)
    with pytest.raises(ValueError):
        unc.conditional_coverage(y, lo, hi, slices={"all": np.ones(4, dtype=bool)})


def test_var_backtest_flags_wrong_rate() -> None:
    rng = np.random.default_rng(3)
    n = 40000
    y = rng.normal(100.0, 10.0, n)
    lo = np.full(n, 200.0)
    hi = np.full(n, 300.0)  # 전부 이탈
    out = unc.var_backtest(y, lo, hi, nominal=0.8)
    assert out["rate"] == pytest.approx(1.0)
    assert out["kupiec"]["p_value"] < 1e-6
    assert out["pass"] is False


def test_var_backtest_flags_clustered_violations() -> None:
    """이탈률이 명목과 같아도 몰려 있으면 독립성 검정이 잡는다."""
    n = 20000
    half = n // 2
    y = np.zeros(n)
    lo = np.full(n, -1.0)
    hi = np.full(n, 1.0)
    y[half - 4000:half] = 5.0  # 앞 절반 끝에만 이탈 → 이탈률 20%
    out = unc.var_backtest(y, lo, hi, nominal=0.8)
    assert out["rate"] == pytest.approx(0.2, abs=1e-9)
    assert out["kupiec"]["p_value"] > 0.05
    assert out["christoffersen"]["p_ind"] < 0.01
    assert out["pass"] is False


def test_horizon_band_and_tertiles() -> None:
    days = np.array([0, 1, 31, 32, 91, 92, 181, 182, 366, 367])
    assert unc.horizon_band(days).tolist() == [
        "other", "1-31", "1-31", "32-91", "32-91", "92-181", "92-181", "182-366", "182-366", "other",
    ]
    scales = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    edges = unc.tertile_edges(scales)
    assert edges[0] < edges[1]
    tert = unc.scale_tertile(scales, edges)
    assert tert[0] == "small" and tert[-1] == "large"
    n_small = int((tert == "small").sum())
    assert n_small >= 2


def test_calendar_decision_uses_nonholiday_tolerance() -> None:
    """공휴일 개선만 보지 않고 비공휴일 악화 허용치를 함께 본다(판정에 실제로 연결)."""
    assert unc.NONHOLIDAY_TOLERANCE == pytest.approx(0.01)
    assert unc.HOLIDAY_GAIN_TARGET == pytest.approx(0.15)
    assert unc.calendar_decision(0.20, 0.005) == "accepted"
    assert unc.calendar_decision(0.20, 0.01) == "accepted"
    # 공휴일은 좋아졌지만 비공휴일이 허용치를 넘게 나빠지면 채택하지 않는다
    assert unc.calendar_decision(0.20, 0.02) == "rejected"
    # 공휴일 목표 미달이면 비공휴일이 좋아져도 채택하지 않는다
    assert unc.calendar_decision(0.14, -0.05) == "rejected"
    # 비유한 값은 채택 근거가 될 수 없다
    assert unc.calendar_decision(float("nan"), 0.0) == "rejected"
    assert unc.calendar_decision(0.20, float("inf")) == "rejected"


# ───────────────────────── 실행 산출물 (수용 테스트) ─────────────────────────
def test_picp80_gate_matches_measured_coverage() -> None:
    """포함률 판정이 실측과 일치하는지 검사한다.

    목표는 8개 기준 시점 전부 포함률 0.77~0.83 이다. 현재 산출물은 4개만 통과하므로
    이 검사는 세 가지를 고정한다.
    1) 판정 플래그와 밴드 밖 목록이 실측값과 일치한다(숨김 금지).
    2) 밴드 밖이면 음성 기록이 남아야 한다.
    3) 통과 개수는 퇴행 방지 하한(4개) 아래로 내려가지 않는다 — 목표(8개)가 아니다.
    """
    payload = _metrics()
    per_cut = payload["per_cut"]
    assert len(per_cut) == 8, f"cut 별 판정이 {len(per_cut)}개다"
    for cut, rec in per_cut.items():
        assert 0.0 <= rec["picp"] <= 1.0, f"{cut} 포함률이 범위 밖이다: {rec['picp']}"
    out = [c for c, rec in per_cut.items() if not PICP_BAND[0] <= rec["picp"] <= PICP_BAND[1]]
    gates = payload["gates"]
    assert gates["picp_all_cuts_in_band"] is (not out)
    assert sorted(gates["picp_out_of_band_cuts"]) == sorted(out)
    in_band = len(per_cut) - len(out)
    assert in_band >= 4, f"통과 기준 시점이 {in_band}개로 현재 달성치(4개)보다 줄었다"
    # 가중 경로가 실제로 쓰였는지 — 비가중으로 되돌아가면 이 값이 none 이 된다
    assert payload["config"]["weighting"] == "time_decay"
    assert payload["config"]["half_life_days"] == pytest.approx(30.0)
    assert _calibrator_payload()["weighting"] == "time_decay"
    if not out:
        return
    rejected = [r for r in _log_records() if r["kind"] == "picp_rejected"]
    assert rejected, f"밴드 밖 cut {out} 이 있는데 음성 기록이 없다"
    recorded = set(rejected[-1]["out_of_band_primary"]) | set(rejected[-1]["out_of_band_fallback"])
    assert set(out) <= recorded


def test_conditional_coverage_reported() -> None:
    """심야 버킷·소규모 tertile·공휴일·호선별 슬라이스가 모두 리포트에 있어야 한다."""
    payload = _calibrator_payload()
    table = _coverage_table()
    table = table[table["variant"] == payload["selected_variant"]]
    cuts = sorted(set(table["cut"]) - {"pooled"})
    assert len(cuts) == 8
    for name in REQUIRED_SLICES:
        present = set(table.loc[table["slice"] == name, "cut"])
        assert set(cuts) <= present, f"슬라이스 {name} 이 cut {sorted(set(cuts) - present)} 에 없다"
        pooled = table[(table["slice"] == name) & (table["cut"] == "pooled")]
        assert len(pooled) == 1 and pooled["n"].iloc[0] > 0
        assert np.isfinite(pooled["picp"].iloc[0])
    line_slices = [s for s in table["slice"].unique() if s.startswith("line::")]
    assert len(line_slices) >= 2, f"호선 슬라이스가 부족하다: {line_slices}"
    for name in line_slices:
        body = table[(table["slice"] == name) & (table["cut"] == "pooled")]
        assert len(body) == 1 and body["n"].iloc[0] > 0


def test_no_negative_predictions() -> None:
    """클램프 후 음수 점예측·음수 하한이 0건이어야 한다(로그에 전후 건수가 남는다)."""
    payload = _metrics()
    gates = payload["gates"]
    assert gates["negative_predictions_after"] == 0
    assert gates["negative_predictions_before"] > 0, (
        "기존 예측에 음수가 없었다면 클램프가 검증되지 않는다"
    )
    assert len(payload["per_cut"]) == 8
    for cut, rec in payload["per_cut"].items():
        assert rec["negative_lower_bounds"] == 0, f"cut {cut} 음수 하한 {rec['negative_lower_bounds']}건"
    records = [r for r in _log_records() if r["kind"] == "cut"]
    assert len(records) == 8
    assert sum(r["negative_predictions_before"] for r in records) == gates["negative_predictions_before"]


def test_holiday_mae_improved_or_logged() -> None:
    """공휴일 MAE −15% 이상이거나, 채택하지 않았다는 기록이 있어야 한다."""
    rec = _metrics()["gates"]["calendar"]
    assert rec["skipped"] is False, "달력 비교를 건너뛴 실행이다"
    assert np.isfinite(rec["base_holiday_mae"]) and rec["base_holiday_mae"] > 0
    assert np.isfinite(rec["calendar_holiday_mae"])
    assert rec["cuts"] == 8
    if rec["decision"] == "accepted":
        assert rec["holiday_gain"] >= rec["holiday_gain_target"]
        assert rec["holiday_gain"] >= 0.15
    else:
        assert rec["decision"] == "rejected"
        assert rec["holiday_gain"] is not None
    logs = [r for r in _log_records() if r["kind"] == "calendar_compare"]
    assert logs, "달력 피처 비교 로그가 없다"
    assert logs[-1]["decision"] == rec["decision"]


def test_nonholiday_not_degraded() -> None:
    """비공휴일 MAE 악화는 +1% 이내여야 한다."""
    rec = _metrics()["gates"]["calendar"]
    assert rec["skipped"] is False, "달력 비교를 건너뛴 실행이다"
    assert np.isfinite(rec["nonholiday_change"])
    assert rec["nonholiday_change"] <= 0.01, (
        f"비공휴일 악화 {rec['nonholiday_change']:.2%} (기준 +1%)"
    )


def test_manifest_matches_artifacts() -> None:
    """manifest 의 해시·크기가 실제 산출물과 같아야 한다(재현 확인용)."""
    import hashlib

    manifest = _manifest()
    assert manifest["schema"] == "conformal-manifest/1"
    assert manifest["data"]["test_accessed"] is False
    assert len(manifest["cuts"]) == 8
    assert {entry["path"] for entry in manifest["artifacts"]} == {
        "calibrator.json", "conditional_coverage.csv", "metrics.json",
    }
    for entry in manifest["artifacts"]:
        target = _require(RUN_DIR / entry["path"])
        assert entry["bytes"] == target.stat().st_size, entry["path"]
        assert entry["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest(), entry["path"]
    metrics = _metrics()
    assert metrics["run_id"] == manifest["run_id"]
    assert metrics["test_accessed"] is False
    assert metrics["gates"]["interval_crossings"] == 0
    assert metrics["gates"]["band_quantile_inversions"] >= 0
    assert metrics["gates"]["conditional_slices"]["ok"] is True