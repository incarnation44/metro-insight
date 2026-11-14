"""평가 프로토콜 계약 테스트(계약 8개 + legacy 재현 표시 테스트 1개).

실제 data/processed/*.csv.gz 는 열지 않는다. 합성 프레임을 tmp 저장소에 써서 배관만 검증하고,
실제 Test 재현은 승인 1회 실행의 증거 파일(models/_exp/legacy_b_baseline.json)로 확인한다.
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

import eval_protocol as ep  # noqa: E402
import verify_manifest as vm  # noqa: E402

from metro_insight.data import guard as guard_mod  # noqa: E402

SPLIT_COLS = [
    "date", "station_code", "line_code", "hour_code", "weekday",
    "is_weekend", "is_holiday", "temperature", "rainfall", "snowfall", "boarding",
]


# ───────────────────────── 합성 데이터 ─────────────────────────
def _synth_frame(start: str, end: str, *, seed: int = 7) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    index = pd.MultiIndex.from_product(
        [dates, [1, 2, 3], [0, 1, 2]], names=["date", "station_code", "hour_code"]
    )
    frame = pd.DataFrame(index=index).reset_index()
    rng = np.random.default_rng(seed)
    frame["line_code"] = frame["station_code"].astype("int16")
    frame["weekday"] = frame["date"].dt.weekday.astype("int8")
    frame["is_weekend"] = (frame["weekday"] >= 5).astype("int8")
    frame["is_holiday"] = (frame["date"].dt.day == 1).astype("int8")
    frame["temperature"] = 15.0
    frame["rainfall"] = 0.0
    frame["snowfall"] = 0.0
    frame["boarding"] = (
        100 + frame["station_code"] * 10 + frame["hour_code"] * 5
        + (frame["weekday"] < 5) * 40 + rng.normal(0, 5, len(frame))
    ).clip(lower=0)
    return frame[SPLIT_COLS]


def _no_frozen_guard(root, **kwargs):
    """합성 저장소에서는 동결표(실제 저장소 해시)를 쓰지 않는다."""
    return guard_mod.DataGuard(root, frozen={}, **kwargs)


def _make_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    (root / "data" / "processed").mkdir(parents=True)
    (root / "models" / "_exp").mkdir(parents=True)
    ranges = {
        "train": ("2025-09-01", "2025-12-31"),
        "valid": ("2026-01-01", "2026-02-14"),
        "test": ("2026-02-15", "2026-06-30"),
    }
    for name, (start, end) in ranges.items():
        _synth_frame(start, end).to_csv(
            root / "data" / "processed" / f"{name}.csv.gz", index=False, compression="gzip"
        )
    monkeypatch.setattr(ep, "ROOT", root)
    monkeypatch.setattr(ep, "DataGuard", _no_frozen_guard)
    return root


# ───────────────────────── 계약 ─────────────────────────
def test_cuts_are_eight_and_ordered() -> None:
    assert len(ep.CUTS) == 8
    dates = [cut.cut_date for cut in ep.CUTS]
    assert dates == sorted(dates)
    assert dates[0] == "2024-11-01" and dates[-1] == "2026-01-01"
    assert [cut.horizon for cut in ep.CUTS] == [
        (182, 366), (182, 365), (182, 366), (182, 364),
        (1, 181), (1, 181), (1, 181), (1, 180),
    ]
    for cut in ep.CUTS:
        expected = "long" if cut.horizon[0] >= 182 else "short"
        assert cut.kind == expected
        assert cut.train_days > 0
    assert ep.PROTOCOL_ID == "rolling8-v1"


def test_test_access_denied_without_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(guard_mod.APPROVAL_ENV, raising=False)
    guard = guard_mod.DataGuard(ROOT, allow_test=True, approval_id=None, frozen={})
    with pytest.raises(guard_mod.TestAccessDenied):
        guard.open_split("test_dev", purpose="승인 없이 Test 를 열면 안 된다")
    guard_with_id = guard_mod.DataGuard(ROOT, allow_test=True, approval_id="APP-X", frozen={})
    with pytest.raises(guard_mod.TestAccessDenied):
        guard_with_id.open_split("test_final", purpose="환경변수 없이는 열 수 없다")


def test_manifest_roundtrip(tmp_path: Path) -> None:
    payload = _payload()
    path = ep.write_manifest(tmp_path, payload)
    assert path.name == "manifest.json"
    doc = json.loads(path.read_text(encoding="utf-8"))

    assert vm.check_manifest(doc) == []
    assert doc["schema"] == "metro-manifest/1"
    assert len(doc) >= 12

    broken = json.loads(json.dumps(doc))
    broken["features"]["names"] = list(reversed(broken["features"]["names"]))
    assert any("order_hash" in problem for problem in vm.check_manifest(broken))

    with pytest.raises(ValueError):
        missing = {k: v for k, v in _payload().items() if k != "seeds"}
        ep.write_manifest(tmp_path / "bad", missing)


def test_reproduces_legacy_b_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """합성 데이터로 이전 기준선(legacy) 재현 경로(승인 · 프로파일 고정 · 채점 · 산출물)를 통과시킨다."""
    root = _make_repo(tmp_path, monkeypatch)
    monkeypatch.setenv(guard_mod.APPROVAL_ENV, "1")
    out = root / "models" / "_exp" / "20260101-P0-R0-01"

    code = ep.main([
        "--protocol", ep.PROTOCOL_ID, "--model", "B", "--feature-set", "base",
        "--out", str(out), "--approval-id", ep.LEGACY_APPROVAL_ID, "--legacy-baseline",
    ])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert code in (0, 3)
    assert vm.check_manifest(manifest) == []
    assert manifest["run_id"] == "20260101-P0-R0-01"
    per_cut = manifest["metrics"]["per_cut"]
    assert len(per_cut) == 1 and per_cut[0]["n"] > 0
    assert per_cut[0]["cut_date"] == ep.LEGACY_CUT.cut_date
    assert per_cut[0]["n"] % 9 == 0  # 날짜 × 역 3 × 시간 3
    assert per_cut[0]["n"] // 9 == 136  # 2026-02-15~2026-06-30 전 구간
    assert manifest["metrics"]["pooled"]["mae"] > 0

    evidence = json.loads((root / "models" / "_exp" / "legacy_b_baseline.json").read_text(encoding="utf-8"))
    assert evidence["approval_id"] == ep.LEGACY_APPROVAL_ID
    # 합성 데이터는 92.72 근처가 아니다 → '통과(pass)' 를 주장하면 안 된다
    assert evidence["verdict"]["result"] == "fail"
    assert evidence["target_mae"] == 92.72
    assert evidence["tolerance"] == 0.5

    log = (root / "models" / "_exp" / "access.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["split"] == "test_dev" for line in log)
    assert any(json.loads(line)["split"] == "test_final" for line in log)


def test_bootstrap_resamples_days_not_rows() -> None:
    days = pd.date_range("2025-01-01", periods=136, freq="D")
    rows_a, rows_b = [], []
    for day in days:
        # 하루 안에 두 행: A 는 (100, 0), B 는 (0, 100) → 날짜 평균은 둘 다 50
        rows_a += [{"date": day, "abs_err": 100.0, "n": 1}, {"date": day, "abs_err": 0.0, "n": 1}]
        rows_b += [{"date": day, "abs_err": 0.0, "n": 1}, {"date": day, "abs_err": 100.0, "n": 1}]
    daily_a = pd.DataFrame(rows_a)
    daily_b = pd.DataFrame(rows_b)

    result = ep.day_block_bootstrap(daily_a, daily_b, n=10_000, seed=42)
    assert result["blocks"] == 136
    assert result["n_boot"] == 10_000
    # 날짜 단위로 뽑으면 같은 날의 두 값이 함께 움직인다 → 차이가 항상 0, CI 폭 0
    assert result["mae_a"] == pytest.approx(50.0)
    assert result["mae_b"] == pytest.approx(50.0)
    assert result["mean_diff_b_minus_a"] == 0.0
    assert result["ci95"] == [0.0, 0.0]

    # 날짜별 평균이 흔들리면 CI 폭이 0보다 커야 한다(테스트가 무의미하게 통과하지 않도록)
    shifted = pd.DataFrame([
        {"date": day, "abs_err": 0.0 if i % 2 else 100.0, "n": 1} for i, day in enumerate(days)
    ])
    result2 = ep.day_block_bootstrap(daily_a, shifted, n=2_000, seed=42)
    assert result2["ci95"][1] - result2["ci95"][0] > 0


def test_pooled_weights_by_n() -> None:
    def entry(cut: str, n: int, mae: float) -> dict:
        return {
            "cut_date": cut, "n": n, "mae": mae,
            "abs_err_sum": mae * n, "sq_err_sum": mae ** 2 * n,
            "signed_err_sum": 0.0, "y_abs_sum": 1_000.0 * n, "negative_predictions": 0,
        }

    agg = ep.pooled([entry("2025-01-01", 100, 10.0), entry("2025-07-01", 300, 20.0)])
    assert agg["n"] == 400
    assert agg["mae"] == pytest.approx(17.5)  # 단순 평균 15.0 이 아니다
    assert agg["mae"] != pytest.approx(15.0)
    assert agg["per_cut_mae"] == {"2025-01-01": 10.0, "2025-07-01": 20.0}
    assert agg["cuts"] == 2


def test_feature_hash_stable() -> None:
    first = ep.feature_order_hash(ep.BASE_FEATURES)
    second = ep.feature_order_hash(list(ep.BASE_FEATURES))
    assert first == second
    assert len(first) == 64
    assert ep.feature_order_hash(list(reversed(ep.BASE_FEATURES))) != first
    # 검증기(독립 구현)가 같은 지문을 계산한다
    assert vm.feature_order_hash(ep.BASE_FEATURES) == first
    # 순서가 다르면 manifest 도 다른 순서로 적는다
    doc = _payload()
    doc["features"]["names"] = list(reversed(ep.BASE_FEATURES))
    assert any("order_hash" in problem for problem in vm.check_manifest(doc))


def test_run_id_format(tmp_path: Path) -> None:
    assert ep.RUN_ID_RE.match("20261114-P0-R0-01")
    assert ep.RUN_ID_RE.match("20261114-P12-R3-07")
    for bad in (
        "2026-11-14-P0-R0-01", "20261114-P0-R0", "20261114-P0-R0-1",
        "2026111-P0-R0-01", "20261114-p0-R0-01", "20261114-P0-R0-001",
    ):
        assert not ep.RUN_ID_RE.match(bad)

    # 데이터를 읽기 전에 run_id 형식부터 막는다
    with pytest.raises(SystemExit):
        ep.main([
            "--protocol", ep.PROTOCOL_ID, "--model", "B", "--feature-set", "base",
            "--out", str(tmp_path / "bad-name"),
        ])


@pytest.mark.legacy_repro
def test_legacy_repro_evidence_recorded() -> None:
    """승인 1회 실행의 증거 파일을 검사한다(합성 데이터가 아닌 실제 Test 재현 기록)."""
    assert ep.LEGACY_TARGET_MAE == 92.72
    assert ep.LEGACY_TOLERANCE == 0.5
    assert ep.LEGACY_CUT.cut_date == "2026-02-14"
    assert ep.LEGACY_CUT.horizon == (1, 136)

    evidence_path = ROOT / "models" / "_exp" / "legacy_b_baseline.json"
    if not evidence_path.exists():
        # 증거 파일이 없는 환경(깨끗한 복제본(clone))에서는 계약 상수만 검사한다 — 건너뛰지 않는다
        return
    doc = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert doc["approval_id"] == ep.LEGACY_APPROVAL_ID
    assert abs(doc["observed_mae"] - ep.LEGACY_TARGET_MAE) <= ep.LEGACY_TOLERANCE
    assert doc["verdict"]["result"] == "pass"
    assert doc["cut"]["days"] == 136


# ───────────────────────── 도우미 ─────────────────────────
def _payload() -> dict:
    names = list(ep.BASE_FEATURES)
    return {
        "schema": ep.MANIFEST_SCHEMA,
        "run_id": "20261114-P0-R0-01",
        "created_at": "2026-11-14T00:00:00+00:00",
        "git": {"commit": "0" * 40, "dirty": False},
        "task": "T2",
        "horizon": {"min": 1, "max": 136},
        "data": {
            "train": {
                "path": "data/processed/train.csv.gz", "sha256": "0" * 64, "rows": 10,
                "date_min": "2025-01-01", "date_max": "2025-12-31",
            },
        },
        "features": {
            "names": names,
            "order_hash": ep.feature_order_hash(names),
            "order_hash_rule": "sha256-newline-join-v1",
            "as_of_rule": ep.AS_OF_RULE,
        },
        "params": dict(ep.PARAMS),
        "seeds": {"model": 42, "bootstrap": 42},
        "protocol": ep.PROTOCOL_ID,
        "metrics": {
            "per_cut": [{
                "cut_date": "2026-02-14", "n": 10, "mae": 1.0, "rmse": 1.5, "wape": 0.1,
                "bias": 0.0, "negative_predictions": 0, "horizon_bucket": {},
                "date_min": "2026-02-15", "date_max": "2026-02-20",
            }],
            "pooled": {"n": 10, "mae": 1.0, "rmse": 1.5, "wape": 0.1, "bias": 0.0},
            "ci": {"ci95": [0.0, 0.0], "n_boot": 10},
        },
        "verdict": {"result": "recorded"},
        "artifacts": {"metrics": {"path": "metrics.json"}},
    }


def test_bootstrap_labels_overlapping_cuts_as_dependent() -> None:
    """cut 창이 겹치면 블록 독립 가정이 깨진다 → 라벨과 경고가 정확해야 한다."""
    rows_a, rows_b = [], []
    for cut in ("2024-11-01", "2024-12-15"):
        for day in pd.date_range("2025-06-01", periods=3, freq="D"):
            rows_a.append({"cut": cut, "date": day, "abs_err": 10.0, "n": 1})
            rows_b.append({"cut": cut, "date": day, "abs_err": 12.0, "n": 1})

    result = ep.day_block_bootstrap(pd.DataFrame(rows_a), pd.DataFrame(rows_b), n=200, seed=42)
    assert result["unit"].startswith("cut+date")
    assert result["blocks"] == 6
    assert result["blocks_are_independent"] is False
    assert "하한" in result["note"]


# ───────────────────────── 다중 지평 ─────────────────────────
def _horizon_frame(start: str = "2025-01-01", days: int = 120) -> pd.DataFrame:
    """밴드·lag 규칙 검증용 합성 프레임. 날짜가 촘촘해 lag 창 계산을 직접 확인할 수 있다."""
    return _synth_frame(start, (pd.Timestamp(start) + pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d"))


def test_horizon_bands_cover_every_day() -> None:
    """1~366일이 빈틈·겹침 없이 덮여야 한다(밴드 밖 지평은 예측할 모델이 없다)."""
    bands = ep.HORIZON_BANDS
    assert bands[0][0] == 1 and bands[-1][1] == 366
    for (_, hi), (next_lo, _) in zip(bands, bands[1:], strict=False):
        assert next_lo == hi + 1, f"밴드 사이에 빈틈이나 겹침이 있다: {hi} → {next_lo}"
    assert list(ep.SPEC_BANDS) == list(bands[:4])
    assert ep.LONG_BAND == bands[4]


def test_band_training_frame_keeps_horizon_inside_band() -> None:
    """학습 표본의 h 는 밴드 안에 있어야 하고, 목표일은 as_of 이전이어야 한다."""
    frame = _horizon_frame()
    features = ep.build_features(frame, as_of="2025-03-01", feature_set="horizon")
    band = (8, 28)
    sample = ep.band_training_frame(features, band=band, as_of="2025-03-01", seed=42)

    h = sample["h"].to_numpy()
    assert h.min() >= band[0] and h.max() <= band[1]
    assert sample["date"].max() <= pd.Timestamp("2025-03-01")
    # 밴드 폭 21일이 골고루 섞여야 한다(한 지평에 몰리면 밴드 학습이 아니다)
    assert len(set(h.tolist())) >= 20
    assert list(sample.attrs["feature_names"]) == ep.feature_names(features)


def test_band_profile_uses_only_prior_days() -> None:
    """밴드 학습 표본의 프로파일은 t=d−h 시점 값이어야 한다 — 그 뒤 값을 고쳐도 변하면 안 된다."""
    frame = _horizon_frame()
    features = ep.build_features(frame, as_of="2025-03-01", feature_set="horizon")
    band = (8, 28)
    clean = ep.band_training_frame(features, band=band, as_of="2025-03-01", seed=42)

    corrupted = frame.copy()
    late = corrupted["date"] > pd.Timestamp("2025-03-01")
    corrupted.loc[late, "boarding"] = corrupted.loc[late, "boarding"] * 5.0
    features_bad = ep.build_features(corrupted, as_of="2025-03-01", feature_set="horizon")
    dirty = ep.band_training_frame(features_bad, band=band, as_of="2025-03-01", seed=42)

    profile_cols = [
        *ep.cf.PROFILE_COLS_SH, *ep.cf.PROFILE_COLS_MONTH, *ep.cf.PROFILE_COLS_WEEKDAY
    ]
    pd.testing.assert_frame_equal(clean[profile_cols], dirty[profile_cols])
    # 목표값 자체도 as_of 이후 값을 쓰지 않는다
    pd.testing.assert_series_equal(clean["boarding"], dirty["boarding"])


def test_banded_model_routes_rows_by_horizon() -> None:
    """행마다 h 에 맞는 밴드 모델이 예측에 쓰여야 한다."""
    class _Const:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(frame), self.value)

    models = {(1, 7): _Const(1.0), (8, 28): _Const(2.0)}
    model = ep.BandedModel(models, ["profile_n"])
    frame = pd.DataFrame({"h": [1, 5, 7, 8, 20, 28]})
    assert list(model.predict(frame)) == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]

    with pytest.raises(ValueError, match="밴드"):
        model.predict(pd.DataFrame({"h": [40]}))


def test_matched_lags_ignore_post_as_of_values() -> None:
    """as_of 이후 값은 lag 재계산에서도 쓰지 않는다 — 학습 lag 는 행 날짜, 평가 lag 는 as_of 기준."""
    frame = _horizon_frame()
    as_of = "2025-03-01"
    clean = ep.build_features(frame, as_of=as_of, feature_set="matched")

    corrupted = frame.copy()
    late = corrupted["date"] > pd.Timestamp(as_of)
    corrupted.loc[late, "boarding"] = corrupted.loc[late, "boarding"] * 7.0
    dirty = ep.build_features(corrupted, as_of=as_of, feature_set="matched")

    for col in ep.MATCHED_EXTRA:
        pd.testing.assert_series_equal(clean[col], dirty[col], check_names=False)


def test_matched_eval_lag_is_frozen_at_as_of() -> None:
    """평가 행의 lag 는 as_of 당일까지의 값으로 고정된다(지평이 길어도 같은 값)."""
    frame = _horizon_frame()
    as_of = "2025-03-01"
    features = ep.build_features(frame, as_of=as_of, feature_set="matched")

    eval_rows = features.loc[features["date"] > pd.Timestamp(as_of)]
    window = (pd.Timestamp(as_of) - pd.Timedelta(days=7), pd.Timestamp(as_of) - pd.Timedelta(days=1))
    before = frame.loc[(frame["date"] >= window[0]) & (frame["date"] <= window[1])]
    expected = before.groupby(["station_code", "hour_code"])["boarding"].mean()

    for _, row in eval_rows.drop_duplicates(["station_code", "hour_code"]).iterrows():
        key = (row["station_code"], row["hour_code"])
        assert row["mean_prev7d"] == pytest.approx(float(expected.loc[key]))

    # 지평이 다른 두 행도 lag 값이 같아야 한다(frozen 이라는 증거)
    pairs = eval_rows.groupby(["station_code", "hour_code"])[["lag_1d", "mean_prev7d"]].nunique()
    assert (pairs.to_numpy() == 1).all()