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


def test_run_log_keeps_one_verdict_line(tmp_path: Path) -> None:
    """한 실행의 판정 줄은 하나만 남는다 — 같은 실행 식별자로 다시 남겨도 늘지 않는다."""
    run = tmp_path / "20260101-P0-R0-01"
    record = {"run_id": "20260101-P0-R0-01", "verdict": "adopt", "pooled_mae": 87.16}
    ep._append_run_log(run, dict(record))
    ep._append_run_log(run, dict(record))
    lines = (run / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == "20260101-P0-R0-01"

    # 다른 실행 식별자는 따로 남는다
    ep._append_run_log(run, {"run_id": "20260101-P0-R0-02", "verdict": "rejected"})
    lines = (run / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_noise_file_counts_as_collision(tmp_path: Path) -> None:
    """이름을 가리지 않고 모든 일반 파일을 충돌로 센다 — report.md 만 있어도 덮어쓰지 않는다."""
    out = tmp_path / "20260101-P0-R0-01"
    out.mkdir()
    (out / "report.md").write_text("# 기록", encoding="utf-8")
    assert ep.existing_artifacts(out) == ["report.md"]
    with pytest.raises(SystemExit) as excinfo:
        ep.main([
            "--protocol", ep.PROTOCOL_ID, "--model", "B", "--feature-set", "base",
            "--out", str(out),
        ])
    assert "충돌" in str(excinfo.value)
    assert (out / "report.md").read_text(encoding="utf-8") == "# 기록"


def test_run_id_overflow_is_system_exit(tmp_path: Path) -> None:
    """순번이 99를 넘으면 트레이스백이 아니라 안내 메시지로 멈춘다."""
    out = tmp_path / "20260101-P0-R0-99"
    out.mkdir()
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        ep.next_free_run_id(out)
    with pytest.raises(SystemExit) as excinfo:
        ep.main([
            "--protocol", ep.PROTOCOL_ID, "--model", "B", "--feature-set", "base",
            "--out", str(out),
        ])
    assert "99" in str(excinfo.value)


def test_existing_run_id_is_refused(tmp_path: Path) -> None:
    """산출물이 있는 디렉터리를 다시 쓰려 하면 거부하고 비어 있는 순번을 알려준다."""
    out = tmp_path / "20260101-P0-R0-01"
    out.mkdir()
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    assert ep.existing_artifacts(out) == ["manifest.json"]

    with pytest.raises(SystemExit) as excinfo:
        ep.main([
            "--protocol", ep.PROTOCOL_ID, "--model", "B", "--feature-set", "base",
            "--out", str(out),
        ])
    assert "충돌" in str(excinfo.value)
    assert ep.next_free_run_id(out) == "20260101-P0-R0-02"

    # 비어 있는 형제 디렉터리도 있으면 그다음 순번으로 넘어간다
    (tmp_path / "20260101-P0-R0-02").mkdir()
    (tmp_path / "20260101-P0-R0-02" / "log.jsonl").write_text("", encoding="utf-8")
    assert ep.next_free_run_id(out) == "20260101-P0-R0-03"
    # 빈 디렉터리는 충돌이 아니다
    empty = tmp_path / "20260101-P0-R0-09"
    empty.mkdir()
    assert ep.existing_artifacts(empty) == []


def test_verify_manifest_compares_real_hashes(tmp_path: Path) -> None:
    """검증기는 기록된 해시와 파일 실측 해시를 실제로 맞춘다(값이 없으면 누락으로 보고한다)."""
    run = tmp_path / "20260101-P0-R0-01"
    run.mkdir()
    metrics = run / "metrics.json"
    metrics.write_text('{"pooled": 1}', encoding="utf-8")
    payload = _payload()
    payload["artifacts"] = {
        "manifest": {"path": "manifest.json"},
        "metrics": {"path": "metrics.json", "sha256": vm.sha256_file(metrics)},
    }
    path = ep.write_manifest(run, payload)
    doc = json.loads(path.read_text(encoding="utf-8"))

    problems, counts = vm.verify_hashes(doc, run)
    assert problems == []
    assert counts["artifacts"] == {"matched": 1, "total": 1}
    assert vm.check_manifest(doc, base_dir=run) == []

    # 파일이 바뀌면 불일치로 잡는다
    metrics.write_text('{"pooled": 2}', encoding="utf-8")
    problems, counts = vm.verify_hashes(doc, run)
    assert any("해시 불일치" in problem for problem in problems)
    assert counts["artifacts"] == {"matched": 0, "total": 1}
    assert any("해시 불일치" in problem for problem in vm.check_manifest(doc, base_dir=run))

    # 기록이 없으면 통과로 넘기지 않고 누락으로 센다
    missing = json.loads(json.dumps(doc))
    missing["artifacts"]["metrics"].pop("sha256")
    problems, counts = vm.verify_hashes(missing, run)
    assert any("누락" in problem for problem in problems)
    assert counts["artifacts"] == {"matched": 0, "total": 1}

    # 파일이 없으면 실패다
    gone = json.loads(json.dumps(doc))
    gone["artifacts"]["metrics"]["path"] = "없는파일.json"
    problems, _ = vm.verify_hashes(gone, run)
    assert any("파일이 없다" in problem for problem in problems)


def test_verify_manifest_compares_data_hashes(tmp_path: Path) -> None:
    """입력 데이터 해시도 실측값과 대조한다 — 기준 디렉터리는 repo_root 로 받는다."""
    data = tmp_path / "data" / "processed"
    data.mkdir(parents=True)
    target = data / "train.csv.gz"
    target.write_text("date,boarding\n2025-01-01,1\n", encoding="utf-8")

    doc = _payload()
    doc["artifacts"] = {"manifest": {"path": "manifest.json"}}
    doc["data"]["train"]["sha256"] = vm.sha256_file(target)
    problems, counts = vm.verify_hashes(doc, tmp_path, repo_root=tmp_path)
    assert problems == []
    assert counts["data"] == {"matched": 1, "total": 1}

    doc["data"]["train"]["sha256"] = "1" * 64
    problems, counts = vm.verify_hashes(doc, tmp_path, repo_root=tmp_path)
    assert any("data.train" in problem and "불일치" in problem for problem in problems)
    assert counts["data"] == {"matched": 0, "total": 1}

    # 기록이 없는 항목(sha256 없음)은 통과가 아니라 누락이다
    doc["data"]["train"]["sha256"] = None
    problems, _ = vm.verify_hashes(doc, tmp_path, repo_root=tmp_path)
    assert any("누락" in problem for problem in problems)


def _conformal_run(tmp_path: Path, *, run_id: str = "20260101-P3-R3-01") -> Path:
    """합성 conformal 실행 디렉터리 — 산출물 목록(list) 형 manifest."""
    run = tmp_path / "20260101-P3-R3-01"
    run.mkdir()
    metrics = run / "metrics.json"
    metrics.write_text('{"gates": {}}', encoding="utf-8")
    (run / "log.jsonl").write_text('{"kind": "run_start"}\n', encoding="utf-8")
    doc = {
        "schema": vm.CONFORMAL_SCHEMA,
        "run_id": run_id,
        "protocol": ep.PROTOCOL_ID,
        "created_at": "2026-11-15T00:00:00+00:00",
        "data": {"splits": ["train", "valid"], "rows": 10, "test_accessed": False},
        "config": {"nominal": 0.8, "weighting": "time_decay", "half_life_days": 30.0},
        "cuts": [{"cut": "2026-01-01", "kind": "short", "train_days": 10, "horizon": [1, 31]}],
        "artifacts": [{
            "path": "metrics.json", "bytes": metrics.stat().st_size,
            "sha256": vm.sha256_file(metrics),
        }],
        "log": {"path": "log.jsonl"},
    }
    (run / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    return run


def test_verify_manifest_treats_list_artifacts_as_schema_mismatch(tmp_path: Path) -> None:
    """목록 형 산출물은 해시를 못 맞춘다 — 조용히 0/0 으로 통과시키지 않고 1건으로 보고한다."""
    doc = _payload()
    doc["artifacts"] = [{"path": "metrics.json", "sha256": "0" * 64}]
    problems, counts = vm.verify_hashes(doc, tmp_path)
    assert len(problems) == 1
    assert "스키마 불일치" in problems[0]
    assert counts["artifacts"] == {"matched": 0, "total": 0}


def test_verify_manifest_dispatches_conformal_schema(tmp_path: Path) -> None:
    """conformal manifest 는 그 계약으로 검증한다 — 평가 프로토콜 필드를 요구해 오진하지 않는다."""
    run = _conformal_run(tmp_path)
    assert vm.verify_run(run / "manifest.json") is True
    # 디렉터리를 인자로 줘도 schema 를 확인하고 알맞은 경로로 분기한다
    assert vm.main([str(run)]) == 0

    # 산출물이 바뀌면 실패다
    (run / "metrics.json").write_text('{"gates": 1}', encoding="utf-8")
    assert vm.verify_run(run / "manifest.json") is False


def test_conformal_manifest_reports_identity_and_log_warnings(tmp_path: Path) -> None:
    """run_id 불일치·로그 부재는 과거 기록이므로 경고로 남기고 실패로 만들지 않는다."""
    run = _conformal_run(tmp_path, run_id="20260101-P3-R3-07")
    doc = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    doc["log"]["path"] = "log-9.jsonl"
    problems, warnings = vm.check_conformal(doc, base_dir=run, dir_name=run.name)
    assert problems == []
    assert any("run_id" in warning for warning in warnings)
    assert any("log.path" in warning for warning in warnings)
    assert vm.verify_run(run / "manifest.json") is True


def test_unknown_manifest_schema_fails_honestly(tmp_path: Path) -> None:
    """모르는 schema 는 통과가 아니라 1건 실패로 보고한다."""
    run = tmp_path / "20260101-P0-R0-01"
    run.mkdir()
    (run / "manifest.json").write_text('{"schema": "other/1"}', encoding="utf-8")
    assert vm.verify_run(run / "manifest.json") is False


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


def test_profile_columns_do_not_depend_on_feature_set() -> None:
    """피처셋을 바꿔도 프로파일 열은 그대로다 — 누수 방지 계약은 피처 추가와 무관해야 한다."""
    frame = _horizon_frame()
    as_of = "2025-03-01"
    profile_cols = [
        *ep.cf.PROFILE_COLS_SH, *ep.cf.PROFILE_COLS_MONTH, *ep.cf.PROFILE_COLS_WEEKDAY
    ]
    base = ep.build_features(frame, as_of=as_of, feature_set="base")
    matched = ep.build_features(frame, as_of=as_of, feature_set="matched")
    pd.testing.assert_frame_equal(base[profile_cols], matched[profile_cols])

    horizon = ep.build_features(frame, as_of=as_of, feature_set="horizon")
    assert list(horizon[profile_cols].columns) == list(base[profile_cols].columns)
    # 지평 피처셋은 빈 월 프로파일만 대체하므로 채워진 값이 줄어들지 않는다
    before = base[profile_cols].notna().sum()
    after = horizon[profile_cols].notna().sum()
    assert all(after[col] >= before[col] for col in profile_cols)

    # 평가 구간 프로파일은 as_of 시점에 고정된 값이라, 이후 값을 고쳐도 변하지 않는다
    corrupted = frame.copy()
    late = corrupted["date"] > pd.Timestamp(as_of)
    corrupted.loc[late, "boarding"] = corrupted.loc[late, "boarding"] * 3.0
    base_bad = ep.build_features(corrupted, as_of=as_of, feature_set="base")
    matched_bad = ep.build_features(corrupted, as_of=as_of, feature_set="matched")
    eval_rows = base["date"] > pd.Timestamp(as_of)
    pd.testing.assert_frame_equal(
        base.loc[eval_rows, profile_cols], base_bad.loc[eval_rows, profile_cols]
    )
    pd.testing.assert_frame_equal(
        matched.loc[eval_rows, profile_cols], matched_bad.loc[eval_rows, profile_cols]
    )


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