"""DataGuard 계약 테스트 — 승인 게이트 · 접근 로그 · 해시 변조 감지.

실제 data/processed/*.csv.gz 는 읽지 않는다. tmp_path 에 같은 상대경로로 합성 데이터를 만들어
가드 로직만 검증한다(Test 접근 금지 규칙을 테스트가 스스로 지킨다).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from metro_insight.data import guard as guard_mod
from metro_insight.data.guard import (
    ACCESS_LOG,
    APPROVAL_ENV,
    TAMPER_TARGETS,
    ArtifactTampered,
    DataGuard,
)

Denied = guard_mod.TestAccessDenied  # 클래스 이름이 Test* 라 pytest 수집 경고가 나므로 별칭을 쓴다

TRAIN_ROWS = 365 * 2 * 2  # 2025-01-01~2025-12-31 × 역 2 × 시간 2
VALID_ROWS = 45 * 2 * 2
TEST_DAYS = 136  # 2026-02-15 ~ 2026-06-30


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    index = pd.MultiIndex.from_product(
        [dates, [1, 2], [0, 1]], names=["date", "station_code", "hour_code"]
    )
    frame = pd.DataFrame(index=index).reset_index()
    frame["line_code"] = 1
    frame["weekday"] = frame["date"].dt.weekday.astype("int8")
    frame["is_weekend"] = (frame["weekday"] >= 5).astype("int8")
    frame["is_holiday"] = 0
    frame["boarding"] = 100 + frame["hour_code"] * 10
    return frame[
        ["date", "station_code", "line_code", "hour_code", "weekday",
         "is_weekend", "is_holiday", "boarding"]
    ]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    _frame("2025-01-01", "2025-12-31").to_csv(
        processed / "train.csv.gz", index=False, compression="gzip"
    )
    _frame("2026-01-01", "2026-02-14").to_csv(
        processed / "valid.csv.gz", index=False, compression="gzip"
    )
    _frame("2026-02-15", "2026-06-30").to_csv(
        processed / "test.csv.gz", index=False, compression="gzip"
    )
    (tmp_path / "models" / "metrics_calendar.json").write_text("{}", encoding="utf-8")
    (tmp_path / "models" / "metrics_sequence.json").write_text("{}", encoding="utf-8")
    return tmp_path


def _frozen(root: Path) -> dict[str, str]:
    return {rel: _sha256(root / rel) for rel in TAMPER_TARGETS}


def _approved(root: Path) -> DataGuard:
    return DataGuard(root, lane="test", approval_id="APP-TEST-1", allow_test=True, frozen={})


# ───────────────────────── 승인 게이트 ─────────────────────────
def test_train_opens_without_approval(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(APPROVAL_ENV, raising=False)
    guard = DataGuard(root, lane="test", frozen={})
    frame = guard.open_split("train", purpose="단위 테스트", columns=["date", "boarding"])
    assert len(frame) == TRAIN_ROWS
    assert list(frame.columns) == ["date", "boarding"]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_test_dev_denied_without_approval(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(APPROVAL_ENV, raising=False)
    guard = DataGuard(root, lane="test", approval_id="APP-TEST-1", allow_test=True, frozen={})
    with pytest.raises(Denied):
        guard.open_split("test_dev", purpose="승인 없이 열면 안 된다")
    # 거부는 파일을 열기 전에 끝난다 — 접근 로그도 남기지 않는다
    assert not (root / ACCESS_LOG).exists()


def test_env_var_alone_is_not_enough(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(APPROVAL_ENV, "1")
    # approval_id 없음
    with pytest.raises(Denied):
        DataGuard(root, lane="test", allow_test=True, frozen={}).open_split("test_dev", purpose="환경변수만")
    # allow_test 없음
    with pytest.raises(Denied):
        DataGuard(root, lane="test", approval_id="APP-TEST-1", frozen={}).open_split(
            "test_final", purpose="환경변수만")


def test_approved_access_passes_with_small_slice(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(APPROVAL_ENV, "1")
    guard = _approved(root)
    dev = guard.open_split("test_dev", purpose="승인 확인", columns=["date", "boarding"])
    final = guard.open_split("test_final", purpose="승인 확인", columns=["date", "boarding"])

    assert dev["date"].min() == pd.Timestamp("2026-02-15")
    assert dev["date"].max() == pd.Timestamp("2026-04-30")
    assert final["date"].min() == pd.Timestamp("2026-05-01")
    assert final["date"].max() == pd.Timestamp("2026-06-30")
    # 두 조각을 합치면 test 전체(136일 × 역 2 × 시간 2)와 같다 — 겹침·누락 없음
    assert len(dev) + len(final) == TEST_DAYS * 2 * 2


def test_access_log_appends_one_json_line(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(APPROVAL_ENV, raising=False)
    guard = DataGuard(root, lane="laptop-test", frozen={})
    guard.open_split("train", purpose="로그 1")
    guard.open_split("valid", purpose="로그 2")

    path = root / ACCESS_LOG
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[1])
    assert record["split"] == "valid"
    assert record["rows"] == VALID_ROWS
    assert record["purpose"] == "로그 2"
    assert record["lane"] == "laptop-test"
    assert record["approval_id"] is None
    assert record["ts"]

    guard.open_split("train", purpose="로그 3")
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 3


# ───────────────────────── 해시 ─────────────────────────
def test_tamper_is_detected(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(APPROVAL_ENV, raising=False)
    frozen = _frozen(root)
    guard = DataGuard(root, lane="test", frozen=frozen)
    guard.assert_unchanged()  # 동결 직후에는 통과

    valid = root / "data" / "processed" / "valid.csv.gz"
    valid.write_bytes(valid.read_bytes() + b"tampered")
    with pytest.raises(ArtifactTampered):
        guard.assert_unchanged()


def test_tamper_after_open_is_detected(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(APPROVAL_ENV, raising=False)
    guard = DataGuard(root, lane="test", frozen={})  # 동결표가 비어도 연 원본은 기억한다
    guard.open_split("train", purpose="변조 감지")
    train = root / "data" / "processed" / "train.csv.gz"
    train.write_bytes(train.read_bytes() + b"x")
    with pytest.raises(ArtifactTampered):
        guard.assert_unchanged()


def test_missing_frozen_file_is_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(APPROVAL_ENV, raising=False)
    guard = DataGuard(tmp_path, lane="test", frozen={"data/processed/valid.csv.gz": "0" * 64})
    with pytest.raises(ArtifactTampered):
        guard.assert_unchanged()


def test_freeze_requires_test_gate(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(APPROVAL_ENV, raising=False)
    with pytest.raises(Denied):
        DataGuard(root, lane="test", frozen={}).freeze()

    monkeypatch.setenv(APPROVAL_ENV, "1")
    hashes = _approved(root).freeze()
    assert set(hashes) == set(TAMPER_TARGETS)
    assert all(len(value) == 64 for value in hashes.values())
    log = (root / ACCESS_LOG).read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(line)["split"] == "test_dev+test_final" for line in log)


def test_module_frozen_covers_four_targets() -> None:
    assert set(guard_mod.FROZEN) == set(TAMPER_TARGETS)
    assert set(guard_mod.SPLIT_FILES) == {"train", "valid", "test_dev", "test_final"}
    assert guard_mod.TEST_WINDOWS["test_dev"] == ("2026-02-15", "2026-04-30")
    assert guard_mod.TEST_WINDOWS["test_final"] == ("2026-05-01", "2026-06-30")
