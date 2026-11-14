"""평가 데이터 접근 통제 — Test 는 2중 승인 게이트, 모든 접근은 append-only 로그.

접근 규칙
    * ``train`` · ``valid`` 는 승인 없이 열 수 있다.
    * ``test_dev`` · ``test_final`` 은 세 조건이 **모두** 있어야 열린다.
          1) ``DataGuard(..., allow_test=True)``
          2) ``approval_id`` 가 비어 있지 않은 문자열
          3) 환경변수 ``ALLOW_TEST_EVAL=1``
      하나라도 빠지면 :class:`TestAccessDenied`.
    * ``test_dev``  = test.csv.gz 의 2026-02-15 ~ 2026-04-30
      ``test_final`` = test.csv.gz 의 2026-05-01 ~ 2026-06-30
    * ``open_split`` 호출은 모두 ``models/_exp/access.jsonl`` 에 1줄 JSON 으로 남는다(append 전용).
      시각은 UTC ISO, 파일은 지우거나 덮어쓰지 않는다.
    * :meth:`DataGuard.assert_unchanged` 는 동결 해시(FROZEN 4개 + 이번에 연 원본)를 다시 계산해
      변조·교체를 잡는다. 불일치는 :class:`ArtifactTampered`.

FROZEN 을 guard.py 리터럴로 두는 이유
    ``models/_exp/*`` 는 .gitignore 대상이다. frozen.json 을 정본으로 두면 clone 마다 사라져
    "동결"의 뜻이 없어진다. 코드 리터럴은 diff 로 리뷰되고, import 시 I/O 가 없고, 값이 바뀌면 바로 보인다.
    최초 1회 계산은 :meth:`DataGuard.freeze` 로 하고(승인 경로에서 실행), 결과를 이 리터럴에 붙인다.

의존성: hashlib · json · pandas 표준 조합만 쓴다(신규 패키지 없음).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

SplitName = Literal["train", "valid", "test_dev", "test_final"]

# 상대경로 → sha256. P0 에서 1회 계산해 동결했다(2026-11-14, approval APPROVAL-2026-11-14-P0-legacy-B).
FROZEN: dict[str, str] = {
    "data/processed/test.csv.gz": "be4407c785630c21b1bdb7d6696d9a5f7dc4f15cf059d4e8adc9ea00f85a1132",
    "data/processed/valid.csv.gz": "25a2bfd9fc0a0eb0c26bd0f990b9f71a71235f5114d580d8ae2ceea41c805213",
    "models/metrics_sequence.json": "e9ac27b053d7e3578d5212d6b1bc485af78e83da970f75907e462065960e8f05",
    "models/metrics_calendar.json": "4da5a76e9c7c1b85c6cd7caea99bfee9f80546c6bcf338d651b5ba3da7001250",
}

# 동결 대상 4개 (FROZEN 키와 같아야 한다)
TAMPER_TARGETS: tuple[str, ...] = (
    "data/processed/test.csv.gz",
    "data/processed/valid.csv.gz",
    "models/metrics_sequence.json",
    "models/metrics_calendar.json",
)

SPLIT_FILES: dict[str, str] = {
    "train": "data/processed/train.csv.gz",
    "valid": "data/processed/valid.csv.gz",
    "test_dev": "data/processed/test.csv.gz",
    "test_final": "data/processed/test.csv.gz",
}

# test 는 한 파일을 기간으로 나눠 쓴다
TEST_WINDOWS: dict[str, tuple[str, str]] = {
    "test_dev": ("2026-02-15", "2026-04-30"),
    "test_final": ("2026-05-01", "2026-06-30"),
}

APPROVAL_ENV = "ALLOW_TEST_EVAL"
ACCESS_LOG = "models/_exp/access.jsonl"


class TestAccessDenied(RuntimeError):
    """승인 게이트(allow_test · approval_id · ALLOW_TEST_EVAL)를 통과하지 못했다."""


class ArtifactTampered(RuntimeError):
    """동결 해시와 실제 파일이 다르다 — 데이터·성적표가 바뀌었다."""


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DataGuard:
    """평가 데이터 접근 지점. 모든 split 읽기는 이 클래스를 통과한다."""

    def __init__(
        self,
        root: Path,
        *,
        lane: str = "unknown",
        approval_id: str | None = None,
        allow_test: bool = False,
        frozen: Mapping[str, str] | None = None,
    ) -> None:
        """frozen 을 주면 모듈 FROZEN 대신 그 표를 쓴다(합성 데이터 테스트용)."""
        self.root = Path(root)
        self.lane = str(lane)
        self.approval_id = approval_id
        self.allow_test = bool(allow_test)
        self.frozen: dict[str, str] = dict(FROZEN if frozen is None else frozen)
        self._opened: dict[str, str] = {}

    # ───────────────────────── 게이트 ─────────────────────────
    def _check_test_gate(self, name: str) -> None:
        missing = []
        if not self.allow_test:
            missing.append("allow_test=True")
        if not (self.approval_id or "").strip():
            missing.append("approval_id")
        if os.environ.get(APPROVAL_ENV) != "1":
            missing.append(f"{APPROVAL_ENV}=1")
        if missing:
            raise TestAccessDenied(
                f"{name} 은 승인 경로 전용이다 — 빠진 조건: {', '.join(missing)}"
            )

    # ───────────────────────── 읽기 ─────────────────────────
    def open_split(
        self,
        name: SplitName,
        *,
        purpose: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """split 을 읽는다. test 계열은 승인 게이트를 통과해야 하고, 모든 호출은 로그에 남는다.

        columns 를 주면 그 컬럼만 돌려준다(내부적으로 date 는 기간 필터에 필요해 함께 읽는다).
        """
        if name not in SPLIT_FILES:
            raise KeyError(f"알 수 없는 split: {name} (가능: {', '.join(SPLIT_FILES)})")
        if not str(purpose).strip():
            raise ValueError("purpose 는 비울 수 없다 — 접근 로그의 감사 근거다")
        if name in TEST_WINDOWS:
            self._check_test_gate(name)

        rel = SPLIT_FILES[name]
        path = self.root / rel
        if not path.exists():
            raise FileNotFoundError(path)

        digest = _sha256(path)
        expected = self.frozen.get(rel)
        if expected and expected != digest:
            raise ArtifactTampered(
                f"{rel} 이 동결 해시와 다르다 — 기대 {expected[:12]}… 실제 {digest[:12]}…"
            )

        read_cols = None
        if columns is not None:
            read_cols = list(dict.fromkeys([*columns, "date"]))
        frame = pd.read_csv(path, usecols=read_cols, compression="gzip")
        frame["date"] = pd.to_datetime(frame["date"])

        if name in TEST_WINDOWS:
            lo, hi = TEST_WINDOWS[name]
            keep = (frame["date"] >= pd.Timestamp(lo)) & (frame["date"] <= pd.Timestamp(hi))
            frame = frame.loc[keep]
        if columns is not None:
            frame = frame.loc[:, list(columns)]
        frame = frame.reset_index(drop=True)

        self._opened[rel] = digest
        self.log_access(name, len(frame), purpose)
        return frame

    def log_access(self, name: str, rows: int, purpose: str) -> None:
        """``models/_exp/access.jsonl`` 에 1줄 append. 덮어쓰지 않는다."""
        record = {
            "ts": _now(),
            "lane": self.lane,
            "split": str(name),
            "rows": int(rows),
            "purpose": str(purpose),
            "approval_id": self.approval_id,
        }
        path = self.root / ACCESS_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ───────────────────────── 무결성 ─────────────────────────
    def assert_unchanged(self) -> None:
        """동결 4개 + 이번에 연 원본의 sha256 을 다시 계산해 비교한다."""
        for rel, expected in self.frozen.items():
            if not expected:
                continue  # 아직 동결 전(빈 값)은 건너뛴다
            path = self.root / rel
            if not path.exists():
                raise ArtifactTampered(f"{rel} 가 없다 — 동결 시점 이후 삭제됐다")
            actual = _sha256(path)
            if actual != expected:
                raise ArtifactTampered(
                    f"{rel} sha256 불일치 — 기대 {expected[:12]}… 실제 {actual[:12]}…"
                )
        for rel, opened_at in self._opened.items():
            actual = _sha256(self.root / rel)
            if actual != opened_at:
                raise ArtifactTampered(f"{rel} 이 이번 실행 중에 바뀌었다")

    def freeze(self, targets: tuple[str, ...] | None = None) -> dict[str, str]:
        """P0 최초 1회용 — 대상 파일의 sha256 을 계산해 돌려준다(파일을 쓰지 않는다).

        test.csv.gz 는 바이트 해시만 계산하지만 승인 게이트는 그대로 요구하고, 그 사실도 로그에 남긴다.
        """
        out: dict[str, str] = {}
        for rel in TAMPER_TARGETS if targets is None else targets:
            if rel == SPLIT_FILES["test_dev"]:
                self._check_test_gate("test_dev")
                self.log_access("test_dev+test_final", 0, "P0 해시 동결 — 행은 읽지 않고 바이트 해시만 계산")
            path = self.root / rel
            if not path.exists():
                raise FileNotFoundError(path)
            out[rel] = _sha256(path)
        return out