"""골든셋 오프라인 재생.

녹화된 도구 출력(``fixtures``)만으로 파이프라인을 다시 돌린다. 예측기·파일·네트워크를
읽지 않으므로 ``pytest -m agent`` 는 모델 로딩 없이 끝난다.

재생은 같은 입력 → 같은 출력을 확인하는 일이다. 숫자 비교는 허용오차 0.01 을 쓴다.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

from metro_insight.agent.contracts import Envelope
from metro_insight.agent.orchestrator import run_turn

REFERENCE_DAY = "2026-11-15"
TODAY_ENV = "METRO_AGENT_TODAY"
MODE_ENV = "METRO_AGENT_MODE"
TOLERANCE = 0.01

GOLDEN_DIR = Path(__file__).resolve().parent


def fixture_key(tool: str, args: dict) -> str:
    """도구 이름 + 인자 → 녹화 키. 같은 호출이면 같은 문자열."""
    return tool + "::" + json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)


def golden_path(split: str) -> Path:
    return GOLDEN_DIR / f"golden_{split}.jsonl"


def load_golden(split: str) -> list[dict]:
    """녹화된 골든 한 묶음. 파일이 없으면 빈 목록."""
    path = golden_path(split)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _envelope_from_dict(payload: dict) -> Envelope:
    return Envelope(
        ok=bool(payload["ok"]),
        tool=str(payload["tool"]),
        version=str(payload.get("version", "1.0")),
        as_of=payload.get("as_of"),
        inputs_hash=payload.get("inputs_hash"),
        elapsed_ms=int(payload.get("elapsed_ms") or 0),
        data=dict(payload.get("data") or {}),
        caveats=list(payload.get("caveats") or []),
    )


@contextmanager
def _patched_tools(fixtures: dict):
    """``_call_tool`` 을 녹화본 조회로 바꾼다. 녹화에 없는 호출은 즉시 실패한다."""
    from metro_insight.agent import orchestrator

    original = orchestrator._call_tool

    def replay_call(name: str, args: dict) -> Envelope:
        key = fixture_key(name, args)
        if key not in fixtures:
            raise KeyError(f"녹화에 없는 도구 호출: {key}")
        return _envelope_from_dict(fixtures[key])

    orchestrator._call_tool = replay_call
    try:
        yield
    finally:
        orchestrator._call_tool = original


@contextmanager
def _environment():
    previous = {name: os.environ.get(name) for name in (TODAY_ENV, MODE_ENV)}
    os.environ[TODAY_ENV] = REFERENCE_DAY
    os.environ[MODE_ENV] = "template"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def replay(case: dict) -> dict:
    """골든 한 건을 오프라인으로 다시 실행한다."""
    with _environment(), _patched_tools(case.get("fixtures") or {}):
        return run_turn(case["utterance"])


def fact_values(bundle: dict) -> dict[str, dict]:
    """fact_id → values. 비교를 값 단위로 하기 위한 평탄화."""
    return {str(fact["fact_id"]): dict(fact.get("values") or {}) for fact in bundle.get("facts") or []}


def numbers_match(left: dict, right: dict, tolerance: float = TOLERANCE) -> bool:
    """두 값 묶음이 같은지. 숫자는 허용오차, 나머지는 같음 비교."""
    if set(left) != set(right):
        return False
    for key, value in left.items():
        other = right[key]
        if isinstance(value, (int, float)) and isinstance(other, (int, float)):
            if abs(float(value) - float(other)) > tolerance:
                return False
        elif value != other:
            return False
    return True