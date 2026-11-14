"""에이전트 계약: 봉투(Envelope)·계획(Plan)·근거 묶음(FactBundle)·판정(Verdict).

이 모듈은 **데이터 계약**만 정의한다. 신규 의존성 없이 표준 라이브러리
(dataclasses·enum·typing·json)만 쓴다. 모든 직렬화는 키 순서를 고정해
같은 내용이면 항상 같은 바이트가 나오게 한다 (재현·해시 비교용).

설계 원칙
* ``Envelope.to_dict()`` 키 순서: ok, tool, version, as_of, inputs_hash, elapsed_ms, data, caveats
* dict 안의 키는 재귀적으로 이름순 정렬한다 (중첩 dict 도 동일).
* 스키마 ``required`` 는 8키 전부다. dataclass 기본값은 편의일 뿐, 봉투에는 항상 있다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# 봉투 키 순서 (직렬화 시 이 순서를 절대 바꾸지 않는다)
ENVELOPE_KEYS = ("ok", "tool", "version", "as_of", "inputs_hash", "elapsed_ms", "data", "caveats")

# 스키마 required — 8키 전부. dataclass 에는 기본값이 있어도 계약상 봉투에는 항상 있어야 한다.
ENVELOPE_REQUIRED_KEYS = ENVELOPE_KEYS

# 호출 상한 — 한 번의 질의에서 허용하는 스텝·도구 호출·LLM 호출 수
PLAN_LIMITS = {"max_steps": 6, "max_calls": 8, "max_llm_calls": 2}

# 실행 경로별 데드라인 (ms)
DEADLINE_MS = {"template": 3000, "cloud": 6000, "local_batch": 25000}


class ErrorCode(str, Enum):  # noqa: UP042 — 계약 표기(str+Enum)를 그대로 고정한다
    """도구 실패 사유 코드. 화면·로그·평가가 같은 문자열을 쓴다."""

    E_INVALID_ARG = "E_INVALID_ARG"
    E_NOT_FOUND = "E_NOT_FOUND"
    E_OUT_OF_RANGE = "E_OUT_OF_RANGE"
    E_UNAVAILABLE = "E_UNAVAILABLE"
    E_STALE = "E_STALE"
    E_CONFLICT = "E_CONFLICT"
    E_INTERNAL = "E_INTERNAL"


class Caveat(str, Enum):  # noqa: UP042 — 계약 표기(str+Enum)를 그대로 고정한다
    """답변에 붙는 한계 표시. 숫자를 숨기지 않고 조건을 밝힌다."""

    HORIZON_PARTIAL = "HORIZON_PARTIAL"
    DE_UNAVAILABLE = "DE_UNAVAILABLE"
    DATA_STALE = "DATA_STALE"
    WEATHER_ASSUMED = "WEATHER_ASSUMED"
    AMBIGUOUS_STATION = "AMBIGUOUS_STATION"
    DATA_MISSING = "DATA_MISSING"
    OUT_OF_RANGE = "OUT_OF_RANGE"


class Intent(str, Enum):  # noqa: UP042 — 계약 표기(str+Enum)를 그대로 고정한다
    """질의 의도 12종. 계획의 ``intent`` 는 이 값 중 하나여야 한다."""

    PREDICT = "PREDICT"
    COMPARE = "COMPARE"
    HISTORY = "HISTORY"
    AGGREGATE = "AGGREGATE"
    SIMULATE = "SIMULATE"
    EXPLAIN = "EXPLAIN"
    RECONCILE = "RECONCILE"
    BACKTEST = "BACKTEST"
    DRIFT = "DRIFT"
    ALERTS = "ALERTS"
    REPORT = "REPORT"
    CLARIFY = "CLARIFY"


# 의도별 필수 슬롯. validate_plan 이 이 목록으로 누락을 잡는다.
REQUIRED_SLOTS: dict[str, tuple[str, ...]] = {
    Intent.PREDICT.value: ("line", "station", "date"),
    Intent.COMPARE.value: ("stations",),
    Intent.HISTORY.value: ("line", "station"),
    Intent.AGGREGATE.value: ("date_from", "date_to"),
    Intent.SIMULATE.value: ("line", "station", "scenario"),
    Intent.EXPLAIN.value: ("target",),
    Intent.RECONCILE.value: ("claim",),
    Intent.BACKTEST.value: ("horizon",),
    Intent.DRIFT.value: ("window",),
    Intent.ALERTS.value: ("threshold",),
    Intent.REPORT.value: ("title",),
    Intent.CLARIFY.value: ("question",),
}

# 계획 최상위 필수 키 (순서 = 직렬화 순서)
PLAN_KEYS = ("plan_id", "intent", "slots", "steps", "limits", "needs_clarification", "refusal")

# 근거 묶음 필수 키 (순서 = 직렬화 순서)
FACT_BUNDLE_KEYS = ("bundle_id", "as_of", "facts", "unavailable", "policy")


def _sorted_json(value: object) -> object:
    """dict 키를 재귀적으로 이름순 정렬한다 (리스트 안 dict 포함). 키가 문자열로 통일된다."""
    if isinstance(value, dict):
        return {str(k): _sorted_json(value[k]) for k in sorted(value, key=lambda k: str(k))}
    if isinstance(value, (list, tuple)):
        return [_sorted_json(v) for v in value]
    return value


def dumps(payload: object) -> str:
    """고정 규칙 JSON 문자열. 계약 테스트가 바이트 동일을 확인할 때 쓴다."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


@dataclass(frozen=True)
class Envelope:
    """도구 한 번 호출의 결과 봉투. 성공·실패가 같은 모양을 쓴다."""

    ok: bool
    tool: str
    version: str = "1.0"
    as_of: str | None = None
    inputs_hash: str | None = None
    elapsed_ms: int = 0
    data: dict = field(default_factory=dict)
    caveats: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """키 순서를 고정한 dict. 같은 내용이면 항상 같은 JSON 바이트가 나온다."""
        return {
            "ok": self.ok,
            "tool": self.tool,
            "version": self.version,
            "as_of": self.as_of,
            "inputs_hash": self.inputs_hash,
            "elapsed_ms": self.elapsed_ms,
            "data": _sorted_json(self.data),
            "caveats": _sorted_json(self.caveats),
        }

    @classmethod
    def ok_(
        cls,
        tool: str,
        data: dict,
        *,
        as_of: str | None = None,
        inputs_hash: str | None = None,
        elapsed_ms: int = 0,
        caveats: tuple[dict, ...] | list[dict] = (),
    ) -> Envelope:
        """성공 봉투. ``ok=True`` 고정."""
        return cls(
            ok=True,
            tool=str(tool),
            as_of=as_of,
            inputs_hash=inputs_hash,
            elapsed_ms=int(elapsed_ms),
            data=dict(data),
            caveats=list(caveats),
        )

    @classmethod
    def error(
        cls,
        tool: str,
        code: ErrorCode,
        human: str,
        *,
        elapsed_ms: int = 0,
    ) -> Envelope:
        """실패 봉투. ``data`` 는 {code, human} 두 키만 갖는다."""
        return cls(
            ok=False,
            tool=str(tool),
            elapsed_ms=int(elapsed_ms),
            data={"code": ErrorCode(code).value, "human": str(human)},
            caveats=[],
        )


@dataclass(frozen=True)
class Plan:
    """질의를 도구 호출 순서로 푼 계획. ``steps`` 는 {step_id, tool, args, depends_on}."""

    plan_id: str
    intent: str
    slots: dict = field(default_factory=dict)
    steps: list[dict] = field(default_factory=list)
    limits: dict = field(default_factory=lambda: dict(PLAN_LIMITS))
    needs_clarification: bool = False
    refusal: str | None = None

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "intent": self.intent,
            "slots": _sorted_json(self.slots),
            "steps": _sorted_json(self.steps),
            "limits": _sorted_json(self.limits),
            "needs_clarification": self.needs_clarification,
            "refusal": self.refusal,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Plan:
        """``to_dict`` 결과를 되돌린다. 빠진 키는 기본값으로 채운다."""
        source = payload if isinstance(payload, dict) else {}
        return cls(
            plan_id=str(source.get("plan_id", "")),
            intent=str(source.get("intent", "")),
            slots=dict(source.get("slots") or {}),
            steps=[dict(s) for s in (source.get("steps") or [])],
            limits=dict(source.get("limits") or PLAN_LIMITS),
            needs_clarification=bool(source.get("needs_clarification", False)),
            refusal=source.get("refusal"),
        )


@dataclass(frozen=True)
class FactBundle:
    """계획이 모은 사실과 '없는 것'을 함께 담는다. 없는 것도 정직하게 남긴다."""

    bundle_id: str
    as_of: str | None = None
    facts: list[dict] = field(default_factory=list)
    unavailable: list[dict] = field(default_factory=list)
    policy: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "as_of": self.as_of,
            "facts": _sorted_json(self.facts),
            "unavailable": _sorted_json(self.unavailable),
            "policy": _sorted_json(self.policy),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> FactBundle:
        source = payload if isinstance(payload, dict) else {}
        return cls(
            bundle_id=str(source.get("bundle_id", "")),
            as_of=source.get("as_of"),
            facts=[dict(f) for f in (source.get("facts") or [])],
            unavailable=[dict(u) for u in (source.get("unavailable") or [])],
            policy=dict(source.get("policy") or {}),
        )


@dataclass(frozen=True)
class DraftAnswer:
    """근거 묶음으로 만든 초안. 문장이 아니라 주장(claim) 목록이다."""

    answer_id: str
    claims: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer_id": self.answer_id,
            "claims": _sorted_json(self.claims),
            "caveats": [str(c) for c in self.caveats],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> DraftAnswer:
        source = payload if isinstance(payload, dict) else {}
        return cls(
            answer_id=str(source.get("answer_id", "")),
            claims=[dict(c) for c in (source.get("claims") or [])],
            caveats=[str(c) for c in (source.get("caveats") or [])],
        )


@dataclass(frozen=True)
class Verdict:
    """초안 검사 결과. pass/repair/reject 셋 중 하나."""

    verdict: Literal["pass", "repair", "reject"]
    violations: list[dict] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    rendered_answer: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "violations": _sorted_json(self.violations),
            "checks": _sorted_json(self.checks),
            "rendered_answer": self.rendered_answer,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Verdict:
        source = payload if isinstance(payload, dict) else {}
        verdict = str(source.get("verdict", "reject"))
        if verdict not in ("pass", "repair", "reject"):
            verdict = "reject"
        return cls(
            verdict=verdict,
            violations=[dict(v) for v in (source.get("violations") or [])],
            checks=dict(source.get("checks") or {}),
            rendered_answer=str(source.get("rendered_answer", "")),
        )


# 수동 작성 JSON Schema (jsonschema 패키지 금지). 계약 변경 시 SCHEMA_HASH 가 바뀐다.
JSON_SCHEMA: dict = {
    "envelope": {
        "type": "object",
        "required": list(ENVELOPE_REQUIRED_KEYS),
        "additionalProperties": False,
        "properties": {
            "ok": {"type": "boolean"},
            "tool": {"type": "string", "minLength": 1},
            "version": {"type": "string", "pattern": r"^\d+\.\d+$"},
            "as_of": {"type": ["string", "null"]},
            "inputs_hash": {"type": ["string", "null"]},
            "elapsed_ms": {"type": "integer", "minimum": 0},
            "data": {"type": "object"},
            "caveats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["code", "human"],
                    "properties": {
                        "code": {"type": "string"},
                        "human": {"type": "string"},
                    },
                },
            },
        },
    },
    "plan": {
        "type": "object",
        "required": list(PLAN_KEYS),
        "additionalProperties": False,
        "properties": {
            "plan_id": {"type": "string", "minLength": 1},
            "intent": {"type": "string", "enum": [i.value for i in Intent]},
            "slots": {"type": "object"},
            "steps": {
                "type": "array",
                "maxItems": PLAN_LIMITS["max_steps"],
                "items": {
                    "type": "object",
                    "required": ["step_id", "tool", "args"],
                    "additionalProperties": False,
                    "properties": {
                        "step_id": {"type": "string", "minLength": 1},
                        "tool": {"type": "string", "minLength": 1},
                        "args": {"type": "object"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "limits": {
                "type": "object",
                "properties": {
                    "max_steps": {"type": "integer", "maximum": PLAN_LIMITS["max_steps"]},
                    "max_calls": {"type": "integer", "maximum": PLAN_LIMITS["max_calls"]},
                    "max_llm_calls": {"type": "integer", "maximum": PLAN_LIMITS["max_llm_calls"]},
                },
            },
            "needs_clarification": {"type": "boolean"},
            "refusal": {"type": ["string", "null"]},
        },
    },
    "fact_bundle": {
        "type": "object",
        "required": list(FACT_BUNDLE_KEYS),
        "additionalProperties": False,
        "properties": {
            "bundle_id": {"type": "string", "minLength": 1},
            "as_of": {"type": ["string", "null"]},
            "facts": {"type": "array", "items": {"type": "object"}},
            "unavailable": {"type": "array", "items": {"type": "object"}},
            "policy": {"type": "object"},
        },
    },
    "draft_answer": {
        "type": "object",
        "required": ["answer_id", "claims", "caveats"],
        "additionalProperties": False,
        "properties": {
            "answer_id": {"type": "string", "minLength": 1},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text"],
                    "properties": {
                        "text": {"type": "string"},
                        "fact_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "caveats": {"type": "array", "items": {"type": "string"}},
        },
    },
    "verdict": {
        "type": "object",
        "required": ["verdict", "violations", "checks", "rendered_answer"],
        "additionalProperties": False,
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "repair", "reject"]},
            "violations": {"type": "array", "items": {"type": "object"}},
            "checks": {"type": "object"},
            "rendered_answer": {"type": "string"},
        },
    },
}


def _allowed_tools() -> frozenset[str]:
    """도구 허용목록. contracts → tools 순환 import 를 피하려고 함수 안에서 가져온다."""
    from metro_insight.agent.tools import ALLOWLIST

    return frozenset(ALLOWLIST)


def _find_cycle(graph: dict[str, list[str]]) -> list[str]:
    """의존 그래프의 순환 경로 하나를 돌려준다. 없으면 빈 리스트."""
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for dep in graph.get(node, ()):
            if dep not in graph:
                continue
            if state.get(dep) == 1:
                return stack[stack.index(dep):] + [dep]
            if state.get(dep, 0) == 0:
                found = visit(dep)
                if found:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for node in graph:
        if state.get(node, 0) == 0:
            found = visit(node)
            if found:
                return found
    return []


def validate_plan(payload: dict) -> list[str]:
    """계획 payload 를 수동 검증한다. 오류 문자열 목록을 돌려주고, 빈 리스트면 통과."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["plan: object expected"]

    missing = [k for k in PLAN_KEYS if k not in payload]
    if missing:
        return [f"plan: missing '{k}'" for k in missing]

    plan_id = payload.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id.strip():
        errors.append("plan.plan_id: non-empty string expected")

    intent = payload.get("intent")
    intents = {i.value for i in Intent}
    if not isinstance(intent, str) or intent not in intents:
        errors.append(f"plan.intent: unknown intent '{intent}'")

    slots = payload.get("slots")
    if not isinstance(slots, dict):
        errors.append("plan.slots: object expected")
        slots = {}

    steps = payload.get("steps")
    if not isinstance(steps, list):
        errors.append("plan.steps: array expected")
        steps = []

    limits = payload.get("limits")
    if not isinstance(limits, dict):
        errors.append("plan.limits: object expected")
        limits = {}

    if not isinstance(payload.get("needs_clarification"), bool):
        errors.append("plan.needs_clarification: boolean expected")

    refusal = payload.get("refusal")
    if refusal is not None and not isinstance(refusal, str):
        errors.append("plan.refusal: string or null expected")

    if isinstance(intent, str) and intent in intents:
        for key in REQUIRED_SLOTS.get(intent, ()):
            if key not in slots:
                errors.append(f"plan.slots: missing '{key}' for {intent}")

    max_steps = PLAN_LIMITS["max_steps"]
    if len(steps) > max_steps:
        errors.append(f"plan.steps: {len(steps)} steps exceeds max_steps={max_steps}")

    for key, ceiling in PLAN_LIMITS.items():
        value = limits.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"plan.limits.{key}: integer expected")
        elif value > ceiling:
            errors.append(f"plan.limits.{key}: {value} exceeds {ceiling}")

    allowed = _allowed_tools()
    graph: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"plan.steps[{index}]: object expected")
            continue
        step_id = step.get("step_id")
        if not isinstance(step_id, str) or not step_id.strip():
            errors.append(f"plan.steps[{index}].step_id: non-empty string expected")
            step_id = f"#{index}"
        elif step_id in seen_ids:
            errors.append(f"plan.steps[{index}].step_id: duplicate '{step_id}'")
        seen_ids.add(step_id)

        tool = step.get("tool")
        if not isinstance(tool, str) or tool not in allowed:
            errors.append(f"plan.steps[{index}].tool: unknown tool '{tool}'")

        args = step.get("args", {})
        if not isinstance(args, dict):
            errors.append(f"plan.steps[{index}].args: object expected")

        deps = step.get("depends_on", [])
        if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
            errors.append(f"plan.steps[{index}].depends_on: array of strings expected")
            deps = []
        graph[step_id] = list(deps)

    unknown = sorted({d for deps in graph.values() for d in deps if d not in graph})
    for dep in unknown:
        errors.append(f"plan.steps: unknown dependency '{dep}'")

    cycle = _find_cycle(graph)
    if cycle:
        errors.append(f"plan.steps: dependency cycle {' -> '.join(cycle)}")

    return errors