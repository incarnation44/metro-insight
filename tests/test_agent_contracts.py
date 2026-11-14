"""에이전트 계약 동결 테스트: 봉투·계획·스키마·도구 허용목록.

계약이 바뀌면 이 파일이 먼저 실패해야 한다. 특히 ``SCHEMA_HASH`` 는 계약의
지문이다 — 의도적으로 바꿀 때만 상수를 갱신한다.
"""
import dataclasses
import hashlib
import inspect
import json

import pytest

from metro_insight.agent import tools
from metro_insight.agent.contracts import (
    DEADLINE_MS,
    ENVELOPE_KEYS,
    ENVELOPE_REQUIRED_KEYS,
    JSON_SCHEMA,
    PLAN_KEYS,
    PLAN_LIMITS,
    Caveat,
    DraftAnswer,
    Envelope,
    ErrorCode,
    FactBundle,
    Intent,
    Plan,
    Verdict,
    validate_plan,
)

# JSON_SCHEMA 의 지문 (sort_keys + 고정 구분자). 계약을 바꾸면 이 값도 바뀐다.
SCHEMA_HASH = "be92bd7a0d95467d383279c59da8505438f40ac3c92a5a8b0e947f6235f093f7"


def _valid_plan_payload() -> dict:
    return {
        "plan_id": "p1",
        "intent": Intent.PREDICT.value,
        "slots": {"line": "2호선", "station": "잠실", "date": "2026-11-14"},
        "steps": [
            {"step_id": "s1", "tool": "resolve_station", "args": {"name": "잠실"}},
            {
                "step_id": "s2",
                "tool": "predict",
                "args": {"line": "2호선", "station": "잠실", "date": "2026-11-14", "hour": 8},
                "depends_on": ["s1"],
            },
        ],
        "limits": dict(PLAN_LIMITS),
        "needs_clarification": False,
        "refusal": None,
    }


# ---------- Envelope ----------


def test_envelope_required_fields_exist():
    env = Envelope.ok_("predict", {"value": 1})
    payload = env.to_dict()
    for key in ENVELOPE_REQUIRED_KEYS:
        assert key in payload
    assert len(ENVELOPE_REQUIRED_KEYS) == 8
    assert set(payload) == set(ENVELOPE_KEYS)
    # 기본값: version 1.0, 메타 없음, 경과 0, 한계 없음
    assert env.ok is True
    assert env.version == "1.0"
    assert env.as_of is None
    assert env.inputs_hash is None
    assert env.elapsed_ms == 0
    assert env.caveats == []


def test_envelope_key_order_is_frozen():
    payload = Envelope.ok_("predict", {"b": 1, "a": 2}).to_dict()
    assert tuple(payload) == ENVELOPE_KEYS
    assert ENVELOPE_KEYS == (
        "ok",
        "tool",
        "version",
        "as_of",
        "inputs_hash",
        "elapsed_ms",
        "data",
        "caveats",
    )


def test_envelope_serialization_byte_identical():
    first = Envelope.ok_(
        "aggregate",
        {"value": 12.5, "n": 4, "coverage": 0.5, "nested": {"z": 1, "a": [{"y": 2, "b": 3}]}},
        as_of="2026-11-14",
        inputs_hash="abc",
        elapsed_ms=7,
        caveats=[{"code": Caveat.DATA_STALE.value, "human": "3일 지남"}],
    )
    second = Envelope.ok_(
        "aggregate",
        {"value": 12.5, "n": 4, "coverage": 0.5, "nested": {"z": 1, "a": [{"y": 2, "b": 3}]}},
        as_of="2026-11-14",
        inputs_hash="abc",
        elapsed_ms=7,
        caveats=[{"code": Caveat.DATA_STALE.value, "human": "3일 지남"}],
    )
    dump = json.dumps(first.to_dict(), ensure_ascii=False, sort_keys=False, separators=(",", ":"))
    other = json.dumps(second.to_dict(), ensure_ascii=False, sort_keys=False, separators=(",", ":"))
    assert dump == other
    assert '"data":{"coverage":0.5,"n":4,"nested":{"a":[{"b":3,"y":2}],"z":1},"value":12.5}' in dump


def test_envelope_ok_defaults():
    env = Envelope.ok_("history", {"points": []})
    assert env.ok is True
    assert env.version == "1.0"
    assert env.as_of is None
    assert env.inputs_hash is None
    assert env.elapsed_ms == 0
    assert env.caveats == []


def test_every_error_code_serializes_and_roundtrips():
    assert len(ErrorCode) == 7
    for code in ErrorCode:
        env = Envelope.error("predict", code, "실패 사유")
        payload = env.to_dict()
        assert payload["ok"] is False
        assert payload["tool"] == "predict"
        assert set(payload["data"]) == {"code", "human"}
        assert payload["data"]["code"] == code.value
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
        assert ErrorCode(payload["data"]["code"]) is code
    assert {c.value for c in ErrorCode} == {
        "E_INVALID_ARG",
        "E_NOT_FOUND",
        "E_OUT_OF_RANGE",
        "E_UNAVAILABLE",
        "E_STALE",
        "E_CONFLICT",
        "E_INTERNAL",
    }
    assert len(Caveat) == 7
    assert len(Intent) == 12


# ---------- Plan / FactBundle / DraftAnswer / Verdict ----------


def _plan() -> Plan:
    return Plan(
        plan_id="p1",
        intent=Intent.PREDICT.value,
        slots={"station": "잠실", "line": "2호선", "date": "2026-11-14"},
        steps=[
            {"step_id": "s1", "tool": "resolve_station", "args": {"name": "잠실"}},
            {
                "step_id": "s2",
                "tool": "predict",
                "args": {"hour": 8, "date": "2026-11-14"},
                "depends_on": ["s1"],
            },
        ],
        limits=dict(PLAN_LIMITS),
        needs_clarification=False,
        refusal=None,
    )


def test_plan_roundtrip():
    plan = _plan()
    payload = plan.to_dict()
    assert tuple(payload) == PLAN_KEYS
    restored = Plan.from_dict(payload)
    assert restored == plan
    assert restored.to_dict() == payload


def test_fact_bundle_roundtrip():
    bundle = FactBundle(
        bundle_id="b1",
        as_of="2026-11-14",
        facts=[{"fact_id": "f1", "value": 12.5, "source": "history"}],
        unavailable=[{"slot": "weather", "reason": "미수집"}],
        policy={"honesty": "no_invention"},
    )
    payload = bundle.to_dict()
    assert tuple(payload) == ("bundle_id", "as_of", "facts", "unavailable", "policy")
    assert FactBundle.from_dict(payload) == bundle
    assert FactBundle.from_dict(payload).to_dict() == payload


def test_draft_answer_and_verdict_roundtrip():
    answer = DraftAnswer(
        answer_id="a1",
        claims=[{"text": "8시 잠실 1.2만명", "fact_ids": ["f1"]}],
        caveats=[Caveat.HORIZON_PARTIAL.value],
    )
    assert DraftAnswer.from_dict(answer.to_dict()) == answer

    verdict = Verdict(
        verdict="repair",
        violations=[{"kind": "missing_fact_id", "claim": 0}],
        checks={"fact_coverage": 0.5},
        rendered_answer="확인 필요",
    )
    assert Verdict.from_dict(verdict.to_dict()) == verdict


# ---------- JSON_SCHEMA ----------


def test_json_schema_keys_match_contract():
    assert set(JSON_SCHEMA) == {"envelope", "plan", "fact_bundle", "draft_answer", "verdict"}
    for name, schema in JSON_SCHEMA.items():
        assert schema["type"] == "object", name
        assert isinstance(schema["properties"], dict), name
        assert isinstance(schema["required"], list), name
        assert set(schema["required"]) <= set(schema["properties"]), name


def test_json_schema_required_fields():
    assert tuple(JSON_SCHEMA["envelope"]["required"]) == ENVELOPE_REQUIRED_KEYS
    assert tuple(JSON_SCHEMA["plan"]["required"]) == PLAN_KEYS
    assert JSON_SCHEMA["plan"]["properties"]["intent"]["enum"] == [i.value for i in Intent]


def test_envelope_schema_requires_all_eight():
    required = JSON_SCHEMA["envelope"]["required"]
    assert len(required) == 8
    assert tuple(required) == ENVELOPE_KEYS
    assert set(required) == set(ENVELOPE_KEYS)


def test_envelope_schema_required_covers_every_dataclass_field():
    fields = tuple(f.name for f in dataclasses.fields(Envelope))
    assert fields == ENVELOPE_KEYS
    assert set(JSON_SCHEMA["envelope"]["required"]) == set(fields)


def test_schema_hash_is_frozen():
    canonical = json.dumps(
        JSON_SCHEMA, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == SCHEMA_HASH


# ---------- validate_plan ----------


def test_validate_plan_accepts_valid_payload():
    assert validate_plan(_valid_plan_payload()) == []


def test_validate_plan_rejects_missing_required_key():
    payload = _valid_plan_payload()
    payload.pop("limits")
    errors = validate_plan(payload)
    assert any("limits" in e for e in errors)


def test_validate_plan_rejects_unknown_intent():
    payload = _valid_plan_payload()
    payload["intent"] = "FLY"
    assert any("intent" in e for e in validate_plan(payload))


def test_validate_plan_rejects_missing_slot():
    payload = _valid_plan_payload()
    payload["slots"] = {"line": "2호선", "date": "2026-11-14"}
    errors = validate_plan(payload)
    assert any("missing 'station'" in e for e in errors)


def test_validate_plan_rejects_too_many_steps():
    payload = _valid_plan_payload()
    payload["steps"] = [
        {"step_id": f"s{i}", "tool": "predict", "args": {}} for i in range(PLAN_LIMITS["max_steps"] + 1)
    ]
    errors = validate_plan(payload)
    assert any("exceeds max_steps" in e for e in errors)


def test_validate_plan_rejects_unknown_tool():
    payload = _valid_plan_payload()
    payload["steps"] = [{"step_id": "s1", "tool": "teleport", "args": {}}]
    assert any("unknown tool" in e for e in validate_plan(payload))


def test_validate_plan_rejects_dependency_cycle():
    payload = _valid_plan_payload()
    payload["steps"] = [
        {"step_id": "a", "tool": "predict", "args": {}, "depends_on": ["b"]},
        {"step_id": "b", "tool": "history", "args": {}, "depends_on": ["a"]},
    ]
    assert any("cycle" in e for e in validate_plan(payload))


def test_validate_plan_rejects_unknown_dependency_and_limits():
    payload = _valid_plan_payload()
    payload["steps"] = [
        {"step_id": "a", "tool": "predict", "args": {}, "depends_on": ["ghost"]},
    ]
    payload["limits"] = {"max_calls": PLAN_LIMITS["max_calls"] + 1}
    errors = validate_plan(payload)
    assert any("unknown dependency" in e for e in errors)
    assert any("max_calls" in e for e in errors)


def test_plan_limits_and_deadlines():
    assert PLAN_LIMITS == {"max_steps": 6, "max_calls": 8, "max_llm_calls": 2}
    assert DEADLINE_MS == {"template": 3000, "cloud": 6000, "local_batch": 25000}


# ---------- tools ----------


def test_allowlist_is_exactly_twelve():
    assert isinstance(tools.ALLOWLIST, tuple)
    assert len(tools.ALLOWLIST) == 12
    assert set(tools.ALLOWLIST) == {
        "resolve_station",
        "predict",
        "compare",
        "history",
        "aggregate",
        "simulate",
        "explain",
        "reconcile",
        "backtest",
        "drift",
        "alerts",
        "report",
    }
    # 쓰기 허용은 report 하나뿐
    assert tools.WRITE_ALLOWED == {"report"}
    assert tools.WRITE_ALLOWED <= set(tools.ALLOWLIST)


def test_check_tool_allowed_accepts_allowlist_and_rejects_others():
    for name in tools.ALLOWLIST:
        assert tools.check_tool_allowed(name) is None
    with pytest.raises(ValueError) as excinfo:
        tools.check_tool_allowed("shell")
    assert "E_INVALID_ARG" in str(excinfo.value)


def test_envelope_wrappers_delegate():
    ok = tools.envelope_ok("predict", {"value": 1}, elapsed_ms=5)
    assert ok.ok is True and ok.elapsed_ms == 5
    bad = tools.envelope_error("predict", ErrorCode.E_NOT_FOUND, "없음")
    assert bad.ok is False and bad.data["code"] == "E_NOT_FOUND"


# 도구 시그니처 계약 (파라미터 이름·순서). 바뀌면 여기서 실패한다.
EXPECTED_SIGNATURES = {
    "resolve_station": ("query", "limit"),
    "predict": ("line", "station", "date", "hour", "weather"),
    "compare": ("line", "station", "date", "hour", "models"),
    "history": ("line", "station", "date_from", "date_to", "hour"),
    "aggregate": ("scope", "range", "metric"),
    "simulate": ("base", "scenarios"),
    "explain": ("line", "station", "date", "hour", "model"),
    "reconcile": ("line", "date", "hour"),
    "backtest": ("bucket", "model", "slice"),
    "drift": (),
    "alerts": ("scope", "range", "z"),
    "report": ("kind", "date", "sections"),
}


def test_tool_parameter_names_match_plan_10_6():
    assert set(EXPECTED_SIGNATURES) == set(tools.ALLOWLIST)
    for name, expected in EXPECTED_SIGNATURES.items():
        actual = tuple(inspect.signature(getattr(tools, name)).parameters)
        assert actual == expected, f"{name}: {actual} != {expected}"


def test_tool_parameter_kinds_match_plan_10_6():
    resolve = inspect.signature(tools.resolve_station).parameters
    assert resolve["query"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert resolve["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    for name in tools.ALLOWLIST:
        if name == "resolve_station":
            continue
        for param in inspect.signature(getattr(tools, name)).parameters.values():
            assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (name, param.name)


def test_aggregate_rejects_non_dict_scope_or_range():
    with pytest.raises(ValueError) as scope_error:
        tools.aggregate(
            "2호선", {"date_from": "2026-01-01", "date_to": "2026-01-02"}
        )
    assert "E_INVALID_ARG" in str(scope_error.value)
    with pytest.raises(ValueError) as range_error:
        tools.aggregate({"line": "2호선"}, "2026-01-01~2026-01-02")
    assert "E_INVALID_ARG" in str(range_error.value)


def test_alerts_rejects_non_dict_scope_or_range():
    with pytest.raises(ValueError) as scope_error:
        tools.alerts("2호선", {"date_from": "2026-01-01", "date_to": "2026-01-02"})
    assert "E_INVALID_ARG" in str(scope_error.value)
    with pytest.raises(ValueError) as range_error:
        tools.alerts({"line": "2호선"}, "2026-01-01~2026-01-02")
    assert "E_INVALID_ARG" in str(range_error.value)


def _dummy_value(annotation: object) -> object:
    text = str(annotation)
    if "list" in text:
        return []
    if "dict" in text:
        return {}
    if "float" in text:
        return 0.0
    if "int" in text:
        return 0
    return "x"


def _call_with_dummies(fn):
    signature = inspect.signature(fn)
    args: list[object] = []
    kwargs: dict[str, object] = {}
    for param in signature.parameters.values():
        if param.default is not inspect.Parameter.empty:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        value = _dummy_value(param.annotation)
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[param.name] = value
        else:
            args.append(value)
    return fn(*args, **kwargs)


def test_tool_bodies_are_callable_and_return_envelopes():
    """도구 12개는 구현되어 있고, 잘못된 인자도 예외 대신 같은 모양의 봉투로 알린다."""
    for name in tools.ALLOWLIST:
        fn = getattr(tools, name)
        assert callable(fn), name
        result = _call_with_dummies(fn)
        assert isinstance(result, Envelope), name
        assert result.tool == name
        assert set(result.to_dict()) == set(ENVELOPE_KEYS)


def test_tool_stubs_annotate_envelope_return():
    for name in tools.ALLOWLIST:
        annotation = inspect.signature(getattr(tools, name)).return_annotation
        assert "Envelope" in str(annotation), name