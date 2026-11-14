"""골든셋 130문항을 오프라인으로 다시 돌려 녹화와 같은지 확인한다.

재생은 녹화된 봉투만 쓰므로 예측기·파일·네트워크를 읽지 않는다. 문항표의 의도·도구는
표가 정본이고, 수치·문장은 녹화가 정본이다. 서로 다른 두 정본이 어긋나면 실패한다.
"""
from __future__ import annotations

from collections import Counter

import pytest

from metro_insight.agent.evals import cases as case_table
from metro_insight.agent.evals import runner
from metro_insight.agent.tools import ALLOWLIST

pytestmark = pytest.mark.agent


@pytest.fixture(scope="session")
def table() -> dict[str, dict]:
    return {case["id"]: case for case in case_table.CASES}


@pytest.fixture(scope="session")
def recorded() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for split in case_table.SPLITS:
        for row in runner.load_golden(split):
            rows[row["id"]] = row
    return rows


@pytest.fixture(scope="session")
def replayed(recorded) -> dict[str, dict]:
    """문항별 (녹화 행, 재생 결과). 재생은 문항당 한 번만 한다."""
    return {case_id: (row, runner.replay(row)) for case_id, row in recorded.items()}


def test_case_table_has_expected_size_and_splits():
    summary = case_table.summary()
    assert summary["total"] == 130
    assert summary["by_split"] == {"dev": 90, "heldout": 40}


def test_category_counts_match_spec():
    assert case_table.summary()["by_category"] == case_table.CATEGORY_COUNTS


def test_case_ids_are_unique():
    identifiers = [case["id"] for case in case_table.CASES]
    assert len(identifiers) == len(set(identifiers))


@pytest.mark.parametrize(("split", "expected"), [("dev", 90), ("heldout", 40)])
def test_golden_file_covers_split(split, expected):
    rows = runner.load_golden(split)
    assert len(rows) == expected
    assert {row["split"] for row in rows} == {split}


@pytest.mark.parametrize(("split", "expected"), [("dev", 90), ("heldout", 40)])
def test_golden_rows_match_case_table(split, expected, table):
    rows = runner.load_golden(split)
    assert len(rows) == expected
    for row in rows:
        case = table[row["id"]]
        assert row["utterance"] == case["utterance"]
        assert row["intent"] == case["intent"]
        assert row["tools"] == case["tools"]
        assert row["forbidden"] == case["forbidden"]


def test_fixtures_only_use_allowlisted_tools(recorded):
    for case_id, row in recorded.items():
        for key in row["fixtures"]:
            tool = key.split("::", 1)[0]
            assert tool in ALLOWLIST, (case_id, tool)


def test_non_refusal_cases_have_fixtures(recorded):
    for case_id, row in recorded.items():
        if row["intent"] == "CLARIFY":
            continue
        assert row["fixtures"], case_id


@pytest.mark.parametrize("case_id", [case["id"] for case in case_table.CASES])
def test_replay_reproduces_recording(case_id, table, replayed):
    case = table[case_id]
    row, result = replayed[case_id]
    plan = result["plan"]
    assert plan["intent"] == case["intent"]
    assert [step["tool"] for step in plan["steps"]] == case["tools"]
    assert result["verdict"]["verdict"] == "pass"
    assert result["answer"] == row["expected"]["answer"]
    assert result["bundle"]["policy"] == row["expected"]["policy"]


@pytest.mark.parametrize("case_id", [case["id"] for case in case_table.CASES])
def test_replay_reproduces_fact_values(case_id, replayed):
    row, result = replayed[case_id]
    expected = {fact["fact_id"]: fact for fact in row["expected"]["facts"]}
    produced = {fact["fact_id"]: fact for fact in result["bundle"]["facts"]}
    assert set(produced) == set(expected)
    for fact_id, fact in expected.items():
        assert produced[fact_id]["kind"] == fact["kind"]
        assert runner.numbers_match(produced[fact_id]["values"], fact["values"])


@pytest.mark.parametrize("case_id", [case["id"] for case in case_table.CASES])
def test_replay_has_no_unbound_numbers(case_id, replayed):
    _, result = replayed[case_id]
    checks = result["verdict"]["checks"]
    assert checks["digit_free"] is True
    assert checks["placeholder_binding"] is True
    assert checks["fact_match"] is True
    codes = {item["code"] for item in result["verdict"]["violations"]}
    assert codes == set()


@pytest.mark.parametrize("case_id", [case["id"] for case in case_table.CASES])
def test_replay_avoids_forbidden_strings(case_id, replayed):
    row, result = replayed[case_id]
    for word in row["forbidden"]:
        assert word not in result["answer"], (case_id, word)


def test_refusals_cover_ten_out_of_scope_questions(replayed, table):
    out_of_scope = [key for key, case in table.items() if case["category"] == "refusal"]
    assert len(out_of_scope) == 10
    for case_id in out_of_scope:
        result = replayed[case_id][1]
        assert result["fallback"] == "refusal", case_id
        assert result["verdict"]["verdict"] == "pass"
        assert result["plan"]["refusal"]
        assert result["plan"]["steps"] == []
        assert result["answer"] == result["plan"]["refusal"]
        assert "tool_calls" not in result["trace"]


def test_adversarial_requests_are_refused(replayed, table):
    adversarial = [key for key, case in table.items() if case["category"] == "adversarial"]
    assert len(adversarial) == 5
    for case_id in adversarial:
        result = replayed[case_id][1]
        assert result["fallback"] == "refusal", case_id
        assert result["verdict"]["verdict"] == "pass"
        assert result["plan"]["steps"] == []


def test_required_phrases_are_always_present(replayed):
    checked = 0
    for case_id, (_, result) in replayed.items():
        policy = result["bundle"]["policy"]
        flags = (
            ("future_partial", "일부 미검증"),
            ("de_missing", "계산 불가"),
            ("stale", "낡음"),
        )
        for flag, phrase in flags:
            if policy.get(flag):
                assert phrase in result["answer"], (case_id, phrase)
                checked += 1
        assert result["verdict"]["checks"]["required_phrases"] is True, case_id
    assert checked >= 50


def test_answers_never_show_missing_values_as_zero(replayed, table):
    checked = 0
    for case_id, (_, result) in replayed.items():
        if not result["bundle"]["policy"].get("de_missing"):
            continue
        checked += 1
        assert table[case_id]["category"] in {"single", "compare", "simulate"}, case_id
        prediction_facts = [
            fact
            for fact in result["bundle"]["facts"]
            if fact["kind"] in {"prediction", "comparison", "simulation"}
        ]
        assert prediction_facts, case_id
        for fact in prediction_facts:
            assert "D" not in fact["values"] and "E" not in fact["values"], case_id
            assert fact["values"].get("D") != 0 and fact["values"].get("E") != 0, case_id
        assert "계산 불가" in result["answer"], case_id
    assert checked >= 10


def test_replay_is_independent_of_process_clock(recorded):
    case_id = "d01"
    row = recorded[case_id]
    first = runner.replay(row)["answer"]
    second = runner.replay(row)["answer"]
    assert first == second


def test_replay_does_not_invent_tool_outputs():
    case = {
        "id": "probe",
        "utterance": "11/16 8시 강남 2호선 예상",
        "intent": "PREDICT",
        "tools": ["resolve_station", "predict"],
        "fixtures": {},
    }
    result = runner.replay(case)
    assert result["fallback"] == "no_facts"
    assert "계산할 수 없습니다" in result["answer"]
    assert result["bundle"]["facts"] == []


def test_recorded_answers_are_plain_text(recorded):
    for case_id, row in recorded.items():
        answer = row["expected"]["answer"]
        assert answer, case_id
        assert "nan" not in answer.lower(), case_id
        assert "None" not in answer, case_id
        assert row["expected"]["answer_sha256"]


def test_category_distribution_matches_table(replayed, table):
    counted = Counter(table[case_id]["category"] for case_id in replayed)
    assert counted == Counter(case_table.CATEGORY_COUNTS)