"""골든셋 녹화. 실제 도구 출력을 입력 해시와 함께 저장해 두고 재생은 그것만 쓴다.

녹화는 예측기를 한 번 읽는다(수십 초). 재생은 읽지 않는다.

사용:
    python -m metro_insight.agent.evals.record           # 골든 다시 녹화
    python -m metro_insight.agent.evals.record --check   # 녹화 없이 표만 확인
"""
from __future__ import annotations

import hashlib
import json
import os
from time import perf_counter

from metro_insight.agent import orchestrator
from metro_insight.agent.evals import cases as cases_module
from metro_insight.agent.evals import runner

MODE_ENV = runner.MODE_ENV
TODAY_ENV = runner.TODAY_ENV


class Recorder:
    """한 문항을 실제 도구로 실행하며 봉투를 모은다."""

    def __init__(self, case: dict) -> None:
        self.case = case
        self.fixtures: dict[str, dict] = {}

    def __call__(self, name: str, args: dict):
        from metro_insight.agent import tools

        envelope = getattr(tools, name)(**dict(args))
        self.fixtures[runner.fixture_key(name, dict(args))] = envelope.to_dict()
        return envelope


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_case(case: dict) -> tuple[dict, list[str]]:
    """(녹화 행, 문제 목록). 기대 의도·도구가 어긋나면 문제로 남긴다."""
    recorder = Recorder(case)
    problems: list[str] = []
    original = orchestrator._call_tool
    orchestrator._call_tool = recorder
    started = perf_counter()
    try:
        result = orchestrator.run_turn(case["utterance"])
    finally:
        orchestrator._call_tool = original
    elapsed = int((perf_counter() - started) * 1000)

    observed_tools = [str(step.get("tool")) for step in result["plan"]["steps"]]
    if result["plan"]["intent"] != case["intent"]:
        problems.append(f"{case['id']}: intent {result['plan']['intent']} != {case['intent']}")
    if observed_tools != list(case["tools"]):
        problems.append(f"{case['id']}: tools {observed_tools} != {list(case['tools'])}")
    if result["verdict"]["verdict"] != "pass":
        problems.append(f"{case['id']}: verdict {result['verdict']['verdict']}")
    if case["intent"] == "CLARIFY" and not result["plan"].get("refusal"):
        problems.append(f"{case['id']}: 거부문이 아닙니다")

    bundle = result["bundle"]
    row = {
        "id": case["id"],
        "split": case["split"],
        "category": case["category"],
        "utterance": case["utterance"],
        "intent": case["intent"],
        "tools": list(case["tools"]),
        "forbidden": list(case["forbidden"]),
        "max_latency_ms": case["max_latency_ms"],
        "expected": {
            "verdict": result["verdict"]["verdict"],
            "policy": dict(bundle.get("policy") or {}),
            "facts": [
                {"fact_id": fact["fact_id"], "kind": fact["kind"], "values": fact["values"]}
                for fact in bundle.get("facts") or []
            ],
            "answer": result["answer"],
            "answer_sha256": _digest(result["answer"]),
            "recorded_latency_ms": elapsed,
        },
        "fixtures": recorder.fixtures,
    }
    return row, problems


def write_split(split: str) -> list[str]:
    """한 split 을 녹화해 ``golden_<split>.jsonl`` 로 쓴다. 문제 목록을 돌려준다."""
    problems: list[str] = []
    rows = []
    for case in cases_module.CASES:
        if case["split"] != split:
            continue
        row, found = record_case(case)
        rows.append(row)
        problems.extend(found)
    path = runner.golden_path(split)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return problems


def main(argv: list[str] | None = None) -> int:
    """녹화 또는 표 검사. 문제가 있으면 1 을 돌려준다."""
    args = list(argv or [])
    os.environ[TODAY_ENV] = runner.REFERENCE_DAY
    os.environ[MODE_ENV] = "template"
    summary = cases_module.summary()
    if summary["total"] != 130 or summary["by_split"] != {"dev": 90, "heldout": 40}:
        print(f"문항표 불일치: {summary}")
        return 1
    if summary["by_category"] != cases_module.CATEGORY_COUNTS:
        print(f"유형 분포 불일치: {summary['by_category']}")
        return 1
    if args and args[0] == "--check":
        for split in cases_module.SPLITS:
            print(f"{split}: 녹화 {len(runner.load_golden(split))}개")
        return 0
    if not orchestrator.warmup():
        print("예측기를 읽지 못해 녹화할 수 없습니다.")
        return 1
    problems: list[str] = []
    for split in cases_module.SPLITS:
        found = write_split(split)
        print(f"{split}: {len(runner.load_golden(split))}개 기록, 문제 {len(found)}건")
        problems.extend(found)
    for line in problems:
        print("  " + line)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())