"""한 턴 실행: 계획 → 도구 실행 → 근거 묶음 → 문장 조립 → 검증 → 답.

숫자는 도구만 만들고 문장은 플레이스홀더만 쓴다. 상한은 스텝 6·호출 8·도구 1.5초·
턴 3초(템플릿)이며, 넘으면 남은 단계를 멈추고 있는 사실만으로 답한다. 문장 조립이
LLM 으로 실패하면 템플릿으로 내려가고, 3연속 실패면 10분 동안 템플릿만 쓴다.
"""
from __future__ import annotations

import os
import time
from time import perf_counter

from metro_insight.agent import planner
from metro_insight.agent import verify as verify_module
from metro_insight.agent.contracts import (
    DEADLINE_MS,
    PLAN_LIMITS,
    DraftAnswer,
    FactBundle,
    Intent,
    Plan,
    Verdict,
)
from metro_insight.agent.tools import ALLOWLIST
from metro_insight.agent.tools._common import predictor, predictor_ready

MODE_ENV = "METRO_AGENT_MODE"
DEFAULT_MODE = "template"
MODES = ("template", "llm", "replay")
TOOL_TIMEOUT_MS = 1500
STEP_LIMIT = PLAN_LIMITS["max_steps"]
CALL_LIMIT = PLAN_LIMITS["max_calls"]


class LlmUnavailable(RuntimeError):
    """이 실행 환경에서는 문장 조립 LLM 을 쓰지 않는다(네트워크 호출 없음)."""


class CircuitBreaker:
    """문장 조립 실패가 이어지면 한동안 템플릿만 쓰게 한다."""

    def __init__(self, threshold: int = 3, cooldown_seconds: float = 600.0) -> None:
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.failures = 0
        self.opened_at: float | None = None

    def allow(self, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        if self.opened_at is None:
            return True
        if moment - self.opened_at >= self.cooldown_seconds:
            self.opened_at = None
            self.failures = 0
            return True
        return False

    def record_failure(self, now: float | None = None) -> None:
        moment = time.monotonic() if now is None else now
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = moment

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None


BREAKER = CircuitBreaker()
_WARMED = False


def mode() -> str:
    """실행 모드. 기본은 템플릿(LLM·네트워크 없이 동작)."""
    raw = os.environ.get(MODE_ENV, DEFAULT_MODE).strip().lower()
    return raw if raw in MODES else DEFAULT_MODE


def warmup() -> bool:
    """예측기와 첫 계산 경로를 미리 데운다(앱 시작 시 1회).

    첫 호출에만 드는 비용(모델 로딩·feature 준비)을 사용자 질의에서 떼어 낸다.
    실패하면 다음 도구 호출에서 사유와 함께 드러나므로 여기서는 False 만 돌려준다.
    """
    global _WARMED
    if _WARMED:
        return True
    try:
        from metro_insight.agent.tools._common import DATA_END as data_end
        from metro_insight.agent.tools.predict import predict_values

        predictor()
        predict_values("2호선", "강남", data_end.isoformat(), 8, None)
    except Exception:  # noqa: BLE001 — 실패는 다음 도구 호출에서 드러난다
        return False
    _WARMED = True
    return True


def compose_with_llm(_bundle: FactBundle) -> DraftAnswer:
    """문장 조립 LLM 호출 자리. 기본 실행 경로에서는 쓰지 않는다."""
    raise LlmUnavailable("문장 조립 LLM 을 쓰지 않는 실행 모드입니다.")


# ---------- 도구 실행 ----------


def _call_tool(name: str, args: dict):
    if name not in ALLOWLIST:
        raise ValueError(f"허용목록 밖 도구: {name}")
    function = getattr(__import__("metro_insight.agent.tools", fromlist=[name]), name)
    return function(**args)


def _execute(plan: Plan) -> tuple[list[dict], list[dict], list[dict]]:
    """(실행 결과 목록, 사용할 수 없음 목록, 추적 목록)."""
    results: list[dict] = []
    unavailable: list[dict] = []
    trace: list[dict] = []
    started = perf_counter()
    deadline = DEADLINE_MS.get(mode(), DEADLINE_MS["template"])
    calls = 0
    done: set[str] = set()
    skipped: set[str] = set()
    cold_turn = not predictor_ready()
    for step in plan.steps[:STEP_LIMIT]:
        step_id = str(step.get("step_id"))
        deps = [str(dep) for dep in (step.get("depends_on") or [])]
        if any(dep in skipped for dep in deps):
            skipped.add(step_id)
            trace.append({"step_id": step_id, "tool": step.get("tool"), "skipped": "dependency"})
            continue
        if calls >= CALL_LIMIT or (not cold_turn and int((perf_counter() - started) * 1000) > deadline):
            skipped.add(step_id)
            unavailable.append(
                {
                    "step_id": step_id,
                    "reason_code": "E_UNAVAILABLE",
                    "human": "호출 상한 또는 턴 제한 시간을 넘어 실행하지 않았습니다.",
                }
            )
            trace.append({"step_id": step_id, "tool": step.get("tool"), "skipped": "limit"})
            continue
        name = str(step.get("tool"))
        calls += 1
        begin = perf_counter()
        try:
            envelope = _call_tool(name, dict(step.get("args") or {}))
            elapsed = int((perf_counter() - begin) * 1000)
        except Exception as exc:  # noqa: BLE001 — 도구 예외도 사유를 남긴다
            elapsed = int((perf_counter() - begin) * 1000)
            unavailable.append(
                {
                    "step_id": step_id,
                    "tool": name,
                    "reason_code": "E_INTERNAL",
                    "human": f"{type(exc).__name__}: {exc}",
                }
            )
            skipped.add(step_id)
            trace.append({"step_id": step_id, "tool": name, "elapsed_ms": elapsed, "ok": False})
            continue
        trace.append(
            {
                "step_id": step_id,
                "tool": name,
                "elapsed_ms": elapsed,
                "ok": bool(envelope.ok),
                "cold": cold_turn,
            }
        )
        if envelope.elapsed_ms > TOOL_TIMEOUT_MS and not cold_turn:
            unavailable.append(
                {
                    "step_id": step_id,
                    "tool": name,
                    "reason_code": "E_UNAVAILABLE",
                    "human": f"도구 응답이 {TOOL_TIMEOUT_MS}ms 를 넘었습니다({envelope.elapsed_ms}ms).",
                }
            )
        if not envelope.ok:
            unavailable.append(
                {
                    "step_id": step_id,
                    "tool": name,
                    "reason_code": envelope.data.get("code"),
                    "human": envelope.data.get("human"),
                }
            )
            skipped.add(step_id)
            continue
        done.add(step_id)
        results.append({"step_id": step_id, "tool": name, "envelope": envelope})
    return results, unavailable, trace


# ---------- 근거 묶음 ----------


def _first(results: list[dict], tool: str) -> dict | None:
    for item in results:
        if item["tool"] == tool:
            return item["envelope"].data
    return None


def _policy(plan: Plan, results: list[dict], unavailable: list[dict]) -> dict:
    prediction = (
        _first(results, "predict")
        or _first(results, "compare")
        or _first(results, "simulate")
        or {}
    )
    drift_data = _first(results, "drift") or {}
    future_partial = bool(prediction.get("horizon_partial")) or (
        bool(prediction.get("is_future")) and int(prediction.get("days_after_data_end") or 0) > 136
    )
    de_missing = bool(prediction.get("missing"))
    stale = bool(drift_data.get("stale")) if drift_data else False
    return {
        "mode": mode(),
        "future_partial": future_partial,
        "de_missing": de_missing,
        "stale": stale,
        "unavailable_count": len(unavailable),
        "intent": plan.intent,
    }


def _bundle(plan: Plan, results: list[dict], unavailable: list[dict], policy: dict) -> FactBundle:
    facts: list[dict] = []
    counter = 0

    def add(kind: str, values: dict, source: str, caveats: list[str] | None = None) -> str:
        nonlocal counter
        counter += 1
        fact_id = f"F{counter}"
        facts.append(
            verify_module.build_fact(
                fact_id,
                kind,
                values,
                source={"tool": source},
                caveats=caveats,
            )
        )
        return fact_id

    prediction = _first(results, "predict")
    compare = _first(results, "compare")
    simulate = _first(results, "simulate")
    explain = _first(results, "explain")
    aggregate = _first(results, "aggregate")
    history = _first(results, "history")
    reconcile = _first(results, "reconcile")
    backtest = _first(results, "backtest")
    drift = _first(results, "drift")
    alerts = _first(results, "alerts")
    report = _first(results, "report")

    if prediction:
        values = {
            "line": prediction["line"],
            "station": prediction["station"],
            "date": prediction["date"],
            "hour": prediction["hour"],
            "days": pred_days(prediction),
            "horizon": prediction["horizon_label"],
        }
        for name, value in prediction["by_model"].items():
            values[name] = value
        add(
            "prediction",
            values,
            "predict",
            [str(item.get("reason_code")) for item in prediction.get("missing", [])],
        )
    if compare:
        values = {
            "line": compare["line"],
            "station": compare["station"],
            "date": compare["date"],
            "hour": compare["hour"],
            "days": pred_days(compare),
            "horizon": compare.get("horizon_label") or "",
        }
        for name, value in compare["by_model"].items():
            values[name] = value
        diffs = compare.get("diffs") or []
        if diffs:
            values["diff"] = diffs[0].get("delta")
            values["pair"] = diffs[0].get("pair")
        add("comparison", values, "compare")
    if simulate:
        for scenario in simulate["scenarios"]:
            values = {
                "line": simulate["line"],
                "station": simulate["station"],
                "date": simulate["date"],
                "hour": simulate["hour"],
                "name": scenario["name"],
                "days": 0,
            }
            for name, value in scenario["by_model"].items():
                values[name] = value
                if name in scenario.get("delta", {}):
                    values[f"delta_{name}"] = scenario["delta"][name]
                if name in scenario.get("delta_pct", {}):
                    values[f"delta_pct_{name}"] = scenario["delta_pct"][name]
            add("simulation", values, "simulate")
    if explain:
        values = {
            "line": explain["line"],
            "station": explain["station"],
            "date": explain["date"],
            "hour": explain["hour"],
            "model": explain["model"],
        }
        contributions = explain.get("contributions") or []
        if contributions:
            item = contributions[0]
            values["model_value"] = item.get("model_value")
            values["baseline"] = item.get("baseline_value")
            values["delta"] = item.get("delta")
            values["features"] = item.get("baseline_setting")
        add("explanation", values, "explain")
    if aggregate:
        metric_label = {"sum": "합계", "mean": "평균", "max": "최대"}.get(
            str(aggregate.get("metric")), str(aggregate.get("metric"))
        )
        values = {
            "line": aggregate.get("scope", {}).get("line") or "전체",
            "date_from": aggregate["range"]["date_from"],
            "date_to": aggregate["range"]["date_to"],
            "metric": metric_label,
            "value": aggregate.get("value"),
            "n": aggregate.get("n"),
            "coverage": aggregate.get("coverage"),
        }
        top = aggregate.get("top") or []
        if top:
            values["top_station"] = top[0]["station"]
            values["top_value"] = top[0]["value"]
        add("aggregate", values, "aggregate")
    if history:
        points = history.get("points") or []
        last = points[-1] if points else {}
        values = {
            "line": history["line"],
            "station": history["station"],
            "date_from": history["date_from"],
            "date_to": history["date_to"],
            "n": history.get("n"),
            "missing_count": history.get("missing_count"),
            "coverage": history.get("coverage"),
            "last_date": last.get("date") or "",
            "last_value": last.get("value"),
        }
        add("history", values, "history")
    if reconcile:
        values = {
            "line": reconcile["line"],
            "date": reconcile["date"],
            "hour": reconcile["hour"],
            "line_sum": reconcile.get("line_sum"),
            "station_sum": reconcile.get("station_sum"),
            "diff": reconcile.get("diff"),
            "stations_with_data": reconcile.get("stations_with_data"),
            "missing_count": len(reconcile.get("missing_stations") or []),
        }
        add("reconcile", values, "reconcile")
    if backtest:
        for row in backtest["rows"]:
            values = {
                "model": row["model"],
                "bucket": row.get("bucket") or "전체",
                "mae": row.get("mae"),
                "rmse": row.get("rmse"),
                "n": row.get("n"),
            }
            add("backtest", values, "backtest")
    if drift:
        values = {
            "data_end": drift["data_end"],
            "days": drift["days_since_data_end"],
            "stale": "낡음" if drift["stale"] else "최신",
            "map_pairs": drift["map_pairs"],
            "map_only": drift["map_only_unresolved"],
        }
        add("drift", values, "drift")
    if alerts:
        values = {
            "date_from": alerts["range"]["date_from"],
            "date_to": alerts["range"]["date_to"],
            "z": alerts.get("z"),
            "n": alerts.get("n"),
        }
        add("alerts", values, "alerts")
    if report:
        values = {
            "kind": report["kind"],
            "date": report["date"],
            "path": report["path"],
            "sections": ", ".join(report["sections"]),
        }
        add("report", values, "report")

    return FactBundle(
        bundle_id=f"b-{plan.plan_id}",
        as_of=None,
        facts=facts,
        unavailable=unavailable,
        policy=policy,
    )


def pred_days(prediction: dict) -> int:
    """데이터 끝 이후 경과일(과거면 0)."""
    return max(int(prediction.get("days_after_data_end") or 0), 0)


# ---------- 문장 조립 ----------


def _models_prefix(values: dict) -> str:
    names = [name for name in ("A", "B", "C", "D", "E") if name in values]
    return ", ".join(f"{name} 모델 {{{{F1.{name}}}}}" for name in names)


def compose(plan: Plan, bundle: FactBundle) -> DraftAnswer:
    """근거 묶음 → 템플릿 문장. 숫자는 전부 플레이스홀더다."""
    claims: list[dict] = []
    policy = bundle.policy

    def claim(text: str, *fact_ids: str) -> None:
        claims.append({"text": text, "fact_refs": list(fact_ids) or ["F1"]})

    def id_of(index: int) -> str:
        return f"F{index}"

    if policy.get("clarify"):
        if not bundle.facts:
            return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=[], caveats=[])
        values = bundle.facts[0]["values"]
        claim("{{F1.question}}", "F1")
        if int(values.get("ambiguous") or 0) > 1:
            claim(
                "후보는 {{F1.c1_line}} {{F1.c1_station}}, {{F1.c2_line}} {{F1.c2_station}}입니다.",
                "F1",
            )
        return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=claims, caveats=[])

    intent = plan.intent
    if intent in (Intent.PREDICT.value, Intent.COMPARE.value, Intent.SIMULATE.value,
                  Intent.EXPLAIN.value):
        if not bundle.facts:
            return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=[], caveats=[])
        values = bundle.facts[0]["values"]
        if intent == Intent.PREDICT.value:
            claim(
                f"{{{{F1.line}}}} {{{{F1.station}}}} {{{{F1.date}}}} {{{{F1.hour}}}} 예상 승차 인원은 "
                f"{_models_prefix(values)}입니다.",
                "F1",
            )
            if policy.get("future_partial"):
                claim(
                    "데이터 끝 이후 {{F1.days}} 지난 날짜라 이 값은 일부 미검증입니다.",
                    "F1",
                )
            if policy.get("de_missing"):
                claim("D와 E 모델은 대상일 전 실측이 없어 계산 불가입니다.", "F1")
        elif intent == Intent.COMPARE.value:
            claim(
                f"{{{{F1.line}}}} {{{{F1.station}}}} {{{{F1.date}}}} {{{{F1.hour}}}} 기준 "
                f"{_models_prefix(values)}입니다.",
                "F1",
            )
            if "diff" in values:
                claim("{{F1.pair}} 차이는 {{F1.diff}}입니다.", "F1")
            if policy.get("future_partial"):
                claim(
                    "데이터 끝 이후 {{F1.days}} 지난 날짜라 이 값은 일부 미검증입니다.",
                    "F1",
                )
            if policy.get("de_missing"):
                claim("D와 E 모델은 대상일 전 실측이 없어 계산 불가입니다.", "F1")
        elif intent == Intent.SIMULATE.value:
            for index in range(1, len(bundle.facts) + 1):
                fact_id = id_of(index)
                values = bundle.facts[index - 1]["values"]
                claim(
                    f"{{{{{fact_id}.name}}}} 조건이면 {{{{F1.line}}}} {{{{F1.station}}}} "
                    f"{{{{{fact_id}.date}}}} {{{{F1.hour}}}} 예상은 {_models_prefix_at(values, fact_id)}이고 "
                    f"기준 대비 C 모델 차이는 {{{{{fact_id}.delta_C}}}}입니다.",
                    fact_id,
                )
            if policy.get("future_partial"):
                claim("데이터 끝 이후 지난 날짜라 이 값은 일부 미검증입니다.", "F1")
            if policy.get("de_missing"):
                claim("D와 E 모델은 대상일 전 실측이 없어 계산 불가입니다.", "F1")
        else:
            if values.get("model_value") is not None:
                claim(
                    "{{F1.model}} 모델 값 {{F1.model_value}}는 날씨를 평년으로 되돌린 "
                    "{{F1.baseline}}과 비교해 {{F1.delta}} 차이입니다.",
                    "F1",
                )
            else:
                claim(
                    "{{F1.model}} 모델은 날씨를 쓰지 않아 feature 기여도를 계산하지 않습니다.",
                    "F1",
                )
        return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=claims, caveats=[])

    if intent == Intent.AGGREGATE.value:
        if not bundle.facts:
            return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=[], caveats=[])
        claim(
            "{{F1.line}} {{F1.date_from}}~{{F1.date_to}} {{F1.metric}} {{F1.value}}입니다.",
            "F1",
        )
        claim("관측 칸 {{F1.n}}개, 커버리지 {{F1.coverage}}입니다.", "F1")
        if "top_station" in (bundle.facts[0]["values"] if bundle.facts else {}):
            claim("가장 큰 역은 {{F1.top_station}}이며 값은 {{F1.top_value}}입니다.", "F1")
        return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=claims, caveats=[])

    if intent == Intent.HISTORY.value:
        if not bundle.facts:
            return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=[], caveats=[])
        claim(
            "{{F1.line}} {{F1.station}} {{F1.date_from}}~{{F1.date_to}} 실측 {{F1.n}}일 중 "
            "관측이 없는 날은 {{F1.missing_count}}일입니다.",
            "F1",
        )
        if bundle.facts[0]["values"].get("last_date"):
            claim("마지막 값은 {{F1.last_date}} {{F1.last_value}}입니다.", "F1")
        return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=claims, caveats=[])

    if intent == Intent.BACKTEST.value:
        for index in range(1, len(bundle.facts) + 1):
            fact_id = id_of(index)
            claim(
                f"{{{{{fact_id}.model}}}} 모델 {{{{F1.bucket}}}} 구간 MAE "
                f"{{{{{fact_id}.mae}}}}, RMSE {{{{{fact_id}.rmse}}}}, 표본 {{{{{fact_id}.n}}}}입니다.",
                fact_id,
            )
        return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=claims, caveats=[])

    if intent == Intent.DRIFT.value:
        if not bundle.facts:
            return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=[], caveats=[])
        claim(
            "데이터 기준일 {{F1.data_end}}부터 {{F1.days}} 지났고 상태는 {{F1.stale}}입니다.",
            "F1",
        )
        claim(
            "지도 (호선, 역) {{F1.map_pairs}}쌍 중 예측 데이터가 없는 쌍은 {{F1.map_only}}개입니다.",
            "F1",
        )
        return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=claims, caveats=[])

    if intent == Intent.ALERTS.value:
        if not bundle.facts:
            return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=[], caveats=[])
        claim(
            "{{F1.date_from}}부터 {{F1.date_to}}까지 기준 {{F1.z}} 시그마를 넘는 날은 {{F1.n}}일입니다"
            "(과거 실측 기준).",
            "F1",
        )
        return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=claims, caveats=[])

    if intent == Intent.RECONCILE.value:
        if not bundle.facts:
            return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=[], caveats=[])
        claim(
            "{{F1.line}} {{F1.date}} {{F1.hour}} 역별 실측 합은 {{F1.station_sum}}, "
            "집계 합은 {{F1.line_sum}}, 차이는 {{F1.diff}}입니다.",
            "F1",
        )
        claim("관측이 없는 역은 {{F1.missing_count}}개입니다.", "F1")
        return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=claims, caveats=[])

    if intent == Intent.REPORT.value:
        if not bundle.facts:
            return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=[], caveats=[])
        claim(
            "{{F1.kind}} 리포트를 {{F1.path}}에 저장했습니다. 섹션은 {{F1.sections}}입니다.",
            "F1",
        )
        return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=claims, caveats=[])

    return DraftAnswer(answer_id=f"a-{plan.plan_id}", claims=claims, caveats=[])


def _models_prefix_at(values: dict, fact_id: str) -> str:
    names = [name for name in ("A", "B", "C", "D", "E") if name in values]
    return ", ".join(f"{name} 모델 {{{{{fact_id}.{name}}}}}" for name in names)


# ---------- 한 턴 ----------


def _clarify_bundle(plan: Plan) -> FactBundle:
    pairs: list[dict] = []
    given_line = plan.slots.get("line")
    for item in plan.slots.get("stations") or []:
        lines = [str(name) for name in (item.get("lines") or [])]
        if given_line:
            lines = [str(given_line)]
        if not lines:
            pairs.append({"line": "없음", "station": item.get("name", "없음")})
            continue
        for line_name in lines:
            pairs.append({"line": line_name, "station": item.get("name", "없음")})
    question = str(plan.slots.get("question") or "조건을 조금 더 알려 주세요.")
    values = {"question": question, "ambiguous": len(pairs)}
    for index, item in enumerate(pairs[:2], start=1):
        values[f"c{index}_line"] = item["line"]
        values[f"c{index}_station"] = item["station"]
    fact = verify_module.build_fact("F1", "clarification", values, source={"tool": "planner"})
    return FactBundle(
        bundle_id=f"b-{plan.plan_id}",
        as_of=None,
        facts=[fact],
        unavailable=[],
        policy={"mode": mode(), "intent": plan.intent, "clarify": True},
    )


def _finalize(plan: Plan, bundle: FactBundle, trace: dict) -> dict:
    draft = None
    if mode() == "llm":
        draft = _llm_attempt(bundle)
        if draft is None:
            trace["fallback"] = "template"
    if draft is None:
        draft = compose(plan, bundle)
    verdict = verify_module.verify(draft, bundle, attempt=1)
    if verdict.verdict == "repair":
        trace["repair"] = True
        verdict = verify_module.verify(draft, bundle, attempt=2)
    if verdict.verdict != "pass":
        trace["fallback"] = "facts_only"
        fallback_text = _facts_only_text(bundle, verdict)
        verdict = verify_module.verify_refusal(fallback_text)
    trace["verdict"] = verdict.verdict
    trace["checks"] = verdict.checks
    return {
        "answer": verdict.rendered_answer,
        "verdict": verdict.to_dict(),
        "draft": draft.to_dict(),
        "bundle": bundle.to_dict(),
    }


def _facts_only_text(bundle: FactBundle, verdict: Verdict) -> str:
    """검증을 통과하지 못했을 때 남기는 사실표. 숫자를 새로 만들지 않는다."""
    lines = ["검증을 통과한 문장이 없어 사실만 남깁니다."]
    for fact in bundle.facts:
        pairs = ", ".join(f"{key}={value}" for key, value in sorted((fact.get("display") or {}).items()))
        if pairs:
            lines.append(f"{fact['kind']}: {pairs}")
    if verdict.violations:
        codes = sorted({str(item.get("code")) for item in verdict.violations})
        lines.append("위반: " + ", ".join(codes))
    return " ".join(lines)


def run_turn(utterance: str, ui_context: dict | None = None) -> dict:
    """질문 한 턴을 끝까지 처리해 답·검증·근거·추적을 돌려준다."""
    started = perf_counter()
    plan = planner.build_plan(utterance, ui_context)
    current_mode = mode()
    trace: dict = {
        "mode": current_mode,
        "intent": plan.intent,
        "limits": {"max_steps": STEP_LIMIT, "max_calls": CALL_LIMIT,
                   "tool_timeout_ms": TOOL_TIMEOUT_MS,
                   "turn_deadline_ms": DEADLINE_MS.get(current_mode, DEADLINE_MS["template"])},
        "steps": [],
        "fallback": None,
        "repair": False,
    }

    if plan.refusal:
        verdict = verify_module.verify_refusal(plan.refusal)
        trace["refusal"] = True
        trace["verdict"] = "pass"
        latency = int((perf_counter() - started) * 1000)
        trace["latency_ms"] = latency
        return {
            "answer": plan.refusal,
            "verdict": verdict.to_dict(),
            "draft": {"answer_id": f"a-{plan.plan_id}", "claims": [], "caveats": []},
            "bundle": FactBundle(
                bundle_id=f"b-{plan.plan_id}",
                as_of=None,
                facts=[],
                unavailable=[],
                policy={"mode": current_mode, "intent": plan.intent, "refusal": True},
            ).to_dict(),
            "plan": plan.to_dict(),
            "mode": current_mode,
            "latency_ms": latency,
            "fallback": "refusal",
            "trace": trace,
        }

    if plan.needs_clarification:
        bundle = _clarify_bundle(plan)
        finalized = _finalize(plan, bundle, trace)
        latency = int((perf_counter() - started) * 1000)
        trace["latency_ms"] = latency
        return {
            **finalized,
            "plan": plan.to_dict(),
            "mode": current_mode,
            "latency_ms": latency,
            "fallback": "clarify",
            "trace": trace,
        }

    results, unavailable, step_trace = _execute(plan)
    trace["steps"] = step_trace
    trace["tool_calls"] = len(step_trace)
    policy = _policy(plan, results, unavailable)
    policy["data_end"] = next(
        (item["envelope"].as_of for item in results if item["envelope"].as_of), None
    )
    bundle = _bundle(plan, results, unavailable, policy)

    if not bundle.facts:
        reason = unavailable[0]["human"] if unavailable else "도구 결과가 비어 있습니다."
        text = f"계산할 수 없습니다: {reason}"
        verdict = verify_module.verify_refusal(text)
        trace["fallback"] = "no_facts"
        latency = int((perf_counter() - started) * 1000)
        trace["latency_ms"] = latency
        return {
            "answer": text,
            "verdict": verdict.to_dict(),
            "draft": {"answer_id": f"a-{plan.plan_id}", "claims": [], "caveats": []},
            "bundle": bundle.to_dict(),
            "plan": plan.to_dict(),
            "mode": current_mode,
            "latency_ms": latency,
            "fallback": "no_facts",
            "trace": trace,
        }

    finalized = _finalize(plan, bundle, trace)
    latency = int((perf_counter() - started) * 1000)
    trace["latency_ms"] = latency
    return {
        **finalized,
        "plan": plan.to_dict(),
        "mode": current_mode,
        "latency_ms": latency,
        "fallback": trace.get("fallback"),
        "trace": trace,
    }


__all__ = [
    "BREAKER",
    "CircuitBreaker",
    "compose",
    "compose_with_llm",
    "mode",
    "run_turn",
]


def _llm_attempt(bundle: FactBundle) -> DraftAnswer | None:
    """문장 조립 시도. 차단기가 열려 있으면 곧바로 템플릿으로 간다."""
    if not BREAKER.allow():
        return None
    try:
        draft = compose_with_llm(bundle)
    except LlmUnavailable:
        BREAKER.record_failure()
        return None
    BREAKER.record_success()
    return draft