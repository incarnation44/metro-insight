"""문장이 근거 묶음에 묶여 있는지 검사한다. 숫자를 문장이 만들지 못하게 하는 장치다.

검사 항목
1. 플레이스홀더를 뺀 본문에 숫자 문자가 없어야 한다.
2. 플레이스홀더는 묶음에 있는 경로만 가리킬 수 있다.
3. 모든 주장은 fact_refs 를 1개 이상 가진다(없으면 그 주장을 버린다).
4. 표시 문자열은 같은 포매터로 다시 계산한 값과 같아야 한다.
5. 금지어('정확'·'보장'·'확실')를 쓰지 않고, 필요한 문구('일부 미검증'·'계산 불가'·'낡음')를
   조건에 맞게 넣는다.
6. 위반이 있으면 한 번 수리하고, 다시 실패하면 템플릿으로 되돌린다(호출부 담당).
7. 템플릿 문장은 숫자를 코드가 넣으므로 만들어 낸 수치가 없다.
"""
from __future__ import annotations

import re

from metro_insight.agent.contracts import DraftAnswer, FactBundle, Verdict

PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
DIGIT = re.compile(r"[0-9０-９]")
FORBIDDEN_WORDS = ("정확", "보장", "확실")

REQUIRED_PHRASES = {
    "future_partial": "일부 미검증",
    "de_missing": "계산 불가",
    "stale": "낡음",
}

_COUNT_KEYS = frozenset(
    {"A", "B", "C", "D", "E", "value", "station_sum", "line_sum", "top_value", "base",
     "model_value", "baseline", "last_value", "n"}
)
_DECIMAL_KEYS = frozenset({"mae", "rmse", "diff", "delta", "delta_pct", "z", "coverage"})
_HOUR_KEYS = frozenset({"hour"})
_DAYS_KEYS = frozenset({"days", "days_since_data_end"})
_INT_KEYS = frozenset(
    {"missing_count", "map_pairs", "map_only", "map_only_exact", "map_only_unresolved",
     "stations_with_data", "station_count", "count"}
)


def format_fact(key: str, value: object) -> str:
    """사실 하나를 표시 문자열로 바꾼다. 같은 값이면 언제나 같은 문자열."""
    if value is None:
        return "계산 불가"
    if key.startswith("delta_pct_"):
        try:
            return f"{float(value):,.2f}%"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "계산 불가"
    if key == "diff" or key.startswith("delta_"):
        try:
            return f"{float(value):,.2f}명"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "계산 불가"
    if key in _COUNT_KEYS:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "계산 불가"
        people = {
            "A",
            "B",
            "C",
            "D",
            "E",
            "value",
            "station_sum",
            "line_sum",
            "top_value",
            "base",
            "model_value",
            "baseline",
            "last_value",
        }
        shown = f"{int(round(number)):,}"
        return f"{shown}명" if key in people else shown
    if key in _DECIMAL_KEYS:
        try:
            return f"{float(value):,.2f}"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "계산 불가"
    if key in _HOUR_KEYS:
        try:
            return f"{int(value):02d}시"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "계산 불가"
    if key in _DAYS_KEYS:
        try:
            return f"{int(value)}일"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "계산 불가"
    if key in _INT_KEYS:
        try:
            return f"{int(value):,}"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "계산 불가"
    return str(value)


def build_fact(
    fact_id: str,
    kind: str,
    values: dict,
    *,
    source: dict | None = None,
    caveats: list[str] | None = None,
) -> dict:
    """사실 하나를 만든다. 표시 문자열은 전부 :func:`format_fact` 가 만든다."""
    return {
        "fact_id": fact_id,
        "kind": kind,
        "values": dict(values),
        "display": {key: format_fact(key, value) for key, value in values.items()},
        "caveats": list(caveats or []),
        "source": dict(source or {}),
    }


def _fact_index(bundle: FactBundle) -> dict[str, dict]:
    return {str(fact.get("fact_id")): fact for fact in bundle.facts}


def _resolve_placeholder(path: str, index: dict[str, dict]) -> tuple[bool, str, str | None]:
    """'F1.station' → (찾음, 표시 문자열, 문제 코드)."""
    fact_id, _, key = path.partition(".")
    fact = index.get(fact_id)
    if fact is None or not key:
        return False, "", "E_UNKNOWN_FACT"
    display = fact.get("display") or {}
    if key not in display:
        return False, "", "E_UNKNOWN_PLACEHOLDER"
    return True, str(display[key]), None


def _required_phrase_flags(policy: dict) -> dict[str, str]:
    wanted = {}
    for name, phrase in REQUIRED_PHRASES.items():
        if policy.get(name):
            wanted[name] = phrase
    return wanted


def verify(draft: DraftAnswer, bundle: FactBundle, *, attempt: int = 1) -> Verdict:
    """초안 검사. 통과하면 렌더 문장을, 아니면 위반 목록을 돌려준다."""
    index = _fact_index(bundle)
    violations: list[dict] = []
    checks = {
        "placeholder_binding": True,
        "digit_free": True,
        "fact_refs": True,
        "fact_match": True,
        "required_phrases": True,
        "forbidden_words": True,
    }
    rendered_claims: list[str] = []
    surviving = 0
    for position, claim in enumerate(draft.claims):
        text = str(claim.get("text", ""))
        where = f"claims[{position}]"
        refs = [str(ref) for ref in (claim.get("fact_refs") or [])]
        if not refs:
            checks["fact_refs"] = False
            violations.append({"code": "E_NO_FACT_REF", "severity": "block", "where": where})
            continue
        unknown_ref = [ref for ref in refs if ref not in index]
        if unknown_ref:
            checks["fact_refs"] = False
            violations.append(
                {"code": "E_UNKNOWN_FACT", "severity": "block", "where": where,
                 "detail": ",".join(unknown_ref)}
            )
            continue

        stripped = PLACEHOLDER.sub("", text)
        if DIGIT.search(stripped):
            checks["digit_free"] = False
            violations.append({"code": "E_NUM_IN_PROSE", "severity": "block", "where": where})
            continue

        rendered = text
        broken = False
        for match in PLACEHOLDER.finditer(text):
            path = match.group(1)
            found, shown, problem = _resolve_placeholder(path, index)
            if not found:
                checks["placeholder_binding"] = False
                violations.append(
                    {"code": problem, "severity": "block", "where": where, "detail": path}
                )
                broken = True
                break
            fact_id, _, key = path.partition(".")
            fact = index[fact_id]
            recomputed = format_fact(key, (fact.get("values") or {}).get(key))
            if recomputed != shown:
                checks["fact_match"] = False
                violations.append(
                    {"code": "E_FACT_MISMATCH", "severity": "block", "where": where, "detail": path}
                )
                broken = True
                break
            rendered = rendered.replace(match.group(0), recomputed, 1)
        if broken:
            continue

        for word in FORBIDDEN_WORDS:
            if word in rendered:
                checks["forbidden_words"] = False
                violations.append(
                    {"code": "E_FORBIDDEN_WORD", "severity": "block", "where": where, "detail": word}
                )
                broken = True
                break
        if broken:
            continue

        surviving += 1
        rendered_claims.append(rendered)

    caveat_texts = [str(item) for item in draft.caveats]
    body = " ".join(rendered_claims + caveat_texts)
    for phrase in _required_phrase_flags(bundle.policy).values():
        if phrase not in body:
            checks["required_phrases"] = False
            violations.append(
                {"code": "E_MISSING_PHRASE", "severity": "block", "where": "answer", "detail": phrase}
            )
    if DIGIT.search(" ".join(PLACEHOLDER.sub("", text) for text in caveat_texts)):
        checks["digit_free"] = False
        violations.append({"code": "E_NUM_IN_PROSE", "severity": "block", "where": "caveats"})
    for word in FORBIDDEN_WORDS:
        if word in body:
            checks["forbidden_words"] = False
            violations.append(
                {"code": "E_FORBIDDEN_WORD", "severity": "block", "where": "answer", "detail": word}
            )
            break

    if surviving == 0:
        return Verdict(
            verdict="reject",
            violations=violations or [{"code": "E_NO_CLAIMS", "severity": "block", "where": "answer"}],
            checks=checks,
            rendered_answer="",
        )
    if violations:
        return Verdict(
            verdict="repair" if attempt < 2 else "reject",
            violations=violations,
            checks=checks,
            rendered_answer=" ".join(rendered_claims + caveat_texts),
        )
    return Verdict(
        verdict="pass",
        violations=[],
        checks=checks,
        rendered_answer=" ".join(rendered_claims + caveat_texts),
    )


def verify_refusal(text: str) -> Verdict:
    """거부 문장 검사. 거부문은 코드가 쓴 정적 문장이라 숫자를 검사하지 않는다."""
    return Verdict(
        verdict="pass",
        violations=[],
        checks={"refusal": True, "digit_free": True, "fact_refs": True,
                "placeholder_binding": True, "fact_match": True, "required_phrases": True},
        rendered_answer=str(text),
    )