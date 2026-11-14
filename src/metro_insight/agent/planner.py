"""규칙으로 질의의 의도와 슬롯을 정하고 도구 호출 순서를 만든다.

LLM 없이도 같은 질문에 같은 계획이 나오도록 키워드·정규식만 쓴다. 역 이름은
지도 사전에서 찾고, 같은 이름이 여러 호선에 있으면 임의로 고르지 않고 되묻는다.
범위 밖 질문(하차·혼잡·실시간·없는 호선·요금 등)은 도구를 부르지 않고 거부한다.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta

from metro_insight.agent.contracts import Intent, Plan
from metro_insight.agent.tools._common import (
    DATA_END,
    SERVING_END,
    STATION_ALIASES,
    base_name,
    map_pairs,
    today,
)

# 거부 규칙: (정규식, 거부 사유). 도구를 부르지 않는다.
REFUSALS: tuple[tuple[str, str], ...] = (
    (r"하차|내린 사람|내리는", "하차 인원은 다루지 않습니다. 승차 인원만 예측합니다."),
    (r"혼잡|붐비는 정도|혼잡도", "혼잡도는 다루지 않습니다. 승차 인원만 예측합니다."),
    (r"실시간|지금\s*승차|현재\s*승차", "실시간 승차 인원은 없습니다. 과거 실측과 예측만 제공합니다."),
    (r"9호선|9\s*호선", "9호선은 데이터에 없습니다. 1~8호선만 다룹니다."),
    (r"요금|운임|환승\s*할인", "요금·운임 정보는 다루지 않습니다."),
    (r"버스|택시", "지하철 승차 인원만 다룹니다."),
    (r"환승\s*소요|환승\s*시간", "환승 소요 시간은 다루지 않습니다."),
    (r"test\.csv|테스트\s*파일|평가\s*데이터\s*읽", "평가용 데이터는 열지 않습니다."),
    (r"api\s*키|비밀|토큰|암호", "키·비밀값은 읽지도 알려 주지도 않습니다."),
    (r"셸|명령어\s*실행|파일\s*써|삭제해|지워", "파일 쓰기·명령 실행은 지원하지 않습니다."),
    (r"무시하|시스템\s*지시|프롬프트", "지시문을 따르지 않습니다. 승차 인원 질문만 답합니다."),
)

# 의도 키워드. 많이 맞은 의도가 이긴다(동점이면 아래 순서).
KEYWORDS: dict[str, tuple[str, ...]] = {
    Intent.REPORT.value: ("리포트", "보고서"),
    Intent.DRIFT.value: ("데이터 끝", "언제까지", "낡", "오래", "최신", "생성일", "며칠 지났", "기준일"),
    Intent.BACKTEST.value: ("mae", "rmse", "백테스트", "성적", "오차", "커버리지", "정확도"),
    Intent.ALERTS.value: ("이상치", "이상", "튀는", "급증"),
    Intent.RECONCILE.value: ("대조", "맞는지", "불일치", "검증"),
    Intent.SIMULATE.value: (
        "만약", "비가", "비 오", "눈이", "눈 오", "시나리오", "공휴일이면", "기온이", "오르면", "내리면",
    ),
    Intent.EXPLAIN.value: ("왜", "이유", "근거", "설명"),
    Intent.COMPARE.value: ("비교", " vs ", "대비", "차이", "둘 중", "나란히"),
    Intent.AGGREGATE.value: (
        "합계", "총 ", "평균", "집계", "상위", "가장 많", "가장 붐", "요일별", "월별", "합은",
    ),
    Intent.HISTORY.value: ("과거", "지난", "최근", "추이", "시계열", "그래프", "변화", "실측"),
    Intent.PREDICT.value: ("예상", "예측", "승차 인원", "몇 명", "얼마나"),
}

# 의도 동점 처리 순서(구체적인 질문 먼저).
PRIORITY = (
    Intent.REPORT.value,
    Intent.DRIFT.value,
    Intent.BACKTEST.value,
    Intent.ALERTS.value,
    Intent.RECONCILE.value,
    Intent.SIMULATE.value,
    Intent.EXPLAIN.value,
    Intent.COMPARE.value,
    Intent.AGGREGATE.value,
    Intent.HISTORY.value,
    Intent.PREDICT.value,
)

WEEKDAYS = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
MODEL_PATTERN = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")
LINE_PATTERN = re.compile(r"([1-9])\s*호선")
HOUR_PATTERN = re.compile(r"(오전|오후)?\s*(\d{1,2})\s*시")
CLOCK_PATTERN = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
ISO_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
SLASH_PATTERN = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)")
KOREAN_DATE_PATTERN = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일")
RELATIVE_DAYS = {"오늘": 0, "내일": 1, "모레": 2, "어제": -1, "그제": -2}

# 역 이름 사전은 한 번만 만든다(긴 이름 먼저 찾아 '강남구청'을 '강남'으로 자르지 않는다).
_STATIONS: list[tuple[str, tuple[str, ...]]] | None = None


def _station_table() -> list[tuple[str, tuple[str, ...]]]:
    global _STATIONS
    if _STATIONS is not None:
        return _STATIONS
    by_name: dict[str, list[str]] = {}
    for item in map_pairs():
        by_name.setdefault(item["station"], [])
        if item["line"] not in by_name[item["station"]]:
            by_name[item["station"]].append(item["line"])
    table = [(name, tuple(sorted(lines))) for name, lines in by_name.items()]
    table.sort(key=lambda item: (-len(item[0]), item[0]))
    _STATIONS = table
    return table


def classify(utterance: str) -> str:
    """질의 → 의도 12종 중 하나."""
    text = str(utterance).lower()
    scores: dict[str, int] = {}
    for intent, words in KEYWORDS.items():
        hits = sum(1 for word in words if word.lower() in text)
        if hits:
            scores[intent] = hits
    if not scores:
        return Intent.CLARIFY.value
    best = max(scores.values())
    for intent in PRIORITY:
        if scores.get(intent) == best:
            return intent
    return Intent.CLARIFY.value


YEAR_PATTERN = re.compile(r"(\d{4})\s*년")


def refusal_for(utterance: str) -> str | None:
    """범위 밖·주입 질의의 거부 사유. 해당 없으면 None."""
    text = str(utterance)
    for pattern, reason in REFUSALS:
        if re.search(pattern, text):
            return reason
    line = LINE_PATTERN.search(text)
    if line and int(line.group(1)) > 8:
        return "9호선 이상은 데이터에 없습니다. 1~8호선만 다룹니다."
    for match in ISO_PATTERN.finditer(text):
        if int(match.group(1)) > SERVING_END.year:
            return f"{SERVING_END.year}년 이후는 조회 범위 밖입니다(마지막 {SERVING_END.isoformat()})."
    for year_text in YEAR_PATTERN.findall(text):
        if int(year_text) > SERVING_END.year:
            return f"{SERVING_END.year}년 이후는 조회 범위 밖입니다(마지막 {SERVING_END.isoformat()})."
    return None


def find_stations(utterance: str) -> list[dict]:
    """발화에 나온 역 이름 후보(긴 이름 우선, 중복 제거)."""
    text = str(utterance)
    found: list[dict] = []
    seen: set[str] = set()
    for name, lines in _station_table():
        if name in seen:
            continue
        if name in text or f"{name}역" in text:
            seen.add(name)
            found.append({"name": name, "lines": list(lines)})
    # 괄호 별칭('이수' → '총신대입구(이수)')과 기본 이름도 확인한다.
    for name, lines in _station_table():
        short = base_name(name)
        if name in seen or short == name or short in seen:
            continue
        if short and short in text:
            seen.add(name)
            found.append({"name": name, "lines": list(lines), "alias": True})
    # 지도 표기와 전혀 다른 통칭('이수' → '총신대입구(이수)')도 후보로 올린다.
    for alias, target in STATION_ALIASES.items():
        if alias in text and target not in seen:
            for name, lines in _station_table():
                if name == target:
                    seen.add(name)
                    found.append({"name": name, "lines": list(lines), "alias": True})
                    break
    # '이수'처럼 지도 표기와 다른 이름은 지도 표기의 기본 이름으로 잡힌다.
    return found


def extract_date(utterance: str, reference: date | None = None) -> str | None:
    """발화 → 날짜(ISO). 없으면 None. 연도가 없으면 기준일 가까운 해로 정한다."""
    text = str(utterance)
    iso_match = ISO_PATTERN.search(text)
    if iso_match:
        return f"{iso_match.group(1)}-{iso_match.group(2)}-{iso_match.group(3)}"
    base = reference or today()
    for word, offset in RELATIVE_DAYS.items():
        if word in text:
            return (base + timedelta(days=offset)).isoformat()
    for pattern in (KOREAN_DATE_PATTERN, SLASH_PATTERN):
        match = pattern.search(text)
        if not match:
            continue
        month, day = int(match.group(1)), int(match.group(2))
        year = base.year
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None
        if candidate < base - timedelta(days=365):
            try:
                candidate = date(year + 1, month, day)
            except ValueError:
                return None
        return candidate.isoformat()
    return None


def extract_hour(utterance: str) -> int | None:
    """발화 → 시간대 시작 시(0~24). '오전/오후'를 반영한다."""
    text = str(utterance)
    clock = CLOCK_PATTERN.search(text)
    if clock:
        hour = int(clock.group(1))
        if 0 <= hour <= 24:
            return hour
    match = HOUR_PATTERN.search(text)
    if not match:
        return None
    meridiem, raw = match.group(1), int(match.group(2))
    if meridiem == "오후" and raw < 12:
        raw += 12
    if meridiem == "오전" and raw == 12:
        raw = 0
    return raw if 0 <= raw <= 24 else None


def extract_weekday(utterance: str) -> int | None:
    for word, value in WEEKDAYS.items():
        if f"{word}요일" in str(utterance):
            return value
    return None


def extract_models(utterance: str) -> list[str] | None:
    found = MODEL_PATTERN.findall(str(utterance))
    if not found:
        return None
    ordered: list[str] = []
    for name in found:
        if name not in ordered:
            ordered.append(name)
    return ordered


MONTH_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*월")


def extract_scenarios(utterance: str) -> list[dict]:
    """발화 → 시뮬레이션 시나리오(비·눈·기온). 최대 3개."""
    text = str(utterance)
    scenarios: list[dict] = []
    if "비" in text:
        scenarios.append({"name": "비", "rainfall": True})
    if "눈" in text:
        scenarios.append({"name": "눈", "rainfall": True, "snowfall": True})
    temperature = re.search(r"기온[이가]?\s*([+-]?\d+)\s*도", text)
    if temperature:
        delta = int(temperature.group(1))
        if "내리" in text:
            delta = -abs(delta)
        elif "오르" in text:
            delta = abs(delta)
        scenarios.append({"name": f"기온 {delta}도", "temperature_delta": delta})
    return scenarios[:3]


def _plan_id(utterance: str, intent: str) -> str:
    digest = hashlib.sha256(f"{intent}:{utterance}".encode()).hexdigest()[:8]
    return f"p-{digest}"


def _resolve_step(step_id: str, station: str) -> dict:
    return {"step_id": step_id, "tool": "resolve_station", "args": {"query": station, "limit": 5}}


def _slot_question(line: str | None, day: str | None, hour: int | None) -> str:
    """빠진 슬롯만 짚는 되묻기 문장. 없는 값을 지어내지 않는다."""
    labels = []
    if line is None:
        labels.append("호선")
    if day is None:
        labels.append("날짜")
    if hour is None:
        labels.append("시각")
    if not labels:
        return "조건을 조금 더 알려 주세요."
    if len(labels) == 1:
        return f"{labels[0]} 정보가 필요합니다."
    return f"{'·'.join(labels)} 정보가 필요합니다."


def _default_range(reference: date) -> tuple[str, str]:
    end = min(reference, DATA_END)
    start = max(end - timedelta(days=29), date(2024, 1, 8))
    return start.isoformat(), end.isoformat()


WEEK_PATTERN = re.compile(r"(\d{1,2})\s*주")
MONTH_COUNT_PATTERN = re.compile(r"(\d{1,2})\s*개월")


def _history_range(text: str, reference: date) -> tuple[str, str]:
    """발화의 기간('2주'·'3개월'·'한 달') → 실측 조회 구간. 없으면 최근 30일."""
    end = min(reference, DATA_END)
    days = 30
    week = WEEK_PATTERN.search(text)
    months = MONTH_COUNT_PATTERN.search(text)
    if week:
        days = 7 * int(week.group(1))
    elif months:
        days = 30 * int(months.group(1))
    days = max(1, min(days, 366))
    start = max(end - timedelta(days=days - 1), date(2024, 1, 8))
    return start.isoformat(), end.isoformat()


def _month_range(reference: date) -> tuple[str, str]:
    first = reference.replace(day=1)
    if first > DATA_END:
        return _default_range(reference)
    return first.isoformat(), min(reference, DATA_END).isoformat()


def _month_span(text: str, reference: date) -> tuple[str, str] | None:
    """발화의 'N월'·'지난달'·'이번 달' → 그 달의 조회 구간(데이터 끝까지 자른다)."""
    anchor = min(reference, DATA_END)
    if "지난달" in text:
        last = anchor.replace(day=1) - timedelta(days=1)
        year, month = last.year, last.month
    elif "이번 달" in text:
        return _month_range(reference)
    else:
        match = MONTH_PATTERN.search(text)
        if not match:
            return None
        month = int(match.group(1))
        if not 1 <= month <= 12:
            return None
        year = anchor.year if month <= anchor.month else anchor.year - 1
    start = date(year, month, 1)
    end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    if end > DATA_END:
        end = DATA_END
    if start > DATA_END:
        return None
    return start.isoformat(), end.isoformat()


def build_plan(utterance: str, ui_context: dict | None = None) -> Plan:
    """발화 → 계획. 범위 밖이면 거부, 역이 모호하면 되묻는다."""
    text = str(utterance).strip()
    reference = today()
    if not text:
        return Plan(
            plan_id=_plan_id("", Intent.CLARIFY.value),
            intent=Intent.CLARIFY.value,
            slots={"question": "무엇을 알고 싶은지 적어 주세요."},
            steps=[],
            needs_clarification=True,
        )

    reason = refusal_for(text)
    if reason is not None:
        return Plan(
            plan_id=_plan_id(text, Intent.CLARIFY.value),
            intent=Intent.CLARIFY.value,
            slots={"question": text},
            steps=[],
            needs_clarification=False,
            refusal=reason,
        )

    intent = classify(text)
    stations = find_stations(text)
    line_match = LINE_PATTERN.search(text)
    line = f"{line_match.group(1)}호선" if line_match else None
    day = extract_date(text, reference)
    hour = extract_hour(text)
    weekday = extract_weekday(text)
    models = extract_models(text)

    # 호선이 지정되면 그 호선의 후보만 남긴다. 지정이 없으면 (역, 호선) 쌍이 여럿일 수 있다.
    if line is not None:
        stations = [item for item in stations if line in item["lines"]]
    pair_candidates = [
        (item["name"], name_line)
        for item in stations
        for name_line in (item["lines"] if line is None else [line])
    ]
    needs_clarification = False
    question = None
    if line is None and len(pair_candidates) > 1:
        needs_clarification = True
        question = "어느 호선인가요?"
    station = stations[0]["name"] if stations else None
    if line is None and stations and not needs_clarification:
        line = stations[0]["lines"][0]

    slots: dict = {"line": line, "station": station, "date": day, "hour": hour}
    if models:
        slots["models"] = models
    if weekday is not None:
        slots["weekday"] = weekday
    if ui_context:
        slots["ui_context"] = dict(ui_context)

    steps: list[dict] = []
    if intent in (
        Intent.PREDICT.value,
        Intent.COMPARE.value,
        Intent.HISTORY.value,
        Intent.EXPLAIN.value,
        Intent.SIMULATE.value,
    ):
        if station is None:
            needs_clarification = True
            question = question or "역 이름을 알려 주세요."
        else:
            steps.append(_resolve_step("s1", station))

    if intent == Intent.PREDICT.value:
        slots.update({"line": line, "station": station, "date": day})
        if not needs_clarification:
            if line is None or day is None or hour is None:
                needs_clarification = True
                question = question or _slot_question(line, day, hour)
            else:
                steps.append(
                    {
                        "step_id": "s2",
                        "tool": "predict",
                        "args": {"line": line, "station": station, "date": day, "hour": hour},
                        "depends_on": ["s1"],
                    }
                )
    elif intent == Intent.COMPARE.value:
        slots.update({"stations": stations, "line": line, "station": station, "date": day})
        if not needs_clarification:
            if line is None or day is None or hour is None:
                needs_clarification = True
                question = question or _slot_question(line, day, hour)
            else:
                args = {"line": line, "station": station, "date": day, "hour": hour}
                if models:
                    args["models"] = models
                steps.append({"step_id": "s2", "tool": "compare", "args": args, "depends_on": ["s1"]})
    elif intent == Intent.HISTORY.value:
        date_from, date_to = _history_range(text, reference)
        if weekday is not None:
            date_from, date_to = "2024-01-08", min(reference, DATA_END).isoformat()
        slots.update({"line": line, "station": station, "date_from": date_from, "date_to": date_to})
        if not needs_clarification:
            if line is None:
                needs_clarification = True
                question = question or "역 이름을 알려 주세요."
            else:
                args = {
                    "line": line,
                    "station": station,
                    "date_from": date_from,
                    "date_to": date_to,
                }
                if hour is not None:
                    args["hour"] = hour
                steps.append({"step_id": "s2", "tool": "history", "args": args, "depends_on": ["s1"]})
    elif intent == Intent.AGGREGATE.value:
        span = _month_span(text, reference)
        date_from, date_to = span or _default_range(reference)
        scope: dict = {}
        if line is not None:
            scope["line"] = line
        if station is not None and line is not None:
            scope["station"] = station
        if hour is not None:
            scope["hour"] = hour
        if weekday is not None:
            scope["weekday"] = weekday
        if re.search(r"상위|가장 많|가장 붐|top", text):
            scope["top"] = 5
        slots.update({"date_from": date_from, "date_to": date_to})
        args = {
            "scope": scope,
            "range": {"date_from": date_from, "date_to": date_to},
            "metric": "mean" if "평균" in text else "sum",
        }
        steps.append({"step_id": "s1", "tool": "aggregate", "args": args})
    elif intent == Intent.SIMULATE.value:
        scenarios = extract_scenarios(text)
        slots.update({"line": line, "station": station, "scenario": scenarios})
        if not needs_clarification:
            if line is None or day is None or hour is None or not scenarios:
                needs_clarification = True
                question = question or _slot_question(line, day, hour)
            else:
                steps.append(
                    {
                        "step_id": "s2",
                        "tool": "simulate",
                        "args": {
                            "base": {"line": line, "station": station, "date": day, "hour": hour},
                            "scenarios": scenarios,
                        },
                        "depends_on": ["s1"],
                    }
                )
    elif intent == Intent.EXPLAIN.value:
        model = (models or ["C"])[0]
        target = f"{line or ''} {station or ''} {day or ''} {hour or ''} {model}".strip()
        slots.update({"target": target, "model": model})
        if not needs_clarification:
            if line is None or day is None or hour is None:
                needs_clarification = True
                question = question or _slot_question(line, day, hour)
            else:
                steps.append(
                    {
                        "step_id": "s2",
                        "tool": "explain",
                        "args": {"line": line, "station": station, "date": day, "hour": hour, "model": model},
                        "depends_on": ["s1"],
                    }
                )
    elif intent == Intent.RECONCILE.value:
        slots.update({"claim": {"line": line, "date": day, "hour": hour}})
        if line is None or day is None or hour is None:
            needs_clarification = True
            question = question or "호선·날짜·시각을 알려 주세요."
        else:
            steps.append(
                {"step_id": "s1", "tool": "reconcile", "args": {"line": line, "date": day, "hour": hour}}
            )
    elif intent == Intent.BACKTEST.value:
        bucket = None
        for name in ("1-31", "32-91", "92-181", "182-366", "367+"):
            if name in text or name.replace("-", "~") in text:
                bucket = name
                break
        slice_name = None
        for name in ("소", "중", "대"):
            if f"{name} 규모" in text or f"tertile:{name}" in text:
                slice_name = f"tertile:{name}"
                break
        slots.update({"horizon": bucket or "overall"})
        args: dict = {}
        if bucket:
            args["bucket"] = bucket
        if models:
            args["model"] = models[0]
        if slice_name:
            args["slice"] = slice_name
        steps.append({"step_id": "s1", "tool": "backtest", "args": args})
    elif intent == Intent.DRIFT.value:
        slots.update({"window": "all"})
        steps.append({"step_id": "s1", "tool": "drift", "args": {}})
    elif intent == Intent.ALERTS.value:
        span = _month_span(text, reference)
        date_from, date_to = span or _default_range(reference)
        scope: dict = {}
        if line is not None:
            scope["line"] = line
        if station is not None and line is not None:
            scope["station"] = station
        if hour is not None:
            scope["hour"] = hour
        z = 3.0
        z_match = re.search(r"(\d+(?:\.\d+)?)\s*시그마", text)
        if z_match:
            z = float(z_match.group(1))
        slots.update({"threshold": z})
        steps.append(
            {
                "step_id": "s1",
                "tool": "alerts",
                "args": {
                    "scope": scope,
                    "range": {"date_from": date_from, "date_to": date_to},
                    "z": z,
                },
            }
        )
    elif intent == Intent.REPORT.value:
        kind = "weekly" if "주간" in text else "daily"
        report_date = day or min(reference, SERVING_END).isoformat()
        sections = ["summary", "drift", "caveats"]
        if "성적" in text or "백테스트" in text:
            sections.append("backtest")
        if "이상" in text:
            sections.append("alerts")
        slots.update({"title": f"{kind} {report_date}"})
        steps.append(
            {
                "step_id": "s1",
                "tool": "report",
                "args": {"kind": kind, "date": report_date, "sections": sections},
            }
        )

    if not steps and not needs_clarification and intent not in (Intent.CLARIFY.value,):
        needs_clarification = True
        question = question or "질문을 이해하지 못했습니다. 역·날짜·시각을 함께 적어 주세요."
    if intent == Intent.CLARIFY.value:
        needs_clarification = True
        question = question or "질문을 이해하지 못했습니다. 역·날짜·시각을 함께 적어 주세요."
        slots.setdefault("question", question)

    plan_slots = {key: value for key, value in slots.items() if value is not None}
    if needs_clarification:
        plan_slots["question"] = question or "조건을 조금 더 알려 주세요."
        plan_slots.setdefault("line", line)
        plan_slots.setdefault("station", station)
        plan_slots.setdefault("date", day)
        plan_slots.setdefault("date_from", _default_range(reference)[0])
        plan_slots.setdefault("date_to", _default_range(reference)[1])
        plan_slots.setdefault("stations", stations)
        plan_slots.setdefault("scenario", [])
        plan_slots.setdefault("target", station or text)
        plan_slots.setdefault("claim", {})
        plan_slots.setdefault("horizon", "overall")
        plan_slots.setdefault("window", "all")
        plan_slots.setdefault("threshold", 3.0)
        plan_slots.setdefault("title", text)
        plan_slots["line"] = plan_slots.get("line")
        plan_slots["station"] = plan_slots.get("station")
        plan_slots["date"] = plan_slots.get("date")
        if plan_slots["line"] is None:
            plan_slots["line"] = ""
        if plan_slots["station"] is None:
            plan_slots["station"] = ""
        if plan_slots["date"] is None:
            plan_slots["date"] = ""

    return Plan(
        plan_id=_plan_id(text, intent),
        intent=intent,
        slots=plan_slots,
        steps=steps,
        needs_clarification=needs_clarification,
        refusal=None,
    )