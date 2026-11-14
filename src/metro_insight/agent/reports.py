"""리포트 파일 생성. 에이전트가 파일을 쓰는 유일한 경로다.

허용 범위는 ``reports/`` 아래뿐이고 파일 이름은 ``<daily|weekly>/<YYYY-MM-DD>.md`` 로
고정한다(사용자 문자열을 경로에 넣지 않는다). 내용은 도구가 계산한 값만 담고 시각·난수를
넣지 않으므로 같은 날짜로 두 번 만들어도 같은 바이트가 나온다.

``report`` 도구는 인자를 검증한 뒤 이 모듈의 :func:`write` 만 호출한다.
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path

from metro_insight.agent.contracts import ErrorCode
from metro_insight.agent.tools._common import (
    DATA_END,
    SERVING_END,
    ToolError,
    clean_float,
    iso,
    repo_root,
    reports_root,
)

KINDS = ("daily", "weekly")
SECTIONS = ("summary", "drift", "backtest", "alerts", "caveats")
DEFAULT_SECTIONS = ("summary", "drift", "caveats")


def check_sections(sections: object) -> list[str]:
    """섹션 이름 검증·중복 제거. 모르는 이름은 E_INVALID_ARG."""
    if sections is None:
        return list(DEFAULT_SECTIONS)
    if not isinstance(sections, (list, tuple)):
        raise ToolError(ErrorCode.E_INVALID_ARG, "sections 는 배열이어야 합니다")
    out: list[str] = []
    for raw in sections:
        name = str(raw).strip()
        if name not in SECTIONS:
            raise ToolError(
                ErrorCode.E_INVALID_ARG,
                f"모르는 섹션입니다: {raw} (가능: {', '.join(SECTIONS)})",
            )
        if name not in out:
            out.append(name)
    if not out:
        raise ToolError(ErrorCode.E_INVALID_ARG, "sections 가 비어 있습니다")
    return out


def target_path(kind: str, day: date) -> Path:
    """``reports/<kind>/<date>.md`` 절대 경로. 범위 밖이면 E_INVALID_ARG."""
    root = reports_root().resolve()
    directory = (root / str(kind)).resolve()
    path = (directory / f"{day.isoformat()}.md").resolve()
    if not path.is_relative_to(root) or path.parent != directory:
        raise ToolError(
            ErrorCode.E_INVALID_ARG, "리포트 경로가 허용 범위(reports/)를 벗어났습니다."
        )
    return path


def _summary_lines() -> list[str]:
    from metro_insight.agent.tools.drift import drift_facts

    drift_data, _ = drift_facts()
    stale = "낡음" if drift_data["stale"] else "최신"
    return [
        "## 요약",
        f"- 데이터 기준일: {drift_data['data_end']} "
        f"(경과 {drift_data['days_since_data_end']}일, {stale})",
        f"- 조회 가능 마지막 날짜: {SERVING_END.isoformat()}",
        f"- 지도 (호선, 역) {drift_data['map_pairs']}쌍 중 예측 데이터 없는 쌍 "
        f"{drift_data['map_only_unresolved']}개",
    ]


def _drift_lines() -> list[str]:
    from metro_insight.agent.tools.drift import drift_facts

    drift_data, _ = drift_facts()
    lines = ["## 아티팩트", "| 파일 | 생성일 | 수정 시각 |", "| --- | --- | --- |"]
    for item in drift_data["artifacts"]:
        lines.append(f"| {item['name']} | {item['generated'] or '-'} | {item['mtime'] or '-'} |")
    return lines


def _backtest_lines() -> list[str]:
    from metro_insight.agent.tools.backtest import backtest

    result = backtest()
    if not result.ok:
        return ["## 성적", f"- 계산하지 못했습니다: {result.data['human']}"]
    lines = ["## 성적", "| 모델 | MAE | RMSE | 표본 |", "| --- | --- | --- | --- |"]
    for row in result.data["rows"]:
        mae = clean_float(row["mae"])
        rmse = clean_float(row["rmse"])
        lines.append(
            f"| {row['model']} | {mae if mae is not None else '-'} | "
            f"{rmse if rmse is not None else '-'} | {row['n'] or '-'} |"
        )
    return lines


def _alert_lines() -> list[str]:
    from metro_insight.agent.tools.alerts import alerts

    start = DATA_END - timedelta(days=30)
    result = alerts({}, {"date_from": iso(start), "date_to": DATA_END.isoformat()}, 3.0)
    if not result.ok:
        return ["## 이상치", f"- 계산하지 못했습니다: {result.data['human']}"]
    rows = result.data["alerts"]
    lines = ["## 이상치(과거 기준)"]
    if not rows:
        lines.append("- 기준을 넘는 날이 없습니다.")
    for item in rows:
        lines.append(f"- {item['date']}: {item['value']} (z {item['z']})")
    return lines


def _caveat_lines() -> list[str]:
    from metro_insight.agent.tools.drift import drift_facts

    drift_data, _ = drift_facts()
    lines = ["## 한계"]
    if drift_data["stale"]:
        lines.append(
            f"- 데이터 기준일이 {drift_data['days_since_data_end']}일 전이라 낡음 상태입니다."
        )
    lines.append("- D/E 모델은 대상일 전 실측 창이 없으면 값을 제공하지 않습니다(0 아님).")
    lines.append("- 이 리포트의 수치는 모두 도구 계산 결과입니다.")
    return lines


_BLOCKS = {
    "summary": _summary_lines,
    "drift": _drift_lines,
    "backtest": _backtest_lines,
    "alerts": _alert_lines,
    "caveats": _caveat_lines,
}


def render(kind: str, day: date, sections: list[str]) -> str:
    """리포트 본문. 새 값을 만들지 않고 섹션 함수 결과만 이어 붙인다."""
    title = "일일 리포트" if kind == "daily" else "주간 리포트"
    lines = [f"# {title} — {day.isoformat()}", ""]
    for section in sections:
        lines.extend(_BLOCKS[section]())
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def write(kind: str, day: date, sections: list[str] | None = None) -> dict:
    """(경로, sha256, 바이트, 섹션) — 파일을 쓴다. 실패는 OSError 로 올린다."""
    chosen = check_sections(sections)
    path = target_path(kind, day)
    content = render(kind, day, chosen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    try:
        shown = path.relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        shown = path.as_posix()
    body = content.encode("utf-8")
    return {
        "kind": kind,
        "date": day.isoformat(),
        "path": shown,
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
        "sections": chosen,
    }