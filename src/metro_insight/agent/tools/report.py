"""리포트 도구. 인자를 검증하고 :mod:`metro_insight.agent.reports` 에 쓰기를 맡긴다.

파일을 쓰는 유일한 도구이며 경로는 ``reports/<daily|weekly>/<YYYY-MM-DD>.md`` 로 고정된다.
"""
from __future__ import annotations

from time import perf_counter

from metro_insight.agent import reports as reports_module
from metro_insight.agent.contracts import Envelope, ErrorCode
from metro_insight.agent.tools._common import ToolError, as_of, inputs_hash, parse_date

KINDS = reports_module.KINDS
SECTIONS = reports_module.SECTIONS
DEFAULT_SECTIONS = reports_module.DEFAULT_SECTIONS


def report(kind: str, date: str, sections: list[str] | None = None) -> Envelope:
    """리포트 파일을 만든다(유일한 쓰기). 경로는 reports/ 아래로 고정된다."""
    started = perf_counter()
    try:
        name = str(kind).strip()
        if name not in KINDS:
            raise ToolError(
                ErrorCode.E_INVALID_ARG, f"kind 는 {KINDS} 중 하나여야 합니다: {kind}"
            )
        day = parse_date(date)
        reports_module.check_sections(sections)
        data = reports_module.write(name, day, sections)
        return Envelope.ok_(
            "report",
            data,
            as_of=as_of(),
            inputs_hash=inputs_hash(
                {"kind": name, "date": day.isoformat(), "sections": data["sections"]}
            ),
            elapsed_ms=int((perf_counter() - started) * 1000),
        )
    except ToolError as exc:
        return Envelope.error(
            "report", exc.code, exc.human, elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except OSError as exc:
        return Envelope.error(
            "report",
            ErrorCode.E_UNAVAILABLE,
            f"파일을 쓰지 못했습니다: {exc}",
            elapsed_ms=int((perf_counter() - started) * 1000),
        )