"""과거 실측 시계열. 없는 날은 0 이 아니라 null 로 남기고 결측 날짜를 함께 알린다.

계산은 :class:`~metro_insight.model.analytics.Analytics` 가 예측기의 이력 배열을
읽어서 한다(전체 스캔 없이 필요한 행·시간대·날짜만 자른다). 미래 날짜는 실측이
없으므로 조회 범위를 넘겼다고 알린다.
"""
from __future__ import annotations

from time import perf_counter

from metro_insight.agent.contracts import Caveat, Envelope, ErrorCode
from metro_insight.agent.tools._common import (
    DATA_END,
    ToolError,
    analytics,
    as_of,
    caveat,
    check_hour,
    check_line,
    check_station,
    clean_float,
    inputs_hash,
    parse_date,
)

MAX_POINTS = 400


def history(
    line: str, station: str, date_from: str, date_to: str, hour: int | None = None
) -> Envelope:
    """과거 실측 시계열(결측일 포함). ``hour`` 가 없으면 하루 합계."""
    started = perf_counter()
    try:
        line = check_line(line)
        station = check_station(line, station)
        start = parse_date(date_from)
        end = parse_date(date_to)
        if start > end:
            raise ToolError(
                ErrorCode.E_INVALID_ARG, f"시작일({start}) 이 종료일({end}) 보다 늦습니다."
            )
        if end > DATA_END:
            raise ToolError(
                ErrorCode.E_OUT_OF_RANGE,
                f"실측 시계열은 {DATA_END.isoformat()}까지만 있습니다"
                f"(가장 가까운 날짜 {DATA_END.isoformat()}).",
            )
        window_hour = None if hour is None else check_hour(hour)

        dates, values = analytics().series(line, station, window_hour)
        points = []
        for day, value in zip(dates, values, strict=True):
            if not start.isoformat() <= day <= end.isoformat():
                continue
            number = clean_float(value)
            points.append({"date": day, "value": number})
        if len(points) > MAX_POINTS:
            points = points[-MAX_POINTS:]
        missing = [item["date"] for item in points if item["value"] is None]
        valid = len(points) - len(missing)
        hints = []
        if missing:
            hints.append(
                caveat(Caveat.DATA_MISSING, f"{len(missing)}일은 관측이 없어 0 대신 비웠습니다.")
            )
        return Envelope.ok_(
            "history",
            {
                "line": line,
                "station": station,
                "hour": window_hour,
                "date_from": start.isoformat(),
                "date_to": end.isoformat(),
                "points": points,
                "n": valid,
                "coverage": round(valid / len(points), 4) if points else 0.0,
                "missing_dates": missing,
                "missing_count": len(missing),
            },
            as_of=as_of(),
            inputs_hash=inputs_hash(
                {
                    "line": line,
                    "station": station,
                    "date_from": start.isoformat(),
                    "date_to": end.isoformat(),
                    "hour": window_hour,
                }
            ),
            elapsed_ms=int((perf_counter() - started) * 1000),
            caveats=hints,
        )
    except ToolError as exc:
        return Envelope.error(
            "history", exc.code, exc.human, elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except Exception as exc:  # noqa: BLE001 — 예기치 못한 실패도 사유를 남긴다
        return Envelope.error(
            "history",
            ErrorCode.E_INTERNAL,
            f"시계열 계산 오류: {type(exc).__name__}: {exc}",
            elapsed_ms=int((perf_counter() - started) * 1000),
        )