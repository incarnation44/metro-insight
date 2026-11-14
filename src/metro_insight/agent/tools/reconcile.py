"""그 시점의 호선 전체 합계를 역별 실측의 합과 대조한다.

역별 값을 더한 합과 집계 경로가 낸 합이 같은지 확인하고, 관측이 없는 역을 함께
알린다. 이 합계는 역별 실측의 합이며 공식 호선 합계가 아니라는 점을 값과 같이 남긴다.
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
    clean_float,
    inputs_hash,
    parse_date,
    round2,
    stations_of_line,
)


def reconcile(line: str, date: str, hour: int) -> Envelope:
    """역 합 vs 집계 합 대조 + 관측 없는 역 목록."""
    started = perf_counter()
    try:
        line = check_line(line)
        day = parse_date(date)
        hour = check_hour(hour)
        if day > DATA_END:
            raise ToolError(
                ErrorCode.E_UNAVAILABLE,
                f"{day.isoformat()} 는 데이터 끝({DATA_END.isoformat()}) 이후라 "
                "실측이 없어 대조할 수 없습니다.",
            )
        engine = analytics()
        total = engine.aggregate(
            line=line,
            hour=hour,
            date_from=day.isoformat(),
            date_to=day.isoformat(),
            metric="sum",
        )
        total_value = clean_float(total["value"])

        station_values = []
        missing = []
        for station in stations_of_line(line):
            dates, values = engine.series(line, station, hour)
            value = None
            for item_date, item_value in zip(dates, values, strict=True):
                if item_date == day.isoformat():
                    value = clean_float(item_value)
                    break
            if value is None:
                missing.append(station)
            else:
                station_values.append(value)
        station_sum = sum(station_values) if station_values else None
        diff = (
            round2(station_sum - total_value)
            if station_sum is not None and total_value is not None
            else None
        )
        matches = diff is not None and abs(diff) < 0.5
        hints = []
        if missing:
            hints.append(
                caveat(Caveat.DATA_MISSING, f"{len(missing)}개 역은 그 시각 관측이 없습니다.")
            )
        return Envelope.ok_(
            "reconcile",
            {
                "line": line,
                "date": day.isoformat(),
                "hour": hour,
                "line_sum": round2(total_value),
                "station_sum": round2(station_sum),
                "diff": diff,
                "consistency": {"matches": matches},
                "station_count": len(stations_of_line(line)),
                "stations_with_data": len(station_values),
                "missing_stations": missing,
                "coverage": round(float(total["coverage"]), 4),
                "basis": "past_actuals",
                "note": "역별 실측의 합이며 공식 호선 합계가 아닙니다.",
            },
            as_of=as_of(),
            inputs_hash=inputs_hash({"line": line, "date": date, "hour": hour}),
            elapsed_ms=int((perf_counter() - started) * 1000),
            caveats=hints,
        )
    except ToolError as exc:
        return Envelope.error(
            "reconcile", exc.code, exc.human, elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except Exception as exc:  # noqa: BLE001 — 예기치 못한 실패도 사유를 남긴다
        return Envelope.error(
            "reconcile",
            ErrorCode.E_INTERNAL,
            f"대조 중 오류: {type(exc).__name__}: {exc}",
            elapsed_ms=int((perf_counter() - started) * 1000),
        )