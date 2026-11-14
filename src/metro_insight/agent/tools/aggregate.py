"""scope(호선·역·시간대·요일) × range(날짜 구간) 집계.

값과 함께 유효값 수 ``n``·커버리지 ``coverage`` 를 돌려준다. 관측이 없으면 값은
``null`` 이고 0 이 아니다. ``top`` 을 주면 그 호선에서 값이 큰 역 순으로 함께 담는다.
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
    round2,
    stations_of_line,
)

_METRICS = ("sum", "mean", "max")
_SCOPE_KEYS = ("line", "station", "hour", "weekday", "top")
_RANGE_KEYS = ("date_from", "date_to")
MAX_TOP = 10


def _check_scope(scope: dict) -> dict:
    if not isinstance(scope, dict):
        raise ToolError(ErrorCode.E_INVALID_ARG, "scope must be an object")
    unknown = sorted(set(scope) - set(_SCOPE_KEYS))
    if unknown:
        raise ToolError(ErrorCode.E_INVALID_ARG, f"scope 에 모르는 키가 있습니다: {', '.join(unknown)}")
    out: dict = {}
    if scope.get("line") not in (None, ""):
        out["line"] = check_line(scope["line"])
    if scope.get("station") not in (None, ""):
        line = out.get("line")
        if line is None:
            raise ToolError(ErrorCode.E_INVALID_ARG, "station 을 쓰려면 line 이 필요합니다")
        out["station"] = check_station(line, scope["station"])
    if scope.get("hour") is not None:
        out["hour"] = check_hour(scope["hour"])
    if scope.get("weekday") is not None:
        try:
            weekday = int(scope["weekday"])
        except (TypeError, ValueError) as exc:
            raise ToolError(
                ErrorCode.E_INVALID_ARG, f"weekday 는 정수여야 합니다: {scope['weekday']}"
            ) from exc
        if not 0 <= weekday <= 6:
            raise ToolError(ErrorCode.E_INVALID_ARG, f"weekday 는 0~6 사이여야 합니다: {scope['weekday']}")
        out["weekday"] = weekday
    if scope.get("top") is not None:
        try:
            top = int(scope["top"])
        except (TypeError, ValueError) as exc:
            raise ToolError(ErrorCode.E_INVALID_ARG, f"top 은 정수여야 합니다: {scope['top']}") from exc
        if not 1 <= top <= MAX_TOP:
            raise ToolError(ErrorCode.E_INVALID_ARG, f"top 은 1~{MAX_TOP} 사이여야 합니다: {scope['top']}")
        if "line" not in out:
            raise ToolError(ErrorCode.E_INVALID_ARG, "top 을 쓰려면 line 이 필요합니다")
        out["top"] = top
    return out


def _check_range(range_: dict) -> tuple[str, str]:
    if not isinstance(range_, dict):
        raise ToolError(ErrorCode.E_INVALID_ARG, "range must be an object")
    unknown = sorted(set(range_) - set(_RANGE_KEYS))
    if unknown:
        raise ToolError(ErrorCode.E_INVALID_ARG, f"range 에 모르는 키가 있습니다: {', '.join(unknown)}")
    start = parse_date(range_.get("date_from"))
    end = parse_date(range_.get("date_to"))
    if start > end:
        raise ToolError(ErrorCode.E_INVALID_ARG, f"시작일({start}) 이 종료일({end}) 보다 늦습니다.")
    if start > DATA_END:
        raise ToolError(
            ErrorCode.E_OUT_OF_RANGE,
            f"집계 가능 범위는 {DATA_END.isoformat()}까지입니다(가장 가까운 날짜 {DATA_END.isoformat()}).",
        )
    if end > DATA_END:
        end = DATA_END
    return start.isoformat(), end.isoformat()


def _top_stations(scope: dict, date_from: str, date_to: str, metric: str, limit: int) -> list[dict]:
    """그 호선에서 값이 큰 역 순(같은 조건). 표본이 없는 역은 넣지 않는다."""
    engine = analytics()
    rows = []
    for station in stations_of_line(scope["line"]):
        result = engine.aggregate(
            line=scope["line"],
            station=station,
            hour=scope.get("hour"),
            weekday=scope.get("weekday"),
            date_from=date_from,
            date_to=date_to,
            metric=metric,
        )
        value = clean_float(result["value"])
        if value is None:
            continue
        rows.append({"station": station, "value": round2(value), "n": result["n"]})
    rows.sort(key=lambda item: (-item["value"], item["station"]))
    return rows[:limit]


def aggregate(scope: dict, range: dict, metric: str = "sum") -> Envelope:
    """scope × range 집계 + 표본 수·커버리지. 관측이 없으면 값은 null."""
    started = perf_counter()
    try:
        if not isinstance(scope, dict):
            raise ToolError(ErrorCode.E_INVALID_ARG, "scope must be an object")
        if not isinstance(range, dict):
            raise ToolError(ErrorCode.E_INVALID_ARG, "range must be an object")
        if metric not in _METRICS:
            raise ToolError(
                ErrorCode.E_INVALID_ARG, f"metric 은 {_METRICS} 중 하나여야 합니다: {metric}"
            )
        clean_scope = _check_scope(scope)
        date_from, date_to = _check_range(range)
        result = analytics().aggregate(
            line=clean_scope.get("line"),
            station=clean_scope.get("station"),
            hour=clean_scope.get("hour"),
            weekday=clean_scope.get("weekday"),
            date_from=date_from,
            date_to=date_to,
            metric=metric,
        )
        value = clean_float(result["value"])
        data = {
            "scope": clean_scope,
            "range": {"date_from": date_from, "date_to": date_to},
            "metric": metric,
            "value": round2(value),
            "n": int(result["n"]),
            "coverage": round(float(result["coverage"]), 4),
            "unit": "명",
        }
        if "top" in clean_scope:
            data["top"] = _top_stations(clean_scope, date_from, date_to, metric, clean_scope["top"])
        hints = []
        if value is None:
            hints.append(
                caveat(Caveat.DATA_MISSING, "그 조건의 관측이 없어 값을 비웠습니다(0 아님).")
            )
        elif data["coverage"] < 1.0:
            hints.append(
                caveat(
                    Caveat.DATA_MISSING,
                    f"기대 칸 중 {data['n']}칸만 관측이 있습니다(커버리지 {data['coverage']:.4f}).",
                )
            )
        return Envelope.ok_(
            "aggregate",
            data,
            as_of=as_of(),
            inputs_hash=inputs_hash({"scope": scope, "range": range, "metric": metric}),
            elapsed_ms=int((perf_counter() - started) * 1000),
            caveats=hints,
        )
    except ToolError as exc:
        return Envelope.error(
            "aggregate", exc.code, exc.human, elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except ValueError as exc:
        return Envelope.error(
            "aggregate", ErrorCode.E_INVALID_ARG, str(exc), elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except Exception as exc:  # noqa: BLE001 — 예기치 못한 실패도 사유를 남긴다
        return Envelope.error(
            "aggregate",
            ErrorCode.E_INTERNAL,
            f"집계 중 오류: {type(exc).__name__}: {exc}",
            elapsed_ms=int((perf_counter() - started) * 1000),
        )