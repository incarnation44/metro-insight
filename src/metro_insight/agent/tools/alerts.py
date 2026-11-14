"""scope × range 에서 평균에서 z 배를 넘는 날을 찾는다. 과거 기준이며 실시간이 아니다.

계산은 :class:`~metro_insight.model.analytics.Analytics` 가 한다. 경계값(정확히 z)은
이상치로 넣지 않고, 관측이 2일 미만이면 표준편차를 만들지 않는다.
"""
from __future__ import annotations

from time import perf_counter

from metro_insight.agent.contracts import Envelope, ErrorCode
from metro_insight.agent.tools._common import (
    DATA_END,
    ToolError,
    analytics,
    as_of,
    check_hour,
    check_line,
    check_station,
    inputs_hash,
    parse_date,
    round2,
)

_SCOPE_KEYS = ("line", "station", "hour")
_RANGE_KEYS = ("date_from", "date_to")


def _check_z(z: object) -> float:
    try:
        value = float(z)
    except (TypeError, ValueError) as exc:
        raise ToolError(ErrorCode.E_INVALID_ARG, f"z 는 숫자여야 합니다: {z}") from exc
    if not 0 < value <= 10:
        raise ToolError(ErrorCode.E_INVALID_ARG, f"z 는 0 보다 크고 10 이하여야 합니다: {z}")
    return value


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
        if "line" not in out:
            raise ToolError(ErrorCode.E_INVALID_ARG, "station 을 쓰려면 line 이 필요합니다")
        out["station"] = check_station(out["line"], scope["station"])
    if scope.get("hour") is not None:
        out["hour"] = check_hour(scope["hour"])
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
            f"이상치 조회 범위는 {DATA_END.isoformat()}까지입니다(가장 가까운 날짜 {DATA_END.isoformat()}).",
        )
    if end > DATA_END:
        end = DATA_END
    return start.isoformat(), end.isoformat()


def alerts(scope: dict, range: dict, z: float = 3.0) -> Envelope:
    """과거 기준 이상치 목록(경계값 제외)."""
    started = perf_counter()
    try:
        if not isinstance(scope, dict):
            raise ToolError(ErrorCode.E_INVALID_ARG, "scope must be an object")
        if not isinstance(range, dict):
            raise ToolError(ErrorCode.E_INVALID_ARG, "range must be an object")
        clean_scope = _check_scope(scope)
        date_from, date_to = _check_range(range)
        threshold = _check_z(z)
        found = analytics().outliers(
            line=clean_scope.get("line"),
            station=clean_scope.get("station"),
            hour=clean_scope.get("hour"),
            date_from=date_from,
            date_to=date_to,
            z=threshold,
        )
        rows = [
            {
                "date": item["date"],
                "value": round2(item["value"]),
                "z": round2(item["z"]),
                "mean": round2(item["mean"]),
                "std": round2(item["std"]),
            }
            for item in found
        ]
        hints: list[dict] = []
        return Envelope.ok_(
            "alerts",
            {
                "scope": clean_scope,
                "range": {"date_from": date_from, "date_to": date_to},
                "z": threshold,
                "alerts": rows,
                "n": len(rows),
                "basis": "past_actuals",
            },
            as_of=as_of(),
            inputs_hash=inputs_hash({"scope": scope, "range": range, "z": threshold}),
            elapsed_ms=int((perf_counter() - started) * 1000),
            caveats=hints,
        )
    except ToolError as exc:
        return Envelope.error(
            "alerts", exc.code, exc.human, elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except ValueError as exc:
        return Envelope.error(
            "alerts", ErrorCode.E_INVALID_ARG, str(exc), elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except Exception as exc:  # noqa: BLE001 — 예기치 못한 실패도 사유를 남긴다
        return Envelope.error(
            "alerts",
            ErrorCode.E_INTERNAL,
            f"이상치 계산 오류: {type(exc).__name__}: {exc}",
            elapsed_ms=int((perf_counter() - started) * 1000),
        )