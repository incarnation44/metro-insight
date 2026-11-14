"""예측 하나의 근거를 모은다. 지금은 날씨 feature 만 분리해 계산한다.

C 모델만 날씨를 입력으로 받으므로 A/B/D/E 는 기여도를 꾸며 내지 않고
``supported=false`` 와 이유를 돌려준다. 기여도는 '그 feature 만 평년/안 옴으로
되돌렸을 때의 차이'로 계산하며, 값은 전부 예측기 결과다.
"""
from __future__ import annotations

from time import perf_counter

from metro_insight.agent.contracts import Envelope, ErrorCode
from metro_insight.agent.tools._common import (
    VALID_MODELS,
    ToolError,
    as_of,
    check_hour,
    check_line,
    check_serving_range,
    check_station,
    climatology_temperature,
    inputs_hash,
    parse_date,
    round2,
)
from metro_insight.agent.tools.predict import predict_values

_FEATURES = ("temperature", "rainfall", "snowfall")


def explain(line: str, station: str, date: str, hour: int, model: str) -> Envelope:
    """한 예측의 날씨 feature 기여도(또는 지원하지 않는다는 사실)."""
    started = perf_counter()
    try:
        name = str(model).strip().upper()
        if name not in VALID_MODELS:
            raise ToolError(
                ErrorCode.E_INVALID_ARG,
                f"모르는 모델입니다: {model} (가능: {', '.join(VALID_MODELS)})",
            )
        line = check_line(line)
        station = check_station(line, station)
        day = parse_date(date)
        check_serving_range(day)
        hour = check_hour(hour)
        if name != "C":
            return Envelope.ok_(
                "explain",
                {
                    "line": line,
                    "station": station,
                    "date": day.isoformat(),
                    "hour": hour,
                    "model": name,
                    "supported": False,
                    "method": "none",
                    "contributions": [],
                    "reason_code": ErrorCode.E_UNAVAILABLE.value,
                    "human": (
                        f"{name} 모델은 날씨를 입력으로 쓰지 않아 feature 기여도를 "
                        "계산하지 않습니다(임의 수치를 만들지 않음)."
                    ),
                },
                as_of=as_of(),
                inputs_hash=inputs_hash({"line": line, "station": station, "date": date,
                                         "hour": hour, "model": name}),
                elapsed_ms=int((perf_counter() - started) * 1000),
            )
        base_data, hints = predict_values(line, station, day.isoformat(), hour, None)
        base_value = base_data["by_model"].get("C")
        # 세 feature 를 한 번에 되돌린 계산(평년 기온·비 없음·눈 없음).
        normal = {
            "temperature": climatology_temperature(day.month),
            "rainfall": False,
            "snowfall": False,
        }
        ablated, _ = predict_values(line, station, day.isoformat(), hour, normal)
        ablated_value = ablated["by_model"].get("C")
        contributions = []
        if base_value is not None and ablated_value is not None:
            contributions.append(
                {
                    "features": list(_FEATURES),
                    "baseline_setting": "평년 기온 · 비 없음 · 눈 없음",
                    "baseline_value": round2(ablated_value),
                    "model_value": round2(base_value),
                    "delta": round2(base_value - ablated_value),
                }
            )
        return Envelope.ok_(
            "explain",
            {
                "line": line,
                "station": station,
                "date": day.isoformat(),
                "hour": hour,
                "model": name,
                "supported": True,
                "method": "weather_ablation",
                "contributions": contributions,
                "reason_code": None,
                "human": "입력한 날씨가 평년·안 옴과 비교해 만든 차이입니다.",
            },
            as_of=as_of(),
            inputs_hash=inputs_hash({"line": line, "station": station, "date": date,
                                     "hour": hour, "model": name}),
            elapsed_ms=int((perf_counter() - started) * 1000),
            caveats=hints,
        )
    except ToolError as exc:
        return Envelope.error(
            "explain", exc.code, exc.human, elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except Exception as exc:  # noqa: BLE001 — 예기치 못한 실패도 사유를 남긴다
        return Envelope.error(
            "explain",
            ErrorCode.E_INTERNAL,
            f"근거 계산 오류: {type(exc).__name__}: {exc}",
            elapsed_ms=int((perf_counter() - started) * 1000),
        )