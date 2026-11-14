"""기준 조건과 가정 시나리오들을 같은 모델로 다시 계산해 차이를 보여 준다.

숫자는 전부 예측기 계산 결과다(이 도구는 더하거나 곱하지 않고 두 계산의 차만 낸다).
시나리오는 날씨·기온만 바꿀 수 있고, 그 밖의 키는 조용히 버리지 않고 거부한다.
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
    check_weather,
    climatology_temperature,
    inputs_hash,
    parse_date,
    round2,
)
from metro_insight.agent.tools.predict import predict_values

_BASE_KEYS = ("line", "station", "date", "hour", "weather")
_SCENARIO_KEYS = ("name", "weather", "temperature_delta", "rainfall", "snowfall")
MAX_SCENARIOS = 5


def _check_base(base: object) -> dict:
    if not isinstance(base, dict):
        raise ToolError(ErrorCode.E_INVALID_ARG, "base must be an object")
    unknown = sorted(set(base) - set(_BASE_KEYS))
    if unknown:
        raise ToolError(ErrorCode.E_INVALID_ARG, f"base 에 모르는 키가 있습니다: {', '.join(unknown)}")
    for key in ("line", "station", "date", "hour"):
        if base.get(key) in (None, ""):
            raise ToolError(ErrorCode.E_INVALID_ARG, f"base.{key} 가 필요합니다")
    out = {
        "line": check_line(base["line"]),
        "station": str(base["station"]),
        "date": parse_date(base["date"]),
        "hour": check_hour(base["hour"]),
        "weather": check_weather(base.get("weather")),
    }
    out["station"] = check_station(out["line"], out["station"])
    check_serving_range(out["date"])
    return out


def _check_scenarios(scenarios: object) -> list[dict]:
    if not isinstance(scenarios, (list, tuple)):
        raise ToolError(ErrorCode.E_INVALID_ARG, "scenarios 는 배열이어야 합니다")
    if not scenarios:
        raise ToolError(ErrorCode.E_INVALID_ARG, "scenarios 가 비어 있습니다")
    if len(scenarios) > MAX_SCENARIOS:
        raise ToolError(
            ErrorCode.E_INVALID_ARG, f"scenarios 는 {MAX_SCENARIOS}개까지입니다: {len(scenarios)}"
        )
    out = []
    for index, raw in enumerate(scenarios):
        if not isinstance(raw, dict):
            raise ToolError(ErrorCode.E_INVALID_ARG, f"scenarios[{index}] 는 객체여야 합니다")
        unknown = sorted(set(raw) - set(_SCENARIO_KEYS))
        if unknown:
            raise ToolError(
                ErrorCode.E_INVALID_ARG,
                f"scenarios[{index}] 에 모르는 키가 있습니다: {', '.join(unknown)}",
            )
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ToolError(ErrorCode.E_INVALID_ARG, f"scenarios[{index}].name 이 필요합니다")
        entry = {"name": name.strip(), "weather": check_weather(raw.get("weather"))}
        if raw.get("temperature_delta") is not None:
            try:
                entry["temperature_delta"] = float(raw["temperature_delta"])
            except (TypeError, ValueError) as exc:
                raise ToolError(
                    ErrorCode.E_INVALID_ARG,
                    f"scenarios[{index}].temperature_delta 는 숫자여야 합니다",
                ) from exc
        for key in ("rainfall", "snowfall"):
            if raw.get(key) is not None:
                entry[key] = bool(raw[key])
        changed = [key for key in ("temperature_delta", "rainfall", "snowfall") if key in entry]
        if not changed and not entry["weather"]:
            raise ToolError(
                ErrorCode.E_INVALID_ARG,
                f"scenarios[{index}] 에 바꿀 조건이 없습니다(비·눈·기온 중 하나 필요)",
            )
        out.append(entry)
    return out


def _merge_weather(base_weather: dict, scenario: dict, base_day) -> dict:
    """기준 날씨 + 시나리오 변경 → 그 시나리오의 날씨."""
    weather = dict(base_weather)
    weather.update(scenario["weather"])
    if "temperature_delta" in scenario:
        current = weather.get("temperature")
        if current is None:
            current = climatology_temperature(base_day.month)
        weather["temperature"] = round(float(current) + scenario["temperature_delta"], 1)
    for key in ("rainfall", "snowfall"):
        if key in scenario:
            weather[key] = scenario[key]
    return weather


def simulate(base: dict, scenarios: list[dict]) -> Envelope:
    """기준 조건과 가정 시나리오들의 모델별 값·차이(델타)."""
    started = perf_counter()
    try:
        clean_base = _check_base(base)
        clean_scenarios = _check_scenarios(scenarios)
        base_data, hints = predict_values(
            clean_base["line"],
            clean_base["station"],
            clean_base["date"].isoformat(),
            clean_base["hour"],
            clean_base["weather"],
        )
        base_values = base_data["by_model"]
        rows = []
        for scenario in clean_scenarios:
            weather = _merge_weather(clean_base["weather"], scenario, clean_base["date"])
            data, _ = predict_values(
                clean_base["line"],
                clean_base["station"],
                clean_base["date"].isoformat(),
                clean_base["hour"],
                weather,
            )
            values = data["by_model"]
            deltas = {}
            pct = {}
            for name in VALID_MODELS:
                if name not in values or name not in base_values:
                    continue
                delta = round2(values[name] - base_values[name])
                deltas[name] = delta
                pct[name] = (
                    round2(delta / base_values[name] * 100) if base_values[name] else None
                )
            rows.append(
                {
                    "name": scenario["name"],
                    "by_model": values,
                    "delta": deltas,
                    "delta_pct": pct,
                    "weather": {
                        "temperature": weather.get("temperature"),
                        "rainfall": bool(weather.get("rainfall", False)),
                        "snowfall": bool(weather.get("snowfall", False)),
                    },
                }
            )
        return Envelope.ok_(
            "simulate",
            {
                "line": clean_base["line"],
                "station": clean_base["station"],
                "date": clean_base["date"].isoformat(),
                "hour": clean_base["hour"],
                "base": {
                    "by_model": base_values,
                    "weather": base_data["weather"],
                },
                "scenarios": rows,
                "horizon_bucket": base_data["horizon_bucket"],
                "is_future": base_data["is_future"],
                "days_after_data_end": base_data["days_after_data_end"],
            },
            as_of=as_of(),
            inputs_hash=inputs_hash({"base": base, "scenarios": scenarios}),
            elapsed_ms=int((perf_counter() - started) * 1000),
            caveats=hints,
        )
    except ToolError as exc:
        return Envelope.error(
            "simulate", exc.code, exc.human, elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except Exception as exc:  # noqa: BLE001 — 예기치 못한 실패도 사유를 남긴다
        return Envelope.error(
            "simulate",
            ErrorCode.E_INTERNAL,
            f"시뮬레이션 중 오류: {type(exc).__name__}: {exc}",
            elapsed_ms=int((perf_counter() - started) * 1000),
        )