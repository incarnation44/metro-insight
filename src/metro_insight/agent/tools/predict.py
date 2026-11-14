"""한 조건의 A~E 예상 승차 인원. 미래 날짜는 실측 창이 없어 D/E 를 키 없이 돌려준다.

계산은 예측기(:class:`~metro_insight.model.predictor.ABCPredictor`)가 한다. 이 도구는
인자 검증·지평 구간·없는 모델의 사유를 붙이는 일만 한다. D/E 가 없을 때 0 을 넣지
않고 키를 빼고 ``missing`` 에 이유를 남긴다.
"""
from __future__ import annotations

from time import perf_counter

from metro_insight.agent.contracts import Caveat, Envelope, ErrorCode
from metro_insight.agent.tools._common import (
    VALID_MODELS,
    ToolError,
    as_of,
    caveat,
    check_hour,
    check_line,
    check_serving_range,
    check_station,
    check_weather,
    days_after_data_end,
    horizon_caveat,
    horizon_label,
    horizon_partial,
    inputs_hash,
    is_future,
    missing_entry,
    parse_date,
    prediction_input,
    predictor,
    round2,
    verified_buckets,
)
from metro_insight.model.predictor import horizon_bucket


def predict_values(
    line: str, station: str, date: str, hour: int, weather: dict | None = None
) -> tuple[dict, list[dict]]:
    """(봉투 ``data``, 한계 목록). 인자 오류는 :class:`ToolError` 로 올린다."""
    line = check_line(line)
    station = check_station(line, station)
    day = parse_date(date)
    check_serving_range(day)
    hour = check_hour(hour)
    weather = check_weather(weather)

    model = predictor()
    values = model.predict(prediction_input(line, station, day, hour, weather))
    by_model = {
        name: round2(values[name])
        for name in VALID_MODELS
        if isinstance(values.get(name), (int, float))
    }
    missing = [
        missing_entry(name, day, model.last_errors.get(name))
        for name in ("D", "E")
        if name not in by_model
    ]
    scores = model.horizon_scores(day)
    bucket = horizon_bucket(day)
    assumed = weather.get("temperature") is None
    data = {
        "line": line,
        "station": station,
        "date": day.isoformat(),
        "hour": hour,
        "by_model": by_model,
        "missing": missing,
        "horizon_bucket": bucket,
        "horizon_label": horizon_label(bucket),
        "horizon_scores": {
            name: {"mae": round2(score.mae), "rmse": round2(score.rmse)}
            for name, score in sorted(scores.items())
        },
        "horizon_verified": bucket in verified_buckets(),
        "is_future": is_future(day),
        "days_after_data_end": days_after_data_end(day),
        "weather": {
            "assumed": assumed,
            "temperature": weather.get("temperature"),
            "rainfall": bool(weather.get("rainfall", False)),
            "snowfall": bool(weather.get("snowfall", False)),
        },
    }
    hints: list[dict] = []
    partial = horizon_caveat(bucket, day)
    if partial is not None:
        hints.append(partial)
    if missing:
        hints.append(
            caveat(
                Caveat.DE_UNAVAILABLE,
                "D/E 모델은 대상일 전 실측 창이 없어 계산하지 않았습니다(0 으로 채우지 않음).",
            )
        )
    if assumed:
        hints.append(
            caveat(
                Caveat.WEATHER_ASSUMED,
                "날씨를 입력하지 않아 기온은 월별 평년값, 비·눈은 '안 옴'으로 가정했습니다.",
            )
        )
    if horizon_partial(day):
        data["horizon_partial"] = True
    return data, hints


def predict(
    line: str, station: str, date: str, hour: int, weather: dict | None = None
) -> Envelope:
    """A~E 모델 예상 승차 인원 + 지평 구간 + 없는 모델의 사유."""
    started = perf_counter()
    payload = {"line": str(line), "station": str(station), "date": str(date), "hour": hour}
    try:
        data, hints = predict_values(line, station, date, hour, weather)
    except ToolError as exc:
        return Envelope.error(
            "predict", exc.code, exc.human, elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except Exception as exc:  # noqa: BLE001 — 예기치 못한 실패도 사유를 남긴다
        return Envelope.error(
            "predict",
            ErrorCode.E_INTERNAL,
            f"예측 중 오류: {type(exc).__name__}: {exc}",
            elapsed_ms=int((perf_counter() - started) * 1000),
        )
    return Envelope.ok_(
        "predict",
        data,
        as_of=as_of(),
        inputs_hash=inputs_hash({**payload, "weather": weather}),
        elapsed_ms=int((perf_counter() - started) * 1000),
        caveats=hints,
    )