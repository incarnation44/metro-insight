"""같은 조건에서 여러 모델을 나란히 비교한다.

예측은 한 번만 계산해 재사용한다(같은 조건임을 보장). 요청한 모델이 데이터에 없으면
0 이 아니라 ``missing`` 에 이유를 남긴다. D/E 가 있으면 '최근 실측 사용' 라벨을 붙인다.
"""
from __future__ import annotations

from time import perf_counter

from metro_insight.agent.contracts import Envelope, ErrorCode
from metro_insight.agent.tools._common import VALID_MODELS, ToolError, as_of, inputs_hash, round2
from metro_insight.agent.tools.predict import predict_values

_LABELS = {"D": "최근 실측 사용", "E": "최근 실측 사용"}


def _check_models(models: object) -> list[str] | None:
    if models is None:
        return None
    if not isinstance(models, (list, tuple)):
        raise ToolError(ErrorCode.E_INVALID_ARG, "models 는 배열이어야 합니다")
    wanted: list[str] = []
    for raw in models:
        name = str(raw).strip().upper()
        if name not in VALID_MODELS:
            raise ToolError(
                ErrorCode.E_INVALID_ARG,
                f"모르는 모델입니다: {raw} (가능: {', '.join(VALID_MODELS)})",
            )
        if name not in wanted:
            wanted.append(name)
    if not wanted:
        raise ToolError(ErrorCode.E_INVALID_ARG, "models 가 비어 있습니다")
    return wanted


def compare(
    line: str, station: str, date: str, hour: int, models: list[str] | None = None
) -> Envelope:
    """같은 조건에서 모델 여러 개를 나란히 비교한다(모델쌍 차이 포함)."""
    started = perf_counter()
    try:
        wanted = _check_models(models)
        data, hints = predict_values(line, station, date, hour, None)
    except ToolError as exc:
        return Envelope.error(
            "compare", exc.code, exc.human, elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except Exception as exc:  # noqa: BLE001 — 예기치 못한 실패도 사유를 남긴다
        return Envelope.error(
            "compare",
            ErrorCode.E_INTERNAL,
            f"비교 중 오류: {type(exc).__name__}: {exc}",
            elapsed_ms=int((perf_counter() - started) * 1000),
        )

    available = dict(data["by_model"])
    missing = list(data["missing"])
    if wanted is not None:
        available = {name: value for name, value in available.items() if name in wanted}
        missing = [item for item in missing if item["model"] in wanted]
    order = list(available)
    diffs = []
    for left, right in zip(order, order[1:], strict=False):
        left_value, right_value = available[left], available[right]
        if left_value is None or right_value is None:
            continue
        delta = round2(left_value - right_value)
        base = right_value if right_value else None
        diffs.append(
            {
                "pair": f"{left}-{right}",
                "delta": delta,
                "delta_pct": round2(delta / base * 100) if base else None,
            }
        )
    payload = {
        "line": data["line"],
        "station": data["station"],
        "date": data["date"],
        "hour": data["hour"],
        "models": order,
        "by_model": available,
        "diffs": diffs,
        "missing": missing,
        "labels": {name: _LABELS[name] for name in order if name in _LABELS},
        "same_condition": True,
        "horizon_bucket": data["horizon_bucket"],
        "horizon_label": data["horizon_label"],
        "is_future": data["is_future"],
        "days_after_data_end": data["days_after_data_end"],
    }
    return Envelope.ok_(
        "compare",
        payload,
        as_of=as_of(),
        inputs_hash=inputs_hash(
            {"line": line, "station": station, "date": date, "hour": hour, "models": wanted}
        ),
        elapsed_ms=int((perf_counter() - started) * 1000),
        caveats=hints,
    )