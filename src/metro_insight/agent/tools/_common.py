"""도구 공통 런타임: 예측기 지연 로딩·역 사전·봉투 도우미·날짜/시간 검증.

도구 12개의 공통 기반만 담는다. 도구 이름은 허용목록
(:mod:`metro_insight.agent.tools`)에만 있고 여기서는 만들지 않는다.

정직성 규칙
* 예측기는 프로세스당 한 번만 읽는다(기존 ``models/boarding_history.npz`` 캐시 사용).
  로딩 실패는 0 으로 속이지 않고 ``E_UNAVAILABLE`` 로 알린다.
* ``data`` 에는 시각·난수를 넣지 않는다(같은 입력이면 같은 바이트). 시각은 봉투
  ``as_of`` 에만 둔다.
* 역 해석은 예측기의 기존 별칭 규칙을 재사용한다(새 별칭 표를 만들지 않는다).
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import date, timedelta
from importlib.resources import files
from pathlib import Path

from metro_insight.agent.contracts import Caveat, ErrorCode
from metro_insight.model.analytics import Analytics
from metro_insight.model.predictor import (
    DATA_END,
    ROOT,
    SERVING_END,
    SERVING_START,
    VERIFIED_HORIZON_BUCKETS,
    ABCPredictor,
    PredictionInput,
    PredictionUnavailable,
    climatology_temperature,
)

# 역 이름 별칭 (지도 표기 ← 원본 표기). 새 표를 만들지 않고 기존 규칙을 그대로 쓴다.
MAP_NAME_BY_DATA_STATION = {("7호선", "이수"): "총신대입구(이수)"}
STATION_ALIASES = {"이수": "총신대입구(이수)", "총신대입구": "총신대입구(이수)"}

VALID_MODELS = ("A", "B", "C", "D", "E")
WEATHER_KEYS = ("temperature", "rainfall", "snowfall")
HOUR_MIN, HOUR_MAX = 0, 24
# 데이터 끝 이후 이 일수를 넘으면 Test 로 검증하지 않은 구간이라 '일부 미검증' 문구가 필요하다.
UNVERIFIED_AFTER_DAYS = 136
# 기준일 고정용 환경변수(테스트·재생의 재현성). 없으면 실제 오늘.
TODAY_ENV = "METRO_AGENT_TODAY"

_lock = threading.Lock()
_predictor: ABCPredictor | None = None
_predictor_error: str | None = None
_map_pairs: list[dict] | None = None
_holidays: dict[int, dict] = {}


class ToolError(Exception):
    """도구 실패를 봉투 코드와 함께 올린다. 도구 함수가 이 예외를 봉투로 바꾼다."""

    def __init__(self, code: ErrorCode, human: str) -> None:
        super().__init__(human)
        self.code = code
        self.human = human


# ---------- 예측기 ----------


def predictor() -> ABCPredictor:
    """예측기 지연 싱글턴. 두 번째 호출부터는 같은 객체를 돌려준다."""
    global _predictor, _predictor_error
    if _predictor is not None:
        return _predictor
    with _lock:
        if _predictor is not None:
            return _predictor
        try:
            _predictor = ABCPredictor()
        except Exception as exc:  # noqa: BLE001 — 어떤 실패든 E_UNAVAILABLE 로 알린다
            _predictor_error = f"{type(exc).__name__}: {exc}"
            raise ToolError(
                ErrorCode.E_UNAVAILABLE,
                f"예측기를 읽지 못했습니다: {_predictor_error}",
            ) from exc
    return _predictor


def analytics() -> Analytics:
    """예측기 이력 배열 위의 집계기(같은 싱글턴을 공유한다)."""
    return Analytics(predictor())


def predictor_failure() -> str | None:
    """예측기 로딩이 실패했으면 그 이유(진단용)."""
    return _predictor_error


def predictor_ready() -> bool:
    """예측기가 이미 메모리에 있는지. 콜드 로딩 시간을 도구 지연으로 세지 않게 한다."""
    return _predictor is not None


# ---------- 시간 ----------


def today() -> date:
    """기준 '오늘'. ``METRO_AGENT_TODAY`` 로 고정할 수 있다(테스트·재생의 재현성)."""
    import os

    raw = os.environ.get(TODAY_ENV)
    if raw:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            pass
    return date.today()


def parse_date(value: object) -> date:
    """'YYYY-MM-DD' → date. 형식 오류는 E_INVALID_ARG."""
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ToolError(ErrorCode.E_INVALID_ARG, f"날짜 형식이 아닙니다: {value}") from exc


def iso(day: date) -> str:
    return day.isoformat()


def check_hour(value: object) -> int:
    """시간대 시작 시(0~24). 범위 밖·비정수는 E_INVALID_ARG."""
    if isinstance(value, bool):
        raise ToolError(ErrorCode.E_INVALID_ARG, f"hour 는 정수여야 합니다: {value}")
    try:
        hour = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ToolError(ErrorCode.E_INVALID_ARG, f"hour 는 정수여야 합니다: {value}") from exc
    if not HOUR_MIN <= hour <= HOUR_MAX:
        raise ToolError(ErrorCode.E_INVALID_ARG, f"hour 는 {HOUR_MIN}~{HOUR_MAX} 사이여야 합니다: {value}")
    return hour


def check_serving_range(day: date) -> None:
    """예측 가능 범위(데이터 시작 ~ 데이터 끝+1년) 밖이면 E_OUT_OF_RANGE."""
    if day < SERVING_START:
        raise ToolError(
            ErrorCode.E_OUT_OF_RANGE,
            f"조회 가능 범위는 {SERVING_START.isoformat()}부터입니다"
            f"(가장 가까운 날짜 {SERVING_START.isoformat()}).",
        )
    if day > SERVING_END:
        raise ToolError(
            ErrorCode.E_OUT_OF_RANGE,
            f"조회 가능 범위는 {SERVING_END.isoformat()}까지입니다(데이터 끝 + 1년).",
        )


def days_after_data_end(day: date) -> int:
    return (day - DATA_END).days


def is_future(day: date) -> bool:
    return day > DATA_END


def horizon_partial(day: date) -> bool:
    """데이터 끝 이후 136일을 넘어 Test 로 검증하지 않은 구간인지."""
    return days_after_data_end(day) > UNVERIFIED_AFTER_DAYS


def horizon_label(bucket: str | None) -> str:
    """지평 구간 이름 → 표시용 ('92-181' → '92~181일'). 없으면 '과거(실측 구간)'."""
    if bucket is None:
        return "과거(실측 구간)"
    return str(bucket).replace("-", "~") + "일"


def verified_buckets() -> tuple[str, ...]:
    return tuple(VERIFIED_HORIZON_BUCKETS)


# ---------- 날씨·공휴일 ----------


def check_weather(weather: object) -> dict:
    """날씨 dict 검증·정규화. 모르는 키는 E_INVALID_ARG(조용히 버리지 않는다)."""
    if weather is None:
        return {}
    if not isinstance(weather, dict):
        raise ToolError(ErrorCode.E_INVALID_ARG, "weather 는 객체여야 합니다")
    unknown = sorted(set(weather) - set(WEATHER_KEYS))
    if unknown:
        raise ToolError(ErrorCode.E_INVALID_ARG, f"weather 에 모르는 키가 있습니다: {', '.join(unknown)}")
    out: dict = {}
    if "temperature" in weather:
        value = weather["temperature"]
        if value is None:
            out["temperature"] = None
        else:
            try:
                out["temperature"] = float(value)
            except (TypeError, ValueError) as exc:
                raise ToolError(ErrorCode.E_INVALID_ARG, f"temperature 는 숫자여야 합니다: {value}") from exc
    if "rainfall" in weather:
        out["rainfall"] = bool(weather["rainfall"])
    if "snowfall" in weather:
        out["snowfall"] = bool(weather["snowfall"])
    return out


def kr_holidays(year: int) -> dict:
    """그 해 한국 공휴일. 학습·앱과 같은 holidays 패키지 목록을 쓴다."""
    if year not in _holidays:
        from holidays import country_holidays

        _holidays[year] = dict(country_holidays("KR", years=year, language="ko"))
    return _holidays[year]


def is_holiday(day: date) -> bool:
    return day in kr_holidays(day.year)


def prediction_input(line: str, station: str, day: date, hour: int, weather: dict) -> PredictionInput:
    """도구 인자 → 예측기 입력 한 행. 날씨가 없으면 평년 가정(앱과 같은 규칙)."""
    temperature = weather.get("temperature")
    return PredictionInput(
        date=day.isoformat(),
        line=str(line),
        station=str(station),
        hour=int(hour),
        weekday=day.weekday(),
        is_weekend=day.weekday() >= 5,
        is_holiday=is_holiday(day),
        temperature=None if temperature is None else round(float(temperature), 1),
        rainfall=bool(weather.get("rainfall", False)),
        snowfall=bool(weather.get("snowfall", False)),
    )


def effective_temperature(day: date, weather: dict) -> tuple[float, bool]:
    """(모델에 들어가는 기온, 평년값 가정 여부)."""
    if weather.get("temperature") is None:
        return climatology_temperature(day.month), True
    return round(float(weather["temperature"]), 1), False


# ---------- 역 사전 ----------


def data_pairs() -> frozenset[tuple[str, str]]:
    """학습 데이터에 있는 (호선, 역) 쌍."""
    return frozenset(predictor()._station_code)


def base_name(name: str) -> str:
    """'잠실(송파구청)' → '잠실' (예측기 별칭 규칙과 같은 정규화)."""
    text = str(name)
    while "(" in text and ")" in text:
        head, _, rest = text.partition("(")
        _, _, tail = rest.partition(")")
        text = head + tail
    return text.strip()


def canonical_map_name(name: str) -> str:
    """사용자가 쓴 이름 → 지도 표기. '이수'·'총신대입구' → '총신대입구(이수)'."""
    text = str(name).strip().removesuffix("역")
    return STATION_ALIASES.get(text, text)


def resolve_availability(line: str, station: str) -> tuple[int | None, str]:
    """(station_code, match). match ∈ {'exact','alias','base','none'}.

    데이터에 정확히 있으면 exact, 괄호·지도 별칭을 거쳐 찾으면 alias/base.
    """
    pairs = predictor()._station_code
    key = (str(line), str(station))
    if key in pairs:
        return int(pairs[key]), "exact"
    merged = MAP_NAME_BY_DATA_STATION.get((str(line), str(station)))
    if merged is not None and (str(line), merged) in pairs:
        return int(pairs[(str(line), merged)]), "alias"
    wanted = base_name(station)
    for (pair_line, pair_station), code in pairs.items():
        if pair_line == str(line) and base_name(pair_station) == wanted:
            return int(code), "base"
    return None, "none"


def map_pairs() -> list[dict]:
    """지도(노선도) 기준 (호선, 역) 쌍. 좌표 포함."""
    global _map_pairs
    if _map_pairs is not None:
        return _map_pairs
    raw = json.loads(
        (files("metro_insight.app") / "resources" / "subway_network.json").read_text(encoding="utf-8")
    )
    line_name = {ln["id"]: ln["name"] for ln in raw["lines"]}
    pairs: list[dict] = []
    for station in raw["stations"]:
        for line_id in station["lines"]:
            pairs.append(
                {
                    "line": line_name[line_id],
                    "station": station["name"],
                    "lat": float(station["lat"]),
                    "lon": float(station["lon"]),
                }
            )
    pairs.sort(key=lambda item: (item["line"], item["station"]))
    _map_pairs = pairs
    return pairs


def map_only_stats() -> dict:
    """지도 vs 데이터 공백 통계.

    ``exact`` 는 (호선, 지도 표기) 쌍을 그대로 비교한 수, ``unresolved`` 는 별칭·괄호
    규칙까지 적용해도 데이터가 없는 수다. 둘의 차이는 데이터에 있는데 지도 표기가
    다른 1쌍(7호선 총신대입구(이수))이다. 0으로 세지 않고 두 수를 모두 남긴다.
    """
    pairs = map_pairs()
    known = data_pairs()
    exact = []
    unresolved = []
    for item in pairs:
        code, _ = resolve_availability(item["line"], item["station"])
        if (item["line"], item["station"]) not in known:
            exact.append(item)
        if code is None:
            unresolved.append(item)
    by_line: dict[str, int] = {}
    for item in unresolved:
        by_line[item["line"]] = by_line.get(item["line"], 0) + 1
    return {
        "map_pairs": len(pairs),
        "data_pairs": len(known),
        "map_only_exact": len(exact),
        "map_only_unresolved": len(unresolved),
        "by_line": dict(sorted(by_line.items())),
    }


def nearest_data_station(line: str, lat: float, lon: float) -> dict | None:
    """데이터가 있는 가장 가까운 역(같은 호선 우선). 지도 공백 역 안내용."""
    best: tuple[float, dict] | None = None
    for item in map_pairs():
        if item["line"] != line:
            continue
        code, _ = resolve_availability(item["line"], item["station"])
        if code is None:
            continue
        dist = _distance_km(lat, lon, item["lat"], item["lon"])
        if best is None or dist < best[0]:
            best = (dist, item)
    if best is None:
        return None
    dist, item = best
    return {
        "line": item["line"],
        "station": item["station"],
        "distance_km": round(dist, 1),
    }


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = (lat1 - lat2) * 111.0
    dlon = (lon1 - lon2) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dlat, dlon)


def stations_of_line(line: str) -> list[str]:
    """그 호선의 데이터 역 이름 목록(정렬)."""
    return sorted(station for (pair_line, station) in data_pairs() if pair_line == str(line))


def check_line(line: object) -> str:
    """호선 이름 검증. 데이터에 없는 호선은 E_NOT_FOUND."""
    name = str(line).strip()
    if name not in predictor()._line_code:
        known = ", ".join(sorted(predictor()._line_code))
        raise ToolError(ErrorCode.E_NOT_FOUND, f"{name} 호선은 데이터에 없습니다(가능: {known}).")
    return name


def check_station(line: str, station: object) -> str:
    """(호선, 역) 검증. 데이터에 없으면 지도에 있는지까지 확인해 이유를 나눈다."""
    name = str(station).strip()
    code, _ = resolve_availability(line, name)
    if code is None:
        raise ToolError(ErrorCode.E_NOT_FOUND, data_missing_reason(line, name))
    return name


def data_missing_reason(line: str, station: str) -> str:
    """없는 역의 이유 문장. 지도에만 있으면 '예측 데이터 없음'으로 구분한다."""
    for item in map_pairs():
        if item["line"] == line and item["station"] == station:
            near = nearest_data_station(line, item["lat"], item["lon"])
            tail = f" 가장 가까운 데이터 역은 {near['station']}입니다." if near else ""
            return f"{station}({line}) 역은 지도에는 있으나 예측 데이터가 없습니다.{tail}"
    return f"{station}({line}) 역은 학습 데이터에 없습니다."


# ---------- 봉투 도우미 ----------


def as_of() -> str:
    """데이터 기준일(봉투 ``as_of``)."""
    return DATA_END.isoformat()


def inputs_hash(payload: object) -> str:
    """입력 지문. 같은 입력이면 항상 같은 문자열(정렬·고정 구분자)."""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def caveat(code: Caveat, human: str) -> dict:
    return {"code": Caveat(code).value, "human": human}


def horizon_caveat(bucket: str | None, day: date) -> dict | None:
    """검증하지 않은 지평이면 HORIZON_PARTIAL."""
    if not is_future(day) or not horizon_partial(day):
        return None
    return caveat(
        Caveat.HORIZON_PARTIAL,
        f"데이터 끝 이후 {days_after_data_end(day)}일 지난 날짜라 일부 미검증 구간입니다.",
    )


def missing_entry(model: str, day: date, reason: str | None) -> dict:
    """D/E 키 부재 사유. 0 으로 채우지 않고 이유를 남긴다."""
    if is_future(day):
        human = "대상일 전 14일 실측이 없어 계산 불가"
    else:
        human = f"시계열 모델을 쓸 수 없어 계산 불가({reason or '모델 없음'})"
    return {"model": model, "reason_code": ErrorCode.E_UNAVAILABLE.value, "human": human}


def fallback_reason(exc: BaseException) -> tuple[ErrorCode, str]:
    """도구가 올린 예외 → (오류 코드, 사람 문장)."""
    if isinstance(exc, ToolError):
        return exc.code, exc.human
    if isinstance(exc, PredictionUnavailable):
        return ErrorCode.E_UNAVAILABLE, str(exc)
    if isinstance(exc, ValueError):
        return ErrorCode.E_INVALID_ARG, str(exc)
    return ErrorCode.E_INTERNAL, f"{type(exc).__name__}: {exc}"


def round2(value: float | None) -> float | None:
    """표시용 2자리 반올림. NaN 은 None(0 으로 속이지 않는다)."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 2)


def clean_float(value: object) -> float | None:
    """NaN/무한 → None. JSON 에 NaN 을 남기지 않는다."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def reports_root() -> Path:
    """리포트 쓰기 허용 루트(기본 ``reports/``). 테스트는 환경변수로 격리한다."""
    import os

    raw = os.environ.get("METRO_AGENT_REPORTS_ROOT")
    return Path(raw) if raw else ROOT / "reports"


def repo_root() -> Path:
    return ROOT


def format_count(value: float | None) -> str:
    """승차 인원 표시: 정수 반올림 + 천 단위 쉼표 + '명'. 없으면 '계산 불가'."""
    number = clean_float(value)
    if number is None:
        return "계산 불가"
    return f"{int(round(number)):,}명"


def format_decimal(value: float | None, digits: int = 2) -> str:
    number = clean_float(value)
    if number is None:
        return "계산 불가"
    return f"{number:,.{digits}f}"


def format_days(days: int) -> str:
    return f"{int(days)}일"


def format_hour(hour: int) -> str:
    return f"{int(hour):02d}시"


def format_rate(value: float | None) -> str:
    number = clean_float(value)
    if number is None:
        return "계산 불가"
    return f"{number * 100:.1f}%"


def tomorrow_of(day: date) -> date:
    return day + timedelta(days=1)