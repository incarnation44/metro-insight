"""에이전트 도구 허용목록과 호출 계약.

도구는 12개로 고정한다. 각 도구의 본체는 ``tools/<이름>.py`` 에 있고, 이 모듈은
허용목록·쓰기 허용목록과 공개 시그니처만 유지한다. 모든 도구는
:class:`~metro_insight.agent.contracts.Envelope` 를 돌려주고, 실패도 같은 모양의
봉투로 알린다(예외를 밖으로 던지지 않는다).

인자 형태는 계획의 인자 형태 그대로다 (위치/키워드 혼용을 임의로 만들지 않는다).
``aggregate``·``alerts`` 는 scope·range 를 dict 로 받는다 — 문자열로 넘기면
E_INVALID_ARG 성격의 ValueError 로 즉시 거부한다.
"""
from __future__ import annotations

from metro_insight.agent.contracts import Envelope, ErrorCode
from metro_insight.agent.tools import aggregate as _aggregate
from metro_insight.agent.tools import alerts as _alerts
from metro_insight.agent.tools import backtest as _backtest
from metro_insight.agent.tools import compare as _compare
from metro_insight.agent.tools import drift as _drift
from metro_insight.agent.tools import explain as _explain
from metro_insight.agent.tools import history as _history
from metro_insight.agent.tools import predict as _predict
from metro_insight.agent.tools import reconcile as _reconcile
from metro_insight.agent.tools import report as _report
from metro_insight.agent.tools import resolve_station as _resolve_station
from metro_insight.agent.tools import simulate as _simulate

# 도구 이름 허용목록 (순서 고정: 화면·평가가 이 순서로 보여 준다)
ALLOWLIST: tuple[str, ...] = (
    "resolve_station",
    "predict",
    "compare",
    "history",
    "aggregate",
    "simulate",
    "explain",
    "reconcile",
    "backtest",
    "drift",
    "alerts",
    "report",
)

# 쓰기(외부 저장·전송)가 허용된 도구. 나머지는 전부 읽기 전용이다.
WRITE_ALLOWED = frozenset({"report"})


def envelope_ok(tool: str, data: dict, **kw) -> Envelope:
    """성공 봉투 얇은 래퍼."""
    return Envelope.ok_(tool, data, **kw)


def envelope_error(tool: str, code: ErrorCode, human: str, **kw) -> Envelope:
    """실패 봉투 얇은 래퍼."""
    return Envelope.error(tool, code, human, **kw)


def check_tool_allowed(name: str) -> None:
    """허용목록 밖 도구 이름을 막는다. LLM 이 만든 이름이 실행 경로로 새지 않게 한다."""
    if name not in ALLOWLIST:
        raise ValueError(f"{ErrorCode.E_INVALID_ARG.value}: unknown tool '{name}'")


def _require_dict(argument: str, value: object) -> dict:
    """dict 인자를 요구한다. 문자열·None 등은 E_INVALID_ARG 성격의 ValueError."""
    if not isinstance(value, dict):
        raise ValueError(f"{ErrorCode.E_INVALID_ARG.value}: {argument} must be an object")
    return value


def resolve_station(query: str, *, limit: int = 5) -> Envelope:
    """역 이름 → (호선, 역) 후보 목록. 모호하면 후보를 여럿 돌려준다."""
    return _resolve_station.resolve_station(query, limit=limit)


def predict(
    line: str, station: str, date: str, hour: int, weather: dict | None = None
) -> Envelope:
    """A~E 모델 예상 승차 인원 + 지평 구간."""
    return _predict.predict(line, station, date, hour, weather)


def compare(
    line: str, station: str, date: str, hour: int, models: list[str] | None = None
) -> Envelope:
    """같은 조건에서 모델 여러 개를 나란히 비교한다."""
    return _compare.compare(line, station, date, hour, models)


def history(
    line: str, station: str, date_from: str, date_to: str, hour: int | None = None
) -> Envelope:
    """과거 실측 시계열."""
    return _history.history(line, station, date_from, date_to, hour)


def aggregate(scope: dict, range: dict, metric: str = "sum") -> Envelope:
    """scope(호선·역·시간대·요일) × range(날짜 구간) 집계 + 표본 수·커버리지."""
    _require_dict("scope", scope)
    _require_dict("range", range)
    return _aggregate.aggregate(scope, range, metric)


def simulate(base: dict, scenarios: list[dict]) -> Envelope:
    """기준 조건(base)과 가정 시나리오들로 예측을 다시 계산한다."""
    return _simulate.simulate(base, scenarios)


def explain(line: str, station: str, date: str, hour: int, model: str) -> Envelope:
    """예측 하나의 근거 feature 를 모은다."""
    return _explain.explain(line, station, date, hour, model)


def reconcile(line: str, date: str, hour: int) -> Envelope:
    """그 시점의 예측과 실측을 대조한다."""
    return _reconcile.reconcile(line, date, hour)


def backtest(
    bucket: str | None = None, model: str | None = None, slice: str | None = None
) -> Envelope:
    """지평 구간·모델·슬라이스별 과거 재현 오차."""
    return _backtest.backtest(bucket, model, slice)


def drift() -> Envelope:
    """최근 분포 이동(드리프트) 지표."""
    return _drift.drift()


def alerts(scope: dict, range: dict, z: float = 3.0) -> Envelope:
    """scope × range 에서 z 를 넘는 이상치 목록."""
    _require_dict("scope", scope)
    _require_dict("range", range)
    return _alerts.alerts(scope, range, z)


def report(kind: str, date: str, sections: list[str] | None = None) -> Envelope:
    """답변·근거를 파일로 저장한다(유일한 쓰기 도구)."""
    return _report.report(kind, date, sections)