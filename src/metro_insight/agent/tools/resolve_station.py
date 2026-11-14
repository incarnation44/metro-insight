"""역 이름 → (호선, 역) 후보. 모호하면 후보를 여럿 돌려주고 임의로 고르지 않는다.

지도(노선도)에 있는 역을 기준으로 찾는다. 지도에만 있고 학습 데이터에 없는 역은
``data_missing`` 으로 표시하고 가장 가까운 데이터 역을 함께 안내한다.
"""
from __future__ import annotations

from time import perf_counter

from metro_insight.agent.contracts import Caveat, Envelope, ErrorCode
from metro_insight.agent.tools._common import (
    ToolError,
    as_of,
    base_name,
    canonical_map_name,
    caveat,
    inputs_hash,
    map_pairs,
    nearest_data_station,
    resolve_availability,
)

MAX_LIMIT = 5
_SCORE = {"exact": 1.0, "alias": 0.95, "base": 0.9, "prefix": 0.7}


def _check_limit(limit: object) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ToolError(ErrorCode.E_INVALID_ARG, f"limit 은 정수여야 합니다: {limit}")
    if not 1 <= limit <= MAX_LIMIT:
        raise ToolError(ErrorCode.E_INVALID_ARG, f"limit 은 1~{MAX_LIMIT} 사이여야 합니다: {limit}")
    return limit


def _match(query: str) -> tuple[list[dict], str]:
    """검색어 → (지도 (호선, 역) 목록, match 방법). 정확 → 괄호 제거 → 접두 순."""
    wanted = canonical_map_name(query)
    exact = [item for item in map_pairs() if item["station"] == wanted]
    if exact:
        method = "exact" if wanted == query else "alias"
        return exact, method
    plain = base_name(wanted)
    loose = [item for item in map_pairs() if base_name(item["station"]) == plain]
    if loose:
        return loose, "base"
    prefix = [
        item
        for item in map_pairs()
        if item["station"].startswith(wanted) or base_name(item["station"]).startswith(plain)
    ]
    return prefix, "prefix"


def resolve_station(query: str, *, limit: int = 5) -> Envelope:
    """역 이름을 (호선, 역) 후보로 바꾼다. 같은 이름이 여러 호선에 있으면 모두 돌려준다."""
    started = perf_counter()
    try:
        text = str(query).strip()
        if not text:
            raise ToolError(ErrorCode.E_INVALID_ARG, "검색어가 비어 있습니다")
        top = _check_limit(limit)
        found, method = _match(text)
        if not found:
            raise ToolError(
                ErrorCode.E_NOT_FOUND,
                f"'{text}' 로 시작하는 역을 지도에서 찾지 못했습니다.",
            )
        candidates = []
        for item in found:
            code, match = resolve_availability(item["line"], item["station"])
            entry = {
                "line": item["line"],
                "station": item["station"],
                "code": code,
                "method": match if code is not None else method,
                "score": _SCORE.get(method, 0.7),
                "data_missing": code is None,
            }
            if code is None:
                entry["nearest_data_station"] = nearest_data_station(
                    item["line"], item["lat"], item["lon"]
                )
            candidates.append(entry)
        candidates.sort(key=lambda c: (-c["score"], c["line"], c["station"]))
        missing = [c for c in candidates if c["data_missing"]]
        ambiguous = len(candidates) > 1
        hints = []
        if ambiguous:
            hints.append(
                caveat(
                    Caveat.AMBIGUOUS_STATION,
                    f"'{text}' 는 {len(candidates)}개 (호선, 역) 후보가 있습니다. 임의로 고르지 않습니다.",
                )
            )
        if missing:
            hints.append(
                caveat(Caveat.DATA_MISSING, f"{len(missing)}개 후보는 예측 데이터가 없습니다.")
            )
        kept = candidates[:top]
        return Envelope.ok_(
            "resolve_station",
            {
                "query": text,
                "normalized": canonical_map_name(text),
                "candidates": kept,
                "count": len(kept),
                "total_candidates": len(candidates),
                "truncated": len(candidates) > len(kept),
                "ambiguous": ambiguous,
                "data_missing_count": len(missing),
            },
            as_of=as_of(),
            inputs_hash=inputs_hash({"query": text, "limit": top}),
            elapsed_ms=int((perf_counter() - started) * 1000),
            caveats=hints,
        )
    except ToolError as exc:
        return Envelope.error(
            "resolve_station", exc.code, exc.human, elapsed_ms=int((perf_counter() - started) * 1000)
        )