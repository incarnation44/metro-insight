"""아티팩트 나이와 데이터 공백 현황. '지금'이라는 주장을 막는 장치다.

데이터 끝 이후 경과일·성적 파일 생성일·파일 수정 시각과 지도/데이터 공백 수를
한 번에 보여 준다. 오래됐으면 ``stale`` 을 참으로 두고 답변에 '낡음' 문구를 붙인다.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from metro_insight.agent.contracts import Caveat, Envelope
from metro_insight.agent.tools._common import (
    DATA_END,
    SERVING_END,
    as_of,
    caveat,
    inputs_hash,
    map_only_stats,
    repo_root,
    today,
)

STALE_AFTER_DAYS = 30
_ARTIFACTS = (
    ("metrics_calendar.json", "generated"),
    ("metrics_sequence.json", "generated"),
    ("backtest_182_366.json", "created"),
)


def _generated(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        value = payload.get(key)
        return str(value) if value not in (None, "") else None
    return None


def drift_facts() -> tuple[dict, list[dict]]:
    """(봉투 ``data``, 한계 목록). 리포트도 같은 계산을 재사용한다."""
    reference = today()
    days = (reference - DATA_END).days
    models_dir = repo_root() / "models"
    artifacts = []
    for name, key in _ARTIFACTS:
        path = models_dir / name
        artifacts.append(
            {
                "name": name,
                "exists": path.exists(),
                "mtime": int(path.stat().st_mtime) if path.exists() else None,
                "generated": _generated(path, key),
            }
        )
    gaps = map_only_stats()
    stale_reasons = []
    if days > STALE_AFTER_DAYS:
        stale_reasons.append(f"데이터 기준일이 {days}일 전입니다")
    stale = bool(stale_reasons)
    data = {
        "data_end": DATA_END.isoformat(),
        "serving_end": SERVING_END.isoformat(),
        "reference_date": reference.isoformat(),
        "days_since_data_end": days,
        "stale": stale,
        "stale_reasons": stale_reasons,
        "stale_after_days": STALE_AFTER_DAYS,
        "artifacts": artifacts,
        "map_pairs": gaps["map_pairs"],
        "data_pairs": gaps["data_pairs"],
        "map_only_exact": gaps["map_only_exact"],
        "map_only_unresolved": gaps["map_only_unresolved"],
        "map_only_by_line": gaps["by_line"],
    }
    hints = []
    if stale:
        hints.append(
            caveat(
                Caveat.DATA_STALE,
                f"데이터 기준일이 {DATA_END.isoformat()}라 {days}일 지났습니다(낡음).",
            )
        )
    return data, hints


def drift() -> Envelope:
    """최근 분포 이동이 아니라 '아티팩트가 얼마나 낡았는지'를 돌려준다."""
    started = perf_counter()
    data, hints = drift_facts()
    return Envelope.ok_(
        "drift",
        data,
        as_of=as_of(),
        inputs_hash=inputs_hash({"tool": "drift"}),
        elapsed_ms=int((perf_counter() - started) * 1000),
        caveats=hints,
    )