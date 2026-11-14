"""지평 구간·모델·슬라이스별 과거 성적(MAE·RMSE·표본 수).

값은 models/ 의 성적 파일을 그대로 읽는다(다시 계산하지 않는다). 182~366일 구간은
백테스트 근사치라 근사 표시를 함께 남기고, 파일이 없으면 0 이 아니라 사유를 돌려준다.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from metro_insight.agent.contracts import Caveat, Envelope, ErrorCode
from metro_insight.agent.tools._common import (
    VALID_MODELS,
    ToolError,
    as_of,
    caveat,
    inputs_hash,
    repo_root,
    round2,
)

BUCKETS = ("1-31", "32-91", "92-181", "182-366", "367+")
SLICES = ("overall", "tertile:소", "tertile:중", "tertile:대")


def _load(name: str) -> dict:
    path = repo_root() / "models" / name
    if not path.exists():
        raise ToolError(ErrorCode.E_UNAVAILABLE, f"성적 파일이 없습니다: models/{name}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolError(ErrorCode.E_UNAVAILABLE, f"성적 파일을 읽지 못했습니다: models/{name}") from exc
    return payload if isinstance(payload, dict) else {}


def _row(model: str, bucket: str | None, entry: dict, source: str, generated: str | None) -> dict:
    return {
        "model": model,
        "bucket": bucket,
        "mae": round2(entry.get("mae")),
        "rmse": round2(entry.get("rmse")),
        "n": int(entry.get("n", 0)) if entry.get("n") is not None else None,
        "source": source,
        "generated": generated,
    }


def backtest(
    bucket: str | None = None, model: str | None = None, slice: str | None = None
) -> Envelope:
    """지평 구간·모델·슬라이스별 과거 재현 오차."""
    started = perf_counter()
    try:
        want_bucket = None if bucket in (None, "") else str(bucket).strip()
        if want_bucket is not None and want_bucket not in BUCKETS:
            raise ToolError(
                ErrorCode.E_INVALID_ARG,
                f"모르는 지평 구간입니다: {bucket} (가능: {', '.join(BUCKETS)})",
            )
        want_model = None if model in (None, "") else str(model).strip().upper()
        if want_model is not None and want_model not in VALID_MODELS:
            raise ToolError(
                ErrorCode.E_INVALID_ARG,
                f"모르는 모델입니다: {model} (가능: {', '.join(VALID_MODELS)})",
            )
        want_slice = None if slice in (None, "") else str(slice).strip()
        if want_slice is not None and want_slice not in SLICES:
            raise ToolError(
                ErrorCode.E_INVALID_ARG,
                f"모르는 슬라이스입니다: {slice} (가능: {', '.join(SLICES)})",
            )

        calendar = _load("metrics_calendar.json")
        sequence = _load("metrics_sequence.json")
        rows: list[dict] = []
        unavailable: list[dict] = []
        hints: list[dict] = []
        sources: dict[str, str] = {}
        approx = False

        def model_filter(name: str) -> bool:
            return want_model is None or want_model == name

        if want_slice is not None and want_slice != "overall":
            tertile = want_slice.split(":", 1)[1]
            for name in ("D", "E"):
                if not model_filter(name):
                    continue
                entry = (sequence.get("test") or {}).get(name) or {}
                item = (entry.get("tertile") or {}).get(tertile)
                if isinstance(item, dict):
                    rows.append(
                        _row(name, None, item, "metrics_sequence.json", sequence.get("generated"))
                    )
                    sources["metrics_sequence.json"] = sequence.get("generated") or ""
                else:
                    unavailable.append(
                        {
                            "model": name,
                            "slice": want_slice,
                            "reason_code": ErrorCode.E_UNAVAILABLE.value,
                            "human": f"{name} 모델의 {want_slice} 성적이 없습니다.",
                        }
                    )
        elif want_bucket in (None, "367+"):
            for name in ("A", "B", "C"):
                if not model_filter(name):
                    continue
                entry = (calendar.get("test") or {}).get(name) or {}
                if entry:
                    rows.append(
                        _row(name, None, entry, "metrics_calendar.json", calendar.get("generated"))
                    )
                    sources["metrics_calendar.json"] = calendar.get("generated") or ""
            for name in ("D", "E"):
                if not model_filter(name):
                    continue
                entry = (sequence.get("test") or {}).get(name) or {}
                if entry:
                    rows.append(
                        _row(name, None, entry, "metrics_sequence.json", sequence.get("generated"))
                    )
                    sources["metrics_sequence.json"] = sequence.get("generated") or ""
            if want_bucket == "367+" and rows:
                unavailable.append(
                    {
                        "bucket": "367+",
                        "reason_code": ErrorCode.E_UNAVAILABLE.value,
                        "human": "367일 이후 구간은 검증 성적이 없습니다(위 값은 전체 성적).",
                    }
                )
        else:
            for name in ("A", "B", "C"):
                if not model_filter(name):
                    continue
                entry = (calendar.get("test") or {}).get(name) or {}
                item = (entry.get("horizon") or {}).get(want_bucket)
                if isinstance(item, dict):
                    rows.append(
                        _row(name, want_bucket, item, "metrics_calendar.json", calendar.get("generated"))
                    )
                    sources["metrics_calendar.json"] = calendar.get("generated") or ""
            for name in ("D", "E"):
                if not model_filter(name):
                    continue
                entry = (sequence.get("test") or {}).get(name) or {}
                item = (entry.get("horizon") or {}).get(want_bucket)
                if isinstance(item, dict):
                    rows.append(
                        _row(name, want_bucket, item, "metrics_sequence.json", sequence.get("generated"))
                    )
                    sources["metrics_sequence.json"] = sequence.get("generated") or ""
            if want_bucket == "182-366":
                rolled = _load("backtest_182_366.json")
                pooled = rolled.get("pooled") or {}
                for name in ("A", "B", "C"):
                    if not model_filter(name):
                        continue
                    entry = (pooled.get("models") or {}).get(name) or {}
                    if not entry:
                        continue
                    rows.append(
                        _row(
                            name,
                            want_bucket,
                            {"mae": entry.get("mae_weighted"), "n": entry.get("n")},
                            "backtest_182_366.json",
                            rolled.get("created"),
                        )
                    )
                    sources["backtest_182_366.json"] = rolled.get("created") or ""
                approx = True
                hints.append(
                    caveat(
                        Caveat.DATA_STALE,
                        "182~366일 구간 값은 롤링 백테스트 근사치이며 Test 점수가 아닙니다.",
                    )
                )

        if not rows:
            raise ToolError(
                ErrorCode.E_UNAVAILABLE,
                f"조건(bucket={bucket}, model={model}, slice={slice})에 맞는 성적이 없습니다.",
            )
        return Envelope.ok_(
            "backtest",
            {
                "rows": rows,
                "sources": dict(sorted(sources.items())),
                "approx": approx,
                "unavailable": unavailable,
                "query": {"bucket": want_bucket, "model": want_model, "slice": want_slice},
            },
            as_of=as_of(),
            inputs_hash=inputs_hash({"bucket": want_bucket, "model": want_model, "slice": want_slice}),
            elapsed_ms=int((perf_counter() - started) * 1000),
            caveats=hints,
        )
    except ToolError as exc:
        return Envelope.error(
            "backtest", exc.code, exc.human, elapsed_ms=int((perf_counter() - started) * 1000)
        )
    except Exception as exc:  # noqa: BLE001 — 예기치 못한 실패도 사유를 남긴다
        return Envelope.error(
            "backtest",
            ErrorCode.E_INTERNAL,
            f"성적 조회 오류: {type(exc).__name__}: {exc}",
            elapsed_ms=int((perf_counter() - started) * 1000),
        )


def metrics_paths() -> tuple[Path, ...]:
    """성적 파일 경로(진단용)."""
    root = repo_root() / "models"
    return tuple(
        root / name
        for name in ("metrics_calendar.json", "metrics_sequence.json", "backtest_182_366.json")
    )