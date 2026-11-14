"""manifest 검증 — 필수 필드·타입·run_id 정규식·피처 순서 지문 재계산.

사용 (프로젝트 루트에서)
    python notebooks/verify_manifest.py models/_exp/20261114-P0-R0-01
    python notebooks/verify_manifest.py models/_exp/20261114-P0-R0-01/manifest.json

종료 코드
    0 = 통과, 1 = 실패(문제를 한 줄씩 출력)

eval_protocol.py 를 import 하지 않는다 — 같은 규칙을 독립으로 다시 구현해야 검증기 노릇을 한다.
규칙 문자열은 manifest.features.order_hash_rule 에 적혀 있고, 여기서 아는 값과 다르면 실패한다.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

RUN_ID_RE = re.compile(r"^[0-9]{8}-P[0-9]+-R[0-9]+-[0-9]{2}$")
MANIFEST_SCHEMA = "metro-manifest/1"
PROTOCOL_ID = "rolling8-v1"
HASH_RULE = "sha256-newline-join-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_TOP = [
    "schema", "run_id", "created_at", "git", "task", "horizon", "data", "features",
    "params", "seeds", "protocol", "metrics", "verdict", "artifacts",
]
PER_CUT_KEYS = ["cut_date", "n", "mae", "rmse", "wape", "bias", "negative_predictions",
                "horizon_bucket", "date_min", "date_max"]
POOLED_KEYS = ["n", "mae", "rmse", "wape", "bias"]
DATA_KEYS = ["path", "sha256", "rows", "date_min", "date_max"]


def feature_order_hash(names: list[str]) -> str:
    return hashlib.sha256((HASH_RULE + "\n" + "\n".join(names)).encode("utf-8")).hexdigest()


def check_manifest(doc: dict) -> list[str]:
    """문제 목록을 돌려준다(빈 목록이면 통과)."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["manifest 가 JSON 객체가 아니다"]

    for key in REQUIRED_TOP:
        if key not in doc:
            problems.append(f"필수 필드 없음: {key}")
    if problems:
        return problems

    if doc["schema"] != MANIFEST_SCHEMA:
        problems.append(f"schema 불일치: {doc['schema']!r} (기대 {MANIFEST_SCHEMA})")
    if not isinstance(doc["run_id"], str) or not RUN_ID_RE.match(doc["run_id"]):
        problems.append(f"run_id 형식 오류: {doc['run_id']!r} (기대 YYYYMMDD-Pn-Rn-NN)")
    if not isinstance(doc["created_at"], str) or not doc["created_at"].strip():
        problems.append("created_at 는 비어 있지 않은 문자열이어야 한다")
    else:
        try:
            datetime.fromisoformat(doc["created_at"])
        except ValueError:
            problems.append(f"created_at 이 ISO 시각이 아니다: {doc['created_at']!r}")

    git = doc["git"]
    if not isinstance(git, dict):
        problems.append("git 은 객체여야 한다")
    else:
        if "commit" not in git:
            problems.append("git.commit 없음")
        elif not (git["commit"] is None or isinstance(git["commit"], str)):
            problems.append("git.commit 은 문자열 또는 null")
        if not isinstance(git.get("dirty"), bool) and git.get("dirty") is not None:
            problems.append("git.dirty 는 참·거짓 또는 비어 있음(null)")

    if doc["task"] not in ("T1", "T2"):
        problems.append(f"task 는 T1 또는 T2: {doc['task']!r}")
    horizon = doc["horizon"]
    if not isinstance(horizon, dict) or not isinstance(horizon.get("min"), int) \
            or not isinstance(horizon.get("max"), int):
        problems.append("horizon 은 {min:int, max:int} 이어야 한다")
    if doc["protocol"] != PROTOCOL_ID:
        problems.append(f"protocol 불일치: {doc['protocol']!r} (기대 {PROTOCOL_ID})")

    data = doc["data"]
    if not isinstance(data, dict) or not data:
        problems.append("data 는 비어 있지 않은 객체여야 한다")
    else:
        for split, meta in data.items():
            if not isinstance(meta, dict):
                problems.append(f"data.{split} 는 객체여야 한다")
                continue
            for key in DATA_KEYS:
                if key not in meta:
                    problems.append(f"data.{split}.{key} 없음")
            sha = meta.get("sha256")
            if sha is not None and not (isinstance(sha, str) and HEX64.match(sha)):
                problems.append(f"data.{split}.sha256 은 64자 16진수 또는 비어 있음(null)")
            if "rows" in meta and not isinstance(meta["rows"], int):
                problems.append(f"data.{split}.rows 는 int")

    features = doc["features"]
    if not isinstance(features, dict):
        problems.append("features 는 객체여야 한다")
    else:
        names = features.get("names")
        if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
            problems.append("features.names 는 비어 있지 않은 문자열 목록")
        elif features.get("order_hash_rule") != HASH_RULE:
            problems.append(f"features.order_hash_rule 을 모른다: {features.get('order_hash_rule')!r}")
        else:
            expected = feature_order_hash(names)
            if features.get("order_hash") != expected:
                problems.append(
                    f"features.order_hash 재계산 불일치 — 적힌 값 {str(features.get('order_hash'))[:12]}… "
                    f"재계산 {expected[:12]}… (피처 순서/목록이 바뀌었다)"
                )
        if not features.get("as_of_rule"):
            problems.append("features.as_of_rule 없음")

    if not isinstance(doc["params"], dict) or not doc["params"]:
        problems.append("params 는 비어 있지 않은 객체")
    if not isinstance(doc["seeds"], dict) or not doc["seeds"]:
        problems.append("seeds 는 비어 있지 않은 객체")

    metrics = doc["metrics"]
    if not isinstance(metrics, dict):
        problems.append("metrics 는 객체여야 한다")
    else:
        per_cut = metrics.get("per_cut")
        if not isinstance(per_cut, list) or not per_cut:
            problems.append("metrics.per_cut 은 비어 있지 않은 목록")
        else:
            for i, entry in enumerate(per_cut):
                if not isinstance(entry, dict):
                    problems.append(f"metrics.per_cut[{i}] 는 객체")
                    continue
                for key in PER_CUT_KEYS:
                    if key not in entry:
                        problems.append(f"metrics.per_cut[{i}].{key} 없음")
                if "mae" in entry and not isinstance(entry["mae"], (int, float)):
                    problems.append(f"metrics.per_cut[{i}].mae 는 숫자")
                if "n" in entry and not isinstance(entry["n"], int):
                    problems.append(f"metrics.per_cut[{i}].n 은 int")
        agg = metrics.get("pooled")
        if not isinstance(agg, dict):
            problems.append("metrics.pooled 는 객체")
        else:
            for key in POOLED_KEYS:
                if key not in agg:
                    problems.append(f"metrics.pooled.{key} 없음")
        if not isinstance(metrics.get("ci"), dict):
            problems.append("metrics.ci 는 객체")

    verdict = doc["verdict"]
    if not isinstance(verdict, dict) or not verdict.get("result"):
        problems.append("verdict.result 없음")
    if not isinstance(doc["artifacts"], dict) or not doc["artifacts"]:
        problems.append("artifacts 는 비어 있지 않은 객체")

    return problems


def resolve_manifest(path: str | Path) -> Path:
    target = Path(path)
    if target.is_dir():
        target = target / "manifest.json"
    return target


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("사용: python notebooks/verify_manifest.py <run 디렉터리 또는 manifest.json>", file=sys.stderr)
        return 1
    path = resolve_manifest(argv[0])
    if not path.exists():
        print(f"FAIL manifest 가 없다: {path}", file=sys.stderr)
        return 1
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"FAIL JSON 파싱 실패: {exc}", file=sys.stderr)
        return 1

    problems = check_manifest(doc)
    if problems:
        print(f"FAIL {path} — 문제 {len(problems)}개", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    features = doc["features"]
    print(f"OK {path}")
    print(f"  run_id={doc['run_id']} · protocol={doc['protocol']} · task={doc['task']} · "
          f"기준일 {len(doc['metrics']['per_cut'])}개 · "
          f"통합 평균절대오차={doc['metrics']['pooled']['mae']:.2f}")
    print(f"  features {len(features['names'])}개 · order_hash={features['order_hash'][:12]}… (재계산 일치)")
    print(f"  판정={doc['verdict'].get('result')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())