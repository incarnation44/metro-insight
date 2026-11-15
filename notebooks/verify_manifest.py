"""manifest 검증 — 필수 필드·타입·run_id 정규식·피처 순서 지문 재계산·실제 파일 해시 대조.

사용 (프로젝트 루트에서)
    python notebooks/verify_manifest.py
    python notebooks/verify_manifest.py models/_exp/20261114-P0-R0-01
    python notebooks/verify_manifest.py models/_exp/20261114-P0-R0-01/manifest.json

인자 없이 실행하면 models/_exp 아래 모든 실행(manifest.json 이 있는 디렉터리)을 차례로 검증한다.

schema 로 검증 경로를 나눈다.
    * ``metro-manifest/1``  — 평가 프로토콜 실행(필수 필드·피처 지문·산출물 해시·입력 데이터 해시).
    * ``conformal-manifest/1`` — 구간 보정 실행(목록 형 산출물 해시·크기, log.path 존재, run_id 와
      디렉터리명 일치). 과거 실행의 신원 불일치·로그 부재·설정 누락은 **경고**로 남기고 실패로
      만들지 않는다(이름·내용을 고치지 않고 보존한다).
    * 그 밖의 schema 는 통과가 아니라 실패 1건으로 보고한다(모르는 것을 통과로 세지 않는다).

종료 코드
    0 = 통과, 1 = 실패(문제를 한 줄씩 출력)

eval_protocol.py 를 import 하지 않는다 — 같은 규칙을 독립으로 다시 구현해야 검증기 노릇을 한다.
규칙 문자열은 manifest.features.order_hash_rule 에 적혀 있고, 여기서 아는 값과 다르면 실패한다.

해시 대조는 기록값과 실측값을 그대로 비교한다. 해시가 기록되지 않은 항목은 통과로 넘기지 않고
'누락'으로 보고한다(대조하지 못한 것을 통과로 세지 않는다).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

RUN_ID_RE = re.compile(r"^[0-9]{8}-P[0-9]+-R[0-9]+-[0-9]{2}$")
MANIFEST_SCHEMA = "metro-manifest/1"
CONFORMAL_SCHEMA = "conformal-manifest/1"
PROTOCOL_ID = "rolling8-v1"
HASH_RULE = "sha256-newline-join-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_NAME = "manifest.json"
METRICS_TOL = 1e-4
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path("models") / "_exp"

REQUIRED_TOP = [
    "schema", "run_id", "created_at", "git", "task", "horizon", "data", "features",
    "params", "seeds", "protocol", "metrics", "verdict", "artifacts",
]
CONFORMAL_REQUIRED = [
    "schema", "run_id", "protocol", "created_at", "data", "config", "cuts", "artifacts",
]
PER_CUT_KEYS = ["cut_date", "n", "mae", "rmse", "wape", "bias", "negative_predictions",
                "horizon_bucket", "date_min", "date_max"]
POOLED_KEYS = ["n", "mae", "rmse", "wape", "bias"]
DATA_KEYS = ["path", "sha256", "rows", "date_min", "date_max"]


def feature_order_hash(names: list[str]) -> str:
    return hashlib.sha256((HASH_RULE + "\n" + "\n".join(names)).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """파일 실측 해시 — 1MB 씩 읽어 메모리를 아낀다."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_entry(path: Path, recorded, label: str, problems: list[str], counter: dict) -> None:
    """한 항목의 기록 해시와 실측 해시를 맞춰 보고한다(대조 불가는 누락으로 센다)."""
    counter["total"] += 1
    if not isinstance(recorded, str) or not HEX64.match(recorded):
        problems.append(f"{label}.sha256 누락 — 실측 해시와 대조할 값이 없다")
        return
    if not path.exists():
        problems.append(f"{label} 파일이 없다: {path}")
        return
    measured = sha256_file(path)
    if measured != recorded:
        problems.append(
            f"{label} 해시 불일치 — 기록 {recorded[:12]}… 실측 {measured[:12]}… ({path.name})"
        )
        return
    counter["matched"] += 1


def verify_hashes(doc: dict, base_dir: str | Path,
                  repo_root: str | Path | None = None) -> tuple[list[str], dict]:
    """산출물·입력 데이터의 실측 해시를 manifest 기록값과 대조한다.

    base_dir 는 manifest 가 있는 디렉터리, repo_root 는 data 경로의 기준 디렉터리다.
    manifest 자신은 자기 해시를 담을 수 없으므로 대조 대상에서 뺀다.
    """
    base_dir = Path(base_dir)
    problems: list[str] = []
    artifacts_count = {"matched": 0, "total": 0}
    data_count = {"matched": 0, "total": 0}

    artifacts = doc.get("artifacts")
    if isinstance(artifacts, dict):
        for name in sorted(artifacts):
            meta = artifacts[name]
            if not isinstance(meta, dict) or not isinstance(meta.get("path"), str):
                problems.append(f"artifacts.{name}.path 없음")
                continue
            recorded = meta.get("sha256")
            # manifest 자신은 자기 해시를 담을 수 없다 — 항목 이름과 경로가 정확히 manifest 일 때만 뺀다.
            # 이름만 보면 하위 경로의 임의 manifest.json 이 대조 없이 통과한다.
            if recorded is None and name == "manifest" and meta["path"] == MANIFEST_NAME:
                continue
            _hash_entry(base_dir / meta["path"], recorded,
                        f"artifacts.{name}", problems, artifacts_count)
    else:
        # 목록(list) 형 산출물을 조용히 0/0 으로 통과시키지 않는다 — 대조 불가를 문제로 남긴다
        problems.append(
            f"artifacts 스키마 불일치 — 객체가 아니라 {type(artifacts).__name__} 이다(해시 대조 불가)"
        )

    data = doc.get("data")
    if isinstance(data, dict) and repo_root is not None:
        root = Path(repo_root)
        for split in sorted(data):
            meta = data[split]
            if not isinstance(meta, dict) or not isinstance(meta.get("path"), str):
                problems.append(f"data.{split}.path 없음")
                continue
            _hash_entry(root / meta["path"], meta.get("sha256"),
                        f"data.{split}", problems, data_count)

    return problems, {"artifacts": artifacts_count, "data": data_count}


def verify_conformal_hashes(doc: dict, base_dir: str | Path) -> tuple[list[str], dict]:
    """conformal manifest 의 산출물(목록 형) 해시·크기를 실측값과 대조한다."""
    problems: list[str] = []
    count = {"matched": 0, "total": 0}
    base_dir = Path(base_dir)
    artifacts = doc.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        problems.append(
            f"artifacts 스키마 불일치 — 비어 있지 않은 목록이어야 한다({type(artifacts).__name__})"
        )
        return problems, count
    for i, entry in enumerate(artifacts):
        if not isinstance(entry, dict):
            problems.append(f"artifacts[{i}] 는 객체여야 한다")
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            problems.append(f"artifacts[{i}].path 가 없다")
            continue
        target = base_dir / path
        if target.exists() and isinstance(entry.get("bytes"), int):
            measured = target.stat().st_size
            if measured != entry["bytes"]:
                count["total"] += 1
                problems.append(
                    f"artifacts[{i}] 크기 불일치 — 기록 {entry['bytes']} 실측 {measured} ({path})"
                )
                continue
        _hash_entry(target, entry.get("sha256"), f"artifacts[{i}]", problems, count)
    return problems, count


def _numeric(value) -> float | None:
    """숫자만 float 로 — bool 은 숫자로 세지 않는다(참/거짓을 1/0 으로 오독하지 않게)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _collect_mismatch(declared, actual, *, prefix: str, out: list[str],
                      tol: float = METRICS_TOL) -> None:
    """manifest 기록값과 파일 내용을 허용오차로 깊이 비교한다(불일치 경로만 남긴다)."""
    if isinstance(declared, dict) and isinstance(actual, dict):
        for key in sorted(set(declared) | set(actual)):
            if key not in declared or key not in actual:
                out.append(f"{prefix}.{key} 가 manifest 와 metrics.json 에서 다르다")
                continue
            _collect_mismatch(declared[key], actual[key], prefix=f"{prefix}.{key}", out=out, tol=tol)
        return
    if isinstance(declared, list) and isinstance(actual, list):
        if len(declared) != len(actual):
            out.append(f"{prefix} 길이 불일치 — manifest {len(declared)} ≠ metrics.json {len(actual)}")
            return
        for i, (left, right) in enumerate(zip(declared, actual, strict=False)):
            _collect_mismatch(left, right, prefix=f"{prefix}[{i}]", out=out, tol=tol)
        return
    declared_num = _numeric(declared)
    actual_num = _numeric(actual)
    if declared_num is not None and actual_num is not None:
        if abs(declared_num - actual_num) > tol:
            out.append(f"{prefix} 불일치 — manifest {declared_num:.6g} ≠ metrics.json {actual_num:.6g}")
        return
    if declared != actual or isinstance(declared, bool) != isinstance(actual, bool):
        out.append(f"{prefix} 불일치 — manifest {declared!r} ≠ metrics.json {actual!r}")


def check_metrics_arithmetic(metrics: dict) -> list[str]:
    """per_cut 행을 n 가중으로 다시 합쳐 pooled 와 맞는지 검산한다.

    해시는 위조 뒤 다시 맞출 수 있지만, cut 별 값과 통합 값의 산술 관계는 맞추기 어렵다.
    그래서 이 검산이 위조·손상의 마지막 그물이다. 형식이 어긋나 검산할 수 없는 항목은
    반쪽 검산으로 통과를 주장하지 않고 넘긴다(형식 문제는 check_manifest 가 따로 보고한다).
    """
    if not isinstance(metrics, dict):
        return []
    per_cut = metrics.get("per_cut")
    pooled = metrics.get("pooled")
    if not isinstance(per_cut, list) or not isinstance(pooled, dict):
        return []
    rows: list[tuple[int, float, float]] = []
    for entry in per_cut:
        if not isinstance(entry, dict):
            return []
        n = entry.get("n")
        mae = _numeric(entry.get("mae"))
        bias = _numeric(entry.get("bias"))
        if not isinstance(n, int) or n < 0 or mae is None or bias is None:
            return []
        rows.append((n, mae, bias))
    if len(rows) != len(per_cut) or not rows:
        return []

    problems: list[str] = []
    total = sum(row[0] for row in rows)
    pooled_n = pooled.get("n")
    if isinstance(pooled_n, int) and pooled_n != total:
        problems.append(
            f"metrics.pooled.n={pooled_n:,} 이 per_cut n 합 {total:,} 과 다르다(위조/손상)"
        )
    if total > 0:
        weighted_mae = sum(n * mae for n, mae, _ in rows) / total
        weighted_bias = sum(n * bias for n, _, bias in rows) / total
        pooled_mae = _numeric(pooled.get("mae"))
        pooled_bias = _numeric(pooled.get("bias"))
        if pooled_mae is not None and abs(weighted_mae - pooled_mae) > METRICS_TOL:
            problems.append(
                f"metrics.pooled.mae={pooled_mae:.6f} 이 n 가중 평균 {weighted_mae:.6f} 과 다르다(위조/손상)"
            )
        if pooled_bias is not None and abs(weighted_bias - pooled_bias) > METRICS_TOL:
            problems.append(
                f"metrics.pooled.bias={pooled_bias:.6f} 이 n 가중 평균 {weighted_bias:.6f} 과 "
                "다르다(위조/손상)"
            )
    return problems


def check_metrics_consistency(doc: dict, base_dir: str | Path) -> list[str]:
    """manifest.metrics · metrics.json · 산술 관계를 서로 대조한다.

    * metrics.json 이 있으면 manifest 기록값과 허용오차(1e-4)로 비교한다.
    * 어느 쪽이든 per_cut → pooled 산술 관계를 다시 계산한다.
    해시가 맞아도 내용이 위조·손상되면 여기서 걸린다.
    """
    if not isinstance(doc, dict):
        return []
    declared = doc.get("metrics")
    artifacts = doc.get("artifacts")
    meta = artifacts.get("metrics") if isinstance(artifacts, dict) else None
    if not isinstance(meta, dict) or not isinstance(meta.get("path"), str):
        return check_metrics_arithmetic(declared) if isinstance(declared, dict) else []

    path = Path(base_dir) / meta["path"]
    if not path.exists():
        # 파일 부재는 해시 대조가 잡는다. 여기서는 manifest 산술만 검산한다.
        return check_metrics_arithmetic(declared) if isinstance(declared, dict) else []
    try:
        with open(path, encoding="utf-8") as f:
            file_metrics = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"metrics.json 을 읽지 못했다: {path.name} — {exc}"]
    if not isinstance(file_metrics, dict):
        return [f"metrics.json 이 JSON 객체가 아니다: {path.name}"]

    problems: list[str] = []
    if isinstance(declared, dict):
        mismatches: list[str] = []
        _collect_mismatch(declared, file_metrics, prefix="metrics", out=mismatches)
        if len(mismatches) > 10:
            head = mismatches[:10]
            head.append(f"… 그 밖에 불일치 {len(mismatches) - 10}건")
            mismatches = head
        problems.extend(
            f"manifest.metrics 와 metrics.json 불일치 — {line}" for line in mismatches
        )
    problems.extend(check_metrics_arithmetic(file_metrics))
    return problems


def _check_conformal_conditional_csv(
    base_dir: Path,
    selected_variant: str | None,
    cut_rows: list[tuple[str, int, float]],
) -> list[str]:
    """conditional_coverage.csv 의 slice=all 및 분할 합이 per_cut 과 맞는지 검증한다."""
    csv_path = base_dir / "conditional_coverage.csv"
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            by_cut: dict[str, dict[str, tuple[int, float | None]]] = {}
            for row in reader:
                variant = row.get("variant")
                if selected_variant and variant and variant != selected_variant:
                    continue
                cut = str(row.get("cut") or "")
                slc = str(row.get("slice") or "")
                if not cut or not slc:
                    continue
                n_raw = row.get("n")
                if n_raw in (None, ""):
                    continue
                n_val = int(float(n_raw))
                picp_raw = row.get("picp")
                picp_val = float(picp_raw) if picp_raw not in (None, "") else None
                by_cut.setdefault(cut, {})[slc] = (n_val, picp_val)
    except (OSError, ValueError) as exc:
        return [f"conditional_coverage.csv 를 읽지 못했다 — {exc}"]

    if not by_cut:
        return []

    problems: list[str] = []
    for cut, n, picp in cut_rows:
        slices = by_cut.get(cut)
        if not slices:
            problems.append(f"conditional_coverage.csv 에 cut={cut} 행이 없다")
            continue
        all_entry = slices.get("all")
        if all_entry is not None:
            csv_n, csv_picp = all_entry
            if csv_n != n:
                problems.append(
                    f"conditional_coverage.csv[{cut}].all.n={csv_n:,} 이 "
                    f"metrics.per_cut.n={n:,} 과 다르다(위조/손상)"
                )
            if csv_picp is not None and abs(csv_picp - picp) > 1e-6:
                problems.append(
                    f"conditional_coverage.csv[{cut}].all.picp={csv_picp:.6f} 이 "
                    f"metrics.per_cut.picp={picp:.6f} 과 다르다(위조/손상)"
                )
        for prefix in ("day_type::", "scale_tertile::", "line::"):
            parts = [v[0] for k, v in slices.items() if k.startswith(prefix)]
            if parts and sum(parts) != n:
                problems.append(
                    f"conditional_coverage.csv[{cut}] {prefix}* 표본 합 {sum(parts):,} 이 "
                    f"per_cut.n={n:,} 과 다르다(위조/손상)"
                )
    return problems


def _check_conformal_base_run_n(
    base_dir: Path,
    manifest: dict | None,
    cut_rows: list[tuple[str, int, float]],
) -> list[str]:
    """base_run 이 로컬에 있으면 cut 별 평가 표본수(n)가 원 실행과 일치하는지 대조한다."""
    if not isinstance(manifest, dict):
        return []
    base_run = manifest.get("base_run")
    if not isinstance(base_run, str) or not base_run.strip():
        return []
    base_metrics_path = base_dir.parent / base_run / "metrics.json"
    if not base_metrics_path.exists():
        return []
    try:
        with open(base_metrics_path, encoding="utf-8") as f:
            base_doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    base_per_cut = base_doc.get("per_cut") if isinstance(base_doc, dict) else None
    base_n_map: dict[str, int] = {}
    if isinstance(base_per_cut, list):
        for entry in base_per_cut:
            if isinstance(entry, dict) and isinstance(entry.get("cut"), str) and isinstance(entry.get("n"), int):
                base_n_map[entry["cut"]] = entry["n"]
    elif isinstance(base_per_cut, dict):
        for k, v in base_per_cut.items():
            if isinstance(v, dict) and isinstance(v.get("n"), int):
                base_n_map[str(k)] = v["n"]
    if not base_n_map:
        return []

    problems: list[str] = []
    for cut, n, _ in cut_rows:
        if cut in base_n_map and base_n_map[cut] != n:
            problems.append(
                f"per_cut[{cut}].n={n:,} 이 base_run({base_run}) 표본수 {base_n_map[cut]:,} 과 "
                "다르다(위조/손상)"
            )
    return problems


def check_conformal_metrics(
    base_dir: str | Path,
    *,
    manifest: dict | None = None,
) -> list[str]:
    """구간 보정 실행의 metrics.json 에서 cut 별 PICP·n·gate 판정이 맞는지 본다.

    per_cut 의 n 가중 PICP 가 게이트의 밴드·통과 목록·conditional_coverage.csv·base_run 과
    어긋나면 위조·손상으로 본다.
    과거 실행(2026-11-15 P2·P3)은 per_cut·gates 가 모두 서로 맞으므로 그대로 통과한다.
    """
    base_path = Path(base_dir)
    path = base_path / "metrics.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"metrics.json 을 읽지 못했다: {path.name} — {exc}"]
    if not isinstance(doc, dict):
        return [f"metrics.json 이 JSON 객체가 아니다: {path.name}"]

    per_cut = doc.get("per_cut")
    gates = doc.get("gates")
    if not isinstance(per_cut, dict) or not per_cut or not isinstance(gates, dict):
        return ["metrics.json 에 per_cut·gates 가 없다 — 가중 PICP 를 대조할 수 없다"]

    problems: list[str] = []
    rows: list[tuple[str, int, float]] = []
    for cut, rec in per_cut.items():
        if not isinstance(rec, dict):
            problems.append(f"metrics.per_cut[{cut}] 가 객체가 아니다")
            continue
        n = rec.get("n")
        picp = _numeric(rec.get("picp"))
        if not isinstance(n, int) or n <= 0 or picp is None:
            problems.append(f"metrics.per_cut[{cut}] 에 n(양의 정수)·picp(숫자) 가 없다")
            continue
        if not math.isfinite(picp) or not 0.0 <= picp <= 1.0:
            problems.append(f"metrics.per_cut[{cut}].picp={picp!r} 이 포함률 범위(0~1) 밖이다")
            continue
        rows.append((str(cut), n, picp))
    if problems:
        return problems

    total = sum(n for _, n, _ in rows)
    weighted = sum(n * picp for _, n, picp in rows) / total
    if not math.isfinite(weighted) or not 0.0 <= weighted <= 1.0:
        problems.append(f"가중 PICP {weighted!r} 이 포함률 범위(0~1) 밖이다")

    band = gates.get("picp_band")
    if isinstance(band, list) and len(band) == 2:
        low, high = _numeric(band[0]), _numeric(band[1])
        if low is not None and high is not None and low <= high:
            for cut, _, picp in rows:
                rec = per_cut.get(cut, {})
                if isinstance(rec, dict) and "in_band" in rec:
                    expected_in_band = low <= picp <= high
                    if bool(rec["in_band"]) != expected_in_band:
                        problems.append(
                            f"per_cut[{cut}].in_band={rec['in_band']!r} 가 "
                            f"picp={picp:.4f} (밴드 [{low}, {high}]) 판정과 다르다"
                        )
            out = [cut for cut, _, picp in rows if not low <= picp <= high]
            declared_out = gates.get("picp_out_of_band_cuts")
            if isinstance(declared_out, list) and sorted(map(str, declared_out)) != sorted(out):
                problems.append(
                    "gates.picp_out_of_band_cuts 가 per_cut 포함률과 다르다 — "
                    f"기록 {sorted(map(str, declared_out))} ≠ 재계산 {sorted(out)}"
                )
            if "picp_all_cuts_in_band" in gates and bool(gates["picp_all_cuts_in_band"]) != (not out):
                problems.append("gates.picp_all_cuts_in_band 가 per_cut 판정과 다르다")
            if not out and not low <= weighted <= high:
                problems.append(
                    f"가중 PICP {weighted:.4f} 가 밴드 [{low}, {high}] 밖이다 — cut 값이 모두 안인데 "
                    "가중 평균이 밖이면 n 이나 picp 가 위조·손상됐다"
                )
        # 밴드가 있으면 명목값과 비교해 남은 진단만 남긴다(불일치는 실패로 만들지 않는다)
    nominal = _numeric(gates.get("nominal"))
    if nominal is not None and abs(weighted - nominal) > 0.5:
        problems.append(
            f"가중 PICP {weighted:.4f} 가 명목 {nominal:.2f} 에서 크게 벗어났다(차 {weighted - nominal:+.4f})"
        )

    selected_variant = doc.get("selected_variant")
    problems.extend(
        _check_conformal_conditional_csv(
            base_path,
            str(selected_variant) if isinstance(selected_variant, str) else None,
            rows,
        )
    )
    problems.extend(_check_conformal_base_run_n(base_path, manifest, rows))
    return problems


def check_conformal(doc: dict, *, base_dir: str | Path,
                    dir_name: str | None = None) -> tuple[list[str], list[str]]:
    """conformal manifest 계약을 확인한다 — (문제, 경고)를 돌려준다.

    문제는 실패(종료 코드 1), 경고는 기록만 한다. 과거 실행 디렉터리는 이름·내용을 고치지
    않고 보존하므로 run_id 불일치와 로그 파일 부재는 경고로 남긴다.
    """
    problems: list[str] = []
    warnings: list[str] = []
    if not isinstance(doc, dict):
        return ["manifest 가 JSON 객체가 아니다"], warnings

    for key in CONFORMAL_REQUIRED:
        if key not in doc:
            problems.append(f"필수 필드 없음: {key}")
    if doc.get("schema") != CONFORMAL_SCHEMA:
        problems.append(f"schema 불일치: {doc.get('schema')!r} (기대 {CONFORMAL_SCHEMA})")

    run_id = doc.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        problems.append(f"run_id 형식 오류: {run_id!r} (기대 YYYYMMDD-Pn-Rn-NN)")
    elif dir_name is not None and run_id != str(dir_name):
        warnings.append(
            f"run_id({run_id}) 가 디렉터리명({dir_name}) 과 다르다 — "
            "과거 기록이라 이름을 고치지 않고 보존한다"
        )

    data = doc.get("data")
    if not isinstance(data, dict):
        problems.append("data 는 객체여야 한다")
    elif data.get("test_accessed") is not False:
        problems.append("data.test_accessed 는 False 여야 한다")

    cuts = doc.get("cuts")
    if not isinstance(cuts, list) or not cuts:
        problems.append("cuts 는 비어 있지 않은 목록이어야 한다")

    config = doc.get("config")
    if not isinstance(config, dict):
        problems.append("config 는 객체여야 한다")
    elif "weighting" not in config:
        warnings.append(
            "config.weighting 없음 — 시간 감쇠 가중 도입(2026-11-15) 이전 기록이라 그대로 보존한다"
        )

    log = doc.get("log")
    if not isinstance(log, dict) or not isinstance(log.get("path"), str):
        problems.append("log.path 없음")
    elif not (Path(base_dir) / log["path"]).exists():
        warnings.append(f"log.path 가 실제 파일이 아니다: {log['path']} — 로그 없는 실행으로 기록한다")

    hash_problems, _ = verify_conformal_hashes(doc, base_dir)
    problems.extend(hash_problems)
    problems.extend(check_conformal_metrics(base_dir, manifest=doc))
    return problems, warnings


def check_manifest(doc: dict, *, base_dir: str | Path | None = None,
                   repo_root: str | Path | None = None) -> list[str]:
    """문제 목록을 돌려준다(빈 목록이면 통과).

    base_dir 를 주면 산출물 파일 해시를, repo_root 를 주면 입력 데이터 해시를 실제로 대조한다.
    """
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
    elif base_dir is not None:
        hash_problems, _ = verify_hashes(doc, base_dir, repo_root)
        problems.extend(hash_problems)
        problems.extend(check_metrics_consistency(doc, base_dir))

    return problems


def check_manifest_with_warnings(
    doc: dict, *, base_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """(문제, 경고)를 함께 돌려준다.

    문제는 실패(종료 코드 1)이고, 경고는 기록만 한다. ``git.dirty=True`` · ``git.commit=null``
    은 과거 실행 9개가 그 상태로 남아 있으므로 실패로 만들지 않고 경고로 분리한다
    (통과를 주장하지 않도록 경고는 항상 출력한다).
    """
    problems = check_manifest(doc, base_dir=base_dir, repo_root=repo_root)
    warnings: list[str] = []
    git = doc.get("git") if isinstance(doc, dict) else None
    if isinstance(git, dict):
        if git.get("dirty") is True:
            warnings.append(
                "git.dirty=True — 커밋되지 않은 변경 위에서 만든 실행이다. 수치는 그대로 두고 "
                "이 사실을 함께 읽는다"
            )
        if git.get("commit") is None:
            warnings.append(
                "git.commit=null — 커밋 해시가 없어 어떤 코드로 만들었는지 git 이력으로 확인할 수 없다"
            )
    return problems, warnings


def resolve_manifest(path: str | Path) -> Path:
    target = Path(path)
    if target.is_dir():
        target = target / "manifest.json"
    return target


def discover_runs(root: str | Path = REPO_ROOT) -> list[Path]:
    """models/_exp 아래 실행 디렉터리 — manifest.json 이 있는 곳만 검증 대상으로 본다."""
    base = Path(root) / EXPERIMENT_DIR
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if (p / MANIFEST_NAME).is_file())


def _report_conformal(path: Path, doc: dict) -> bool:
    """conformal 산출물(구간 보정 실행)을 그 계약으로 검증한다."""
    problems, warnings = check_conformal(doc, base_dir=path.parent, dir_name=path.parent.name)
    if problems:
        print(f"FAIL {path} — 문제 {len(problems)}개", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return False
    _, count = verify_conformal_hashes(doc, path.parent)
    config = doc.get("config", {})
    print(f"OK {path}")
    print(f"  run_id={doc['run_id']} · schema={CONFORMAL_SCHEMA} · protocol={doc['protocol']} · "
          f"가중={config.get('weighting')} · 기준 시점 {len(doc['cuts'])}개")
    print(f"  산출물 해시 {count['matched']}/{count['total']} 일치 · "
          f"test_accessed={doc['data'].get('test_accessed')}")
    for warning in warnings:
        print(f"  경고: {warning}")
    return True


def verify_run(path: Path, *, repo_root: str | Path = REPO_ROOT) -> bool:
    """manifest 하나를 검증하고 결과를 출력한다. 통과면 참.

    schema 로 분기한다 — conformal 산출물에 평가 프로토콜 필수 필드를 요구하면 사실이 아닌
    문제가 쏟아지고, 반대로 평가 manifest 를 conformal 계약으로 보면 해시를 못 맞춘다.
    """
    path = Path(path)
    if not path.exists():
        print(f"FAIL manifest 가 없다: {path}", file=sys.stderr)
        return False
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"FAIL JSON 파싱 실패: {path} — {exc}", file=sys.stderr)
        return False

    schema = doc.get("schema") if isinstance(doc, dict) else None
    if schema == CONFORMAL_SCHEMA:
        return _report_conformal(path, doc)
    if schema != MANIFEST_SCHEMA:
        print(f"FAIL {path} — 문제 1개", file=sys.stderr)
        print(f"  - schema 불일치: {schema!r} — 이 검증기는 {MANIFEST_SCHEMA} 또는 "
              f"{CONFORMAL_SCHEMA} 만 안다", file=sys.stderr)
        return False

    problems, warnings = check_manifest_with_warnings(doc)
    hash_problems, counts = verify_hashes(doc, path.parent, repo_root)
    problems.extend(hash_problems)
    problems.extend(check_metrics_consistency(doc, path.parent))
    if problems:
        print(f"FAIL {path} — 문제 {len(problems)}개", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        for warning in warnings:
            print(f"  경고: {warning}", file=sys.stderr)
        return False

    features = doc["features"]
    artifacts = counts["artifacts"]
    data = counts["data"]
    print(f"OK {path}")
    print(f"  run_id={doc['run_id']} · protocol={doc['protocol']} · task={doc['task']} · "
          f"기준일 {len(doc['metrics']['per_cut'])}개 · "
          f"통합 평균절대오차={doc['metrics']['pooled']['mae']:.2f}")
    print(f"  features {len(features['names'])}개 · order_hash={features['order_hash'][:12]}… (재계산 일치)")
    print(f"  산출물 해시 {artifacts['matched']}/{artifacts['total']} 일치 · "
          f"입력 데이터 해시 {data['matched']}/{data['total']} 일치")
    print(f"  판정={doc['verdict'].get('result')}")
    for warning in warnings:
        print(f"  경고: {warning}")
    return True


def manifest_schema(path: Path) -> str | None:
    """manifest 의 schema 값 — 다른 종류의 산출물을 구분하는 데만 쓴다."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return doc.get("schema") if isinstance(doc, dict) else None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        paths = [resolve_manifest(arg) for arg in argv]
    else:
        paths = [run / MANIFEST_NAME for run in discover_runs(REPO_ROOT)]
        if not paths:
            print(f"FAIL 검증할 실행이 없다: {REPO_ROOT / EXPERIMENT_DIR}", file=sys.stderr)
            return 1

    failed = 0
    for path in paths:
        if not verify_run(path):
            failed += 1
    if failed:
        print(f"FAIL {failed}/{len(paths)} 실행이 검증을 통과하지 못했다", file=sys.stderr)
        return 1
    if len(paths) > 1:
        print(f"전부 통과: {len(paths)}개 실행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())