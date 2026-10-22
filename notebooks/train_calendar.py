"""달력(프로필) 모델 학습 — 2단계. A/B/C 3개 모델을 저장한다.

실행 (프로젝트 루트에서)
    python notebooks/train_calendar.py
    python notebooks/train_calendar.py --recency-weighted   # 최근성 가중 프로필 실험

계약 (feature 순서 그대로)
    A = [profile_month_mean, profile_n, station_code, line_code]
    B = A + [profile_weekday_mean, profile_weekday_n, hour_code, weekday, is_weekend, is_holiday]
    C = B + [temperature, rainfall, snowfall]

핵심
    * train 행은 as-of 프로필(그 날짜 이전만), valid 행은 train_end 시점 frozen 프로필을 쓴다.
    * 그리드 튜닝은 train 으로만 하고 valid 로 고른다 (test 는 학습·튜닝·평가에 쓰지 않는다).
    * 최종 모델은 train+valid 로 재학습한다 (그때도 as-of 프로필).
    * 기존 model_A/B/C.joblib · metrics_abc.json 은 건드리지 않는다.

저장 (기본)
    models/cal_A.joblib · cal_B.joblib · cal_C.joblib
    models/cal_profiles.npz        서빙용: 전체 이력(2024-01-08~2026-06-30) 프로필
    models/cal_profiles_eval.npz   평가용: train+valid(2026-02-14까지) 프로필
    models/metrics_calendar.json   feature 목록 · 파라미터 · valid 지표(horizon 구간 포함) · 프로필 정보

저장 (--recency-weighted)
    기존 cal_A/B/C.joblib 는 cal_A_prev.joblib 등으로 **이름만 바꿔 보존**하고,
    가중 프로필로 재학습한 모델을 cal_A_w/B_w/C_w.joblib 로 저장한다.
    프로필도 cal_profiles_w.npz · cal_profiles_eval_w.npz 로 새로 쓰고(기존 파일 덮어쓰지 않음),
    metrics_calendar.json 은 기존 키를 보존한 채 "recency_weighted" 키만 추가한다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calendar_features as cf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

SEED = 42
MAX_ITER = 300  # train_abc.py 와 동일
GRID = [
    {"learning_rate": 0.05, "max_leaf_nodes": 31},
    {"learning_rate": 0.05, "max_leaf_nodes": 63},
    {"learning_rate": 0.1, "max_leaf_nodes": 31},
    {"learning_rate": 0.1, "max_leaf_nodes": 63},
]
TUNE_TRAIN = 200_000
TUNE_VALID = 50_000

USE_COLS = [
    "date", "station_code", "line_code", "hour_code", "weekday",
    "is_weekend", "is_holiday", "temperature", "rainfall", "snowfall", "boarding",
]

A_COLS = ["profile_month_mean", "profile_n", "station_code", "line_code"]
B_COLS = A_COLS + ["profile_weekday_mean", "profile_weekday_n",
                   "hour_code", "weekday", "is_weekend", "is_holiday"]
C_COLS = B_COLS + ["temperature", "rainfall", "snowfall"]
FEATS = {"A": A_COLS, "B": B_COLS, "C": C_COLS}


def _profile_cols_present(df: pd.DataFrame) -> list[str]:
    """재학습 전에 기존 프로필 컬럼을 지우기 위한 목록."""
    return [c for c in cf.PROFILE_COLS_SH + cf.PROFILE_COLS_MONTH + cf.PROFILE_COLS_WEEKDAY
            if c in df.columns]


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def metrics(y: np.ndarray, pred: np.ndarray, horizon: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype="float64")
    err = y - np.asarray(pred, dtype="float64")
    out = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))), "n": int(len(y))}
    if horizon is not None:
        bucket = cf.horizon_bucket(horizon)
        out["horizon"] = {}
        for lo, hi in cf.HORIZON_BUCKETS:
            label = f"{lo}-{hi}"
            m = bucket == label
            if m.any():
                e = err[m]
                out["horizon"][label] = {
                    "mae": float(np.mean(np.abs(e))),
                    "rmse": float(np.sqrt(np.mean(e ** 2))),
                    "n": int(m.sum()),
                }
    return out


def make_model(lr: float, leaf: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=MAX_ITER, learning_rate=lr, max_leaf_nodes=leaf, random_state=SEED
    )


def tune(name: str, cols: list[str], tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    """train 샘플로 학습, valid 샘플로 고른다 (상대 비교용 — train_abc.py 와 같은 방식)."""
    tr_s = tr.sample(n=min(TUNE_TRAIN, len(tr)), random_state=SEED)
    va_s = va.sample(n=min(TUNE_VALID, len(va)), random_state=SEED)
    y_va = va_s["boarding"].to_numpy(dtype="float64")
    best = None
    for g in GRID:
        m = make_model(g["learning_rate"], g["max_leaf_nodes"])
        m.fit(tr_s[cols], tr_s["boarding"])
        sc = metrics(y_va, m.predict(va_s[cols]))
        print(f"  [{name}] lr={g['learning_rate']} leaf={g['max_leaf_nodes']:<3} "
              f"valid MAE={sc['mae']:8.2f} RMSE={sc['rmse']:8.2f}")
        if best is None or sc["mae"] < best[0]["mae"]:
            best = (sc, g)
    print(f"  [{name}] best: lr={best[1]['learning_rate']} leaf={best[1]['max_leaf_nodes']}")
    return best[1]


def profile_meta(path: Path, frame: pd.DataFrame) -> dict:
    return {
        "file": path.name,
        "rows_used": int(len(frame)),
        "date_min": frame["date"].min().strftime("%Y-%m-%d"),
        "date_max": frame["date"].max().strftime("%Y-%m-%d"),
        "keys_month": int(frame.groupby(cf.KEY_SH_MONTH, observed=True).ngroups),
        "keys_station_hour": int(frame.groupby(cf.KEY_SH, observed=True).ngroups),
    }


def preserve_models() -> None:
    """기존 cal_A/B/C.joblib 를 cal_A_prev.joblib 등으로 이름만 바꿔 보존한다.

    이미 보존본이 있으면 덮어쓰지 않는다 (원본은 항상 _prev 에 남긴다).
    """
    for name in ("A", "B", "C"):
        src = MODELS / f"cal_{name}.joblib"
        dst = MODELS / f"cal_{name}_prev.joblib"
        if not src.exists():
            continue
        if dst.exists():
            print(f"  보존본 유지: {dst.name} (기존 {src.name} 은 승격 때 교체된다)")
            continue
        src.replace(dst)
        print(f"  보존: {src.name} → {dst.name}")


def main(weighted: bool = False) -> None:
    suffix = "_w" if weighted else ""
    header(f"1) 데이터 로드{' (최근성 가중 프로필 실험)' if weighted else ''}")
    tr = cf.load_processed(DATA / "train.csv.gz", columns=USE_COLS)
    va = cf.load_processed(DATA / "valid.csv.gz", columns=USE_COLS)
    train_end = tr["date"].max()
    print(f"train {len(tr):,}행 ({tr['date'].min().date()}~{train_end.date()}) · "
          f"valid {len(va):,}행 ({va['date'].min().date()}~{va['date'].max().date()})")

    header("2) 프로필 — train as-of, valid train_end 고정" + (" · 가중치 2 (최근 182일)" if weighted else ""))
    t0 = time.time()
    tr = pd.concat([tr, cf.asof_profile_frame(tr, recency=weighted)], axis=1)
    va_prof = cf.frozen_profile_frame(tr, va, train_end, recency=weighted)
    va = pd.concat([va, va_prof], axis=1)
    print(f"train as-of profile_n=0 비율 {float((tr['profile_n'] == 0).mean()):.3f} · "
          f"valid profile_n 최소 {int(va['profile_n'].min())} · {time.time() - t0:.1f}s")
    sh_missing = int(va[cf.PROFILE_COLS_SH].isna().any(axis=1).sum())
    mo_missing = int(va[cf.PROFILE_COLS_MONTH[0]].isna().sum())
    wd_missing = int(va[cf.PROFILE_COLS_WEEKDAY[0]].isna().sum())
    assert sh_missing == 0, f"valid 에 (역,시간대) 프로필이 없는 행이 {sh_missing}개 있다 — train 범위를 확인한다"
    print(f"valid 프로필 결측: station×hour {sh_missing}행 · month 평균 {mo_missing}행 · "
          f"weekday 평균 {wd_missing}행 (없으면 NaN 그대로 두고 모델이 처리한다)")

    y_va = va["boarding"].to_numpy(dtype="float64")
    horizon = (va["date"] - train_end).dt.days.to_numpy()

    header("3) 그리드 튜닝 (train 샘플 → valid 샘플)")
    best_params = {name: tune(name, cols, tr, va) for name, cols in FEATS.items()}

    header("4) 홀드아웃 지표 (train 만 학습 → valid 평가)")
    holdout_models = {}
    results: dict[str, dict] = {"params": {}, "valid": {}}
    for name, cols in FEATS.items():
        p = best_params[name]
        m = make_model(p["learning_rate"], p["max_leaf_nodes"])
        m.fit(tr[cols], tr["boarding"])
        pred = m.predict(va[cols])
        sc = metrics(y_va, pred, horizon)
        holdout_models[name] = m
        results["params"][name] = {"max_iter": MAX_ITER, **p, "random_state": SEED}
        results["valid"][name] = sc
        print(f"  {name}: valid MAE={sc['mae']:8.2f} RMSE={sc['rmse']:8.2f}  " +
              " ".join(f"{k}={v['mae']:.1f}" for k, v in sc["horizon"].items()))
    naive = metrics(y_va, va[cf.PROFILE_COLS_SH[0]].to_numpy(dtype="float64"), horizon)
    results["valid"]["naive_profile_mean"] = naive
    print(f"  naive(sh_mean): valid MAE={naive['mae']:8.2f} RMSE={naive['rmse']:8.2f}")

    header("5) 최종 모델 — train+valid 재학습 (as-of 프로필)")
    both = pd.concat([tr, va], ignore_index=True)
    both = both.drop(columns=_profile_cols_present(both))
    both = pd.concat([both, cf.asof_profile_frame(both, recency=weighted)], axis=1)
    if weighted:
        preserve_models()  # 기존 정식 모델은 _prev 로 보존 (덮어쓰지 않는다)
    for name, cols in FEATS.items():
        p = best_params[name]
        m = make_model(p["learning_rate"], p["max_leaf_nodes"])
        m.fit(both[cols], both["boarding"])
        joblib.dump(m, MODELS / f"cal_{name}{suffix}.joblib")
        pred = m.predict(both[cols])
        sc = metrics(both["boarding"].to_numpy(dtype="float64"), pred)
        results.setdefault("refit", {})[name] = {"train_valid_mae": sc["mae"], "train_valid_rmse": sc["rmse"], "n": sc["n"]}
        print(f"  cal_{name}{suffix}.joblib 저장 (train+valid MAE={sc['mae']:.2f} RMSE={sc['rmse']:.2f})")

    header("6) 프로필 저장")
    serve_test = cf.load_processed(DATA / "test.csv.gz", columns=USE_COLS)
    full = cf.add_month(pd.concat([tr[USE_COLS], va[USE_COLS], serve_test[USE_COLS]], ignore_index=True))
    serve_path = MODELS / f"cal_profiles{suffix}.npz"
    eval_path = MODELS / f"cal_profiles_eval{suffix}.npz"
    print(f"  서빙 프로필 키 계산: {cf.save_profile_npz(serve_path, full, recency=weighted)}")
    print(f"  평가 프로필 키 계산: {cf.save_profile_npz(eval_path, both[USE_COLS], recency=weighted)}")
    print("  ※ test.csv.gz 는 서빙용 전체 이력 프로필에만 쓴다 (학습·튜닝·평가에 쓰지 않는다)")

    header("7) metrics_calendar.json")
    doc = {
        "features": FEATS,
        "params": results["params"],
        "grid": GRID,
        "max_iter": MAX_ITER,
        "seed": SEED,
        "train_end": train_end.strftime("%Y-%m-%d"),
        "eval_end": va["date"].max().strftime("%Y-%m-%d"),
        "valid": results["valid"],
        "refit": results.get("refit", {}),
        "profiles": {
            "serving": profile_meta(serve_path, full),
            "eval": profile_meta(eval_path, both),
            "profiles_npz_keys": ["station_code", "hour_code", "month", "mean", "count",
                                  "station_code_sh", "hour_code_sh", "mean_sh", "count_sh",
                                  "station_code_wd", "hour_code_wd", "weekday_wd", "mean_wd", "count_wd"],
        },
        "notes": [
            "train 행 프로필은 as-of(그 날짜 이전만), valid 행 프로필은 train_end 시점 고정.",
            "valid 지표는 train 만 학습한 모델의 홀드아웃 값이다. cal_*.joblib 은 train+valid 재학습본이다.",
            "test 는 모델 학습·튜닝·평가에 쓰지 않았다. 서빙 프로필에만 포함된다.",
        ],
    }
    out = MODELS / "metrics_calendar.json"
    if weighted:
        # 기존 키를 보존하고 recency_weighted 블록만 추가한다
        old: dict = {}
        if out.exists():
            with open(out, encoding="utf-8") as f:
                old = json.load(f)
        old["recency_weighted"] = {
            "features": FEATS,
            "params": results["params"],
            "grid": GRID,
            "max_iter": MAX_ITER,
            "seed": SEED,
            "train_end": doc["train_end"],
            "eval_end": doc["eval_end"],
            "recency_weight_days": cf.RECENCY_WEIGHT_DAYS,
            "recency_weight": cf.RECENCY_WEIGHT,
            "valid": results["valid"],
            "refit": results.get("refit", {}),
            "profiles": doc["profiles"],
            "notes": [
                "프로필 평균만 최근성 가중(기준일로부터 182일 이내 관측 가중치 2)으로 바꾼 실험이다.",
                "feature 계약·GRID·seed 는 기존과 같다. 모델 파일은 cal_*_w.joblib, 프로필은 cal_profiles*_w.npz.",
                "채택 여부는 notebooks/backtest_calendar.py --recency-weighted 비교로 판정한다.",
            ],
        }
        doc = old
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"  저장: {out}")

    header("요약 (valid MAE)")
    for k in ("naive_profile_mean", "A", "B", "C"):
        print(f"  {k:<20} {results['valid'][k]['mae']:10.2f}")


if __name__ == "__main__":
    main("--recency-weighted" in sys.argv[1:])
