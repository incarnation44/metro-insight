"""182~366일 지평 백테스트 — cut 마다 모델을 새로 학습해 '6~12개월 뒤'를 채점한다.

실행 (프로젝트 루트에서)
    python notebooks/backtest_182_366.py

용도
    앱·서빙 모델에는 쓰지 않는다. 앱이 '미검증 장기 추정'으로 표시하는 구간의
    오차를 숫자로 남겨 README·보고서의 근거로만 쓴다. 결과 JSON은 작은 집계라
    커밋 대상이고, joblib/npz 같은 모델 바이너리와는 성격이 다르다.

원칙
    * test.csv.gz 는 열지 않는다. 실제값은 train+valid(≤2026-02-14)만 쓴다.
    * 서빙 모델(cal_*.joblib)은 건드리지 않는다. cut 시점 데이터로 메모리에서만 학습한다.
    * 서빙 모델은 2026-02-14까지 학습했으므로 과거 cut 평가에 그대로 쓰면 누수다 → cut 마다 재학습.
    * target = cut+182 ~ cut+366일 날짜를 5일 간격으로 표본 추출. 전부 '182-366' 버킷이다.
    * C 의 날씨 입력은 앱의 미래 경로와 같게 평년 기온 + 비·눈 없음으로 둔다.
    * 결과는 models/backtest_182_366.json 에만 쓴다 (metrics_calendar.json 은 Test 기록이라 건드리지 않음).

feature 계약·하이퍼파라미터는 train_calendar.py / metrics_calendar.json 과 동일하다.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import calendar_features as cf  # noqa: E402
import train_calendar as tc  # noqa: E402
from metro_insight.model.predictor import MONTHLY_TEMPERATURE_NORMALS  # noqa: E402

DATA = ROOT / "data" / "processed"
OUT = ROOT / "models" / "backtest_182_366.json"
VALID_END = pd.Timestamp("2026-02-14")
CUTS = ["2024-11-01", "2024-12-15", "2025-02-13"]
STEP_DAYS = 5
PARAMS = {"learning_rate": 0.05, "max_leaf_nodes": 31}
BUCKET = "182-366"


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    err = np.asarray(y, dtype="float64") - np.asarray(pred, dtype="float64")
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "n": int(len(err)),
        "coverage_mae": float(np.mean(abs_err <= mae)),
        "p50_abs_err": float(np.quantile(abs_err, 0.50)),
        "p90_abs_err": float(np.quantile(abs_err, 0.90)),
    }


def run_cut(cut: str, base: pd.DataFrame) -> dict:
    cut_ts = pd.Timestamp(cut)
    fit = base[base["date"] <= cut_ts].copy()
    dates = pd.date_range(cut_ts + pd.Timedelta(days=182),
                          cut_ts + pd.Timedelta(days=366), freq=f"{STEP_DAYS}D")
    tgt = base[base["date"].isin(dates)].copy()
    horizon = (tgt["date"] - cut_ts).dt.days.to_numpy()
    assert len(tgt) and horizon.min() >= 182 and horizon.max() <= 366
    assert tgt["date"].max() <= VALID_END, "valid 끝을 넘는 실제값을 쓰려 한다 — Test 접촉"

    print(f"\ncut {cut}: train {len(fit):,}행 ({fit['date'].min().date()}~{fit['date'].max().date()}) · "
          f"target {len(tgt):,}행 ({tgt['date'].min().date()}~{tgt['date'].max().date()}, "
          f"{horizon.min()}~{horizon.max()}일)", flush=True)

    t0 = time.time()
    fit = pd.concat([fit, cf.asof_profile_frame(fit)], axis=1)
    tgt = pd.concat([tgt, cf.frozen_profile_frame(fit, tgt, cut_ts)], axis=1)
    month_missing = int(tgt["profile_month_mean"].isna().sum())
    print(f"  프로필 계산 {time.time() - t0:.1f}s · 월 프로필 결측 {month_missing:,}행 "
          f"(모델은 NaN 그대로 처리)", flush=True)

    y = tgt["boarding"].to_numpy(dtype="float64")
    pred_frame = tgt.copy()
    pred_frame["temperature"] = pred_frame["date"].dt.month.map(MONTHLY_TEMPERATURE_NORMALS).astype("float64")
    pred_frame["rainfall"] = 0
    pred_frame["snowfall"] = 0

    out: dict = {"n": int(len(tgt)), "horizon_min": int(horizon.min()),
                 "horizon_max": int(horizon.max()), "month_profile_missing": month_missing,
                 "models": {}}
    # baseline: 월 프로필 평균, 없으면 (역·시간) 평균으로 대체
    naive = (pred_frame[cf.PROFILE_COLS_MONTH[0]]
             .fillna(pred_frame[cf.PROFILE_COLS_SH[0]]).to_numpy(dtype="float64"))
    if np.isnan(naive).any():
        raise AssertionError("naive 기준에 빈 키가 남아 있다")
    out["models"]["naive_profile_mean"] = score(y, naive)

    for name, cols in tc.FEATS.items():
        t0 = time.time()
        m = tc.make_model(PARAMS["learning_rate"], PARAMS["max_leaf_nodes"])
        m.fit(fit[cols], fit["boarding"])
        sc = score(y, m.predict(pred_frame[cols]))
        out["models"][name] = sc
        print(f"  {name}: MAE {sc['mae']:7.2f} · RMSE {sc['rmse']:7.2f} · "
              f"±MAE 커버 {sc['coverage_mae']:.1%} · fit {time.time() - t0:.1f}s", flush=True)
        del m
        gc.collect()

    del fit, tgt, pred_frame
    gc.collect()
    return out


def main() -> None:
    print("데이터 로드 (test.csv.gz 는 읽지 않는다)", flush=True)
    frames = [cf.load_processed(DATA / f, columns=tc.USE_COLS) for f in ("train.csv.gz", "valid.csv.gz")]
    base = pd.concat(frames, ignore_index=True)
    print(f"train+valid {len(base):,}행 · {base['date'].min().date()} ~ {base['date'].max().date()}",
          flush=True)
    del frames
    gc.collect()

    per_cut = {cut: run_cut(cut, base) for cut in CUTS}

    names = list(tc.FEATS) + ["naive_profile_mean"]
    pooled: dict = {"n": sum(per_cut[cut]["models"]["A"]["n"] for cut in CUTS), "models": {}}
    for name in names:
        n_total = sum(per_cut[cut]["models"][name]["n"] for cut in CUTS)
        weighted_mae = sum(per_cut[cut]["models"][name]["mae"] * per_cut[cut]["models"][name]["n"]
                           for cut in CUTS) / n_total
        pooled["models"][name] = {"mae_weighted": weighted_mae, "n": n_total}
        for cut in CUTS:
            pooled["models"][name][f"mae_{cut}"] = per_cut[cut]["models"][name]["mae"]

    doc = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "purpose": "182~366일 지평(미검증 구간)의 실제 오차 측정 — 롤링 오리진 백테스트",
        "test_untouched": True,
        "actuals_range": f"train+valid (≤{VALID_END.date()})",
        "cuts": CUTS,
        "step_days": STEP_DAYS,
        "params": {"max_iter": tc.MAX_ITER, **PARAMS, "random_state": tc.SEED},
        "features": tc.FEATS,
        "weather_for_C": "월 평년 기온 + rainfall=0 + snowfall=0 (앱 미래 경로와 동일)",
        "per_cut": per_cut,
        "pooled": pooled,
        "caveats": [
            "cut 마다 데이터가 적은 모델로 학습했다 — 서빙 모델(2026-02-14까지 학습)보다 보수적이다.",
            "target 날짜는 5일 간격 표본이라 전수 평가가 아니다.",
            "실제값은 valid(2026-02-14)까지만 쓴다. Test는 열지 않았다.",
            "이 결과는 Test 성적이 아니므로 '검증됨'으로 표기하면 안 된다.",
        ],
    }
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {OUT} (문서용 집계 — 서빙 미사용, 커밋 대상)", flush=True)

    print("\n" + "=" * 60)
    print(f"pooled 182-366일 (cut {len(CUTS)}개, n={pooled['n']:,})")
    print("=" * 60)
    rows = []
    for name in names:
        v = pooled["models"][name]
        rows.append({"model": name, "MAE(가중)": round(v["mae_weighted"], 2),
                     **{cut: round(v[f"mae_{cut}"], 2) for cut in CUTS}})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
