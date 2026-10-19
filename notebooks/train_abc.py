"""metro-insight A/B/C 모델 학습 스크립트."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from metro_insight.model.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

FEATS = {
    "A": ["lag_1d", "lag_7d", "mean_prev7d"],
    "B": ["lag_1d", "lag_7d", "mean_prev7d", "hour_code", "weekday",
          "is_weekend", "is_holiday", "station_code", "line_code"],
    "C": ["lag_1d", "lag_7d", "mean_prev7d", "hour_code", "weekday",
          "is_weekend", "is_holiday", "station_code", "line_code",
          "temperature", "rainfall", "snowfall"],
}

GRID = [
    {"learning_rate": 0.05, "max_leaf_nodes": 31},
    {"learning_rate": 0.05, "max_leaf_nodes": 63},
    {"learning_rate": 0.1, "max_leaf_nodes": 31},
    {"learning_rate": 0.1, "max_leaf_nodes": 63},
]

MAX_ITER = 300
RANDOM_STATE = 42


def tune(cols, tr, va):
    # 튜닝은 샘플로 빠르게 (상대 비교용)
    tr_s = tr.sample(n=min(200000, len(tr)), random_state=RANDOM_STATE)
    va_s = va.sample(n=min(50000, len(va)), random_state=RANDOM_STATE)
    best = None
    for g in GRID:
        m = HistGradientBoostingRegressor(
            max_iter=MAX_ITER,
            learning_rate=g["learning_rate"],
            max_leaf_nodes=g["max_leaf_nodes"],
            random_state=RANDOM_STATE,
        )
        m.fit(tr_s[cols], tr_s.boarding)
        pred = m.predict(va_s[cols])
        sc = evaluate(va_s.boarding, pred)
        print(f"  tune lr={g['learning_rate']} leaf={g['max_leaf_nodes']} "
              f"valid MAE={sc.mae:.2f} RMSE={sc.rmse:.2f}")
        if best is None or sc.mae < best[0].mae:
            best = (sc, g, None)
    return best


def main():
    print("데이터 로드...")
    tr = pd.read_csv(DATA / "train.csv.gz")
    va = pd.read_csv(DATA / "valid.csv.gz")
    te = pd.read_csv(DATA / "test.csv.gz")
    print(f"train={len(tr)} valid={len(va)} test={len(te)}")

    results = {}

    # baseline
    base_valid = evaluate(va.boarding, va.lag_7d)
    base_test = evaluate(te.boarding, te.lag_7d)
    base_train = evaluate(tr.boarding, tr.lag_7d)
    print(f"\nbaseline  train MAE={base_train.mae:.2f} RMSE={base_train.rmse:.2f}"
          f"  valid MAE={base_valid.mae:.2f} RMSE={base_valid.rmse:.2f}"
          f"  test MAE={base_test.mae:.2f} RMSE={base_test.rmse:.2f}")
    results["baseline"] = {
        "train": {"mae": base_train.mae, "rmse": base_train.rmse},
        "valid": {"mae": base_valid.mae, "rmse": base_valid.rmse},
        "test": {"mae": base_test.mae, "rmse": base_test.rmse},
    }

    for name, cols in FEATS.items():
        print(f"\n=== {name} ({len(cols)} features) ===")
        print("튜닝 (valid):")
        _, best_g, _ = tune(cols, tr, va)
        print(f"  best: lr={best_g['learning_rate']} leaf={best_g['max_leaf_nodes']}")

        final = HistGradientBoostingRegressor(
            max_iter=MAX_ITER,
            learning_rate=best_g["learning_rate"],
            max_leaf_nodes=best_g["max_leaf_nodes"],
            random_state=RANDOM_STATE,
        )
        final.fit(tr[cols], tr.boarding)

        train_pred = final.predict(tr[cols])
        valid_pred = final.predict(va[cols])
        test_pred = final.predict(te[cols])

        tr_sc = evaluate(tr.boarding, train_pred)
        va_sc = evaluate(va.boarding, valid_pred)
        te_sc = evaluate(te.boarding, test_pred)

        # 24_01 버킷 (hour_code == 19) 별도
        mask24 = te.hour_code == 19
        sc24 = evaluate(te.loc[mask24, "boarding"], test_pred[mask24])

        print(f"  train MAE={tr_sc.mae:.2f} RMSE={tr_sc.rmse:.2f}")
        print(f"  valid MAE={va_sc.mae:.2f} RMSE={va_sc.rmse:.2f}")
        print(f"  test  MAE={te_sc.mae:.2f} RMSE={te_sc.rmse:.2f}")
        print(f"  test  24_01 MAE={sc24.mae:.2f} RMSE={sc24.rmse:.2f} (n={mask24.sum()})")
        print(f"  과적합 격차 (test-train MAE) = {te_sc.mae - tr_sc.mae:.2f}")

        joblib.dump(final, MODELS / f"model_{name}.joblib")
        print(f"  저장: models/model_{name}.joblib")

        results[name] = {
            "best": best_g,
            "train": {"mae": tr_sc.mae, "rmse": tr_sc.rmse},
            "valid": {"mae": va_sc.mae, "rmse": va_sc.rmse},
            "test": {"mae": te_sc.mae, "rmse": te_sc.rmse},
            "test_24_01": {"mae": sc24.mae, "rmse": sc24.rmse, "n": int(mask24.sum())},
            "overfit_gap_mae": te_sc.mae - tr_sc.mae,
        }

    out = ROOT / "models" / "metrics_abc.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n메트릭 저장: {out}")

    print("\n=== 요약 (test MAE) ===")
    for k in ["baseline", "A", "B", "C"]:
        r = results[k]
        print(f"  {k:9s} MAE={r['test']['mae']:10.2f}  RMSE={r['test']['rmse']:10.2f}")


if __name__ == "__main__":
    main()
