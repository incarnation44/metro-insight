"""달력(프로필) 모델 최종 test 평가 — 1회용. 콘솔 표 + metrics_calendar.json 의 'test' 블록 갱신.

실행 (프로젝트 루트에서, 1회만)
    python notebooks/evaluate_calendar.py

입력
    models/cal_A.joblib · cal_B.joblib · cal_C.joblib
    models/cal_profiles_eval.npz   ← train+valid(2026-02-14까지) 로 만든 고정 프로필. **이것만 쓴다**
    data/processed/test.csv.gz

원칙
    * 프로필은 npz 에서만 룩업한다. test 데이터로 프로필을 다시 만들지 않는다 (평가 누수 방지).
    * feature 순서는 각 모델의 feature_names_in_ 을 따른다.
    * horizon = test 날짜 - 2026-02-14 (일수) → 1-31 / 32-91 / 92-181 / 182-366 버킷.
    * metrics_calendar.json 은 기존 키를 보존하고 'test' 블록만 추가/갱신한다.

feature 계약
    A = [profile_month_mean, profile_n, station_code, line_code,
         profile_weekday_mean, profile_weekday_n, hour_code, weekday]     역·시간·요일·월
    B = A + [is_holiday]                                                  + 공휴일
    C = B + [temperature, rainfall, snowfall]                             + 날씨
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calendar_features as cf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PROFILES = MODELS / "cal_profiles_eval.npz"
METRICS = MODELS / "metrics_calendar.json"

# 프로필을 만든 마지막 날 (= eval 프로필 기준일). horizon 계산 기준.
PROFILE_END = "2026-02-14"

USE_COLS = [
    "date", "station_code", "line_code", "hour_code", "weekday",
    "is_weekend", "is_holiday", "temperature", "rainfall", "snowfall", "boarding",
]

A_COLS = ["profile_month_mean", "profile_n", "station_code", "line_code",
          "profile_weekday_mean", "profile_weekday_n", "hour_code", "weekday"]
B_COLS = A_COLS + ["is_holiday"]
C_COLS = B_COLS + ["temperature", "rainfall", "snowfall"]
FEATS = {"A": A_COLS, "B": B_COLS, "C": C_COLS}


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def metrics(y: np.ndarray, pred: np.ndarray, horizon: np.ndarray) -> dict:
    """overall + horizon 구간별 MAE/RMSE/n."""
    y = np.asarray(y, dtype="float64")
    err = y - np.asarray(pred, dtype="float64")
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "n": int(len(y)),
        "horizon": {},
    }
    bucket = cf.horizon_bucket(horizon)
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


def load_profiles(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """npz → (월별 프로필, 요일별 프로필) 데이터프레임. 여기서 test 데이터는 쓰지 않는다."""
    if not path.exists():
        raise FileNotFoundError(f"평가용 프로필이 없다: {path}")
    z = np.load(path)
    need = ["station_code", "hour_code", "month", "mean", "count",
            "station_code_wd", "hour_code_wd", "weekday_wd", "mean_wd", "count_wd"]
    missing = [k for k in need if k not in z.files]
    if missing:
        raise KeyError(f"npz 에 없는 배열: {missing} (train_calendar.py 로 다시 만들어야 한다)")
    month = pd.DataFrame({
        "station_code": z["station_code"].astype("int64"),
        "hour_code": z["hour_code"].astype("int64"),
        "month": z["month"].astype("int64"),
        "mean": z["mean"],
        "count": z["count"],
    })
    weekday = pd.DataFrame({
        "station_code": z["station_code_wd"].astype("int64"),
        "hour_code": z["hour_code_wd"].astype("int64"),
        "weekday": z["weekday_wd"].astype("int64"),
        "mean": z["mean_wd"],
        "count": z["count_wd"],
    })
    return month, weekday


def attach_profiles(df: pd.DataFrame, month: pd.DataFrame, weekday: pd.DataFrame) -> pd.DataFrame:
    """고정 프로필을 npz 테이블에서 룩업해 붙인다 (cf.lookup — 정수 키 reindex)."""
    out = df.copy()
    out["profile_month_mean"] = cf.lookup(month, out, cf.KEY_SH_MONTH, "mean").astype("float32")
    out["profile_n"] = np.nan_to_num(
        cf.lookup(month, out, cf.KEY_SH_MONTH, "count"), nan=0.0).astype("int32")
    out["profile_weekday_mean"] = cf.lookup(weekday, out, cf.KEY_SH_WEEKDAY, "mean").astype("float32")
    out["profile_weekday_n"] = np.nan_to_num(
        cf.lookup(weekday, out, cf.KEY_SH_WEEKDAY, "count"), nan=0.0).astype("int32")
    return out


def feature_order(model, name: str) -> list[str]:
    """모델이 학습된 feature 순서. 계약과 컬럼 집합이 다르면 즉시 실패한다."""
    cols = list(getattr(model, "feature_names_in_", []))
    if not cols:
        raise ValueError(f"cal_{name} 에 feature_names_in_ 이 없다 — 그 순서로 예측할 수 없다")
    expected = FEATS[name]
    if set(cols) != set(expected):
        raise ValueError(f"cal_{name} 피처가 계약과 다르다.\n  모델: {cols}\n  계약: {expected}")
    if cols != expected:
        print(f"  ※ cal_{name}: 모델 저장 순서가 계약과 달라 모델 순서를 쓴다 → {cols}")
    return cols


def main() -> None:
    header("1) 로드")
    test = cf.load_processed(DATA / "test.csv.gz", columns=USE_COLS)
    print(f"test {len(test):,}행 · {test['date'].min().date()} ~ {test['date'].max().date()} · "
          f"역 {test['station_code'].nunique()}개")
    month, weekday = load_profiles(PROFILES)
    print(f"프로필: {PROFILES.name} — 월별 {len(month):,}키 · 요일별 {len(weekday):,}키 "
          f"(기준일 {PROFILE_END}, test 로 만들지 않음)")

    test = attach_profiles(test, month, weekday)
    miss = {c: int(test[c].isna().sum()) for c in ("profile_month_mean", "profile_weekday_mean")}
    print(f"프로필 결측: month {miss['profile_month_mean']:,}행 · weekday {miss['profile_weekday_mean']:,}행")

    horizon = (test["date"] - pd.Timestamp(PROFILE_END)).dt.days.to_numpy()
    print(f"horizon {horizon.min()}~{horizon.max()}일 · 구간별 행수: " +
          ", ".join(f"{lo}-{hi}={int(((horizon >= lo) & (horizon <= hi)).sum()):,}"
                    for lo, hi in cf.HORIZON_BUCKETS))
    y = test["boarding"].to_numpy(dtype="float64")

    header("2) test 평가 (모델별)")
    results: dict[str, dict] = {}
    for name in ("A", "B", "C"):
        path = MODELS / f"cal_{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"모델이 없다: {path}")
        model = joblib.load(path)
        cols = feature_order(model, name)
        pred = model.predict(test[cols])
        sc = metrics(y, pred, horizon)
        results[name] = sc
        print(f"  cal_{name}: MAE {sc['mae']:8.2f}  RMSE {sc['rmse']:8.2f}  n {sc['n']:,}")

    header("3) horizon 구간별 (MAE / RMSE / n)")
    rows = []
    for name, sc in results.items():
        for label, v in sc["horizon"].items():
            rows.append({"model": name, "horizon": label,
                         "MAE": round(v["mae"], 2), "RMSE": round(v["rmse"], 2), "n": v["n"]})
        rows.append({"model": name, "horizon": "overall",
                     "MAE": round(sc["mae"], 2), "RMSE": round(sc["rmse"], 2), "n": sc["n"]})
    print(pd.DataFrame(rows).to_string(index=False))

    header("4) models/metrics_calendar.json 'test' 블록 갱신")
    doc = {}
    if METRICS.exists():
        with open(METRICS, encoding="utf-8") as f:
            doc = json.load(f)          # 기존 키는 그대로 보존한다
    doc["test"] = {name: sc for name, sc in results.items()}
    with open(METRICS, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"  저장: {METRICS} (기존 키 {len([k for k in doc if k != 'test'])}개 보존 + test 추가)")

    header("요약 (test MAE)")
    for name, sc in results.items():
        print(f"  {name}: {sc['mae']:10.2f}   RMSE {sc['rmse']:10.2f}")


if __name__ == "__main__":
    main()
