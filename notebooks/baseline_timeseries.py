"""D/E 비교용 시계열 기준선 — 앱에는 연결하지 않고 성적표만 만든다.

① 학습 없는 기준선 3종(계절 나이브 lag_7d · 나이브 lag_1d · mean_prev7d)은 Test 742,560행 전체.
② ARIMA·Holt-Winters 는 5,460 시리즈 전체 fit 이 CPU 에서 비현실적이라
   볼륨 3분위 × 8호선 층화 표본(기본 72 시리즈)만 train+valid 로 학습해 Test 136일을 예측한다.
   같은 표본에서 기준선 3종도 다시 재서 공정하게 비교한다.

결과: models/metrics_timeseries.json (표본 수·선정 규칙·실패 시리즈를 함께 기록)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "models" / "metrics_timeseries.json"

EVAL_END = "2026-02-14"


def load(name: str, cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv.gz", usecols=cols, parse_dates=["date"])


def score(actual: np.ndarray, pred: np.ndarray) -> dict:
    ok = ~(np.isnan(actual) | np.isnan(pred))
    err = actual[ok] - pred[ok]
    return {
        "n": int(ok.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def naive_scores(df: pd.DataFrame) -> dict:
    actual = df["boarding"].to_numpy(dtype="float64")
    return {
        "seasonal_naive_lag_7d": score(actual, df["lag_7d"].to_numpy(dtype="float64")),
        "naive_lag_1d": score(actual, df["lag_1d"].to_numpy(dtype="float64")),
        "mean_prev7d": score(actual, df["mean_prev7d"].to_numpy(dtype="float64")),
    }


def select_series(train: pd.DataFrame, test: pd.DataFrame, per_cell: int) -> pd.DataFrame:
    """호선별 볼륨 3분위 안에서 per_cell 개씩 고른다 (기본 3개 → 8호선 × 9 = 72)."""
    mean = train.groupby(["station_code", "hour_code"], observed=True)["boarding"].mean()
    mean = mean.rename("train_mean").reset_index()
    station_line = test[["station_code", "line_code"]].drop_duplicates("station_code")
    pool = mean.merge(station_line, on="station_code", how="left")
    chosen = []
    for line, part in pool.groupby("line_code", observed=True):
        part = part.sort_values("train_mean").reset_index(drop=True)
        edges = np.quantile(part["train_mean"], [1 / 3, 2 / 3])
        part["tertile"] = np.digitize(part["train_mean"], edges)
        for tertile, cell in part.groupby("tertile", observed=True):
            picks = np.unique(np.linspace(0, len(cell) - 1, per_cell).round().astype(int))
            take = cell.iloc[picks].copy()
            take["line_code"] = line
            take["tertile"] = int(tertile)
            chosen.append(take)
    return pd.concat(chosen, ignore_index=True)


def fit_forecast(y: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    """train+valid 시계열 → Test horizon 예측. 실패한 모델은 결과에서 뺀다."""
    out: dict[str, np.ndarray] = {}
    try:
        fit = ARIMA(y, order=(2, 1, 2), trend="n").fit()
        out["arima_2_1_2"] = np.asarray(fit.forecast(horizon), dtype="float64")
    except Exception as exc:  # noqa: BLE001 - 시리즈별 실패를 기록하고 계속한다
        out["arima_error"] = str(exc)[:200]
    try:
        fit = ExponentialSmoothing(
            y, trend=None, seasonal="add", seasonal_periods=7,
            initialization_method="estimated",
        ).fit()
        out["holt_winters_add_7"] = np.asarray(fit.forecast(horizon), dtype="float64")
    except Exception as exc:  # noqa: BLE001
        out["holt_winters_error"] = str(exc)[:200]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cell", type=int, default=3,
                        help="호선×볼륨분위 셀당 시리즈 수 (3 → 72 시리즈)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")  # statsmodels 수렴 경고가 72개 시리즈만큼 쏟아진다

    cols = ["date", "station_code", "line_code", "hour_code", "boarding",
            "lag_1d", "lag_7d", "mean_prev7d"]
    print("[1/4] CSV 로드")
    train = load("train", cols)
    valid = load("valid", cols)
    test = load("test", cols)

    print("[2/4] 전체 행 기준선")
    all_rows = {"test": naive_scores(test), "valid": naive_scores(valid)}

    print("[3/4] 층화 표본 선정")
    sample = select_series(train, test, args.per_cell)
    keys = pd.MultiIndex.from_frame(sample[["station_code", "hour_code"]])
    frames = pd.concat([train, valid, test], ignore_index=True)
    del train, valid
    frames = frames.set_index(["station_code", "hour_code"]).loc[keys].reset_index()
    dates = pd.date_range(frames["date"].min(), frames["date"].max(), freq="D")
    test_dates = pd.date_range(test["date"].min(), test["date"].max(), freq="D")
    horizon = len(test_dates)
    print(f"  표본 {len(sample)} 시리즈 · Test {horizon}일")

    print("[4/4] 시리즈별 학습·예측")
    rows: list[dict] = []
    pooled: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    t0 = time.perf_counter()
    for _, meta in sample.iterrows():
        key = (meta["station_code"], meta["hour_code"])
        s = frames[(frames["station_code"] == key[0]) & (frames["hour_code"] == key[1])]
        s = s.set_index("date").sort_index()
        y_full = s["boarding"].reindex(dates).astype("float64")
        y_fit = y_full.loc[:pd.Timestamp(EVAL_END)].interpolate(limit_direction="both").to_numpy()
        actual = y_full.loc[test_dates].to_numpy()
        preds = fit_forecast(y_fit, horizon)
        naive = {
            "naive_lag_7d": s["lag_7d"].reindex(test_dates).to_numpy("float64"),
            "naive_lag_1d": s["lag_1d"].reindex(test_dates).to_numpy("float64"),
            "mean_prev7d": s["mean_prev7d"].reindex(test_dates).to_numpy("float64"),
        }
        row = {
            "station_code": int(key[0]), "hour_code": int(key[1]),
            "line_code": int(meta["line_code"]), "tertile": int(meta["tertile"]),
            "train_mean": float(meta["train_mean"]),
        }
        for name, pred in naive.items():
            row[name] = score(actual, pred)
            pooled.setdefault(name, []).append((actual, pred))
        for name in ("arima_2_1_2", "holt_winters_add_7"):
            if name in preds:
                row[name] = score(actual, preds[name])
                pooled.setdefault(name, []).append((actual, preds[name]))
        for name in ("arima_error", "holt_winters_error"):
            if name in preds:
                row[name] = preds[name]
        rows.append(row)

    def pooled_score(name: str) -> dict:
        if name not in pooled:
            return {"n": 0, "mae": float("nan"), "rmse": float("nan")}
        a = np.concatenate([p[0] for p in pooled[name]])
        b = np.concatenate([p[1] for p in pooled[name]])
        return score(a, b)

    stratified = {
        "series": int(len(sample)),
        "horizon_days": int(horizon),
        "selection": f"8호선 × 볼륨 3분위 × 셀당 {args.per_cell} 시리즈 (train 평균 기준)",
        "fit_period": f"2024-01-08 ~ {EVAL_END}",
        "score_period": f"{test_dates[0].date()} ~ {test_dates[-1].date()}",
        "arima_2_1_2": pooled_score("arima_2_1_2"),
        "holt_winters_add_7": pooled_score("holt_winters_add_7"),
        "naive_lag_7d": pooled_score("naive_lag_7d"),
        "naive_lag_1d": pooled_score("naive_lag_1d"),
        "mean_prev7d": pooled_score("mean_prev7d"),
        "failures": {
            "arima": sum("arima_error" in r for r in rows),
            "holt_winters": sum("holt_winters_error" in r for r in rows),
        },
    }
    elapsed = time.perf_counter() - t0

    document = {
        "generated": "2026-10-22",
        "protocol": {
            "source": "data/processed/{train,valid,test}.csv.gz",
            "fit": "train+valid only (Test 미사용)",
            "naive_all_rows": "Test 742,560행 전체",
            "arima": "ARIMA(2,1,2), trend 없음, 계절항 없음",
            "holt_winters": "ExponentialSmoothing trend 없음, seasonal='add', period 7",
            "note": "ARIMA·Holt-Winters 는 층화 표본 기준이라 전체 행 기준선과 n 이 다르다",
        },
        "all_rows": all_rows,
        "stratified": stratified,
        "sample_series": rows,
        "elapsed_seconds": float(elapsed),
    }
    OUT.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_rows_test": all_rows["test"], "stratified": stratified},
                     ensure_ascii=False, indent=2))
    print(f"저장: {OUT}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
