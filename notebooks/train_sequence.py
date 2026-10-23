"""D/E 시계열 모델(LSTM·GRU) 학습·평가 — 1회용 스크립트.

실행 (프로젝트 루트에서)
    python notebooks/train_sequence.py --verify-only   # G1: 데이터셋·누수 검증만
    python notebooks/train_sequence.py --smoke         # G2: 1에폭 속도·Valid 확인
    python notebooks/train_sequence.py                 # 전체 학습 + Test 1회

계약 (2026-10-22 지시서)
    * 소스는 data/processed/{train,valid,test}.csv.gz 단일 진실. npz 는 교차검증용.
    * WINDOW=14, 창은 대상일 전날까지. 창에 NaN 이 있으면 그 행은 제외.
    * 시리즈별 정규화: x / (그 역·시간대의 train 평균). 통계는 train 기간만.
    * 대상일 조건에 as-of 프로파일 4개를 넣는다 (train as-of, valid train_end 고정, test eval npz).
    * D=LSTM, E=GRU. 셀만 다르고 나머지는 전부 동일.
    * Test 는 학습·선택이 끝난 뒤 1회만 채점한다. 선택은 전부 Valid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

import calendar_features as cf  # noqa: E402

from metro_insight.model import sequence as seq  # noqa: E402

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PROFILE_EVAL = MODELS / "cal_profiles_eval.npz"
HISTORY_NPZ = MODELS / "boarding_history.npz"
METRICS = MODELS / "metrics_sequence.json"

TRAIN_END = "2025-10-01"
EVAL_END = "2026-02-14"
DATA_END = "2026-06-30"

BASE_COLS = [
    "date", "station_code", "line_code", "hour_code", "weekday", "is_holiday",
    "temperature", "rainfall", "snowfall", "boarding",
]
LAG_COLS = ["lag_1d", "lag_7d", "mean_prev7d"]
DTYPES = {
    "station_code": "int16", "line_code": "int8", "hour_code": "int8",
    "weekday": "int8", "is_holiday": "int8", "rainfall": "int8", "snowfall": "int8",
    "boarding": "float32", "temperature": "float32",
    "lag_1d": "float32", "lag_7d": "float32", "mean_prev7d": "float32",
}


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(
        DATA / f"{name}.csv.gz", usecols=BASE_COLS + LAG_COLS, dtype=DTYPES,
        parse_dates=["date"],
    )


def build_cube(frames: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(역, 시간대, 날짜) 승차 큐브. 날짜는 하루도 안 비고 연속이다."""
    all_df = pd.concat(frames.values(), ignore_index=True)
    station_codes = np.sort(all_df["station_code"].unique()).astype("int64")
    dates = np.sort(all_df["date"].unique()).astype("datetime64[D]").astype(str)
    start = np.datetime64(dates[0], "D")
    day_index = (all_df["date"].to_numpy("datetime64[D]") - start).astype("int64")
    sc_to_row = np.full(int(station_codes.max()) + 1, -1, dtype="int64")
    sc_to_row[station_codes] = np.arange(len(station_codes))
    history = np.full((len(station_codes), seq.HOURS_PER_DAY, len(dates)),
                      np.nan, dtype="float32")
    history[sc_to_row[all_df["station_code"].to_numpy()],
            all_df["hour_code"].to_numpy(), day_index] = all_df["boarding"].to_numpy()
    return history, station_codes, sc_to_row


def calendar_and_weather(frames: dict[str, pd.DataFrame], start: str,
                         n_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """날짜별 weekday·공휴일, (시간대, 날짜)별 기온·강수·적설. 값은 역 공통임을 확인했다."""
    all_df = pd.concat(frames.values(), ignore_index=True)
    day_index = (all_df["date"].to_numpy("datetime64[D]")
                 - np.datetime64(start, "D")).astype("int64")
    weekday = np.full(n_days, -1, dtype="int8")
    holiday = np.zeros(n_days, dtype="int8")
    cal = all_df.assign(_d=day_index)[["_d", "weekday", "is_holiday"]].drop_duplicates("_d")
    weekday[cal["_d"].to_numpy()] = cal["weekday"].to_numpy()
    holiday[cal["_d"].to_numpy()] = cal["is_holiday"].to_numpy()

    temp = np.full((seq.HOURS_PER_DAY, n_days), np.nan, dtype="float32")
    rain = np.zeros((seq.HOURS_PER_DAY, n_days), dtype="float32")
    snow = np.zeros((seq.HOURS_PER_DAY, n_days), dtype="float32")
    w = all_df.assign(_d=day_index)[
        ["_d", "hour_code", "temperature", "rainfall", "snowfall"]
    ].drop_duplicates(["_d", "hour_code"])
    temp[w["hour_code"].to_numpy(), w["_d"].to_numpy()] = w["temperature"].to_numpy()
    rain[w["hour_code"].to_numpy(), w["_d"].to_numpy()] = w["rainfall"].to_numpy()
    snow[w["hour_code"].to_numpy(), w["_d"].to_numpy()] = w["snowfall"].to_numpy()
    return weekday, holiday, temp, rain, snow


def test_profile_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    with np.load(PROFILE_EVAL, allow_pickle=False) as z:
        month = pd.DataFrame({
            "station_code": z["station_code"].astype("int64"),
            "hour_code": z["hour_code"].astype("int64"),
            "month": z["month"].astype("int64"),
            "mean": z["mean"], "count": z["count"],
        })
        weekday = pd.DataFrame({
            "station_code": z["station_code_wd"].astype("int64"),
            "hour_code": z["hour_code_wd"].astype("int64"),
            "weekday": z["weekday_wd"].astype("int64"),
            "mean": z["mean_wd"], "count": z["count_wd"],
        })
    return month, weekday


def scatter_profiles(
    target: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    df: pd.DataFrame, sc_to_row: np.ndarray, start: str,
    month_mean: np.ndarray, month_n: np.ndarray,
    weekday_mean: np.ndarray, weekday_n: np.ndarray,
) -> None:
    pm, pn, pw, pwn = target
    di = (df["date"].to_numpy("datetime64[D]") - np.datetime64(start, "D")).astype("int64")
    si = sc_to_row[df["station_code"].to_numpy()]
    hc = df["hour_code"].to_numpy()
    pm[si, hc, di] = month_mean
    pn[si, hc, di] = np.nan_to_num(month_n, nan=0.0)
    pw[si, hc, di] = weekday_mean
    pwn[si, hc, di] = np.nan_to_num(weekday_n, nan=0.0)


def fill_profiles(
    frames: dict[str, pd.DataFrame], sc_to_row: np.ndarray, start: str,
    n_st: int, n_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pm = np.full((n_st, seq.HOURS_PER_DAY, n_days), np.nan, dtype="float32")
    pn = np.zeros((n_st, seq.HOURS_PER_DAY, n_days), dtype="int32")
    pw = np.full((n_st, seq.HOURS_PER_DAY, n_days), np.nan, dtype="float32")
    pwn = np.zeros((n_st, seq.HOURS_PER_DAY, n_days), dtype="int32")
    target = (pm, pn, pw, pwn)

    train = frames["train"]
    asof_month = cf.asof_profile(
        train, cf.KEY_SH_MONTH, mean_col="profile_month_mean", n_col="profile_n")
    asof_weekday = cf.asof_profile(
        train, cf.KEY_SH_WEEKDAY, mean_col="profile_weekday_mean", n_col="profile_weekday_n")
    scatter_profiles(target, train, sc_to_row, start,
                     asof_month["profile_month_mean"].to_numpy(),
                     asof_month["profile_n"].to_numpy(),
                     asof_weekday["profile_weekday_mean"].to_numpy(),
                     asof_weekday["profile_weekday_n"].to_numpy())
    del asof_month, asof_weekday

    # valid 는 train_end 시점에 고정한 프로파일을 쓴다 (A/B/C 와 같은 프로토콜).
    frozen_month = cf.frozen_profile(
        train, TRAIN_END, cf.KEY_SH_MONTH,
        mean_col="profile_month_mean", n_col="profile_n")
    frozen_weekday = cf.frozen_profile(
        train, TRAIN_END, cf.KEY_SH_WEEKDAY,
        mean_col="profile_weekday_mean", n_col="profile_weekday_n")
    valid = frames["valid"]
    scatter_profiles(
        target, valid, sc_to_row, start,
        cf.lookup(frozen_month, valid, cf.KEY_SH_MONTH, "profile_month_mean"),
        cf.lookup(frozen_month, valid, cf.KEY_SH_MONTH, "profile_n"),
        cf.lookup(frozen_weekday, valid, cf.KEY_SH_WEEKDAY, "profile_weekday_mean"),
        cf.lookup(frozen_weekday, valid, cf.KEY_SH_WEEKDAY, "profile_weekday_n"),
    )

    # test 는 2026-02-14 동결 프로파일만 쓴다 (test 데이터로 다시 만들지 않는다).
    month, weekday = test_profile_tables()
    test = frames["test"]
    scatter_profiles(
        target, test, sc_to_row, start,
        cf.lookup(month, test, cf.KEY_SH_MONTH, "mean"),
        cf.lookup(month, test, cf.KEY_SH_MONTH, "count"),
        cf.lookup(weekday, test, cf.KEY_SH_WEEKDAY, "mean"),
        cf.lookup(weekday, test, cf.KEY_SH_WEEKDAY, "count"),
    )
    return target


def lag_check(name: str, df: pd.DataFrame, history_flat: np.ndarray,
              sc_to_row: np.ndarray, start: str, window: int) -> dict:
    """생성한 창이 CSV 의 lag_1d·lag_7d·mean_prev7d 와 일치하는지 (무료 정렬 검증)."""
    di = (df["date"].to_numpy("datetime64[D]") - np.datetime64(start, "D")).astype("int64")
    si = sc_to_row[df["station_code"].to_numpy()] * seq.HOURS_PER_DAY + df["hour_code"].to_numpy()
    keep = di >= window
    di, si = di[keep], si[keep]
    lag1 = history_flat[si, di - 1]
    lag7 = history_flat[si, di - 7]
    with np.errstate(invalid="ignore"):
        mean7 = np.nanmean(history_flat[si[:, None], di[:, None] - np.arange(1, 8)], axis=1)
    ref1 = df["lag_1d"].to_numpy()[keep]
    ref7 = df["lag_7d"].to_numpy()[keep]
    refm = df["mean_prev7d"].to_numpy()[keep]
    both = ~(np.isnan(lag1) | np.isnan(ref1))
    return {
        "rows": int(keep.sum()),
        "lag_1d": float(np.mean(np.isclose(lag1[both], ref1[both], rtol=1e-4, atol=1e-3))),
        "lag_7d": float(np.mean(np.isclose(lag7[both], ref7[both], rtol=1e-4, atol=1e-3))),
        "mean_prev7d": float(np.mean(np.isclose(mean7[both], refm[both], atol=0.06))),
    }


def leakage_check(seq_flat: np.ndarray, series: np.ndarray, day: np.ndarray, window: int) -> dict:
    """창의 마지막 날짜가 대상일보다 앞서는지, 미래 타깃을 뒤섞어도 창이 변하지 않는지.

    마지막 날의 타깃 값을 셔플한 뒤 표본 행의 창을 다시 gather 해 비교한다.
    마지막 날은 어떤 표본 행의 창에도 들어가지 않으므로(창 최대 = 대상일-1) 입력이 변하면 누수다.
    """
    n = min(len(day), 1000)
    series, day = series[:n], day[:n]
    offsets = day[:, None] - window + np.arange(window, dtype="int64")
    last_before = bool(np.all(offsets.max(axis=1) < day))
    before = seq_flat[series[:, None], offsets]
    changed = seq_flat.copy()
    rng = np.random.default_rng(seq.SEED)
    for s in np.unique(series):
        changed[s, -1, :] = rng.permutation(changed[s, -1, :])
    after = changed[series[:, None], offsets]
    return {"window_before_target": last_before,
            "target_change_keeps_input": bool(np.array_equal(before, after, equal_nan=True)),
            "sampled": n}


def make_model(cell: str, n_stations: int, n_lines: int, hidden: int) -> nn.Module:
    return seq.SequenceRegressor(cell, n_stations, n_lines, hidden=hidden)


def to_tensors(station_of_series, line_of_series, hour_of_series, series):
    return (torch.from_numpy(station_of_series[series]),
            torch.from_numpy(line_of_series[series]),
            torch.from_numpy(hour_of_series[series]))


def predict_people(
    model: nn.Module, seq_flat: np.ndarray, ctx_flat: np.ndarray, target_flat: np.ndarray,
    series: np.ndarray, day: np.ndarray, scale_flat: np.ndarray,
    station_of_series, line_of_series, hour_of_series, window: int, batch: int = 1024,
) -> np.ndarray:
    model.eval()
    out = np.empty(len(series), dtype="float32")
    with torch.no_grad():
        for i in range(0, len(series), batch):
            s, d = series[i:i + batch], day[i:i + batch]
            xs, xc, _ = seq.gather_batch(seq_flat, ctx_flat, target_flat, s, d, window)
            st, ln, hc = to_tensors(station_of_series, line_of_series, hour_of_series, s)
            pred = model(torch.from_numpy(xs), torch.from_numpy(xc), st, ln, hc)
            out[i:i + len(s)] = pred.numpy()
    return out * scale_flat[series]


def train_one_epoch(
    model, optimizer, criterion, seq_flat, ctx_flat, target_flat,
    series, day, scale_flat, station_of_series, line_of_series, hour_of_series,
    window: int, batch: int, rng: np.random.Generator,
) -> float:
    model.train()
    perm = rng.permutation(len(series))
    total, count = 0.0, 0
    for i in range(0, len(perm), batch):
        take = perm[i:i + batch]
        s, d = series[take], day[take]
        xs, xc, y = seq.gather_batch(seq_flat, ctx_flat, target_flat, s, d, window)
        st, ln, hc = to_tensors(station_of_series, line_of_series, hour_of_series, s)
        optimizer.zero_grad()
        loss = criterion(model(torch.from_numpy(xs), torch.from_numpy(xc), st, ln, hc),
                         torch.from_numpy(y))
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(take)
        count += len(take)
    return total / max(count, 1)


def metrics_block(y: np.ndarray, pred: np.ndarray, day: np.ndarray,
                  holiday: np.ndarray, horizon_base: int,
                  tertile: np.ndarray, series: np.ndarray) -> dict:
    err = y - pred
    out: dict = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "n": int(len(y)),
        "negative": {"count": int((pred < 0).sum()), "min": float(pred.min())},
        "horizon": {},
        "tertile": {},
        "holiday": {},
    }
    horizon = day - horizon_base
    for lo, hi in cf.HORIZON_BUCKETS:
        m = (horizon >= lo) & (horizon <= hi)
        if m.any():
            e = err[m]
            out["horizon"][f"{lo}-{hi}"] = {
                "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                "n": int(m.sum()),
            }
    for label, code in (("소", 0), ("중", 1), ("대", 2)):
        m = tertile[series] == code
        if m.any():
            e = err[m]
            out["tertile"][label] = {
                "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                "n": int(m.sum()),
            }
    for label, code in (("공휴일", 1), ("비공휴일", 0)):
        m = holiday[day] == code
        if m.any():
            e = err[m]
            out["holiday"][label] = {
                "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                "n": int(m.sum()),
            }
    return out


def checkpoint_payload(model, cell, hidden, window, arrays: dict, trained: dict) -> dict:
    return {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cell": cell,
        "hidden": hidden,
        "dropout": seq.DROPOUT,
        "window": window,
        "seq_features": list(seq.SEQ_FEATURES),
        "ctx_features": list(seq.CTX_FEATURES),
        "embedding_dims": dict(seq.EMBEDDING_DIMS),
        "trained": trained,
        **arrays,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cells", default="D,E")
    parser.add_argument("--max-epochs", type=int, default=seq.MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=seq.BATCH_SIZE)
    parser.add_argument("--hidden", type=int, default=seq.HIDDEN)
    parser.add_argument("--window", type=int, default=seq.WINDOW)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--no-refit", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads or min(12, os.cpu_count() or 1))
    np.random.seed(seq.SEED)
    torch.manual_seed(seq.SEED)
    t_start = time.perf_counter()

    print("[1/6] CSV 로드")
    frames = {name: load_split(name) for name in ("train", "valid", "test")}
    frames = {name: cf.add_month(df) for name, df in frames.items()}
    for name, df in frames.items():
        print(f"  {name}: {len(df):,}행 · {df['date'].min().date()} ~ {df['date'].max().date()}")

    print("[2/6] 큐브 생성 + npz 교차검증")
    history, station_codes, sc_to_row = build_cube(frames)
    dates = np.unique(np.concatenate(
        [df["date"].to_numpy("datetime64[D]") for df in frames.values()])).astype(str)
    start = str(dates[0])
    n_st, n_days = history.shape[0], history.shape[2]
    cross = {"npz": False}
    if HISTORY_NPZ.exists():
        with np.load(HISTORY_NPZ, allow_pickle=False) as z:
            old = z["boarding"]
            old_codes = z["station_codes"]
            old_dates = np.asarray([str(d)[:10] for d in z["dates"]])
            same_grid = (len(old_codes) == n_st and int(old_codes.max()) == n_st - 1
                         and np.array_equal(old_dates, dates))
            if same_grid:
                both = np.isfinite(history) & np.isfinite(old)
                cross = {
                    "npz": True, "same_grid": True,
                    "max_abs_diff": float(np.max(np.abs(history[both] - old[both]))),
                    "compared": int(both.sum()),
                }
            else:
                cross = {"npz": True, "same_grid": False, "old_days": len(old_dates)}
    print(f"  큐브 {history.shape} · {n_days}일 · 교차검증 {cross}")

    print("[3/6] 달력·날씨·프로파일")
    weekday, holiday, temp, rain, snow = calendar_and_weather(frames, start, n_days)
    train_end_idx = int(np.searchsorted(dates, TRAIN_END, side="right"))
    valid_end_idx = int(np.searchsorted(dates, EVAL_END, side="right"))
    test_end_idx = int(np.searchsorted(dates, DATA_END, side="right"))
    profiles = fill_profiles(frames, sc_to_row, start, n_st, n_days)

    print("[4/6] 텐서 생성 (창은 미리 만들지 않는다)")
    scale = seq.series_mean(history, train_end_idx)
    scale_flat = scale.reshape(-1)
    temp_train = temp[:, :train_end_idx]
    temp_mean = float(np.nanmean(temp_train))
    temp_std = float(np.nanstd(temp_train)) or 1.0
    seq_t = seq.sequence_tensor(history, scale, weekday, holiday, temp, rain, snow,
                                temp_mean, temp_std)
    ctx_t = seq.context_tensor(
        weekday=weekday, is_holiday=holiday, temperature=temp, rainfall=rain, snowfall=snow,
        profile_month_mean=profiles[0], profile_n=profiles[1],
        profile_weekday_mean=profiles[2], profile_weekday_n=profiles[3],
        scale=scale, temp_mean=temp_mean, temp_std=temp_std,
    )
    profile_scaler = seq.fit_profile_scaler(ctx_t, train_end_idx)
    target = (history / scale[:, :, None]).astype("float32")
    seq_flat = seq_t.reshape(-1, n_days, len(seq.SEQ_FEATURES))
    ctx_flat = ctx_t.reshape(-1, n_days, len(seq.CTX_FEATURES))
    target_flat = target.reshape(-1, n_days)
    history_flat = history.reshape(-1, n_days)
    station_of_series = np.repeat(np.arange(n_st), seq.HOURS_PER_DAY)
    hour_of_series = np.tile(np.arange(seq.HOURS_PER_DAY), n_st)
    line_of_station = np.zeros(n_st, dtype="int64")
    pairs = pd.concat(frames.values(), ignore_index=True)[["station_code", "line_code"]] \
        .drop_duplicates("station_code")
    line_of_station[sc_to_row[pairs["station_code"].to_numpy()]] = pairs["line_code"].to_numpy()
    line_of_series = line_of_station[station_of_series]
    n_lines = int(line_of_station.max()) + 1

    mask = seq.window_mask(history, args.window)
    train_series, train_day = seq.split_index(mask, args.window, train_end_idx)
    valid_series, valid_day = seq.split_index(mask, train_end_idx, valid_end_idx)
    test_series, test_day = seq.split_index(mask, valid_end_idx, test_end_idx)
    excluded = {
        "train": int(len(frames["train"]) - len(train_series)),
        "valid": int(len(frames["valid"]) - len(valid_series)),
        "test": int(len(frames["test"]) - len(test_series)),
    }
    print(f"  train {len(train_series):,} · valid {len(valid_series):,} · test {len(test_series):,}")
    print(f"  제외(행 부족·NaN): {excluded}")

    lag_valid = lag_check("valid", frames["valid"], history_flat, sc_to_row, start, args.window)
    lag_test = lag_check("test", frames["test"], history_flat, sc_to_row, start, args.window)
    leak = leakage_check(seq_flat, test_series[:1000], test_day[:1000], args.window)
    print(f"  lag 일치 valid {lag_valid} · test {lag_test}")
    print(f"  누수 {leak}")

    report = {
        "window": args.window,
        "split_days": {"train": train_end_idx, "valid": valid_end_idx - train_end_idx,
                       "test": test_end_idx - valid_end_idx},
        "rows": {"train": len(train_series), "valid": len(valid_series), "test": len(test_series)},
        "excluded": excluded,
        "lag": {"valid": lag_valid, "test": lag_test},
        "leakage": leak,
        "cube": cross,
        "scale": {
            "min": float(scale_flat.min()), "p50": float(np.median(scale_flat)),
            "max": float(scale_flat.max()),
            "examples": {str(code): float(scale.reshape(-1)[code])
                         for code in np.argsort(scale_flat)[[0, len(scale_flat) // 2, -1]]},
        },
    }
    if args.verify_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    tertile_edges = np.quantile(scale_flat, [1 / 3, 2 / 3])
    tertile = np.digitize(scale_flat, tertile_edges)
    y_test = history_flat[test_series, test_day]
    horizon_base = valid_end_idx - 1
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    max_epochs = 1 if args.smoke else args.max_epochs
    results: dict[str, dict] = {}

    for cell in cells:
        name = {"D": "LSTM", "E": "GRU"}.get(cell)
        if name is None:
            raise SystemExit(f"셀 이름은 D 또는 E 여야 한다: {cell}")
        print(f"\n[{cell}] {name} 학습 시작 (hidden {args.hidden}, batch {args.batch_size}, "
              f"max {max_epochs} epoch)")
        torch.manual_seed(seq.SEED)
        model = make_model(name, n_st, n_lines, args.hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=seq.LEARNING_RATE)
        criterion = nn.SmoothL1Loss()
        rng = np.random.default_rng(seq.SEED)
        best_mae, best_epoch, best_state, bad = float("inf"), -1, None, 0
        epoch_log = []
        for epoch in range(max_epochs):
            t0 = time.perf_counter()
            loss = train_one_epoch(
                model, optimizer, criterion, seq_flat, ctx_flat, target_flat,
                train_series, train_day, scale_flat,
                station_of_series, line_of_series, hour_of_series,
                args.window, args.batch_size, rng,
            )
            valid_pred = predict_people(
                model, seq_flat, ctx_flat, target_flat, valid_series, valid_day, scale_flat,
                station_of_series, line_of_series, hour_of_series, args.window,
            )
            valid_y = history_flat[valid_series, valid_day]
            valid_mae = float(np.mean(np.abs(valid_y - valid_pred)))
            seconds = time.perf_counter() - t0
            epoch_log.append({"epoch": epoch + 1, "loss": loss, "valid_mae": valid_mae,
                              "seconds": seconds})
            print(f"  epoch {epoch + 1:2d}  loss {loss:.5f}  valid MAE {valid_mae:8.2f}  "
                  f"{seconds:6.1f}s")
            if valid_mae < best_mae - 1e-6:
                best_mae, best_epoch, bad = valid_mae, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= seq.PATIENCE:
                    print(f"  조기 종료 (patience {seq.PATIENCE})")
                    break

        if args.smoke:
            results[cell] = {"smoke": True, "epochs": epoch_log, "best_valid_mae": best_mae,
                             "epoch_seconds": [e["seconds"] for e in epoch_log]}
            continue

        valid_model = make_model(name, n_st, n_lines, args.hidden)
        valid_model.load_state_dict(best_state)
        valid_pred = predict_people(
            valid_model, seq_flat, ctx_flat, target_flat, valid_series, valid_day, scale_flat,
            station_of_series, line_of_series, hour_of_series, args.window,
        )
        y_valid = history_flat[valid_series, valid_day]
        valid_block = metrics_block(y_valid, valid_pred, valid_day, holiday,
                                    train_end_idx - 1, tertile, valid_series)

        refit_epochs = 0 if args.no_refit else best_epoch + 1
        if refit_epochs:
            print(f"  train+valid refit {refit_epochs} epoch (Valid 최고 epoch 재사용)")
            torch.manual_seed(seq.SEED)
            final_model = make_model(name, n_st, n_lines, args.hidden)
            optimizer = torch.optim.Adam(final_model.parameters(), lr=seq.LEARNING_RATE)
            rng = np.random.default_rng(seq.SEED)
            combined_series, combined_day = seq.split_index(mask, args.window, valid_end_idx)
            for epoch in range(refit_epochs):
                t0 = time.perf_counter()
                loss = train_one_epoch(
                    final_model, optimizer, criterion, seq_flat, ctx_flat, target_flat,
                    combined_series, combined_day, scale_flat,
                    station_of_series, line_of_series, hour_of_series,
                    args.window, args.batch_size, rng,
                )
                print(f"    refit {epoch + 1:2d}/{refit_epochs}  loss {loss:.5f}  "
                      f"{time.perf_counter() - t0:6.1f}s")
        else:
            final_model = valid_model

        test_pred = predict_people(
            final_model, seq_flat, ctx_flat, target_flat, test_series, test_day, scale_flat,
            station_of_series, line_of_series, hour_of_series, args.window,
        )
        test_block = metrics_block(y_test, test_pred, test_day, holiday, horizon_base,
                                   tertile, test_series)
        valid_ckpt_pred = predict_people(
            valid_model, seq_flat, ctx_flat, target_flat, test_series, test_day, scale_flat,
            station_of_series, line_of_series, hour_of_series, args.window,
        )
        valid_ckpt_block = metrics_block(y_test, valid_ckpt_pred, test_day, holiday, horizon_base,
                                         tertile, test_series)
        print(f"  Test(headline, refit): MAE {test_block['mae']:.2f}  "
              f"RMSE {test_block['rmse']:.2f}  n {test_block['n']:,}")
        print(f"  Test(valid ckpt)     : MAE {valid_ckpt_block['mae']:.2f}  "
              f"RMSE {valid_ckpt_block['rmse']:.2f}")

        arrays = {
            "series_mean": scale,
            "profile_mean": profile_scaler.mean,
            "profile_std": profile_scaler.std,
            "temp_mean": np.float32(temp_mean),
            "temp_std": np.float32(temp_std),
            "dates": np.asarray(dates, dtype=str),
            "weekday": weekday,
            "is_holiday": holiday,
            "temperature": temp,
            "rainfall": rain,
            "snowfall": snow,
            "station_codes": station_codes,
            "line_of_station": line_of_station,
            "n_stations": n_st,
            "n_lines": n_lines,
        }
        trained = {
            "best_epoch": best_epoch + 1,
            "epochs_run": len(epoch_log),
            "best_valid_mae": best_mae,
            "epoch_seconds": [e["seconds"] for e in epoch_log],
            "total_seconds": float(sum(e["seconds"] for e in epoch_log)),
            "refit_epochs": refit_epochs,
            "seed": seq.SEED,
            "batch_size": args.batch_size,
            "learning_rate": seq.LEARNING_RATE,
            "hidden": args.hidden,
            "loss": "SmoothL1Loss",
        }
        payload = checkpoint_payload(final_model, name, args.hidden, args.window, arrays, trained)
        path = MODELS / f"sequence_{cell}.pt"
        seq.save_checkpoint(path, payload)
        results[cell] = {
            "cell": name,
            "valid_checkpoint": valid_block,
            "headline": test_block,
            "test_valid_checkpoint": valid_ckpt_block,
            "epochs": epoch_log,
            "trained": trained,
            "checkpoint": str(path.relative_to(ROOT)),
        }
        print(f"  저장: {path}")

    if args.smoke:
        print(json.dumps({"smoke": True, "results": results}, ensure_ascii=False, indent=2))
        return

    document = {
        "generated": "2026-10-22",
        "protocol": {
            "window": args.window,
            "source": "data/processed/{train,valid,test}.csv.gz",
            "normalization": "boarding / series train mean (per station-hour)",
            "profiles": "train as-of · valid frozen at 2025-10-01 · test cal_profiles_eval.npz (2026-02-14)",
            "selection": "Valid MAE only; Test once after training",
            "headline": "train+valid refit at best valid epoch",
        },
        "dataset": report,
        "tertile_edges_train_mean": [float(v) for v in tertile_edges],
        "models": results,
        "test": {cell: r["headline"] for cell, r in results.items()},
        "elapsed_seconds": float(time.perf_counter() - t_start),
    }
    METRICS.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {METRICS}")
    print(f"총 소요 {document['elapsed_seconds'] / 60:.1f}분")


if __name__ == "__main__":
    main()
