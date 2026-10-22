"""달력(프로필) 피처 백테스트 — 1단계. 콘솔 표만 출력하고 파일은 저장하지 않는다.

실행 (프로젝트 루트에서)
    python notebooks/backtest_calendar.py
    python notebooks/backtest_calendar.py --recency-weighted   # 기존 vs 가중 프로필 채택 판정

무엇을 보는가
    * train = date <= cut  /  eval = cut < date <= 2026-02-14 (valid 끝)
    * train 프로필은 as-of (그 날짜 이전만), eval 프로필은 cut 시점 frozen
    * 사다리 8런그: A0 → A(티어 A) → +hour_code → +weekday(요일 코드만, 비교용) →
      +weekday_profile(역×시간대×요일 as-of 프로필) → +is_weekend → +is_holiday(티어 B)
      → +temperature/rainfall/snowfall(티어 C)
    * 지표는 horizon 구간(1-31 / 32-91 / 92-181 / 182-366일)별 + 전체
    * baseline: 계절 naive — 프로필 평균을 그대로 예측값으로 쓴다

누수 검사
    (a) 샘플 행의 프로필이 정말 그 날짜 이전 행만으로 계산됐는지 직접 재계산해 대조
    (b) eval 타깃을 셔플해도 예측이 변하지 않는지
    (c) profile_n 분위별 오차 (표본이 적은 구간에서 오차가 큰지)

주의
    test.csv.gz 는 읽지 않는다.
    학습 파라미터는 런그마다 완전히 동일하다 (오직 피처만 다르다).
    학습 행을 줄여야 하면 환경변수 METRO_CAL_MAX_TRAIN_ROWS=500000 처럼 지정한다 (기본: 전부).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calendar_features as cf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

CUTS = ["2025-03-31", "2025-10-01"]
EVAL_END = "2026-02-14"
SEED = 42

# train_abc.py 와 같은 값 (모든 런그 동일)
PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "random_state": SEED,
}

USE_COLS = [
    "date", "station_code", "line_code", "hour_code", "weekday",
    "is_weekend", "is_holiday", "temperature", "rainfall", "snowfall", "boarding",
]

BASE = ["station_code", "line_code"]
A_COLS = ["profile_month_mean", "profile_n"] + BASE          # 티어 A (train_calendar.py 계약과 동일)
WD_COLS = ["profile_weekday_mean", "profile_weekday_n"]      # 티어 B 의 요일 프로필
B_EXTRA = WD_COLS + ["hour_code", "weekday", "is_weekend", "is_holiday"]  # 티어 B 추가분
C_EXTRA = ["temperature", "rainfall", "snowfall"]               # 티어 C 추가분

LADDER = [
    ("A0", ["profile_sh_mean", "profile_sh_n"] + BASE, "-"),
    ("A", A_COLS, "A"),
    ("A+hour_code", A_COLS + ["hour_code"], "A"),
    ("A+weekday", A_COLS + ["hour_code", "weekday", "is_weekend"], "A"),  # 요일 코드만 (비교용)
    ("A+weekday_profile", A_COLS + WD_COLS + ["hour_code", "weekday"], "A"),
    ("A+weekday_profile+weekend", A_COLS + WD_COLS + ["hour_code", "weekday", "is_weekend"], "A"),
    ("A+is_holiday", A_COLS + B_EXTRA, "B"),
    ("A+weather", A_COLS + B_EXTRA + C_EXTRA, "C"),
]

# 최근성 가중 비교에 쓰는 생산 티어 3개 (A/B/C). 이름은 feature 계약과 같다.
PROD_RUNGS = [
    ("A", A_COLS, "A"),
    ("B", A_COLS + B_EXTRA, "B"),
    ("C", A_COLS + B_EXTRA + C_EXTRA, "C"),
]

# 채택 기준: 92~181·182~366 구간이 기존보다 나쁘지 않고, 182~366 이 평균 5명 이상 개선
ADOPT_MIN_GAIN = 5.0


# ───────────────────────── 유틸 ─────────────────────────
def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def score(y: np.ndarray, pred: np.ndarray, horizon: np.ndarray | None = None) -> dict:
    """전체 + horizon 구간별 MAE/RMSE. NaN 예측은 없어야 한다."""
    y = np.asarray(y, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    err = y - pred
    out = {
        "overall": {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "n": int(len(y)),
        }
    }
    if horizon is not None:
        bucket = cf.horizon_bucket(horizon)
        for lo, hi in cf.HORIZON_BUCKETS:
            label = f"{lo}-{hi}"
            m = bucket == label
            if m.any():
                e = err[m]
                out[label] = {
                    "mae": float(np.mean(np.abs(e))),
                    "rmse": float(np.sqrt(np.mean(e ** 2))),
                    "n": int(m.sum()),
                }
    return out


def fmt_row(name: str, tier: str, sc: dict) -> dict:
    row = {"rung": name, "tier": tier,
           "MAE": round(sc["overall"]["mae"], 2), "RMSE": round(sc["overall"]["rmse"], 2)}
    for lo, hi in cf.HORIZON_BUCKETS:
        label = f"{lo}-{hi}"
        row[label] = round(sc[label]["mae"], 2) if label in sc else np.nan
    return row


def build_eval_features(eval_df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """eval 피처 = 고정 프로필 + 달력·날씨. eval 의 boarding 은 보지 않는다 (누수 검사 b 의 근거)."""
    out = pd.DataFrame(index=eval_df.index)
    for col in cf.PROFILE_COLS_SH + cf.PROFILE_COLS_MONTH + cf.PROFILE_COLS_WEEKDAY:
        out[col] = profiles[col].to_numpy()
    for col in ["station_code", "line_code", "hour_code", "weekday", "is_weekend", "is_holiday",
                "temperature", "rainfall", "snowfall"]:
        out[col] = eval_df[col].to_numpy()
    return out


# ───────────────────────── 누수 검사 ─────────────────────────
def check_asof_direct(df: pd.DataFrame, cols: list[str], key_cols: list[str], n_sample: int, label: str) -> None:
    """(a) 샘플 행의 프로필을 원본에서 직접 재계산해 대조한다. 이전 날짜만 쓰는지 확인."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    n_col, mean_col = cols[1], cols[0]
    bad_mean = bad_n = 0
    for i in np.sort(idx):
        row = df.iloc[i]
        prior = df[df["date"] < row["date"]]
        for k in key_cols:
            prior = prior[prior[k] == row[k]]
        if int(row[n_col]) != len(prior):
            bad_n += 1
        if len(prior):
            expected = float(prior["boarding"].mean())
            if not np.isclose(float(row[mean_col]), expected, rtol=1e-5, atol=1e-2):  # float32 저장 오차 허용
                bad_mean += 1
        elif not np.isnan(float(row[mean_col])):
            bad_mean += 1
    status = "OK" if (bad_n == 0 and bad_mean == 0) else "FAIL"
    print(f"  (a) {label:<26} 샘플 {len(idx):>3}행  표본수 불일치 {bad_n} · 평균 불일치 {bad_mean}  → {status}")
    assert bad_n == 0 and bad_mean == 0, f"as-of 프로필이 이전 날짜만 쓰지 않는다 ({label})"


def check_target_shuffle(model, X1: pd.DataFrame, X2: pd.DataFrame) -> None:
    """(b) eval 타깃을 셔플해도 피처·예측이 그대로인지."""
    same_x = np.array_equal(X1.to_numpy(dtype="float64", na_value=np.nan),
                            X2.to_numpy(dtype="float64", na_value=np.nan), equal_nan=True)
    p1 = model.predict(X1)
    p2 = model.predict(X2)
    same_p = np.allclose(p1, p2, atol=0.0)
    print(f"  (b) eval 타깃 셔플 후        피처 동일 {same_x} · 예측 동일 {same_p}  → {'OK' if same_x and same_p else 'FAIL'}")
    assert same_x and same_p, "eval 타깃이 피처·예측에 새고 있다"


def profile_n_table(y: np.ndarray, pred: np.ndarray, n: np.ndarray) -> pd.DataFrame:
    """(c) profile_n 분위별 MAE. 표본수가 모두 같으면 한 줄로 보여준다."""
    n = np.asarray(n)
    if pd.Series(n).nunique() < 2:
        q = pd.Series("all", index=range(len(n)))
    else:
        q = pd.qcut(pd.Series(n), 5, duplicates="drop")
    df = pd.DataFrame({"q": q.to_numpy(), "abs_err": np.abs(np.asarray(y, dtype="float64") - pred), "n": n})
    g = df.groupby("q", observed=True).agg(MAE=("abs_err", "mean"), rows=("n", "size"),
                                           profile_n_median=("n", "median"))
    g["MAE"] = g["MAE"].round(2)
    return g.reset_index(drop=True)


# ───────────────────────── 한 cut 실행 ─────────────────────────
def run_cut(cut: str, base: pd.DataFrame, max_rows: int, weighted: bool = False,
            ladder: list[tuple[str, list[str], str]] | None = None,
            check_leaks: bool = True) -> tuple[pd.DataFrame, dict[str, dict]]:
    ladder = LADDER if ladder is None else ladder
    cut_ts = pd.Timestamp(cut)
    train = base[base["date"] <= cut_ts].copy()
    eval_mask = (base["date"] > cut_ts) & (base["date"] <= pd.Timestamp(EVAL_END))
    ev = base.loc[eval_mask].copy()
    horizon = (ev["date"] - cut_ts).dt.days.to_numpy()
    print(f"\n{'-' * 78}\ncut = {cut}   train {len(train):,}행 "
          f"({train['date'].min().date()}~{train['date'].max().date()})   "
          f"eval {len(ev):,}행 ({ev['date'].min().date()}~{ev['date'].max().date()}, "
          f"{horizon.min()}~{horizon.max()}일 뒤)")

    # train = as-of, eval = cut 시점 frozen
    t0 = time.time()
    train = pd.concat([train, cf.asof_profile_frame(train, recency=weighted)], axis=1)
    profiles = cf.frozen_profile_frame(train, ev, cut, recency=weighted)
    ev_x = build_eval_features(ev, profiles)
    label = "가중 프로필" if weighted else "기본 프로필"
    print(f"  {label} · 프로필 계산 {time.time() - t0:.1f}s · as-of profile_n=0 비율 "
          f"{float((train['profile_n'] == 0).mean()):.3f} · eval profile_n 최소 {int(profiles['profile_n'].min())}")

    if check_leaks:
        check_asof_direct(train, cf.PROFILE_COLS_SH, cf.KEY_SH, 20, "station×hour")
        check_asof_direct(train, cf.PROFILE_COLS_MONTH, cf.KEY_SH_MONTH, 10, "station×hour×month")
        check_asof_direct(train, cf.PROFILE_COLS_WEEKDAY, cf.KEY_SH_WEEKDAY, 15, "station×hour×weekday")

    y_train = train["boarding"].to_numpy(dtype="float64")
    y_eval = ev["boarding"].to_numpy(dtype="float64")

    fit_idx = None
    if max_rows and len(train) > max_rows:
        rng = np.random.default_rng(SEED)
        fit_idx = np.sort(rng.choice(len(train), size=max_rows, replace=False))
        print(f"  학습 행 제한: {max_rows:,} (METRO_CAL_MAX_TRAIN_ROWS)")

    rows = []
    raw: dict[str, dict] = {}
    # baseline: 계절 naive (프로필 평균 그대로)
    naive = ev_x["profile_sh_mean"].to_numpy(dtype="float64")
    assert not np.isnan(naive).any(), "frozen 프로필에 빈 키가 있다"
    raw["naive(sh_mean)"] = score(y_eval, naive, horizon)
    rows.append(fmt_row("naive(sh_mean)", "-", raw["naive(sh_mean)"]))

    best_model = None
    for name, cols, tier in ladder:
        t0 = time.time()
        model = HistGradientBoostingRegressor(**PARAMS)
        if fit_idx is None:
            model.fit(train[cols], y_train)
        else:
            model.fit(train.iloc[fit_idx][cols], y_train[fit_idx])
        pred = model.predict(ev_x[cols])
        sc = score(y_eval, pred, horizon)
        raw[name] = sc
        rows.append(fmt_row(name, tier, sc))
        best_model = (model, cols, pred)
        print(f"  {name:<14} fit {time.time() - t0:5.1f}s  전체 MAE {sc['overall']['mae']:8.2f}  "
              f"RMSE {sc['overall']['rmse']:8.2f}")

    # (b) 타깃 셔플
    rng = np.random.default_rng(SEED)
    ev_shuf = ev.copy()
    ev_shuf["boarding"] = rng.permutation(ev_shuf["boarding"].to_numpy())
    ev_x_shuf = build_eval_features(ev_shuf, profiles)
    model, cols, pred = best_model
    check_target_shuffle(model, ev_x[cols], ev_x_shuf[cols])

    # (c) profile_n 분위별 오차
    print("  (c) eval profile_n 분위별 오차 (최종 런그):")
    tbl = profile_n_table(y_eval, pred, profiles["profile_n"].to_numpy())
    print(tbl.to_string(index=False))

    return pd.DataFrame(rows), raw


# ───────────────────────── main ─────────────────────────
def pooled(entries: list[dict], bucket: str) -> tuple[float, int]:
    """여러 cut 의 구간 점수를 행수(n) 가중으로 합친다."""
    parts = [e[bucket] for e in entries if bucket in e]
    n = sum(p["n"] for p in parts)
    if not n:
        return (float("nan"), 0)
    return (sum(p["mae"] * p["n"] for p in parts) / n, n)


def compare_weighted(base: pd.DataFrame, max_rows: int) -> bool:
    """기존 프로필 vs 최근성 가중 프로필을 같은 사다리(A/B/C)로 비교하고 채택 여부를 판정한다.

    채택 기준: 92~181·182~366 구간 MAE 가 세 모델 모두 기존보다 나쁘지 않고,
    182~366 구간 개선폭이 세 모델 평균 5명 이상.
    """
    header("최근성 가중 프로필 비교 — 기존 vs 가중 (같은 사다리: A/B/C)")
    raws: dict[tuple[str, str], dict[str, dict]] = {}
    for cut in CUTS:
        for variant, weighted in (("base", False), ("weighted", True)):
            _, raw = run_cut(cut, base, max_rows, weighted=weighted,
                             ladder=PROD_RUNGS, check_leaks=False)
            raws[(cut, variant)] = raw

    header("cut 별 비교 (MAE, 명 — 낮을수록 좋음)")
    for cut in CUTS:
        rows = []
        for model in ("A", "B", "C"):
            b = raws[(cut, "base")][model]
            w = raws[(cut, "weighted")][model]
            row = {"model": model, "전체 기존": round(b["overall"]["mae"], 2),
                   "전체 가중": round(w["overall"]["mae"], 2)}
            for bucket in ("92-181", "182-366"):
                if bucket in b and bucket in w:
                    row[f"{bucket} 기존"] = round(b[bucket]["mae"], 2)
                    row[f"{bucket} 가중"] = round(w[bucket]["mae"], 2)
            rows.append(row)
        print(f"\ncut = {cut}")
        print(pd.DataFrame(rows).to_string(index=False))

    header("cut 합산 (행수 가중 평균) 과 판정")
    rows = []
    checks = []
    for model in ("A", "B", "C"):
        be = [raws[(cut, "base")][model] for cut in CUTS]
        we = [raws[(cut, "weighted")][model] for cut in CUTS]
        row = {"model": model}
        for bucket in ("92-181", "182-366"):
            bm, bn = pooled(be, bucket)
            wm, wn = pooled(we, bucket)
            gain = bm - wm
            row[f"{bucket} 기존"] = round(bm, 2)
            row[f"{bucket} 가중"] = round(wm, 2)
            row[f"{bucket} 개선"] = round(gain, 2)
            checks.append((model, bucket, gain))
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    nonworse = all(gain >= -1e-9 for _, _, gain in checks)
    gains_182 = [gain for _, bucket, gain in checks if bucket == "182-366"]
    mean_gain = float(np.mean(gains_182)) if gains_182 else float("nan")
    adopt = bool(nonworse and mean_gain >= ADOPT_MIN_GAIN)
    print(f"\n  나쁜 구간 없음: {nonworse} · 182~366 평균 개선: {mean_gain:.2f}명 "
          f"(기준 {ADOPT_MIN_GAIN:.0f}명 이상)")
    print(f"  판정: {'채택' if adopt else '폐기'}")
    return adopt


def main() -> None:
    max_rows = int(os.environ.get("METRO_CAL_MAX_TRAIN_ROWS", "0") or 0)
    compare = "--recency-weighted" in sys.argv[1:]
    header("데이터 로드 (test.csv.gz 는 읽지 않는다)")
    frames = [cf.load_processed(DATA / f, columns=USE_COLS) for f in ("train.csv.gz", "valid.csv.gz")]
    base = pd.concat(frames, ignore_index=True)
    base["date"] = pd.to_datetime(base["date"])
    print(f"train+valid {len(base):,}행 · {base['date'].min().date()} ~ {base['date'].max().date()} · "
          f"역 {base['station_code'].nunique()}개 · 월 컬럼 파생 {sorted(base['month'].unique())}")

    if compare:
        compare_weighted(base, max_rows)
        return

    results = {}
    for cut in CUTS:
        table, _ = run_cut(cut, base, max_rows)
        results[cut] = table

        header(f"cut = {cut} — 사다리 (horizon 구간별 MAE, 단위 명)")
        show = results[cut].copy()
        show.columns = ["런그", "티어", "MAE", "RMSE", "1-31일", "32-91일", "92-181일", "182-366일"]
        print(show.to_string(index=False))

    header("요약 — 전체 MAE (낮을수록 좋음)")
    summary = pd.DataFrame({cut: df.set_index("rung")["MAE"] for cut, df in results.items()})
    tiers = dict(zip([r[0] for r in LADDER], [r[2] for r in LADDER]))
    summary.insert(0, "tier", [tiers.get(i, "-") for i in summary.index])
    print(summary.to_string())
    print("\n해석: A0 → 티어A → 티어B → 티어C 로 갈 때 MAE 가 실제로 줄어드는지 본다. "
          "A+weekday(요일 코드만) 와 A+weekday_profile(역×시간대×요일 프로필) 의 차이가 요일 정보를 "
          "어떻게 넣었을 때의 값인지 보여준다. 기울기가 없으면 그 피처군은 이 데이터에서 값을 못 한다.")


if __name__ == "__main__":
    main()
