"""달력(프로필) 피처 계산 — 역·시간대·월별 과거 평균을 '그 날짜 이전'만으로 만든다.

핵심 규칙
    * as-of  : 각 행의 날짜 **이전** 데이터만 쓴다. 타깃 행 자신의 boarding 은 절대 들어가지 않는다.
    * frozen : cut 시점 이전 데이터만으로 키별 평균·표본수를 고정한다 (백테스트 평가용, 한 번만 계산).

구현 방식 (4.9M 행에서 동작해야 한다)
    * 정렬 후 groupby.cumsum() - 자기값  /  groupby.cumcount() 로 벡터화한다. 행마다 루프를 돌지 않는다.
    * 룩업은 정수 키(np.int64) + pd.Series.reindex 로 한다 (merge 보다 빠르고 메모리도 적다).

키
    SH      = (station_code, hour_code)
    SH_MONTH= (station_code, hour_code, month)

전제
    (station_code, hour_code, date) 조합은 유일하다 (preprocess.py 가 그렇게 만든다).
    같은 날짜에 같은 키가 두 줄 이상이면 as-of 가 그 날의 다른 줄을 보게 되므로 즉시 예외를 던진다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

KEY_SH = ["station_code", "hour_code"]
KEY_SH_MONTH = ["station_code", "hour_code", "month"]
KEY_SH_WEEKDAY = ["station_code", "hour_code", "weekday"]

HOURS_PER_DAY = 20          # hour_code 0~19
SLOTS = 13                  # 월(1~12) 또는 요일(0~6) 슬롯. 키에 없으면 0
_STATION_STRIDE = HOURS_PER_DAY * SLOTS

PROFILE_COLS_SH = ["profile_sh_mean", "profile_sh_n"]
PROFILE_COLS_MONTH = ["profile_month_mean", "profile_n"]
PROFILE_COLS_WEEKDAY = ["profile_weekday_mean", "profile_weekday_n"]

# 백테스트 horizon 구간 (cut 으로부터 며칠 뒤인지)
HORIZON_BUCKETS = [(1, 31), (32, 91), (92, 181), (182, 366)]

# 최근성 가중 프로필 (옵션) — 프로필 기준일로부터 182일 이내 관측에 가중치 2, 그 밖은 1.
# 기본값(False)은 기존과 같은 단순 평균이다.
RECENCY_WEIGHT_DAYS = 182
RECENCY_WEIGHT = 2.0
_RECENCY_KEY_STRIDE = 100_000  # (그룹키, 날짜) 합성키. 날짜 일수는 6자리 미만이라 겹치지 않는다.


# ───────────────────────── 로드 ─────────────────────────
def load_processed(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """data/processed/{train,valid,test}.csv.gz 를 읽고 date(날짜형)·month(1~12) 를 붙인다.

    상대 경로는 data/processed 기준으로 해석한다 (예: "train.csv.gz").
    """
    path = Path(path)
    if not path.is_absolute():
        path = DATA / path
    df = pd.read_csv(path, usecols=columns, compression="gzip")
    df["date"] = pd.to_datetime(df["date"])
    return add_month(df)


def add_month(df: pd.DataFrame) -> pd.DataFrame:
    """date 에서 month(1~12, int8) 를 파생한다. 이미 있으면 그대로 둔다."""
    if "month" not in df.columns:
        df = df.copy()
        df["month"] = df["date"].dt.month.astype("int8")
        return df
    if not pd.api.types.is_integer_dtype(df["month"]):
        df = df.copy()
        df["month"] = df["month"].astype("int8")
    return df


# ───────────────────────── 공통 ─────────────────────────
def _check_unique(df: pd.DataFrame, key_cols: list[str]) -> None:
    dup = int(df.duplicated(key_cols + ["date"]).sum())
    if dup:
        raise ValueError(f"{key_cols + ['date']} 조합이 {dup}줄 겹친다. "
                         "as-of 계산이 같은 날 다른 줄을 보게 된다.")


def _keys(df: pd.DataFrame, key_cols: list[str]) -> np.ndarray:
    """키를 단일 int64 로 인코딩한다 (룩업용). 키 구성이 다르면 값도 다르다.

    슬롯은 하나뿐이다 — key_cols 에 weekday 가 있으면 요일(0~6), month 가 있으면 월(1~12), 없으면 0.
    (두 키를 동시에 쓰지 않는다. 조합이 필요하면 키를 새로 정의한다.)
    """
    station = df["station_code"].to_numpy(dtype=np.int64)
    hour = df["hour_code"].to_numpy(dtype=np.int64)
    if "weekday" in key_cols:
        slot = df["weekday"].to_numpy(dtype=np.int64)
    elif "month" in key_cols:
        slot = df["month"].to_numpy(dtype=np.int64)
    else:
        slot = np.zeros(len(df), dtype=np.int64)
    return station * _STATION_STRIDE + hour * SLOTS + slot


# ───────────────────────── as-of ─────────────────────────
def _asof_recency_mean(
    order: pd.DataFrame,
    prev_sum: np.ndarray,
    prev_n: np.ndarray,
    key_cols: list[str],
    value: str,
) -> np.ndarray:
    """정렬된 프레임에서 '직전 182일 관측에 가중치 2'를 준 as-of 평균.

    가중평균 = (전체 이전 합 + 최근 182일 합) / (전체 이전 수 + 최근 182일 수).
    최근 구간은 정렬된 (그룹키, 날짜) 합성키에 searchsorted 한 번으로 찾는다 (행별 루프 없음).
    """
    keys = _keys(order, key_cols)
    days = order["date"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    composite = keys * _RECENCY_KEY_STRIDE + days
    cutoff = keys * _RECENCY_KEY_STRIDE + (days - RECENCY_WEIGHT_DAYS)
    start = np.searchsorted(composite, cutoff, side="left")

    values = order[value].to_numpy(dtype="float64")
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    pos = np.arange(len(order), dtype=np.int64)
    recent_sum = prefix[pos] - prefix[start]
    recent_n = pos - start

    denom = prev_n + recent_n
    return np.divide(
        prev_sum + recent_sum,
        denom,
        out=np.full(len(order), np.nan, dtype="float64"),
        where=denom > 0,
    )


def asof_profile(
    df: pd.DataFrame,
    key_cols: list[str],
    value: str = "boarding",
    mean_col: str = "mean",
    n_col: str = "n",
    recency: bool = False,
) -> pd.DataFrame:
    """각 행의 날짜 **이전** 행들만으로 키별 평균·표본수를 계산해 df.index 순서로 돌려준다.

    표본이 없는 첫 구간은 mean=NaN, n=0 이다 (HistGradientBoosting 은 NaN 을 그대로 처리한다).
    recency=True 면 그 행 날짜로부터 182일 이내 관측에 가중치 2를 준 가중평균을 쓴다.
    n 은 옵션과 무관하게 관측 수 그대로다 (feature 계약 유지).
    """
    for col in key_cols + [value, "date"]:
        if col not in df.columns:
            raise KeyError(f"컬럼이 없다: {col}")
    _check_unique(df, key_cols)

    order = df.sort_values(key_cols + ["date"], kind="mergesort")
    grp = order.groupby(key_cols, sort=False, observed=True)[value]
    # cumsum 은 자기 자신을 포함하므로 빼고, cumcount 가 곧 '이전 표본수'다
    prev_sum = grp.cumsum().to_numpy(dtype="float64") - order[value].to_numpy(dtype="float64")
    prev_n = grp.cumcount().to_numpy(dtype="int64")

    if recency:
        mean = _asof_recency_mean(order, prev_sum, prev_n, key_cols, value)
    else:
        mean = np.divide(
            prev_sum,
            prev_n,
            out=np.full(len(prev_n), np.nan, dtype="float64"),
            where=prev_n > 0,
        )

    back = pd.DataFrame({"_mean": mean, "_n": prev_n}, index=order.index).reindex(df.index)
    n = back["_n"].to_numpy(dtype="int64")
    return pd.DataFrame(
        {mean_col: back["_mean"].to_numpy(dtype="float64").astype("float32"),
         n_col: n.astype("int32")},
        index=df.index,
    )


def asof_profile_frame(df: pd.DataFrame, value: str = "boarding",
                       recency: bool = False) -> pd.DataFrame:
    """A0용 (역,시간대) + A용 (역,시간대,월) + B용 (역,시간대,요일) as-of 프로필 6컬럼을 한 번에 만든다."""
    sh = asof_profile(df, KEY_SH, value=value, mean_col=PROFILE_COLS_SH[0], n_col=PROFILE_COLS_SH[1],
                      recency=recency)
    mo = asof_profile(df, KEY_SH_MONTH, value=value, mean_col=PROFILE_COLS_MONTH[0],
                      n_col=PROFILE_COLS_MONTH[1], recency=recency)
    wd = asof_profile(df, KEY_SH_WEEKDAY, value=value,
                      mean_col=PROFILE_COLS_WEEKDAY[0], n_col=PROFILE_COLS_WEEKDAY[1],
                      recency=recency)
    return pd.concat([sh, mo, wd], axis=1)


# ───────────────────────── frozen ─────────────────────────
def frozen_profile(
    df: pd.DataFrame,
    cut: str | pd.Timestamp,
    key_cols: list[str],
    value: str = "boarding",
    mean_col: str = "mean",
    n_col: str = "n",
    recency: bool = False,
) -> pd.DataFrame:
    """cut **이전까지만** 써서 키별 평균·표본수를 고정한다. cut 은 포함(<=)이다.

    recency=True 면 cut 으로부터 182일 이내 관측에 가중치 2를 준 가중평균을 쓴다.
    n 은 옵션과 무관하게 관측 수 그대로다 (feature 계약 유지).
    """
    cut = pd.Timestamp(cut)
    keep = df["date"] <= cut
    part = df.loc[keep, key_cols + ["date", value]].copy()
    if not recency:
        out = (
            part.groupby(key_cols, observed=True)[value]
            .agg(["mean", "count"])
            .rename(columns={"mean": mean_col, "count": n_col})
            .reset_index()
        )
    else:
        start = cut - pd.Timedelta(days=RECENCY_WEIGHT_DAYS)
        recent = part["date"].to_numpy(dtype="datetime64[ns]") >= np.datetime64(start)
        work = part[key_cols].copy()
        work["_v"] = part[value].to_numpy(dtype="float64")
        work["_w"] = np.where(recent, RECENCY_WEIGHT, 1.0)
        work["_wx"] = work["_v"] * work["_w"]
        out = (
            work.groupby(key_cols, observed=True)
            .agg(_wsum=("_wx", "sum"), _wn=("_w", "sum"), _count=("_v", "count"))
            .reset_index()
        )
        out[mean_col] = (out["_wsum"] / out["_wn"]).astype("float32")
        out[n_col] = out["_count"].astype("int32")
        return out[key_cols + [mean_col, n_col]]
    out[n_col] = out[n_col].astype("int32")
    out[mean_col] = out[mean_col].astype("float32")
    return out


def lookup(profile: pd.DataFrame, df: pd.DataFrame, key_cols: list[str], col: str) -> np.ndarray:
    """고정 프로필을 df 각 행에 붙인다. 없는 키는 NaN. (정수 키 reindex — 벡터화)"""
    if col not in profile.columns:
        raise KeyError(f"프로필에 컬럼이 없다: {col}")
    table = pd.Series(profile[col].to_numpy(), index=_keys(profile, key_cols))
    if table.index.has_duplicates:
        raise ValueError(f"프로필 키가 겹친다: {key_cols}")
    return table.reindex(_keys(df, key_cols)).to_numpy()


def frozen_profile_frame(
    df_fit: pd.DataFrame,
    df_target: pd.DataFrame,
    cut: str | pd.Timestamp,
    value: str = "boarding",
    recency: bool = False,
) -> pd.DataFrame:
    """df_fit 의 cut 이전 데이터로 프로필을 고정해 df_target 에 붙인다 (df_target.boarding 은 보지 않는다)."""
    sh = frozen_profile(df_fit, cut, KEY_SH, value=value,
                        mean_col=PROFILE_COLS_SH[0], n_col=PROFILE_COLS_SH[1], recency=recency)
    mo = frozen_profile(df_fit, cut, KEY_SH_MONTH, value=value,
                        mean_col=PROFILE_COLS_MONTH[0], n_col=PROFILE_COLS_MONTH[1], recency=recency)
    wd = frozen_profile(df_fit, cut, KEY_SH_WEEKDAY, value=value,
                        mean_col=PROFILE_COLS_WEEKDAY[0], n_col=PROFILE_COLS_WEEKDAY[1], recency=recency)
    out = pd.DataFrame(index=df_target.index)
    for prof, key_cols, cols in ((sh, KEY_SH, PROFILE_COLS_SH),
                                 (mo, KEY_SH_MONTH, PROFILE_COLS_MONTH),
                                 (wd, KEY_SH_WEEKDAY, PROFILE_COLS_WEEKDAY)):
        out[cols[0]] = lookup(prof, df_target, key_cols, cols[0]).astype("float32")
        # 프로필에 없는 키(과거에 한 번도 없던 조합)는 표본수 0 으로 둔다
        out[cols[1]] = np.nan_to_num(lookup(prof, df_target, key_cols, cols[1]), nan=0.0).astype("int32")
    return out


# ───────────────────────── 편의 ─────────────────────────
def horizon_bucket(days: np.ndarray) -> np.ndarray:
    """cut 으로부터 경과일 → 구간 라벨('1-31' 등). 범위 밖은 'other'."""
    days = np.asarray(days)
    out = np.full(days.shape, "other", dtype=object)
    for lo, hi in HORIZON_BUCKETS:
        out[(days >= lo) & (days <= hi)] = f"{lo}-{hi}"
    return out


def save_profile_npz(
    path: str | Path,
    df: pd.DataFrame,
    value: str = "boarding",
    recency: bool = False,
) -> dict[str, int]:
    """서빙/평가용 프로필을 npz 로 저장한다.

    배열: station_code, hour_code, month, mean, count                 (월별 키 — A)
          station_code_sh, hour_code_sh, mean_sh, count_sh            (A0, 월 무시)
          station_code_wd, hour_code_wd, weekday_wd, mean_wd, count_wd (B, 요일별)
    recency=True 면 마지막 관측일로부터 182일 이내 관측에 가중치 2를 준 평균을 쓴다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = add_month(df)
    last = df["date"].max()
    sh = frozen_profile(df, last, KEY_SH, value=value, mean_col="mean", n_col="count", recency=recency)
    mo = frozen_profile(df, last, KEY_SH_MONTH, value=value, mean_col="mean", n_col="count",
                        recency=recency)
    wd = frozen_profile(df, last, KEY_SH_WEEKDAY, value=value, mean_col="mean", n_col="count",
                        recency=recency)
    np.savez_compressed(
        path,
        station_code=mo["station_code"].to_numpy(dtype=np.int32),
        hour_code=mo["hour_code"].to_numpy(dtype=np.int16),
        month=mo["month"].to_numpy(dtype=np.int8),
        mean=mo["mean"].to_numpy(dtype=np.float32),
        count=mo["count"].to_numpy(dtype=np.int32),
        station_code_sh=sh["station_code"].to_numpy(dtype=np.int32),
        hour_code_sh=sh["hour_code"].to_numpy(dtype=np.int16),
        mean_sh=sh["mean"].to_numpy(dtype=np.float32),
        count_sh=sh["count"].to_numpy(dtype=np.int32),
        station_code_wd=wd["station_code"].to_numpy(dtype=np.int32),
        hour_code_wd=wd["hour_code"].to_numpy(dtype=np.int16),
        weekday_wd=wd["weekday"].to_numpy(dtype=np.int8),
        mean_wd=wd["mean"].to_numpy(dtype=np.float32),
        count_wd=wd["count"].to_numpy(dtype=np.int32),
        date_min=np.asarray(df["date"].min().strftime("%Y-%m-%d")),
        date_max=np.asarray(df["date"].max().strftime("%Y-%m-%d")),
        rows=np.asarray(len(df), dtype=np.int64),
    )
    return {"keys_month": int(len(mo)), "keys_sh": int(len(sh)),
            "keys_weekday": int(len(wd)), "rows": int(len(df))}


def _self_check() -> None:  # pragma: no cover - 자체 점검 진입점
    tr = load_processed(DATA / "train.csv.gz", columns=["date", "station_code", "line_code",
                                                        "hour_code", "weekday", "is_weekend",
                                                        "is_holiday", "boarding"])
    prof = asof_profile_frame(tr)
    print(tr.head(3).to_string(index=False))
    print(prof.head(3).to_string(index=False))
    print("NaN 비율", prof.isna().mean().round(4).to_dict())


# ───────────────────────── 공휴일 달력 ─────────────────────────
# 음력 설날·추석. 패키지 없이 표로 고정한다(설날, 추석 순서 고정).
# 2024~2026 설날은 data/processed 의 is_holiday 와 대조해 확인했다.
# 2026 추석·2027 은 아직 공표 일정과 대조하지 못했으므로 미확인 상태다.
LUNAR_HOLIDAYS: dict[int, list[str]] = {
    2024: ["2024-02-10", "2024-09-17"],
    2025: ["2025-01-29", "2025-10-06"],
    2026: ["2026-02-17", "2026-09-25"],
    2027: ["2027-02-07", "2027-09-15"],
}

# 양력 고정 공휴일 (월, 일). 부처님오신날·선거일·임시공휴일은 표에 없다.
FIXED_HOLIDAYS: tuple[tuple[int, int], ...] = (
    (1, 1), (3, 1), (5, 5), (6, 6), (8, 15), (10, 3), (10, 9), (12, 25),
)
SPECIAL_DAYS: tuple[tuple[int, int], ...] = ((5, 5), (12, 25))  # 어린이날·성탄
VACATION_MONTHS: tuple[int, ...] = (1, 2, 7, 8)
# 확인된 수능일만 넣는다(시행일은 매년 교육부 발표). 이 데이터 구간은 두 날짜로 덮인다.
EXAM_DAYS: tuple[str, ...] = ("2024-11-14", "2025-11-13")
HOLIDAY_WINDOW = 3
HOLIDAY_PAD_DAYS = 14
_ORDINAL_EPOCH = pd.Timestamp("1970-01-01").toordinal()

CALENDAR_FEATURE_COLS: list[str] = [
    "cal_run_len", "cal_pos_in_run", "cal_seollal", "cal_chuseok",
    "cal_substitute", "cal_vacation", "cal_exam", "cal_fixed",
]


def _holiday_table(years: list[int]) -> tuple[set[pd.Timestamp], set[pd.Timestamp]]:
    """연도 목록 → (공휴일 집합, 대체공휴일 집합).

    연휴는 설날·추석 전날과 다음날까지 3일로 잡는다. 대체공휴일은 "연휴 덩어리에 주말이 끼면
    그 덩어리 다음의 첫 비공휴일 평일" 규칙으로 근사한다. 실제 지정(겹침형 대체공휴일,
    임시공휴일)과 다른 날이 생길 수 있으므로 공휴일 판정의 정본은 데이터의 is_holiday 다.
    """
    days: set[pd.Timestamp] = set()
    for year in years:
        for month, day in FIXED_HOLIDAYS:
            days.add(pd.Timestamp(year=year, month=month, day=day))
        for text in LUNAR_HOLIDAYS.get(year, []):
            base = pd.Timestamp(text)
            for offset in (-1, 0, 1):
                days.add(base + pd.Timedelta(days=offset))

    substitutes: set[pd.Timestamp] = set()
    ordered = sorted(days)
    blocks: list[list[pd.Timestamp]] = []
    for stamp in ordered:
        if blocks and (stamp - blocks[-1][-1]).days == 1:
            blocks[-1].append(stamp)
        else:
            blocks.append([stamp])
    for block in blocks:
        if not any(stamp.weekday() >= 5 for stamp in block):
            continue
        candidate = block[-1] + pd.Timedelta(days=1)
        while candidate in days or candidate in substitutes or candidate.weekday() >= 5:
            candidate += pd.Timedelta(days=1)
        substitutes.add(candidate)
    days |= substitutes
    return days, substitutes


def holiday_context(dates: pd.Series) -> pd.DataFrame:
    """날짜 열 → 공휴일 맥락 피처(모두 int8).

    열
        run_len        : 그 날이 속한 연휴 길이(공휴일이 아니면 0)
        pos_in_run     : 가장 가까운 공휴일까지의 일수(-3..+3, 공휴일 당일 0, 3일 넘으면 ±3)
        seollal_window : 설날 ±3일
        chuseok_window : 추석 ±3일
        substitute     : 규칙으로 계산한 대체공휴일
        vacation       : 방학월(1·2·7·8월)
        exam_day       : 수능일
        fixed_day      : 어린이날·성탄절
        ctx_is_holiday : 위 표 기준 공휴일 여부(진단용, 데이터 is_holiday 와 다를 수 있다)
    """
    series = pd.Series(dates)
    original_index = series.index
    series = series.reset_index(drop=True)
    stamps = pd.to_datetime(series).dt.normalize()
    if stamps.empty:
        return pd.DataFrame(columns=[*CALENDAR_FEATURE_COLS, "ctx_is_holiday"], index=original_index)
    years = sorted({int(y) for y in stamps.dt.year.unique()})
    span = [years[0] - 1, *years, years[-1] + 1]
    holidays, substitutes = _holiday_table(span)

    ordinals = stamps.to_numpy(dtype="datetime64[D]").astype("int64") + _ORDINAL_EPOCH
    h_ord = np.sort(np.array([stamp.toordinal() for stamp in holidays], dtype="int64"))
    idx = np.searchsorted(h_ord, ordinals)
    prev = h_ord[np.clip(idx - 1, 0, len(h_ord) - 1)]
    nxt = h_ord[np.clip(idx, 0, len(h_ord) - 1)]
    dist = np.where(np.abs(ordinals - prev) <= np.abs(nxt - ordinals),
                    ordinals - prev, ordinals - nxt)
    pos_in_run = np.clip(dist, -HOLIDAY_WINDOW, HOLIDAY_WINDOW).astype("int8")

    # 연휴 길이는 입력 범위 밖까지 봐야 정확하다(한 날짜만 넘겨도 같은 값이 나오도록 앞뒤를 채운다)
    pad = pd.Timedelta(days=HOLIDAY_PAD_DAYS)
    day_index = pd.date_range(stamps.min() - pad, stamps.max() + pad, freq="D")
    flag_series = pd.Series(day_index.to_numpy(), index=day_index).isin(holidays).astype(bool)
    group_id = (flag_series != flag_series.shift(1)).cumsum()
    block_len = flag_series.groupby(group_id).transform("sum")
    run_len = pd.Series(0, index=day_index, dtype="int16")
    run_len[flag_series.to_numpy()] = block_len[flag_series.to_numpy()].to_numpy(dtype="int16")
    run_map = run_len.reindex(stamps).to_numpy()

    seollal = np.zeros(len(stamps), dtype="int8")
    chuseok = np.zeros(len(stamps), dtype="int8")
    for year in span:
        pair = LUNAR_HOLIDAYS.get(year)
        if not pair or len(pair) < 2:
            continue
        seollal[np.abs(ordinals - pd.Timestamp(pair[0]).toordinal()) <= HOLIDAY_WINDOW] = 1
        chuseok[np.abs(ordinals - pd.Timestamp(pair[1]).toordinal()) <= HOLIDAY_WINDOW] = 1

    substitute_flag = stamps.isin(substitutes).to_numpy().astype("int8")
    exam = stamps.isin([pd.Timestamp(d) for d in EXAM_DAYS]).to_numpy().astype("int8")
    fixed = np.array(
        [(m, d) in SPECIAL_DAYS for m, d in zip(stamps.dt.month, stamps.dt.day, strict=True)],
        dtype="int8",
    )
    vacation = stamps.dt.month.isin(VACATION_MONTHS).to_numpy().astype("int8")
    ctx_holiday = stamps.isin(holidays).to_numpy().astype("int8")

    out = pd.DataFrame({
        "run_len": np.asarray(run_map, dtype="int16"),
        "pos_in_run": pos_in_run,
        "seollal_window": seollal,
        "chuseok_window": chuseok,
        "substitute": substitute_flag,
        "vacation": vacation,
        "exam_day": exam,
        "fixed_day": fixed,
        "ctx_is_holiday": ctx_holiday,
    })
    out.index = original_index
    return out


def calendar_feature_frame(dates: pd.Series) -> pd.DataFrame:
    """holiday_context 중 모델 피처로 쓸 열만 cal_ 접두어로 돌려준다."""
    ctx = holiday_context(dates)
    out = pd.DataFrame(index=ctx.index)
    mapping = {
        "cal_run_len": "run_len",
        "cal_pos_in_run": "pos_in_run",
        "cal_seollal": "seollal_window",
        "cal_chuseok": "chuseok_window",
        "cal_substitute": "substitute",
        "cal_vacation": "vacation",
        "cal_exam": "exam_day",
        "cal_fixed": "fixed_day",
    }
    for target, source in mapping.items():
        out[target] = ctx[source].to_numpy(dtype="int8")
    return out[CALENDAR_FEATURE_COLS]


if __name__ == "__main__":  # 간단 자체 점검
    _self_check()
