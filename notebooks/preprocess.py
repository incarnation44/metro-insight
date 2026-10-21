"""데이터 전처리 — 1~8호선 승차 인원(2024-01-01 ~ 2026-06-30)을 모델 입력 형태로 바꿔 저장한다.

파이프라인
    1) 원본 3개 로드(cp949)   2) 승객유형 합산   3) 이상 데이터 제외   4) 와이드→롱, 승차만
    5) 역 이름 통일   6) 요일·주말·공휴일   7) 날씨 붙이기   8) 과거값   9) 인코딩·시간순 분할·저장

실행 (프로젝트 루트에서)
    python notebooks/fetch_weather.py        # 먼저: data/processed/weather_시간대별.csv 를 만든다
    python notebooks/preprocess.py

입력 : 원본 CSV 폴더 (METRO_DATA_DIR 로 위치 지정), data/processed/weather_시간대별.csv, holidays 패키지
결과 : data/processed/{train,valid,test}.csv.gz
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.unicode.east_asian_width", True)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# 원본 CSV 기본 위치는 저장소 안 data/raw/subway. 밖에서 가져오려면 METRO_DATA_DIR 로 지정
DATA_DIR = Path(os.environ.get("METRO_DATA_DIR", ROOT / "data" / "raw" / "subway"))
ENCODING = "cp949"

# 원본 20개 시간대 열 → 짧은 라벨 (순서 1:1)
# 핵심: '06시이전'=04:00~05:59(2시간, 원본이 묶어서 준다), '24시이후'=24:00~익일 00:59(1시간).
# 업무일자가 04:00 에 시작하고 01~04시는 운행 중단이라 경계를 그대로 쓴다
HOUR_LABELS = [
    "04_06",
    "06_07", "07_08", "08_09", "09_10", "10_11", "11_12", "12_13",
    "13_14", "14_15", "15_16", "16_17", "17_18", "18_19", "19_20",
    "20_21", "21_22", "22_23", "23_24",
    "24_01",
]
ID_COLUMNS = ["date", "line", "station_num", "station", "승하차구분"]
KEY = ["date", "line", "station_num", "station", "hour"]

# 기간 제외 없음
EXCLUDE_DATE_RANGES: list[tuple[str, str]] = []
# 7호선 까치울은 관측 기간이 거의 없다 → 제외
EXCLUDE_STATIONS = [("7호선", 2753)]

TARGET_BOARD = "boarding"      # 예측 대상(y). 하차 인원은 원본에만 둔다
LAG_DAYS = 7                  # 과거값은 직전 7일이 모두 있어야 만든다
TRAIN_END_RATIO = 0.70
VALID_END_RATIO = 0.85

COLUMNS = [
    "date", "line", "station_num", "station", "hour", TARGET_BOARD,
    "weekday", "is_weekend", "is_holiday", "temperature", "rainfall", "snowfall",
    "lag_1d", "lag_7d", "mean_prev7d", "station_code", "line_code", "hour_code",
]


def header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ───────────────────────── 1~3. 읽기·합산·제외 ─────────────────────────
def normalize_wide(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """원본 한 파일분을 (식별자 5개 + 시간대 20개) 모양으로 맞춘다. 2026 파일은 승객유형을 합산한다."""
    df = df.rename(columns={"수송일자": "date", "호선": "line", "호선명": "line",
                            "역번호": "station_num", "역명": "station"})
    df = df[df["date"].notna()].copy()                       # 2025 파일 끝의 빈 줄 제거
    df = df.rename(columns=dict(zip(df.columns[-20:], HOUR_LABELS)))   # 이름이 파일마다 달라서 위치(마지막 20열)로 맞춘다
    df["date"] = pd.to_datetime(df["date"])
    df["station_num"] = df["station_num"].astype(int)
    df[HOUR_LABELS] = df[HOUR_LABELS].apply(pd.to_numeric)
    n_types = df["승객유형"].nunique() if "승객유형" in df.columns else 0
    if n_types:
        # 국내 6종 화이트리스트 — 외국어 6종(영어/일어/중국어 × 일반/어린이)은 제외
        kr_types = ["일반", "어린이", "중고생", "청소년", "우대권", "직원"]
        foreign = ~df["승객유형"].isin(kr_types)
        if foreign.any():
            print(f"  외국어 승객유형 제외: {sorted(df.loc[foreign, '승객유형'].unique())} = {int(foreign.sum()):,}행")
            df = df[~foreign].copy()
        df = df.groupby(ID_COLUMNS, as_index=False, sort=False)[HOUR_LABELS].sum()
    print(f"  {name[:60]:<62} {df['date'].min().date()} ~ {df['date'].max().date()}  "
          f"{df['date'].nunique()}일 {len(df):>8,}행" + (f"  (승객유형 {n_types}종 합산)" if n_types else ""))
    return df[ID_COLUMNS + HOUR_LABELS]


def read_one(path: Path) -> pd.DataFrame:
    return normalize_wide(pd.read_csv(path, encoding=ENCODING, low_memory=False), path.name)


def load_wide() -> pd.DataFrame:
    header("1~2) 원본 로드 · 승객유형 합산")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"데이터 폴더가 없다: {DATA_DIR} (METRO_DATA_DIR 로 지정)")
    files = [p for p in sorted(DATA_DIR.glob("*.csv")) if "보기용" not in p.name and "_sample" not in p.name]   # 보기용 복사본·샘플은 읽지 않는다
    if not files:
        raise FileNotFoundError(f"CSV 가 없다: {DATA_DIR}")
    wide = pd.concat([read_one(p) for p in files], ignore_index=True)
    dup = int(wide.duplicated(["date", "line", "station_num", "승하차구분"]).sum())
    if dup:
        raise ValueError(f"같은 (날짜, 역, 승하차)가 {dup}줄 겹친다. 파일 기간이 겹치지 않는지 확인한다.")
    print(f"합계 : {len(wide):,}행, {wide['date'].nunique()}일, (호선·역번호) {wide.groupby(['line', 'station_num']).ngroups}개")
    return wide


def drop_bad(wide: pd.DataFrame) -> pd.DataFrame:
    header("3) 이상 데이터 제외")
    mask = pd.Series(False, index=wide.index)
    for start, end in EXCLUDE_DATE_RANGES:
        hit = wide["date"].between(start, end)
        print(f"기간 제외 {start} ~ {end} : {int(hit.sum()):,}행 ({wide.loc[hit, 'date'].nunique()}일)")
        mask |= hit
    for line, no in EXCLUDE_STATIONS:
        hit = (wide["line"] == line) & (wide["station_num"] == no)
        print(f"역 제외 {line} {no} : {int(hit.sum()):,}행")
        mask |= hit
    return wide[~mask].reset_index(drop=True)


# ───────────────────────── 4~5. 롱 변환 · 역 이름 ─────────────────────────
def unify_station_names(wide: pd.DataFrame) -> pd.DataFrame:
    """핵심: 역 키는 (line, station_num). 이름이 바뀐 역(당고개→불암산 등)은 가장 최근 이름 하나로 통일한다."""
    header("5) 역 이름 통일")
    latest = (
        wide.sort_values("date").drop_duplicates(["line", "station_num"], keep="last")
        [["line", "station_num", "station"]].rename(columns={"station": "_latest"})
    )
    changed = wide[["line", "station_num", "station"]].drop_duplicates().merge(latest, on=["line", "station_num"])
    changed = changed[changed["station"] != changed["_latest"]]
    for _, r in changed.iterrows():
        print(f"  {r['line']} {r['station_num']}: {r['station']} → {r['_latest']}")
    out = wide.merge(latest, on=["line", "station_num"], how="left")
    out["station"] = out.pop("_latest")
    return out


def to_long(wide: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """시간대 20열을 행으로 내리고 승차만 한 줄씩 만든다. 하차 줄은 원본에만 남기고 여기서 뺀다."""
    header("4) 와이드 → 롱 · 승차만")
    ride = wide[wide["승하차구분"] == "승차"]
    alight_rows = int((wide["승하차구분"] == "하차").sum())
    total = int(ride[HOUR_LABELS].to_numpy().sum())
    long = ride.melt(id_vars=["date", "line", "station_num", "station"], value_vars=HOUR_LABELS,
                     var_name="hour", value_name=TARGET_BOARD)
    long[TARGET_BOARD] = long[TARGET_BOARD].astype("int32")
    print(f"승차 {len(ride):,}줄 → 시간대 행 {len(long):,}행 (하차 {alight_rows:,}줄은 통합본에서 뺌, 승차 합계 {total:,})")
    return long, total


# ───────────────────────── 6~8. 달력·날씨·과거값 ─────────────────────────
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """요일·주말·공휴일. 날짜별로 한 번만 계산해서 합친다 (줄마다 하면 느리다)."""
    header("6) 요일 · 주말 · 공휴일")
    try:
        import holidays
    except ImportError as exc:
        raise ImportError("holidays 패키지가 필요하다: pip install holidays") from exc
    days = pd.DataFrame({"date": np.sort(df["date"].unique())})
    days["weekday"] = days["date"].dt.dayofweek                       # 0=월 … 6=일
    days["is_weekend"] = (days["weekday"] >= 5).astype(int)
    years = range(days["date"].dt.year.min(), days["date"].dt.year.max() + 1)
    # 핵심: 대체·임시공휴일·선거일도 패키지 목록에 들어 있다. 주말과 겹쳐도 is_holiday=1 (is_weekend 와 둘 다)
    hol = pd.DatetimeIndex(list(holidays.country_holidays("KR", years=years).keys()))
    days["is_holiday"] = days["date"].isin(hol).astype(int)
    print(f"공휴일 {int(days['is_holiday'].sum())}일 (남은 {len(days)}일 중), 주말과 겹친 공휴일 {int(((days.is_holiday == 1) & (days.is_weekend == 1)).sum())}일")
    return df.merge(days, on="date", how="left")


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """날씨(기온·비·눈)를 (date, hour)로 붙인다. 날씨는 서울 한 곳 값이라 모든 역에 같다."""
    header("7) 날씨 붙이기")
    path = PROCESSED / "weather_시간대별.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} 이 없다. 먼저 python notebooks/fetch_weather.py 를 실행한다.")
    weather = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
    before = len(df)
    out = df.merge(weather, on=["date", "hour"], how="left")
    assert len(out) == before, "날씨를 붙였더니 줄 수가 달라졌다"
    nan = out[["temperature", "rainfall", "snowfall"]].isna().sum()
    if nan.any():
        raise ValueError(f"날씨가 안 붙은 줄이 있다:\n{nan}")
    print(f"붙임 : temperature · rainfall · snowfall  (NaN 0, 줄 수 {len(out):,} 그대로)")
    return out


def add_lags(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """과거값: 어제·지난주 같은 시간대, 직전 7일 평균. 직전 7일이 모두 없는 줄은 뺀다.

    핵심: shift 가 아니라 '빠진 날을 NaN 으로 채운 전체 달력'에서 계산한다.
          빠진 날이 있으면 그냥 shift 하면 엉뚱한 날의 값이 들어온다.
    """
    header("8) 과거값 (lag_1d · lag_7d · mean_prev7d)")
    df = df.copy()
    df["_stn"] = df["line"] + "_" + df["station_num"].astype(str)
    stations = sorted(df["_stn"].unique())
    calendar = pd.date_range(df["date"].min(), df["date"].max())
    index = pd.MultiIndex.from_product([stations, HOUR_LABELS, calendar], names=["_stn", "hour", "date"])
    series = df.set_index(["_stn", "hour", "date"])[TARGET_BOARD]
    grid = series.reindex(index).astype("float32")                      # 없는 날은 NaN
    by = grid.groupby(level=[0, 1], sort=False)
    lag1 = by.shift(1)
    lag7 = by.shift(LAG_DAYS)
    mean7 = by.transform(lambda s: s.shift(1).rolling(LAG_DAYS, min_periods=LAG_DAYS).mean())
    df["lag_1d"] = lag1.reindex(series.index).to_numpy()
    df["lag_7d"] = lag7.reindex(series.index).to_numpy()
    df["mean_prev7d"] = mean7.reindex(series.index).to_numpy()

    bad = df[["lag_1d", "lag_7d", "mean_prev7d"]].isna().any(axis=1)
    pairs = df.loc[bad, ["_stn", "date"]].drop_duplicates().shape[0]
    print(f"직전 {LAG_DAYS}일이 모자라 뺀 (역,날짜) {pairs:,}쌍 = {int(bad.sum()):,}줄")
    df = df[~bad].copy()
    df[["lag_1d", "lag_7d"]] = df[["lag_1d", "lag_7d"]].astype("int32")
    df["mean_prev7d"] = df["mean_prev7d"].round(1)
    return df.drop(columns="_stn"), pairs


# ───────────────────────── 9. 인코딩 · 분할 · 저장 ─────────────────────────
def split_and_save(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """날짜 순서로 70/15/15 분할해 저장한다. 코드맵은 train 에서만 뽑는다. 셔플하지 않는다 (섞으면 미래 누수)."""
    header("9) 인코딩 · 시간순 분할 · 저장")
    out = df.sort_values(["date", "line", "station_num", "hour"]).reset_index(drop=True)
    dates = pd.Series(out["date"].unique()).sort_values()
    train_end = dates.iloc[int(len(dates) * TRAIN_END_RATIO) - 1]
    valid_end = dates.iloc[int(len(dates) * VALID_END_RATIO) - 1]
    train = out[out["date"] <= train_end].copy()
    valid = out[(out["date"] > train_end) & (out["date"] <= valid_end)].copy()
    test = out[out["date"] > valid_end].copy()

    out_key = lambda f: f["line"] + "_" + f["station_num"].astype(str)   # noqa: E731
    station_codes = sorted(out_key(train).unique())
    line_codes = sorted(train["line"].unique())
    for frame in (train, valid, test):
        frame["station_code"] = pd.Categorical(out_key(frame), categories=station_codes).codes
        frame["line_code"] = pd.Categorical(frame["line"], categories=line_codes).codes
        frame["hour_code"] = frame["hour"].map({label: i for i, label in enumerate(HOUR_LABELS)})

    total = len(out)
    print(f"{'split':<7}{'시작일':<13}{'종료일':<13}{'일수':>5}{'행 수':>12}{'비율':>9}")
    for name, frame in (("train", train), ("valid", valid), ("test", test)):
        print(f"{name:<7}{str(frame['date'].min().date()):<13}{str(frame['date'].max().date()):<13}"
              f"{frame['date'].nunique():>5}{len(frame):>12,}{len(frame) / total:>9.4f}")

    overlap = set(train["date"]) & set(valid["date"]) | set(valid["date"]) & set(test["date"])
    unseen = sorted(set(out_key(valid)) | set(out_key(test)) - set(out_key(train)))
    print(f"누수 점검 : 날짜 겹침 {len(overlap)}일(0 이어야 함) · valid/test 에만 있는 역 {len([s for s in unseen if s not in set(out_key(train))])}개")

    for name, frame in (("train", train), ("valid", valid), ("test", test)):
        path = PROCESSED / f"{name}.csv.gz"
        frame[COLUMNS].to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
        print(f"저장 : {path.name:<14} {len(frame):>10,}행  {path.stat().st_size / 1024 / 1024:6.1f} MB")
    return {"train": train, "valid": valid, "test": test}


def verify(board_total: int, agg: pd.DataFrame, splits: dict[str, pd.DataFrame], pairs_dropped: int) -> None:
    header("검증")
    kept = pd.concat(splits.values())
    print(f"줄 수 : 합계 {len(kept):,} = train {len(splits['train']):,} + valid {len(splits['valid']):,} + test {len(splits['test']):,}")
    ride_ok = int(agg[TARGET_BOARD].sum()) == board_total
    print(f"합계 보존 : 승차 {'일치' if ride_ok else '불일치'} ({board_total:,})")
    print(f"중복 (날짜·역·시간대) : {int(kept.duplicated(KEY).sum())}개")
    print(f"역 {kept.groupby(['line', 'station_num']).ngroups}개 · 시간대 {kept['hour'].nunique()}종 · 날짜 {kept['date'].nunique()}일")
    print(f"NaN : {int(kept[COLUMNS].isna().sum().sum())}개")

    train = splits["train"]
    print("\ntrain.shape:", train.shape)
    print(train[COLUMNS].head(3).to_string(index=False))
    train[COLUMNS].info()

    print("\n상식 점검 (평균 승차 인원)")
    print("  요일별(0=월):", kept.groupby("weekday")[TARGET_BOARD].mean().round(0).astype(int).to_dict())
    for col, label in (("is_holiday", "공휴일"), ("rainfall", "비"), ("snowfall", "눈")):
        g = kept.groupby(col)[TARGET_BOARD].mean().round(1)
        print(f"  {label} 아님 {g.get(0, float('nan')):>7.1f} / {label} {g.get(1, float('nan')):>7.1f}")


def main() -> None:
    print(f"전처리 시작 — 1~8호선 승차 인원 · Python {sys.version.split()[0]} · pandas {pd.__version__}")
    wide = drop_bad(load_wide())
    wide = unify_station_names(wide)
    agg, total = to_long(wide)
    del wide
    agg = add_calendar(agg)
    agg = add_weather(agg)
    full = agg
    agg, pairs = add_lags(agg)
    splits = split_and_save(agg)
    verify(total, full, splits, pairs)
    print("\n완료 — 라벨: 승차인원(예측 대상)만. 하차 인원은 원본에만 있고 통합본에 없다")


if __name__ == "__main__":
    main()
