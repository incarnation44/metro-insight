"""날씨 수집 — 기상청 API허브 ASOS 시간자료(서울 108)를 받아 지하철 시간대에 맞는 날씨표를 만든다.

결과 변수는 세 가지다: 기온(temperature), 비 왔나(is_rain 0/1), 눈 왔나(is_snow 0/1).

실행 (프로젝트 루트에서)
    $env:KMA_API_KEY = "<인증키>"
    python notebooks/fetch_weather.py                  # 받기 + 변환 + 검증 (이미 받은 달은 건너뜀)
    python notebooks/fetch_weather.py --force          # 원본을 전부 다시 받기
    python notebooks/fetch_weather.py --report 경로    # 검증 보고서를 파일로 저장 (기본은 화면 출력)

입력 : 기상청 API허브 kma_sfctm3.php (지상관측 시간자료, 기간 조회, 한 번에 최대 31일)
결과 : data/raw/weather/kma_sfctm3_stn108_YYYYMM.txt   받은 그대로 (월별, 수정 금지)
       data/processed/weather.csv                       시간별: date, hour, temperature, is_rain, is_snow
       data/processed/weather_시간대별.csv               지하철용: 수송일자, 시간대, temperature, is_rain, is_snow

인증키는 코드·파일에 적지 않고 환경변수 KMA_API_KEY 에서만 읽는다. 원본이 모두 있으면 키 없이도 변환·검증은 돌아간다.
"""

from __future__ import annotations

import calendar
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ───────────────────────── 설정 ─────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "weather"
CSV_PATH = ROOT / "data" / "processed" / "weather.csv"
BIN_CSV_PATH = ROOT / "data" / "processed" / "weather_시간대별.csv"

API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
STN = 108                                   # 서울
START = pd.Timestamp("2024-01-01 00:00")    # 지하철 데이터 시작
LAST_DAY = pd.Timestamp("2026-06-30")       # 지하철 데이터 마지막 날(포함)
ONE_HOUR = pd.Timedelta(hours=1)

# 핵심: 지하철 시간대 '24_28' 은 익일 0~4시다 → 마지막 날(06-30)에 07-01 0~3시 날씨가 필요하다.
#       또 h시의 비·눈은 'h+1시' 관측으로 계산하므로 07-01 04시 관측까지 받는다.
EXTRA_HOURS = 4
END_TS = LAST_DAY + pd.Timedelta(days=1, hours=EXTRA_HOURS)   # 2026-07-01 04:00

# preprocess.py 의 시간대 라벨과 같다. '04_06'=04~06시(2시간), '24_28'=익일 00~04시(4시간)
BIN_LABELS = ["04_06"] + [f"{h:02d}_{h + 1:02d}" for h in range(6, 24)] + ["24_28"]
BIN_HOURS = {lb: 1 for lb in BIN_LABELS}
BIN_HOURS["04_06"], BIN_HOURS["24_28"] = 2, 4

# 시간자료 46개 열 (기상청 안내문 순서)
COLS = [
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR",
    "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3", "SD_DAY",
    "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN", "CT", "CT_TOP", "CT_MID",
    "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01", "TE_02", "TE_03",
    "ST_SEA", "WH", "BF", "IR", "IX",
]
NUM_COLS = ["STN", "TA", "RN", "RN_DAY", "SD_DAY"]   # 실제로 쓰는 열만 숫자로 바꾼다
TOL = 0.05              # 강수 비교 허용 오차(mm). 원자료가 0.1 단위라 사실상 '정확히 같다'는 뜻
TA_MISSING_BELOW = -50  # 핵심: 기온 결측은 -99.9 같은 값. -9.0 은 실제 영하 9도라 결측으로 보면 안 된다
TEMP_FILL_LIMIT = 12    # 기온 공백은 12시간까지만 직선으로 메운다

HOURLY_COLS = ["date", "hour", "temperature", "is_rain", "is_snow"]
BIN_COLS = ["수송일자", "시간대", "temperature", "is_rain", "is_snow"]


# ───────────────────────── 1. 수집 ─────────────────────────
def month_requests() -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """(시작시각, 끝시각, 기대 행 수) 목록. API 가 31일까지만 주므로 월 단위로 나눈다."""
    out = []
    y, m = START.year, START.month
    while (y, m) <= (LAST_DAY.year, LAST_DAY.month):
        tm1 = pd.Timestamp(year=y, month=m, day=1, hour=0)
        last = calendar.monthrange(y, m)[1]
        tm2 = pd.Timestamp(year=y, month=m, day=last, hour=23)
        if (y, m) == (LAST_DAY.year, LAST_DAY.month):
            tm2 = END_TS
        out.append((tm1, tm2, int((tm2 - tm1) / ONE_HOUR) + 1))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def raw_path(tm1: pd.Timestamp) -> Path:
    return RAW_DIR / f"kma_sfctm3_stn{STN}_{tm1:%Y%m}.txt"


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp949", errors="replace")


def data_lines(text: str) -> list[str]:
    """'#' 로 시작하는 설명 줄과 빈 줄을 뺀 데이터 줄만."""
    return [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def validate_text(text: str, tm1: pd.Timestamp, tm2: pd.Timestamp, expected_rows: int) -> tuple[bool, int, str]:
    """원본 한 달치가 구조상 정상이고 요청 구간을 끝까지 덮는지 검사한다. (정상여부, 행수, 메모)

    행이 모자란 것은 실패로 보지 않는다 (원자료 공백일 수 있다). 대신 첫·끝 시각은 요청과 같아야 한다.
    끝이 짧으면 응답이 잘렸을 수 있으므로 다시 받는다.
    """
    if text.lstrip().startswith("{"):
        return False, 0, "JSON 오류 응답(인증·권한·한도 문제 가능)"
    if "#7777END" not in text:
        return False, 0, "끝 표시(#7777END) 없음 - 응답이 잘렸을 수 있음"
    lines = data_lines(text)
    if not lines:
        return False, 0, "데이터 행 없음"
    bad = [ln for ln in lines if len(ln.split()) != len(COLS)]
    if bad:
        return False, len(lines), f"열 개수가 {len(COLS)}개가 아닌 줄 {len(bad)}개"
    first, last = lines[0].split()[0], lines[-1].split()[0]
    if first != f"{tm1:%Y%m%d%H%M}" or last != f"{tm2:%Y%m%d%H%M}":
        return False, len(lines), f"구간 불일치 ({first} ~ {last}, 요청 {tm1:%Y%m%d%H%M} ~ {tm2:%Y%m%d%H%M})"
    if len(lines) > expected_rows:
        return False, len(lines), f"행 수 {len(lines)} > 기대 {expected_rows}"
    if len(lines) < expected_rows:
        return True, len(lines), f"원자료 공백 {expected_rows - len(lines)}시간"
    return True, len(lines), "정상"


def fetch(tm1: pd.Timestamp, tm2: pd.Timestamp, key: str) -> bytes:
    qs = f"tm1={tm1:%Y%m%d%H%M}&tm2={tm2:%Y%m%d%H%M}&stn={STN}&help=0&authKey={key}"
    last_err = ""
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(f"{API_URL}?{qs}", timeout=90) as resp:
                return resp.read()
        except Exception as exc:  # 네트워크/HTTP 오류는 최대 3번 다시 시도
            last_err = str(exc).replace(key, "***")   # 핵심: 오류 문구에 키가 섞여도 가린다
            time.sleep(2 * attempt)
    raise RuntimeError(last_err)


def collect(force: bool) -> list[dict]:
    """월별 원본을 받아 저장한다. 이미 정상인 달은 건너뛴다(이어받기)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("KMA_API_KEY", "").strip()
    results = []
    for tm1, tm2, rows in month_requests():
        path = raw_path(tm1)
        label = f"{tm1:%Y-%m}"
        if path.exists() and not force:
            ok, n, msg = validate_text(decode(path.read_bytes()), tm1, tm2, rows)
            if ok:
                results.append(dict(month=label, rows=n, expected=rows, ok=True, status=f"기존 파일 사용 ({msg})"))
                continue
        if not key:
            results.append(dict(month=label, rows=0, expected=rows, ok=False,
                                status="원본 없음/불량 + 환경변수 KMA_API_KEY 없음"))
            continue
        try:
            raw = fetch(tm1, tm2, key)
        except RuntimeError as exc:
            results.append(dict(month=label, rows=0, expected=rows, ok=False, status=f"받기 실패: {exc}"))
            continue
        ok, n, msg = validate_text(decode(raw), tm1, tm2, rows)
        if ok:
            path.write_bytes(raw)   # 핵심: 받은 그대로 저장한다 (가공은 변환 단계에서만)
        results.append(dict(month=label, rows=n, expected=rows, ok=ok,
                            status=(f"새로 받음 ({msg})" if ok else f"받았으나 불량: {msg}")))
        print(f"  {label} {'OK' if ok else 'FAIL'} rows={n}/{rows} {msg}", flush=True)
        time.sleep(1.0)   # 서버 부담을 줄이려고 호출 사이에 쉰다
    return results


# ───────────────────────── 2. 변환 ─────────────────────────
def load_raw() -> tuple[pd.DataFrame, int]:
    rows = []
    for tm1, _tm2, _n in month_requests():
        for ln in data_lines(decode(raw_path(tm1).read_bytes())):
            rows.append(ln.split())          # 공백으로 나뉜 46열
    df = pd.DataFrame(rows, columns=COLS)
    df["TM"] = pd.to_datetime(df["TM"], format="%Y%m%d%H%M")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("TM")
    dup = int(df["TM"].duplicated().sum())
    return df.drop_duplicates("TM", keep="last"), dup


def cumulative_to_hourly(raw: pd.Series, day_key: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """'그날 00시부터의 누적값'(RN_DAY, SD_DAY)을 '그 시간에 새로 생긴 양'으로 바꾼다.

    핵심: 11~3월에는 RN(강수량)이 3시간마다만 있어 시간별 비를 직접 만들 수 없다.
          매시간 채워지는 누적값의 앞뒤 차이로 대신 만든다 (원자료 RN 과 전부 일치함을 검증한다).
    관측시각 t 의 값은 't-1시~t시'까지의 누적이다. 00시 행은 '전날 24시'(전날 합계)이므로
    하루 묶음을 (t - 1시간)의 날짜로 만든다 → 묶음 = 01시, 02시, ... 23시, 다음날 00시.
    묶음의 첫 행은 0에서 시작하고, 나머지는 바로 앞 행과의 차이다.
    반환: (관측값(-9 는 NaN), 시간별 증가량)
    """
    obs = raw.where(raw >= 0)                                  # -9(없음/미관측) → NaN
    filled = obs.groupby(day_key).ffill().fillna(0.0)          # 같은 날 안에서 앞 값을 이어 쓰고, 아직 없으면 0
    hourly = filled.groupby(day_key).diff().fillna(filled)     # 묶음 첫 행은 누적값 그대로
    hourly.iloc[0] = np.nan                                    # 맨 첫 행은 전날 값이라 계산 불가(결과표에는 안 쓰임)
    return obs, hourly.clip(lower=0)                           # 눈은 다져져서 줄 수 있으므로 음수는 0


def build_weather(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """시간별 날씨표(상세)를 만든다. 저장할 때는 HOURLY_COLS 만 쓴다."""
    idx = pd.date_range(START, END_TS, freq=ONE_HOUR)
    d = df.set_index("TM").reindex(idx)
    row_missing = d["STN"].isna()                              # 원자료에 아예 없는 시각
    day_key = (idx - ONE_HOUR).normalize()

    ta_raw = d["TA"].where(d["TA"] > TA_MISSING_BELOW)
    ta = ta_raw.interpolate(limit=TEMP_FILL_LIMIT, limit_area="inside")   # 공백은 앞뒤 값을 직선으로 연결
    temp_filled = ta_raw.isna() & ta.notna()

    rain_obs, rain_row = cumulative_to_hourly(d["RN_DAY"], day_key)
    _snow_obs, snow_row = cumulative_to_hourly(d["SD_DAY"], day_key)

    # 핵심: 지하철 'h시' = h시~h+1시. 그 시간의 비·눈 = (h+1시 누적) - (h시 누적) = h+1시 행의 값 → 한 칸 당긴다
    rain_h = rain_row.shift(-1)
    snow_h = snow_row.shift(-1)
    gap_h = row_missing.shift(-1, fill_value=False) | row_missing   # h시 또는 h+1시 관측이 없으면 그 시간은 공백

    n = len(idx) - 1    # 2024-01-01 00시 ~ 2026-07-01 03시 (912일 x 24시간 + 마지막 날 24_28 용 4시간)
    # is_rain · is_snow 는 서로 독립이다 (눈이 녹거나 섞이면 둘 다 1). 겹쳐도 지우지 않는다.
    detail = pd.DataFrame({
        "date": idx[:n].strftime("%Y-%m-%d"),
        "hour": idx[:n].hour,
        "temperature": ta.iloc[:n].round(1).values,
        "rainfall": rain_h.iloc[:n].round(1).values,              # 검증용(mm). 저장하지 않는다
        "is_rain": (rain_h.iloc[:n] > 0).astype(int).values,      # 비가 한 방울이라도 기록됐나 (0.1mm 이상)
        "snowfall": snow_h.iloc[:n].round(1).values,              # 검증용(cm). 저장하지 않는다
        "is_snow": (snow_h.iloc[:n] > 0).astype(int).values,      # 새 눈이 0.1cm 이상. is_rain 과 독립이다
        "temp_filled": temp_filled.iloc[:n].astype(int).values,
        "obs_missing": gap_h.iloc[:n].astype(int).values,
    })
    parts = dict(idx=idx, d=d, ta=ta, ta_raw=ta_raw, rain_row=rain_row,
                 row_missing=row_missing, temp_filled=temp_filled)
    return detail, parts


def to_bins(detail: pd.DataFrame) -> pd.DataFrame:
    """시간별 → 지하철 시간대(수송일자 x 20개). preprocess.py 와 같은 키(수송일자, 시간대).

    지하철 시간대는 1시간이 아닌 것이 둘 있다.
        04_06 : 같은 날 04, 05시            (2시간)
        24_28 : 다음날 00, 01, 02, 03시      (4시간) — 날짜는 '전날' 수송일자에 속한다
    핵심: 여러 시간을 묶을 때 기온은 평균, 비·눈은 '그 안에 하나라도 있었으면 1'로 한다.
    """
    t = pd.to_datetime(detail["date"]) + pd.to_timedelta(detail["hour"], unit="h")
    h = detail["hour"].to_numpy()
    biz = (t - pd.to_timedelta((h < 4).astype(int), unit="D")).dt.normalize()        # 0~3시는 전날 수송일자
    label = np.where(h < 4, "24_28", np.where(h < 6, "04_06", [f"{x:02d}_{x + 1:02d}" for x in h]))
    work = detail.assign(수송일자=biz.values, 시간대=label)
    work = work[(work["수송일자"] >= START.normalize()) & (work["수송일자"] <= LAST_DAY)]
    out = work.groupby(["수송일자", "시간대"], sort=False).agg(
        temperature=("temperature", "mean"),
        rainfall=("rainfall", "sum"),
        snowfall=("snowfall", "sum"),
        obs_missing=("obs_missing", "max"),
        n_hours=("hour", "size"),
    ).reset_index()
    out["is_rain"] = (out["rainfall"] > 0).astype(int)
    out["is_snow"] = (out["snowfall"] > 0).astype(int)
    out["temperature"] = out["temperature"].round(1)
    order = {lb: i for i, lb in enumerate(BIN_LABELS)}
    out = out.sort_values(["수송일자", "시간대"], key=lambda s: s.map(order) if s.name == "시간대" else s).reset_index(drop=True)
    out["수송일자"] = out["수송일자"].dt.strftime("%Y-%m-%d")
    return out


# ───────────────────────── 3. 검증 ─────────────────────────
def gap_segments(idx: pd.DatetimeIndex, missing: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """빠진 시각을 연속 구간으로 묶는다."""
    segs, start, prev = [], None, None
    for t in idx[missing.values]:
        if start is None:
            start = prev = t
        elif t - prev == ONE_HOUR:
            prev = t
        else:
            segs.append((start, prev, int((prev - start) / ONE_HOUR) + 1))
            start = prev = t
    if start is not None:
        segs.append((start, prev, int((prev - start) / ONE_HOUR) + 1))
    return segs


def verify(detail: pd.DataFrame, bins: pd.DataFrame, parts: dict, dup_rows: int, results: list[dict]) -> str:
    idx, d = parts["idx"], parts["d"]
    rain_row = parts["rain_row"]
    ta, ta_raw, row_missing = parts["ta"], parts["ta_raw"], parts["row_missing"]
    L: list[str] = []
    P = L.append
    days = (LAST_DAY - START).days + 1

    P("=" * 78)
    P(f"날씨 수집·변환 검증 (지점 {STN} 서울, {START.date()} ~ {LAST_DAY.date()})")
    P("=" * 78)

    P("\n[1] 월별 원본")
    for r in results:
        P(f"  {r['month']}  {r['rows']:>4}/{r['expected']:<4}행  {r['status']}")

    P("\n[2] 시각 빠짐·중복")
    P(f"  기대 시각 {len(idx):,}개 (= {days}일 x 24시간 + 마지막 날 다음날 0~{EXTRA_HOURS}시 {EXTRA_HOURS + 1}개), 받은 시각 {int((~row_missing).sum()):,}개")
    P(f"  원자료 공백 {int(row_missing.sum())}시간, 중복 제거 {dup_rows}개")
    for s, e, n in gap_segments(idx, row_missing):
        P(f"    - {s:%Y-%m-%d %H시} ~ {e:%Y-%m-%d %H시} ({n}시간)")
    P(f"  시간별 표 {len(detail):,}행 (기대 {days * 24 + EXTRA_HOURS:,})")

    P("\n[3] 기온(TA)")
    P(f"  결측(공백 포함) {int(ta_raw.isna().sum())}개 → 직선 보간 {int(parts['temp_filled'].sum())}개, 보간 후 남은 결측 {int(ta.iloc[:-1].isna().sum())}개")
    P(f"  범위 {ta.min():.1f} ~ {ta.max():.1f}℃ (평균 {ta.mean():.1f}℃)")
    exact = ta_raw[ta_raw == -9.0]
    pv, nx = ta_raw.shift(1), ta_raw.shift(-1)
    sus = [t for t in exact.index if abs(-9.0 - pv[t]) > 4 and abs(-9.0 - nx[t]) > 4]
    P(f"  값이 정확히 -9.0 인 시각 {len(exact)}개 (앞뒤와 4℃ 넘게 튀는 의심 값 {len(sus)}개)")

    P("\n[4] 비 계산 검증 (원자료 RN 과, RN_DAY 차이로 만든 값을 비교)")
    rn = d["RN"].where(d["RN"] >= 0)
    summer = pd.Series((idx.month >= 4) & (idx.month <= 10), index=idx)
    rain3 = rain_row.rolling(3).sum()
    cmp_s = rn[summer & rn.notna()]
    mis_s = cmp_s[(cmp_s - rain_row[cmp_s.index]).abs() > TOL]
    P(f"  4~10월(RN=1시간 강수): 비교 {len(cmp_s):,}건, 불일치 {len(mis_s)}건")
    cmp_w = rn[~summer & rn.notna() & rain3.notna()]
    mis_w = cmp_w[(cmp_w - rain3[cmp_w.index]).abs() > TOL]
    P(f"  11~3월(RN=3시간 강수): 비교 {len(cmp_w):,}건, 불일치 {len(mis_w)}건")
    for name, ser, ref in (("4~10월", mis_s, rain_row), ("11~3월", mis_w, rain3)):
        for t in ser.index[:8]:
            P(f"    불일치[{name}] {t}  RN={ser[t]}  계산={ref[t]:.1f}")

    P("\n[5] 연도별 요약 (시간별 표)")
    o = detail.assign(year=detail["date"].str[:4])
    for y, g in o.groupby("year"):
        P(f"  {y}년: 비 온 시간 {int(g['is_rain'].sum())}, 눈 온 시간 {int(g['is_snow'].sum())}, 강수합 {g['rainfall'].sum():.1f}mm, 새눈합 {g['snowfall'].sum():.1f}cm")

    P("\n[6] 월별 요약 (상식 점검: 장마철·겨울 눈이 맞는지)")
    o["ym"] = o["date"].str[:7]
    for ym, g in o.groupby("ym"):
        P(f"  {ym}  비 온 시간 {int(g['is_rain'].sum()):>3}  눈 온 시간 {int(g['is_snow'].sum()):>3}  강수합 {g['rainfall'].sum():>7.1f}mm  평균기온 {g['temperature'].mean():>5.1f}℃")

    P("\n[7] 이상값")
    neg = int((detail["rainfall"] < 0).sum() + (detail["snowfall"] < 0).sum())
    P(f"  음수 비·눈 {neg}개 (0이어야 함)")
    P(f"  NaN → 기온 {int(detail['temperature'].isna().sum())}, 비 {int(detail['rainfall'].isna().sum())}, 눈 {int(detail['snowfall'].isna().sum())}")
    P(f"  공백 시간 표시 {int(detail['obs_missing'].sum())}시간")

    P("\n[7-2] 눈이면서 강수량도 기록된 시간 (is_rain=1 & is_snow=1)")
    both_h = detail[(detail["is_rain"] == 1) & (detail["is_snow"] == 1)]
    snow_h_n = int((detail["is_snow"] == 1).sum())
    both_b = bins[(bins["is_rain"] == 1) & (bins["is_snow"] == 1)]
    snow_b_n = int((bins["is_snow"] == 1).sum())
    P(f"  시간별 : 눈 {snow_h_n}시간 중 {len(both_h)}시간({len(both_h) / max(snow_h_n, 1):.1%})에 강수량도 기록")
    if len(both_h):
        P(f"    강수량(mm) 평균 {both_h['rainfall'].mean():.2f} · 중앙값 {both_h['rainfall'].median():.2f} · 최대 {both_h['rainfall'].max():.1f}")
    P(f"  시간대별 : 눈 {snow_b_n}칸 중 {len(both_b)}칸({len(both_b) / max(snow_b_n, 1):.1%})에 강수량도 기록")

    P("\n[8] 시간대별 표 (지하철 수송일자 x 시간대에 붙이는 표)")
    P(f"  행 수 {len(bins):,} (기대 {days * 20:,} = {days}일 x 20), (수송일자, 시간대) 중복 {int(bins.duplicated(['수송일자', '시간대']).sum())}개")
    wrong = {lb: int((bins.loc[bins['시간대'] == lb, 'n_hours'] != n).sum()) for lb, n in BIN_HOURS.items()}
    P(f"  묶은 시간 수가 틀린 행 {sum(wrong.values())}개 (04_06=2, 24_28=4, 나머지 1), 시간대 라벨 {bins['시간대'].nunique()}종 (기대 {len(BIN_LABELS)})")
    P(f"  NaN → 기온 {int(bins['temperature'].isna().sum())}, 비 {int(bins['is_rain'].isna().sum())}, 눈 {int(bins['is_snow'].isna().sum())}")
    P(f"  비 온 (날·시간대) {int(bins['is_rain'].sum()):,}개 / 눈 {int(bins['is_snow'].sum()):,}개 / 공백 포함 {int(bins['obs_missing'].sum())}개")

    P("\n[9] 결론")
    problems = []
    if ta.iloc[:-1].isna().any():
        problems.append(f"기온 결측 {int(ta.iloc[:-1].isna().sum())}개")
    if len(mis_s) or len(mis_w):
        problems.append(f"비 계산 불일치 {len(mis_s) + len(mis_w)}건")
    if neg:
        problems.append(f"음수 {neg}개")
    if len(detail) != days * 24 + EXTRA_HOURS:
        problems.append("시간별 표 행 수 불일치")
    if len(bins) != days * 20 or sum(wrong.values()) or bins.duplicated(["수송일자", "시간대"]).any():
        problems.append("시간대별 표 불일치")
    P("  " + ("문제 없음 (원자료 공백은 [2]에 기록)" if not problems else "확인 필요: " + ", ".join(problems)))
    return "\n".join(L)


# ───────────────────────── main ─────────────────────────
def main() -> None:
    force = "--force" in sys.argv
    report_path = Path(sys.argv[sys.argv.index("--report") + 1]) if "--report" in sys.argv else None
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # 콘솔 인코딩 때문에 글자 하나로 멈추지 않게
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] 월별 원본 확인·수집 ...", flush=True)
    results = collect(force)
    failed = [r for r in results if not r["ok"]]
    if failed:   # 핵심: 한 달이라도 못 받으면 구멍 난 표를 만들지 않고 멈춘다
        for r in failed:
            print(f"  실패 {r['month']}: {r['status']}")
        sys.exit(1)

    print("[2/3] 시간별·시간대별 날씨표 만들기 ...", flush=True)
    df, dup = load_raw()
    detail, parts = build_weather(df)
    bins = to_bins(detail)
    detail[HOURLY_COLS].to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    bins[BIN_COLS].to_csv(BIN_CSV_PATH, index=False, encoding="utf-8-sig")

    print("[3/3] 검증 ...", flush=True)
    report = verify(detail, bins, parts, dup, results)
    if report_path:
        report_path.write_text(report, encoding="utf-8")
        print(f"검증 보고서 저장: {report_path}")
    else:
        print(report)
    print(f"DONE hourly={len(detail)} bins={len(bins)} -> data/processed/{CSV_PATH.name}, {BIN_CSV_PATH.name}")


if __name__ == "__main__":
    main()
