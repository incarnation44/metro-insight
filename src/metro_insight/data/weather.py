"""서울 날씨 데이터 (기상청 API허브 지상관측 시간자료, 서울 108번 관측소).

날씨는 서울 전체에 한 값이므로 지하철 이용 데이터와 date + hour 로 결합한다.

* temperature: 기온 (℃)
* rainfall: 강수 여부 (0/1). 눈도 강수에 포함한다
* snowfall: 눈 여부 (0/1)

강수·눈 여부는 누적값의 직전 시각과의 차이로만 판단한다 (학습 규칙 notebooks/fetch_weather.py 와 같다).

* 시간자료의 강수량(RN)은 11~3월에 3시간마다만 기록되므로 쓰지 않고,
  매시 기록되는 일강수량(RN_DAY)의 직전 시각과의 차이를 1시간 강수량으로 쓴다
* 적설(SD_DAY)도 같은 방식으로 차이를 내고, 0.1cm 이상 쌓였으면 눈으로 본다
* 일강수량은 01시부터 쌓이고 00시 값은 전날 합계이다. 따라서 01시는 그 값 자체가 1시간 강수량이다
* 현재일기 코드(WC)는 쓰지 않는다. 코드만 있고 수치가 없는 시간을 강수로 세면 학습표와 어긋난다

인증키는 환경변수 KMA_APIHUB_AUTH_KEY 또는 저장소 루트의 .env.production 에서 읽는다.

날씨 조회: python -m metro_insight.data.weather [YYYY-MM-DDTHH] (생략하면 현재)
"""
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
SEOUL_STATION = 108
KST = ZoneInfo("Asia/Seoul")

AUTH_KEY_ENV = "KMA_APIHUB_AUTH_KEY"
ENV_FILE = Path(__file__).resolve().parents[3] / ".env.production"

# API가 간헐적으로 30초 대기 후 504를 돌려주므로 재시도한다
TIMEOUT_SEC = 35
RETRIES = 3
RETRY_WAIT_SEC = 2

# 정시 자료는 정시 후 수십 분 뒤에 올라오고 가끔 빠지기도 하므로 이 시간만큼 거슬러 찾는다
LOOKBACK_HOURS = 3

# 응답 데이터 행의 열 순서 (help=1 로 조회하면 설명이 나온다)
COLUMNS = (
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR", "TA", "TD", "HM", "PV",
    "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3", "SD_DAY", "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID",
    "CH_MIN", "CT", "CT_TOP", "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01",
    "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR", "IX",
)  # fmt: skip

# 학습 때 받아 둔 월별 원자료(같은 기상청 자료). 여기 있으면 네트워크 없이 그 관측값을 쓴다.
RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "weather"
ARCHIVE_PREFIX = "kma_sfctm3_stn108_"
# 원자료의 열 순서 (기상청 안내문 순서, 46열)
ARCHIVE_COLS = (
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR",
    "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3", "SD_DAY",
    "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN", "CT", "CT_TOP", "CT_MID",
    "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01", "TE_02", "TE_03",
    "ST_SEA", "WH", "BF", "IR", "IX",
)  # fmt: skip

# 결측은 -9 (기온은 -99) 로 표시된다
MISSING = -9.0
MISSING_TEMPERATURE = -50.0

# 강수·적설 판정 임계값 (학습 규칙 notebooks/fetch_weather.py 와 같다)
PRECIPITATION_THRESHOLD = 0.1  # mm
SNOWFALL_THRESHOLD = 0.1  # cm


class WeatherApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class Observation:
    """관측소 시간자료 한 행에서 날씨 Feature 계산에 필요한 값."""

    time: datetime
    temperature: float | None
    daily_rain: float | None  # 일강수량 (mm). 결측(-9)은 None (학습 규칙처럼 앞 값으로 채운다)
    daily_snow: float | None  # 일적설 (cm). 결측은 None
    weather_code: int | None
    source: str = "기상청"  # 어디서 온 관측값인가 (기상청 API / 저장해 둔 원자료)


@dataclass(frozen=True)
class HourlyWeather:
    date: str  # yyyy-mm-dd
    hour: int
    temperature: float | None
    rainfall: int
    snowfall: int
    source: str = "기상청"


def parse_observations(text: str) -> list[Observation]:
    observations = []
    for line in text.splitlines():
        if not line[:1].isdigit():
            continue
        values = line.split()
        if len(values) != len(COLUMNS):
            raise WeatherApiError(f"응답 열 개수가 {len(COLUMNS)}개가 아니다: {len(values)}개")
        row = dict(zip(COLUMNS, values, strict=True))
        temperature = float(row["TA"])
        daily_rain = float(row["RN_DAY"])
        daily_snow = float(row["SD_DAY"])
        weather_code = int(row["WC"])
        observations.append(
            Observation(
                time=datetime.strptime(row["TM"], "%Y%m%d%H%M").replace(tzinfo=KST),
                temperature=temperature if temperature > MISSING_TEMPERATURE else None,
                daily_rain=daily_rain if daily_rain != MISSING else None,
                daily_snow=daily_snow if daily_snow != MISSING else None,
                weather_code=weather_code if weather_code != MISSING else None,
            )
        )
    return observations


def hourly_precipitation(obs: Observation, prev: Observation | None) -> float | None:
    """직전 1시간 강수량 (mm). 직전 시각 자료가 없으면 None."""
    if obs.time.hour == 1:
        return obs.daily_rain
    if prev is None or obs.daily_rain is None or prev.daily_rain is None:
        return None
    return max(obs.daily_rain - prev.daily_rain, 0.0)


def hourly_snowfall(obs: Observation, prev: Observation | None) -> float | None:
    """직전 1시간 새로 쌓인 눈 (cm). 직전 시각 자료가 없으면 None."""
    if obs.time.hour == 1:
        return obs.daily_snow
    if prev is None or obs.daily_snow is None or prev.daily_snow is None:
        return None
    return max(obs.daily_snow - prev.daily_snow, 0.0)


def to_hourly_weather(obs: Observation, prev: Observation | None) -> HourlyWeather:
    precipitation = hourly_precipitation(obs, prev)
    snowfall = hourly_snowfall(obs, prev)
    # 학습과 같은 규칙: 강수량 ≥0.1mm 또는 적설 ≥0.1cm (눈도 강수에 포함). 일기코드는 보지 않는다.
    return HourlyWeather(
        date=obs.time.strftime("%Y-%m-%d"),
        hour=obs.time.hour,
        temperature=obs.temperature,
        rainfall=int((precipitation or 0.0) >= PRECIPITATION_THRESHOLD
                     or (snowfall or 0.0) >= SNOWFALL_THRESHOLD),
        snowfall=int((snowfall or 0.0) >= SNOWFALL_THRESHOLD),
        source=obs.source,
    )


# ───────────────────────── 저장해 둔 원자료 ─────────────────────────

_ARCHIVE_CACHE: dict[str, dict[datetime, Observation]] = {}


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp949", errors="replace")


def _observation_from_row(row: dict[str, str]) -> Observation | None:
    """원자료 한 줄 → 관측값. 해석 안 되는 줄은 None."""
    try:
        when = datetime.strptime(row["TM"], "%Y%m%d%H%M").replace(tzinfo=KST)
        temperature = float(row["TA"])
        daily_rain = float(row["RN_DAY"])
        daily_snow = float(row["SD_DAY"])
        code = float(row["WC"])
    except (KeyError, ValueError):
        return None
    return Observation(
        time=when,
        temperature=temperature if temperature > MISSING_TEMPERATURE else None,
        daily_rain=daily_rain if daily_rain != MISSING else None,
        daily_snow=daily_snow if daily_snow != MISSING else None,
        weather_code=int(code) if code != MISSING else None,
        source="저장 자료",
    )


def archive_observations(tm: datetime) -> dict[datetime, Observation]:
    """그 달 원자료(data/raw/weather)의 시간별 관측값. 파일이 없으면 빈 dict."""
    key = f"{tm:%Y%m}"
    if key not in _ARCHIVE_CACHE:
        found: dict[datetime, Observation] = {}
        path = RAW_DIR / f"{ARCHIVE_PREFIX}{key}.txt"
        if path.exists():
            for line in _decode(path.read_bytes()).splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != len(ARCHIVE_COLS):
                    continue
                obs = _observation_from_row(dict(zip(ARCHIVE_COLS, parts, strict=True)))
                if obs is not None:
                    found[obs.time] = obs
        _ARCHIVE_CACHE[key] = found
    return _ARCHIVE_CACHE[key]


def archive_observation(tm: datetime) -> Observation | None:
    """저장 원자료에서 그 시각 관측값. 없으면 None."""
    obs = archive_observations(tm).get(tm)
    if obs is not None:
        return obs
    # 달 경계는 전날 파일에 다음 날 0~4시가 들어 있다 (학습 때와 같은 구성)
    before = tm - timedelta(hours=5)
    if before.month != tm.month:
        return archive_observations(before).get(tm)
    return None


def archive_available(tm: datetime) -> bool:
    """그 달 원자료 파일이 있는가."""
    if (RAW_DIR / f"{ARCHIVE_PREFIX}{tm:%Y%m}.txt").exists():
        return True
    before = tm - timedelta(hours=5)
    return before.month != tm.month and (
        RAW_DIR / f"{ARCHIVE_PREFIX}{before:%Y%m}.txt"
    ).exists()


def load_auth_key() -> str:
    if key := os.environ.get(AUTH_KEY_ENV):
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == AUTH_KEY_ENV:
                return value.strip().strip("\"'")
    raise WeatherApiError(f"인증키가 없다: 환경변수 {AUTH_KEY_ENV} 또는 {ENV_FILE} 에 설정한다")


_KEY_CACHE: str | None = None


def _auth_key(auth_key: str | None) -> str:
    """인증키. 저장 자료로 충분하면 API 키가 필요 없다."""
    global _KEY_CACHE
    if auth_key:
        return auth_key
    if _KEY_CACHE is None:
        _KEY_CACHE = load_auth_key()
    return _KEY_CACHE


def _observation_at(tm: datetime, auth_key: str | None) -> Observation | None:
    """그 시각 관측값. 저장 원자료가 있는 달은 그것만 보고, 없으면 기상청 API를 쓴다."""
    if archive_available(tm):
        return archive_observation(tm)
    return fetch_observation(tm, _auth_key(auth_key))


def _group_start(tm: datetime) -> datetime:
    """학습 표의 묶음 시작 (그날 01시). 묶음 = 그날 01시 ~ 다음 날 00시."""
    group_day = (tm - timedelta(hours=1)).date()
    return datetime(group_day.year, group_day.month, group_day.day, 1, tzinfo=KST)


def _filled_cum(tm: datetime, auth_key: str | None) -> tuple[float, float]:
    """tm까지의 누적 강수·적설. 관측이 없으면 같은 묶음의 앞 값으로 채운다(학습 규칙)."""
    rain: float | None = None
    snow: float | None = None
    cur = tm
    start = _group_start(tm)
    while cur > start and (rain is None or snow is None):
        obs = _observation_at(cur, auth_key)
        if obs is not None:
            if rain is None and obs.daily_rain is not None:
                rain = obs.daily_rain
            if snow is None and obs.daily_snow is not None:
                snow = obs.daily_snow
        cur -= timedelta(hours=1)
    return (0.0 if rain is None else rain), (0.0 if snow is None else snow)


def slot_hours(tm: datetime) -> int:
    """그 시간대가 포함하는 관측 시간 수 (학습 표의 시간대와 같다).

    '24시' = 24:00~익일 01:00 (1시간), '04~06시' = 04, 05시 (2시간), 나머지 1시간.
    """
    if tm.hour == 0:
        return 1
    if tm.hour == 4:
        return 2
    return 1


def slot_weather(tm: datetime, auth_key: str | None = None) -> HourlyWeather:
    """학습(notebooks/fetch_weather.py)과 같은 규칙으로 그 시간대의 날씨를 만든다.

    지하철 'h시' = h시~h+1시. 그 시간의 비·눈 = (h+1시 누적) - (h시 누적).
    관측이 없는 시각은 묶음 안에서 앞 값으로 채운다(학습 규칙과 동일).
    """
    tm = tm.astimezone(KST).replace(minute=0, second=0, microsecond=0)
    count = slot_hours(tm)
    rainfall = snowfall = 0
    temperatures: list[float] = []
    source = "기상청"
    for k in range(count):
        hour = tm + timedelta(hours=k)
        next_hour = hour + timedelta(hours=1)
        if next_hour.hour == 1:  # 묶음 첫 시간의 증분은 누적값 자체
            cum_rain, cum_snow = _filled_cum(next_hour, auth_key)
            inc_rain, inc_snow = round(cum_rain, 1), round(cum_snow, 1)
        else:
            rain_now, snow_now = _filled_cum(next_hour, auth_key)
            rain_prev, snow_prev = _filled_cum(next_hour - timedelta(hours=1), auth_key)
            # 학습 표처럼 소수 1자리로 반올림한다 (0.1 경계에서 어긋나지 않게)
            inc_rain = round(max(rain_now - rain_prev, 0.0), 1)
            inc_snow = round(max(snow_now - snow_prev, 0.0), 1)
        if inc_rain >= PRECIPITATION_THRESHOLD or inc_snow >= SNOWFALL_THRESHOLD:
            rainfall = 1
        if inc_snow >= SNOWFALL_THRESHOLD:
            snowfall = 1
        obs = _observation_at(hour, auth_key)
        if obs is not None:
            source = obs.source
            if obs.temperature is not None:
                temperatures.append(obs.temperature)
    temperature = round(sum(temperatures) / len(temperatures), 1) if temperatures else None
    return HourlyWeather(
        date=tm.strftime("%Y-%m-%d"),
        hour=tm.hour,
        temperature=temperature,
        rainfall=rainfall,
        snowfall=snowfall,
        source=source,
    )


def fetch_observation(tm: datetime, auth_key: str) -> Observation | None:
    """tm 정시의 서울 관측자료. 아직 올라오지 않았으면 None."""
    params = {"tm": tm.strftime("%Y%m%d%H00"), "stn": SEOUL_STATION, "help": 0, "authKey": auth_key}
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(API_URL, params=params, timeout=TIMEOUT_SEC)
        except requests.RequestException as e:
            # 예외 메시지에 인증키가 담긴 URL이 들어갈 수 있어 원인 예외를 붙이지 않는다
            error = f"기상청 API 연결 실패 ({type(e).__name__})"
        else:
            if response.ok:
                observations = parse_observations(response.content.decode("euc-kr", errors="replace"))
                return observations[0] if observations else None
            if response.status_code < 500 and response.status_code not in (408, 429):
                raise WeatherApiError(f"기상청 API 오류 {response.status_code}: {response.text.strip()}")
            # 5xx 와 429(요청 과다)·408 는 잠깐 기다렸다 다시 시도한다
            error = f"기상청 API 오류 {response.status_code}"
        if attempt < RETRIES:
            time.sleep(RETRY_WAIT_SEC)
    raise WeatherApiError(f"{error} ({RETRIES}회 시도)")


def weather_at(tm: datetime, auth_key: str | None = None) -> HourlyWeather:
    """tm 정시의 서울 날씨. 그 정시 자료가 없으면 가장 가까운 이전 정시 자료."""
    tm = tm.astimezone(KST).replace(minute=0, second=0, microsecond=0)
    requested = tm
    for _ in range(LOOKBACK_HOURS):
        if obs := _observation_at(tm, auth_key):
            break
        tm -= timedelta(hours=1)
    else:
        raise WeatherApiError(f"{requested:%Y-%m-%d %H}시까지 {LOOKBACK_HOURS}시간 동안의 관측자료가 없다")
    prev = None if tm.hour == 1 else _observation_at(tm - timedelta(hours=1), auth_key)
    return to_hourly_weather(obs, prev)


def current_weather(now: datetime | None = None, auth_key: str | None = None) -> HourlyWeather:
    """가장 최근 정시의 서울 날씨."""
    return weather_at(now or datetime.now(KST), auth_key)


if __name__ == "__main__":
    tm = datetime.fromisoformat(sys.argv[1]).replace(tzinfo=KST) if len(sys.argv) > 1 else datetime.now(KST)
    print(json.dumps(asdict(weather_at(tm)), ensure_ascii=False))
