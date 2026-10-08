"""서울 날씨 데이터 (기상청 API허브 지상관측 시간자료, 서울 108번 관측소).

날씨는 서울 전체에 한 값이므로 지하철 이용 데이터와 date + hour 로 결합한다.

* temperature: 기온 (℃)
* rainfall: 강수 여부 (0/1). 눈도 강수에 포함한다
* snowfall: 눈 여부 (0/1)

강수·눈 여부는 직전 1시간 강수량과 현재일기 코드로 판단한다.

* 시간자료의 강수량(RN)은 11~3월에 3시간마다만 기록되므로 쓰지 않고,
  매시 기록되는 일강수량(RN_DAY)의 직전 시각과의 차이를 1시간 강수량으로 쓴다
* 일강수량은 01시부터 쌓이고 00시 값은 전날 합계이다. 따라서 01시는 그 값 자체가 1시간 강수량이다
* 현재일기 코드(WC, GTS Code 4677)는 특이 날씨가 없으면 생략(-9)되므로 강수량과 함께 본다

인증키는 환경변수 KMA_APIHUB_AUTH_KEY 또는 실행 위치의 .env.production 에서 읽는다.

현재 날씨 조회: python -m metro_insight.data.weather
"""
import json
import os
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
ENV_FILE = Path(".env.production")

# API가 간헐적으로 30초 대기 후 504를 돌려주므로 재시도한다
TIMEOUT_SEC = 35
RETRIES = 3
RETRY_WAIT_SEC = 2

# 정시 자료는 정시 후 수십 분 뒤에 올라오므로 최신 자료를 이 시간만큼 거슬러 찾는다
LOOKBACK_HOURS = 3

# 응답 데이터 행의 열 순서 (help=1 로 조회하면 설명이 나온다)
COLUMNS = (
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR", "TA", "TD", "HM", "PV",
    "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3", "SD_DAY", "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID",
    "CH_MIN", "CT", "CT_TOP", "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01",
    "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR", "IX",
)  # fmt: skip

# 현재일기 코드: 20~27 은 직전 1시간 동안의 강수, 50~99 는 관측 시각의 강수·뇌전
PRECIPITATION_CODES = frozenset([*range(20, 28), *range(50, 100)])
SNOW_CODES = frozenset([22, 23, 26, 68, 69, *range(70, 80), 83, 84, 85, 86, 87, 88, 93, 94])

# 결측은 -9 (기온은 -99) 로 표시된다
MISSING = -9.0
MISSING_TEMPERATURE = -50.0


class WeatherApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class Observation:
    """관측소 시간자료 한 행에서 날씨 Feature 계산에 필요한 값."""

    time: datetime
    temperature: float | None
    daily_rain: float  # 일강수량 (mm), 강수 없음은 0
    weather_code: int | None


@dataclass(frozen=True)
class HourlyWeather:
    date: str  # yyyy-mm-dd
    hour: int
    temperature: float | None
    rainfall: int
    snowfall: int


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
        weather_code = int(row["WC"])
        observations.append(
            Observation(
                time=datetime.strptime(row["TM"], "%Y%m%d%H%M").replace(tzinfo=KST),
                temperature=temperature if temperature > MISSING_TEMPERATURE else None,
                daily_rain=daily_rain if daily_rain != MISSING else 0.0,
                weather_code=weather_code if weather_code != MISSING else None,
            )
        )
    return observations


def hourly_precipitation(obs: Observation, prev: Observation | None) -> float | None:
    """직전 1시간 강수량 (mm). 직전 시각 자료가 없으면 None."""
    if obs.time.hour == 1:
        return obs.daily_rain
    if prev is None:
        return None
    return max(obs.daily_rain - prev.daily_rain, 0.0)


def to_hourly_weather(obs: Observation, prev: Observation | None) -> HourlyWeather:
    precipitation = hourly_precipitation(obs, prev)
    rained = (precipitation or 0.0) > 0 or obs.weather_code in PRECIPITATION_CODES
    return HourlyWeather(
        date=obs.time.strftime("%Y-%m-%d"),
        hour=obs.time.hour,
        temperature=obs.temperature,
        rainfall=int(rained),
        snowfall=int(obs.weather_code in SNOW_CODES),
    )


def load_auth_key() -> str:
    if key := os.environ.get(AUTH_KEY_ENV):
        return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == AUTH_KEY_ENV:
                return value.strip().strip("\"'")
    raise WeatherApiError(f"인증키가 없다: 환경변수 {AUTH_KEY_ENV} 또는 {ENV_FILE} 에 설정한다")


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
            if response.status_code < 500:
                raise WeatherApiError(f"기상청 API 오류 {response.status_code}: {response.text.strip()}")
            error = f"기상청 API 오류 {response.status_code}"
        if attempt < RETRIES:
            time.sleep(RETRY_WAIT_SEC)
    raise WeatherApiError(f"{error} ({RETRIES}회 시도)")


def current_weather(now: datetime | None = None, auth_key: str | None = None) -> HourlyWeather:
    """가장 최근 정시의 서울 날씨."""
    auth_key = auth_key or load_auth_key()
    tm = (now or datetime.now(KST)).astimezone(KST).replace(minute=0, second=0, microsecond=0)
    for _ in range(LOOKBACK_HOURS):
        if obs := fetch_observation(tm, auth_key):
            break
        tm -= timedelta(hours=1)
    else:
        raise WeatherApiError(f"최근 {LOOKBACK_HOURS}시간 동안의 관측자료가 없다")
    prev = None if tm.hour == 1 else fetch_observation(tm - timedelta(hours=1), auth_key)
    return to_hourly_weather(obs, prev)


if __name__ == "__main__":
    print(json.dumps(asdict(current_weather()), ensure_ascii=False))
