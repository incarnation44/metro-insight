from datetime import datetime

import pytest

from metro_insight.data import weather
from metro_insight.data.weather import (
    COLUMNS,
    KST,
    HourlyWeather,
    WeatherApiError,
    current_weather,
    load_auth_key,
    parse_observations,
    to_hourly_weather,
    weather_at,
)

# 실제 응답 행 (2024-11-27 03시, 서울 폭설)
SNOW_LINE = (
    "202411270300 108  27  2.7  29 11.0  105 1000.3 1011.0  7  -1.3   0.3  -0.2  96.0   6.0    5.8    5.8"
    "    5.8   -9.0    4.8    4.8    5.4 73 77 05                      10   9    2 StNs      -9   2   7"
    "    93 -9.0 -9.00 -9   0.8   5.5   5.8   8.2   9.5  -9 -9.0 -9  4  1"
)
# 실제 응답 행 (2026-10-08 09시, 맑음)
CLEAR_LINE = (
    "202610080900 108   5  1.7  -9 -9.0   -9 1015.8 1026.1  2   1.3  15.9   8.6  62.0  11.2   -9.0   -9.0"
    "   -9.0   -9.0   -9.0   -9.0   -9.0 -9 -9 -                        0   0   -9 -         -9  -9  -9"
    "  3053  1.0  1.17 -9  13.3  15.1  16.0  16.7  19.7  -9 -9.0 -9  3  2"
)


def line(tm, temperature="0.0", daily_rain="-9.0", daily_snow="-9.0", code="-9"):
    values = dict(zip(COLUMNS, SNOW_LINE.split(), strict=True))
    values.update(TM=tm, TA=temperature, RN_DAY=daily_rain, SD_DAY=daily_snow, WC=code)
    return " ".join(values.values())


def observe(*args, **kwargs):
    return parse_observations(line(*args, **kwargs))[0]


def test_parse_real_response_rows():
    text = f"#START7777\n# YYMMDDHHMI STN ...\n{CLEAR_LINE}\n#7777END\n"
    (obs,) = parse_observations(text)
    assert obs.time == datetime(2026, 10, 8, 9, tzinfo=KST)
    assert obs.temperature == 15.9
    assert obs.daily_rain is None  # -9 는 결측 (학습 규칙처럼 앞 값으로 채운다)
    assert obs.daily_snow is None
    assert obs.weather_code is None

    (snow,) = parse_observations(SNOW_LINE)
    assert (snow.temperature, snow.daily_rain, snow.daily_snow, snow.weather_code) == (0.3, 5.8, 4.8, 73)


def test_clear_weather():
    obs = parse_observations(CLEAR_LINE)[0]
    prev = observe("202610080800", "12.5")
    assert to_hourly_weather(obs, prev) == HourlyWeather("2026-10-08", 9, 15.9, rainfall=0, snowfall=0)


def test_snow_counts_as_both_precipitation_and_snow():
    obs = parse_observations(SNOW_LINE)[0]
    prev = observe("202411270200", daily_rain="2.4", daily_snow="3.4")
    assert to_hourly_weather(obs, prev) == HourlyWeather("2024-11-27", 3, 0.3, rainfall=1, snowfall=1)


def test_weather_code_alone_no_longer_counts():
    # 08시: 일강수량·일적설 변화 없음, 현재일기 22 (직전 1시간 눈). 학습 규칙은 수치만 본다.
    obs = observe("202411270800", "-1.3", "16.3", "16.3", "22")
    prev = observe("202411270700", "-1.0", "16.3", "16.3", "71")
    weather = to_hourly_weather(obs, prev)
    assert (weather.rainfall, weather.snowfall) == (0, 0)


def test_snow_detected_from_daily_snow_without_rain():
    # 눈만 0.4cm 쌓이고 강수량은 그대로 → 눈은 강수에도 포함된다
    obs = observe("202411270800", "-1.3", "16.3", "4.0")
    prev = observe("202411270700", "-1.0", "16.3", "3.6")
    weather = to_hourly_weather(obs, prev)
    assert (weather.rainfall, weather.snowfall) == (1, 1)


def test_amounts_below_threshold_are_not_precipitation():
    obs = observe("202411270800", "0.2", "16.3", "3.6")
    prev = observe("202411270700", "0.2", "16.3", "3.6")
    weather = to_hourly_weather(obs, prev)
    assert (weather.rainfall, weather.snowfall) == (0, 0)


def test_omitted_weather_code_and_no_precipitation():
    # 16시: 현재일기 생략, 일강수량 변화 없음
    obs = observe("202411271600", "0.4", "19.6")
    prev = observe("202411271500", "-0.3", "19.6", "22")
    weather = to_hourly_weather(obs, prev)
    assert (weather.rainfall, weather.snowfall) == (0, 0)


def test_rain_code():
    obs = observe("202208081200", "28.0", "3.3", "60")
    weather = to_hourly_weather(obs, observe("202208081100", "27.8", "3.1", "60"))
    assert (weather.rainfall, weather.snowfall) == (1, 0)


def test_rain_detected_from_daily_rain_when_code_is_omitted():
    obs = observe("202208081200", "28.0", "3.3")
    assert to_hourly_weather(obs, observe("202208081100", "27.8", "3.1")).rainfall == 1


def test_daily_rain_restarts_at_one_oclock():
    # 00시 값은 전날 합계, 01시 값부터 새로 쌓인다
    obs = observe("202411270100", "1.2", "0.3")
    prev = observe("202411270000", "1.0", "26.0")
    assert to_hourly_weather(obs, prev).rainfall == 1
    assert to_hourly_weather(obs, None).rainfall == 1
    assert to_hourly_weather(observe("202411270100", "1.2"), prev).rainfall == 0


def test_missing_previous_hour_counts_no_precipitation():
    # 직전 시각 자료가 없으면 차이를 낼 수 없으므로 강수·눈 없음으로 둔다 (일기코드는 보지 않는다)
    assert to_hourly_weather(observe("202411270900", daily_rain="16.3"), None).rainfall == 0
    assert to_hourly_weather(observe("202411270900", daily_rain="16.3", code="70"), None).rainfall == 0


def test_daily_snow_restarts_at_one_oclock():
    # 일적설도 01시부터 쌓이므로 01시는 그 값 자체가 1시간 적설이다
    obs = observe("202411270100", "-2.0", "0.0", "0.4")
    assert to_hourly_weather(obs, observe("202411270000", "-2.0", "0.0", "26.0")).snowfall == 1
    assert to_hourly_weather(obs, None).snowfall == 1


def test_minus_nine_degrees_is_a_temperature_not_missing():
    assert observe("202501100900", "-9.0").temperature == -9.0
    assert observe("202501100900", "-99.0").temperature is None


def test_unexpected_column_count_raises():
    with pytest.raises(WeatherApiError):
        parse_observations(CLEAR_LINE + " 1")


def test_load_auth_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv(weather.AUTH_KEY_ENV, raising=False)
    monkeypatch.setattr(weather, "ENV_FILE", tmp_path / ".env.production")
    with pytest.raises(WeatherApiError):
        load_auth_key()
    (tmp_path / ".env.production").write_text("OTHER=1\nKMA_APIHUB_AUTH_KEY=abc123\n")
    assert load_auth_key() == "abc123"
    monkeypatch.setenv(weather.AUTH_KEY_ENV, "from-env")
    assert load_auth_key() == "from-env"


def test_current_weather_falls_back_to_latest_available_hour(monkeypatch):
    available = {
        datetime(2026, 10, 8, 9, tzinfo=KST): parse_observations(CLEAR_LINE)[0],
        datetime(2026, 10, 8, 8, tzinfo=KST): observe("202610080800", "12.5"),
    }
    requested = []

    def fake_fetch(tm, auth_key):
        requested.append(tm.hour)
        return available.get(tm)

    monkeypatch.setattr(weather, "archive_available", lambda tm: False)  # 저장 자료 없이 API 만 본다
    monkeypatch.setattr(weather, "fetch_observation", fake_fetch)
    result = current_weather(datetime(2026, 10, 8, 10, 5, tzinfo=KST), auth_key="key")
    assert result == HourlyWeather("2026-10-08", 9, 15.9, rainfall=0, snowfall=0)
    assert requested == [10, 9, 8]


def test_weather_at_past_hour(monkeypatch):
    requested = []

    def fake_fetch(tm, auth_key):
        requested.append(tm)
        return observe(tm.strftime("%Y%m%d%H00"), "-4.5", "1.0" if tm.hour == 7 else "0.0")

    monkeypatch.setattr(weather, "archive_available", lambda tm: False)  # 저장 자료 없이 API 만 본다
    monkeypatch.setattr(weather, "fetch_observation", fake_fetch)
    result = weather_at(datetime(2025, 1, 9, 7, 40, tzinfo=KST), auth_key="key")
    assert result == HourlyWeather("2025-01-09", 7, -4.5, rainfall=1, snowfall=0)
    assert requested == [datetime(2025, 1, 9, 7, tzinfo=KST), datetime(2025, 1, 9, 6, tzinfo=KST)]


def test_weather_at_without_observations_raises(monkeypatch):
    monkeypatch.setattr(weather, "archive_available", lambda tm: False)
    monkeypatch.setattr(weather, "fetch_observation", lambda tm, auth_key: None)
    with pytest.raises(WeatherApiError, match="2025-01-09 07시"):
        weather_at(datetime(2025, 1, 9, 7, tzinfo=KST), auth_key="key")


def test_archive_weather_needs_no_api_key():
    """저장해 둔 원자료가 있으면 기상청 API 키 없이 그 관측값을 쓴다."""
    result = weather_at(datetime(2024, 1, 1, 3, tzinfo=KST))
    assert result.source == "저장 자료"
    assert result.date == "2024-01-01"
    assert result.temperature is not None


def test_archive_keeps_next_day_hours_at_month_end():
    """달 마지막 날 24시(익일 0시)는 전날 파일에 들어 있다."""
    obs = weather.archive_observation(datetime(2026, 7, 1, 0, tzinfo=KST))
    assert obs is not None
    assert obs.time == datetime(2026, 7, 1, 0, tzinfo=KST)


def test_api_error_429_is_retried(monkeypatch):
    """요청 과다(429)는 잠깐 기다렸다 다시 시도한다."""
    calls = []

    class FakeResponse:
        def __init__(self, status):
            self.status_code = status
            self.ok = status == 200
            self.text = "너무 많음"
            self.content = b""

        def json(self):
            return {}

    def fake_get(url, params=None, timeout=None):
        calls.append(params["tm"])
        return FakeResponse(429)

    monkeypatch.setattr(weather.requests, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    with pytest.raises(WeatherApiError):
        weather.fetch_observation(datetime(2026, 10, 8, 9, tzinfo=KST), "key")
    assert len(calls) == 3  # 429 를 세 번 시도했다


def test_api_error_429_is_retried_then_succeeds(monkeypatch):
    """429 가 한 번 나면 다시 시도해서 관측값을 받는다."""
    calls = []

    class FakeResponse:
        def __init__(self, status, content=b""):
            self.status_code = status
            self.ok = status == 200
            self.text = "너무 많음"
            self.content = content

    def fake_get(url, params=None, timeout=None):
        calls.append(params["tm"])
        if len(calls) == 1:
            return FakeResponse(429)
        return FakeResponse(200, CLEAR_LINE.encode("euc-kr"))

    monkeypatch.setattr(weather.requests, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    obs = weather.fetch_observation(datetime(2026, 10, 8, 9, tzinfo=KST), "key")
    assert obs is not None and obs.temperature == 15.9
    assert len(calls) == 2


def _training_table() -> dict:
    """학습 때 만든 날씨표(20버킷)를 읽는다."""
    import csv
    import glob
    from pathlib import Path

    processed = Path(weather.RAW_DIR).parent.parent / "processed"
    for path in glob.glob(str(processed / "weather*.csv")):
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and rows[0]["hour"] == "04_06":
            return {(r["date"], r["hour"]): (int(r["rainfall"]), int(r["snowfall"])) for r in rows}
    raise AssertionError("학습 때 만든 날씨표가 없다")


def test_slot_weather_matches_training_table():
    """화면 시간대의 날씨가 학습 때 만든 날씨표와 같다 (표본 검사)."""
    from datetime import timedelta

    table = _training_table()
    items = sorted(table.items())
    assert len(items) > 1000
    for (date, label), expected in items[::97]:
        day = datetime.fromisoformat(date).replace(tzinfo=KST)
        if label == "24_01":
            tm = day + timedelta(days=1)  # 다음 날 00시
        elif label == "04_06":
            tm = day.replace(hour=4)
        else:
            tm = day.replace(hour=int(label[:2]))
        got = weather.slot_weather(tm)
        assert (got.rainfall, got.snowfall) == expected, f"{date} {label}"


def test_slot_weather_reads_archive_without_api_key():
    """저장 원자료로 계산한다 (기상청 API 키 없이)."""
    got = weather.slot_weather(datetime(2024, 1, 1, 4, tzinfo=KST))
    assert got.source == "저장 자료"
    assert got.rainfall == 0 and got.snowfall == 0
