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


def line(tm, temperature="0.0", daily_rain="-9.0", code="-9"):
    values = dict(zip(COLUMNS, SNOW_LINE.split(), strict=True))
    values.update(TM=tm, TA=temperature, RN_DAY=daily_rain, WC=code)
    return " ".join(values.values())


def observe(*args, **kwargs):
    return parse_observations(line(*args, **kwargs))[0]


def test_parse_real_response_rows():
    text = f"#START7777\n# YYMMDDHHMI STN ...\n{CLEAR_LINE}\n#7777END\n"
    (obs,) = parse_observations(text)
    assert obs.time == datetime(2026, 10, 8, 9, tzinfo=KST)
    assert obs.temperature == 15.9
    assert obs.daily_rain == 0.0
    assert obs.weather_code is None

    (snow,) = parse_observations(SNOW_LINE)
    assert (snow.temperature, snow.daily_rain, snow.weather_code) == (0.3, 5.8, 73)


def test_clear_weather():
    obs = parse_observations(CLEAR_LINE)[0]
    prev = observe("202610080800", "12.5")
    assert to_hourly_weather(obs, prev) == HourlyWeather("2026-10-08", 9, 15.9, rainfall=0, snowfall=0)


def test_snow_counts_as_both_precipitation_and_snow():
    obs = parse_observations(SNOW_LINE)[0]
    prev = observe("202411270200", daily_rain="2.4", code="73")
    assert to_hourly_weather(obs, prev) == HourlyWeather("2024-11-27", 3, 0.3, rainfall=1, snowfall=1)


def test_snow_in_past_hour_code_without_new_precipitation():
    # 08시: 일강수량 변화 없음, 현재일기 22 (직전 1시간 눈)
    obs = observe("202411270800", "-1.3", "16.3", "22")
    prev = observe("202411270700", "-1.0", "16.3", "71")
    weather = to_hourly_weather(obs, prev)
    assert (weather.rainfall, weather.snowfall) == (1, 1)


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


def test_missing_previous_hour_uses_weather_code_only():
    assert to_hourly_weather(observe("202411270900", daily_rain="16.3"), None).rainfall == 0
    assert to_hourly_weather(observe("202411270900", daily_rain="16.3", code="70"), None).rainfall == 1


def test_minus_nine_degrees_is_a_temperature_not_missing():
    assert observe("202501100900", "-9.0").temperature == -9.0
    assert observe("202501100900", "-99.0").temperature is None


def test_unexpected_column_count_raises():
    with pytest.raises(WeatherApiError):
        parse_observations(CLEAR_LINE + " 1")


def test_load_auth_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv(weather.AUTH_KEY_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
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

    monkeypatch.setattr(weather, "fetch_observation", fake_fetch)
    result = current_weather(datetime(2026, 10, 8, 10, 5, tzinfo=KST), auth_key="key")
    assert result == HourlyWeather("2026-10-08", 9, 15.9, rainfall=0, snowfall=0)
    assert requested == [10, 9, 8]
