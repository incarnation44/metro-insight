import os
import threading
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 화면 없이 위젯 테스트

from PySide6.QtCore import QDate, QEventLoop, QTimer  # noqa: E402
from PySide6.QtGui import QInputMethodEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from metro_insight.app.station_panel import (  # noqa: E402
    WEATHER_DEBOUNCE_MS,
    StationPanel,
    WeatherLoader,
    hour_slot,
)
from metro_insight.app.subway_data import load_network, search_stations  # noqa: E402
from metro_insight.data.weather import KST, HourlyWeather, WeatherApiError  # noqa: E402
from metro_insight.model.evaluation import ModelScore  # noqa: E402

SUNNY = HourlyWeather("2026-10-12", 9, 21.6, rainfall=0, snowfall=0)
SNOWY = HourlyWeather("2026-01-20", 7, -3.2, rainfall=1, snowfall=1)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def network():
    return load_network()


def select_hour(panel, hour):
    panel._hour.setCurrentIndex(panel._hour.findData(hour))


@pytest.fixture
def panel(app, network):
    return StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY))


def station(network, name):
    return next(s for s in network[1] if s.name == name)


def process_events(ms=200):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def test_search_by_prefix(network):
    names = [s.name for s in search_stations(network[1], "강")]
    assert "강남구청" in names and "강남" in names
    assert all(n.startswith("강") for n in names)
    assert names == sorted(names)
    assert search_stations(network[1], " 강남구 ")[0].name == "강남구청"
    assert search_stations(network[1], "") == []
    assert search_stations(network[1], "남구청") == []  # 가운데 일치는 찾지 않는다


def result_names(panel):
    results = panel._results
    return [results.item(i).data(256).name for i in range(results.count()) if results.item(i).data(256)]


def compose(panel, preedit, commit=""):
    """한글 입력기 이벤트: commit 은 확정된 글자, preedit 는 조합 중인 글자."""
    event = QInputMethodEvent(preedit, [])
    event.setCommitString(commit)
    QApplication.sendEvent(panel._search, event)
    process_events(50)


def test_search_while_composing_korean(panel):
    compose(panel, "강")  # 아직 조합 중: text() 는 비어 있다
    assert panel._search.text() == ""
    assert "강남구청" in result_names(panel) and "강변" in result_names(panel)

    compose(panel, "나", commit="강")  # '강' 확정, '나' 조합 중
    assert result_names(panel) == []  # '강나'로 시작하는 역은 없다
    compose(panel, "남")
    assert result_names(panel) == ["강남", "강남구청"]

    compose(panel, "", commit="남")  # 조합 끝
    assert panel._search.text() == "강남"
    assert result_names(panel) == ["강남", "강남구청"]


def test_no_input_before_station_selected(panel):
    assert panel.prediction_input() is None


def test_defaults_are_now(panel, network):
    panel.set_station(station(network, "강남구청"))
    values = panel.prediction_input()
    now = datetime.now(KST)
    day = now - timedelta(hours=4)  # 00~04시는 전날 업무일
    assert values.station == "강남구청"
    assert values.line == "7호선"
    assert values.date == day.strftime("%Y-%m-%d")
    assert values.hour == hour_slot(now.hour)
    assert values.weekday == day.weekday()
    assert values.is_weekend == (day.weekday() >= 5)
    assert values.is_holiday is False  # 공휴일은 계산하지 않는다
    assert values.temperature is None  # 날씨를 받기 전
    assert not hasattr(values, "boarding")  # 승차인원은 입력받지 않는다


def test_hour_choices_follow_data_time_slots(panel):
    labels = [panel._hour.itemText(i) for i in range(panel._hour.count())]
    assert labels == ["04~06시"] + [f"{hour:02d}시" for hour in range(6, 25)]


def test_hour_slot():
    assert [hour_slot(hour) for hour in (0, 3, 4, 5, 6, 23)] == [24, 24, 4, 4, 6, 23]


def test_hour_24_is_next_day_midnight(panel):
    panel._date.setDate(QDate(2025, 12, 31))
    select_hour(panel, 24)
    assert panel.selected_time() == datetime(2026, 1, 1, 0, tzinfo=KST)
    select_hour(panel, 4)
    assert panel.selected_time() == datetime(2025, 12, 31, 4, tzinfo=KST)


def test_date_change_updates_weekday(panel, network):
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2026, 10, 5))  # 월
    values = panel.prediction_input()
    assert (values.date, values.weekday, values.is_weekend) == ("2026-10-05", 0, False)

    panel._date.setDate(QDate(2026, 10, 10))  # 토
    values = panel.prediction_input()
    assert (values.weekday, values.is_weekend) == (5, True)

    panel._weekday.setCurrentIndex(2)  # 요일을 직접 바꾸면 주말 여부도 따라간다
    assert not panel.prediction_input().is_weekend


def test_holiday_is_manual_input(panel, network):
    panel.set_station(station(network, "강남"))
    panel._holiday.setChecked(True)
    assert panel.prediction_input().is_holiday
    panel._date.setDate(QDate(2026, 10, 9))  # 날짜를 바꿔도 직접 체크한 값을 유지한다
    panel.reset_to_now(load_weather=False)
    assert panel.prediction_input().is_holiday


def test_snow_counts_as_rain(panel, network):
    panel.set_station(station(network, "강남"))
    panel._snowfall.setChecked(True)
    values = panel.prediction_input()
    assert values.rainfall and values.snowfall
    assert not panel._rainfall.isEnabled()

    panel._snowfall.setChecked(False)
    assert panel._rainfall.isEnabled()


def test_line_choice_for_transfer_station(panel, network):
    panel.set_station(station(network, "고속터미널"))
    assert [panel._line.itemText(i) for i in range(panel._line.count())] == ["3호선", "7호선"]
    assert panel._line.isEnabled()
    panel._line.setCurrentIndex(1)
    assert panel.prediction_input().line == "7호선"

    panel.set_station(station(network, "강남구청"))  # 한 호선만 지나면 고를 수 없다
    assert not panel._line.isEnabled()


def test_temperature_input(panel, network):
    panel.set_station(station(network, "강남"))
    panel._temperature.setValue(-5.3)
    assert panel.prediction_input().temperature == -5.3


def test_weather_fills_fields(panel, network):
    panel.set_station(station(network, "강남"))
    panel.apply_weather(SNOWY)
    values = panel.prediction_input()
    assert (values.temperature, values.rainfall, values.snowfall) == (-3.2, True, True)


def test_loaded_weather_fills_fields(panel, network):
    panel.set_station(station(network, "강남"))
    panel.refresh_weather()
    process_events()
    assert panel.prediction_input().temperature == 21.6
    assert "09시 관측" in panel._weather_note.text()


def test_loaded_weather_keeps_user_edits(panel, network):
    panel.set_station(station(network, "강남"))
    panel.refresh_weather()
    panel._temperature.setValue(30.0)  # 날씨를 받는 사이에 직접 입력
    process_events()
    assert panel.prediction_input().temperature == 30.0
    assert "유지" in panel._weather_note.text()


def recording_panel(network):
    """받은 날씨 대신 요청한 시각을 기록하는 패널."""
    requested = []

    def fetch(tm):
        requested.append(tm)
        return HourlyWeather(tm.strftime("%Y-%m-%d"), tm.hour, 5.5, rainfall=1, snowfall=0)

    return StationPanel(*network, weather=WeatherLoader(fetch)), requested


def test_changing_date_and_hour_loads_that_weather(app, network):
    panel, requested = recording_panel(network)
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2025, 1, 9))
    select_hour(panel, 7)
    select_hour(panel, 8)  # 연달아 바꾸면 마지막 값만 받는다
    process_events(WEATHER_DEBOUNCE_MS + 200)
    assert requested == [datetime(2025, 1, 9, 8, tzinfo=KST)]
    values = panel.prediction_input()
    assert (values.temperature, values.rainfall) == (5.5, True)
    assert panel._weather_note.text() == "기상청 서울 2025-01-09 08시 관측"


def test_changing_date_overwrites_edited_weather(app, network):
    panel, requested = recording_panel(network)
    panel.set_station(station(network, "강남"))
    panel._temperature.setValue(30.0)
    panel._date.setDate(QDate(2025, 1, 9))
    process_events(WEATHER_DEBOUNCE_MS + 200)
    assert panel.prediction_input().temperature == 5.5


def test_fallback_to_earlier_hour_is_noted(app, network):
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY))  # 2026-10-12 09시
    panel._date.setDate(QDate(2026, 10, 12))
    select_hour(panel, 10)
    panel.refresh_weather()
    process_events()
    assert "직전 관측값" in panel._weather_note.text()


def test_changing_date_drops_pending_response(app, network):
    slow = threading.Event()

    def fetch(tm):
        slow.wait(1)
        return SNOWY

    panel = StationPanel(*network, weather=WeatherLoader(fetch))
    panel._date.setDate(QDate(2026, 1, 20))
    select_hour(panel, 7)
    panel.refresh_weather()
    panel._date.setDate(QDate(2025, 1, 9))  # 응답이 오기 전에 날짜를 바꾼다
    slow.set()
    process_events(100)  # 새 조회를 시작하기 전에 이전 응답이 도착
    assert panel._temperature.value() == panel._temperature.minimum()


def test_future_time_is_not_loaded(app, network):
    panel, requested = recording_panel(network)
    panel.set_station(station(network, "강남"))
    tomorrow = datetime.now(KST) + timedelta(days=1)
    panel._date.setDate(QDate(tomorrow.year, tomorrow.month, tomorrow.day))
    process_events(WEATHER_DEBOUNCE_MS + 200)
    assert requested == []
    assert "미래" in panel._weather_note.text()
    assert panel.prediction_input().temperature is None


def test_reset_to_now_loads_once(app, network):
    panel, requested = recording_panel(network)
    panel._date.setDate(QDate(2025, 1, 9))
    panel.reset_to_now()
    process_events(WEATHER_DEBOUNCE_MS + 200)
    assert requested == [panel.selected_time()]  # 날짜·시를 바꾸며 잡힌 조회는 취소된다


def test_reset_button_loads_current_weather(app, network):
    panel, requested = recording_panel(network)
    panel._date.setDate(QDate(2025, 1, 9))
    reset = next(b for b in panel.findChildren(QPushButton) if b.text() == "현재로")
    reset.click()
    process_events(WEATHER_DEBOUNCE_MS + 200)
    assert requested == [panel.selected_time()]
    assert panel.selected_time().date() == datetime.now(KST).date()


def test_weather_failure_is_shown(app, network):
    def fail(tm):
        raise WeatherApiError("인증키가 없다")

    panel = StationPanel(*network, weather=WeatherLoader(fail))
    panel.refresh_weather()
    process_events()
    assert "인증키가 없다" in panel._weather_note.text()
    assert panel._weather_note.property("error") is True


def test_loader_drops_stale_response(app):
    responses = iter([SNOWY, SUNNY])
    loader = WeatherLoader(lambda tm: next(responses))
    received = []
    loader.loaded.connect(received.append)
    tm = datetime(2026, 10, 12, 9, tzinfo=KST)
    loader.load(tm)
    loader.load(tm)  # 첫 응답이 오기 전에 다시 요청
    process_events()
    assert len(received) == 1


class FakePredictor:
    def __init__(self):
        self.received = []

    def predict(self, values):
        self.received.append(values)
        return {"A": 11850.4, "B": 12240, "C": 12500, "D": 12610, "E": 12580}

    def scores(self):
        return {
            "A": ModelScore(820.25, 1210.0),
            "B": ModelScore(540.0, 801.5),
            "C": ModelScore(498.0, 760.0),
            "D": ModelScore(455.5, 702.0),
            "E": ModelScore(462.0, 698.25),
        }


def result_texts(panel):
    conditions = {name: label.text() for name, label in panel._result_conditions.items()}
    boarding = {model: label.text() for model, label in panel._result_boarding.items()}
    scores = {model: (mae.text(), rmse.text()) for model, (_, mae, rmse) in panel._result_scores.items()}
    return conditions, boarding, scores


def test_predict_shows_model_results(app, network):
    predictor = FakePredictor()
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY), predictor=predictor)
    panel.predict()  # 역을 고르기 전에는 아무것도 하지 않는다
    assert predictor.received == [] and panel._result.isHidden()

    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2025, 9, 1))
    select_hour(panel, 8)
    panel._temperature.setValue(23.1)
    panel.predict()

    assert predictor.received == [panel.prediction_input()]
    assert not panel._result.isHidden()
    conditions, boarding, scores = result_texts(panel)
    assert conditions == {"역": "강남 (2호선)", "날짜": "2025-09-01 (월)", "시간": "08시", "기온": "23.1℃"}
    assert boarding == {"A": "11,850명", "B": "12,240명", "C": "12,500명", "D": "12,610명", "E": "12,580명"}
    assert scores == {
        "A": ("820.2", "1,210.0"),
        "B": ("540.0", "801.5"),
        "C": ("498.0", "760.0"),
        "D": ("455.5", "702.0"),
        "E": ("462.0", "698.2"),
    }
    assert [m for m, (name, *_) in panel._result_scores.items() if name.property("best")] == ["D"]
    assert panel._result_note.isHidden()


def test_predict_without_model(panel, network):
    panel.set_station(station(network, "강남"))
    panel.predict()
    conditions, boarding, scores = result_texts(panel)
    assert conditions["역"] == "강남 (2호선)"
    assert set(boarding.values()) == {"-"}
    assert set(scores.values()) == {("-", "-")}
    assert "모델이 아직 없어" in panel._result_note.text()


def test_predict_error_is_shown(app, network):
    class Broken(FakePredictor):
        def predict(self, values):
            raise KeyError("강남")

    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY), predictor=Broken())
    panel.set_station(station(network, "강남"))
    panel.predict()
    assert set(result_texts(panel)[1].values()) == {"-"}
    assert "KeyError" in panel._result_note.text()
    assert panel._result_note.property("error") is True
