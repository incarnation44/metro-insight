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
    kr_holidays,
)
from metro_insight.app.subway_data import Station, load_network, search_stations  # noqa: E402
from metro_insight.data.weather import KST, HourlyWeather, WeatherApiError  # noqa: E402
from metro_insight.model.evaluation import ModelScore  # noqa: E402
from metro_insight.model.predictor import PredictionUnavailable  # noqa: E402

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
    assert values.is_holiday is (day in kr_holidays(day.year))  # 날짜로 자동 판정
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


def test_weekday_is_read_only(panel, network):
    """요일은 직접 고를 수 없다 (날짜에서만 자동 결정)."""
    panel.set_station(station(network, "강남"))
    assert not panel._weekday.isEnabled()
    panel._date.setDate(QDate(2026, 10, 5))  # 월
    panel._weekday.setCurrentIndex(2)  # 코드로 바꿔도 예측은 날짜를 따른다
    values = panel.prediction_input()
    assert (values.weekday, values.is_weekend) == (0, False)


def test_holiday_follows_date(panel, network):
    """공휴일은 날짜에서 자동으로 정해진다 (체크박스 없음)."""
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2026, 10, 3))  # 개천절
    assert panel.prediction_input().is_holiday
    panel._date.setDate(QDate(2026, 10, 7))  # 평일
    assert not panel.prediction_input().is_holiday
    panel.reset_to_now(load_weather=False)
    today = panel._date.date().toPython()
    assert panel.prediction_input().is_holiday is (today in kr_holidays(today.year))


def test_weather_sets_rain_and_snow(panel, network):
    """강수·눈은 날씨 관측값에서 자동으로 정해진다 (체크박스 없음)."""
    panel.set_station(station(network, "강남"))
    panel.apply_weather(SNOWY)
    values = panel.prediction_input()
    assert values.rainfall and values.snowfall

    panel.apply_weather(SUNNY)
    values = panel.prediction_input()
    assert not values.rainfall and not values.snowfall


def test_manual_checkboxes_are_gone(panel):
    """주말·공휴일·강수·눈 체크박스는 화면에서 제거됐다."""
    for name in ("_weekend", "_holiday", "_rainfall", "_snowfall"):
        assert not hasattr(panel, name)


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


def test_changing_date_resets_edited_weather(app, network):
    """날짜를 바꾸면 직접 고친 날씨는 그 날짜의 값이 아니므로 새로 조회한 값으로 바뀐다."""
    panel, requested = recording_panel(network)
    panel.set_station(station(network, "강남"))
    select_hour(panel, 8)
    panel._temperature.setValue(30.0)  # 직접 입력한 기온
    panel._date.setDate(QDate(2025, 1, 9))
    process_events(WEATHER_DEBOUNCE_MS + 200)
    assert requested == [datetime(2025, 1, 9, 8, tzinfo=KST)]  # 그 날짜를 다시 조회한다
    assert panel.prediction_input().temperature == 5.5  # 조회한 값으로 바뀐다
    assert "유지" not in panel._weather_note.text()


def test_changing_hour_resets_edited_weather(app, network):
    """시간을 바꿔도 직접 입력한 날씨는 그 시각의 값이 아니므로 새로 조회한 값으로 바뀐다."""
    panel, requested = recording_panel(network)
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2025, 1, 9))
    select_hour(panel, 8)
    process_events(WEATHER_DEBOUNCE_MS + 200)
    panel._temperature.setValue(30.0)  # 직접 입력한 기온
    select_hour(panel, 9)
    process_events(WEATHER_DEBOUNCE_MS + 200)
    assert requested[-1] == datetime(2025, 1, 9, 9, tzinfo=KST)
    assert panel.prediction_input().temperature == 5.5


def test_future_date_clears_edited_weather(app, network):
    """미래 날짜로 바꾸면 직접 입력한 기온과 이전 비·눈이 남지 않는다."""
    panel, requested = recording_panel(network)
    panel.set_station(station(network, "강남"))
    panel.apply_weather(SNOWY)  # 비·눈이 True인 날씨
    panel._temperature.setValue(30.0)  # 직접 입력한 기온
    future = datetime.now(KST) + timedelta(days=30)
    panel._date.setDate(QDate(future.year, future.month, future.day))
    process_events(WEATHER_DEBOUNCE_MS + 200)
    assert requested == []  # 미래 시각은 조회하지 않는다
    values = panel.prediction_input()
    assert values.temperature is None
    assert (values.rainfall, values.snowfall) == (False, False)


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


def test_future_date_uses_climatology_default(app, network):
    """미래 날짜는 자동조회 대신 평년 기온·비 안 옴을 기본 가정으로 둔다."""
    panel, requested = recording_panel(network)
    panel.set_station(station(network, "강남"))
    future = datetime.now(KST) + timedelta(days=30)
    panel._date.setDate(QDate(future.year, future.month, future.day))
    process_events(WEATHER_DEBOUNCE_MS + 200)
    assert requested == []
    assert "평년" in panel._weather_note.text()
    values = panel.prediction_input()
    assert values.temperature is None  # 예측기가 평년값으로 채운다
    assert (values.rainfall, values.snowfall) == (False, False)


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
    assert all(not panel._result_boarding[m].isHidden() for m in ("A", "B", "C"))
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


def test_predict_unavailable_is_shown(app, network):
    class Unavailable(FakePredictor):
        def predict(self, values):
            raise PredictionUnavailable("학습 데이터에 없는 역입니다.")

    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY), predictor=Unavailable())
    panel.set_station(station(network, "강남"))
    panel.predict()
    assert set(result_texts(panel)[1].values()) == {"-"}
    # 예측은 못 해도 성적표(MAE/RMSE)는 그대로 보여 준다
    assert set(result_texts(panel)[2].values()) != {("-", "-")}
    assert panel._result_note.text() == "학습 데이터에 없는 역입니다."
    assert panel._result_note.property("error") is True


def test_set_date_range_limits_calendar(panel):
    panel.set_date_range("2024-01-08", "2029-12-31")
    assert panel._date.minimumDate() == QDate(2024, 1, 8)
    assert panel._date.maximumDate() == QDate(2029, 12, 31)

    # 범위 밖 날짜는 달력이 잘라 낸다 (직접 넣어도)
    panel._date.setDate(QDate(2030, 1, 1))
    assert panel._date.date() == QDate(2029, 12, 31)
    panel._date.setDate(QDate(2020, 1, 1))
    assert panel._date.date() == QDate(2024, 1, 8)

    # 날짜 안내 문구·장기 추정 배너는 화면에서 제거됐다
    assert not hasattr(panel, "_date_note")
    assert not hasattr(panel, "_long_range_banner")


def test_date_cap_is_six_months(panel):
    """상한(오늘+182일)을 넘는 날짜는 고를 수 없다."""
    end = datetime.now().date() + timedelta(days=182)
    panel.set_date_range("2024-01-08", end.isoformat())
    assert panel._date.maximumDate() == QDate(end.year, end.month, end.day)
    panel._date.setDate(QDate(end.year + 1, 1, 1))
    assert panel._date.date() == QDate(end.year, end.month, end.day)


class HorizonPredictor(FakePredictor):
    """지평 정보까지 돌려주는 예측기 (±MAE 표시 확인용)."""

    def predict(self, values):
        self.received.append(values)
        return {
            "A": 11850.4, "B": 12240.0, "C": 12500.0,
            "horizon_bucket": "32-91",
            "horizon_scores": {
                "A": ModelScore(820.25, 1210.0),
                "B": ModelScore(540.0, 801.5),
                "C": ModelScore(498.0, 760.0),
            },
        }


def test_result_shows_bucket_error_next_to_models(app, network):
    """예상 승차 인원 옆에는 그 지평 구간의 ±MAE 만 붙는다."""
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY),
                         predictor=HorizonPredictor())
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2027, 3, 3))
    panel._temperature.setValue(20.0)
    panel.predict()

    assert panel._result_boarding["A"].text() == "11,850명 ±820명"
    assert panel._result_boarding["C"].text() == "12,500명 ±498명"
    # 날씨 가정 줄은 없고, 미검증 안내 줄은 검증 구간(32~91일)에서 숨겨진다
    assert not hasattr(panel, "_result_weather_note")
    assert panel._result_unverified.isHidden()


def test_pm_tooltip_says_bucket_average(app, network):
    """± 는 구간 평균 MAE 라는 걸 툴팁으로 밝힌다."""
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY),
                         predictor=HorizonPredictor())
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2027, 3, 3))
    panel.predict()

    tip = panel._result_boarding["A"].toolTip()
    assert "평균 MAE" in tip
    assert "32~91일" in tip
    assert "이 한 건" in tip


def test_result_computes_c_without_temperature(app, network):
    """기온을 넣지 않아도 C 는 계산한다 (안내 줄 없이)."""
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY),
                         predictor=HorizonPredictor())
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2027, 3, 3))
    panel.predict()

    assert panel.prediction_input().temperature is None
    assert panel._result_boarding["C"].text() != "-"  # C 를 건너뛰지 않는다


class PastPredictor(HorizonPredictor):
    """과거 날짜: 지평 자체가 없다 (실측 구간)."""

    def predict(self, values):
        self.received.append(values)
        return {"A": 11850.4, "B": 12240.0, "C": 12500.0,
                "horizon_bucket": None, "horizon_scores": {}}


def test_past_date_has_no_pm(app, network):
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY),
                         predictor=PastPredictor())
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2026, 6, 30))
    panel._temperature.setValue(20.0)
    panel.predict()

    assert panel._result_boarding["A"].text() == "11,850명"  # ±없음
    assert panel._result_unverified.isHidden()  # 과거 날짜는 경고 대상이 아니다


class Unverified182Predictor(HorizonPredictor):
    """182일 이후 구간 (Test 성적이 없는 장기 추정)."""

    def predict(self, values):
        self.received.append(values)
        return {"A": 11850.4, "B": 12240.0, "C": 12500.0,
                "horizon_bucket": "182-366", "horizon_scores": {}}


def test_unverified_horizon_shows_warning(app, network):
    """12/29 이후처럼 검증 안 된 구간에는 '미검증' 안내가 붙는다."""
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY),
                         predictor=Unverified182Predictor())
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2027, 1, 15))
    panel.predict()

    assert panel._result_unverified.isHidden() is False
    text = panel._result_unverified.text()
    assert "미검증" in text
    assert "2026-06-30" in text
    assert "199일 뒤" in text  # 2027-01-15 - 2026-06-30
    assert "오차를 측정한 적이 없습니다" in panel._result_unverified.toolTip()


def test_unverified_horizon_hides_pm(app, network):
    """미검증 구간에는 ± 를 붙이지 않는다 (오차를 모른다)."""
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY),
                         predictor=Unverified182Predictor())
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2027, 1, 15))
    panel.predict()

    assert panel._result_boarding["A"].text() == "11,850명"  # ± 없음
    assert panel._result_boarding["A"].toolTip() == ""


class UnverifiedPredictor(HorizonPredictor):
    """지평은 있는데 검증 점수가 없는 경우 (367+ — 실측 구간 없음)."""

    def predict(self, values):
        self.received.append(values)
        return {"A": 11850.4, "B": 12240.0, "C": 12500.0,
                "horizon_bucket": "367+", "horizon_scores": {}}


def test_no_scores_means_no_pm(app, network):
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY),
                         predictor=UnverifiedPredictor())
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2029, 7, 1))
    panel._temperature.setValue(20.0)
    panel.predict()

    assert panel._result_boarding["A"].text() == "11,850명"  # 점수가 없으니 ±없음
    assert panel._result_unverified.isHidden() is False  # 미검증 안내는 뜬다


class PredictorWithData(FakePredictor):
    """지정한 (호선, 역)만 데이터가 있다고 답하는 예측기."""

    def __init__(self, known: set[tuple[str, str]]):
        super().__init__()
        self.known = known

    def has_data(self, line, station):
        return (line, station) in self.known


def test_transfer_station_defaults_to_line_with_data(app, network):
    """까치산은 데이터가 5호선에만 있다. 기본 2호선을 잡아 '예측 불가'로 오인하던 것을 막는다."""
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY),
                         predictor=PredictorWithData({("5호선", "까치산")}))
    panel.set_station(station(network, "까치산"))

    texts = [panel._line.itemText(i) for i in range(panel._line.count())]
    assert texts == ["2호선 (데이터 없음)", "5호선"]
    assert panel._line.currentText() == "5호선"  # 기본은 데이터가 있는 첫 호선
    assert panel.prediction_input().line == "5호선"
    assert not panel._line.model().item(0).isEnabled()  # 2호선은 고를 수 없다
    assert panel._line.model().item(1).isEnabled()


def test_station_without_any_data_keeps_all_lines(app, network):
    """신내는 어느 호선에도 데이터가 없다. 전부 고를 수 있게 두고 예측 때 안내문구가 뜨게 한다."""
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY),
                         predictor=PredictorWithData(set()))
    panel.set_station(station(network, "신내"))

    texts = [panel._line.itemText(i) for i in range(panel._line.count())]
    assert texts == ["6호선"]
    assert panel._line.model().item(0).isEnabled()
    assert panel.prediction_input().line == "6호선"


def test_holiday_is_detected_from_date(panel, network):
    """공휴일은 고른 날짜로 자동 판정한다 (학습 때 쓴 공휴일 목록과 같은 것)."""
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2026, 10, 3))  # 개천절
    assert panel.prediction_input().is_holiday is True
    panel._date.setDate(QDate(2026, 10, 5))  # 개천절 대체 휴일
    assert panel.prediction_input().is_holiday is True
    panel._date.setDate(QDate(2026, 10, 7))  # 평일
    assert panel.prediction_input().is_holiday is False


def test_holiday_years_up_to_2029(panel, network):
    """서빙 기간 끝(2029년)까지 공휴일을 자동 판정한다 (학습 때와 같은 holidays 목록)."""
    panel.set_station(station(network, "강남"))
    for year in (2027, 2028, 2029):
        assert kr_holidays(year)  # 그 해 공휴일 목록이 비어 있지 않다
    panel._date.setDate(QDate(2029, 1, 1))  # 신정
    assert panel.prediction_input().is_holiday is True
    panel._date.setDate(QDate(2028, 3, 1))  # 삼일절
    assert panel.prediction_input().is_holiday is True


def test_holiday_is_marked_in_red(panel):
    """공휴일이면 날짜 옆에 빨간색 표시가 붙는다."""
    panel._date.setDate(QDate(2026, 10, 3))  # 개천절
    assert panel._holiday_tag.isHidden() is False
    assert "개천절" in panel._holiday_tag.text()
    assert panel._holiday_tag.property("error") is True
    panel._date.setDate(QDate(2026, 10, 7))  # 평일
    assert panel._holiday_tag.isHidden() is True


def test_weather_failure_clears_stale_values(panel, network):
    """날씨를 못 받으면 지난 시각의 값이 남지 않게 비운다."""
    panel.set_station(station(network, "강남"))
    panel.apply_weather(SNOWY)
    assert panel._temperature.value() == -3.2
    assert panel.prediction_input().rainfall is True
    panel._on_weather_failed("테스트 오류")
    assert panel._temperature.value() == panel._temperature.minimum()  # 미입력
    assert panel.prediction_input().rainfall is False


def test_weather_failure_keeps_manual_values(panel):
    """직접 입력한 값은 날씨를 못 받아도 그대로 둔다."""
    panel.apply_weather(SUNNY)
    panel._temperature.setValue(25.0)  # 직접 입력
    panel._on_weather_failed("테스트 오류")
    assert panel._temperature.value() == 25.0


def test_station_without_lines_does_not_crash(panel):
    """호선 정보가 없는 역에서도 예외 없이 처리한다."""
    panel.set_station(Station(name="테스트역", lat=0.0, lon=0.0, lines=()))
    assert panel._line.currentIndex() == -1

