"""시작 성능과 모델 로딩 상태 검증.

* 첫 창이 뜨기까지 걸리는 시간을 잰다 (모듈 import 는 이미 끝난 상태에서 창 구성·첫 그리기까지).
* 모델을 배경에서 읽는 동안·읽지 못했을 때 화면이 어떻게 되는지 본다.
* 예측 결과 화면의 위젯 속성 이름과 문구가 그대로 남아 있는지 확인한다.
"""
import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 화면 없이 위젯 검사

from PySide6.QtCore import QDate, QEventLoop, QObject, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from metro_insight.app import __main__ as app_main  # noqa: E402
from metro_insight.app.main_window import (  # noqa: E402
    MainWindow,
    PredictorLoader,
    warm_up,
)
from metro_insight.app.station_panel import (  # noqa: E402
    MODEL_ROWS,
    SEARCH_ROW_HEIGHT,
    StationPanel,
    WeatherLoader,
)
from metro_insight.app.styles import apply_style  # noqa: E402
from metro_insight.app.subway_data import load_network  # noqa: E402
from metro_insight.data.weather import HourlyWeather  # noqa: E402
from metro_insight.model.evaluation import ModelScore  # noqa: E402
from metro_insight.model.predictor import climatology_temperature  # noqa: E402

FIRST_WINDOW_LIMIT = 2.0  # 첫 화면(창 표시)까지 허용하는 시간(초)

SUNNY = HourlyWeather("2026-10-12", 9, 21.6, rainfall=0, snowfall=0)

# 예측 결과 화면이 쓰는 위젯 속성 이름 (화면 구조를 바꾸지 않기 위해 고정)
FROZEN_ATTRIBUTES = (
    "_result",
    "_result_boarding",
    "_result_scores",
    "_result_note",
    "_result_unverified",
    "_result_boarding_box",
    "_result_scores_box",
    "_scores_header",
    "_boarding_title",
    "_temperature",
    "_weekday",
    "_holiday_tag",
    "_weather_note",
    "_line",
    "_date",
    "_hour",
    "_rain",
    "_snow",
    "_search",
    "_results",
)

# 예측 결과 화면이 쓰는 문구 (11자리, 이름 → 화면에 보이는 글자). 목록을 줄이면 검사가
# 조용히 약해지므로 아래 테스트에서 개수를 함께 확인한다.
FROZEN_TEXTS = {
    "boarding_a": "11,850명",
    "score_a_rmse": "1,210.0",
    "boarding_title": "예상 승차 인원",
    "empty": "-",
    "unverified": "테스트 데이터 없음\n(참고용 숫자)",
    "partial": "테스트 데이터는 136일까지만\n(일부만 확인한 참고용 숫자)",
    "weather_note": "기상청 서울 2025-01-09 08시 관측",
    "line_labels": ("2호선 (데이터 없음)", "5호선"),
    "reset_button": "현재로",
    "climatology": "평년 {value:.1f}℃",
    "weekday": "월요일",
}


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    apply_style(application)
    return application


@pytest.fixture(scope="module")
def network():
    return load_network()


def station(network, name):
    return next(s for s in network[1] if s.name == name)


def process_events(ms=50):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


class ResultsPredictor:
    """정해진 숫자를 돌려주는 예측기 (화면 문구 확인용)."""

    def __init__(self, lines_with_data=None, date_range=("2024-01-08", "2029-07-01")):
        self.received = []
        self._lines_with_data = lines_with_data
        self._date_range = date_range

    def predict(self, values):
        self.received.append(values)
        return {"A": 11850.4, "B": 12240.0, "C": 12500.0, "D": 12610.0, "E": 12580.0}

    def scores(self):
        return {
            "A": ModelScore(820.25, 1210.0),
            "B": ModelScore(540.0, 801.5),
            "C": ModelScore(498.0, 760.0),
            "D": ModelScore(455.5, 702.0),
            "E": ModelScore(462.0, 698.25),
        }

    def has_data(self, line, station):
        if self._lines_with_data is None:
            return True
        return (line, station) in self._lines_with_data

    def date_range(self):
        return self._date_range


class HorizonPredictor(ResultsPredictor):
    """지평 구간을 함께 돌려주는 예측기 (미검증 안내 확인용)."""

    def __init__(self, bucket):
        super().__init__()
        self._bucket = bucket

    def predict(self, values):
        self.received.append(values)
        return {"A": 11850.4, "B": 12240.0, "C": 12500.0, "horizon_bucket": self._bucket}


class FakeLoader(QObject):
    """화면이 배경 로딩을 어떻게 다루는지 보기 위한 로더 (실제 모델을 읽지 않는다)."""

    ready = Signal(object)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.starts = []

    def start(self, lines=(), stations=()):
        self.starts.append((tuple(lines), tuple(stations)))


def make_panel(network, predictor=None):
    return StationPanel(
        *network, weather=WeatherLoader(lambda tm: SUNNY), predictor=predictor
    )


# ---------- 첫 창 표시 시간 ----------


def test_first_window_appears_within_limit(app):
    """창을 만들고 첫 그리기까지가 2초를 넘지 않는다 (모델을 읽느라 미루지 않는다).

    여기서 재는 값은 화면을 띄우는 데 실제로 쓴 CPU 시간이다. 다른 프로그램이 CPU 를 쓰면
    벽시계 시간은 늘어나므로, 이 스레드(화면 스레드)가 쓴 시간으로 본다. 세 번 재어 가운데
    값을 쓴다.

    실제 기동(프로세스 시작 → 창 표시) 측정값은 따로 있다. 안내 창은 0.283초에 떴고(CPU 부하
    27%), 본 창은 부하 100%에서 4.557초, 그때 이 스레드가 쓴 시간은 2.016초였다. 본 창 2.0초
    게이트는 부하 25% 이하에서 재야 하는데 세 번 모두 82~100%여서 조건을 맞추지 못했다.
    그래서 게이트 판정은 미판정(저부하 미확보)이고, 이 테스트는 창 구성 자체의 비용만 본다.
    """
    cpu_times, wall_times = [], []
    for _ in range(3):
        started_cpu, started_wall = time.thread_time(), time.perf_counter()
        window = MainWindow(loader=FakeLoader())
        window.resize(1000, 700)
        window.show()
        app.processEvents()
        cpu_times.append(time.thread_time() - started_cpu)
        wall_times.append(time.perf_counter() - started_wall)
        window.close()
        app.processEvents()

    elapsed = sorted(cpu_times)[1]
    assert elapsed < FIRST_WINDOW_LIMIT, (
        f"첫 창 표시까지 CPU {elapsed:.3f}초 "
        f"(세 번 CPU {[round(v, 3) for v in cpu_times]}, 실제 시간 {[round(v, 3) for v in wall_times]})"
    )


def test_splash_screen_is_ready_before_main_window(app):
    """안내 창은 화면 구성 전에 뜰 수 있게 스스로 서고, 주 창이 뜨면 닫힌다."""
    splash = app_main.SplashScreen()
    splash.show()
    app.processEvents()
    assert splash.isVisible()
    assert splash.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert splash.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert splash.size() == splash.minimumSize() == splash.maximumSize()  # 크기가 흔들리지 않는다

    window = MainWindow(loader=FakeLoader())
    splash.finish(window)
    assert splash.isHidden()
    window.close()


def test_splash_closes_after_window_is_shown(app):
    """안내 창은 주 창이 뜬 뒤에 닫히고, 주 창은 화면에 그대로 남는다."""
    splash = app_main.SplashScreen()
    splash.show()
    app.processEvents()
    assert splash.isVisible()

    window = MainWindow(loader=FakeLoader())
    window.show()
    app.processEvents()

    splash.finish(window)
    assert splash.isHidden()
    assert window.isVisible()  # 주 창을 닫아 버리지 않는다
    window.close()


# ---------- 배경 로딩 ----------


def test_predictor_load_starts_after_window_is_built(app):
    """창을 만든 직후에는 읽기를 시작하지 않는다 (첫 화면을 먼저 그린다)."""
    loader = FakeLoader()
    window = MainWindow(loader=loader)
    assert loader.starts == []
    assert window.station_panel.predictor_state == "loading"
    assert window.station_panel._predictor is None
    assert not window._status_chip.isHidden()
    assert window._status_chip.text() == "모델을 불러오는 중…"

    process_events()
    assert len(loader.starts) == 1
    window.close()


def test_ready_predictor_is_applied(app, network):
    """읽기가 끝나면 날짜 범위가 달력에 반영되고 안내가 사라진다."""
    loader = FakeLoader()
    window = MainWindow(loader=loader)
    panel = window.station_panel
    panel.set_station(station(network, "강남"))
    assert panel._predictor is None  # 읽기 전에는 예측기가 없다

    predictor = ResultsPredictor()
    loader.ready.emit(predictor)

    assert window.predictor is predictor
    assert panel.predictor_state == "ready"
    assert panel._predictor is predictor
    assert panel._loading_note.isHidden()
    assert panel._retry_button.isHidden()
    assert window._status_chip.isHidden()
    assert panel._date.minimumDate() == QDate(2024, 1, 8)
    assert panel._date.maximumDate() == QDate(2029, 7, 1)

    # 읽기가 끝나면 그 예측기로 예측 결과가 나온다
    panel.predict()
    assert panel._result_boarding["A"].text() == "11,850명"
    window.close()


def test_failed_load_shows_reason_and_can_retry(app):
    """읽지 못하면 사유를 남기고, 다시 시도할 수 있게 한다."""
    loader = FakeLoader()
    window = MainWindow(loader=loader)
    panel = window.station_panel
    process_events()

    reason = "FileNotFoundError: models/cal_A.joblib"
    loader.failed.emit(reason)

    assert panel.predictor_state == "failed"
    assert not panel._loading_note.isHidden()
    assert reason in panel._loading_note.text()
    assert panel._loading_note.property("error") is True
    assert not panel._retry_button.isHidden()
    assert not window._status_chip.isHidden()
    assert reason in window._status_chip.toolTip()

    panel._retry_button.click()
    assert panel.predictor_state == "loading"
    assert panel._retry_button.isHidden()
    assert len(loader.starts) == 2
    window.close()


def test_predict_while_loading_keeps_previous_values(app, network):
    """읽는 중에 예측을 눌러도 화면의 직전 값을 지우지 않고 기다리라고만 알린다."""
    predictor = ResultsPredictor()
    panel = make_panel(network, predictor)
    panel.set_station(station(network, "강남"))
    panel.predict()
    assert panel._result_boarding["A"].text() == "11,850명"
    called = len(predictor.received)

    panel.begin_predictor_load()  # 모델을 다시 읽기 시작한 상황
    panel.predict()

    assert panel._result_boarding["A"].text() == "11,850명"  # 직전 값 유지
    assert len(predictor.received) == called  # 예측기를 부르지 않는다
    assert not panel._loading_note.isHidden()
    assert "불러오는 중" in panel._loading_note.text()


def test_predict_while_loading_does_not_touch_model_module(app, network, monkeypatch):
    """읽는 중에 예측을 눌러도 모델 모듈을 읽지 않는다 (읽는 데 몇 초 걸려 화면이 멈춘다)."""
    predictor = ResultsPredictor()
    panel = make_panel(network, predictor)
    panel.set_station(station(network, "강남"))
    panel.predict()
    before = panel._result_boarding["A"].text()
    called = len(predictor.received)

    panel.begin_predictor_load()

    def refuse():
        raise AssertionError("읽는 중에 모델 모듈을 읽었다")

    monkeypatch.setattr("metro_insight.app.station_panel._model_api", refuse)
    panel.predict()  # 모델 모듈을 읽으려 하면 여기서 실패한다

    assert panel._result_boarding["A"].text() == before
    assert len(predictor.received) == called
    assert not panel._loading_note.isHidden()


def test_future_date_while_loading_defers_climatology(app, network, monkeypatch):
    """읽는 중에 미래 날짜로 바꿔도 모델 모듈을 읽지 않고, 다 읽은 뒤에 평년 기온을 채운다."""
    panel = make_panel(network)
    panel.set_station(station(network, "강남"))
    panel.begin_predictor_load()

    def refuse():
        raise AssertionError("읽는 중에 모델 모듈을 읽었다")

    monkeypatch.setattr("metro_insight.app.station_panel._model_api", refuse)
    panel._date.setDate(QDate(2027, 3, 3))
    panel.refresh_weather()  # 미래 시각이라 평년값 자리로 간다

    assert panel._temperature.specialValueText() == "미입력"  # 아직 채우지 않는다
    assert "평년값" in panel._weather_note.text()

    monkeypatch.undo()  # 실제 모듈을 다시 쓰도록 되돌린다
    panel.set_predictor(ResultsPredictor())

    month = panel._date.date().toPython().month
    expected = FROZEN_TEXTS["climatology"].format(value=climatology_temperature(month))
    assert panel._temperature.specialValueText() == expected
    assert panel._temperature.property("climatology") is True


def test_set_predictor_keeps_chosen_line(app, network):
    """모델을 다 읽어 호선 목록을 다시 채워도, 사용자가 고른 호선을 첫 호선으로 되돌리지 않는다."""
    panel = make_panel(network)
    panel.set_station(station(network, "교대"))
    assert panel._line.count() > 1  # 호선이 둘 이상인 역에서 확인한다
    chosen_index = panel._line.count() - 1  # 첫 호선이 아닌 다른 호선을 고른다
    panel._line.setCurrentIndex(chosen_index)
    chosen = panel._line.currentData()

    panel.set_predictor(ResultsPredictor())  # 모든 호선에 데이터가 있는 예측기가 붙는다

    assert panel.predictor_state == "ready"
    assert panel._line.currentData() == chosen
    assert panel._line.currentText() == panel._lines[chosen].name


def test_loader_reads_predictor_off_ui_thread(app):
    """예측기 생성은 UI 스레드가 아닌 곳에서 이뤄진다."""
    created_in = []
    loader = PredictorLoader(factory=lambda: created_in.append(threading.current_thread().name))
    received = []
    loader.ready.connect(received.append)

    loader.start()
    process_events(300)

    assert received, "준비 신호가 오지 않음"
    assert created_in and created_in[0] != threading.main_thread().name


def test_warm_up_touches_past_and_future(app, network):
    """워밍업은 데이터 안·밖 날짜를 한 번씩 계산해 첫 예측을 빠르게 한다."""

    class Recording(ResultsPredictor):
        def has_data(self, line, station):
            return (line, station) == ("5호선", "여의도")

    predictor = Recording()
    warm_up(predictor, *network)

    days = [values.date for values in predictor.received]
    assert len(days) == 2
    assert days[0] < days[1]
    assert all(values.line == "5호선" and values.station == "여의도"
               for values in predictor.received)


def test_warm_up_skips_when_no_station_has_data(app, network):
    predictor = ResultsPredictor(lines_with_data=set())
    warm_up(predictor, *network)
    assert predictor.received == []


# ---------- 화면 구조 고정 ----------


def test_result_widgets_keep_their_names(app, network):
    panel = make_panel(network)
    missing = [name for name in FROZEN_ATTRIBUTES if not hasattr(panel, name)]
    assert not missing, f"없어진 위젯 속성: {missing}"
    assert set(panel._result_boarding) == {"A", "B", "C", "D", "E"}
    assert set(panel._result_scores) == {"A", "B", "C", "D", "E"}


def test_frozen_lists_are_not_shortened(app):
    """고정 목록의 개수를 함께 못 박는다 (항목을 지우면 검사가 조용히 약해진다)."""
    assert len(FROZEN_ATTRIBUTES) == 20
    assert len(set(FROZEN_ATTRIBUTES)) == 20
    assert len(FROZEN_TEXTS) == 11
    assert FROZEN_TEXTS["line_labels"] == ("2호선 (데이터 없음)", "5호선")


def test_model_rows_match_model_module(app):
    """화면 쪽 모델 목록이 모델 모듈의 정의와 같은지 확인한다 (한쪽만 고치면 어긋난다)."""
    from metro_insight.model.predictor import MODELS  # 비교할 때 읽는다

    assert dict(MODEL_ROWS) == MODELS
    assert tuple(model for model, _ in MODEL_ROWS) == tuple(MODELS)  # 카드 순서도 같다


def test_result_texts_are_unchanged(app, network):
    """결과 카드의 숫자 서식·안내 문구가 그대로다."""
    panel = make_panel(network, ResultsPredictor())
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2025, 9, 1))
    panel.predict()

    boarding = {model: label.text() for model, label in panel._result_boarding.items()}
    scores = {
        model: (mae.text(), rmse.text())
        for model, (_name, mae, rmse) in panel._result_scores.items()
    }
    assert boarding == {"A": "11,850명", "B": "12,240명", "C": "12,500명",
                        "D": "12,610명", "E": "12,580명"}
    assert scores["A"] == ("820.2", FROZEN_TEXTS["score_a_rmse"])
    assert panel._boarding_title.text() == FROZEN_TEXTS["boarding_title"]
    assert panel._weekday.text() == FROZEN_TEXTS["weekday"]

    # 모델이 없을 때는 '-' 로 비운다
    empty = make_panel(network)
    empty.set_station(station(network, "강남"))
    empty.predict()
    assert set(label.text() for label in empty._result_boarding.values()) == {
        FROZEN_TEXTS["empty"]
    }


def test_unverified_notes_are_unchanged(app, network):
    panel = make_panel(network, HorizonPredictor("182-366"))
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2027, 1, 15))
    panel.predict()
    assert panel._result_unverified.text() == FROZEN_TEXTS["unverified"]

    partial = make_panel(network, HorizonPredictor("92-181"))
    partial.set_station(station(network, "강남"))
    partial._date.setDate(QDate(2026, 11, 20))
    partial.predict()
    assert partial._result_unverified.text() == FROZEN_TEXTS["partial"]


def test_weather_note_and_climatology_are_unchanged(app, network):
    observed = HourlyWeather("2025-01-09", 8, 5.5, rainfall=1, snowfall=0)
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: observed))
    panel.set_station(station(network, "강남"))
    panel._date.setDate(QDate(2025, 1, 9))
    panel._hour.setCurrentIndex(panel._hour.findData(8))
    panel.refresh_weather()
    process_events()
    assert panel._weather_note.text() == FROZEN_TEXTS["weather_note"]

    future = make_panel(network)
    future.set_station(station(network, "강남"))
    future._date.setDate(QDate(2027, 3, 3))
    future.refresh_weather()
    month = future._date.date().toPython().month
    expected = FROZEN_TEXTS["climatology"].format(value=climatology_temperature(month))
    assert future._temperature.text() == expected
    assert future._temperature.property("climatology") is True


def test_line_label_for_missing_data_is_unchanged(app, network):
    panel = make_panel(network, ResultsPredictor(lines_with_data={("5호선", "까치산")}))
    panel.set_station(station(network, "까치산"))
    texts = [panel._line.itemText(i) for i in range(panel._line.count())]
    assert texts == list(FROZEN_TEXTS["line_labels"])
    assert panel._line.currentText() == FROZEN_TEXTS["line_labels"][1]


def test_reset_button_text_is_unchanged(app, network):
    panel = make_panel(network)
    texts = [button.text() for button in panel.findChildren(QPushButton)]
    assert FROZEN_TEXTS["reset_button"] in texts
    assert isinstance(panel._weekday, QLabel)


def test_search_row_height_matches_panel_padding(app, network):
    """검색 결과 목록 높이는 줄 높이(34)와 QSS 안쪽 여백(4px)으로만 정해진다."""
    panel = make_panel(network)
    panel.resize(340, 700)
    panel._search.setText("강남")
    rows = min(panel._results.count(), 7)

    assert SEARCH_ROW_HEIGHT == 34
    assert panel._results.height() == rows * SEARCH_ROW_HEIGHT + 2 * panel._results.frameWidth() + 8
