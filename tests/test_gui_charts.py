"""예측값 도트플롯이 숫자와 같은 값을 그리는지, 축·막대 계산이 맞는지 확인한다."""
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 화면 없이 위젯 검사

from PySide6.QtCore import QEvent, QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from metro_insight.app.charts import (  # noqa: E402
    AXIS_PAD_RATIO,
    Axis,
    DotPlot,
    coverage_share,
    error_caption,
    plot_axis,
    spread_note,
    value_x,
    whisker_span,
)
from metro_insight.app.station_panel import StationPanel, WeatherLoader  # noqa: E402
from metro_insight.app.subway_data import load_network  # noqa: E402
from metro_insight.data.weather import HourlyWeather  # noqa: E402
from metro_insight.model.evaluation import ModelScore  # noqa: E402

SUNNY = HourlyWeather("2026-10-12", 9, 21.6, rainfall=0, snowfall=0)

VALUES = {"A": 11850.4, "B": 12240.0, "C": 12500.0, "D": 12610.0, "E": 12580.0}
MAE = {"A": 820.25, "B": 540.0, "C": 498.0, "D": 455.5, "E": 462.0}
BEST = "D"  # MAE 가 가장 작은 모델


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def network():
    return load_network()


@pytest.fixture
def plot(app):
    widget = DotPlot()
    widget.resize(320, 200)
    widget.set_data(VALUES, MAE)
    return widget


# ---------- 축 ----------


def test_axis_does_not_start_at_zero():
    """0 부터 그리면 모델 사이 차이가 보이지 않으므로, 가장 낮은 값보다 아래에서 시작한다."""
    axis = plot_axis(VALUES, MAE)
    assert axis is not None
    assert axis.lo > 0
    assert axis.lo < min(VALUES.values())
    assert axis.hi > max(VALUES.values())


def test_axis_covers_every_whisker():
    """축은 모든 모델의 예측값 ±MAE 를 담아야 한다 (막대가 잘리지 않게)."""
    axis = plot_axis(VALUES, MAE)
    for model, value in VALUES.items():
        assert axis.lo <= value - MAE[model]
        assert axis.hi >= value + MAE[model]


def test_axis_padding_is_ratio_of_data_span():
    """양끝 여백은 (막대 끝 ~ 막대 끝) 폭의 15% 다."""
    axis = plot_axis(VALUES, MAE)
    low = min(VALUES[m] - MAE[m] for m in VALUES)
    high = max(VALUES[m] + MAE[m] for m in VALUES)
    data_span = high - low
    assert axis.lo == pytest.approx(low - data_span * AXIS_PAD_RATIO)
    assert axis.hi == pytest.approx(high + data_span * AXIS_PAD_RATIO)


def test_axis_handles_missing_and_single_values():
    assert plot_axis({}, {}) is None
    single = plot_axis({"A": 5000.0}, {})
    assert single is not None and single.hi > single.lo
    # 값이 없거나 None 인 모델이 섞여 있어도 계산되는 모델만 본다
    mixed = plot_axis({"A": 5000.0, "B": None}, {"A": 100.0})
    assert mixed is not None and mixed.lo < 4900


def test_value_x_moves_with_value():
    axis = Axis(0.0, 10.0)
    points = [value_x(axis, v, 10.0, 100.0) for v in (0.0, 5.0, 10.0)]
    assert points == pytest.approx([10.0, 60.0, 110.0])

    left, right = whisker_span(axis, 5.0, 2.0, 10.0, 100.0)
    assert (left, right) == pytest.approx((40.0, 80.0))  # ±2 는 축의 ±20%


# ---------- 그림 ----------


def test_rows_are_numbered_in_model_order(plot):
    assert plot.models() == list(VALUES)
    assert plot.row_y("A") < plot.row_y("B") < plot.row_y("C")


def test_dots_sit_inside_plot_and_follow_values(plot):
    rect = plot.plot_rect()
    dots = {model: plot.dot_position(model) for model in VALUES}
    for point in dots.values():
        assert isinstance(point, QPointF)
        assert rect.left() <= point.x() <= rect.right()
    assert dots["A"].x() < dots["B"].x() < dots["C"].x() < dots["E"].x() < dots["D"].x()


def test_whisker_brackets_each_dot(plot):
    for model in VALUES:
        left, right = plot.whisker_pixels(model)
        dot = plot.dot_position(model)
        assert left < dot.x() < right
        assert right - left == pytest.approx(
            2 * MAE[model] / plot.axis.span * plot.plot_rect().width(), rel=1e-6
        )


def test_best_model_is_lowest_mae(plot):
    assert plot.best_model == BEST


def test_report_matches_chart(plot):
    """그림의 점·막대는 표에 적은 숫자와 같아야 한다."""
    assert plot.values == VALUES
    assert plot.errors == MAE
    assert plot.hover_model  # 행 단위 강조 신호가 있다
    assert plot.best_model == min(MAE, key=MAE.get)


def test_model_at_maps_pointer_to_row(plot):
    assert plot.model_at(plot.row_y("C")) == "C"
    assert plot.model_at(-10) is None


# ---------- 설명 문장 ----------


def test_coverage_share_is_read_from_backtest_file():
    """오차 막대 포함률은 182~366일 백테스트 기록에서 중앙값으로 계산한다 (상수로 적지 않는다)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "models", "backtest_182_366.json"), encoding="utf-8") as f:
        data = json.load(f)
    shares = sorted(
        entry["coverage_mae"]
        for cut in data["per_cut"].values()
        for model, entry in cut["models"].items()
        if model in ("A", "B", "C")
    )
    middle = len(shares) // 2
    expected = shares[middle] if len(shares) % 2 else (shares[middle - 1] + shares[middle]) / 2
    assert coverage_share() == pytest.approx(expected)
    assert 0.5 < coverage_share() < 1.0


def test_caption_says_mae_and_real_share():
    caption = error_caption(coverage_share())
    assert "±MAE" in caption
    assert "약 73%" in caption
    assert "80%" not in caption  # 80% 구간처럼 읽히지 않게 한다
    assert error_caption(None) == "오차 막대는 ±MAE 기준입니다."


def test_spread_note_compares_difference_with_error():
    """모델 사이 차이가 오차보다 작으면 순위를 단정하지 않는다."""
    smaller = spread_note({"A": 100.0, "B": 105.0}, {"A": 60.0, "B": 108.0})
    assert "5개" not in smaller  # 모델 수는 세어서 적는다
    assert "우열을 가르기 어렵습니다" in smaller

    between = spread_note({"A": 100.0, "B": 160.0}, {"A": 30.0, "B": 90.0})
    assert "뒤집힐 수 있습니다" in between

    larger = spread_note({"A": 100.0, "B": 400.0}, {"A": 30.0, "B": 90.0})
    assert "가장 큰 오차보다도 큽니다" in larger

    assert spread_note({"A": 100.0}, {"A": 10.0}) == ""


# ---------- 패널에 붙은 그림 ----------


class FakePredictor:
    def __init__(self, predictions=None, scores=None):
        self._predictions = predictions if predictions is not None else dict(VALUES)
        self._scores = scores if scores is not None else dict(MAE)

    def predict(self, values):
        return dict(self._predictions)

    def scores(self):
        return {model: ModelScore(mae, mae * 1.5) for model, mae in self._scores.items()}


def station(network, name):
    return next(s for s in network[1] if s.name == name)


def predict_with(app, network, predictor):
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY), predictor=predictor)
    panel.set_station(station(network, "강남"))
    panel.predict()
    return panel


def test_chart_shows_the_same_numbers_as_the_table(app, network):
    panel = predict_with(app, network, FakePredictor())
    chart = panel._chart
    labels = {model: label.text() for model, label in panel._result_boarding.items()}
    assert labels["A"] == "11,850명"  # 표의 숫자는 그대로 둔다
    assert chart.values["A"] == pytest.approx(11850.4)
    assert chart.errors["A"] == pytest.approx(820.25)
    assert chart.best_model == BEST
    assert panel._chart_caption.text() == error_caption(coverage_share())
    assert "760명" in panel._chart_readout.text()  # 모델 사이 차이를 자료에서 계산해 적는다


def test_chart_is_a_sibling_of_the_number_block(app, network):
    """그림은 숫자 줄을 대신하지 않고 그 옆에 더해진다."""
    panel = predict_with(app, network, FakePredictor())
    assert panel._chart.parent() is not panel._result_boarding_box
    assert not panel._result_boarding_box.isHidden()
    assert not panel._chart.isHidden()

    card = panel._result.layout()
    order = [
        card.itemAt(i).widget()
        for i in range(card.count())
        if card.itemAt(i).widget() is not None
    ]
    block = panel._chart.parentWidget()  # 그림을 담은 칸
    assert block is not None and block is not panel._result_boarding_box
    assert order.index(block) == order.index(panel._result_boarding_box) + 1


def test_models_without_a_value_are_left_out_of_the_chart(app, network):
    """D·E 는 최근 실측이 없으면 값이 없다. 빈 모델을 점으로 그리면 축이 왜곡된다."""
    partial = {"A": 11850.4, "B": 12240.0, "C": 12500.0, "D": None, "E": None}
    panel = predict_with(app, network, FakePredictor(predictions=partial))
    assert panel._chart.models() == ["A", "B", "C"]
    assert "D" not in panel._chart.values


def test_highlight_pairs_chart_and_table(app, network):
    panel = predict_with(app, network, FakePredictor())
    summary = panel._chart_readout.text()
    panel._highlight_model("B")
    assert panel._chart.highlight == "B"
    assert panel._result_boarding["B"].property("hover") is True
    assert panel._result_boarding["A"].property("hover") is False
    assert "±540" in panel._chart_readout.text()

    panel._highlight_model(None)
    assert panel._chart.highlight is None
    assert all(label.property("hover") is False for label in panel._result_boarding.values())
    assert panel._chart_readout.text() == summary  # 손을 떼면 해석 문장으로 돌아온다


def test_hovering_a_table_row_highlights_the_chart(app, network):
    panel = predict_with(app, network, FakePredictor())
    label = panel._result_boarding["C"]
    QApplication.sendEvent(label, QEvent(QEvent.Type.Enter))
    assert panel._chart.highlight == "C"
    QApplication.sendEvent(label, QEvent(QEvent.Type.Leave))
    assert panel._chart.highlight is None


def test_highlight_is_reset_when_a_new_prediction_arrives(app, network):
    panel = predict_with(app, network, FakePredictor())
    panel._highlight_model("A")
    panel.predict()
    assert panel._chart.highlight is None
    assert all(label.property("hover") is False for label in panel._result_boarding.values())