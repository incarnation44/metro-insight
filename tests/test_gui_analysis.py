"""분석 화면이 평가 파일의 숫자를 그대로 보여 주는지 확인한다 (새로 계산하지 않는다)."""
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 화면 없이 위젯 검사

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from metro_insight.app.analysis import (  # noqa: E402
    FAR_HORIZON,
    HEAT_STEPS,
    AnalysisData,
    heat_color,
    heat_step,
    horizon_insight,
    load_analysis,
    tertile_note,
)
from metro_insight.app.analysis_view import MISSING_CELL, AnalysisView, Heatmap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(name):
    with open(os.path.join(ROOT, "models", name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def data():
    return load_analysis()


@pytest.fixture(scope="module")
def view(app, data):
    widget = AnalysisView(data)
    widget.resize(1200, 900)
    widget.show()
    return widget


# ---------- 자료 읽기 ----------


def test_model_scores_come_from_the_metric_files(data):
    calendar = read("metrics_calendar.json")["test"]
    sequence = read("metrics_sequence.json")["test"]
    for stat in data.models:
        source = (calendar if stat.name in ("A", "B", "C") else sequence)[stat.name]
        assert stat.mae == pytest.approx(source["mae"])
        assert stat.rmse == pytest.approx(source["rmse"])


def test_models_are_sorted_by_error(data):
    maes = [stat.mae for stat in data.models]
    assert maes == sorted(maes)
    assert [stat.name for stat in data.models] == ["E", "D", "B", "C", "A"]
    assert data.best().name == "E"


def test_horizon_values_match_the_files(data):
    calendar = read("metrics_calendar.json")["test"]
    for model in ("A", "B", "C"):
        for horizon in ("1-31", "32-91", "92-181"):
            assert data.value_at(model, horizon) == pytest.approx(calendar[model]["horizon"][horizon]["mae"])
    sequence = read("metrics_sequence.json")["test"]
    for model in ("D", "E"):
        for horizon in ("1-31", "32-91", "92-181"):
            assert data.value_at(model, horizon) == pytest.approx(sequence[model]["horizon"][horizon]["mae"])


def test_far_horizon_comes_from_the_backtest(data):
    pooled = read("backtest_182_366.json")["pooled"]["models"]
    assert FAR_HORIZON in data.horizons
    for model in ("A", "B", "C"):
        assert data.value_at(model, FAR_HORIZON) == pytest.approx(pooled[model]["mae_weighted"])
        assert data.pooled[model] == pytest.approx(pooled[model]["mae_weighted"])
    # 백테스트는 A~C 만 다시 맞혀 봤다
    assert data.value_at("D", FAR_HORIZON) is None


def test_cut_dates_and_values_match_the_backtest(data):
    per_cut = read("backtest_182_366.json")["per_cut"]
    assert list(data.cut_dates) == sorted(per_cut)
    for cut, values in data.cut_mae.items():
        for model, value in values.items():
            assert value == pytest.approx(per_cut[cut]["models"][model]["mae"])


def test_tertile_values_match_the_sequence_file(data):
    sequence = read("metrics_sequence.json")["test"]
    for model in ("D", "E"):
        for label in ("소", "중", "대"):
            assert data.tertiles[model][label] == pytest.approx(sequence[model]["tertile"][label]["mae"])
    assert data.tertiles["D"]["소"] == pytest.approx(17.2395, rel=1e-4)
    assert data.tertiles["D"]["대"] == pytest.approx(118.2955, rel=1e-4)


# ---------- 그림 계산 ----------


def test_heat_colour_is_one_hue_in_six_steps():
    assert len(HEAT_STEPS) == 6
    low, high = 60.0, 136.0
    assert heat_step(low, low, high) == 0
    assert heat_step(high, low, high) == 5
    assert heat_color(low, low, high) == HEAT_STEPS[0]
    assert heat_color(high, low, high) == HEAT_STEPS[-1]
    assert heat_step(80.0, low, high) < heat_step(120.0, low, high)


def test_heat_range_brackets_every_cell(data):
    low, high = data.heat_range()
    values = [value for row in data.horizon_mae.values() for value in row.values()]
    assert low == min(values) and high == max(values)


def test_insight_follows_the_data(data):
    text = horizon_insight(data)
    assert "A" in text and "136" in text  # 1~31일에서 가장 부정확한 모델
    assert "92" in text and "101" in text  # C 는 지평이 길어질수록 나빠진다
    assert "줄어드는" in text and "늘어나는" in text
    # 달력 모델만 세지 않고 다섯 모델을 모두 본다 (D·E 도 지평이 길면 오차가 줄어든다)
    assert "D(" in text and "E(" in text


def test_tertile_note_uses_the_numbers(data):
    note = tertile_note(data)
    assert "17.2" in note and "118.3" in note


def test_view_builds_when_files_are_missing(tmp_path, app):
    """평가 파일이 없어도 화면이 비어 있을 뿐 멈추지 않는다."""
    empty = AnalysisData((), (), {}, {}, (), {}, {}, {})
    view = AnalysisView(empty)
    assert view.heatmap.rows() == []
    assert view.insight.text() == "지평 구간별 기록이 없습니다."


# ---------- 화면 ----------


def test_model_cards_follow_the_ranking(view, data):
    assert list(view.cards) == [stat.name for stat in data.models]
    values = [view.cards[name].findChild(QLabel, "modelCardValue").text() for name in view.cards]
    assert values == [f"{stat.mae:,.1f}" for stat in data.models]
    assert values[0] == "59.9"  # E


def test_model_card_click_is_reported(view):
    received = []
    view.model_selected.connect(received.append)
    view.cards["B"].clicked.emit("B")
    assert received == ["B"]


def test_heatmap_rows_and_columns_match_the_data(view, data):
    assert view.heatmap.rows() == [stat.name for stat in data.models]
    assert view.data.horizons == data.horizons
    assert len(data.horizons) == 4


def test_heatmap_cells_use_the_value_colour(view, data):
    low, high = data.heat_range()
    for row, model in enumerate(view.heatmap.rows()):
        for column, horizon in enumerate(data.horizons):
            value = data.value_at(model, horizon)
            expected = MISSING_CELL if value is None else heat_color(value, low, high)
            assert view.heatmap.cell_color(row, column) == expected
            assert view.heatmap.cell_rect(row, column).width() > 0


def test_heatmap_paints_the_colour_it_reports(view, data):
    """값을 묻는 색과 실제로 칠한 픽셀이 같다 (값이 없는 칸 포함)."""
    heatmap = view.heatmap
    heatmap.resize(600, heatmap.sizeHint().height())
    pixmap = heatmap.grab()
    image = pixmap.toImage()
    ratio = pixmap.devicePixelRatio()
    checked = 0
    for row, model in enumerate(heatmap.rows()):
        for column, horizon in enumerate(data.horizons):
            box = heatmap.cell_rect(row, column)
            expected = heatmap.cell_color(row, column).lower()
            point = QPointF(box.left() + 8, box.top() + 8)
            pixel = image.pixelColor(int(point.x() * ratio), int(point.y() * ratio)).name().lower()
            assert pixel == expected, (model, horizon, expected, pixel)
            checked += 1
    assert checked >= 20


def test_analysis_paint_colours_come_from_the_tokens():
    """그림 색도 토큰 한 곳에서 온다 (규모별 막대·빈 칸)."""
    from metro_insight.app import analysis_view, chart_colors

    assert analysis_view.BAR == chart_colors.BAR
    assert analysis_view.BAR_MUTED == chart_colors.BAR_MUTED
    assert analysis_view.MISSING_CELL == chart_colors.MISSING
    assert analysis_view.SWATCH_EDGE == chart_colors.SWATCH_EDGE


def test_heatmap_paints_without_error(view):
    """그림을 실제로 한 번 그려 본다 (캐시 픽스맵 경로)."""
    view.heatmap.resize(600, view.heatmap.sizeHint().height())
    pixmap = view.heatmap.grab()
    assert not pixmap.isNull()
    assert pixmap.width() == view.heatmap.width()


def test_backtest_table_shows_the_cut_values(view, data):
    for cut, values in data.cut_mae.items():
        for model, value in values.items():
            assert view.backtest_rows[cut][model] == f"{value:,.1f}"
    assert view.backtest_rows[data.cut_dates[0]]["A"] == "109.5"


def test_tertile_chart_shows_three_scales(view, data):
    labels = [label.text() for label in view.tertiles.findChildren(QLabel, "statValue")]
    for model in data.tertiles:
        for label in ("소", "중", "대"):
            assert f"{data.tertiles[model][label]:,.1f}명" in labels


def test_analysis_is_never_imported_eagerly():
    """분석 화면 코드는 열 때 읽는다 (앱을 켜는 시간에 더하지 않게)."""
    source = os.path.join(ROOT, "src", "metro_insight", "app", "main_window.py")
    with open(source, encoding="utf-8") as f:
        text = f.read()
    head, _, _ = text.partition("def show_page")
    assert "analysis_view" not in head
    assert "from metro_insight.app.analysis_view import AnalysisView" in text


def test_heatmap_uses_a_pixmap_cache(view):
    assert isinstance(view.heatmap, Heatmap)
    view.heatmap.grab()
    assert view.heatmap._pixmap is not None
    assert view.heatmap._pixmap.size() == view.heatmap.size()


def test_segment_switches_pages(app):
    """헤더의 노선도·분석 단추로 화면을 바꾼다 (분석 화면은 처음 열 때 만든다)."""
    from PySide6.QtCore import QObject, Signal

    from metro_insight.app.main_window import MainWindow

    class Loader(QObject):
        ready = Signal(object)
        failed = Signal(str)

        def start(self, *args, **kwargs):
            pass

    window = MainWindow(loader=Loader())
    window.show()
    app.processEvents()
    assert window._stack.count() == 1  # 분석 화면은 아직 없다
    window.segment_analysis.click()
    app.processEvents()
    assert window._stack.count() == 2
    assert window._analysis_page is not None
    assert window.segment_analysis.isChecked()

    window._analysis_page.model_selected.emit("C")
    app.processEvents()
    assert not window.segment_analysis.isChecked()
    assert window._stack.currentWidget() is window.map_view.parentWidget()
    assert "C 모델" in window._model_chip.text()
    assert window._model_chip.isVisibleTo(window)
    window.close()


def test_map_page_keeps_its_widgets_after_switching(app):
    """화면을 오가도 노선도·패널은 그대로 살아 있다."""
    from PySide6.QtCore import QObject, Signal

    from metro_insight.app.main_window import MainWindow

    class Loader(QObject):
        ready = Signal(object)
        failed = Signal(str)

        def start(self, *args, **kwargs):
            pass

    window = MainWindow(loader=Loader())
    window.show()
    app.processEvents()
    map_page = window._stack.widget(0)
    window.segment_analysis.click()
    app.processEvents()
    window.segment_map.click()
    app.processEvents()
    assert window._stack.currentWidget() is map_page
    assert window.map_view.isVisible()
    assert window._panel_toggle.isEnabled()
    window.segment_analysis.click()
    app.processEvents()
    assert not window._panel_toggle.isEnabled()  # 역 정보 패널은 노선도 화면의 것
    window.close()