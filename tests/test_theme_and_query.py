"""라이트·다크 화면 전환과 질의 화면을 확인한다.

화면 모드는 세 곳에 함께 반영돼야 한다 — QSS(글자·바탕), QPalette(기본 위젯), 직접 그리는
그림 색(chart_colors). 질의 화면은 계산을 배경 스레드로 넘겨 창을 멈추지 않아야 하고,
보여 주는 숫자는 돌려받은 값을 그대로 옮겨 적어야 한다.
"""
import os
import re
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 화면 없이 위젯 검사

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtGui import QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from metro_insight.app import analysis_view, chart_colors, palette, styles  # noqa: E402
from metro_insight.app.analysis import load_analysis  # noqa: E402
from metro_insight.app.analysis_view import AnalysisView  # noqa: E402
from metro_insight.app.main_window import PAGE_QUERY, MainWindow  # noqa: E402
from metro_insight.app.query_view import (  # noqa: E402
    KIND_LABELS,
    QUICK_QUERIES,
    QueryView,
    fact_summary,
)
from metro_insight.app.station_map import StationMapView  # noqa: E402
from metro_insight.app.styles import qss, tokens  # noqa: E402
from metro_insight.app.subway_data import load_network  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_WINDOW = os.path.join(ROOT, "src", "metro_insight", "app", "main_window.py")
QUERY_VIEW = os.path.join(ROOT, "src", "metro_insight", "app", "query_view.py")


def token(mode, name):
    return tokens.resolve(mode)["color"][name].lower()


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def restore_light(app):
    """검사가 끝나면 화면을 라이트로 되돌린다 (다른 검사가 라이트 값을 기대한다)."""
    yield
    styles.apply_style(app, mode="light")
    chart_colors.set_mode("light")


def wait_until(app, predicate, timeout=5.0):
    """이 스레드에서 화면 사건을 풀어 주며 조건이 참이 되기를 기다린다."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---------- 팔레트 ----------


def test_build_palette_follows_the_mode():
    light = styles.build_palette("light")
    dark = styles.build_palette("dark")
    assert light.color(QPalette.ColorRole.Window).name().lower() == token("light", "bg")
    assert dark.color(QPalette.ColorRole.Window).name().lower() == token("dark", "bg")
    assert light.color(QPalette.ColorRole.Text).name().lower() == token("light", "text-1")
    assert dark.color(QPalette.ColorRole.Text).name().lower() == token("dark", "text-1")
    assert light.color(QPalette.ColorRole.Window) != dark.color(QPalette.ColorRole.Window)


def test_build_palette_keeps_the_disabled_text_readable_in_each_mode():
    for mode in tokens.MODES:
        value = styles.build_palette(mode).color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        assert value.name().lower() == token(mode, "disabled")


def test_build_palette_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        styles.build_palette("sepia")


def test_apply_style_switches_the_stylesheet_and_records_the_mode(app):
    styles.apply_style(app, mode="dark")
    assert styles.current_mode() == "dark"
    sheet = app.styleSheet()
    assert token("dark", "surface") in sheet
    assert not qss.unresolved(sheet)  # 색 자리가 남아 있으면 안 된다

    styles.apply_style(app, mode="light")
    assert styles.current_mode() == "light"
    assert token("light", "surface") in app.styleSheet()


# ---------- 직접 그리는 그림 색 ----------


def test_colors_for_returns_a_complete_bundle_per_mode():
    light = chart_colors.colors_for("light")
    dark = chart_colors.colors_for("dark")
    assert set(light) == set(dark)
    assert len(light["HEAT"]) == 6
    assert light["TEXT"] == token("light", "text-1")
    assert dark["TEXT"] == token("dark", "text-1")
    assert light["SURFACE"] != dark["SURFACE"]


def test_module_constants_stay_on_the_light_values():
    """다른 모듈이 import 시점에 읽어 간 값은 라이트 그대로다 (기본 화면 보존)."""
    assert chart_colors.get_mode() == "light"
    assert chart_colors.HEAT == tuple(token("light", f"heat-{step}") for step in range(1, 7))
    assert chart_colors.TEXT == token("light", "text-1")


def test_set_mode_swaps_the_module_constants():
    dark = chart_colors.set_mode("dark")
    try:
        assert chart_colors.get_mode() == "dark"
        assert chart_colors.HEAT == dark["HEAT"]
        assert chart_colors.TEXT == token("dark", "text-1")
        assert chart_colors.STATION_FILL == token("dark", "map-station-fill")
    finally:
        chart_colors.set_mode("light")
    assert chart_colors.TEXT == token("light", "text-1")


def test_set_mode_rejects_an_unknown_mode():
    with pytest.raises(KeyError):
        chart_colors.set_mode("sepia")
    with pytest.raises(KeyError):
        chart_colors.colors_for("sepia")


def test_band_follows_the_mode():
    light = chart_colors.band()
    assert light.alpha() == 14
    assert light.name().lower() == token("light", "chart-dot")
    chart_colors.set_mode("dark")
    try:
        assert chart_colors.band().name().lower() == token("dark", "chart-dot")
    finally:
        chart_colors.set_mode("light")


def test_gap_palette_has_three_colours_per_mode():
    light = palette.gap_palette("light")
    dark = palette.gap_palette("dark")
    assert set(light) == set(dark) == {"GAP_BORDER", "GAP_FILL", "GAP_LABEL"}
    assert light["GAP_BORDER"] == token("light", "map-gap-border")
    assert dark["GAP_BORDER"] == token("dark", "map-gap-border")
    assert light["GAP_FILL"] != dark["GAP_FILL"]
    assert palette.GAP_BORDER == light["GAP_BORDER"]  # 모듈 상수는 라이트 값


# ---------- 노선도 ----------


@pytest.fixture(scope="module")
def map_view(app):
    view = StationMapView()
    view.resize(900, 700)
    view.show_initial_view()
    return view


def gap_station_name(view):
    """예측 데이터가 없는 역 하나 (없으면 건너뛴다)."""
    for name, item in view.station_items.items():
        if item.is_gap:
            return name
    pytest.skip("예측 데이터가 없는 역이 없다")


def test_map_repaints_its_gap_stations_for_the_dark_mode(map_view):
    name = gap_station_name(map_view)
    item = map_view.station_items[name]
    before = item.pen().color().name().lower()
    assert before == token("light", "map-gap-border")

    map_view.apply_theme("dark")
    try:
        assert map_view.mode == "dark"
        assert item.pen().color().name().lower() == token("dark", "map-gap-border")
        assert item.brush().color().name().lower() == token("dark", "map-gap-fill")
        assert item.label._color == token("dark", "map-gap-label")
        casings = {path.pen().color().name().lower() for path in map_view._casing_items}
        assert casings == {token("dark", "map-label-halo")}
    finally:
        map_view.apply_theme("light")
    assert item.pen().color().name().lower() == token("light", "map-gap-border")


def test_map_repaints_the_selected_station_for_the_dark_mode(map_view):
    network = load_network()
    station = next(s for s in network[1] if s.name == "강남")
    map_view.select_station(station)
    item = map_view.station_items["강남"]
    map_view.apply_theme("dark")
    try:
        assert item.brush().color().name().lower() == token("dark", "map-selected-fill")
        assert item.label._selected_color == token("dark", "map-selected-label")
    finally:
        map_view.apply_theme("light")


def test_map_keeps_the_official_line_colours_in_dark_mode(map_view):
    """노선 선은 공식 노선색이라 모드와 상관없이 그대로다 (바탕색 테두리만 바뀐다)."""
    official = {line.color.lower() for line in map_view.lines}
    map_view.apply_theme("dark")
    try:
        after = {path.pen().color().name().lower() for path, _width in map_view._line_items}
    finally:
        map_view.apply_theme("light")
    assert official <= after


# ---------- 분석 화면 ----------


@pytest.fixture(scope="module")
def analysis_page(app):
    view = AnalysisView(load_analysis())
    view.resize(1200, 900)
    return view


def test_analysis_repaints_for_the_dark_mode(analysis_page):
    heatmap = analysis_page.heatmap
    heatmap.resize(600, heatmap.sizeHint().height())
    light_cell = heatmap.cell_color(0, 0)
    assert light_cell in set(chart_colors.colors_for("light")["HEAT"]) | {
        token("light", "chart-missing")
    }
    heatmap._pixmap = None
    heatmap.grab()

    analysis_page.apply_theme("dark")
    try:
        dark_cell = heatmap.cell_color(0, 0)
        assert dark_cell in set(chart_colors.colors_for("dark")["HEAT"]) | {
            token("dark", "chart-missing")
        }
        assert light_cell != dark_cell or light_cell == token("light", "chart-missing")
        assert heatmap._pixmap is None  # 캐시한 그림을 버리고 다음 그리기에서 새 색을 쓴다
        bar_colors = {bar.color for bar, muted in analysis_page.tertiles._bars if not muted}
        assert bar_colors <= {token("dark", "chart-bar")}
    finally:
        analysis_page.apply_theme("light")
    assert analysis_page.heatmap.cell_color(0, 0) in set(chart_colors.colors_for("light")["HEAT"]) | {
        token("light", "chart-missing")
    }


def test_analysis_module_constants_stay_on_the_light_values():
    """그림 색 상수(검사 기준)는 라이트 값 그대로 둔다 — 화면은 인스턴스 속성으로 칠한다."""
    assert analysis_view.BAR == chart_colors.BAR
    assert analysis_view.MISSING_CELL == chart_colors.MISSING


# ---------- 헤더의 화면 모드 단추 ----------


class Loader(QObject):
    """예측기를 읽는 흉내 (검사에서는 아무 일도 하지 않는다)."""

    ready = Signal(object)
    failed = Signal(str)

    def start(self, *args, **kwargs):
        pass


@pytest.fixture(scope="module")
def window(app):
    widget = MainWindow(loader=Loader())
    widget.show()
    app.processEvents()
    return widget


def test_theme_toggle_says_where_it_goes(window, app):
    toggle = window.theme_toggle
    assert toggle.objectName() == "themeToggle"
    assert not toggle.isChecked()
    assert toggle.text() == "다크 모드"  # 지금 모드가 아니라 누르면 갈 모드를 적는다
    assert window.mode == "light"

    toggle.click()
    app.processEvents()
    try:
        assert window.mode == "dark"
        assert toggle.isChecked()
        assert toggle.text() == "라이트 모드"
        assert chart_colors.get_mode() == "dark"
        assert window.map_view.mode == "dark"
        assert styles.current_mode() == "dark"
    finally:
        toggle.click()
        app.processEvents()
    assert window.mode == "light"
    assert toggle.text() == "다크 모드"


def test_a_window_created_after_a_style_change_follows_it(app):
    """스타일을 먼저 바꿔 둔 경우에도 창이 그 모드로 시작한다 (화면과 색이 어긋나지 않게)."""
    styles.apply_style(app, mode="dark")
    later = MainWindow(loader=Loader())
    try:
        assert later.mode == "dark"
        assert later.theme_toggle.isChecked()
        assert later.theme_toggle.text() == "라이트 모드"
        assert later.map_view.mode == "dark"
    finally:
        later.close()
        styles.apply_style(app, mode="light")


def test_theme_switch_keeps_the_open_page(window, app):
    window.segment_analysis.click()
    app.processEvents()
    page = window._stack.currentWidget()
    window.theme_toggle.click()
    app.processEvents()
    try:
        assert window._stack.currentWidget() is page
        assert window.segment_analysis.isChecked()
    finally:
        window.theme_toggle.click()
        app.processEvents()
    window.segment_map.click()
    app.processEvents()


# ---------- 질의 화면 (지연 로딩) ----------


def test_query_page_is_created_on_first_open(window, app):
    assert window._query_page is None
    assert window._pages.get(PAGE_QUERY) is None
    before = window._stack.count()

    window.segment_query.click()
    app.processEvents()
    assert window._query_page is not None
    assert window._pages[PAGE_QUERY] is window._query_page
    assert window._stack.count() == before + 1
    assert window.segment_query.isChecked()
    assert window._stack.currentWidget() is window._query_page
    assert "질의" in window._meta.text()
    assert not window._panel_toggle.isEnabled()  # 역 정보 패널은 노선도 화면의 것


def test_query_view_is_never_imported_eagerly():
    with open(MAIN_WINDOW, encoding="utf-8") as handle:
        text = handle.read()
    head, _, _ = text.partition("def show_page")
    assert "query_view" not in head
    assert "from metro_insight.app.query_view import QueryView" in text


def test_query_page_shows_the_selected_station(window, app):
    window.segment_query.click()
    app.processEvents()
    page = window._query_page
    assert page.context_label.text() == "기준 역 없음"
    assert page._chips[0].isEnabled() is False  # 역이 필요한 빠른 질의
    assert page._chips[2].isEnabled() is True  # 역과 상관없는 빠른 질의


# ---------- 질의 화면 (주어진 값을 그대로 보여 준다) ----------


def sample_result(**overrides):
    result = {
        "answer": "강남역 2026-11-16 08시 예측은 2,684명입니다.",
        "latency_ms": 42,
        "plan": {
            "slots": {"line": "2호선", "station": "강남", "date": "2026-11-16", "hour": 8},
        },
        "bundle": {
            "facts": [
                {
                    "kind": "prediction",
                    "display": {
                        "line": "2호선",
                        "station": "강남",
                        "date": "2026-11-16",
                        "hour": "08시",
                        "value": "2,684명",
                    },
                }
            ],
            "unavailable": [],
            "policy": {},
        },
    }
    result.update(overrides)
    return result


def make_view(app, result=None, context=None):
    calls = []

    def turn(utterance, ui_context):
        calls.append((utterance, dict(ui_context or {})))
        if isinstance(result, Exception):
            raise result
        return result if result is not None else sample_result()

    view = QueryView(turn=turn, context=context or (lambda: {}))
    view.resize(1000, 800)
    return view, calls


def test_answer_copies_the_values_it_was_given(app):
    view, calls = make_view(app)
    view.ask("강남 내일 08시 예측", wait=True)
    assert calls == [("강남 내일 08시 예측", {})]
    assert view.answer.text() == sample_result()["answer"]
    assert view.meta.text() == "처리 42ms · 근거 1건"
    assert view.conditions == ["2호선", "강남", "2026-11-16", "08시"]
    assert len(view.evidence_texts) == 1
    assert "2,684명" in view.evidence_texts[0].text()
    assert view.gap_card.isHidden()
    assert view.empty.isHidden()


def test_evidence_kind_names_are_plain_words(app):
    view, _calls = make_view(app)
    view.ask("질의", wait=True)
    tags = [label.text() for label in view.findChildren(type(view.status), "queryEvidenceKind")]
    assert tags == [KIND_LABELS["prediction"]]


def test_unavailable_items_are_listed_separately(app):
    result = sample_result()
    result["bundle"] = {
        "facts": [],
        "unavailable": [{"human": "5호선 여의도는 대상일 실측이 아직 없습니다."}],
        "policy": {"de_missing": True},
    }
    view, _calls = make_view(app, result)
    view.ask("질의", wait=True)
    texts = [label.text() for label in view.gap_texts]
    assert any("여의도" in text for text in texts)
    assert any("D·E" in text for text in texts)
    assert len(texts) == 2  # 겹치는 사유는 한 번만
    assert not view.gap_card.isHidden()


def test_failure_is_shown_instead_of_raised(app):
    view, _calls = make_view(app, RuntimeError("자료를 읽지 못했습니다"))
    assert view.ask("질의", wait=True) is None
    assert view.answer.text() == "질의를 처리하지 못했습니다."
    assert any("자료를 읽지 못했습니다" in text.text() for text in view.gap_texts)
    assert view.submit.isEnabled()  # 다시 시도할 수 있다


def test_a_long_query_does_not_freeze_the_window(app):
    def slow(utterance, ui_context):
        time.sleep(0.4)
        return sample_result()

    view = QueryView(turn=slow, context=lambda: {})
    view.resize(800, 600)
    started = time.monotonic()
    view.ask("강남 내일 08시 예측")
    elapsed = time.monotonic() - started
    assert elapsed < 0.3, f"질의가 화면을 붙잡았다: {elapsed:.3f}s"
    assert not view.submit.isEnabled()  # 계산 중에는 다시 보내지 않는다
    assert wait_until(app, lambda: view.meta.text() != "")
    assert view.answer.text() == sample_result()["answer"]
    assert view.submit.isEnabled()


def test_late_results_of_an_older_query_are_dropped(app):
    """계산 중에 새 질의를 보내면 먼저 보낸 결과는 버린다."""
    seen = []
    first = []

    def turn(utterance, ui_context):
        if utterance.endswith("예측"):
            time.sleep(0.3)
            return sample_result(answer="먼저 보낸 결과")
        return sample_result(answer="나중 결과")

    view = QueryView(turn=turn, context=lambda: {})
    view.ask("강남 내일 08시 예측")
    view.ask("데이터 최신 상태 점검")
    assert wait_until(app, lambda: view.answer.text() == "나중 결과")
    app.processEvents()
    assert view.answer.text() == "나중 결과"
    seen.append(view.answer.text())
    assert seen == ["나중 결과"]
    assert first == []


def test_quick_queries_need_a_station(app):
    view, calls = make_view(app, context=lambda: {"station": "강남", "line": "2호선"})
    assert view.context_label.text() == "기준 역 강남 · 2호선"
    for chip in view._chips:
        assert chip.isEnabled()
    view._chips[0].click()
    assert wait_until(app, lambda: bool(calls))
    assert calls[0][0] == "강남 내일 08시 예측"
    assert calls[0][1] == {"station": "강남", "line": "2호선"}


def test_quick_queries_are_disabled_without_a_station(app):
    view, _calls = make_view(app)
    for chip, (_text, _template, needs_station) in zip(view._chips, QUICK_QUERIES, strict=True):
        assert chip.isEnabled() is (not needs_station), chip.text()


def test_selected_station_is_filled_into_the_sentence(app):
    view, calls = make_view(app, context=lambda: {"station": "강남"}, result=sample_result())
    view.ask("이 역 내일 08시 예측", wait=True)
    assert calls[0][0] == "강남 내일 08시 예측"
    assert view.input.text() == "강남 내일 08시 예측"  # 화면에도 해석한 문장을 남긴다


def test_blank_query_is_ignored(app):
    view, calls = make_view(app)
    assert view.ask("   ", wait=True) is None
    assert calls == []


def test_fact_summary_lists_the_place_and_the_value():
    fact = {
        "kind": "prediction",
        "display": {
            "line": "2호선",
            "station": "강남",
            "date": "2026-11-16",
            "hour": "08시",
            "value": "2,684명",
        },
    }
    text = fact_summary(fact)
    assert text.startswith("2호선 강남")
    assert "2026-11-16 08시" in text
    assert "2,684명" in text


def test_fact_summary_puts_model_values_before_the_changes():
    fact = {
        "display": {
            "line": "2호선",
            "station": "강남",
            "A": "2,684명",
            "B": "2,652명",
            "C": "2,649명",
            "delta_A": "0.00명",
            "delta_C": "-14.03명",
            "days": "0일",
        }
    }
    assert fact_summary(fact) == (
        "2호선 강남 · A 2,684명 · B 2,652명 · C 2,649명 · 기준 대비 C -14.03명"
    )


def test_fact_summary_labels_short_keys_that_are_not_models():
    fact = {"display": {"date_from": "2026-06-01", "date_to": "2026-06-30", "n": "0", "z": "3.00"}}
    assert fact_summary(fact) == "2026-06-01~2026-06-30 · 관측 0 · 기준 3.00"


def test_fact_summary_keeps_the_data_status_in_front():
    fact = {"display": {"data_end": "2026-06-30", "days": "138일", "stale": "낡음"}}
    text = fact_summary(fact)
    assert text.startswith("상태 낡음")
    assert "경과일 138일" in text


# ---------- 쓰지 않기로 한 말 ----------

FORBIDDEN = ("AI", "LLM", "GPT", "Claude", "Gemini", "에이전트", "프롬프트")


STYLESHEET = os.path.join(ROOT, "src", "metro_insight", "app", "styles", "modern_gray.qss")


@pytest.mark.parametrize("path", [QUERY_VIEW, STYLESHEET])
def test_query_text_avoids_tool_jargon(path):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    found = [word for word in FORBIDDEN if word in text]
    assert not found, f"{os.path.basename(path)} 에 남은 말: {found}"


def test_layout_cache_is_packaged():
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as handle:
        text = handle.read()
    line = next(line for line in text.splitlines() if line.startswith('"metro_insight.app"'))
    assert "layout_cache.json" in line
    assert os.path.exists(os.path.join(ROOT, "src", "metro_insight", "app", "layout_cache.json"))


def test_query_styles_exist_in_the_stylesheet():
    text = qss.load("modern_gray")
    for name in ("queryCard", "queryAnswer", "queryEvidenceKind", "queryGapText", "themeToggle"):
        assert re.search(rf"#{name}\b", text), name