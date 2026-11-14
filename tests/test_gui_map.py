"""노선도의 글자 크기·확대 기준·키보드 이동과, 호선 배지 글자 대비를 확인한다."""
import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 화면 없이 위젯 검사

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from metro_insight.app import station_map  # noqa: E402
from metro_insight.app.map_overlays import LineBadge  # noqa: E402
from metro_insight.app.palette import badge_palette, darken  # noqa: E402
from metro_insight.app.station_map import (  # noqa: E402
    LABEL_FADE_MS,
    LABEL_MIN_ZOOM,
    LABEL_POINT_SIZE,
    StationMapView,
)
from metro_insight.app.styles import qss  # noqa: E402
from metro_insight.app.styles.tokens import AA_NORMAL_TEXT, contrast_ratio  # noqa: E402
from metro_insight.app.subway_data import load_network  # noqa: E402

WHITE = "#ffffff"
INK = "#111827"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def network():
    return load_network()


@pytest.fixture(scope="module")
def view(app, network):
    widget = StationMapView()
    widget.resize(900, 700)
    widget.show_initial_view()
    return widget


def press(view, key):
    event = QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(view, event)
    return event


def station(network, name):
    return next(s for s in network[1] if s.name == name)


# ---------- 이름 표시 기준 ----------


def test_label_shows_from_thirty_percent_zoom():
    """확대를 조금만 해도 역 이름이 보이도록 기준을 낮췄다."""
    assert LABEL_MIN_ZOOM == pytest.approx(0.30)
    assert LABEL_POINT_SIZE >= 9.0  # 가장 작을 때도 9pt 아래로 내려가지 않는다


def test_labels_appear_past_the_threshold(view):
    item = view.station_items["강남"]
    assert not item.station.is_transfer
    assert not item.wants_label(LABEL_MIN_ZOOM - 0.02)
    assert item.wants_label(LABEL_MIN_ZOOM + 0.02)
    assert item.wants_label(3.0)


def test_selected_station_always_shows_its_name(view, network):
    view.select_station(station(network, "강남"))
    item = view.station_items["강남"]
    assert item.wants_label(0.05)
    assert item.label.isVisible()  # 고른 역은 확대와 상관없이 이름이 보인다


def test_new_labels_fade_in(view):
    """새로 나타난 이름은 툭 나타나지 않고 잠깐에 걸쳐 진해진다 (90ms)."""
    assert station_map.LABEL_FADE_MS == 90
    view._fade_timer.stop()
    view._fading.clear()
    for name in view.station_items:
        view.station_items[name].label.setVisible(False)
    view._update_labels()
    fading = [label for label in view._fading if label.isVisible()]
    assert fading, "새로 보이는 이름이 하나는 있어야 한다"
    assert all(label.opacity() == 0.0 for label in fading)

    steps = 0
    while view._fading and steps < 20:
        view._on_label_fade()
        steps += 1
    assert steps * station_map.LABEL_FADE_STEP_MS <= 2 * LABEL_FADE_MS
    assert all(
        label.opacity() == 1.0
        for label in (item.label for item in view._stations.values())
        if label.isVisible()
    )


# ---------- 키보드 ----------


def test_arrow_moves_to_a_neighbour(view, network):
    view.select_station(station(network, "강남"))
    before = view.selected_station.name
    press(view, Qt.Key.Key_Right)
    after = view.selected_station.name
    assert after != before
    neighbours = view._neighbors[before]
    assert after in neighbours  # 바로 옆 역으로만 움직인다


def test_every_arrow_key_finds_a_station_or_stays(view, network):
    view.select_station(station(network, "강남"))
    for key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
        target = view.arrow_target(station_map.ARROW_DIRECTIONS[key])
        assert target is not None, key
        assert target.name != view.selected_station.name


def test_enter_opens_the_panel(view, network):
    received = []
    view.station_activated.connect(received.append)
    view.select_station(station(network, "강남"))
    press(view, Qt.Key.Key_Return)
    assert [s.name for s in received] == ["강남"]


def test_escape_asks_to_close_the_panel(view, network):
    received = []
    view.panel_close_requested.connect(lambda: received.append(True))
    view.select_station(station(network, "강남"))
    press(view, Qt.Key.Key_Escape)
    assert received == [True]


def test_map_takes_keyboard_focus(view):
    assert view.focusPolicy() == Qt.FocusPolicy.StrongFocus


# ---------- 호선 배지 ----------


def test_badge_text_is_readable_on_every_line(network):
    """① 흰 글자 ② 먹색 글자 ③ 배경을 조금 어둡게 — 어느 쪽이든 4.5:1 을 넘긴다."""
    for line in network[0]:
        text, background = badge_palette(line.color)
        assert contrast_ratio(text, background) >= AA_NORMAL_TEXT, line.name


def test_badge_rule_three_steps(network):
    colors = {line.name: line.color for line in network[0]}
    assert badge_palette(colors["1호선"]) == (WHITE, "#0052A4")  # 흰 글자가 읽힌다
    assert badge_palette(colors["2호선"]) == (INK, "#00A84D")  # 흰 글자는 안 읽혀 먹색
    text, background = badge_palette(colors["5호선"])
    assert text == WHITE and background == "#8A619B"  # 둘 다 안 되면 배경을 10% 어둡게
    assert darken("#996CAC") == "#8A619B"
    assert contrast_ratio(WHITE, colors["5호선"]) < AA_NORMAL_TEXT  # 원래 색으로는 부족하다


def test_badge_widget_uses_the_computed_colors(app, network):
    line = next(line for line in network[0] if line.name == "2호선")
    badge = LineBadge(line)
    style = badge.styleSheet()
    assert f"background: {line.color}" in style  # 배지는 노선색을 그대로
    assert f"color: {INK}" in style  # 글자만 대비가 되는 먹색으로


def test_line_strokes_keep_the_official_colours(view):
    """노선 선과 역 점 테두리는 공식 노선색을 그대로 쓴다 (배지와 달리 글자가 없다)."""
    item = view.station_items["역삼"]  # 2호선 한 곳만 지나는 역 (선택하지 않은 상태)
    assert not item.station.is_transfer and not item.is_selected
    line = next(line for line in view.lines if line.id == item.station.lines[0])
    assert item.pen().color().name().lower() == line.color.lower()
    strokes = {path.pen().color().name().lower() for path, _width in view._line_items}
    assert {line.color.lower() for line in view.lines} <= strokes


# ---------- 글자 크기·간격 ----------


def test_font_sizes_follow_the_type_scale():
    sizes = {int(value) for value in re.findall(r"font-size:\s*(\d+)px", qss.load("modern_gray"))}
    assert sizes == {12, 13, 15, 20, 26, 28}
    assert 11 not in sizes  # 11px 설명 글자는 쓰지 않는다


def test_spacing_values_sit_on_the_four_pixel_grid():
    text = qss.load("modern_gray")
    odd = set()
    for match in re.finditer(r"(?:padding|margin|spacing|border-radius)[a-z-]*:\s*([^;]+);", text):
        for value in re.findall(r"(\d+)px", match.group(1)):
            number = int(value)
            if number in (1, 2, 3, 999):
                continue  # 1px 선, 작은 표식, 알약 모양
            if number % 4:
                odd.add((match.group(0)[:40], number))
    assert odd == set()


def test_radius_scale_is_limited():
    radii = {
        int(value)
        for value in re.findall(r"border-radius:\s*(\d+)px", qss.load("modern_gray"))
    }
    assert radii <= {4, 8, 12, 999}


def test_body_text_contrast_is_enough():
    from metro_insight.app.styles.tokens import check_contrast

    assert min(check_contrast("light").values()) >= AA_NORMAL_TEXT