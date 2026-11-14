"""예측 데이터가 없는 (호선, 역) 쌍을 노선도와 패널이 하나도 빠짐없이 다루는지 확인한다.

노선도는 1~8호선 420쌍을 그리지만 예측기는 학습 자료에 있는 273쌍만 계산할 수 있다.
남은 147쌍은 점선·회색으로 그리고, 골랐을 때 "예측 데이터 없음"이라 알린다.

같은 역을 다르게 적은 경우('이수'↔'총신대입구(이수)', '잠실'↔'잠실(송파구청)')는
예측기와 같은 규칙으로 붙여서, 데이터가 있는 역을 없다고 표시하지 않는다.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 화면 없이 위젯 검사

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from metro_insight.app.coverage import (  # noqa: E402
    MAP_NAME_BY_DATA_STATION,
    canonical,
    format_count,
    load_coverage,
    normalize,
)
from metro_insight.app.map_overlays import CoverageBanner, LineLegend, SelectionCard  # noqa: E402
from metro_insight.app.station_map import StationMapView  # noqa: E402
from metro_insight.app.station_panel import StationPanel, WeatherLoader  # noqa: E402
from metro_insight.app.subway_data import load_network  # noqa: E402
from metro_insight.data.weather import HourlyWeather  # noqa: E402

SUNNY = HourlyWeather("2026-10-12", 9, 21.6, rainfall=0, snowfall=0)

# models/boarding_history.npz 의 (호선, 역) 목록과 노선도를 같은 열쇠로 비교한 값
EXPECTED_PAIRS = 420
EXPECTED_KNOWN = 273
EXPECTED_MISSING = 147
EXPECTED_FULL_GAP_STATIONS = 136  # 어느 호선에도 데이터가 없는 역
EXPECTED_PER_LINE = {
    "1호선": 92,
    "2호선": 1,
    "3호선": 11,
    "4호선": 25,
    "5호선": 0,
    "6호선": 2,
    "7호선": 11,
    "8호선": 5,
}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def network():
    return load_network()


@pytest.fixture(scope="module")
def coverage(network):
    return load_coverage(lines=network[0], stations=network[1])


@pytest.fixture(scope="module")
def map_view(app, coverage):
    return StationMapView(coverage=coverage)


def station(network, name):
    return next(s for s in network[1] if s.name == name)


# ---------- 목록 계산 ----------


def test_pair_counts(coverage):
    assert len(coverage.pairs) == EXPECTED_PAIRS
    assert len(coverage.known) == EXPECTED_KNOWN
    assert len(coverage.missing) == EXPECTED_MISSING
    assert coverage.known <= coverage.pairs


def test_missing_pairs_per_line(coverage):
    for line, count in EXPECTED_PER_LINE.items():
        assert coverage.line_missing(line) == count, line
    assert sum(EXPECTED_PER_LINE.values()) == EXPECTED_MISSING


def test_line_totals_add_up(coverage):
    assert sum(coverage.line_total(line) for line in coverage.line_names) == EXPECTED_PAIRS
    for line in coverage.line_names:
        assert coverage.line_total(line) == coverage.line_known(line) + coverage.line_missing(line)


def test_first_line_is_the_thin_one(coverage):
    """1호선은 102개 역 중 10개만 예측할 수 있어 안내 띠 대상이다."""
    assert coverage.line_total("1호선") == 102
    assert coverage.line_known("1호선") == 10
    assert coverage.line_gap_ratio("1호선") > 0.5
    assert coverage.alert_lines() == ("1호선",)
    assert format_count(10, 102) == "10/102"


def test_names_are_matched_across_spellings():
    assert normalize("총신대입구(이수)") == "총신대입구"
    assert normalize("잠실(송파구청)") == "잠실"
    assert MAP_NAME_BY_DATA_STATION[("7호선", "이수")] == "총신대입구"
    assert canonical("7호선", "이수") == canonical("7호선", "총신대입구(이수)")
    assert canonical("2호선", "잠실(송파구청)") == ("2호선", "잠실")


def test_differently_spelled_pair_is_not_marked_missing(coverage, network):
    """7호선 원본은 '이수'다. 이 역을 '데이터 없음'으로 그리면 있는 데이터를 없다고 말하게 된다."""
    assert ("7호선", "총신대입구(이수)") in coverage.known
    assert not coverage.is_missing("7호선", "총신대입구(이수)")
    assert coverage.has_data("총신대입구(이수)")
    # 4호선 쪽은 원본 이름이 그대로라 두 호선 모두 데이터가 있다
    assert not coverage.is_missing("4호선", "총신대입구(이수)")
    assert coverage.missing_lines_of("총신대입구(이수)") == ()


def test_every_missing_pair_belongs_to_a_station_on_the_map(coverage, network):
    """147쌍이 모두 노선도의 역과 호선으로 이루어져 있는지 (짝이 맞지 않는 계산이 없는지)."""
    drawn = {(line.name, s.name) for s in network[1] for line in network[0] if line.id in s.lines}
    assert drawn == set(coverage.pairs)
    for pair in coverage.missing:
        assert pair in drawn


# ---------- 노선도 ----------


def test_every_missing_pair_is_drawn_as_a_gap(map_view, coverage):
    """지도 위 역마다 '데이터 없는 호선'이 정확히 표시되고, 147쌍이 하나도 빠지지 않는다."""
    names = {line.id: line.name for line in map_view.lines}
    seen = set()
    for name, item in map_view.station_items.items():
        expected = tuple(
            names[line_id] for line_id in item.station.lines if coverage.is_missing(names[line_id], name)
        )
        assert item.missing_lines == expected, name
        assert item.is_gap is (len(expected) == len(item.station.lines))
        seen.update((line, name) for line in expected)
    assert seen == coverage.missing
    assert len(seen) == EXPECTED_MISSING


def test_gap_stations_are_drawn_dotted_and_grey(map_view):
    """데이터가 하나도 없는 역은 점선·회색. 환승역만 기존 규칙대로 진한 테두리를 쓴다."""
    dotted = 0
    gap_transfers = 0
    for name, item in map_view.station_items.items():
        if not item.is_gap:
            continue
        if item.station.is_transfer:
            gap_transfers += 1
            continue
        assert item.pen().style() == Qt.PenStyle.DashLine, name
        assert item.pen().color().name().lower() == "#b8bcc2", name
        assert item.label._color == "#6b7078", name  # 이름도 회색
        dotted += 1
    assert dotted + gap_transfers == EXPECTED_FULL_GAP_STATIONS
    assert dotted == EXPECTED_FULL_GAP_STATIONS - gap_transfers


def test_stations_with_data_stay_solid(map_view, coverage):
    solid = [name for name, item in map_view.station_items.items() if not item.is_gap]
    assert len(solid) == len(map_view.station_items) - EXPECTED_FULL_GAP_STATIONS
    item = map_view.station_items["강남"]
    assert item.pen().style() == Qt.PenStyle.SolidLine
    assert not item.missing_lines


def test_partial_gap_station_keeps_its_normal_look(map_view, coverage):
    """데이터가 있는 호선이 하나라도 있으면 역 모양은 그대로 두고 툴팁으로만 알린다."""
    item = map_view.station_items["까치산"]
    assert item.missing_lines == ("2호선",)
    assert not item.is_gap
    assert item.brush().color().name().lower() == "#ffffff"  # 회색으로 칠하지 않는다


def test_tooltip_tells_which_line_has_no_data(map_view):
    assert "예측 데이터 없음: 1호선" in map_view.station_items["석계"].toolTip()
    assert "예측 데이터 없음" in map_view.station_items["가능"].toolTip()
    assert "예측 데이터 없음" not in map_view.station_items["강남"].toolTip()
    assert "예측 데이터 없음" not in map_view.station_items["총신대입구(이수)"].toolTip()


# ---------- 지도 위 카드 ----------


def test_legend_shows_counts_and_gap_total(app, network, coverage):
    legend = LineLegend(network[0], coverage)
    counts = [label.text() for label in legend.findChildren(QLabel, "legendCount")]
    assert counts[:8] == [
        format_count(coverage.line_known(line.name), coverage.line_total(line.name))
        for line in network[0]
    ]
    assert network[0][0].name == "1호선" and counts[0] == "10/102"
    assert f"예측 데이터 없음 {EXPECTED_MISSING}곳" in counts
    thin = [label.text() for label in legend.findChildren(QLabel, "legendCount") if label.property("thin")]
    assert thin == ["10/102"]  # 데이터가 드문 호선만 경고색


def test_selection_card_marks_missing_line(app, network, coverage):
    card = SelectionCard(network[0], coverage)
    card.show_station(station(network, "가능"))
    assert not card.gap_tag.isHidden()
    assert "1호선 예측 데이터 없음" in card.gap_tag.text()

    card.show_station(station(network, "강남"))
    assert card.gap_tag.isHidden()


def test_banner_lists_thin_line_numbers(app, coverage):
    banner = CoverageBanner()
    banner.show_lines(coverage)
    assert not banner.isHidden()
    assert "1호선은 102개 역 중 10개 역만" in banner.message
    banner.show_lines(coverage, ("5호선",))  # 데이터가 다 있는 호선만 고르면 띠를 숨긴다
    assert banner.isHidden()


# ---------- 역 정보 패널 ----------


def test_panel_shows_no_data_card_for_gap_station(app, network, coverage):
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY), coverage=coverage)
    panel.set_station(station(network, "가능"))
    assert not panel._gap_note.isHidden()
    assert "예측 데이터 없음" in panel._gap_note.text()
    assert "1호선" in panel._gap_note.text()

    panel.set_station(station(network, "강남"))
    assert panel._gap_note.isHidden()


def test_panel_builds_the_gap_table_without_being_given_one(app, network):
    """coverage 를 넘기지 않아도 역을 고를 때 같은 표를 스스로 만든다."""
    panel = StationPanel(*network, weather=WeatherLoader(lambda tm: SUNNY))
    panel.set_station(station(network, "가능"))
    assert not panel._gap_note.isHidden()
    assert set(panel._coverage_table().missing) == set(load_coverage().missing)