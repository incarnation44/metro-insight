import math

import pytest

from metro_insight.app.schematic_layout import STRAIGHT_TOLERANCE_DEG, metro_layout
from metro_insight.app.subway_data import load_network


@pytest.fixture(scope="module")
def network():
    return load_network()


@pytest.fixture(scope="module")
def layout(network):
    return metro_layout(*network)


def test_every_station_and_edge_is_placed(network, layout):
    lines, stations = network
    assert set(layout.positions) == {s.name for s in stations}
    for line in lines:
        for a, b in line.edges:
            assert (line.id, a, b) in layout.paths or (line.id, b, a) in layout.paths


def test_paths_use_only_horizontal_vertical_and_diagonal_segments(layout):
    for points in layout.paths.values():
        for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
            angle = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 45
            assert min(angle, 45 - angle) <= STRAIGHT_TOLERANCE_DEG + 1e-6


def test_stations_do_not_collide(layout):
    points = list(layout.positions.values())
    closest = min(math.dist(p, q) for i, p in enumerate(points) for q in points[i + 1 :])
    assert closest > 5  # 씬 단위 (기본 역 간격 40)
