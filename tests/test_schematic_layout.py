import json
import math
import sys
from dataclasses import replace

import pytest

from metro_insight.app import schematic_layout
from metro_insight.app.schematic_layout import STRAIGHT_TOLERANCE_DEG, metro_layout
from metro_insight.app.subway_data import load_network

TOLERANCE = 1e-9  # 캐시로 읽은 값과 계산한 값의 허용 차이 (사실상 같아야 한다)

# 모듈 로드 시점(어떤 픽스처도 metro_layout 을 불러 파일을 덮어쓰기 전)에
# 저장소에 커밋된 캐시 원문을 스냅샷으로 확보해 둔다.
_INITIAL_CACHE_TEXT = (
    schematic_layout.CACHE_PATH.read_text(encoding="utf-8")
    if schematic_layout.CACHE_PATH.exists()
    else None
)


@pytest.fixture(scope="module")
def network():
    return load_network()


@pytest.fixture(scope="module")
def layout(network):
    return metro_layout(*network)


@pytest.fixture(scope="module")
def live(network):
    """캐시를 거치지 않은 실시간 계산 결과 (여러 테스트가 함께 쓴다)."""
    return schematic_layout._compute_layout(*network)


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


# ---------- 배치 캐시 ----------


def assert_same_layout(cached, computed):
    """두 배치가 허용 오차 안에서 같은지 (역·경로·보여 줄 범위 전부)."""
    assert set(cached.positions) == set(computed.positions)
    for name, (x, y) in computed.positions.items():
        cx, cy = cached.positions[name]
        assert abs(cx - x) <= TOLERANCE and abs(cy - y) <= TOLERANCE, name

    assert set(cached.paths) == set(computed.paths)
    for key, points in computed.paths.items():
        got = cached.paths[key]
        assert len(got) == len(points), key
        for (gx, gy), (x, y) in zip(got, points, strict=True):
            assert abs(gx - x) <= TOLERANCE and abs(gy - y) <= TOLERANCE, key

    for got, want in zip(cached.focus, computed.focus, strict=True):
        assert abs(got - want) <= TOLERANCE


def test_fingerprint_is_deterministic_and_order_independent(network):
    lines, stations = network
    base = schematic_layout._fingerprint(lines, stations)
    assert base == schematic_layout._fingerprint(list(reversed(lines)), list(reversed(stations)))
    assert len(base) == 64 and int(base, 16) >= 0  # sha256 16진수


def test_fingerprint_follows_input_changes(network):
    lines, stations = network
    base = schematic_layout._fingerprint(lines, stations)

    moved = [replace(stations[0], lat=stations[0].lat + 1e-6), *stations[1:]]
    assert schematic_layout._fingerprint(lines, moved) != base  # 역 좌표가 달라지면 다른 지문

    renamed = [replace(stations[0], name=stations[0].name + "역"), *stations[1:]]
    assert schematic_layout._fingerprint(lines, renamed) != base

    first = lines[0]
    reordered = [replace(first, edges=tuple(reversed(first.edges))), *lines[1:]]
    assert schematic_layout._fingerprint(reordered, stations) == base  # 간선 순서만 뒤집힌 것은 같은 입력


def test_committed_cache_matches_live_computation(network, live):
    """저장소에 둔 layout_cache.json 이 테스트 실행 전부터 지금 입력·계산과 맞는지 확인한다."""
    assert _INITIAL_CACHE_TEXT is not None, "layout_cache.json 이 저장소에 없다"
    fingerprint = schematic_layout._fingerprint(*network)
    data = json.loads(_INITIAL_CACHE_TEXT)
    assert data["schema"] == schematic_layout.CACHE_SCHEMA
    assert data["fingerprint"] == fingerprint, (
        "layout_cache.json 이 현재 입력과 다릅니다. 지우고 metro_layout() 을 다시 실행해 갱신하세요."
    )

    assert_same_layout(metro_layout(*network), live)


def test_cache_hit_reuses_stored_layout_without_computing(tmp_path, network, monkeypatch, live):
    """캐시가 맞으면 비싼 계산(=_stress_layout)을 아예 부르지 않는다."""
    monkeypatch.setattr(schematic_layout, "CACHE_PATH", tmp_path / "layout_cache.json")
    schematic_layout._write_cache(schematic_layout._fingerprint(*network), live)

    def refuse(*args, **kwargs):
        raise AssertionError("캐시가 맞는데 다시 계산했다")

    monkeypatch.setattr(schematic_layout, "_stress_layout", refuse)
    monkeypatch.setattr(schematic_layout, "_octilinearize", refuse)
    assert_same_layout(schematic_layout.metro_layout(*network), live)


@pytest.mark.parametrize(
    "payload",
    [
        "{ 깨진 json",  # 파일이 깨짐
        json.dumps({"schema": "metro-layout-cache/0", "fingerprint": "x"}),  # 스키마가 다름
        json.dumps({"schema": "metro-layout-cache/1", "fingerprint": "0" * 64}),  # 지문이 다름
        json.dumps({"schema": "metro-layout-cache/1"}),  # 항목이 빠짐
    ],
)
def test_stale_or_broken_cache_is_recomputed(tmp_path, network, monkeypatch, live, payload):
    path = tmp_path / "layout_cache.json"
    path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(schematic_layout, "CACHE_PATH", path)

    layout = metro_layout(*network)
    assert_same_layout(layout, live)

    # 계산한 뒤에는 다음 기동에서 바로 쓰도록 캐시를 다시 써 둔다
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == schematic_layout.CACHE_SCHEMA
    assert data["fingerprint"] == schematic_layout._fingerprint(*network)


def test_cache_hit_does_not_import_scipy(tmp_path, network, monkeypatch, live):
    """캐시로 읽을 때는 scipy 를 읽지 않는다 (읽는 데만 1초 가까이 걸린다)."""
    monkeypatch.setattr(schematic_layout, "CACHE_PATH", tmp_path / "layout_cache.json")
    schematic_layout._write_cache(schematic_layout._fingerprint(*network), live)

    for name in [name for name in sys.modules if name.split(".")[0] == "scipy"]:
        monkeypatch.delitem(sys.modules, name)

    metro_layout(*network)  # 캐시 적중 경로

    loaded = sorted(name for name in sys.modules if name.split(".")[0] == "scipy")
    assert loaded == [], f"캐시로 읽었는데 scipy 를 읽었다: {loaded}"


def test_missing_cache_is_computed_and_written(tmp_path, network, monkeypatch, live):
    path = tmp_path / "layout_cache.json"
    monkeypatch.setattr(schematic_layout, "CACHE_PATH", path)
    assert not path.exists()

    layout = metro_layout(*network)
    assert path.exists()
    assert_same_layout(schematic_layout._read_cache(schematic_layout._fingerprint(*network)), layout)
    assert_same_layout(layout, live)


def test_cache_write_failure_still_returns_layout(tmp_path, network, monkeypatch, live):
    """캐시를 저장할 수 없어도(읽기 전용 등) 계산 결과는 그대로 돌려준다."""
    blocked = tmp_path / "no-such-dir" / "layout_cache.json"
    monkeypatch.setattr(schematic_layout, "CACHE_PATH", blocked)

    layout = metro_layout(*network)
    assert_same_layout(layout, live)
    assert not blocked.exists()