"""지하철 노선도 배치 (가로·세로·45° 선만 쓰는 공식 노선도 스타일).

1. 역 간격 고르기: 실제 위치에서 출발해, 연결된 역끼리 비슷한 간격이 되도록 stress majorization으로
   다시 배치한다. 도심 밀집은 풀리고 외곽의 긴 구간은 줄어든다.
2. 각도 맞추기: 환승역·종점·분기역(기준점) 사이 구간이 가로·세로·45° 방향이 되도록 기준점 위치를
   최소제곱으로 조정한다.
3. 그리기: 기준점 사이를 많아야 한 번 꺾이는 가로·세로·45° 선으로 잇고, 그 위에 역을 같은 간격으로 놓는다.
"""
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from metro_insight.app.subway_data import Line, Station

KM_PER_DEG = 111.0
UNIT = 40.0  # 씬 좌표에서 기본 역 간격

# 1. 역 간격: 목표 간격 = (실제 거리 km)^EDGE_LENGTH_POWER. 0이면 모두 같은 간격, 1이면 실제 비율
EDGE_LENGTH_POWER = 0.15
STRESS_ITERATIONS = 150  # 150회면 400회와 위치 차이 1 미만
# 도심에서 TAIL_START_KM보다 먼 구간은 간격을 줄여 1호선 천안·연천 방면 같은 긴 꼬리를 짧게 한다
TAIL_START_KM = 15.0
TAIL_HALF_KM = 25.0  # 여기서 TAIL_HALF_KM만큼 더 멀어질 때마다 간격이 절반씩 가까워진다

# 2. 각도 맞추기: 반복 횟수와, 원래 자리에 머무르려는 힘, 구간 안 역 하나당 최소 간격
OCTILINEAR_ITERATIONS = 30
ANCHOR_STAY_WEIGHT = 0.25
MIN_STATION_GAP = 1.0

# 3. 가로·세로·45°에서 이 각도 이내로 벗어난 구간은 꺾지 않고 곧게 잇는다
STRAIGHT_TOLERANCE_DEG = 3.0
DETOUR_OFFSET = 0.45  # 다른 노선과 겹치는 직선 구간을 옆으로 비켜 그리는 폭 (UNIT 배수)
BEND_COST = 0.3  # 경로 고를 때 꺾임 하나당 벌점: 같은 조건이면 단순한 경로를 고른다

FOCUS_LINE = "2"  # 처음 화면은 2호선 일대
DIRECTIONS = [(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)]

Point = tuple[float, float]


@dataclass(frozen=True)
class SchematicLayout:
    positions: dict[str, Point]  # 역 이름 → 씬 좌표 (y는 아래 방향)
    paths: dict[tuple[str, str, str], list[Point]]  # (호선 id, 역 a, 역 b) → a에서 b까지 꺾임점 포함 경로
    focus: tuple[float, float, float, float]  # 처음 화면에 보여줄 범위 (x, y, w, h)


def metro_layout(lines: list[Line], stations: list[Station]) -> SchematicLayout:
    names = [s.name for s in stations]
    index = {n: i for i, n in enumerate(names)}
    geo = _geo_km(stations)

    edges = sorted({tuple(sorted(e)) for line in lines for e in line.edges})
    center = geo.mean(axis=0)
    lengths = np.array([_edge_length(geo[index[a]], geo[index[b]], center) for a, b in edges])
    x = _stress_layout(geo, [(index[a], index[b]) for a, b in edges], lengths)
    pos = {n: (float(x[i, 0]), float(x[i, 1])) for i, n in enumerate(names)}  # y는 북쪽

    chains = _all_chains(lines, stations)
    anchors = sorted({c[0] for _, c in chains} | {c[-1] for _, c in chains})
    links = [(c[0], c[-1], MIN_STATION_GAP * (len(c) - 1)) for _, c in chains]
    anchor_pos = _octilinearize({a: pos[a] for a in anchors}, links)

    positions: dict[str, Point] = {}
    paths: dict[tuple[str, str, str], list[Point]] = {}
    drawn: list[tuple[Point, Point]] = []
    for line_id, chain in sorted(chains, key=lambda c: -len(c[1])):
        route = _route(anchor_pos[chain[0]], anchor_pos[chain[-1]], drawn, anchor_pos)
        drawn.extend(zip(route, route[1:], strict=False))
        points, pieces = _place_along(route, len(chain) - 1)
        for name, p in zip(chain, points, strict=True):
            positions.setdefault(name, p)
        for (a, b), piece in zip(zip(chain, chain[1:], strict=False), pieces, strict=True):
            paths[(line_id, a, b)] = piece

    def to_scene(p: Point) -> Point:
        return (p[0] * UNIT, -p[1] * UNIT)

    focus = [to_scene(positions[n]) for line in lines if line.id == FOCUS_LINE for e in line.edges for n in e]
    xs, ys = [p[0] for p in focus], [p[1] for p in focus]
    return SchematicLayout(
        {n: to_scene(p) for n, p in positions.items()},
        {k: [to_scene(p) for p in v] for k, v in paths.items()},
        (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
    )


# ---------- 1. 역 간격 ----------


def _geo_km(stations: list[Station]) -> np.ndarray:
    lat0 = sum(s.lat for s in stations) / len(stations)
    lon_km = KM_PER_DEG * math.cos(math.radians(lat0))
    return np.array([(s.lon * lon_km, s.lat * KM_PER_DEG) for s in stations])


def _dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _edge_length(p, q, center) -> float:
    """두 역 사이 목표 간격: 실제 거리를 약하게 반영하고, 도심에서 먼 구간은 줄인다."""
    length = max(_dist(p, q), 0.3) ** EDGE_LENGTH_POWER
    far = _dist(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2), center) - TAIL_START_KM
    return length * 0.5 ** (max(far, 0.0) / TAIL_HALF_KM)


def _stress_layout(geo: np.ndarray, edges: list[tuple[int, int]], lengths: np.ndarray) -> np.ndarray:
    """그래프 최단 거리를 목표 거리로 하는 SMACOF stress majorization (실제 위치에서 시작)."""
    n = len(geo)
    rows, cols = zip(*edges, strict=True)
    graph = coo_matrix((lengths, (rows, cols)), shape=(n, n))
    target = shortest_path(graph, directed=False)

    weights = np.zeros_like(target)
    nonzero = target > 0
    weights[nonzero] = target[nonzero] ** -2
    v = -weights
    np.fill_diagonal(v, weights.sum(axis=1))
    v_pinv = np.linalg.pinv(v)

    edge_km = np.array([_dist(geo[a], geo[b]) for a, b in edges])
    x = (geo - geo.mean(axis=0)) * (lengths.mean() / edge_km.mean())
    for _ in range(STRESS_ITERATIONS):
        sq = (x**2).sum(axis=1)
        dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0))
        b = -weights * target / np.maximum(dist, 1e-9)
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(b, -b.sum(axis=1))
        x = v_pinv @ (b @ x)
    return x - x.mean(axis=0)


# ---------- 2. 각도 맞추기 ----------


def _all_chains(lines: list[Line], stations: list[Station]) -> list[tuple[str, list[str]]]:
    """(호선 id, 기준점에서 다음 기준점까지의 역 목록). 기준점 = 환승역·종점·분기역."""
    transfers = {s.name for s in stations if s.is_transfer}
    result = []
    for line in lines:
        adj = defaultdict(list)
        for a, b in line.edges:
            adj[a].append(b)
            adj[b].append(a)
        fixed = transfers | {n for n, nbrs in adj.items() if len(nbrs) != 2}
        seen = set()
        for start in sorted(fixed & set(adj)):
            for nxt in adj[start]:
                chain = [start, nxt]
                while chain[-1] not in fixed:
                    a, b = adj[chain[-1]]
                    chain.append(b if a == chain[-2] else a)
                key = frozenset(frozenset(e) for e in zip(chain, chain[1:], strict=False))  # 방향 무관
                if key not in seen:
                    seen.add(key)
                    result.append((line.id, chain))
    return result


def _nearest_direction(v) -> Point:
    return max(DIRECTIONS, key=lambda d: d[0] * v[0] + d[1] * v[1])


def _octilinearize(anchor_pos: dict[str, Point], links: list[tuple[str, str, float]]) -> dict[str, Point]:
    """기준점 사이 방향이 가로·세로·45°에 가까워지도록 기준점 위치를 최소제곱으로 반복 조정한다.

    목표 벡터 = (가장 가까운 팔방위) × (처음 길이와 최소 길이 중 큰 값): 방향만 맞추고 간격은 유지한다.
    """
    names = list(anchor_pos)
    idx = {n: i for i, n in enumerate(names)}
    start = np.array([anchor_pos[n] for n in names])
    targets = [max(_dist(anchor_pos[a], anchor_pos[b]), min_len) for a, b, min_len in links if a != b]
    links = [(idx[a], idx[b]) for a, b, _ in links if a != b]

    # 고정된 행렬: 구간마다 (p_b - p_a) 두 줄 + 기준점마다 제자리 유지 두 줄
    n = len(names)
    matrix = np.zeros((2 * len(links) + 2 * n, 2 * n))
    for row, (a, b) in enumerate(links):
        for axis in range(2):
            matrix[2 * row + axis, 2 * b + axis] = 1
            matrix[2 * row + axis, 2 * a + axis] = -1
    matrix[2 * len(links) :, :] = ANCHOR_STAY_WEIGHT * np.eye(2 * n)
    stay = ANCHOR_STAY_WEIGHT * start.reshape(-1)
    solver = np.linalg.pinv(matrix)

    p = start.copy()
    for _ in range(OCTILINEAR_ITERATIONS):
        rhs = np.empty(2 * len(links))
        for row, ((a, b), length) in enumerate(zip(links, targets, strict=True)):
            d = _nearest_direction(p[b] - p[a])
            rhs[2 * row], rhs[2 * row + 1] = d[0] * length, d[1] * length
        p = (solver @ np.concatenate([rhs, stay])).reshape(-1, 2)
    return {name: (float(p[i, 0]), float(p[i, 1])) for i, name in enumerate(names)}


# ---------- 3. 그리기 ----------


def _route(a: Point, b: Point, drawn, anchor_pos) -> list[Point]:
    """a→b를 가로·세로·45° 선으로 잇는다. 후보 중 이미 그린 선·역과 덜 겹치는 경로를 고른다.

    후보: 곧은 선(방향이 거의 맞을 때) 또는 한 번 꺾은 선 두 가지,
    그리고 곧은 선이 다른 노선과 겹칠 때를 위한 옆으로 비켜 가는 선 두 가지.
    """
    v = (b[0] - a[0], b[1] - a[1])
    length = math.hypot(*v)
    if length < 1e-9:
        return [a, b]
    k = int((math.atan2(v[1], v[0]) % (2 * math.pi)) // (math.pi / 4))
    d1, d2 = DIRECTIONS[k % 8], DIRECTIONS[(k + 1) % 8]
    det = d1[0] * d2[1] - d1[1] * d2[0]
    s = (v[0] * d2[1] - v[1] * d2[0]) / det  # v = s·d1 + t·d2
    t = (d1[0] * v[1] - d1[1] * v[0]) / det

    deviation = math.degrees(math.atan2(v[1], v[0])) % 45  # 가장 가까운 팔방위에서 벗어난 각도
    deviation = min(deviation, 45 - deviation)
    if deviation > STRAIGHT_TOLERANCE_DEG:
        options = [
            [a, (a[0] + s * d1[0], a[1] + s * d1[1]), b],
            [a, (a[0] + t * d2[0], a[1] + t * d2[1]), b],
        ]
    else:
        options = [[a, b]]
        w = DETOUR_OFFSET
        if length > 3 * w:
            d = (v[0] / length, v[1] / length)
            for sign in (1, -1):
                n = (-d[1] * sign, d[0] * sign)  # 45°로 빠져나가 나란히 가다가 45°로 돌아온다
                options.append([
                    a,
                    (a[0] + w * (d[0] + n[0]), a[1] + w * (d[1] + n[1])),
                    (b[0] + w * (n[0] - d[0]), b[1] + w * (n[1] - d[1])),
                    b,
                ])
    return min(options, key=lambda r: _route_penalty(r, drawn, anchor_pos, a, b))


def _route_penalty(route, drawn, anchor_pos, a, b) -> float:
    penalty = BEND_COST * (len(route) - 2)
    others = [p for p in anchor_pos.values() if p not in (a, b)]
    for p in others:
        for bend in route[1:-1]:  # 꺾인 점이 다른 기준점 근처에 오지 않게
            penalty += max(0.0, 0.8 - _dist(bend, p)) * 10
        for q, r in zip(route, route[1:], strict=False):  # 다른 역 위를 지나가지 않게
            if _point_segment_distance(p, q, r) < 0.3:
                penalty += 3
    for p, q in zip(route, route[1:], strict=False):
        for r, s in drawn:
            if _segments_cross(p, q, r, s):
                penalty += 1
            penalty += _overlap(p, q, r, s) * 5
    return penalty


def _point_segment_distance(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _segments_cross(p, q, r, s) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    if min(_dist(p, r), _dist(p, s), _dist(q, r), _dist(q, s)) < 1e-6:
        return False  # 끝점을 공유하는 건 교차가 아니다
    return orient(p, q, r) * orient(p, q, s) < 0 and orient(r, s, p) * orient(r, s, q) < 0


def _overlap(p, q, r, s) -> float:
    """거의 평행하고 가까이 붙은 두 선분이 나란히 겹치는 길이."""
    length, other = _dist(p, q), _dist(r, s)
    if length < 1e-9 or other < 1e-9:
        return 0.0
    u = ((q[0] - p[0]) / length, (q[1] - p[1]) / length)
    w = ((s[0] - r[0]) / other, (s[1] - r[1]) / other)
    if abs(u[0] * w[1] - u[1] * w[0]) > 0.17:
        return 0.0  # 10° 넘게 기울어짐
    def off_line(pt):
        return abs(u[0] * (pt[1] - p[1]) - u[1] * (pt[0] - p[0]))

    if min(off_line(r), off_line(s)) > 0.3:
        return 0.0  # 나란하지만 떨어져 있음
    t0 = u[0] * (r[0] - p[0]) + u[1] * (r[1] - p[1])
    t1 = u[0] * (s[0] - p[0]) + u[1] * (s[1] - p[1])
    return max(0.0, min(length, max(t0, t1)) - max(0.0, min(t0, t1)))


def _place_along(route: list[Point], segments: int) -> tuple[list[Point], list[list[Point]]]:
    """경로 위에 segments+1개의 점을 같은 간격으로 놓고, 이웃한 두 점 사이 경로(꺾임점 포함)도 돌려준다."""
    cumulative = [0.0]
    for p, q in zip(route, route[1:], strict=False):
        cumulative.append(cumulative[-1] + _dist(p, q))
    total = cumulative[-1]

    def point_at(offset: float) -> Point:
        for i in range(len(route) - 1):
            if offset <= cumulative[i + 1] or i == len(route) - 2:
                span = cumulative[i + 1] - cumulative[i]
                t = 0.0 if span == 0 else min(max((offset - cumulative[i]) / span, 0.0), 1.0)
                p, q = route[i], route[i + 1]
                return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
        return route[-1]

    offsets = [total * i / segments for i in range(segments + 1)]
    points = [point_at(o) for o in offsets]
    pieces = []
    for i in range(segments):
        bends = [route[j] for j in range(1, len(route) - 1) if offsets[i] < cumulative[j] < offsets[i + 1]]
        pieces.append([points[i], *bends, points[i + 1]])
    return points, pieces
