"""노선도용 지하철 역·노선 데이터 (1~8호선).

역 좌표와 역 순서는 OpenStreetMap의 노선(route) 정보에서 만든다 (© OpenStreetMap contributors, ODbL).
급행·특급 운행 계통은 역을 건너뛰므로 제외하고,
일반 운행 계통의 정차역 순서를 합쳐 지선까지 포함한 노선을 만든다.

재생성: python -m metro_insight.app.subway_data
"""
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from pathlib import Path

import requests

NETWORK_FILE = files("metro_insight.app") / "resources" / "subway_network.json"

LINES = {
    "1": ("1호선", "#0052A4"),
    "2": ("2호선", "#00A84D"),
    "3": ("3호선", "#EF7C1C"),
    "4": ("4호선", "#00A5DE"),
    "5": ("5호선", "#996CAC"),
    "6": ("6호선", "#CD7C2F"),
    "7": ("7호선", "#747F00"),
    "8": ("8호선", "#E6186C"),
}

# 역을 건너뛴 구간 판정: 직접 연결을 뺀 우회 경로가 이 홉 수 이내, 이 거리 비율 이하이면 건너뛴 구간으로 본다
SHORTCUT_MAX_HOPS = 4
SHORTCUT_DISTANCE_RATIO = 1.25

# 호선마다 역 이름이 다른 환승역은 하나의 역으로 합친다
STATION_ALIASES = {"총신대입구": "총신대입구(이수)", "이수": "총신대입구(이수)"}

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_BBOX = "36.5,126.3,38.2,127.6"  # 수도권 + 1호선 천안·아산 구간
EXPRESS_PATTERN = re.compile("급행|특급")
STOP_ROLES = ("stop", "stop_entry_only", "stop_exit_only")


@dataclass(frozen=True)
class Station:
    name: str
    lat: float
    lon: float
    lines: tuple[str, ...]  # 지나는 호선 id ("1" ~ "8")

    @property
    def is_transfer(self) -> bool:
        return len(self.lines) > 1


@dataclass(frozen=True)
class Line:
    id: str
    name: str
    color: str
    edges: tuple[tuple[str, str], ...]  # 인접한 두 역 이름


def load_network() -> tuple[list[Line], list[Station]]:
    data = json.loads(NETWORK_FILE.read_text(encoding="utf-8"))
    lines = [
        Line(ln["id"], ln["name"], ln["color"], tuple(tuple(e) for e in ln["edges"])) for ln in data["lines"]
    ]
    stations = [Station(s["name"], s["lat"], s["lon"], tuple(s["lines"])) for s in data["stations"]]
    return lines, stations


# ---------- OpenStreetMap에서 생성 ----------


def normalize_name(name: str) -> str:
    """'강변(동서울터미널)', '방배 (백석예술대)', '동암역;동암' → '강변', '방배', '동암'."""
    name = min(name.split(";"), key=len)
    name = re.sub(r"\s*\(.*?\)", "", name).strip()
    return STATION_ALIASES.get(name, name)


def fetch_line(line_id: str, retries: int = 3) -> dict:
    query = f"""[out:json][timeout:120];
relation["route"="subway"]["ref"="{line_id}"]["network"="수도권 전철"]({OVERPASS_BBOX})->.r;
.r out body;
node(r.r)->.n;
.n out;"""
    headers = {"User-Agent": "metro-insight/0.1 (subway map build)"}
    for attempt in range(retries):
        for url in OVERPASS_URLS:
            try:
                resp = requests.post(url, data={"data": query}, headers=headers, timeout=150)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError):
                continue
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{line_id}호선 데이터를 Overpass API에서 받지 못했습니다.")


def parse_line(osm: dict) -> tuple[set[tuple[str, str]], dict[str, list[tuple[float, float]]]]:
    """일반 운행 계통의 정차역 순서 → (인접 역 쌍, 역 이름별 정차 위치 목록)."""
    nodes = {e["id"]: e for e in osm["elements"] if e["type"] == "node"}
    edges: set[tuple[str, str]] = set()
    coords: dict[str, list[tuple[float, float]]] = {}
    for rel in osm["elements"]:
        if rel["type"] != "relation" or EXPRESS_PATTERN.search(rel["tags"].get("name", "")):
            continue
        stops = []
        for m in rel["members"]:
            node = nodes.get(m["ref"]) if m["type"] == "node" and m["role"] in STOP_ROLES else None
            if node is None or "name" not in node.get("tags", {}):
                continue
            name = normalize_name(node["tags"]["name"])
            coords.setdefault(name, []).append((node["lat"], node["lon"]))
            if not stops or stops[-1] != name:
                stops.append(name)
        edges.update(tuple(sorted(pair)) for pair in zip(stops, stops[1:], strict=False))
    return edges, coords


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    dlat = (a[0] - b[0]) * 111.0
    dlon = (a[1] - b[1]) * 111.0 * math.cos(math.radians(a[0]))
    return math.hypot(dlat, dlon)


def find_shortcuts(edges: set[tuple[str, str]], pos: dict[str, tuple[float, float]]) -> set[tuple[str, str]]:
    """OSM 운행 계통에 정차역이 빠져 생긴, 중간 역을 건너뛴 구간을 찾는다 (예: 덕정–동두천중앙)."""
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def detour_km(a: str, b: str) -> float:
        best = math.inf
        stack = [(a, (a,), 0.0)]
        while stack:
            node, path, dist = stack.pop()
            for nxt in adj[node]:
                if {node, nxt} == {a, b} or nxt in path:
                    continue
                d = dist + distance_km(pos[node], pos[nxt])
                if nxt == b:
                    best = min(best, d)
                elif len(path) < SHORTCUT_MAX_HOPS:
                    stack.append((nxt, (*path, nxt), d))
        return best

    return {
        (a, b) for a, b in edges if detour_km(a, b) <= distance_km(pos[a], pos[b]) * SHORTCUT_DISTANCE_RATIO
    }


def build_network() -> dict:
    line_edges: dict[str, set[tuple[str, str]]] = {}
    coords: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for line_id, (name, _) in LINES.items():
        print(f"{name} 받는 중...")
        line_edges[line_id], line_coords = parse_line(fetch_line(line_id))
        for station, points in line_coords.items():
            coords[station].extend(points)

    pos = {}
    for station, points in coords.items():
        pos[station] = (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
        far = max(distance_km(pos[station], p) for p in points)
        if far > 1.0:
            print(f"  경고: {station} 정차 위치가 평균에서 {far:.1f}km 떨어져 있습니다.")

    lines = []
    station_lines: dict[str, set[str]] = defaultdict(set)
    for line_id, (name, color) in LINES.items():
        edges = line_edges[line_id]
        for a, b in sorted(shortcuts := find_shortcuts(edges, pos)):
            print(f"  {name}: 건너뛴 구간 제외 {a}–{b}")
        edges -= shortcuts
        for a, b in edges:
            station_lines[a].add(line_id)
            station_lines[b].add(line_id)
        lines.append({"id": line_id, "name": name, "color": color, "edges": sorted(map(list, edges))})

    stations = [
        {"name": n, "lat": round(pos[n][0], 6), "lon": round(pos[n][1], 6), "lines": sorted(station_lines[n])}
        for n in sorted(station_lines)
    ]
    return {
        "source": "© OpenStreetMap contributors (ODbL)",
        "generated": date.today().isoformat(),
        "lines": lines,
        "stations": stations,
    }


def main() -> None:
    network = build_network()
    path = Path(str(NETWORK_FILE))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(network, ensure_ascii=False, indent=1), encoding="utf-8")
    edge_count = sum(len(ln["edges"]) for ln in network["lines"])
    print(f"저장: {path} (역 {len(network['stations'])}개, 구간 {edge_count}개)")


if __name__ == "__main__":
    main()
