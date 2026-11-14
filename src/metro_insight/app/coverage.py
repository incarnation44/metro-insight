"""노선도의 (호선, 역) 쌍 중 예측 데이터가 없는 곳을 찾는다.

노선도는 1~8호선 420개(호선, 역) 쌍을 그리지만, 예측기는 학습 데이터에 있는 쌍만 계산할 수
있다. 그 차이를 미리 알아 두면 지도에서 데이터 없는 역을 다른 모양으로 그리고, 그 역을
골랐을 때 "예측 데이터 없음"이라고 바로 알릴 수 있다.

예측기 역 목록은 `models/boarding_history.npz` 의 (호선, 역) 목록을 쓴다. 예측기를 만드는
것(모델 파일·승차 이력 전체 읽기)은 느리므로, 여기서는 그 파일 안의 이름 목록만 읽는다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from metro_insight.app.subway_data import Line, Station, load_network

ROOT = Path(__file__).resolve().parents[3]
HISTORY_NAME = "boarding_history.npz"

# 데이터 쪽 역 이름이 지도 이름과 다른 경우 (예측기와 같은 값)
MAP_NAME_BY_DATA_STATION = {("7호선", "이수"): "총신대입구"}

# 이 비율 이상의 역에 데이터가 없으면 그 호선을 '데이터가 드문 호선'으로 안내한다
ALERT_GAP_RATIO = 0.5

_PARENS = re.compile(r"\(.*?\)")


def normalize(name: str) -> str:
    """'잠실(송파구청)' → '잠실'. 지도와 데이터의 역 이름을 맞출 때 쓴다."""
    return _PARENS.sub("", str(name)).strip()


def canonical(line: str, station: str) -> tuple[str, str]:
    """이름 표기가 달라도 같은 역으로 묶는 열쇠: 부역명을 떼고, 원본 표기를 지도 이름으로 바꾼다.

    예측기도 같은 규칙으로 역을 찾는다('잠실(송파구청)' 과 '잠실', 7호선 원본 '이수' 와
    지도 이름 '총신대입구'). 그래서 여기서도 예측 데이터가 있는지 같은 기준으로 판단한다.
    """
    short = normalize(station)
    return (line, MAP_NAME_BY_DATA_STATION.get((line, short), short))


@dataclass(frozen=True)
class Coverage:
    """지도 전체 쌍과 예측 데이터가 있는 쌍의 차이."""

    pairs: frozenset[tuple[str, str]]  # (호선 이름, 역 이름) — 지도가 그리는 전체
    known: frozenset[tuple[str, str]]  # 그중 예측 데이터가 있는 쌍
    line_names: tuple[str, ...]  # 지도 호선 이름 (1호선 …)
    source: str  # 목록을 읽은 파일 (안내·검사용)

    @property
    def missing(self) -> frozenset[tuple[str, str]]:
        return self.pairs - self.known

    def is_missing(self, line: str, station: str) -> bool:
        """그 (호선, 역)에 예측 데이터가 없는가. 지도에 없는 쌍은 False."""
        return (line, station) in self.missing

    def missing_lines_of(self, station: str) -> tuple[str, ...]:
        """그 역에서 예측 데이터가 없는 호선 이름들 (지도에 있는 호선만)."""
        return tuple(line for line in self.line_names if (line, station) in self.missing)

    def has_data(self, station: str) -> bool:
        """그 역의 어느 호선이든 예측 데이터가 있는가."""
        return any((line, station) in self.known for line in self.line_names)

    def line_total(self, line: str) -> int:
        return sum(1 for name, _ in self.pairs if name == line)

    def line_known(self, line: str) -> int:
        return sum(1 for name, _ in self.known if name == line)

    def line_missing(self, line: str) -> int:
        return self.line_total(line) - self.line_known(line)

    def line_gap_ratio(self, line: str) -> float:
        total = self.line_total(line)
        return 0.0 if total == 0 else self.line_missing(line) / total

    def alert_lines(self) -> tuple[str, ...]:
        """역 대부분에 데이터가 없는 호선 (안내 띠를 띄우는 기준)."""
        return tuple(line for line in self.line_names if self.line_gap_ratio(line) >= ALERT_GAP_RATIO)


def _history_pairs(path: Path) -> frozenset[tuple[str, str]]:
    """예측기 역 목록(호선, 역). 파일이 없거나 읽지 못하면 빈 목록."""
    if not path.exists():
        return frozenset()
    try:
        import numpy as np  # noqa: PLC0415 - 이 목록을 실제로 읽을 때만 numpy 를 쓴다

        with np.load(path, allow_pickle=False) as data:
            lines = [str(value) for value in data["pair_lines"]]
            stations = [str(value) for value in data["pair_stations"]]
    except (OSError, ValueError, KeyError):
        return frozenset()
    out: set[tuple[str, str]] = set()
    for line, station in zip(lines, stations, strict=True):
        out.add(canonical(line, station))
    return frozenset(out)


_CACHE: dict[Path, Coverage] = {}


def load_coverage(
    models_dir: str | Path | None = None,
    lines: list[Line] | None = None,
    stations: list[Station] | None = None,
) -> Coverage:
    """노선도와 예측기 역 목록을 비교해 데이터 공백을 계산한다 (한 번 계산한 결과는 재사용)."""
    if lines is None or stations is None:
        network = load_network()
        lines = lines or network[0]
        stations = network[1] if stations is None else stations
    directory = Path(models_dir) if models_dir is not None else ROOT / "models"
    key = directory.resolve()
    if key in _CACHE:
        return _CACHE[key]

    names = {line.id: line.name for line in lines}
    line_names = tuple(line.name for line in lines)
    pairs = frozenset(
        (names[line_id], station.name) for station in stations for line_id in station.lines
    )
    path = directory / HISTORY_NAME
    # 예측 데이터 쪽 이름도 같은 열쇠로 바꿔 두어야 '이수'·'총신대입구(이수)' 처럼 표기가 달라도 붙는다
    known_keys = _history_pairs(path)
    known = frozenset(pair for pair in pairs if canonical(*pair) in known_keys)
    coverage = Coverage(pairs, known, line_names, str(path))
    _CACHE[key] = coverage
    return coverage


def format_count(known: int, total: int) -> str:
    """'10/102' — 범례·안내 띠에 쓰는 짧은 표기."""
    return f"{known}/{total}"