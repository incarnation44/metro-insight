"""그림 색: QSS 를 거치지 않고 직접 그리는 곳이 쓰는 색.

도트플롯·히트맵·막대·노선도는 QWidget 위에 직접 그리므로 QSS 규칙이 닿지 않는다. 그래도 색의
출처는 하나여야 하므로 여기서 tokens.json 의 값을 읽어 이름만 붙인다. 색을 바꿀 때는
tokens.json 만 고치면 된다.

모드는 두 가지다. 모듈 수준 상수(SURFACE, HEAT …)는 라이트 값이고, 화면 모드를 바꿀 때는
`set_mode("dark")` 로 상수를 그 모드 값으로 갈아 끼우거나 `colors_for("dark")` 로 묶음을 받아
그 자리에서 칠한다.

이 모듈은 Qt 를 import 하지 않는다(색 이름만 담는다). 그래야 Qt 없이 도는 계산 모듈
(analysis.py)도 같은 값을 쓸 수 있다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from metro_insight.app.styles.tokens import load_tokens

if TYPE_CHECKING:
    from PySide6.QtGui import QColor

DEFAULT_MODE = "light"
MODES = ("light", "dark")

# 상수 이름 → 토큰 이름. 색의 출처는 tokens.json 하나다.
_TOKEN_KEYS = {
    "SURFACE": "surface",  # 카드 바탕 (점 테두리·기준 표찰 글자)
    "TEXT": "text-1",  # 점 옆 숫자·축 이름
    "MUTED": "text-3",  # 축 눈금·보조 글자
    "GRID": "chart-grid",  # 축 기준선
    "WHISKER": "chart-whisker",  # ±MAE 막대
    "DOT": "chart-dot",  # 예측 점
    "BAR": "chart-bar",  # 규모별 오차 진한 막대
    "BAR_MUTED": "chart-bar-muted",  # 규모별 오차 옅은 막대
    "MISSING": "chart-missing",  # 값이 없는 칸
    "SWATCH_EDGE": "border",  # 히트맵 범례 칸 테두리
    "STATION_FILL": "map-station-fill",  # 역 점 바탕
    "TRANSFER_BORDER": "map-transfer-border",  # 환승역 테두리
    "SELECTED_FILL": "map-selected-fill",  # 고른 역 바탕
    "LABEL_COLOR": "map-label",  # 역 이름
    "SELECTED_LABEL_COLOR": "map-selected-label",  # 고른 역 이름
    "LABEL_HALO": "map-label-halo",  # 이름 뒤 테두리 (선 위에서도 읽히게)
}
# 히트맵: 한 가지 색의 6단계 (옅음 → 진함)
_HEAT_KEYS = tuple(f"heat-{step}" for step in range(1, 7))

_COLOR_TABLES = load_tokens()["modes"]


def _build(mode: str) -> dict[str, object]:
    color = _COLOR_TABLES[mode]["color"]
    colors: dict[str, object] = {
        name: str(color[token]).lower() for name, token in _TOKEN_KEYS.items()
    }
    colors["HEAT"] = tuple(str(color[token]).lower() for token in _HEAT_KEYS)
    return colors


_TABLES = {mode: _build(mode) for mode in MODES}
_CONSTANT_NAMES = (*_TOKEN_KEYS, "HEAT")

_mode = DEFAULT_MODE

# 라이트 값 (모듈 상수). 다른 모듈이 import 시점에 이 값을 읽어 가도 라이트 화면은 그대로다.
_LIGHT = _TABLES[DEFAULT_MODE]
SURFACE = _LIGHT["SURFACE"]
TEXT = _LIGHT["TEXT"]
MUTED = _LIGHT["MUTED"]
GRID = _LIGHT["GRID"]
WHISKER = _LIGHT["WHISKER"]
DOT = _LIGHT["DOT"]
BAR = _LIGHT["BAR"]
BAR_MUTED = _LIGHT["BAR_MUTED"]
MISSING = _LIGHT["MISSING"]
SWATCH_EDGE = _LIGHT["SWATCH_EDGE"]
HEAT = _LIGHT["HEAT"]
STATION_FILL = _LIGHT["STATION_FILL"]
TRANSFER_BORDER = _LIGHT["TRANSFER_BORDER"]
SELECTED_FILL = _LIGHT["SELECTED_FILL"]
LABEL_COLOR = _LIGHT["LABEL_COLOR"]
SELECTED_LABEL_COLOR = _LIGHT["SELECTED_LABEL_COLOR"]
LABEL_HALO = _LIGHT["LABEL_HALO"]


def colors_for(mode: str | None = None) -> dict[str, object]:
    """그 모드의 그림 색 묶음(사본). 모드를 생략하면 지금 모드."""
    name = _mode if mode is None else str(mode).strip().lower()
    if name not in _TABLES:
        raise KeyError(f"알 수 없는 화면 모드: {mode!r} (가능: {list(MODES)})")
    return dict(_TABLES[name])


def set_mode(mode: str) -> dict[str, object]:
    """그 모드의 색을 모듈 상수에 반영한다 (직접 그리는 그림이 다음 그리기부터 쓴다)."""
    global _mode
    name = str(mode).strip().lower()
    if name not in _TABLES:
        raise KeyError(f"알 수 없는 화면 모드: {mode!r} (가능: {list(MODES)})")
    _mode = name
    table = _TABLES[name]
    globals().update({constant: table[constant] for constant in _CONSTANT_NAMES})
    return dict(table)


def get_mode() -> str:
    """지금 그림 색이 어느 모드인지."""
    return _mode


def band(alpha: int = 14) -> QColor:
    """마우스를 올린 줄에 까는 옅은 띠 (지금 모드의 점 색)."""
    from PySide6.QtGui import QColor  # noqa: PLC0415  Qt 는 실제로 칠할 때만 읽는다

    color = QColor(str(_TABLES[_mode]["DOT"]))
    color.setAlpha(alpha)
    return color