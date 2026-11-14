"""그림 색: QSS 를 거치지 않고 직접 그리는 곳이 쓰는 색.

도트플롯·히트맵·막대는 QWidget 위에 직접 그리므로 QSS 규칙이 닿지 않는다. 그래도 색의
출처는 하나여야 하므로 여기서 tokens.json 의 값을 읽어 이름만 붙인다. 색을 바꿀 때는
tokens.json 만 고치면 된다.
"""
from __future__ import annotations

from PySide6.QtGui import QColor

from metro_insight.app.styles.tokens import load_tokens

MODE = "light"  # 화면은 라이트 모드 하나만 쓴다

_COLOR = load_tokens()["modes"][MODE]["color"]


def _hex(name: str) -> str:
    """토큰 이름 → #rrggbb (없는 이름이면 import 시점에 바로 KeyError)."""
    return str(_COLOR[name]).lower()


SURFACE = _hex("surface")  # 카드 바탕 (점 테두리·기준 표찰 글자)
TEXT = _hex("text-1")  # 점 옆 숫자·축 이름
MUTED = _hex("text-3")  # 축 눈금·보조 글자
GRID = _hex("chart-grid")  # 축 기준선
WHISKER = _hex("chart-whisker")  # ±MAE 막대
DOT = _hex("chart-dot")  # 예측 점
BAR = _hex("chart-bar")  # 규모별 오차 진한 막대
BAR_MUTED = _hex("chart-bar-muted")  # 규모별 오차 옅은 막대
MISSING = _hex("chart-missing")  # 값이 없는 칸
SWATCH_EDGE = _hex("border")  # 히트맵 범례 칸 테두리


def band(alpha: int = 14) -> QColor:
    """마우스를 올린 줄에 까는 옅은 띠."""
    color = QColor(DOT)
    color.setAlpha(alpha)
    return color