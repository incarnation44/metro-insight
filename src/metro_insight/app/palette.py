"""호선 색과 회색 계열의 계산 규칙.

노선 선·지도 점 테두리 같은 '노선을 나타내는 색'은 공식 노선색을 그대로 쓴다. 반면 배지처럼
그 색 위에 글자를 얹는 곳은 읽히는지(대비 4.5:1)를 계산해서 글자색이나 배경을 조금 바꾼다.

예측 데이터가 없는 역(점선·회색)의 세 색은 tokens.json 의 map-gap-* 이다. 화면 모드가 바뀌면
`gap_palette("dark")` 처럼 그 모드 값을 받아 쓴다.
"""
from __future__ import annotations

from metro_insight.app.styles.tokens import AA_NORMAL_TEXT, contrast_ratio, resolve

WHITE = "#ffffff"
INK = "#111827"  # 흰색 글자가 안 읽힐 때 쓰는 먹색

# 배지 배경을 얼마나 어둡게 파생할지 (3단 규칙의 마지막 단계)
DERIVE_DARKEN = 0.9

# 예측 데이터가 없는 역 (점선 테두리·바탕·이름) → tokens.json 의 map-gap-*
_GAP_KEYS = {
    "GAP_BORDER": "map-gap-border",
    "GAP_FILL": "map-gap-fill",
    "GAP_LABEL": "map-gap-label",
}


def gap_palette(mode: str = "light") -> dict[str, str]:
    """그 모드에서 '예측 데이터 없음' 역을 그리는 색 묶음."""
    color = resolve(mode)["color"]
    return {name: str(color[token]).lower() for name, token in _GAP_KEYS.items()}


# 모듈 상수는 라이트 값이다 (기본 화면).
_LIGHT_GAP = gap_palette("light")
GAP_BORDER = _LIGHT_GAP["GAP_BORDER"]
GAP_FILL = _LIGHT_GAP["GAP_FILL"]
GAP_LABEL = _LIGHT_GAP["GAP_LABEL"]


def darken(color: str, factor: float = DERIVE_DARKEN) -> str:
    """색을 같은 비율로 어둡게 만든다. 노선색의 성격(색상)은 그대로 둔다."""
    value = color.lstrip("#")
    parts = [int(value[index : index + 2], 16) for index in (0, 2, 4)]
    return "#" + "".join(f"{max(0, min(255, round(part * factor))):02X}" for part in parts)


def badge_palette(line_color: str) -> tuple[str, str]:
    """호선 배지의 (글자색, 배경색).

    ① 배경 위 흰 글자가 4.5:1 을 넘으면 흰 글자
    ② 아니면 먹색 글자가 4.5:1 을 넘는지 보고 되면 먹 글자
    ③ 둘 다 안 되면 배경만 조금 어둡게 파생하고 흰 글자를 쓴다
    """
    if contrast_ratio(WHITE, line_color) >= AA_NORMAL_TEXT:
        return WHITE, line_color
    if contrast_ratio(INK, line_color) >= AA_NORMAL_TEXT:
        return INK, line_color
    return WHITE, darken(line_color)