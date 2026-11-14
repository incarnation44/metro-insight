"""호선 색과 회색 계열의 계산 규칙.

노선 선·지도 점 테두리 같은 '노선을 나타내는 색'은 공식 노선색을 그대로 쓴다. 반면 배지처럼
그 색 위에 글자를 얹는 곳은 읽히는지(대비 4.5:1)를 계산해서 글자색이나 배경을 조금 바꾼다.
"""
from __future__ import annotations

from metro_insight.app.styles.tokens import AA_NORMAL_TEXT, contrast_ratio

WHITE = "#ffffff"
INK = "#111827"  # 흰색 글자가 안 읽힐 때 쓰는 먹색

# 노선도에서 예측 데이터가 없는 역 (점선·회색)
GAP_BORDER = "#b8bcc2"
GAP_FILL = "#f4f5f7"
GAP_LABEL = "#6b7078"

# 배지 배경을 얼마나 어둡게 파생할지 (3단 규칙의 마지막 단계)
DERIVE_DARKEN = 0.9


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