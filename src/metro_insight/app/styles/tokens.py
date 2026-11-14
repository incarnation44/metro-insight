"""디자인 토큰 로더·검증·대비 계산 (표준 라이브러리만 사용).

`tokens.json` 이 색·글꼴·간격·모서리의 단일 출처다. 이 모듈은 그 파일을 읽어
구조를 검증하고 WCAG 2.x 대비비를 계산한다. PySide6 에 의존하지 않으므로 Qt 없이도
검증할 수 있다. 실제로 화면에 적용하는 일(QPalette·QSS 생성)은 후속 작업 범위다.

대비 검사 대상은 "본문·보조·강조 텍스트"이고, 기준은 WCAG 2.x AA 일반 텍스트 4.5:1 이다.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

TOKENS_PATH = Path(__file__).with_name("tokens.json")
SCHEMA_ID = "metro-tokens/1"
MODES = ("light", "dark")
GROUPS = ("color", "font", "space", "radius")
REQUIRED_COLOR_KEYS = (
    "bg",
    "surface",
    "text-1",
    "text-2",
    "text-3",
    "border",
    "accent",
    "accent-text",
    "danger",
    "success",
    "warning",
)
REQUIRED_FONT_KEYS = ("family", "size-body", "size-h1", "size-display", "size-num-lg")
REQUIRED_RADIUS_KEYS = ("sm", "md", "lg", "pill")
AA_NORMAL_TEXT = 4.5

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def load_tokens(path: str | Path | None = None) -> dict[str, Any]:
    """토큰 파일을 읽고 구조를 검증해 돌려준다.

    형식이 어긋나면 ValueError 에 무엇이 잘못됐는지 담아 올린다.
    """
    target = Path(path) if path is not None else TOKENS_PATH
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"토큰 파일이 없습니다: {target}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"토큰 파일이 JSON 이 아닙니다: {target} ({exc})") from exc
    _validate(data, target)
    return data


def resolve(mode: str) -> dict[str, Any]:
    """한 모드의 토큰 묶음을 사본으로 돌려준다(수정해도 원본에 영향 없음)."""
    data = load_tokens()
    modes = data["modes"]
    if mode not in modes:
        raise KeyError(f"알 수 없는 모드: {mode!r} (가능: {sorted(modes)})")
    return copy.deepcopy(modes[mode])


def relative_luminance(color: str) -> float:
    """WCAG 2.x 상대휘도. 0(검정) ~ 1(흰색)."""
    value = _parse_hex(color)
    channels = []
    for index in (0, 2, 4):
        raw = int(value[index : index + 2], 16) / 255
        channels.append(raw / 12.92 if raw <= 0.03928 else ((raw + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(fg: str, bg: str) -> float:
    """두 색의 대비비. 1.0(같은 색) ~ 21.0(검정/흰색). 반올림하지 않는다."""
    high, low = sorted((relative_luminance(fg), relative_luminance(bg)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def check_contrast(mode: str) -> dict[str, float]:
    """본문·보조·강조 텍스트를 배경별로 재 본 대비비 묶음.

    키는 "<텍스트 토큰>/<배경 토큰>" 이고, 강조 버튼 글자는 "accent-text/accent" 다.
    값은 반올림하지 않은 실측치다.
    """
    color = resolve(mode)["color"]
    result: dict[str, float] = {}
    for name in ("text-1", "text-2", "text-3", "accent"):
        for background in ("bg", "surface"):
            result[f"{name}/{background}"] = contrast_ratio(color[name], color[background])
    result["accent-text/accent"] = contrast_ratio(color["accent-text"], color["accent"])
    return result


def qss_color(name: str, mode: str) -> str:
    """토큰 이름으로 QSS 에 넣을 색 문자열을 얻는다. 예: qss_color("text-2", "light").

    "color.text-2" 처럼 묶음 접두사를 붙여도 된다.
    """
    color = resolve(mode)["color"]
    key = name[len("color.") :] if name.startswith("color.") else name
    try:
        return str(color[key]).lower()
    except KeyError:
        raise KeyError(f"색 토큰이 없습니다: {name!r} ({mode})") from None


def _parse_hex(color: str) -> str:
    if not isinstance(color, str) or not _HEX_RE.match(color):
        raise ValueError(f"#rrggbb 형식이 아닙니다: {color!r}")
    return color[1:].lower()


def _validate(data: Any, path: Path) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 최상위가 객체가 아닙니다")
    if data.get("schema") != SCHEMA_ID:
        raise ValueError(f"{path}: schema 는 {SCHEMA_ID!r} 이어야 합니다 (현재 {data.get('schema')!r})")
    modes = data.get("modes")
    if not isinstance(modes, dict):
        raise ValueError(f"{path}: modes 객체가 없습니다")
    missing_modes = [mode for mode in MODES if mode not in modes]
    if missing_modes:
        raise ValueError(f"{path}: 모드가 없습니다 {missing_modes}")
    for mode in MODES:
        group_map = modes[mode]
        if not isinstance(group_map, dict):
            raise ValueError(f"{path}: modes.{mode} 가 객체가 아닙니다")
        for group in GROUPS:
            if group not in group_map:
                raise ValueError(f"{path}: modes.{mode}.{group} 묶음이 없습니다")
        _validate_color(group_map["color"], mode, path)
        _validate_font(group_map["font"], mode, path)
        _validate_space(group_map["space"], mode, path)
        _validate_radius(group_map["radius"], mode, path)

    light = modes["light"]
    dark = modes["dark"]
    for group in GROUPS:
        only_light = sorted(set(light[group]) - set(dark[group]))
        only_dark = sorted(set(dark[group]) - set(light[group]))
        if only_light or only_dark:
            raise ValueError(
                f"{path}: {group} 키가 모드마다 다릅니다 (light 전용 {only_light}, dark 전용 {only_dark})"
            )


def _validate_color(color: Any, mode: str, path: Path) -> None:
    if not isinstance(color, dict):
        raise ValueError(f"{path}: modes.{mode}.color 가 객체가 아닙니다")
    missing = [key for key in REQUIRED_COLOR_KEYS if key not in color]
    if missing:
        raise ValueError(f"{path}: modes.{mode}.color 필수 키 누락 {missing}")
    invalid = {
        key: value for key, value in color.items() if not (isinstance(value, str) and _HEX_RE.match(value))
    }
    if invalid:
        raise ValueError(f"{path}: modes.{mode}.color 값이 #rrggbb 가 아닙니다 {invalid}")


def _validate_font(font: Any, mode: str, path: Path) -> None:
    if not isinstance(font, dict):
        raise ValueError(f"{path}: modes.{mode}.font 가 객체가 아닙니다")
    missing = [key for key in REQUIRED_FONT_KEYS if key not in font]
    if missing:
        raise ValueError(f"{path}: modes.{mode}.font 필수 키 누락 {missing}")
    family = font["family"]
    if not isinstance(family, list) or not family or not all(isinstance(item, str) for item in family):
        raise ValueError(f"{path}: modes.{mode}.font.family 는 문자열 목록이어야 합니다")
    for key in REQUIRED_FONT_KEYS[1:]:
        if not _is_positive_int(font[key]):
            raise ValueError(f"{path}: modes.{mode}.font.{key} 는 1 이상의 정수여야 합니다")


def _validate_space(space: Any, mode: str, path: Path) -> None:
    if not isinstance(space, dict):
        raise ValueError(f"{path}: modes.{mode}.space 가 객체가 아닙니다")
    if not _is_positive_int(space.get("unit")):
        raise ValueError(f"{path}: modes.{mode}.space.unit 은 1 이상의 정수여야 합니다")
    scale = space.get("scale")
    if not isinstance(scale, list) or len(scale) < 2 or not all(_is_non_negative_int(item) for item in scale):
        raise ValueError(f"{path}: modes.{mode}.space.scale 은 0 이상 정수 목록이어야 합니다")
    if scale != sorted(set(scale)):
        raise ValueError(f"{path}: modes.{mode}.space.scale 은 중복 없이 오름차순이어야 합니다")


def _validate_radius(radius: Any, mode: str, path: Path) -> None:
    if not isinstance(radius, dict):
        raise ValueError(f"{path}: modes.{mode}.radius 가 객체가 아닙니다")
    missing = [key for key in REQUIRED_RADIUS_KEYS if key not in radius]
    if missing:
        raise ValueError(f"{path}: modes.{mode}.radius 필수 키 누락 {missing}")
    for key in REQUIRED_RADIUS_KEYS:
        if not _is_positive_int(radius[key]):
            raise ValueError(f"{path}: modes.{mode}.radius.{key} 는 1 이상의 정수여야 합니다")


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0