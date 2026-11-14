"""QSS 파일의 색 자리를 토큰 값으로 채운다.

`modern_gray.qss` 는 색을 직접 적지 않고 `{{이름}}` 자리만 남긴다. 이 모듈이 tokens.json 의
값을 읽어 그 자리를 채우므로, 색을 바꿀 때는 토큰 파일 한 곳만 고치면 된다.
"""
from __future__ import annotations

import re

from metro_insight.app.styles import tokens

STYLES_DIR = tokens.TOKENS_PATH.parent
DEFAULT_MODE = "light"

# {{text-2}}, {{accent}} 처럼 이름만 남긴 자리. QSS 의 규칙 괄호({ })와 겹치지 않게 두 겹으로 쓴다.
_PLACEHOLDER = re.compile(r"\{\{\s*([a-z0-9][a-z0-9.\-]*)\s*\}\}")


def render(template: str, mode: str = DEFAULT_MODE) -> str:
    """템플릿의 색 자리를 그 모드의 토큰 값(#rrggbb)으로 바꾼다.

    모르는 이름이 남아 있으면 조용히 넘기지 않고 KeyError 로 알린다.
    """
    color = tokens.resolve(mode)["color"]
    missing: list[str] = []

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        key = name[len("color.") :] if name.startswith("color.") else name
        if key not in color:
            missing.append(name)
            return match.group(0)
        return str(color[key]).lower()

    text = _PLACEHOLDER.sub(fill, template)
    if missing:
        raise KeyError(f"QSS 에서 쓰는 색 토큰이 없습니다: {sorted(set(missing))} ({mode})")
    return text


def template(name: str) -> str:
    """이 폴더의 QSS 템플릿을 읽는다."""
    return (STYLES_DIR / f"{name}.qss").read_text(encoding="utf-8")


def load(name: str, mode: str = DEFAULT_MODE) -> str:
    """템플릿을 읽어 색을 채운 QSS 문자열을 돌려준다."""
    return render(template(name), mode)


def unresolved(text: str) -> list[str]:
    """아직 남아 있는 색 자리 이름들 (검사용)."""
    return _PLACEHOLDER.findall(text)