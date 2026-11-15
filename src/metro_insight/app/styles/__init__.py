"""앱 스타일(QSS) 로드와 화면 팔레트(라이트/다크).

색의 출처는 `tokens.json` 하나다. QSS 의 색 자리는 그 파일 값으로 채우고, QSS 가 닿지 않는
곳(창 바탕·툴팁·달력 팝업 같은 기본 위젯)은 같은 값으로 만든 QPalette 로 맞춘다. 그래서
한 번의 `apply_style(app, mode)` 로 화면 전체의 색이 함께 바뀐다.
"""
from importlib.resources import files

from PySide6 import QtSvg  # noqa: F401  # QSS 의 SVG 아이콘(화살표·체크 등) 렌더러 등록
from PySide6.QtCore import QDir
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from metro_insight.app.styles import qss as qss_builder
from metro_insight.app.styles import tokens

STYLES_DIR = files(__name__)
DEFAULT_STYLE = "modern_gray"
DEFAULT_MODE = "light"
MODES = tokens.MODES

# 설치된 첫 번째 폰트가 쓰인다 (Windows: 맑은 고딕, WSL/Linux: Noto Sans CJK)
FONT_FAMILIES = ["Pretendard", "Malgun Gothic", "Noto Sans CJK KR", "Apple SD Gothic Neo"]
FONT_POINT_SIZE = 10

# 라이트 테마 팔레트 (QSS 의 색 토큰과 같은 값). 검사·다른 모듈이 참조하므로 그대로 둔다.
LIGHT_PALETTE = {
    QPalette.ColorRole.Window: "#f4f5f7",
    QPalette.ColorRole.WindowText: "#1f2328",
    QPalette.ColorRole.Base: "#ffffff",
    QPalette.ColorRole.AlternateBase: "#f1f2f4",
    QPalette.ColorRole.Text: "#1f2328",
    QPalette.ColorRole.Button: "#f1f2f4",
    QPalette.ColorRole.ButtonText: "#1f2328",
    QPalette.ColorRole.BrightText: "#1f2328",
    QPalette.ColorRole.Highlight: "#374151",
    QPalette.ColorRole.HighlightedText: "#ffffff",
    QPalette.ColorRole.ToolTipBase: "#ffffff",
    QPalette.ColorRole.ToolTipText: "#1f2328",
    QPalette.ColorRole.PlaceholderText: "#9aa0a8",
    QPalette.ColorRole.Link: "#374151",
    QPalette.ColorRole.LinkVisited: "#374151",
}

# QPalette 역할 → 토큰 이름. 두 모드가 같은 키를 가지므로 이 표는 모드와 무관하다.
_ROLE_TOKENS = (
    (QPalette.ColorRole.Window, "bg"),
    (QPalette.ColorRole.WindowText, "text-1"),
    (QPalette.ColorRole.Base, "surface"),
    (QPalette.ColorRole.AlternateBase, "surface-alt"),
    (QPalette.ColorRole.Text, "text-1"),
    (QPalette.ColorRole.Button, "surface-alt"),
    (QPalette.ColorRole.ButtonText, "text-1"),
    (QPalette.ColorRole.BrightText, "text-1"),
    (QPalette.ColorRole.Highlight, "accent"),
    (QPalette.ColorRole.HighlightedText, "accent-text"),
    (QPalette.ColorRole.ToolTipBase, "surface"),
    (QPalette.ColorRole.ToolTipText, "text-1"),
    (QPalette.ColorRole.PlaceholderText, "text-3-legacy"),
    (QPalette.ColorRole.Link, "accent"),
    (QPalette.ColorRole.LinkVisited, "accent"),
)

# 글자를 쓸 수 없게 된 버튼·입력 칸의 글자색
_DISABLED_TEXT_TOKEN = "disabled"
_DISABLED_TEXT_ROLES = (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText)

_ACTIVE_MODE = DEFAULT_MODE


def normalize_mode(mode: str) -> str:
    """모드 이름을 확인해 돌려준다. 모르는 이름이면 그 자리에서 알린다."""
    name = str(mode).strip().lower()
    if name not in MODES:
        raise ValueError(f"알 수 없는 화면 모드: {mode!r} (가능: {list(MODES)})")
    return name


def build_palette(mode: str = DEFAULT_MODE) -> QPalette:
    """그 모드의 토큰으로 QPalette 를 만든다.

    QSS 로 지정하지 않은 부분(툴팁·달력 팝업·메뉴·기본 위젯)까지 같은 색계로 맞춘다.
    """
    color = tokens.resolve(normalize_mode(mode))["color"]
    palette = QPalette()
    for role, token in _ROLE_TOKENS:
        palette.setColor(role, QColor(color[token]))
    for role in _DISABLED_TEXT_ROLES:
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(color[_DISABLED_TEXT_TOKEN]))
    return palette


def apply_style(app: QApplication, name: str = DEFAULT_STYLE, mode: str = DEFAULT_MODE) -> None:
    """Fusion 기반 위에 폰트·팔레트·QSS 를 함께 적용한다.

    QSS의 색 자리는 tokens.json 값으로 채우고, QSS에서는 url(styles:파일명)으로 이 폴더의
    이미지를 참조한다.
    """
    global _ACTIVE_MODE
    mode = normalize_mode(mode)
    QDir.addSearchPath("styles", str(STYLES_DIR))
    app.setStyle("Fusion")

    # Fusion 은 시스템 팔레트를 따라가서, QSS 로 지정하지 않은 부분(지도 캔버스·달력 팝업·
    # 툴팁·메뉴)이 OS 설정에 따라 검게 또는 희게 보인다. 그래서 모드별 팔레트를 고정한다.
    app.setPalette(build_palette(mode))

    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPointSize(FONT_POINT_SIZE)
    app.setFont(font)

    app.setStyleSheet(qss_builder.load(name, mode))
    _ACTIVE_MODE = mode


def current_mode() -> str:
    """지금 화면에 적용된 모드("light" 또는 "dark")."""
    return _ACTIVE_MODE