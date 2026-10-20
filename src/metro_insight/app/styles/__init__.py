"""앱 스타일(QSS) 로드. 스타일 파일은 이 폴더의 <이름>.qss 로 관리한다."""
from importlib.resources import files

from PySide6.QtCore import QDir
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

STYLES_DIR = files(__name__)
DEFAULT_STYLE = "modern_gray"

# 설치된 첫 번째 폰트가 쓰인다 (Windows: 맑은 고딕, WSL/Linux: Noto Sans CJK)
FONT_FAMILIES = ["Pretendard", "Malgun Gothic", "Noto Sans CJK KR", "Apple SD Gothic Neo"]
FONT_POINT_SIZE = 10

# 라이트 테마 팔레트 (QSS 의 색 토큰과 같은 값)
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


def apply_style(app: QApplication, name: str = DEFAULT_STYLE) -> None:
    """Fusion 기반 위에 폰트와 QSS를 적용한다.

    QSS에서는 url(styles:파일명)으로 이 폴더의 이미지를 참조한다.
    """
    QDir.addSearchPath("styles", str(STYLES_DIR))
    app.setStyle("Fusion")

    # Windows 다크 모드에서는 Fusion 이 시스템 팔레트를 따라가서, QSS 로 지정하지 않은 부분
    # (지도 캔버스·달력 팝업·툴팁·메뉴)만 검게 보인다. 이 앱은 라이트 테마라 팔레트를 고정한다.
    palette = QPalette()
    for role, color in LIGHT_PALETTE.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#b8bcc2"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#b8bcc2"))
    app.setPalette(palette)

    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPointSize(FONT_POINT_SIZE)
    app.setFont(font)

    app.setStyleSheet((STYLES_DIR / f"{name}.qss").read_text(encoding="utf-8"))
