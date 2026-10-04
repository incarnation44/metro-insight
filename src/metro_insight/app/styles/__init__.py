"""앱 스타일(QSS) 로드. 스타일 파일은 이 폴더의 <이름>.qss 로 관리한다."""
from importlib.resources import files

from PySide6.QtCore import QDir
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

STYLES_DIR = files(__name__)
DEFAULT_STYLE = "modern_gray"

# 설치된 첫 번째 폰트가 쓰인다 (Windows: 맑은 고딕, WSL/Linux: Noto Sans CJK)
FONT_FAMILIES = ["Pretendard", "Malgun Gothic", "Noto Sans CJK KR", "Apple SD Gothic Neo"]
FONT_POINT_SIZE = 10


def apply_style(app: QApplication, name: str = DEFAULT_STYLE) -> None:
    """Fusion 기반 위에 폰트와 QSS를 적용한다.

    QSS에서는 url(styles:파일명)으로 이 폴더의 이미지를 참조한다.
    """
    QDir.addSearchPath("styles", str(STYLES_DIR))
    app.setStyle("Fusion")

    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPointSize(FONT_POINT_SIZE)
    app.setFont(font)

    app.setStyleSheet((STYLES_DIR / f"{name}.qss").read_text(encoding="utf-8"))
