"""실행: metro-insight  또는  python -m metro_insight.app"""
import os
import shutil
import sys

from PySide6.QtWidgets import QApplication

from metro_insight.app.main_window import MainWindow
from metro_insight.app.styles import apply_style

WINDOW_SCREEN_RATIO = 1


def use_ibus_on_linux() -> None:
    """Linux(WSL 포함)에서 ibus 가 설치되어 있으면 한글 입력에 쓴다.

    WSLg 의 Wayland 에서는 Qt 가 입력기를 스스로 찾지 못해 QT_IM_MODULE 로 지정해야 한다.
    이미 다른 입력기(fcitx 등)를 지정했다면 그대로 둔다.
    """
    if sys.platform.startswith("linux") and shutil.which("ibus-daemon"):
        os.environ.setdefault("QT_IM_MODULE", "ibus")


def main() -> None:
    use_ibus_on_linux()  # QApplication 을 만들기 전에 정해야 한다
    app = QApplication(sys.argv)
    apply_style(app)
    window = MainWindow()
    available = app.primaryScreen().availableGeometry()
    window.resize(available.size() * WINDOW_SCREEN_RATIO)
    window.move(available.center() - window.rect().center())
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
