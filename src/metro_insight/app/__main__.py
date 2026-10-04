"""실행: metro-insight  또는  python -m metro_insight.app"""
import sys

from PySide6.QtWidgets import QApplication

from metro_insight.app.main_window import MainWindow
from metro_insight.app.styles import apply_style

WINDOW_SCREEN_RATIO = 1


def main() -> None:
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
