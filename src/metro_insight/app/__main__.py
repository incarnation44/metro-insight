"""실행: metro-insight  또는  python -m metro_insight.app"""
import os
import shutil
import sys
import threading

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from metro_insight.app.styles import apply_style

WINDOW_SCREEN_RATIO = 1
SPLASH_WIDTH, SPLASH_HEIGHT = 360, 148


class SplashScreen(QWidget):
    """첫 창을 만드는 동안 보여 주는 작은 안내 창.

    노선도 화면을 만드는 모듈을 읽는 데 1초 넘게 걸려서, 그동안 아무것도 보이지 않으면
    실행이 안 된 것처럼 보인다. 이 창은 가벼운 모듈만 쓰므로 먼저 떠서 그 시간을 덮는다.
    """

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("splash")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(SPLASH_WIDTH, SPLASH_HEIGHT)

        card = QFrame()
        card.setObjectName("splashCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(2)

        title = QLabel("Metro Insight")
        title.setObjectName("splashTitle")
        subtitle = QLabel("지하철 이용량 예측")
        subtitle.setObjectName("splashSubtitle")
        bar = QProgressBar()
        bar.setObjectName("splashBar")
        bar.setRange(0, 0)  # 남은 시간을 알 수 없으므로 흐르는 막대로만 보여 준다
        bar.setTextVisible(False)
        bar.setFixedHeight(4)
        note = QLabel("노선도를 준비하는 중…")
        note.setObjectName("splashNote")

        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addStretch()
        card_layout.addWidget(bar)
        card_layout.addSpacing(8)
        card_layout.addWidget(note)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)
        self._center()

    def _center(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.move(available.center() - self.rect().center())

    def showEvent(self, event):
        super().showEvent(event)
        self._center()  # 화면 구성이 늦게 정해져도 가운데에 둔다

    def finish(self, window: QWidget) -> None:
        """주 창이 화면에 나온 것을 확인한 뒤 안내 창을 닫는다.

        안내 창을 먼저 닫으면 주 창이 그려지기 전에 화면이 비는 순간이 생긴다. 그래서 주 창을
        맨 앞으로 올리고 한 번 그리게 한 뒤에 닫는다 (주 창을 넘겨받는 이유가 이것이다).
        """
        if window.isVisible():
            window.raise_()
            window.activateWindow()
            window.repaint()  # 닫기 전에 주 창을 한 번 그린다
        self.close()


def preload_layout_modules() -> threading.Thread:
    """노선도 배치에 쓰는 numpy·scipy 를 안내 창이 떠 있는 동안 미리 읽기 시작한다.

    두 모듈은 읽는 데 반 초 넘게 걸린다. 노선도를 만드는 순간(주 스레드)에 읽으면 그만큼
    첫 화면이 늦어지므로, 안내 창을 띄운 뒤 다른 스레드에서 먼저 읽어 둔다. 읽지 못해도
    노선도를 만들 때 다시 시도하므로 창은 뜬다.
    """

    def run() -> None:
        try:
            import numpy  # noqa: F401, PLC0415 - 미리 읽어 두기 위한 import
            import scipy.sparse.csgraph  # noqa: F401, PLC0415
        except Exception:  # 미리 읽기만 하는 일이라 실패해도 그대로 둔다
            pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


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

    splash = SplashScreen()
    splash.show()
    app.processEvents()  # 안내 창을 먼저 그린다
    preload_layout_modules()  # 노선도용 모듈을 배경에서 먼저 읽는다

    # 노선도·예측기 모듈은 여기서 읽는다. 안내 창이 뜬 뒤라 읽는 동안에도 화면이 비지 않는다.
    from metro_insight.app.main_window import MainWindow

    window = MainWindow()
    available = app.primaryScreen().availableGeometry()
    window.resize(available.size() * WINDOW_SCREEN_RATIO)
    window.move(available.center() - window.rect().center())
    window.show()
    splash.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
