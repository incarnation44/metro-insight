"""노선도 위에 겹쳐 띄우는 카드 위젯: 호선 범례, 선택한 역, 확대·축소 버튼, 출처 표기."""
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from metro_insight.app.subway_data import Line, Station

LINE_BADGE_SIZE = 20


def add_shadow(widget: QWidget) -> None:
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(24)
    shadow.setOffset(0, 4)
    shadow.setColor(QColor(17, 24, 39, 36))
    widget.setGraphicsEffect(shadow)


class LineBadge(QLabel):
    """호선 색 원 안에 호선 번호를 쓴 배지."""

    def __init__(self, line: Line, size: int = LINE_BADGE_SIZE):
        super().__init__(line.id)
        self.setObjectName("lineBadge")
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background: {line.color}; border-radius: {size // 2}px;")


class MapCard(QFrame):
    """그림자가 있는 흰 카드. 맨 위에 작은 제목을 둔다."""

    def __init__(self, caption: str):
        super().__init__()
        self.setObjectName("mapCard")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(14, 12, 14, 14)
        self.body.setSpacing(8)
        caption_label = QLabel(caption)
        caption_label.setObjectName("cardCaption")
        self.body.addWidget(caption_label)
        add_shadow(self)


class LineLegend(MapCard):
    def __init__(self, lines: list[Line]):
        super().__init__("노선")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        for i, line in enumerate(lines):
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(LineBadge(line))
            name = QLabel(line.name)
            name.setObjectName("legendText")
            row.addWidget(name)
            grid.addLayout(row, i // 2, i % 2)
        self.body.addLayout(grid)


class SelectionCard(MapCard):
    """선택한 역 이름과 지나는 호선을 보여주는 카드."""

    def __init__(self, lines: list[Line]):
        super().__init__("선택한 역")
        self._lines = {line.id: line for line in lines}
        self.setMinimumWidth(220)

        self.name = QLabel()
        self.name.setObjectName("stationName")
        self.hint = QLabel("노선도에서 역을 클릭하세요")
        self.hint.setObjectName("cardHint")
        self.badges = QHBoxLayout()
        self.badges.setSpacing(4)
        self.line_names = QLabel()
        self.line_names.setObjectName("cardHint")

        self.badge_row = QWidget()  # 역을 선택하기 전에는 자리를 차지하지 않게 숨긴다
        badge_row = QHBoxLayout(self.badge_row)
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.setSpacing(8)
        badge_row.addLayout(self.badges)
        badge_row.addWidget(self.line_names)
        badge_row.addStretch()

        self.body.addWidget(self.name)
        self.body.addWidget(self.hint)
        self.body.addWidget(self.badge_row)
        self.name.hide()
        self.badge_row.hide()

    def show_station(self, station: Station) -> None:
        while self.badges.count():
            self.badges.takeAt(0).widget().deleteLater()
        for line_id in station.lines:
            self.badges.addWidget(LineBadge(self._lines[line_id]))
        self.line_names.setText(" · ".join(self._lines[i].name for i in station.lines))
        self.name.setText(station.name)
        self.name.show()
        self.badge_row.show()
        self.hint.hide()
        self.adjustSize()


class ZoomControls(QFrame):
    """확대·축소·처음 화면 버튼."""

    zoom_in = Signal()
    zoom_out = Signal()
    reset = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("zoomControls")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        for icon, tip, signal in [
            ("styles:plus.svg", "확대", self.zoom_in),
            ("styles:minus.svg", "축소", self.zoom_out),
            ("styles:fit.svg", "처음 화면", self.reset),
        ]:
            button = QToolButton()
            button.setIcon(QIcon(icon))
            button.setIconSize(QSize(16, 16))
            button.setToolTip(tip)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(signal.emit)
            layout.addWidget(button)
        add_shadow(self)


class Attribution(QLabel):
    def __init__(self, text: str):
        super().__init__(text)
        self.setObjectName("mapAttribution")
