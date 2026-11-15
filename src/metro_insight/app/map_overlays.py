"""노선도 위에 겹쳐 띄우는 카드 위젯: 호선 범례, 선택한 역, 확대·축소 버튼, 출처 표기.

호선 배지는 노선색 위에 글자를 얹으므로, 대비가 4.5:1 을 넘도록 글자색이나 배경을 계산해
고른다(palette.badge_palette). 지도 점·노선 선은 공식 노선색을 그대로 쓴다.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen
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

from metro_insight.app import chart_colors
from metro_insight.app.coverage import ALERT_GAP_RATIO, Coverage, format_count
from metro_insight.app.palette import badge_palette, gap_palette
from metro_insight.app.subway_data import Line, Station

LINE_BADGE_SIZE = 22
GAP_SAMPLE_SIZE = 16


def add_shadow(widget: QWidget) -> None:
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(24)
    shadow.setOffset(0, 4)
    shadow.setColor(QColor(17, 24, 39, 36))
    widget.setGraphicsEffect(shadow)


class LineBadge(QLabel):
    """호선 색 원 안에 호선 번호를 쓴 배지. 글자색은 배경과의 대비로 정한다."""

    def __init__(self, line: Line, size: int = LINE_BADGE_SIZE):
        super().__init__(line.id)
        self.setObjectName("lineBadge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(size, size)
        self._apply_color(line.color)

    def _apply_color(self, color: str) -> None:
        text, background = badge_palette(color)
        self.setStyleSheet(
            f"background: {background}; color: {text}; border-radius: {self.height() // 2}px;"
        )


class GapDot(QWidget):
    """예측 데이터가 없는 역이 지도에서 어떻게 보이는지 알려 주는 점선 원 표본."""

    def __init__(self, size: int = GAP_SAMPLE_SIZE):
        super().__init__()
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # 지도 위 표본과 같은 색을 쓴다 (화면 모드가 바뀌면 같이 바뀐다)
        gap = gap_palette(chart_colors.get_mode())
        pen = QPen(QColor(gap["GAP_BORDER"]), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QColor(gap["GAP_FILL"]))
        inset = 2.0
        painter.drawEllipse(QRectF(inset, inset, self.width() - 2 * inset, self.height() - 2 * inset))
        painter.end()


class MapCard(QFrame):
    """그림자가 있는 흰 카드. 맨 위에 작은 제목을 둔다."""

    def __init__(self, caption: str):
        super().__init__()
        self.setObjectName("mapCard")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(16, 12, 16, 12)
        self.body.setSpacing(8)
        caption_label = QLabel(caption)
        caption_label.setObjectName("cardCaption")
        self.body.addWidget(caption_label)
        add_shadow(self)


class LineLegend(MapCard):
    """호선 8개와, 호선마다 예측 데이터가 있는 역 수."""

    def __init__(self, lines: list[Line], coverage: Coverage | None = None):
        super().__init__("노선")
        self._coverage = coverage
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        for i, line in enumerate(lines):
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(LineBadge(line))
            name = QLabel(line.name)
            name.setObjectName("legendText")
            row.addWidget(name)
            if coverage is not None:
                count = QLabel(
                    format_count(coverage.line_known(line.name), coverage.line_total(line.name))
                )
                count.setObjectName("legendCount")
                count.setProperty("thin", coverage.line_gap_ratio(line.name) >= ALERT_GAP_RATIO)
                count.setToolTip(
                    f"{line.name} 예측 데이터 {coverage.line_known(line.name)}개 역 / "
                    f"전체 {coverage.line_total(line.name)}개 역"
                )
                row.addWidget(count)
            row.addStretch()
            grid.addLayout(row, i // 2, i % 2)
        self.body.addLayout(grid)

        if coverage is not None and coverage.missing:
            self.body.addWidget(_divider())
            note = QHBoxLayout()
            note.setSpacing(8)
            note.addWidget(GapDot())
            gap = QLabel(f"예측 데이터 없음 {len(coverage.missing)}곳")
            gap.setObjectName("legendCount")
            gap.setToolTip("점선·회색으로 그린 역은 학습 데이터가 없어 예측할 수 없습니다")
            note.addWidget(gap)
            note.addStretch()
            self.body.addLayout(note)


class SelectionCard(MapCard):
    """선택한 역 이름과 지나는 호선, 그리고 예측 데이터 유무를 보여주는 카드."""

    def __init__(self, lines: list[Line], coverage: Coverage | None = None):
        super().__init__("선택한 역")
        self._lines = {line.id: line for line in lines}
        self._coverage = coverage
        self.setMinimumWidth(240)

        self.name = QLabel()
        self.name.setObjectName("stationDisplay")
        self.hint = QLabel("노선도에서 역을 클릭하세요")
        self.hint.setObjectName("cardHint")
        self.gap_tag = QLabel()
        self.gap_tag.setObjectName("gapTag")
        self.gap_tag.setVisible(False)
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
        badge_row.addWidget(self.gap_tag)
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
        self._show_gap(station)
        self.name.show()
        self.badge_row.show()
        self.hint.hide()
        self.adjustSize()

    def _show_gap(self, station: Station) -> None:
        """예측 데이터가 없는 호선을 표찰로 알린다 (없으면 표찰을 숨긴다)."""
        missing = self._coverage.missing_lines_of(station.name) if self._coverage else ()
        if not missing:
            self.gap_tag.setVisible(False)
            return
        self.gap_tag.setText(f"{'·'.join(m.split('호선')[0] for m in missing)}호선 예측 데이터 없음")
        self.gap_tag.setToolTip("이 호선은 학습 데이터에 없어 예측할 수 없습니다")
        self.gap_tag.setVisible(True)


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


class CoverageBanner(QFrame):
    """데이터가 드문 호선을 골랐을 때 노선도 위에 띄우는 안내 띠.

    숫자는 모두 coverage 에서 계산해 넣는다 (호선마다 다르므로 화면에 적어 두지 않는다).
    """

    def __init__(self):
        super().__init__()
        self.setObjectName("coverageBanner")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)
        self._title = QLabel()
        self._title.setObjectName("bannerTitle")
        self._text = QLabel()
        self._text.setObjectName("bannerText")
        self._text.setWordWrap(True)
        layout.addWidget(self._title)
        layout.addWidget(self._text, 1)
        self.setVisible(False)

    @property
    def title(self) -> str:
        return self._title.text()

    @property
    def message(self) -> str:
        return self._text.text()

    def show_lines(self, coverage: Coverage, line_names: tuple[str, ...] | None = None) -> None:
        """그 호선들의 데이터 공백을 안내한다. 알릴 만큼 데이터가 없는 호선이 없으면 띠를 숨긴다."""
        chosen = line_names if line_names is not None else coverage.alert_lines()
        alerts = tuple(line for line in chosen if coverage.line_gap_ratio(line) >= ALERT_GAP_RATIO)
        if not alerts:
            self.setVisible(False)
            return
        parts = [
            f"{line}은 {coverage.line_total(line)}개 역 중 {coverage.line_known(line)}개 역만 "
            f"예측 데이터가 있습니다"
            for line in alerts
        ]
        self._title.setText("데이터 안내")
        self._text.setText(
            " · ".join(parts) + ". 점선이나 회색으로 그린 역은 예측할 수 없습니다."
        )
        self.setVisible(True)


def _divider() -> QFrame:
    line = QFrame()
    line.setObjectName("resultDivider")
    line.setFixedHeight(1)
    return line