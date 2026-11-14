"""분석 화면: 모델별 오차, 지평 구간 히트맵, 규모별 오차, 182~366일 재확인.

숫자는 모두 models/ 의 평가 기록에서 읽는다(analysis.py). 그림은 Qt 로 직접 그려
matplotlib 같은 추가 의존성을 쓰지 않는다.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from metro_insight.app.analysis import (
    HEAT_STEPS,
    HORIZON_LABELS,
    AnalysisData,
    backtest_note,
    heat_color,
    heat_step,
    horizon_insight,
    load_analysis,
    tertile_note,
)

TEXT = "#1f2328"
MUTED = "#6b7078"
BAR = "#374151"
BAR_LIGHT = "#9aa4b2"
SWATCH_EDGE = "#e3e5e8"
CARD_MARGIN = 16


class ModelCard(QFrame):
    """모델 하나의 MAE 를 크게 보여 주는 카드. 누르면 그 모델 기준으로 노선도를 본다."""

    clicked = Signal(str)

    def __init__(self, rank: int, name: str, mae: float, rmse: float, description: str):
        super().__init__()
        self.setObjectName("modelCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            f"{name} 모델: {description}\n"
            f"Test 평균 오차(MAE) {mae:,.1f}명 · 제곱평균오차(RMSE) {rmse:,.1f}명\n"
            "누르면 노선도로 돌아가 이 모델 기준으로 봅니다"
        )
        self.model = name
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        head = QHBoxLayout()
        head.setSpacing(8)
        badge = QLabel(f"{rank}")
        badge.setObjectName("modelCardRank")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setMinimumWidth(20)
        title = QLabel(name)
        title.setObjectName("modelCardName")
        head.addWidget(badge)
        head.addWidget(title)
        head.addStretch()
        layout.addLayout(head)

        value = QLabel(f"{mae:,.1f}")
        value.setObjectName("modelCardValue")
        layout.addWidget(value)
        unit = QLabel("MAE · 명")
        unit.setObjectName("modelCardUnit")
        layout.addWidget(unit)

        self._rmse = QLabel(f"RMSE {rmse:,.1f}")
        self._rmse.setObjectName("modelCardUnit")
        layout.addWidget(self._rmse)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.model)
        super().mouseReleaseEvent(event)


class Heatmap(QWidget):
    """지평 구간 × 모델 MAE 히트맵. 한 가지 색의 6단계로 옅음·진함을 나타낸다."""

    LABEL_WIDTH = 34
    HEADER_HEIGHT = 26
    CELL_HEIGHT = 34
    LEGEND_HEIGHT = 30
    GAP = 2

    def __init__(self, data: AnalysisData):
        super().__init__()
        self._data = data
        self._pixmap: QPixmap | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(320)

    def rows(self) -> list[str]:
        return [stat.name for stat in self._data.models]

    def sizeHint(self) -> QSize:
        return QSize(420, self.HEADER_HEIGHT + self.CELL_HEIGHT * len(self.rows()) + self.LEGEND_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def cell_rect(self, row: int, column: int) -> QRectF:
        """(행, 열) 칸의 화면 사각형. 검사와 그리기가 같은 계산을 쓴다."""
        columns = len(self._data.horizons)
        width = max(self.width() - self.LABEL_WIDTH, 1.0) / max(columns, 1)
        return QRectF(
            self.LABEL_WIDTH + column * width,
            self.HEADER_HEIGHT + row * self.CELL_HEIGHT,
            width - self.GAP,
            self.CELL_HEIGHT - self.GAP,
        )

    def cell_color(self, row: int, column: int) -> str:
        model = self.rows()[row]
        horizon = self._data.horizons[column]
        value = self._data.value_at(model, horizon)
        if value is None:
            return "#ffffff"
        low, high = self._data.heat_range()
        return heat_color(value, low, high)

    def _render(self) -> QPixmap:
        ratio = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
        pixmap = QPixmap(self.size() * ratio)
        pixmap.setDevicePixelRatio(ratio)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(QRectF(0, 0, self.width(), self.height()), QColor("#ffffff"))
        self._paint(painter)
        painter.end()
        return pixmap

    def _paint(self, painter: QPainter) -> None:
        low, high = self._data.heat_range()
        small = QFont()
        small.setPointSizeF(9.0)
        small.setWeight(QFont.Weight.DemiBold)
        painter.setFont(small)

        for column, horizon in enumerate(self._data.horizons):
            cell = self.cell_rect(0, column)
            painter.setPen(QColor(MUTED))
            painter.drawText(
                QRectF(cell.left(), 2, cell.width(), self.HEADER_HEIGHT - 6),
                int(Qt.AlignmentFlag.AlignCenter),
                HORIZON_LABELS.get(horizon, horizon),
            )

        for row, model in enumerate(self.rows()):
            cell = self.cell_rect(row, 0)
            painter.setPen(QColor(TEXT))
            painter.drawText(
                QRectF(0, cell.top(), self.LABEL_WIDTH - 6, cell.height()),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                model,
            )
            for column, horizon in enumerate(self._data.horizons):
                box = self.cell_rect(row, column)
                value = self._data.value_at(model, horizon)
                painter.setPen(Qt.PenStyle.NoPen)
                if value is None:
                    painter.setBrush(QColor("#f4f5f7"))
                    painter.drawRoundedRect(box, 4, 4)
                    painter.setPen(QColor(MUTED))
                    painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), "-")
                    continue
                color = heat_color(value, low, high)
                painter.setBrush(QColor(color))
                painter.drawRoundedRect(box, 4, 4)
                painter.setPen(QColor("#ffffff" if heat_step(value, low, high) >= 3 else TEXT))
                painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), f"{value:,.1f}")

        self._paint_legend(painter, low, high)

    def _paint_legend(self, painter: QPainter, low: float, high: float) -> None:
        top = self.HEADER_HEIGHT + self.CELL_HEIGHT * len(self.rows()) + 6
        size = 12
        x = self.LABEL_WIDTH
        steps = len(HEAT_STEPS)
        painter.setPen(Qt.PenStyle.NoPen)
        for index in range(steps):
            value = low + (high - low) * index / (steps - 1)
            box = QRectF(x + index * (size + 2), top, size, size)
            painter.setPen(QPen(QColor(SWATCH_EDGE), 1))
            painter.setBrush(QColor(heat_color(value, low, high)))
            painter.drawRoundedRect(box, 2, 2)
        font = QFont()
        font.setPointSizeF(8.5)
        painter.setFont(font)
        painter.setPen(QColor(MUTED))
        painter.drawText(
            QRectF(x + steps * (size + 2) + 8, top - 2, 260, size + 4),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"옅음 {low:,.0f}명 · 진함 {high:,.0f}명",
        )

    def paintEvent(self, event):
        if self._pixmap is None or self._pixmap.size() != self.size():
            self._pixmap = self._render()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.end()


class Bar(QFrame):
    """가로 막대 하나 (규모별 오차). 값의 비율만큼 길어진다."""

    MAX_WIDTH = 180
    HEIGHT = 8

    def __init__(self, color: str = BAR):
        super().__init__()
        self._color = color
        self.setFixedHeight(self.HEIGHT)
        self.setStyleSheet(f"background: {color}; border-radius: 4px;")

    def set_value(self, value: float, maximum: float) -> None:
        ratio = 0.0 if maximum <= 0 else max(0.0, min(1.0, value / maximum))
        self.setFixedWidth(max(4, int(self.MAX_WIDTH * ratio)))


class TertileChart(QWidget):
    """이용량 규모(소·중·대)별 오차를 D·E 두 모델로 나란히 보여 준다."""

    ROW_HEIGHT = 30
    LABEL_WIDTH = 28
    VALUE_WIDTH = 62

    def __init__(self, data: AnalysisData):
        super().__init__()
        self._data = data
        models = [name for name in ("D", "E") if name in data.tertiles]
        self._models = models
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        maximum = max(
            (value for values in data.tertiles.values() for value in values.values()), default=0.0
        )
        for row, label in enumerate(("소", "중", "대"), start=1):
            scale = QLabel(label)
            scale.setObjectName("statLabel")
            layout.addWidget(scale, row, 0)
            for column, model in enumerate(models, start=1):
                value = data.tertiles.get(model, {}).get(label)
                if value is None:
                    continue
                cell = QWidget()
                line = QHBoxLayout(cell)
                line.setContentsMargins(0, 0, 0, 0)
                line.setSpacing(8)
                bar = Bar(BAR if model == "D" else BAR_LIGHT)
                bar.set_value(value, maximum)
                text = QLabel(f"{value:,.1f}명")
                text.setObjectName("statValue")
                line.addWidget(bar, 0, Qt.AlignmentFlag.AlignVCenter)
                line.addWidget(text)
                line.addStretch()
                layout.addWidget(cell, row, column)

        # 어느 막대가 어느 모델인지 맨 위에 이름을 둔다 (색만으로 구분하지 않게)
        for column, model in enumerate(models, start=1):
            head = QLabel(model)
            head.setObjectName("statLabel")
            layout.addWidget(head, 0, column)
        layout.setColumnStretch(len(models) + 1, 1)
        self.setToolTip("이용량이 많은 역·시간대일수록 오차가 큽니다 (D 진한 막대, E 옅은 막대)")


class AnalysisView(QScrollArea):
    """분석 화면 전체. 처음 열 때 만들어지고, 모델 카드를 누르면 model_selected 를 보낸다."""

    model_selected = Signal(str)

    def __init__(self, data: AnalysisData | None = None):
        super().__init__()
        self.setObjectName("analysisScroll")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.data = data if data is not None else load_analysis()
        self.setWidget(self._build_page())

    # ---------- 구성 ----------

    def _build_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("analysisPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(CARD_MARGIN, CARD_MARGIN, CARD_MARGIN, CARD_MARGIN)
        layout.setSpacing(12)
        layout.addWidget(self._build_models())
        layout.addWidget(self._build_heatmap())
        layout.addWidget(self._build_lower())
        layout.addStretch()
        return page

    def _card(self, title: str, caption: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("analysisCard")
        body = QVBoxLayout(card)
        body.setContentsMargins(16, 14, 16, 16)
        body.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("analysisTitle")
        body.addWidget(heading)
        if caption:
            note = QLabel(caption)
            note.setObjectName("analysisCaption")
            note.setWordWrap(True)
            body.addWidget(note)
        return card, body

    def _build_models(self) -> QFrame:
        card, body = self._card(
            "모델별 오차",
            "Test 구간(136일)에서 실제 승차 인원과 비교한 평균 오차입니다. 작을수록 정확합니다. "
            "카드를 누르면 노선도로 돌아가 그 모델 기준으로 봅니다.",
        )
        row = QHBoxLayout()
        row.setSpacing(8)
        self.cards: dict[str, ModelCard] = {}
        for rank, stat in enumerate(self.data.models, start=1):
            card_widget = ModelCard(rank, stat.name, stat.mae, stat.rmse, _description(stat.name))
            card_widget.clicked.connect(self.model_selected.emit)
            row.addWidget(card_widget, 1)
            self.cards[stat.name] = card_widget
        body.addLayout(row)
        return card

    def _build_heatmap(self) -> QFrame:
        card, body = self._card(
            "지평 구간별 오차",
            "예측 날짜가 얼마나 먼지에 따라 오차가 어떻게 달라지는지 봅니다. 칸이 진할수록 오차가 큽니다 "
            "(단위: 명). 182~366일 줄은 기준일을 옮겨 가며 다시 맞혀 본 결과입니다.",
        )
        self.heatmap = Heatmap(self.data)
        body.addWidget(self.heatmap)
        self.insight = QLabel(horizon_insight(self.data))
        self.insight.setObjectName("insightText")
        self.insight.setWordWrap(True)
        body.addWidget(self.insight)
        return card

    def _build_lower(self) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        row.addWidget(self._build_tertiles(), 1)
        row.addWidget(self._build_backtest(), 1)
        return holder

    def _build_tertiles(self) -> QFrame:
        card, body = self._card(
            "이용량 규모별 오차",
            "같은 모델도 붐비는 역·시간대에서는 오차가 커집니다.",
        )
        self.tertiles = TertileChart(self.data)
        body.addWidget(self.tertiles)
        note = QLabel(tertile_note(self.data))
        note.setObjectName("analysisCaption")
        note.setWordWrap(True)
        body.addWidget(note)
        body.addStretch()
        return card

    def _build_backtest(self) -> QFrame:
        card, body = self._card(
            "182~366일 앞 재확인",
            "182일 이후는 한 번도 맞혀 본 적이 없어, 기준일을 세 번 옮겨 다시 학습하고 확인했습니다.",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        models = [name for name in ("A", "B", "C") if name in self.data.pooled]
        for column, name in enumerate(("기준일", *models)):
            label = QLabel(name)
            label.setObjectName("statLabel")
            if column:
                label.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(label, 0, column)
        self.backtest_rows: dict[str, dict[str, str]] = {}
        for row, cut in enumerate(self.data.cut_dates, start=1):
            values = self.data.cut_mae.get(cut, {})
            cell = QLabel(cut)
            cell.setObjectName("statValue")
            grid.addWidget(cell, row, 0)
            row_values = {}
            for column, name in enumerate(models, start=1):
                value = values.get(name)
                text = QLabel("-" if value is None else f"{value:,.1f}")
                text.setObjectName("statValue")
                text.setAlignment(Qt.AlignmentFlag.AlignRight)
                grid.addWidget(text, row, column)
                row_values[name] = text.text()
            self.backtest_rows[cut] = row_values
        total_row = len(self.data.cut_dates) + 1
        total = QLabel("전체 평균")
        total.setObjectName("statLabel")
        grid.addWidget(total, total_row, 0)
        for column, name in enumerate(models, start=1):
            value = self.data.pooled.get(name)
            text = QLabel("-" if value is None else f"{value:,.1f}")
            text.setObjectName("statValue")
            text.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(text, total_row, column)
        grid.setColumnStretch(len(models) + 1, 1)
        body.addLayout(grid)
        note = QLabel(backtest_note(self.data))
        note.setObjectName("analysisCaption")
        note.setWordWrap(True)
        body.addWidget(note)
        body.addStretch()
        return card


def _description(model: str) -> str:
    """모델 설명은 결과 카드와 같은 문구를 쓴다 (두 곳에서 다르게 읽히지 않게)."""
    from metro_insight.app.station_panel import MODEL_ROWS

    return dict(MODEL_ROWS).get(model, model)