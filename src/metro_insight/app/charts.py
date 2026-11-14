"""예측값 비교 도트플롯: 다섯 모델의 예측을 점으로, 오차(±MAE)를 가로 막대로 그린다.

축은 0에서 시작하지 않는다. 0부터 그리면 모델 사이의 작은 차이(수백 명)가 보이지 않기 때문에,
(가장 낮은 예측−오차)부터 (가장 높은 예측+오차)까지를 축으로 잡고 양끝에 15% 여백을 둔다.
다만 승차 인원은 음수가 될 수 없으므로 축 하한과 오차 막대의 왼쪽 끝은 0에서 멈춘다.
막대는 ±MAE 다. 이 범위에 실제값이 드는 비율은 100%가 아니므로 옆에 그 비율을 함께 적는다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from metro_insight.app import chart_colors

ROOT = Path(__file__).resolve().parents[3]
BACKTEST_NAME = "backtest_182_366.json"
CALENDAR_MODELS = ("A", "B", "C")  # ±MAE 막대를 붙이는 모델

AXIS_PAD_RATIO = 0.15

# 씬·차트 색은 QSS 를 거치지 않으므로 토큰 값을 이름으로만 붙여 쓴다
TEXT = chart_colors.TEXT
MUTED = chart_colors.MUTED
GRID = chart_colors.GRID
WHISKER = chart_colors.WHISKER
DOT = chart_colors.DOT
SURFACE = chart_colors.SURFACE
BAND = chart_colors.band()


@dataclass(frozen=True)
class Axis:
    """값 → 가로 위치를 정하는 축."""

    lo: float
    hi: float

    @property
    def span(self) -> float:
        return self.hi - self.lo

    def fraction(self, value: float) -> float:
        """축 안에서의 위치 (0 = 왼쪽 끝, 1 = 오른쪽 끝)."""
        if self.span <= 0:
            return 0.5
        return (value - self.lo) / self.span


def plot_axis(values: dict[str, float], errors: dict[str, float], pad: float = AXIS_PAD_RATIO) -> Axis | None:
    """점과 막대가 모두 들어가는 축. 예측값이 없으면 None. 하한은 0 아래로 내려가지 않는다."""
    known = {model: float(value) for model, value in values.items() if value is not None}
    if not known:
        return None
    # 승차 인원은 음수가 될 수 없다. 오차가 예측값보다 커도 막대는 0에서 멈춘다.
    lows = [max(value - abs(errors.get(model, 0.0)), 0.0) for model, value in known.items()]
    highs = [value + abs(errors.get(model, 0.0)) for model, value in known.items()]
    lo, hi = min(lows), max(highs)
    if hi <= lo:  # 한 모델만 있거나 모두 같은 값이면 눈금이 0이 되므로 폭을 준다
        pad_value = max(abs(lo) * 0.05, 1.0)
        lo, hi = lo - pad_value, hi + pad_value
    margin = (hi - lo) * pad
    return Axis(max(lo - margin, 0.0), hi + margin)


def value_x(axis: Axis, value: float, left: float, width: float) -> float:
    """값 하나의 가로 위치(픽셀)."""
    return left + axis.fraction(value) * width


def whisker_span(
    axis: Axis, value: float, error: float, left: float, width: float
) -> tuple[float, float]:
    """오차 막대의 (왼쪽 끝, 오른쪽 끝) 픽셀 위치. 왼쪽 끝은 0명에서 멈춘다."""
    error = abs(error)
    return (
        value_x(axis, max(value - error, 0.0), left, width),
        value_x(axis, value + error, left, width),
    )


def coverage_share(path: str | Path | None = None) -> float | None:
    """±MAE 범위에 실제값이 든 비율의 중앙값.

    182~366일 백테스트 기록의 오차 막대 포함률(A·B·C)을 쓴다. 그 기록이 없으면 None.
    """
    target = Path(path) if path is not None else ROOT / "models" / BACKTEST_NAME
    key = str(target)
    if key in _SHARE_CACHE:
        return _SHARE_CACHE[key]
    share = _read_share(target)
    _SHARE_CACHE[key] = share
    return share


_SHARE_CACHE: dict[str, float | None] = {}


def _read_share(target: Path) -> float | None:
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    shares: list[float] = []
    for cut in (data.get("per_cut") or {}).values():
        for model, entry in (cut.get("models") or {}).items():
            if model not in CALENDAR_MODELS:
                continue
            share = entry.get("coverage_mae") if isinstance(entry, dict) else None
            if isinstance(share, (int, float)) and 0 < share <= 1:
                shares.append(float(share))
    if not shares:
        return None
    shares.sort()
    middle = len(shares) // 2
    if len(shares) % 2:
        return shares[middle]
    return (shares[middle - 1] + shares[middle]) / 2


def error_caption(share: float | None) -> str:
    """오차 막대가 무엇을 뜻하는지 밝히는 문장.

    포함률은 Test 구간이 아니라 182~366일 백테스트에서 잰 값이라, 어느 자료에서 나온
    숫자인지 문장 안에 적는다. 기록이 없으면 숫자를 지어내지 않는다.
    """
    if share is None:
        return "오차 막대는 ±MAE 기준입니다."
    return (
        "오차 막대는 ±MAE 기준입니다. "
        f"182~366일 백테스트에서 이 방식의 실제 포함률은 약 {share:.0%}였습니다"
    )


def spread_note(values: dict[str, float], errors: dict[str, float]) -> str:
    """모델 사이 차이와 오차 크기를 견줘, 순위를 믿을 수 있는지 말해 주는 문장."""
    known = {model: float(value) for model, value in values.items() if value is not None}
    if len(known) < 2:
        return ""
    spread = max(known.values()) - min(known.values())
    errs = [abs(errors[model]) for model in known if model in errors]
    if not errs:
        return f"모델 사이 차이는 {spread:,.0f}명입니다."
    smallest, largest = min(errs), max(errs)
    if spread < smallest:
        return (
            f"{len(known)}개 모델의 차이({spread:,.0f}명)가 가장 작은 오차(±{smallest:,.0f})보다 "
            "작아, 이 순서로 우열을 가르기 어렵습니다."
        )
    if spread < largest:
        return (
            f"{len(known)}개 모델의 차이는 {spread:,.0f}명입니다. 가장 작은 오차(±{smallest:,.0f})보다는 "
            f"크지만 가장 큰 오차(±{largest:,.0f})보다는 작아, 순위가 뒤집힐 수 있습니다."
        )
    return (
        f"{len(known)}개 모델의 차이는 {spread:,.0f}명(오차 ±{smallest:,.0f}~{largest:,.0f})으로, "
        "가장 큰 오차보다도 큽니다."
    )


class DotPlot(QWidget):
    """모델별 예측 점과 ±MAE 막대. 행에 마우스를 올리면 그 행만 강조한다."""

    hover_model = Signal(object)  # 강조할 모델 이름 또는 None

    ROW_HEIGHT = 24
    GUTTER_LEFT = 48
    GUTTER_RIGHT = 68
    AXIS_HEIGHT = 28
    DOT_RADIUS = 4.5
    HIGHLIGHT_RADIUS = 6.0

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumHeight(self.AXIS_HEIGHT + self.ROW_HEIGHT)
        self._values: dict[str, float] = {}
        self._errors: dict[str, float] = {}
        self._axis: Axis | None = None
        self._highlight: str | None = None

    # ---------- 데이터 ----------

    def set_data(self, values: dict[str, float], errors: dict[str, float]) -> None:
        """모델별 예측값과 오차(MAE). 없는 모델은 넘기지 않는다."""
        self._values = {model: float(value) for model, value in values.items() if value is not None}
        self._errors = {
            model: abs(float(value))
            for model, value in errors.items()
            if value is not None and model in self._values
        }
        self._axis = plot_axis(self._values, self._errors)
        self.setMinimumHeight(self.AXIS_HEIGHT + self.ROW_HEIGHT * max(len(self._values), 1))
        self.updateGeometry()
        self.update()

    @property
    def values(self) -> dict[str, float]:
        return dict(self._values)

    @property
    def errors(self) -> dict[str, float]:
        return dict(self._errors)

    @property
    def axis(self) -> Axis | None:
        return self._axis

    @property
    def best_model(self) -> str | None:
        """오차가 가장 작은 모델 (막대가 없으면 None)."""
        if not self._errors:
            return None
        return min(self._errors, key=lambda model: self._errors[model])

    def models(self) -> list[str]:
        return list(self._values)

    def plot_rect(self) -> QRectF:
        return QRectF(
            self.GUTTER_LEFT,
            0.0,
            max(self.width() - self.GUTTER_LEFT - self.GUTTER_RIGHT, 1.0),
            self._plot_height(),
        )

    def _plot_height(self) -> float:
        return max(self.height() - self.AXIS_HEIGHT, self.ROW_HEIGHT)

    def row_index(self, model: str) -> int:
        return self.models().index(model)

    def row_y(self, model: str) -> float:
        """행 가운데의 세로 위치."""
        return self.row_index(model) * self.ROW_HEIGHT + self.ROW_HEIGHT / 2

    def model_at(self, y: float) -> str | None:
        models = self.models()
        if not models or y < 0 or y > self._plot_height():
            return None
        index = int(y // self.ROW_HEIGHT)
        return models[index] if index < len(models) else None

    def dot_position(self, model: str) -> QPointF:
        """점의 픽셀 위치 (데이터가 없으면 (0, 0))."""
        if self._axis is None or model not in self._values:
            return QPointF(0, 0)
        rect = self.plot_rect()
        x = value_x(self._axis, self._values[model], rect.left(), rect.width())
        return QPointF(x, self.row_y(model))

    def whisker_pixels(self, model: str) -> tuple[float, float]:
        """막대의 (왼쪽, 오른쪽) 픽셀. 오차가 없으면 두 값이 같다."""
        if self._axis is None or model not in self._values:
            return 0.0, 0.0
        rect = self.plot_rect()
        return whisker_span(
            self._axis, self._values[model], self._errors.get(model, 0.0), rect.left(), rect.width()
        )

    # ---------- 강조 ----------

    def set_highlight(self, model: str | None) -> None:
        if model == self._highlight:
            return
        self._highlight = model
        self.update()

    @property
    def highlight(self) -> str | None:
        return self._highlight

    def mouseMoveEvent(self, event):
        model = self.model_at(event.position().y())
        if model != self._highlight:
            self._highlight = model
            self.hover_model.emit(model)
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._highlight is not None:
            self._highlight = None
            self.hover_model.emit(None)
            self.update()
        super().leaveEvent(event)

    # ---------- 그리기 ----------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._axis is None or not self._values:
            painter.end()
            return
        rect = self.plot_rect()
        best = self.best_model
        for model in self.models():
            y = self.row_y(model)
            if model == self._highlight:
                painter.fillRect(
                    QRectF(0, y - self.ROW_HEIGHT / 2, self.width(), self.ROW_HEIGHT), BAND
                )
            self._paint_label(painter, model, y, model == best)
            self._paint_row(painter, model, y, rect)
        self._paint_axis(painter, rect)
        painter.end()

    def _paint_label(self, painter: QPainter, model: str, y: float, is_best: bool) -> None:
        font = QFont()
        font.setPointSizeF(9.5)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor(TEXT)))
        painter.drawText(
            QRectF(4, y - self.ROW_HEIGHT / 2, 20, self.ROW_HEIGHT),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            model,
        )
        if is_best:
            chip = QRectF(22, y - 8, 24, 16)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(DOT))
            painter.drawRoundedRect(chip, 4, 4)
            chip_font = QFont()
            chip_font.setPointSizeF(7.5)
            chip_font.setWeight(QFont.Weight.Bold)
            painter.setFont(chip_font)
            painter.setPen(QPen(QColor(SURFACE)))
            painter.drawText(chip, int(Qt.AlignmentFlag.AlignCenter), "기준")

    def _paint_row(self, painter: QPainter, model: str, y: float, rect: QRectF) -> None:
        value = self._values[model]
        error = self._errors.get(model, 0.0)
        left_x, right_x = self.whisker_pixels(model)
        if error > 0:
            pen = QPen(QColor(WHISKER), 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(left_x, y), QPointF(right_x, y))
            cap = QPen(QColor(WHISKER), 2)
            cap.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(cap)
            for x in (left_x, right_x):
                painter.drawLine(QPointF(x, y - 4), QPointF(x, y + 4))
        dot = self.dot_position(model)
        radius = self.HIGHLIGHT_RADIUS if model == self._highlight else self.DOT_RADIUS
        painter.setPen(QPen(QColor(SURFACE), 1.5))
        painter.setBrush(QColor(DOT))
        painter.drawEllipse(dot, radius, radius)

        value_font = QFont()
        value_font.setPointSizeF(9.0)
        value_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(value_font)
        painter.setPen(QPen(QColor(TEXT)))
        painter.drawText(
            QRectF(
                self.width() - self.GUTTER_RIGHT + 4,
                y - self.ROW_HEIGHT / 2,
                self.GUTTER_RIGHT - 12,
                self.ROW_HEIGHT,
            ),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            f"{value:,.0f}",
        )

    def _paint_axis(self, painter: QPainter, rect: QRectF) -> None:
        base = self._plot_height()
        painter.setPen(QPen(QColor(GRID), 1))
        painter.drawLine(QPointF(rect.left(), base), QPointF(rect.right(), base))
        font = QFont()
        font.setPointSizeF(8.5)
        painter.setFont(font)
        painter.setPen(QPen(QColor(MUTED)))
        for fraction, align in (
            (0.0, Qt.AlignmentFlag.AlignLeft),
            (0.5, Qt.AlignmentFlag.AlignHCenter),
            (1.0, Qt.AlignmentFlag.AlignRight),
        ):
            value = self._axis.lo + self._axis.span * fraction
            x = rect.left() + rect.width() * fraction
            painter.drawLine(QPointF(x, base), QPointF(x, base + 4))
            painter.drawText(
                QRectF(x - 40, base + 6, 80, 16),
                int(align | Qt.AlignmentFlag.AlignTop),
                f"{value:,.0f}",
            )