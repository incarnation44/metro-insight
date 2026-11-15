"""1~8호선 노선도. 가로·세로·45° 선의 노선도 배치(schematic_layout)로 그리고, 역을 클릭해 선택한다.

확대 정도에 따라 역 이름을 띄우고, 예측 데이터가 없는 역은 점선·회색으로, 환승역은 회색으로
그린다. 화살표 키로 이웃 역을 옮겨 다니고 Enter 로 패널을 열 수 있다.
"""
import math

from PySide6.QtCore import QEasingCurve, QEvent, QPointF, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QTransform
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from metro_insight.app import chart_colors
from metro_insight.app.coverage import Coverage, load_coverage
from metro_insight.app.map_overlays import Attribution, LineLegend, ZoomControls
from metro_insight.app.palette import gap_palette
from metro_insight.app.schematic_layout import UNIT, SchematicLayout, metro_layout
from metro_insight.app.subway_data import Line, Station, load_network

# 처음 화면: 2호선 일대 범위에 이만큼 여백을 더해 보여준다 (범위 대비 비율)
INITIAL_VIEW_PADDING = 0.06
INITIAL_MIN_ZOOM = 0.85  # 창이 작아도 이보다 작게 시작하지 않는다 (글씨가 너무 작아지지 않게)
CORNER_RADIUS = 0.6 * UNIT  # 노선이 꺾이는 곳의 둥근 모서리 반지름

# 확대 배율(씬 1단위당 화면 픽셀)이 이 값 이상일 때 역 이름 표시 (환승역은 더 일찍)
TRANSFER_LABEL_MIN_ZOOM = 0.2
LABEL_MIN_ZOOM = 0.30
LABEL_SIDES = ("right", "left", "above", "below")  # 겹치면 다음 위치를 시도한다
MAX_ZOOM = 5.0
WHEEL_ZOOM_STEP = 1.2
BUTTON_ZOOM_STEP = 1.6
ZOOM_ANIMATION_MS = 220
OVERLAY_MARGIN = 16
FOCUS_MIN_ZOOM = 1.2  # 검색으로 고른 역으로 이동할 때 이보다 작으면 이만큼 확대한다
REVEAL_MARGIN = 60  # 선택한 역이 화면 가장자리에서 이보다 가까우면 가운데로 옮긴다

# 새로 나타난 역 이름은 이 시간 동안 서서히 진해진다 (툭 나타나지 않게)
LABEL_FADE_MS = 90
LABEL_FADE_STEP_MS = 15

# 씬 아이템에는 QSS가 적용되지 않아 색상을 여기서 지정한다 (값은 styles/tokens.json → chart_colors).
# 실제 색은 StationMapView 가 들고 있는 색 묶음(모드별)에서 온다 — apply_theme(mode) 로 바뀐다.
SELECTED_HALO_ALPHA = 46  # 고른 역 뒤에 까는 테두리의 투명도 (0~255)
LINE_WIDTH = 4.5
LINE_CASING_EXTRA = 3  # 노선 아래 흰 테두리: 교차·인접 구간을 또렷하게
LABEL_POINT_SIZE = 9.0  # 가장 작게 보일 때의 역 이름 크기 (pt). 이보다 작게 줄이지 않는다
SELECTED_LABEL_POINT_SIZE = 11.0

# 확대할수록 글씨·역 점·노선을 키운다: 크기 배율 = (확대 배율 / SIZE_REFERENCE_ZOOM)^SIZE_ZOOM_POWER
# 위 크기들은 SIZE_REFERENCE_ZOOM에서의 크기. 너무 작아지거나 커지지 않게 범위를 둔다
SIZE_REFERENCE_ZOOM = 1.0
SIZE_ZOOM_POWER = 0.6
SIZE_MIN, SIZE_MAX = 1.0, 1.9
SIZE_STEP = 0.05  # 크기 배율을 이 단위로 끊어, 확대 중 글자 모양을 매번 다시 만들지 않게 한다

# 화살표 키 → (오른쪽, 아래) 단위 벡터
ARROW_DIRECTIONS = {
    Qt.Key.Key_Left: (-1.0, 0.0),
    Qt.Key.Key_Right: (1.0, 0.0),
    Qt.Key.Key_Up: (0.0, -1.0),
    Qt.Key.Key_Down: (0.0, 1.0),
}
ARROW_MIN_ALIGN = 0.2  # 이보다 더 옆으로 벗어난 역은 그 방향의 이웃으로 보지 않는다


def rounded_path(points: list[tuple[float, float]], radius: float) -> QPainterPath:
    """꺾이는 점마다 모서리를 둥글린 경로."""
    path = QPainterPath(QPointF(*points[0]))
    for prev, corner, nxt in zip(points, points[1:], points[2:], strict=False):
        d_in, d_out = math.dist(prev, corner), math.dist(corner, nxt)
        r = min(radius, d_in / 2, d_out / 2)
        if r <= 0:
            path.lineTo(QPointF(*corner))
            continue
        path.lineTo(_toward(corner, prev, r / d_in))
        path.quadTo(QPointF(*corner), _toward(corner, nxt, r / d_out))
    path.lineTo(QPointF(*points[-1]))
    return path


def _toward(p: tuple[float, float], q: tuple[float, float], t: float) -> QPointF:
    return QPointF(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def direction_score(dx: float, dy: float, direction: tuple[float, float]) -> float:
    """(dx, dy) 방향이 화살표 방향과 얼마나 잘 맞는지. 1에 가까울수록 정면이다.

    방향이 반대이거나 옆으로 크게 벗어나면 음수가 될 수 있다.
    """
    length = math.hypot(dx, dy)
    if length == 0:
        return -1.0
    unit_x, unit_y = dx / length, dy / length
    along = unit_x * direction[0] + unit_y * direction[1]
    across = abs(unit_x * direction[1] - unit_y * direction[0])
    return along - across


class StationLabel(QGraphicsItem):
    """테두리(halo)를 두른 역 이름. 노선 위에 겹쳐도 읽히게 한다."""

    def __init__(
        self,
        text: str,
        parent: QGraphicsItem,
        emphasized: bool,
        color: str | None = None,
        halo: str | None = None,
    ):
        super().__init__(parent)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._text = text
        self._emphasized = emphasized  # 환승역은 조금 굵게
        self._color = color or str(chart_colors.LABEL_COLOR)
        self._halo = halo or str(chart_colors.LABEL_HALO)
        self._selected_color = str(chart_colors.SELECTED_LABEL_COLOR)
        self._selected = False
        self._size = 1.0
        self._build_path()

    def set_selected(self, selected: bool) -> None:
        self.prepareGeometryChange()
        self._selected = selected
        self._build_path()

    def set_color(self, color: str) -> None:
        if color != self._color:
            self._color = color
            self.update()

    def apply_theme(self, colors: dict) -> None:
        """그 모드의 이름 테두리 색으로 바꾼다 (글자색은 역 상태에 따라 따로 정한다)."""
        self._halo = str(colors["LABEL_HALO"])
        self._selected_color = str(colors["SELECTED_LABEL_COLOR"])
        self.update()

    def set_size(self, size: float) -> None:
        self.prepareGeometryChange()
        self._size = size
        self._build_path()

    def _build_path(self) -> None:
        font = QFont()
        if self._selected:
            font.setPointSizeF(SELECTED_LABEL_POINT_SIZE * self._size)
            font.setWeight(QFont.Weight.Bold)
        else:
            font.setPointSizeF(LABEL_POINT_SIZE * self._size)
            font.setWeight(QFont.Weight.DemiBold if self._emphasized else QFont.Weight.Normal)
        self._path = QPainterPath()
        self._path.addText(0, 0, font, self._text)

    def text_rect(self) -> QRectF:
        return self._path.boundingRect()

    def boundingRect(self) -> QRectF:
        return self.text_rect().adjusted(-3, -3, 3, 3)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        halo = QPen(QColor(self._halo), 3.5 * self._size)
        halo.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(self._path, halo)
        painter.fillPath(self._path, QColor(self._selected_color if self._selected else self._color))


class StationItem(QGraphicsEllipseItem):
    """노선도 위의 역 한 개. 확대·축소해도 화면상 크기가 같다."""

    RADIUS = 3.5
    TRANSFER_RADIUS = 5.5
    HOVER_GROW = 1.5
    SELECTED_GROW = 2
    HALO = 7  # 선택한 역 둘레의 반투명 원

    def __init__(
        self,
        station: Station,
        pos: QPointF,
        line_color: str,
        on_click,
        missing_lines: tuple[str, ...] = (),
        colors: dict | None = None,
        gap_colors: dict | None = None,
    ):
        super().__init__()
        self.station = station
        self.missing_lines = tuple(missing_lines)  # 예측 데이터가 없는 호선 이름들
        self._line_color = line_color
        self._on_click = on_click
        self._colors = dict(colors or chart_colors.colors_for())
        self._gap_colors = dict(gap_colors or gap_palette(chart_colors.get_mode()))
        self._selected = False
        self._hovered = False
        self._size = 1.0
        self._label_side = LABEL_SIDES[0]

        self.setPos(pos)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(self._tooltip())
        self.setZValue(2 if station.is_transfer else 1)

        self.label = StationLabel(
            station.name,
            self,
            emphasized=station.is_transfer,
            color=self._label_color(gap=self.is_gap),
            halo=str(self._colors["LABEL_HALO"]),
        )
        self._refresh()

    def _label_color(self, gap: bool) -> str:
        return str(self._gap_colors["GAP_LABEL"] if gap else self._colors["LABEL_COLOR"])

    def apply_theme(self, colors: dict, gap_colors: dict | None = None) -> None:
        """그 모드의 색으로 역 점·테두리·이름을 다시 칠한다."""
        self._colors = dict(colors)
        if gap_colors is not None:
            self._gap_colors = dict(gap_colors)
        self.label.apply_theme(self._colors)
        self._refresh()

    def _tooltip(self) -> str:
        lines = " · ".join(f"{n}호선" for n in self.station.lines)
        text = f"{self.station.name}\n{lines}"
        if self.is_gap:
            return f"{text}\n예측 데이터 없음"
        if self.missing_lines:
            return f"{text}\n예측 데이터 없음: {', '.join(self.missing_lines)}"
        return text

    @property
    def is_gap(self) -> bool:
        """이 역의 어느 호선에도 예측 데이터가 없는가."""
        return bool(self.station.lines) and len(self.missing_lines) == len(self.station.lines)

    @property
    def has_data(self) -> bool:
        return not self.is_gap

    @property
    def is_selected(self) -> bool:
        return self._selected

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.label.set_selected(selected)
        self.setZValue(3 if selected else (2 if self.station.is_transfer else 1))
        self._refresh()

    @property
    def label_priority(self) -> tuple[bool, int]:
        """이름이 겹칠 때 먼저 표시할 순서: 선택한 역 → 지나는 호선이 많은 역."""
        return (self._selected, len(self.station.lines))

    def wants_label(self, zoom: float) -> bool:
        min_zoom = TRANSFER_LABEL_MIN_ZOOM if self.station.is_transfer else LABEL_MIN_ZOOM
        return self._selected or zoom >= min_zoom

    def set_size(self, size: float) -> None:
        """확대 정도에 따른 크기 배율: 점·테두리·이름을 함께 키운다."""
        self.prepareGeometryChange()
        self._size = size
        self.label.set_size(size)
        self._refresh()

    def radius(self) -> float:
        r = self.TRANSFER_RADIUS if self.station.is_transfer else self.RADIUS
        if self._selected:
            r += self.SELECTED_GROW
        elif self._hovered:
            r += self.HOVER_GROW
        return r * self._size

    def label_offset(self, side: str) -> QPointF:
        """역 중심에서 이름 기준점까지의 화면 픽셀 오프셋."""
        r, text, gap = self.radius(), self.label.text_rect(), 4 * self._size
        if side == "right":
            return QPointF(r + gap, -text.center().y())
        if side == "left":
            return QPointF(-r - gap - text.right(), -text.center().y())
        if side == "above":
            return QPointF(-text.center().x(), -r - gap - text.bottom())
        return QPointF(-text.center().x(), r + gap - text.top())

    def place_label(self, side: str) -> None:
        self._label_side = side
        self.label.setPos(self.label_offset(side))

    def _refresh(self) -> None:
        r = self.radius()
        self.setRect(-r, -r, 2 * r, 2 * r)
        gap = self.is_gap and not self._selected
        if self.station.is_transfer or self._selected:
            border, style = str(self._colors["TRANSFER_BORDER"]), Qt.PenStyle.SolidLine
        elif gap:
            border, style = str(self._gap_colors["GAP_BORDER"]), Qt.PenStyle.DashLine
        else:
            border, style = self._line_color, Qt.PenStyle.SolidLine
        pen = QPen(QColor(border), (2.25 if self.station.is_transfer else 2) * self._size)
        pen.setStyle(style)
        self.setPen(pen)
        if self._selected:
            fill = str(self._colors["SELECTED_FILL"])
        elif gap:
            fill = str(self._gap_colors["GAP_FILL"])
        else:
            fill = str(self._colors["STATION_FILL"])
        self.setBrush(QBrush(QColor(fill)))
        self.label.set_color(self._label_color(gap=gap))
        self.place_label(self._label_side)

    def boundingRect(self) -> QRectF:
        h = self.HALO * self._size
        return super().boundingRect().adjusted(-h, -h, h, h)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._selected:
            r = self.radius() + self.HALO * self._size
            halo = QColor(str(self._colors["SELECTED_FILL"]))
            halo.setAlpha(SELECTED_HALO_ALPHA)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(QPointF(0, 0), r, r)
        super().paint(painter, option, widget)

    def hoverEnterEvent(self, event):
        self._hovered = True
        self._refresh()

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self._refresh()

    def mousePressEvent(self, event):
        event.accept()  # 지도 드래그가 시작되지 않게 한다
        self._on_click(self.station)


class StationMapView(QGraphicsView):
    """1~8호선 노선도. 휠로 확대·축소, 드래그로 이동, 역을 클릭하면 station_selected 시그널을 보낸다."""

    station_selected = Signal(object)  # Station
    station_activated = Signal(object)  # Enter 로 연 역 (패널 열기)
    panel_close_requested = Signal()  # Esc

    def __init__(self, coverage: Coverage | None = None):
        super().__init__()
        self.setObjectName("stationMap")
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # 드래그로 이동
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # 화살표 키로 역을 옮겨 다닌다
        self.setMinimumSize(480, 480)

        self._stations: dict[str, StationItem] = {}
        self._line_items: list[tuple[QGraphicsPathItem, float]] = []  # (노선 경로, 기본 두께)
        self._casing_items: list[QGraphicsPathItem] = []  # 노선 아래 테두리 (모드마다 색이 바뀐다)
        self._mode = chart_colors.DEFAULT_MODE
        self._colors = chart_colors.colors_for(self._mode)
        self._gap_colors = gap_palette(self._mode)
        self._size = None
        self._selected: StationItem | None = None
        self._initial_view_done = False
        self._refit_on_open = True  # 패널이 처음 열려 지도가 좁아지면 그 폭에 맞춰 다시 맞춘다
        self._overlays: list[tuple[QWidget, Qt.Corner]] = []
        self._fading: set[StationLabel] = set()
        self._fade_elapsed = 0
        self._zoom_animation = QVariantAnimation(self)
        self._zoom_animation.setDuration(ZOOM_ANIMATION_MS)
        self._zoom_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._zoom_animation.valueChanged.connect(self._on_zoom_animation)
        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(LABEL_FADE_STEP_MS)
        self._fade_timer.timeout.connect(self._on_label_fade)

        self.lines, self.stations = load_network()
        if coverage is None:
            coverage = load_coverage(lines=self.lines, stations=self.stations)
        self.coverage = coverage
        self.layout = metro_layout(self.lines, self.stations)
        self._neighbors = self._build_neighbors(self.lines)
        self._draw(self.lines, self.stations, self.layout)

        controls = ZoomControls()
        controls.zoom_in.connect(lambda: self.animate_zoom(BUTTON_ZOOM_STEP))
        controls.zoom_out.connect(lambda: self.animate_zoom(1 / BUTTON_ZOOM_STEP))
        controls.reset.connect(self.animate_to_initial_view)
        self.add_overlay(LineLegend(self.lines, self.coverage), Qt.Corner.TopLeftCorner)
        self.add_overlay(controls, Qt.Corner.TopRightCorner)
        self.add_overlay(Attribution("© OpenStreetMap contributors"), Qt.Corner.BottomRightCorner)

    @staticmethod
    def _build_neighbors(lines: list[Line]) -> dict[str, tuple[str, ...]]:
        """역 이름 → 바로 옆 역들 (호선 구간으로 이어진 역)."""
        adjacent: dict[str, set[str]] = {}
        for line in lines:
            for a, b in line.edges:
                adjacent.setdefault(a, set()).add(b)
                adjacent.setdefault(b, set()).add(a)
        return {name: tuple(sorted(peers)) for name, peers in adjacent.items()}

    def _draw(self, lines: list[Line], stations: list[Station], layout: SchematicLayout) -> None:
        scene = self.scene()
        colors = {line.id: line.color for line in lines}
        names = {line.id: line.name for line in lines}
        pos = {name: QPointF(x, y) for name, (x, y) in layout.positions.items()}
        casing = str(self._colors["LABEL_HALO"])

        def line_pen(color: str, width: float) -> QPen:
            pen = QPen(QColor(color), width)
            pen.setCosmetic(True)  # 확대해도 선 두께 유지
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            return pen

        for line in lines:
            # (z, 색, 두께, 테두리 여부). 테두리는 바탕색이라 모드에 따라 바뀐다.
            layers = ((-1, casing, LINE_WIDTH + LINE_CASING_EXTRA, True), (0, line.color, LINE_WIDTH, False))
            for z, color, width, is_casing in layers:
                for a, b in line.edges:
                    points = layout.paths.get((line.id, a, b)) or layout.paths[(line.id, b, a)][::-1]
                    item = QGraphicsPathItem(rounded_path(points, CORNER_RADIUS))
                    item.setPen(line_pen(color, width))
                    item.setZValue(z)
                    scene.addItem(item)
                    self._line_items.append((item, width))
                    if is_casing:
                        self._casing_items.append(item)

        for station in stations:
            missing = tuple(
                names[line_id]
                for line_id in station.lines
                if self.coverage.is_missing(names[line_id], station.name)
            )
            item = StationItem(
                station,
                pos[station.name],
                colors[station.lines[0]],
                self.select_station,
                missing_lines=missing,
                colors=self._colors,
                gap_colors=self._gap_colors,
            )
            scene.addItem(item)
            self._stations[station.name] = item

        # 가장자리 역이 잘리지 않게 여백을 두고 이동 범위를 정한다
        margin = 2 * UNIT
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-margin, -margin, margin, margin))

    def apply_theme(self, mode: str = "light") -> None:
        """노선도 색을 그 모드로 바꾼다 (바탕색은 QSS 가 함께 바뀐다).

        노선 선은 공식 노선색이라 그대로 두고, 바탕에 가까운 색(노선 테두리·역 점·이름·
        테두리·미수집 역)만 그 모드 값으로 갈아 끼운다.
        """
        self._mode = str(mode)
        self._colors = chart_colors.colors_for(mode)
        self._gap_colors = gap_palette(mode)
        casing = QColor(str(self._colors["LABEL_HALO"]))
        for item in self._casing_items:
            pen = item.pen()
            pen.setColor(casing)
            item.setPen(pen)
        for station in self._stations.values():
            station.apply_theme(self._colors, self._gap_colors)
        scene = self.scene()
        if scene is not None:
            scene.update()
        self.viewport().update()

    @property
    def mode(self) -> str:
        """지금 노선도가 쓰는 화면 모드."""
        return self._mode

    @property
    def selected_station(self) -> Station | None:
        return None if self._selected is None else self._selected.station

    @property
    def station_items(self) -> dict[str, StationItem]:
        """역 이름 → 지도 위의 역 (검사·안내용)."""
        return dict(self._stations)

    def select_station(self, station: Station) -> None:
        if self._selected is not None:
            self._selected.set_selected(False)
        self._selected = self._stations[station.name]
        self._selected.set_selected(True)
        self._update_labels()
        self.station_selected.emit(station)

    def focus_station(self, station: Station) -> None:
        """역을 선택하고 그 역이 화면 가운데에 오도록 부드럽게 이동한다."""
        self.select_station(station)
        self._animate_to(max(self.zoom(), FOCUS_MIN_ZOOM), self._stations[station.name].pos())

    def reveal_selected(self) -> None:
        """선택한 역이 화면 밖이나 가장자리에 있으면 가운데로 옮긴다 (패널이 열려 지도가 좁아졌을 때)."""
        if self._selected is None:
            # 고른 역이 없으면 지도 전체가 보이도록 다시 맞춘다.
            # 처음 한 번만 한다 (그 뒤에 사용자가 옮긴 화면을 되돌리지 않게).
            if self._refit_on_open:
                self._refit_on_open = False
                self.animate_to_initial_view()
            return
        if self._zoom_animation.state() == QVariantAnimation.State.Running:
            return
        m = REVEAL_MARGIN
        visible = self.viewport().rect().adjusted(m, m, -m, -m)
        if not visible.contains(self.mapFromScene(self._selected.pos())):
            self._animate_to(self.zoom(), self._selected.pos())

    # ---------- 키보드 ----------

    def arrow_target(self, direction: tuple[float, float]) -> Station | None:
        """지금 고른 역에서 그 방향으로 가장 잘 놓인 이웃 역.

        바로 옆 역을 먼저 보고, 그 방향에 이웃이 없으면 같은 방향에 있는 역 중 가까운 역을 고른다
        (노선 끝에서도 화살표로 계속 옮겨 다닐 수 있게).
        """
        if self._selected is None:
            return None
        origin = self._selected.pos()
        neighbors = self._neighbors.get(self._selected.station.name, ())
        for candidates in (neighbors, tuple(self._stations)):
            best, best_score = None, ARROW_MIN_ALIGN
            for name in candidates:
                if name == self._selected.station.name:
                    continue
                point = self._stations[name].pos()
                dx, dy = point.x() - origin.x(), point.y() - origin.y()
                score = direction_score(dx, dy, direction)
                if score <= best_score:
                    continue
                # 방향이 비슷하면 가까운 역을 고른다
                best_score = score
                best = self._stations[name].station
            if best is not None:
                return best
        return None

    def keyPressEvent(self, event):
        key = event.key()
        if key in ARROW_DIRECTIONS:
            target = self.arrow_target(ARROW_DIRECTIONS[key])
            if target is not None:
                self.focus_station(target)
                event.accept()
                return
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._selected is not None:
                self.station_activated.emit(self._selected.station)
                event.accept()
                return
        elif key == Qt.Key.Key_Escape:
            self.panel_close_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    # ---------- 지도 위 카드 ----------

    def add_overlay(self, widget: QWidget, corner: Qt.Corner) -> None:
        """지도 모서리에 위젯을 겹쳐 띄운다. 같은 모서리에 여러 개면 안쪽으로 쌓는다."""
        widget.setParent(self)
        widget.installEventFilter(self)  # 크기가 바뀌면 다시 배치
        self._overlays.append((widget, corner))
        widget.show()
        self._place_overlays()

    def eventFilter(self, watched, event) -> bool:
        if event.type() in (QEvent.Type.LayoutRequest, QEvent.Type.Resize):
            self._place_overlays()
        return super().eventFilter(watched, event)

    def _place_overlays(self) -> None:
        stacked: dict[Qt.Corner, int] = {}
        for widget, corner in self._overlays:
            size = widget.sizeHint().expandedTo(widget.minimumSizeHint())
            if widget.size() != size:
                widget.resize(size)
            left = corner in (Qt.Corner.TopLeftCorner, Qt.Corner.BottomLeftCorner)
            top = corner in (Qt.Corner.TopLeftCorner, Qt.Corner.TopRightCorner)
            offset = stacked.get(corner, OVERLAY_MARGIN)
            x = OVERLAY_MARGIN if left else self.width() - size.width() - OVERLAY_MARGIN
            y = offset if top else self.height() - size.height() - offset
            widget.move(x, y)
            stacked[corner] = offset + size.height() + 8

    # ---------- 확대·축소 ----------

    def zoom(self) -> float:
        """확대 배율: 씬 1단위당 화면 픽셀 수."""
        return self.transform().m11()

    def min_zoom(self) -> float:
        rect = self.sceneRect()
        view = self.viewport().size()
        return min(view.width() / rect.width(), view.height() / rect.height())

    def _update_labels(self) -> None:
        """확대 정도에 맞춰 역 이름을 표시한다.

        우선순위가 높은 역부터 오른쪽·왼쪽·위·아래 순으로 자리를 찾고,
        이미 놓인 이름이나 다른 역 점과 겹치는 자리밖에 없으면 이름을 숨긴다.
        새로 나타난 이름은 잠깐에 걸쳐 진해진다.
        """
        self._sync_size()
        zoom = self.zoom()
        to_viewport = self.viewportTransform()
        area = QRectF(self.viewport().rect()).adjusted(-50, -50, 50, 50)
        centers = {item: to_viewport.map(item.pos()) for item in self._stations.values()}
        in_view = [item for item, c in centers.items() if area.contains(c)]

        dots = _RectGrid()
        for item in in_view:
            r = item.radius() + 1
            dots.add(QRectF(centers[item].x() - r, centers[item].y() - r, 2 * r, 2 * r), item)
        placed = _RectGrid()
        labeled = set()

        for item in sorted(in_view, key=lambda i: i.label_priority, reverse=True):
            if not item.wants_label(zoom):
                continue
            for side in LABEL_SIDES:
                rect = item.label.boundingRect().translated(centers[item] + item.label_offset(side))
                if not placed.hits(rect) and not dots.hits(rect, ignore=item):
                    break
            else:
                if not item.is_selected:
                    continue
                side = LABEL_SIDES[0]  # 선택한 역 이름은 겹쳐도 표시
                rect = item.label.boundingRect().translated(centers[item] + item.label_offset(side))
            item.place_label(side)
            placed.add(rect, item)
            labeled.add(item)

        for item in self._stations.values():
            label = item.label
            shown = item in labeled
            if shown and not label.isVisible():
                label.setOpacity(0.0)  # 새로 나타난 이름은 0에서 시작해 서서히 진해진다
                self._fading.add(label)
            label.setVisible(shown)
        if self._fading and not self._fade_timer.isActive():
            self._fade_elapsed = 0
            self._fade_timer.start()

    def _on_label_fade(self) -> None:
        self._fade_elapsed += LABEL_FADE_STEP_MS
        progress = min(1.0, self._fade_elapsed / LABEL_FADE_MS)
        for label in list(self._fading):
            if not label.isVisible():
                self._fading.discard(label)
                continue
            label.setOpacity(progress)
            if progress >= 1.0:
                self._fading.discard(label)
        if not self._fading:
            self._fade_timer.stop()

    def _sync_size(self) -> None:
        """확대 배율에 맞춰 글씨·역 점·노선 두께를 바꾼다 (SIZE_STEP 단위로 바뀔 때만)."""
        size = (self.zoom() / SIZE_REFERENCE_ZOOM) ** SIZE_ZOOM_POWER
        size = round(min(max(size, SIZE_MIN), SIZE_MAX) / SIZE_STEP) * SIZE_STEP
        if size == self._size:
            return
        self._size = size
        for item, width in self._line_items:
            pen = item.pen()
            pen.setWidthF(width * size)
            item.setPen(pen)
        for station in self._stations.values():
            station.set_size(size)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self._update_labels()  # 드래그로 이동하면 화면에 들어온 역 이름을 다시 배치한다

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_overlays()
        self._update_labels()

    def _initial_view(self) -> tuple[float, QPointF]:
        """처음 화면의 (확대 배율, 중심): 2호선 일대를, 양옆 카드에 가리지 않는 영역 가운데에 맞춘다."""
        focus = QRectF(*self.layout.focus)
        dx, dy = focus.width() * INITIAL_VIEW_PADDING, focus.height() * INITIAL_VIEW_PADDING
        focus = focus.adjusted(-dx, -dy, dx, dy)

        view = self.viewport().size()
        left, right = self._overlay_insets()
        if left + right > view.width() / 2:  # 창이 좁으면 카드 위로 겹쳐도 지도를 넓게
            left = right = 0
        width = view.width() - left - right
        zoom = max(min(width / focus.width(), view.height() / focus.height()), INITIAL_MIN_ZOOM)
        # 보이는 영역의 가운데가 화면 가운데에서 (left - right) / 2 만큼 비켜 있다
        return zoom, focus.center() - QPointF((left - right) / 2 / zoom, 0)

    def _overlay_insets(self) -> tuple[int, int]:
        """양옆 모서리 카드가 가리는 너비 (왼쪽, 오른쪽)."""
        left = right = 0
        for widget, corner in self._overlays:
            width = widget.width() + 2 * OVERLAY_MARGIN
            if corner in (Qt.Corner.TopLeftCorner, Qt.Corner.BottomLeftCorner):
                left = max(left, width)
            elif widget.height() > 2 * OVERLAY_MARGIN + 30:  # 출처 표기처럼 낮은 띠는 제외
                right = max(right, width)
        return left, right

    def _set_view(self, zoom: float, center: QPointF) -> None:
        self.setTransform(QTransform.fromScale(zoom, zoom))
        self.centerOn(center)
        self._update_labels()

    def _view_center(self) -> QPointF:
        return self.mapToScene(self.viewport().rect().center())

    def show_initial_view(self) -> None:
        self._set_view(*self._initial_view())

    def animate_to_initial_view(self) -> None:
        self._animate_to(*self._initial_view())

    def animate_zoom(self, step: float) -> None:
        """화면 중심을 기준으로 부드럽게 확대·축소한다."""
        running = self._zoom_animation.state() == QVariantAnimation.State.Running
        zoom, center = self._animation_target if running else (self.zoom(), self._view_center())
        self._animate_to(zoom * step, center)  # 연속 클릭하면 진행 중인 목표에서 이어서 확대

    def _animate_to(self, zoom: float, center: QPointF) -> None:
        zoom = min(max(zoom, self.min_zoom()), MAX_ZOOM)
        self._zoom_animation.stop()
        self._animation_start = (self.zoom(), self._view_center())
        self._animation_target = (zoom, center)
        self._zoom_animation.setStartValue(0.0)
        self._zoom_animation.setEndValue(1.0)
        self._zoom_animation.start()

    def _on_zoom_animation(self, t: float) -> None:
        (z0, c0), (z1, c1) = self._animation_start, self._animation_target
        zoom = z0 * (z1 / z0) ** t  # 배율은 곱셈으로 보간해야 속도가 일정해 보인다
        self._set_view(zoom, c0 + (c1 - c0) * t)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_view_done:  # 뷰 크기가 정해진 뒤에 맞춘다
            self._initial_view_done = True
            self.show_initial_view()

    def wheelEvent(self, event):
        step = WHEEL_ZOOM_STEP if event.angleDelta().y() > 0 else 1 / WHEEL_ZOOM_STEP
        target = min(max(self.zoom() * step, self.min_zoom()), MAX_ZOOM)
        factor = target / self.zoom()
        self.scale(factor, factor)
        self._update_labels()


class _RectGrid:
    """화면 사각형 겹침 검사를 빠르게 하기 위한 격자 버킷."""

    CELL = 64

    def __init__(self):
        self._cells: dict[tuple[int, int], list[tuple[QRectF, object]]] = {}

    def _keys(self, rect: QRectF):
        c = self.CELL
        for x in range(int(rect.left() // c), int(rect.right() // c) + 1):
            for y in range(int(rect.top() // c), int(rect.bottom() // c) + 1):
                yield x, y

    def add(self, rect: QRectF, owner) -> None:
        for key in self._keys(rect):
            self._cells.setdefault(key, []).append((rect, owner))

    def hits(self, rect: QRectF, ignore=None) -> bool:
        return any(
            owner is not ignore and rect.intersects(other)
            for key in self._keys(rect)
            for other, owner in self._cells.get(key, ())
        )