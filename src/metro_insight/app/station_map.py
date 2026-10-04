"""1~8호선 노선도. 가로·세로·45° 선의 노선도 배치(schematic_layout)로 그리고, 역을 클릭해 선택한다."""
import math

from PySide6.QtCore import QEasingCurve, QEvent, QPointF, QRectF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QTransform
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from metro_insight.app.map_overlays import Attribution, LineLegend, ZoomControls
from metro_insight.app.schematic_layout import UNIT, SchematicLayout, metro_layout
from metro_insight.app.subway_data import Line, Station, load_network

# 처음 화면: 2호선 일대 범위에 이만큼 여백을 더해 보여준다 (범위 대비 비율)
INITIAL_VIEW_PADDING = 0.06
INITIAL_MIN_ZOOM = 0.85  # 창이 작아도 이보다 작게 시작하지 않는다 (글씨가 너무 작아지지 않게)
CORNER_RADIUS = 0.6 * UNIT  # 노선이 꺾이는 곳의 둥근 모서리 반지름

# 확대 배율(씬 1단위당 화면 픽셀)이 이 값 이상일 때 역 이름 표시 (환승역은 더 일찍)
TRANSFER_LABEL_MIN_ZOOM = 0.2
LABEL_MIN_ZOOM = 0.38
LABEL_SIDES = ("right", "left", "above", "below")  # 겹치면 다음 위치를 시도한다
MAX_ZOOM = 5.0
WHEEL_ZOOM_STEP = 1.2
BUTTON_ZOOM_STEP = 1.6
ZOOM_ANIMATION_MS = 220
OVERLAY_MARGIN = 14

# 씬 아이템에는 QSS가 적용되지 않아 색상을 여기서 지정한다 (styles/modern_gray.qss 팔레트와 맞춤)
STATION_FILL = "#ffffff"
TRANSFER_BORDER = "#1f2937"
SELECTED_FILL = "#1f2937"
SELECTED_HALO = QColor(31, 41, 55, 46)
LABEL_COLOR = "#374151"
SELECTED_LABEL_COLOR = "#111827"
LABEL_HALO = "#ffffff"
LINE_WIDTH = 4.5
LINE_CASING_EXTRA = 3  # 노선 아래 흰 테두리: 교차·인접 구간을 또렷하게
LABEL_POINT_SIZE = 8.5
SELECTED_LABEL_POINT_SIZE = 10

# 확대할수록 글씨·역 점·노선을 키운다: 크기 배율 = (확대 배율 / SIZE_REFERENCE_ZOOM)^SIZE_ZOOM_POWER
# 위 크기들은 SIZE_REFERENCE_ZOOM에서의 크기. 너무 작아지거나 커지지 않게 범위를 둔다
SIZE_REFERENCE_ZOOM = 1.0
SIZE_ZOOM_POWER = 0.6
SIZE_MIN, SIZE_MAX = 0.85, 1.9
SIZE_STEP = 0.05  # 크기 배율을 이 단위로 끊어, 확대 중 글자 모양을 매번 다시 만들지 않게 한다


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


class StationLabel(QGraphicsItem):
    """흰 테두리(halo)를 두른 역 이름. 노선 위에 겹쳐도 읽히게 한다."""

    def __init__(self, text: str, parent: QGraphicsItem, emphasized: bool):
        super().__init__(parent)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._text = text
        self._emphasized = emphasized  # 환승역은 조금 굵게
        self._selected = False
        self._size = 1.0
        self._build_path()

    def set_selected(self, selected: bool) -> None:
        self.prepareGeometryChange()
        self._selected = selected
        self._build_path()

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
        halo = QPen(QColor(LABEL_HALO), 3.5 * self._size)
        halo.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(self._path, halo)
        painter.fillPath(self._path, QColor(SELECTED_LABEL_COLOR if self._selected else LABEL_COLOR))


class StationItem(QGraphicsEllipseItem):
    """노선도 위의 역 한 개. 확대·축소해도 화면상 크기가 같다."""

    RADIUS = 3.5
    TRANSFER_RADIUS = 5.5
    HOVER_GROW = 1.5
    SELECTED_GROW = 2
    HALO = 7  # 선택한 역 둘레의 반투명 원

    def __init__(self, station: Station, pos: QPointF, line_color: str, on_click):
        super().__init__()
        self.station = station
        self._line_color = line_color
        self._on_click = on_click
        self._selected = False
        self._hovered = False
        self._size = 1.0
        self._label_side = LABEL_SIDES[0]

        self.setPos(pos)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{station.name}\n{' · '.join(f'{n}호선' for n in station.lines)}")
        self.setZValue(2 if station.is_transfer else 1)

        self.label = StationLabel(station.name, self, emphasized=station.is_transfer)
        self._refresh()

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
        border = TRANSFER_BORDER if self.station.is_transfer or self._selected else self._line_color
        self.setPen(QPen(QColor(border), (2.25 if self.station.is_transfer else 2) * self._size))
        self.setBrush(QBrush(QColor(SELECTED_FILL if self._selected else STATION_FILL)))
        self.place_label(self._label_side)

    def boundingRect(self) -> QRectF:
        h = self.HALO * self._size
        return super().boundingRect().adjusted(-h, -h, h, h)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if self._selected:
            r = self.radius() + self.HALO * self._size
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(SELECTED_HALO)
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

    def __init__(self):
        super().__init__()
        self.setObjectName("stationMap")
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # 드래그로 이동
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumSize(480, 480)

        self._stations: dict[str, StationItem] = {}
        self._line_items: list[tuple[QGraphicsPathItem, float]] = []  # (노선 경로, 기본 두께)
        self._size = None
        self._selected: StationItem | None = None
        self._initial_view_done = False
        self._overlays: list[tuple[QWidget, Qt.Corner]] = []
        self._zoom_animation = QVariantAnimation(self)
        self._zoom_animation.setDuration(ZOOM_ANIMATION_MS)
        self._zoom_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._zoom_animation.valueChanged.connect(self._on_zoom_animation)

        self.lines, stations = load_network()
        self.layout = metro_layout(self.lines, stations)
        self._draw(self.lines, stations, self.layout)

        controls = ZoomControls()
        controls.zoom_in.connect(lambda: self.animate_zoom(BUTTON_ZOOM_STEP))
        controls.zoom_out.connect(lambda: self.animate_zoom(1 / BUTTON_ZOOM_STEP))
        controls.reset.connect(self.animate_to_initial_view)
        self.add_overlay(LineLegend(self.lines), Qt.Corner.TopLeftCorner)
        self.add_overlay(controls, Qt.Corner.TopRightCorner)
        self.add_overlay(Attribution("© OpenStreetMap contributors"), Qt.Corner.BottomRightCorner)

    def _draw(self, lines: list[Line], stations: list[Station], layout: SchematicLayout) -> None:
        scene = self.scene()
        colors = {line.id: line.color for line in lines}
        pos = {name: QPointF(x, y) for name, (x, y) in layout.positions.items()}

        def line_pen(color: str, width: float) -> QPen:
            pen = QPen(QColor(color), width)
            pen.setCosmetic(True)  # 확대해도 선 두께 유지
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            return pen

        for line in lines:
            layers = ((-1, LABEL_HALO, LINE_WIDTH + LINE_CASING_EXTRA), (0, line.color, LINE_WIDTH))
            for z, color, width in layers:
                for a, b in line.edges:
                    points = layout.paths.get((line.id, a, b)) or layout.paths[(line.id, b, a)][::-1]
                    item = QGraphicsPathItem(rounded_path(points, CORNER_RADIUS))
                    item.setPen(line_pen(color, width))
                    item.setZValue(z)
                    scene.addItem(item)
                    self._line_items.append((item, width))

        for station in stations:
            item = StationItem(station, pos[station.name], colors[station.lines[0]], self.select_station)
            scene.addItem(item)
            self._stations[station.name] = item

        # 가장자리 역이 잘리지 않게 여백을 두고 이동 범위를 정한다
        margin = 2 * UNIT
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-margin, -margin, margin, margin))

    def select_station(self, station: Station) -> None:
        if self._selected is not None:
            self._selected.set_selected(False)
        self._selected = self._stations[station.name]
        self._selected.set_selected(True)
        self._update_labels()
        self.station_selected.emit(station)

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
            stacked[corner] = offset + size.height() + 10

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
            item.label.setVisible(item in labeled)

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
