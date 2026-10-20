"""메인 윈도우: 상단 헤더 + 1~8호선 노선도 + 오른쪽 역 정보 패널."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from metro_insight.app.map_overlays import SelectionCard
from metro_insight.app.station_map import StationMapView
from metro_insight.app.station_panel import PANEL_WIDTH, SlidePanel, StationPanel
from metro_insight.app.subway_data import Station
from metro_insight.model.predictor import ABCPredictor

BODY_MARGIN = 16


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metro Insight - 지하철 이용량 예측")
        self.setMinimumSize(800, 600)

        self.map_view = StationMapView()
        self.map_view.station_selected.connect(self.on_station_selected)
        self.selection_card = SelectionCard(self.map_view.lines)
        self.map_view.add_overlay(self.selection_card, Qt.Corner.BottomLeftCorner)

        predictor = ABCPredictor()  # models/cal_*.joblib + cal_profiles.npz + boarding_history.npz
        self.station_panel = StationPanel(
            self.map_view.lines, self.map_view.stations, predictor=predictor
        )
        if date_range := predictor.date_range():
            self.station_panel.set_date_range(*date_range)
        self.station_panel.station_chosen.connect(self.map_view.focus_station)
        self.slide = SlidePanel(self.station_panel, gap=BODY_MARGIN)
        self.station_panel.close_requested.connect(lambda: self.slide.set_open(False))
        self.slide.opened.connect(self.map_view.reveal_selected)

        body = QHBoxLayout()
        body.setContentsMargins(BODY_MARGIN, BODY_MARGIN, BODY_MARGIN, BODY_MARGIN)
        body.setSpacing(0)  # 패널과의 간격은 SlidePanel 이 함께 열고 닫는다
        body.addWidget(self.map_view, 1)
        body.addWidget(self.slide)
        # 패널을 열어도 노선도가 최소 크기보다 작아지지 않게
        min_width = self.map_view.minimumWidth() + BODY_MARGIN * 3 + PANEL_WIDTH
        self.setMinimumWidth(max(self.minimumWidth(), min_width))

        root = QWidget()
        root.setObjectName("centralWidget")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())
        layout.addLayout(body, 1)
        self.setCentralWidget(root)

        QShortcut(QKeySequence.StandardKey.Find, self, self.open_search)
        self.station_panel.refresh_weather()

    def _build_header(self) -> QWidget:
        title = QLabel("Metro Insight")
        title.setObjectName("appTitle")
        subtitle = QLabel("지하철 이용량 예측")
        subtitle.setObjectName("appSubtitle")
        station_count = len({s for line in self.map_view.lines for edge in line.edges for s in edge})
        meta = QLabel(f"1~8호선 · 역 {station_count}개")
        meta.setObjectName("headerMeta")

        toggle = QPushButton("역 정보")
        toggle.setObjectName("panelToggle")
        toggle.setCheckable(True)
        toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        toggle.setToolTip("역 정보 패널 열기·닫기 (역 검색: Ctrl+F)")
        toggle.clicked.connect(self.slide.set_open)
        self.slide.open_changed.connect(toggle.setChecked)

        header = QWidget()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()
        layout.addWidget(meta)
        layout.addWidget(toggle)
        return header

    def open_search(self) -> None:
        self.slide.set_open(True)
        self.station_panel.focus_search()

    def on_station_selected(self, station: Station) -> None:
        self.selection_card.show_station(station)
        self.station_panel.set_station(station)
        self.slide.set_open(True)
