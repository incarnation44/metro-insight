"""메인 윈도우: 상단 헤더 + 1~8호선 노선도."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from metro_insight.app.map_overlays import SelectionCard
from metro_insight.app.station_map import StationMapView
from metro_insight.app.subway_data import Station


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metro Insight - 지하철 이용량 예측")
        self.setMinimumSize(800, 600)

        self.map_view = StationMapView()
        self.map_view.station_selected.connect(self.on_station_selected)
        self.selection_card = SelectionCard(self.map_view.lines)
        self.map_view.add_overlay(self.selection_card, Qt.Corner.BottomLeftCorner)

        body = QVBoxLayout()
        body.setContentsMargins(16, 16, 16, 16)
        body.addWidget(self.map_view)

        root = QWidget()
        root.setObjectName("centralWidget")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())
        layout.addLayout(body)
        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        title = QLabel("Metro Insight")
        title.setObjectName("appTitle")
        subtitle = QLabel("지하철 이용량 예측")
        subtitle.setObjectName("appSubtitle")
        station_count = len({s for line in self.map_view.lines for edge in line.edges for s in edge})
        meta = QLabel(f"1~8호선 · 역 {station_count}개")
        meta.setObjectName("headerMeta")

        header = QWidget()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()
        layout.addWidget(meta)
        return header

    def on_station_selected(self, station: Station) -> None:
        self.selection_card.show_station(station)
