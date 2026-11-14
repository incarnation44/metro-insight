"""메인 윈도우: 상단 헤더 + 1~8호선 노선도 + 오른쪽 역 정보 패널."""
import threading
from datetime import timedelta

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from metro_insight.app.map_overlays import SelectionCard
from metro_insight.app.station_map import StationMapView
from metro_insight.app.station_panel import (
    PANEL_WIDTH,
    SlidePanel,
    StationPanel,
    kr_holidays,
    set_flag,
)
from metro_insight.app.subway_data import Station

BODY_MARGIN = 16

# 예측기는 모델 파일·프로파일·승차 이력을 읽느라 만드는 데 몇 초가 걸린다. 그동안 창이 뜨지 않으면
# 앱이 멈춘 것처럼 보이므로, 창을 먼저 띄우고 예측기는 배경에서 만든다.
LOADING_STATUS = "모델을 불러오는 중…"
LOADING_TIP = "models/ 의 모델 파일과 승차 이력을 읽고 있습니다."
FAILED_STATUS = "모델을 불러오지 못했습니다"
FAILED_TIP = "모델 파일을 확인한 뒤 눌러서 다시 시도합니다.\n{}"

# 워밍업 날짜. 데이터 안(과거) 날짜는 그 시점까지의 이력으로 프로파일을 다시 계산하고 D·E
# 시계열 모델까지 계산하며, 데이터 끝 이후 날짜는 서빙용 프로파일을 쓴다. 두 갈래를 미리
# 한 번씩 계산해 두면 사용자가 처음 예측할 때 그 비용을 치르지 않는다.
WARMUP_DAYS_BEFORE = 7
WARMUP_DAYS_AFTER = 7
WARMUP_HOUR = 8


class PredictorLoader(QObject):
    """예측기를 배경 스레드에서 만들어 워밍업한 뒤 화면에 넘긴다.

    예측기 생성은 수 초 동안 파이썬 코드와 파일 읽기를 함께 쓰므로 UI 스레드에서 만들면
    그동안 창이 그려지지 않는다. 신호는 UI 스레드로 전달되므로 화면은 결과만 받는다.
    """

    ready = Signal(object)  # 준비된 예측기
    failed = Signal(str)  # 화면에 보여 줄 실패 사유

    def __init__(self, factory=None):
        super().__init__()
        self._factory = factory  # 시험용으로 다른 예측기를 넣을 수 있다
        self._thread: threading.Thread | None = None

    def start(self, lines=(), stations=()) -> None:
        """배경에서 읽기를 시작한다. 이미 읽는 중이면 아무것도 하지 않는다."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, args=(tuple(lines), tuple(stations)), daemon=True
        )
        self._thread.start()

    def _run(self, lines: tuple, stations: tuple) -> None:
        try:
            factory = self._factory
            if factory is None:
                # 모델 쪽 모듈은 pandas·joblib 를 함께 읽어 무거우므로 읽기 스레드에서 처음 읽는다
                from metro_insight.model.predictor import ABCPredictor  # noqa: PLC0415

                factory = ABCPredictor
            predictor = factory()
        except Exception as e:  # 모델 파일이 없거나 깨져도 창은 뜨고 이유를 알린다
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        try:
            warm_up(predictor, lines, stations)
        except Exception:  # 워밍업이 실패해도 예측 자체는 할 수 있다 (첫 예측이 느려질 뿐)
            pass
        self.ready.emit(predictor)


def warmup_targets(predictor, lines, stations) -> list[tuple[str, str]]:
    """워밍업에 쓸 (호선, 역) 하나를 고른다. 학습 데이터가 있는 조합만 쓴다."""
    names = {line.id: line.name for line in lines}
    has_data = getattr(predictor, "has_data", None)
    for station in stations:
        for line_id in station.lines:
            name = names.get(line_id)
            if name is None:
                continue
            if has_data is None or bool(has_data(name, station.name)):
                return [(name, station.name)]
    return []


def warm_up(predictor, lines, stations) -> None:
    """대표 조건으로 한 번씩 계산해 두어 첫 예측이 느려지지 않게 한다."""
    targets = warmup_targets(predictor, lines, stations)
    if not targets:
        return
    from metro_insight.model.predictor import DATA_END, PredictionInput  # noqa: PLC0415

    for offset in (-WARMUP_DAYS_BEFORE, WARMUP_DAYS_AFTER):
        day = DATA_END + timedelta(days=offset)
        weekday = day.weekday()
        for line, station in targets:
            try:
                predictor.predict(
                    PredictionInput(
                        date=day.isoformat(),
                        line=line,
                        station=station,
                        hour=WARMUP_HOUR,
                        weekday=weekday,
                        is_weekend=weekday >= 5,
                        is_holiday=day in kr_holidays(day.year),
                        temperature=None,
                        rainfall=False,
                        snowfall=False,
                    )
                )
            except Exception:  # 조건을 만족하지 못하는 모델이 있어도 나머지는 데운다
                continue


class MainWindow(QMainWindow):
    def __init__(self, predictor=None, loader: PredictorLoader | None = None):
        super().__init__()
        self.setWindowTitle("Metro Insight - 지하철 이용량 예측")
        self.setMinimumSize(800, 600)

        self.map_view = StationMapView()
        self.map_view.station_selected.connect(self.on_station_selected)
        self.selection_card = SelectionCard(self.map_view.lines)
        self.map_view.add_overlay(self.selection_card, Qt.Corner.BottomLeftCorner)

        # predictor 를 넘기지 않으면 창을 띄운 뒤 배경에서 읽는다 (날짜 범위·호선 표시는 그때 맞춘다)
        self._predictor = predictor
        self.station_panel = StationPanel(
            self.map_view.lines, self.map_view.stations, predictor=predictor
        )
        self.station_panel.station_chosen.connect(self.map_view.focus_station)
        self.station_panel.predictor_retry_requested.connect(self.reload_predictor)
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

        if predictor is None:
            self._loader = loader or PredictorLoader()
            self._loader.ready.connect(self._on_predictor_ready)
            self._loader.failed.connect(self._on_predictor_failed)
            self.station_panel.begin_predictor_load()
            self._show_status(LOADING_STATUS, tip=LOADING_TIP)
            # 첫 화면을 그린 뒤에 읽기 시작한다 (화면을 그리는 동안 CPU 를 뺏지 않게)
            QTimer.singleShot(0, self.load_predictor)
        else:
            self._loader = loader
            self.station_panel.set_predictor(predictor)
            self._show_status("")

    def _build_header(self) -> QWidget:
        title = QLabel("Metro Insight")
        title.setObjectName("appTitle")
        subtitle = QLabel("지하철 이용량 예측")
        subtitle.setObjectName("appSubtitle")
        station_count = len({s for line in self.map_view.lines for edge in line.edges for s in edge})
        meta = QLabel(f"1~8호선 · 역 {station_count}개")
        meta.setObjectName("headerMeta")

        # 모델을 읽는 중·읽지 못했을 때만 보이는 상태 표시. 실패했을 때는 눌러서 다시 읽는다.
        self._status_chip = QPushButton("")
        self._status_chip.setObjectName("statusChip")
        self._status_chip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._status_chip.setVisible(False)
        self._status_chip.clicked.connect(self.reload_predictor)

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
        layout.addWidget(self._status_chip)
        layout.addWidget(meta)
        layout.addWidget(toggle)
        return header

    # ---------- 예측기 ----------

    @property
    def predictor(self):
        """지금 화면이 쓰는 예측기. 아직 읽는 중이면 None."""
        return self._predictor

    def load_predictor(self) -> None:
        """배경에서 예측기 읽기를 시작한다."""
        if self._loader is not None:
            self._loader.start(self.map_view.lines, self.map_view.stations)

    def reload_predictor(self) -> None:
        """읽지 못한 예측기를 다시 읽는다 (상태 표시·패널의 '다시 시도')."""
        if self._loader is None or self.station_panel.predictor_state == "loading":
            return
        self.station_panel.begin_predictor_load()
        self._show_status(LOADING_STATUS, tip=LOADING_TIP)
        self.load_predictor()

    def _on_predictor_ready(self, predictor) -> None:
        self._predictor = predictor
        self.station_panel.set_predictor(predictor)
        self._show_status("")

    def _on_predictor_failed(self, reason: str) -> None:
        self.station_panel.set_predictor_error(reason)
        self._show_status(FAILED_STATUS, error=True, tip=FAILED_TIP.format(reason))

    def _show_status(self, text: str, error: bool = False, tip: str = "") -> None:
        self._status_chip.setText(text)
        self._status_chip.setToolTip(tip)
        set_flag(self._status_chip, "error", error)
        self._status_chip.setVisible(bool(text))

    # ---------- 화면 ----------

    def open_search(self) -> None:
        self.slide.set_open(True)
        self.station_panel.focus_search()

    def on_station_selected(self, station: Station) -> None:
        self.selection_card.show_station(station)
        self.station_panel.set_station(station)
        self.slide.set_open(True)
