"""화면 오른쪽에서 밀려 나오는 역 정보 패널: 역 검색, 예측 조건 입력, A~E 모델 예측 결과.

조건의 기본값은 지금이다. 날짜·시는 현재 시각, 요일·주말 여부는 날짜에서 계산하고,
날씨는 고른 날짜·시의 서울 날씨(기상청)를 백그라운드에서 받아 채운다. 날짜·시를 바꾸면 다시 받고,
미래 시각은 관측자료가 없으므로 받지 않는다. 모든 값은 직접 바꿀 수 있다.
공휴일 여부는 계산하지 않으므로 직접 체크한다.

"예측하기"를 누르면 아래에 A~E 모델의 예상 승차 인원과 Test 데이터 평가(MAE·RMSE)를 보여준다.
"""
import threading
from datetime import datetime, timedelta

from PySide6.QtCore import (
    QDate,
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from metro_insight.app.map_overlays import LineBadge
from metro_insight.app.subway_data import Line, Station, search_stations
from metro_insight.data.weather import KST, HourlyWeather, WeatherApiError, weather_at
from metro_insight.model.predictor import MODELS, PredictionInput, PredictionUnavailable, Predictor

PANEL_WIDTH = 340
SLIDE_MS = 240
SEARCH_MAX_ROWS = 7  # 검색 결과 목록이 이보다 많으면 스크롤
SEARCH_ROW_HEIGHT = 34
SEARCH_BADGE_SIZE = 16
SCROLLBAR_GUTTER = 8  # 폼 오른쪽에 스크롤바 자리를 비운다

WEEKDAY_NAMES = ("월", "화", "수", "목", "금", "토", "일")  # date.weekday() 순서
WEEKEND_FROM = 5  # 토요일부터 주말

# 학습 데이터의 시간대 (시작 시 → 표시). 업무일자는 04시에 시작하고, 04~06시는 한 칸,
# '24시'는 그 날짜 업무일의 마지막 시간대(다음 날 00~04시 4시간)다. 01~03시는 데이터가 없어 고르지 않는다
DAY_START_HOUR = 4
HOUR_SLOTS = {4: "04~06시", **{hour: f"{hour:02d}시" for hour in range(6, 25)}}


def hour_slot(hour: int) -> int:
    """시계의 시(0~23) → 그 시각이 속한 시간대의 시작 시. 데이터가 없는 01~03시는 직전 시간대(24시)로 본다."""
    if hour < DAY_START_HOUR:
        return 24
    return DAY_START_HOUR if hour < 6 else hour


# 입력하지 않은 값은 범위 최솟값에 두고 '미입력'으로 표시한다 (QAbstractSpinBox.specialValueText)
UNSET_TEXT = "미입력"
TEMPERATURE_MIN, TEMPERATURE_MAX = -40.0, 45.0

# 날짜·시를 바꾸고 이 시간 동안 더 바꾸지 않으면 날씨를 받는다 (화살표로 연달아 바꿀 때 요청이 쌓이지 않게)
WEATHER_DEBOUNCE_MS = 500


# ---------- 입력 위젯 ----------


class _WheelOnlyWhenFocused:
    """포커스가 없으면 휠을 부모(스크롤 영역)에 넘긴다. 패널을 스크롤하다 값이 바뀌지 않게 한다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class ComboBox(_WheelOnlyWhenFocused, QComboBox):
    pass


class DateEdit(_WheelOnlyWhenFocused, QDateEdit):
    pass


class DoubleSpinBox(_WheelOnlyWhenFocused, QDoubleSpinBox):
    pass


def color_dot(color: str, size: int = 10) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    icon = QIcon(pixmap)
    icon.addPixmap(pixmap, QIcon.Mode.Disabled)  # 호선이 하나라 비활성화한 칸에서도 호선 색을 보인다
    return icon


def styled_label(text: str, name: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName(name)
    return label


def set_flag(label: QLabel, name: str, on: bool) -> None:
    """QSS의 [name="true"] 스타일을 켜고 끈다."""
    label.setProperty(name, on)
    label.style().unpolish(label)
    label.style().polish(label)


def set_error(label: QLabel, error: bool) -> None:
    set_flag(label, "error", error)


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("resultDivider")
    line.setFixedHeight(1)
    return line


def format_people(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}명"


def format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:,.1f}"


# ---------- 날씨 ----------


class WeatherLoader(QObject):
    """서울 날씨를 백그라운드 스레드에서 받는다 (API 응답이 수십 초 걸리기도 한다)."""

    loaded = Signal(object)  # HourlyWeather
    failed = Signal(str)
    _done = Signal(int, object, str)  # (요청 번호, 날씨, 오류) — 작업 스레드 → UI 스레드

    def __init__(self, fetch=weather_at):
        super().__init__()
        self._fetch = fetch  # (시각) -> HourlyWeather
        self._request = 0
        self._done.connect(self._on_done)

    def load(self, tm: datetime) -> None:
        """tm 정시의 날씨를 받는다."""
        self._request += 1
        # 앱을 닫을 때 응답을 기다리지 않도록 daemon 스레드로 받는다
        threading.Thread(target=self._run, args=(self._request, tm), daemon=True).start()

    def cancel(self) -> None:
        """받는 중인 응답을 버린다."""
        self._request += 1

    def _run(self, request: int, tm: datetime) -> None:
        try:
            self._done.emit(request, self._fetch(tm), "")
        except WeatherApiError as e:
            self._done.emit(request, None, str(e))
        except Exception as e:  # 스레드에서 끝나면 화면이 '불러오는 중'에 멈추므로 모두 받는다
            self._done.emit(request, None, f"알 수 없는 오류 ({type(e).__name__})")

    def _on_done(self, request: int, weather: HourlyWeather | None, error: str) -> None:
        if request != self._request:  # 그사이 다시 요청했으면 이전 응답은 버린다
            return
        if weather is None:
            self.failed.emit(error)
        else:
            self.loaded.emit(weather)


# ---------- 패널 ----------


class StationPanel(QFrame):
    """역 검색 + 선택한 역의 예측 조건 입력 + 예측 결과."""

    station_chosen = Signal(object)  # 검색 결과에서 고른 Station
    close_requested = Signal()

    def __init__(
        self,
        lines: list[Line],
        stations: list[Station],
        weather: WeatherLoader | None = None,
        predictor: Predictor | None = None,
    ):
        super().__init__()
        self.setObjectName("stationPanel")
        self.setFixedWidth(PANEL_WIDTH)
        self._lines = {line.id: line for line in lines}
        self._stations = stations
        self._station: Station | None = None
        self._applying_weather = False
        self._weather_edited = False  # 날씨를 받는 동안 사용자가 고친 값은 덮어쓰지 않는다
        self._predictor = predictor  # 학습한 모델이 없으면 None

        self._weather = weather or WeatherLoader()
        self._weather.loaded.connect(self._on_weather_loaded)
        self._weather.failed.connect(self._on_weather_failed)
        self._weather_timer = QTimer(self)
        self._weather_timer.setSingleShot(True)
        self._weather_timer.setInterval(WEATHER_DEBOUNCE_MS)
        self._weather_timer.timeout.connect(self.refresh_weather)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18 - SCROLLBAR_GUTTER, 0)
        layout.setSpacing(12)
        top = QVBoxLayout()  # 폼 필드와 오른쪽 끝을 맞춘다
        top.setContentsMargins(0, 0, SCROLLBAR_GUTTER, 0)
        top.setSpacing(12)
        top.addLayout(self._build_header())
        top.addWidget(self._build_search())
        layout.addLayout(top)

        self._empty = styled_label("노선도에서 역을 클릭하거나\n위에서 역 이름을 검색하세요", "cardHint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty, 1)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("panelScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidget(self._build_form())
        self._scroll.hide()
        layout.addWidget(self._scroll, 1)

        self._results = self._build_results()  # 검색창 아래에 겹쳐 띄운다 (레이아웃 밖)
        QApplication.instance().focusChanged.connect(self._on_focus_changed)

        self.reset_to_now(load_weather=False)

    # ---------- 구성 ----------

    def _build_header(self) -> QHBoxLayout:
        close = QToolButton()
        close.setObjectName("panelClose")
        close.setIcon(QIcon("styles:close.svg"))
        close.setIconSize(QSize(14, 14))
        close.setToolTip("패널 닫기")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.clicked.connect(self.close_requested.emit)

        header = QHBoxLayout()
        header.addWidget(styled_label("역 정보", "panelTitle"))
        header.addStretch()
        header.addWidget(close)
        return header

    def _build_search(self) -> QLineEdit:
        self._search = QLineEdit()
        self._search.setObjectName("stationSearch")
        self._search.setPlaceholderText("역 이름 검색 (예: 강남)")
        self._search.setClearButtonEnabled(True)
        self._search.addAction(QIcon("styles:search.svg"), QLineEdit.ActionPosition.LeadingPosition)
        self._search.textChanged.connect(self._update_results)
        self._search.returnPressed.connect(self._choose_first_result)
        self._search.installEventFilter(self)  # ↓ 로 결과 목록, Esc 로 지우기
        return self._search

    def _build_results(self) -> QListWidget:
        results = QListWidget(self)
        results.setObjectName("searchResults")
        results.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        results.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        results.itemClicked.connect(self._choose_result)
        results.installEventFilter(self)  # Enter 로 선택
        results.hide()
        return results

    def _build_form(self) -> QWidget:
        form = QWidget()
        form.setObjectName("panelForm")
        layout = QVBoxLayout(form)
        layout.setContentsMargins(0, 4, SCROLLBAR_GUTTER, 18)
        layout.setSpacing(6)

        # 선택한 역
        self._name = styled_label("", "stationName")
        self._badges = QHBoxLayout()
        self._badges.setSpacing(4)
        self._line_names = styled_label("", "cardHint")
        badge_row = QHBoxLayout()
        badge_row.setSpacing(8)
        badge_row.addLayout(self._badges)
        badge_row.addWidget(self._line_names)
        badge_row.addStretch()
        layout.addWidget(styled_label("선택한 역", "cardCaption"))
        layout.addWidget(self._name)
        layout.addLayout(badge_row)

        # 일시
        reset = QPushButton("현재로")
        reset.setObjectName("textButton")
        reset.setToolTip("날짜·시간·날씨를 지금 값으로 되돌립니다")
        reset.setCursor(Qt.CursorShape.PointingHandCursor)
        # clicked 가 넘기는 checked(False)가 load_weather 로 들어가지 않게 인자 없이 부른다
        reset.clicked.connect(lambda: self.reset_to_now())

        self._line = ComboBox()
        self._date = DateEdit()
        self._date.setCalendarPopup(True)
        self._date.setDisplayFormat("yyyy-MM-dd")
        self._date.dateChanged.connect(self._apply_weekday)
        self._date.dateChanged.connect(self._schedule_weather)
        self._hour = ComboBox()
        for hour, label in HOUR_SLOTS.items():
            self._hour.addItem(label, hour)
        self._hour.setMaxVisibleItems(12)
        self._hour.currentIndexChanged.connect(self._schedule_weather)
        self._weekday = ComboBox()
        self._weekday.addItems([f"{name}요일" for name in WEEKDAY_NAMES])
        self._weekday.currentIndexChanged.connect(lambda i: self._weekend.setChecked(i >= WEEKEND_FROM))
        self._weekend = QCheckBox("주말")
        self._holiday = QCheckBox("공휴일")

        grid = self._section(layout, "조건", reset)
        self._add_row(grid, "호선", self._line)
        self._add_row(grid, "날짜", self._date)
        self._add_row(grid, "시간", self._hour)
        self._add_row(grid, "요일", self._weekday, self._weekend, self._holiday)

        # 날씨
        self._temperature = DoubleSpinBox()
        self._temperature.setDecimals(1)
        self._temperature.setSingleStep(0.5)
        self._temperature.setRange(TEMPERATURE_MIN - 0.1, TEMPERATURE_MAX)  # 최솟값 = 미입력
        self._temperature.setSpecialValueText(UNSET_TEXT)
        self._temperature.setSuffix(" ℃")
        self._temperature.setValue(self._temperature.minimum())
        self._rainfall = QCheckBox("강수")
        self._rainfall.setToolTip("비 또는 눈. 눈이 오면 함께 체크됩니다")
        self._snowfall = QCheckBox("눈")
        self._snowfall.toggled.connect(self._on_snowfall)
        self._weather_note = styled_label("", "fieldNote")
        self._weather_note.setWordWrap(True)
        for signal in (self._temperature.valueChanged, self._rainfall.toggled, self._snowfall.toggled):
            signal.connect(self._on_weather_edited)

        self._add_row(grid, "날씨", self._temperature, self._rainfall, self._snowfall)
        self._add_row(grid, "", self._weather_note)

        self._temperature.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._temperature.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # 예측
        predict = QPushButton("예측하기")
        predict.setObjectName("primaryButton")
        predict.setToolTip("A~E 모델로 위 조건의 예상 승차 인원을 계산합니다")
        predict.setCursor(Qt.CursorShape.PointingHandCursor)
        predict.clicked.connect(self.predict)
        layout.addSpacing(14)
        layout.addWidget(predict)
        layout.addSpacing(8)
        layout.addWidget(self._build_result())
        layout.addStretch()
        return form

    def _build_result(self) -> QFrame:
        """예측 결과 카드: 조건 / 모델별 예상 승차 인원 / 모델별 MAE·RMSE."""
        self._result = QFrame()
        self._result.setObjectName("resultCard")
        box = QVBoxLayout(self._result)
        box.setContentsMargins(14, 12, 14, 12)
        box.setSpacing(10)

        self._result_conditions = {}
        conditions = self._result_grid()
        for name in ("역", "날짜", "시간", "기온"):
            _, (self._result_conditions[name],) = self._result_line(conditions, name, "resultValue")
        box.addLayout(conditions)
        box.addWidget(divider())

        box.addWidget(styled_label("예상 승차 인원", "cardCaption"))
        self._result_boarding = {}
        boarding = self._result_grid()
        for model, features in MODELS.items():
            name, (value,) = self._result_line(boarding, f"{model} 모델", "resultNumber")
            name.setToolTip(features)
            self._result_boarding[model] = value
        box.addLayout(boarding)
        box.addWidget(divider())

        caption = styled_label("모델 평가", "cardCaption")
        caption.setToolTip("Test 데이터에서 실제 승차 인원과 비교한 오차 (단위: 명). 낮을수록 좋다")
        box.addWidget(caption)
        self._result_scores = {}
        scores = self._result_grid()
        for column, header in enumerate(("MAE", "RMSE"), start=1):
            label = styled_label(header, "fieldLabel")
            label.setAlignment(Qt.AlignmentFlag.AlignRight)
            scores.addWidget(label, 0, column)
        scores.setColumnStretch(2, 1)
        for model, features in MODELS.items():
            name, values = self._result_line(scores, model, "resultNumber", "resultNumber")
            name.setToolTip(features)
            self._result_scores[model] = (name, *values)
        box.addLayout(scores)

        self._result_note = styled_label("", "fieldNote")
        self._result_note.setWordWrap(True)
        box.addWidget(self._result_note)

        self._result.hide()
        return self._result

    @staticmethod
    def _result_grid() -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.setColumnMinimumWidth(0, 52)
        grid.setColumnStretch(1, 1)
        return grid

    @staticmethod
    def _result_line(grid: QGridLayout, label: str, *value_names: str) -> tuple[QLabel, list[QLabel]]:
        """이름 + 오른쪽 정렬 값 칸들. 값 칸은 비워 두고 예측할 때 채운다."""
        row = grid.rowCount()
        name = styled_label(label, "fieldLabel")
        grid.addWidget(name, row, 0)
        values = []
        for column, value_name in enumerate(value_names, start=1):
            value = styled_label("", value_name)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(value, row, column)
            values.append(value)
        return name, values

    @staticmethod
    def _section(layout: QVBoxLayout, caption: str, action: QWidget | None = None) -> QGridLayout:
        title = QHBoxLayout()
        title.setContentsMargins(0, 14, 0, 2)
        title.addWidget(styled_label(caption, "cardCaption"))
        title.addStretch()
        if action is not None:
            title.addWidget(action)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 52)
        grid.setColumnStretch(1, 1)
        layout.addLayout(title)
        layout.addLayout(grid)
        return grid

    @staticmethod
    def _add_row(grid: QGridLayout, label: str, field: QWidget, *extras: QWidget) -> None:
        row = grid.rowCount()
        grid.addWidget(styled_label(label, "fieldLabel"), row, 0)
        if not extras:
            grid.addWidget(field, row, 1)
            return
        box = QHBoxLayout()
        box.setSpacing(12)
        box.addWidget(field, 1)
        for extra in extras:
            box.addWidget(extra)
        grid.addLayout(box, row, 1)

    # ---------- 역 ----------

    @property
    def station(self) -> Station | None:
        return self._station

    def set_station(self, station: Station) -> None:
        """선택한 역을 보여준다. 일시·날씨 조건은 역을 바꿔도 유지한다."""
        self._station = station
        self._name.setText(station.name)
        while self._badges.count():
            self._badges.takeAt(0).widget().deleteLater()
        for line_id in station.lines:
            self._badges.addWidget(LineBadge(self._lines[line_id]))
        self._line_names.setText(" · ".join(self._lines[i].name for i in station.lines))

        self._line.clear()
        for line_id in station.lines:
            line = self._lines[line_id]
            self._line.addItem(color_dot(line.color), line.name, line_id)
        self._line.setEnabled(len(station.lines) > 1)

        self._empty.hide()
        self._scroll.show()

    def focus_search(self) -> None:
        self._search.setFocus()
        self._search.selectAll()

    # ---------- 입력값 ----------

    def prediction_input(self) -> PredictionInput | None:
        """지금 입력된 예측 조건. 역을 고르기 전에는 None."""
        if self._station is None:
            return None
        temperature = self._temperature.value()
        return PredictionInput(
            date=self._date.date().toString("yyyy-MM-dd"),
            line=self._line.currentText(),
            station=self._station.name,
            hour=self._hour.currentData(),
            weekday=self._weekday.currentIndex(),
            is_weekend=self._weekend.isChecked(),
            is_holiday=self._holiday.isChecked(),
            temperature=None if temperature == self._temperature.minimum() else round(temperature, 1),
            rainfall=self._rainfall.isChecked(),
            snowfall=self._snowfall.isChecked(),
        )

    def selected_time(self) -> datetime:
        """고른 날짜·시간대의 시작 시각 (KST 정시). 24시는 다음 날 00시다."""
        date = self._date.date().toPython()
        return datetime(date.year, date.month, date.day, tzinfo=KST) + timedelta(hours=self._hour.currentData())

    def reset_to_now(self, load_weather: bool = True) -> None:
        """날짜·시는 현재 시각으로, 요일·주말은 그 날짜로, 날씨는 서울 현재 날씨로. 공휴일은 그대로 둔다.

        00~04시는 전날 업무일에 속하므로 날짜를 하루 앞으로 두고 24시를 고른다.
        """
        now = datetime.now(KST)
        date = (now - timedelta(hours=DAY_START_HOUR)).date()
        self._date.setDate(QDate(date.year, date.month, date.day))
        self._apply_weekday()  # 날짜가 그대로면 dateChanged 가 오지 않으므로 직접 다시 계산
        self._hour.setCurrentIndex(self._hour.findData(hour_slot(now.hour)))
        self._weather_timer.stop()  # 날짜·시를 바꾸며 잡힌 조회는 기다리지 않는다
        if load_weather:
            self.refresh_weather()

    def set_date_range(self, start: str, end: str) -> None:
        """예측 가능한 날짜 범위를 달력에 반영한다. 범위 밖 날짜는 고를 수 없다."""
        first = QDate.fromString(start, "yyyy-MM-dd")
        last = QDate.fromString(end, "yyyy-MM-dd")
        if first.isValid() and last.isValid() and first <= last:
            self._date.setMinimumDate(first)
            self._date.setMaximumDate(last)

    def _apply_weekday(self) -> None:
        weekday = self._date.date().toPython().weekday()
        self._weekday.setCurrentIndex(weekday)
        self._weekend.setChecked(weekday >= WEEKEND_FROM)  # 요일이 그대로면 시그널이 오지 않으므로 직접

    def _on_snowfall(self, snowing: bool) -> None:
        if snowing:
            self._rainfall.setChecked(True)  # 눈도 강수
        self._rainfall.setEnabled(not snowing)

    # ---------- 예측 ----------

    def predict(self) -> None:
        """지금 조건으로 A~E 모델을 돌려 결과 카드에 보여준다."""
        values = self.prediction_input()
        if values is None:
            return
        predictions, scores, error = {}, {}, ""
        if self._predictor is None:
            error = "학습된 모델이 아직 없어 예상 이용량을 계산할 수 없습니다."
        else:
            try:
                scores = self._predictor.scores()
                predictions = self._predictor.predict(values)
            except PredictionUnavailable as e:
                predictions, error = {}, str(e)
            except Exception as e:  # 모델 오류로 화면이 멈추지 않게 보여 주고 넘어간다
                predictions, scores = {}, {}
                error = f"예측하지 못했습니다. ({type(e).__name__}: {e})"
        self.show_result(values, predictions, scores, error)

    def show_result(self, values: PredictionInput, predictions: dict, scores: dict, error: str = "") -> None:
        holiday = " · 공휴일" if values.is_holiday else ""
        conditions = {
            "역": f"{values.station} ({values.line})",
            "날짜": f"{values.date} ({WEEKDAY_NAMES[values.weekday]}){holiday}",
            "시간": HOUR_SLOTS[values.hour],
            "기온": UNSET_TEXT if values.temperature is None else f"{values.temperature:.1f}℃",
        }
        for name, text in conditions.items():
            self._result_conditions[name].setText(text)

        for model, label in self._result_boarding.items():
            label.setText(format_people(predictions.get(model)))

        # MAE 가 가장 낮은 모델을 굵게 표시한다
        best = min(scores, key=lambda model: scores[model].mae, default=None)
        for model, (name, mae, rmse) in self._result_scores.items():
            score = scores.get(model)
            mae.setText(format_score(score and score.mae))
            rmse.setText(format_score(score and score.rmse))
            for label in (name, mae, rmse):
                set_flag(label, "best", model == best)

        self._result_note.setText(error)
        set_error(self._result_note, bool(error))
        self._result_note.setVisible(bool(error))
        self._result.show()
        # 카드가 레이아웃에 자리 잡은 뒤 보이도록 스크롤한다
        QTimer.singleShot(0, lambda: self._scroll.ensureWidgetVisible(self._result))

    # ---------- 날씨 ----------

    def refresh_weather(self) -> None:
        """고른 날짜·시의 서울 날씨를 받는다. 받으면 직접 고친 날씨도 덮어쓴다."""
        self._weather_timer.stop()
        self._weather_edited = False
        set_error(self._weather_note, False)
        tm = self.selected_time()
        if tm > datetime.now(KST):
            self._weather.cancel()
            self._weather_note.setText("미래 시각이라 관측된 날씨가 없습니다. 예상 날씨를 직접 입력하세요.")
            return
        self._weather_note.setText(f"서울 {tm:%Y-%m-%d %H}시 날씨를 불러오는 중…")
        self._weather.load(tm)

    def _schedule_weather(self) -> None:
        """날짜·시가 바뀌면 이전 시각의 응답은 버리고, 잠시 뒤 새 시각의 날씨를 받는다."""
        self._weather.cancel()
        self._weather_timer.start()

    def apply_weather(self, weather: HourlyWeather) -> None:
        self._applying_weather = True
        try:
            temperature = weather.temperature
            self._temperature.setValue(self._temperature.minimum() if temperature is None else temperature)
            self._snowfall.setChecked(bool(weather.snowfall))
            self._rainfall.setChecked(bool(weather.rainfall or weather.snowfall))
        finally:
            self._applying_weather = False

    def _on_weather_edited(self) -> None:
        if not self._applying_weather:
            self._weather_edited = True

    def _on_weather_loaded(self, weather: HourlyWeather) -> None:
        source = f"기상청 서울 {weather.date} {weather.hour:02d}시 관측"
        selected = self.selected_time()
        if (weather.date, weather.hour) != (f"{selected:%Y-%m-%d}", selected.hour):
            source += " (고른 시각의 자료가 없어 직전 관측값)"
        if self._weather_edited:
            temperature = "-" if weather.temperature is None else f"{weather.temperature:.1f}℃"
            self._weather_note.setText(f"{source} {temperature} · 직접 입력한 값을 유지합니다")
            return
        self.apply_weather(weather)
        self._weather_note.setText(source)

    def _on_weather_failed(self, error: str) -> None:
        self._weather_note.setText(f"날씨를 불러오지 못했습니다. 직접 입력하세요.\n{error}")
        set_error(self._weather_note, True)

    # ---------- 검색 ----------

    def _update_results(self, query: str) -> None:
        self._results.clear()
        if not query.strip():
            self._results.hide()
            return
        matches = search_stations(self._stations, query)
        for station in matches:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, station)
            item.setSizeHint(QSize(0, SEARCH_ROW_HEIGHT))
            self._results.addItem(item)
            self._results.setItemWidget(item, self._result_row(station))
        if not matches:
            item = QListWidgetItem(f"'{query.strip()}'(으)로 시작하는 역이 없습니다")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            item.setSizeHint(QSize(0, SEARCH_ROW_HEIGHT))
            self._results.addItem(item)
        self._place_results()
        self._results.setVisible(self._search.hasFocus())
        self._results.raise_()

    def _result_row(self, station: Station) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(4)
        layout.addWidget(styled_label(station.name, "resultName"))
        layout.addStretch()
        for line_id in station.lines:
            layout.addWidget(LineBadge(self._lines[line_id], SEARCH_BADGE_SIZE))
        return row

    def _place_results(self) -> None:
        frame = 2 * self._results.frameWidth() + 8  # 테두리 + QSS 안쪽 여백
        rows = min(max(self._results.count(), 1), SEARCH_MAX_ROWS)
        top_left = self._search.mapTo(self, QPoint(0, self._search.height() + 4))
        height = rows * SEARCH_ROW_HEIGHT + frame
        self._results.setGeometry(top_left.x(), top_left.y(), self._search.width(), height)

    def _choose_result(self, item: QListWidgetItem) -> None:
        station = item.data(Qt.ItemDataRole.UserRole)
        if station is None:  # '검색 결과 없음' 안내
            return
        self.station_chosen.emit(station)
        self._search.setFocus()  # 이어서 다른 역을 검색할 수 있게
        # 클릭 시그널 처리 중에 목록 항목을 지우지 않도록 다음 이벤트 루프에서 지운다
        QTimer.singleShot(0, self._search.clear)

    def _choose_first_result(self) -> None:
        if self._results.isVisible() and self._results.count():
            self._choose_result(self._results.item(0))

    def _on_focus_changed(self, old: QWidget | None, new: QWidget | None) -> None:
        """검색창·결과 목록 밖을 누르면 결과 목록을 닫고, 검색창으로 돌아오면 다시 연다."""
        searching = new is self._search or new is self._results
        self._results.setVisible(searching and bool(self._search.text().strip()))

    def _query_with_preedit(self, preedit: str) -> str:
        text, cursor = self._search.text(), self._search.cursorPosition()
        return text[:cursor] + preedit + text[cursor:]

    def eventFilter(self, watched, event) -> bool:
        if watched is self._search and event.type() == QEvent.Type.InputMethod:
            # 한글은 조합 중인 글자('강')가 text() 에 아직 없어 textChanged 가 오지 않는다.
            # 조합 중인 글자까지 넣어 검색하도록, 입력기 이벤트가 처리된 뒤 다시 검색한다
            preedit = event.preeditString()
            QTimer.singleShot(0, lambda: self._update_results(self._query_with_preedit(preedit)))
            return super().eventFilter(watched, event)
        if event.type() != QEvent.Type.KeyPress:
            return super().eventFilter(watched, event)
        key = event.key()
        if watched is self._search:
            if key == Qt.Key.Key_Down and self._results.isVisible():
                self._results.setFocus()
                self._results.setCurrentRow(0)
                return True
            if key == Qt.Key.Key_Escape and self._search.text():
                self._search.clear()
                return True
        elif watched is self._results:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and (item := self._results.currentItem()):
                self._choose_result(item)
                return True
            if key == Qt.Key.Key_Up and self._results.currentRow() == 0:
                self._search.setFocus()
                return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._results.isVisible():
            self._place_results()


# ---------- 슬라이드 ----------


class SlidePanel(QWidget):
    """패널을 오른쪽에서 밀어 넣고 빼는 컨테이너.

    컨테이너 너비를 0 ↔ (간격 + 패널 너비)로 바꾸고 패널은 오른쪽 끝에 붙여 두므로,
    패널 내용이 찌그러지지 않고 오른쪽에서 밀려 나오는 것처럼 보인다.
    """

    open_changed = Signal(bool)  # 열기·닫기 시작
    opened = Signal()  # 다 열림

    def __init__(self, panel: QWidget, gap: int):
        super().__init__()
        self.panel = panel
        panel.setParent(self)
        panel.hide()
        self._open = False
        self._open_width = gap + panel.width()
        self.setFixedWidth(0)

        self._animation = QVariantAnimation(self)
        self._animation.setDuration(SLIDE_MS)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(lambda width: self.setFixedWidth(int(width)))
        self._animation.finished.connect(self._on_finished)

    def is_open(self) -> bool:
        return self._open

    def set_open(self, open_: bool) -> None:
        if open_ == self._open:
            return
        self._open = open_
        if open_:
            self.panel.show()
        self._animation.stop()
        self._animation.setStartValue(self.width())
        self._animation.setEndValue(self._open_width if open_ else 0)
        self._animation.start()
        self.open_changed.emit(open_)

    def toggle(self) -> None:
        self.set_open(not self._open)

    def _on_finished(self) -> None:
        if self._open:
            self.opened.emit()
        else:
            self.panel.hide()  # 가려진 패널에 키보드 포커스가 가지 않게

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.panel.setGeometry(self.width() - self.panel.width(), 0, self.panel.width(), self.height())
