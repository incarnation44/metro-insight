"""화면 오른쪽에서 밀려 나오는 역 정보 패널: 역 검색, 예측 조건 입력, A~E 모델 예측 결과.

조건의 기본값은 지금이다. 날짜·시는 현재 시각, 요일·주말 여부는 날짜에서 계산하고,
날씨는 고른 날짜·시의 서울 날씨(기상청)를 백그라운드에서 받아 채운다. 날짜·시를 바꾸면 다시 받고,
미래 시각은 관측자료가 없으므로 받지 않는다(기온은 평년값, 비·눈은 '안 옴'으로 계산).
기온과 비·눈은 직접 바꿀 수 있고, 직접 고친 값은 날짜를 바꿔도 덮어쓰지 않는다.
공휴일 여부는 고른 날짜로 자동 판정한다(학습 때 쓴 공휴일 목록과 같은 것).

"예측하기"를 누르면 아래에 A~E 모델의 예상 승차 인원을 항상 보여주고,
Test 데이터 평가(MAE·RMSE)는 접었다 펼 수 있는 칸으로 보여준다.
"""
import threading
from datetime import date, datetime, timedelta

from holidays import country_holidays
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
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from metro_insight.app.map_overlays import LineBadge
from metro_insight.app.subway_data import Line, Station, search_stations
from metro_insight.data.weather import KST, HourlyWeather, WeatherApiError, slot_weather
from metro_insight.model.predictor import (
    DATA_END,
    MODELS,
    VERIFIED_HORIZON_BUCKETS,
    PredictionInput,
    PredictionUnavailable,
    Predictor,
    climatology_temperature,
)

PANEL_WIDTH = 340
SLIDE_MS = 240
SEARCH_MAX_ROWS = 7  # 검색 결과 목록이 이보다 많으면 스크롤
SEARCH_ROW_HEIGHT = 34
SEARCH_BADGE_SIZE = 16
SCROLLBAR_GUTTER = 8  # 폼 오른쪽에 스크롤바 자리를 비운다

WEEKDAY_NAMES = ("월", "화", "수", "목", "금", "토", "일")  # date.weekday() 순서
WEEKEND_FROM = 5  # 토요일부터 주말

# 데이터 끝 182일 이후 구간(예: 2026-12-29~)에는 Test 성적이 없다.
UNVERIFIED_NOTE = "※ A~C 장기 추정"
UNVERIFIED_TOOLTIP = (
    "Test 로는 검증하지 않았습니다. 과거 3개 시점에서 다시 학습해 182~366일 앞을 채점한 "
    "백테스트 결과만 있습니다 (MAE A 108.6 · B 87.2 · C 85.8, ±MAE 안에 든 비율 72~78%)."
)

# Test 구간이 136일(2026-02-15~06-30)이라 '92-181' 구간의 근거도 136일까지만 있다.
EVAL_SPAN_DAYS = 136
PARTIAL_NOTE = "※ A~C 정확도 일부 미확인"
PARTIAL_TOOLTIP = (
    "Test 구간이 {span}일이라 {span}일을 넘는 구간은 실제 오차를 잰 적이 없습니다. "
    "표시된 숫자는 {span}일까지의 성적으로 추정한 값입니다."
)

# 성적표 — A~C 는 프로파일을 동결해 잰 장기 예측, D·E 는 전날까지 실측을 넣은 하루 앞 예측.
SEQUENCE_MODELS = ("D", "E")

# D·E 는 대상일 전날까지 14일 실측이 있어야 계산한다 — 데이터 끝 이후는 이유를 남긴다.
NO_HISTORY_TEXT = "실측 불가"
NO_HISTORY_TIP = (
    "D·E 는 대상일 전날까지 14일 실측이 필요합니다. 데이터가 {end} 에서 끝나 "
    "이 날짜는 계산하지 않습니다 (A~C 는 달력 패턴으로 계산)."
)

# 학습 데이터의 시간대 (시작 시 → 표시). 업무일자는 04시에 시작하고, 04~06시는 한 칸,
# '24시'는 그 날짜 업무일의 마지막 시간대(다음 날 00~04시 4시간)다. 01~03시는 데이터가 없어 고르지 않는다
DAY_START_HOUR = 4
HOUR_SLOTS = {4: "04~06시", **{hour: f"{hour:02d}시" for hour in range(6, 25)}}


def hour_slot(hour: int) -> int:
    """시계의 시(0~23) → 그 시각이 속한 시간대의 시작 시. 데이터가 없는 01~03시는 직전 시간대(24시)로 본다."""
    if hour < DAY_START_HOUR:
        return 24
    return DAY_START_HOUR if hour < 6 else hour


_HOLIDAY_YEARS: dict[int, dict] = {}


def kr_holidays(year: int) -> dict:
    """그 연도 한국 공휴일 {날짜: 이름}. 학습(notebooks/preprocess.py)이 쓴 holidays 패키지 목록과 같은 것."""
    if year not in _HOLIDAY_YEARS:
        _HOLIDAY_YEARS[year] = dict(country_holidays("KR", years=year, language="ko"))
    return _HOLIDAY_YEARS[year]


def holiday_label(name: str) -> str:
    """'개천절 대체 휴일' → '개천절 (대체 휴일)' (이름 뒤에 대체 표시를 가로로 붙인다)."""
    parts = []
    for part in name.split("; "):
        text = part.replace(" 대체 휴일", "").strip()
        parts.append(f"{text} (대체 휴일)" if text and text != part else part)
    return "; ".join(parts)


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

    def __init__(self, fetch=slot_weather):
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
        # 날씨를 받는 동안 사용자가 고친 칸만 덮어쓰지 않는다.
        self._temperature_edited = False
        self._rain_edited = False
        self._snow_edited = False
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
        badge_row = QHBoxLayout()
        badge_row.setSpacing(8)
        badge_row.addLayout(self._badges)
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
        self._date.dateChanged.connect(self._reset_weather_edit)
        self._date.dateChanged.connect(self._schedule_weather)
        self._hour = ComboBox()
        for hour, label in HOUR_SLOTS.items():
            self._hour.addItem(label, hour)
        self._hour.setMaxVisibleItems(12)
        self._hour.currentIndexChanged.connect(self._reset_weather_edit)
        self._hour.currentIndexChanged.connect(self._schedule_weather)
        # 요일은 날짜에서만 정해 날짜 바로 아래에 글자로 보여 준다 (직접 고를 수 없음).
        self._weekday = styled_label("", "weekdayText")
        self._weekday.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        # 공휴일은 고른 날짜로 자동 판정해 요일 옆에 빨간색으로 알린다.
        self._holiday_tag = styled_label("", "fieldNote")
        self._holiday_tag.setWordWrap(True)
        self._holiday_tag.setVisible(False)
        date_cell = QWidget()
        date_layout = QVBoxLayout(date_cell)
        date_layout.setContentsMargins(0, 0, 0, 0)
        date_layout.setSpacing(4)
        date_layout.addWidget(self._date)
        day_row = QHBoxLayout()
        day_row.setContentsMargins(2, 0, 0, 0)
        day_row.setSpacing(8)
        day_row.addWidget(self._weekday)
        day_row.addWidget(self._holiday_tag)
        day_row.addStretch()
        date_layout.addLayout(day_row)

        grid = self._section(layout, "조건", reset)
        self._add_row(grid, "호선", self._line)
        row = grid.rowCount()
        grid.addWidget(styled_label("날짜", "fieldLabel"), row, 0, Qt.AlignmentFlag.AlignTop)
        grid.addWidget(date_cell, row, 1)
        self._add_row(grid, "시간", self._hour)

        # 날씨
        self._temperature = DoubleSpinBox()
        self._temperature.setDecimals(1)
        self._temperature.setSingleStep(0.5)
        self._temperature.setRange(TEMPERATURE_MIN - 0.1, TEMPERATURE_MAX)  # 최솟값 = 미입력
        self._temperature.setSpecialValueText(UNSET_TEXT)
        self._temperature.setSuffix(" ℃")
        self._temperature.setValue(self._temperature.minimum())
        # 비·눈은 그 시각의 관측값을 기본으로 채우고, 사용자가 직접 바꿀 수 있다.
        # 미래 날짜는 관측값이 없어 '안 옴'으로 둔다.
        self._rain = QCheckBox("비")
        self._snow = QCheckBox("눈")
        for box in (self._rain, self._snow):
            box.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rain.setToolTip("비가 오는 날로 계산합니다 (눈도 비에 포함)")
        self._snow.setToolTip("눈이 오는 날로 계산합니다")
        self._weather_note = styled_label("", "fieldNote")
        self._weather_note.setWordWrap(True)
        self._temperature.valueChanged.connect(self._on_temperature_edited)
        self._rain.toggled.connect(self._on_rain_toggled)
        self._snow.toggled.connect(self._on_snow_toggled)

        self._add_row(grid, "날씨", self._temperature, self._rain, self._snow)
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
        """예측 결과 카드: '예상 승차 인원'은 항상 보이고, '모델 평가'만 접었다 편다."""
        self._result = QFrame()
        self._result.setObjectName("resultCard")
        box = QVBoxLayout(self._result)
        box.setContentsMargins(14, 10, 14, 12)
        box.setSpacing(6)

        self._boarding_title = styled_label("예상 승차 인원", "resultSectionTitle")
        self._result_boarding_box = QWidget()
        boarding_box = QVBoxLayout(self._result_boarding_box)
        boarding_box.setContentsMargins(0, 0, 0, 6)
        boarding_box.setSpacing(6)
        self._result_boarding = {}
        boarding = self._result_grid()
        for model, features in MODELS.items():
            name, (value,) = self._result_line(boarding, f"{model} 모델", "resultNumber")
            name.setToolTip(features)
            self._result_boarding[model] = value
        boarding_box.addLayout(boarding)

        # 검증되지 않은 장기 추정 구간에만 뜨는 안내 줄
        self._result_unverified = styled_label("", "resultWarning")
        self._result_unverified.setWordWrap(True)
        self._result_unverified.setVisible(False)
        boarding_box.addWidget(self._result_unverified)
        box.addWidget(self._boarding_title)
        box.addWidget(self._result_boarding_box)
        box.addWidget(divider())

        self._scores_header = self._accordion_header("모델 평가", expanded=False)
        self._scores_header.setToolTip(
            "Test 데이터에서 실제 승차 인원과 비교한 오차 (단위: 명). 낮을수록 좋다"
        )
        self._result_scores_box = QWidget()
        scores_box = QVBoxLayout(self._result_scores_box)
        scores_box.setContentsMargins(0, 0, 0, 6)
        scores_box.setSpacing(6)
        self._result_scores = {}
        scores = self._result_grid()
        for column, header in enumerate(("MAE", "RMSE"), start=1):
            label = styled_label(header, "fieldLabel")
            label.setAlignment(Qt.AlignmentFlag.AlignRight)
            scores.addWidget(label, 0, column)
        for model, features in MODELS.items():
            name, (mae, rmse) = self._result_line(
                scores, model, "resultNumber", "resultNumber"
            )
            name.setToolTip(features)
            self._result_scores[model] = (name, mae, rmse)
        scores_box.addLayout(scores)
        box.addWidget(self._scores_header)
        box.addWidget(self._result_scores_box)

        self._result_note = styled_label("", "fieldNote")
        self._result_note.setWordWrap(True)
        box.addWidget(self._result_note)

        self._result_scores_box.setVisible(False)
        self._scores_header.toggled.connect(self._scores_toggled)

        self._result.hide()
        return self._result

    @staticmethod
    def _accordion_header(title: str, expanded: bool) -> QPushButton:
        header = QPushButton()
        header.setObjectName("accordionHeader")
        header.setCheckable(True)
        header.setChecked(expanded)
        header.setCursor(Qt.CursorShape.PointingHandCursor)
        header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header.setProperty("sectionTitle", title)
        header.setText(f"{'▼' if expanded else '▶'}  {title}")
        return header

    @staticmethod
    def _set_accordion_text(header: QPushButton, expanded: bool) -> None:
        header.setText(f"{'▼' if expanded else '▶'}  {header.property('sectionTitle')}")

    def _scores_toggled(self, expanded: bool) -> None:
        """'모델 평가'만 접었다 편다. '예상 승차 인원'은 항상 보인다."""
        self._result_scores_box.setVisible(expanded)
        self._set_accordion_text(self._scores_header, expanded)

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

        self._line.clear()
        known = [self._line_has_data(self._lines[i].name, station.name) for i in station.lines]
        if not any(known):  # 데이터가 있는 호선이 하나도 없으면 전부 고를 수 있게 둔다 (예측 시 안내문구)
            known = [True] * len(station.lines)
        for line_id, has in zip(station.lines, known, strict=True):
            line = self._lines[line_id]
            self._line.addItem(color_dot(line.color),
                               line.name if has else f"{line.name} (데이터 없음)", line_id)
            if not has:  # 데이터 없는 호선은 고르지 못하게 막는다 (기본 호선을 잘못 잡아 오인하지 않도록)
                self._line.model().item(self._line.count() - 1).setEnabled(False)
        self._line.setCurrentIndex(known.index(True) if True in known else -1)
        self._line.setEnabled(len(station.lines) > 1)

        self._empty.hide()
        self._scroll.show()

    def _line_has_data(self, line: str, station: str) -> bool:
        """그 호선·역의 학습 데이터가 있는지. 예측기가 없거나 물어볼 수 없으면 있는 것으로 본다."""
        if self._predictor is None:
            return True
        has_data = getattr(self._predictor, "has_data", None)
        return True if has_data is None else bool(has_data(line, station))

    def focus_search(self) -> None:
        self._search.setFocus()
        self._search.selectAll()

    # ---------- 입력값 ----------

    def prediction_input(self) -> PredictionInput | None:
        """지금 입력된 예측 조건. 역을 고르기 전에는 None.

        주말·공휴일은 날짜에서, 비·눈은 체크박스에서 정한다. 체크박스의 기본값은
        그 시각의 관측 날씨이고, 미래 날짜는 관측값이 없어 '안 옴'으로 둔다.
        """
        if self._station is None:
            return None
        temperature = self._temperature.value()
        # 콤보 표시명에 '(데이터 없음)'이 붙을 수 있으므로 호선 id 로 이름을 찾는다
        line_id = self._line.currentData()
        line = self._lines[line_id].name if line_id in self._lines else self._line.currentText()
        chosen = self._date.date().toPython()
        weekday = chosen.weekday()  # 요일은 날짜에서만 정한다 (화면에는 읽기 전용 글자로 표시)
        return PredictionInput(
            date=chosen.isoformat(),
            line=line,
            station=self._station.name,
            hour=self._hour.currentData(),
            weekday=weekday,
            is_weekend=weekday >= WEEKEND_FROM,
            is_holiday=chosen in kr_holidays(chosen.year),
            temperature=None if temperature == self._temperature.minimum() else round(temperature, 1),
            rainfall=self._rain.isChecked(),
            snowfall=self._snow.isChecked(),
        )

    def selected_time(self) -> datetime:
        """고른 날짜·시간대의 시작 시각 (KST 정시). 24시는 다음 날 00시다."""
        date = self._date.date().toPython()
        midnight = datetime(date.year, date.month, date.day, tzinfo=KST)
        return midnight + timedelta(hours=self._hour.currentData())

    def reset_to_now(self, load_weather: bool = True) -> None:
        """날짜·시는 현재 시각으로, 요일·주말은 그 날짜로, 날씨는 서울 현재 날씨로. 공휴일은 그대로 둔다.

        00~04시는 전날 업무일에 속하므로 날짜를 하루 앞으로 두고 24시를 고른다.
        """
        now = datetime.now(KST)
        date = (now - timedelta(hours=DAY_START_HOUR)).date()
        self._reset_weather_edit()  # '현재로'도 직접 입력한 날씨를 버리고 지금 날씨를 다시 받는다
        self._date.setDate(QDate(date.year, date.month, date.day))
        self._apply_weekday()  # 날짜가 그대로면 dateChanged 가 오지 않으므로 직접 다시 계산
        self._hour.setCurrentIndex(self._hour.findData(hour_slot(now.hour)))
        self._weather_timer.stop()  # 날짜·시를 바꾸며 잡힌 조회는 기다리지 않는다
        if load_weather:
            self.refresh_weather()

    def set_date_range(self, start: str, end: str) -> None:
        """예측 가능한 날짜 범위를 달력에 반영한다. 범위 밖 날짜는 아예 고를 수 없다."""
        first = QDate.fromString(start, "yyyy-MM-dd")
        last = QDate.fromString(end, "yyyy-MM-dd")
        if first.isValid() and last.isValid() and first <= last:
            self._date.setMinimumDate(first)
            self._date.setMaximumDate(last)

    def _apply_weekday(self) -> None:
        weekday = self._date.date().toPython().weekday()
        self._weekday.setText(f"{WEEKDAY_NAMES[weekday]}요일")
        # 공휴일은 고른 날짜로 자동 판정한다 (학습 때 쓴 목록과 같은 공휴일).
        chosen = self._date.date().toPython()
        holidays = kr_holidays(chosen.year)
        is_holiday = chosen in holidays
        if is_holiday:  # 공휴일이면 빨간색으로 알린다
            self._holiday_tag.setText(f"공휴일 · {holiday_label(holidays[chosen])}")
            set_error(self._holiday_tag, True)
            self._holiday_tag.setVisible(True)
        else:
            self._holiday_tag.setVisible(False)

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
            else:
                # 모델 하나가 계산되지 않으면 이유를 알려 준다 (그냥 '-' 로만 두지 않는다)
                failed = getattr(self._predictor, "last_errors", None) or {}
                if failed:
                    error = " · ".join(f"{name} 모델 계산 실패 ({msg})" for name, msg in failed.items())
        self.show_result(values, predictions, scores, error)

    @staticmethod
    def _is_future(values: PredictionInput) -> bool:
        """데이터 끝(2026-06-30) 이후 날짜인지 — D·E 가 실측 부족으로 비는 구간."""
        try:
            return date.fromisoformat(str(values.date)[:10]) > DATA_END
        except ValueError:
            return False

    def show_result(self, values: PredictionInput, predictions: dict, scores: dict, error: str = "") -> None:
        future = self._is_future(values)
        for model, label in self._result_boarding.items():
            value = predictions.get(model)
            label.setText(format_people(value))
            label.setToolTip("")
            # 왜 비었는지 화면에 남긴다 ('-' 만 두면 오류로 읽힌다)
            missing = value is None and future and model in SEQUENCE_MODELS and bool(predictions)
            if missing:
                label.setText(NO_HISTORY_TEXT)
                label.setToolTip(NO_HISTORY_TIP.format(end=DATA_END.isoformat()))
            set_flag(label, "missing", missing)

        self._show_unverified_note(predictions, values)

        for model, (_name, mae, rmse) in self._result_scores.items():
            score = scores.get(model)
            mae.setText(format_score(score and score.mae))
            rmse.setText(format_score(score and score.rmse))

        self._result_note.setText(error)
        set_error(self._result_note, bool(error))
        self._result_note.setVisible(bool(error))
        self._result.show()
        # 카드가 레이아웃에 자리 잡은 뒤 보이도록 스크롤한다
        QTimer.singleShot(0, lambda: self._scroll.ensureWidgetVisible(self._result))

    def _show_unverified_note(self, predictions: dict, values: PredictionInput) -> None:
        """검증 근거가 약한 구간에만 경고 줄을 띄운다.

        182일 이후는 잰 적이 없어 '미검증', 137~181일은 근거가 136일까지만 있어 '일부 미검증'.
        과거 날짜와 1~136일 구간에는 숫자만 보여 준다.
        """
        bucket = predictions.get("horizon_bucket")
        if not bucket:
            self._hide_unverified_note()
            return
        try:
            day = date.fromisoformat(values.date[:10])
        except ValueError:
            self._hide_unverified_note()
            return
        end = DATA_END.isoformat()
        days = (day - DATA_END).days
        if bucket in VERIFIED_HORIZON_BUCKETS:
            if bucket == "92-181" and days > EVAL_SPAN_DAYS:
                self._result_unverified.setText(PARTIAL_NOTE)
                self._result_unverified.setToolTip(PARTIAL_TOOLTIP.format(span=EVAL_SPAN_DAYS))
                self._result_unverified.setVisible(True)
                return
            self._hide_unverified_note()
            return
        self._result_unverified.setText(UNVERIFIED_NOTE)
        self._result_unverified.setToolTip(UNVERIFIED_TOOLTIP.format(end=end))
        self._result_unverified.setVisible(True)

    def _hide_unverified_note(self) -> None:
        self._result_unverified.setText("")
        self._result_unverified.setToolTip("")
        self._result_unverified.setVisible(False)

    # ---------- 날씨 ----------

    def refresh_weather(self) -> None:
        """고른 날짜·시의 서울 날씨를 받는다. 직접 고친 값은 덮어쓰지 않는다.

        미래 시각은 관측 자료가 없으므로 자동 조회하지 않는다. 기온은 평년값, 비·눈은
        '안 옴'으로 두고 계산한다 (예측기가 기온 없음을 평년값으로 채운다).
        """
        self._weather_timer.stop()
        set_error(self._weather_note, False)
        tm = self.selected_time()
        if tm > datetime.now(KST):
            self._weather.cancel()
            self._clear_weather()
            self._show_climatology(tm.month)
            self._weather_note.setText(
                "미래 시각이라 관측된 날씨가 없습니다. 비는 안 오고 기온은 평년값으로 계산합니다."
            )
            return
        self._weather_note.setText(f"서울 {tm:%Y-%m-%d %H}시 날씨를 불러오는 중…")
        self._weather.load(tm)

    def _schedule_weather(self) -> None:
        """날짜·시가 바뀌면 이전 시각의 응답은 버리고, 잠시 뒤 새 시각의 날씨를 받는다."""
        self._weather.cancel()
        self._weather_timer.start()

    def _reset_weather_edit(self) -> None:
        """날짜·시가 바뀌면 직접 입력한 날씨는 그 시각의 값이 아니므로 자동 조회를 다시 따른다.

        풀지 않으면 이전 날짜·시에 입력한 기온·비·눈이 남아 엉뚱한 날씨로 예측하게 된다.
        """
        self._temperature_edited = False
        self._rain_edited = False
        self._snow_edited = False

    def apply_weather(self, weather: HourlyWeather) -> None:
        self._applying_weather = True
        self._clear_climatology()  # 관측값이 왔으므로 평년 가정 표시를 지운다
        try:
            if not self._temperature_edited:
                temperature = weather.temperature
                self._temperature.setValue(
                    self._temperature.minimum() if temperature is None else temperature
                )
            # 눈이면 비(강수)도 함께 켠다.
            if not self._snow_edited:
                self._snow.setChecked(bool(weather.snowfall))
            if not self._rain_edited:
                self._rain.setChecked(bool(weather.rainfall or weather.snowfall))
        finally:
            self._applying_weather = False

    def _on_temperature_edited(self, value: float) -> None:
        if self._applying_weather:
            return
        self._temperature_edited = True
        if value != self._temperature.minimum():
            self._clear_climatology()  # 직접 넣은 값은 가정값이 아니다

    def _on_rain_toggled(self, checked: bool) -> None:
        if self._applying_weather:
            return
        self._rain_edited = True
        if not checked and self._snow.isChecked():  # 눈도 비에 포함되므로 함께 끈다
            self._snow_edited = True
            self._applying_weather = True
            try:
                self._snow.setChecked(False)
            finally:
                self._applying_weather = False

    def _on_snow_toggled(self, checked: bool) -> None:
        if self._applying_weather:
            return
        self._snow_edited = True
        if checked and not self._rain.isChecked():
            self._rain_edited = True
            self._applying_weather = True
            try:
                self._rain.setChecked(True)
            finally:
                self._applying_weather = False

    def _on_weather_loaded(self, weather: HourlyWeather) -> None:
        source = f"{weather.source} 서울 {weather.date} {weather.hour:02d}시 관측"
        selected = self.selected_time()
        if (weather.date, weather.hour) != (f"{selected:%Y-%m-%d}", selected.hour):
            source += " (고른 시각의 자료가 없어 직전 관측값)"
        self.apply_weather(weather)
        if self._temperature_edited or self._rain_edited or self._snow_edited:
            temperature = "-" if weather.temperature is None else f"{weather.temperature:.1f}℃"
            self._weather_note.setText(f"{source} {temperature} · 직접 입력한 값을 유지합니다")
        else:
            self._weather_note.setText(source)

    def _on_weather_failed(self, error: str) -> None:
        self._clear_weather()  # 지난 시각의 값이 그대로 남아 조용히 쓰이지 않게
        self._weather_note.setText(f"날씨를 불러오지 못했습니다. 직접 입력하세요.\n{error}")
        set_error(self._weather_note, True)

    def _show_climatology(self, month: int) -> None:
        """미래 시각의 기온 칸에 평년값을 회색으로 보여 준다.

        값 자체는 '미입력'으로 두어 예측기가 평년값을 쓰게 하고, 화면에만 가정값을 적는다.
        """
        value = climatology_temperature(month)
        self._applying_weather = True
        try:
            if self._temperature_edited:
                return  # 직접 넣은 값이 있으면 가정값으로 덮지 않는다
            self._temperature.setValue(self._temperature.minimum())
            self._temperature.setSpecialValueText(f"평년 {value:.1f}℃")
            self._set_climatology_style(True)
            self._temperature.setToolTip(
                f"관측값이 없는 미래 시각이라 {month}월 평년 기온 {value:.1f}℃로 계산합니다. "
                "직접 입력할 수 있습니다."
            )
        finally:
            self._applying_weather = False

    def _clear_climatology(self) -> None:
        """평년값 표시를 지우고 '미입력' 자리로 되돌린다 (관측값이 오거나 직접 입력하면)."""
        self._temperature.setSpecialValueText(UNSET_TEXT)
        self._temperature.setSuffix(" ℃")
        self._temperature.setToolTip("")
        self._set_climatology_style(False)

    def _set_climatology_style(self, on: bool) -> None:
        """가정값임을 알리려고 글자색을 회색으로 바꾼다 (QSS 의 [climatology] 규칙)."""
        if on:
            self._temperature.setSuffix("")  # 평년값 문구에 이미 ℃ 가 붙어 있다
        self._temperature.setProperty("climatology", on)
        style = self._temperature.style()
        style.unpolish(self._temperature)
        style.polish(self._temperature)

    def _clear_weather(self) -> None:
        """날씨 입력란을 비운다. 직접 입력한 칸은 건드리지 않는다."""
        self._clear_climatology()
        self._applying_weather = True
        try:
            if not self._temperature_edited:
                self._temperature.setValue(self._temperature.minimum())
            if not self._snow_edited:
                self._snow.setChecked(False)
            if not self._rain_edited:
                self._rain.setChecked(False)
        finally:
            self._applying_weather = False

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
