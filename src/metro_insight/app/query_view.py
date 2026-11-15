"""질의 탐색 화면: 한 줄 질의로 예측·비교·실측·자료 상태를 확인하고 근거를 함께 보여 준다.

계산은 배경 스레드에서 돌리고 화면은 결과만 받는다(오래 걸리는 질의에도 창이 멈추지 않게).
답·숫자는 질의 도구가 돌려준 값을 그대로 옮겨 적는다 — 화면이 숫자를 새로 만들지 않는다.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from metro_insight.app.station_panel import set_flag

DEFAULT_MODE = "light"

# 빠른 질의: (단추 글자, 질의 문장, 고른 역이 있어야 하는지)
QUICK_QUERIES = (
    ("내일 08시 예측", "{station} 내일 08시 예측", True),
    ("비 오는 날 비교", "{station} 내일 08시 비 오는 날 비교", True),
    ("데이터 최신 상태 점검", "데이터 최신 상태 점검", False),
    ("최근 이상치 일수", "최근 이상치 일수", False),
)

# '이 역' 처럼 지금 화면을 가리키는 말 → 고른 역·호선 이름으로 바꾼다
STATION_WORDS = ("선택 역", "현재 역", "이 역")
LINE_WORDS = ("선택 호선", "이 호선")

# 근거 한 건의 종류 이름 (질의 도구가 돌려주는 kind)
KIND_LABELS = {
    "prediction": "예측",
    "comparison": "비교",
    "simulation": "조건 비교",
    "explanation": "근거 설명",
    "aggregate": "집계",
    "history": "실측",
    "reconcile": "합계 대조",
    "backtest": "재확인",
    "drift": "자료 상태",
    "alerts": "이상치",
    "report": "리포트",
    "clarification": "조건 확인",
}

# 근거 줄에서 값 앞에 붙일 이름 (없는 이름은 그대로 쓴다)
VALUE_LABELS = {
    "value": "값",
    "n": "관측",
    "coverage": "커버리지",
    "mae": "MAE",
    "rmse": "RMSE",
    "model": "모델",
    "bucket": "구간",
    "days": "경과일",
    "stale": "상태",
    "data_end": "기준일",
    "last_date": "마지막 날짜",
    "last_value": "마지막 값",
    "z": "기준",
    "diff": "차이",
    "delta": "차이",
    "delta_pct": "차이(%)",
    "metric": "지표",
    "base": "기준값",
    "top_station": "가장 큰 역",
    "top_value": "값",
    "model_value": "값",
    "baseline": "평년 비교",
    "features": "조건",
    "kind": "종류",
    "path": "경로",
    "sections": "섹션",
    "question": "질문",
    "horizon": "지평",
    "missing_count": "빠진 날",
    "days_since_data_end": "경과일",
    "line_sum": "집계 합",
    "station_sum": "역별 합",
    "stations_with_data": "자료 있는 역",
    "station_count": "역 수",
    "count": "건수",
    "map_pairs": "노선도 대상 역",
    "map_only": "예측 없는 역",
    "map_only_exact": "이름이 맞는 역",
    "map_only_unresolved": "이름을 못 찾은 역",
}

# 근거 줄에서 이미 앞부분에 적은 값 (뒤에 또 적지 않는다)
_LEAD_KEYS = {"line", "station", "date", "hour", "date_from", "date_to", "name", "stale"}
# 모델 이름 한 글자(A~E). 이 값은 줄 앞쪽에 나란히 세운다.
_MODEL_KEYS = frozenset({"A", "B", "C", "D", "E"})


def _default_turn(utterance: str, ui_context: dict | None):
    """질의 한 턴을 실제로 계산한다. 질의 도구는 처음 쓸 때 읽는다."""
    from metro_insight.agent.orchestrator import run_turn  # noqa: PLC0415

    return run_turn(utterance, ui_context)


def _rest_items(display: dict) -> list[str]:
    """근거 줄에 이어 적을 '이름 값' 목록.

    순서는 모델 값 → 기준 대비 차이(큰 것부터) → 비율 차이 → 나머지다. 그래야 줄 끝이
    4개에서 잘려도 무엇이 달라졌는지 남는다. '0일 지남'은 적지 않는다(바로 그날 자료라는 뜻).
    """
    models: list[str] = []
    deltas: list[tuple[float, str]] = []
    percents: list[str] = []
    others: list[str] = []
    for key, value in display.items():
        if key in _LEAD_KEYS or value in (None, ""):
            continue
        text = str(value)
        if key == "days" and text.startswith("0"):
            continue
        if key in _MODEL_KEYS:  # A·B·C 모델 값
            models.append(f"{key} {text}")
        elif key.startswith("delta_pct_"):
            percents.append(f"{key[len('delta_pct_') :]} 비율 차이 {text}")
        elif key.startswith("delta_") or key == "diff":
            letter = key[len("delta_") :] if key.startswith("delta_") else "값"
            try:
                weight = abs(float(text.rstrip("명%").replace(",", "")))
            except ValueError:
                weight = 0.0
            deltas.append((weight, f"기준 대비 {letter} {text}"))
        else:
            others.append(f"{VALUE_LABELS.get(key, key)} {text}")
    deltas.sort(key=lambda item: item[0], reverse=True)
    return [*models, *(text for _weight, text in deltas), *percents, *others]


def fact_summary(fact: dict) -> str:
    """근거 한 건을 한 줄로: '2호선 강남 · 2026-11-16 08시 · A 2,684명'."""
    display = fact.get("display") or {}
    head: list[str] = []
    where = " ".join(str(part) for part in (display.get("line"), display.get("station")) if part)
    if where:
        head.append(where)
    when = " ".join(str(part) for part in (display.get("date"), display.get("hour")) if part)
    if when:
        head.append(when)
    if display.get("date_from"):
        span = f"{display['date_from']}~{display.get('date_to') or display['date_from']}"
        head.append(span)
    if display.get("name"):
        head.append(f"{display['name']} 조건")
    if display.get("stale"):
        head.append(f"상태 {display['stale']}")
    return " · ".join([*head, *_rest_items(display)[:4]])


class QueryView(QWidget):
    """질의 화면. 계산은 배경 스레드에서 돌리고 결과만 화면에 붙인다."""

    turn_done = Signal(object)  # 결과 묶음
    turn_failed = Signal(str)  # 화면에 보여 줄 실패 사유

    def __init__(self, turn=None, context=None):
        super().__init__()
        self.setObjectName("queryPage")
        self._turn = turn or _default_turn  # 시험에서는 다른 계산기로 바꿔 끼울 수 있다
        self._context = context or (lambda: {})  # 지금 고른 역·호선을 주는 함수
        self._thread: threading.Thread | None = None
        self._request = 0  # 마지막 질의 번호 (늦게 온 지난 결과를 버린다)
        self._mode = DEFAULT_MODE
        self._chips: list[QPushButton] = []
        self._chip_needs_station: list[bool] = []
        self.conditions: list[str] = []
        self.evidence_texts: list[QLabel] = []
        self.gap_texts: list[QLabel] = []
        self.turn_done.connect(self.show_result)
        self.turn_failed.connect(self.show_failure)
        self.setLayout(self._build_page())
        self.refresh_context()

    # ---------- 구성 ----------

    def _build_page(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        layout.addWidget(self._build_form())
        layout.addWidget(self._build_results(), 1)
        return layout

    def _build_form(self) -> QFrame:
        card = QFrame()
        card.setObjectName("queryCard")
        body = QVBoxLayout(card)
        body.setContentsMargins(20, 20, 20, 20)
        body.setSpacing(12)

        heading = QLabel("조건을 적어 확인합니다")
        heading.setObjectName("queryHeading")
        caption = QLabel(
            "역·날짜·시각을 함께 적으면 그 조건으로 계산합니다. "
            "노선도에서 고른 역은 따로 적지 않아도 붙습니다."
        )
        caption.setObjectName("queryCaption")
        caption.setWordWrap(True)

        self.input = QLineEdit()
        self.input.setObjectName("queryInput")
        self.input.setPlaceholderText("예: 강남 내일 08시 예측")
        self.input.setClearButtonEnabled(True)
        self.input.returnPressed.connect(self.submit_query)

        self.submit = QPushButton("질의")
        self.submit.setObjectName("querySubmit")
        self.submit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.submit.setToolTip("적은 문장으로 계산합니다 (Enter)")
        self.submit.clicked.connect(self.submit_query)

        line = QHBoxLayout()
        line.setSpacing(8)
        line.addWidget(self.input, 1)
        line.addWidget(self.submit)

        quick = QHBoxLayout()
        quick.setSpacing(8)
        quick_label = QLabel("빠른 질의")
        quick_label.setObjectName("queryExampleCaption")
        quick.addWidget(quick_label)
        for text, template, needs_station in QUICK_QUERIES:
            chip = QPushButton(text)
            chip.setObjectName("queryChip")
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.clicked.connect(lambda _=False, sentence=template: self.ask(sentence))
            quick.addWidget(chip)
            self._chips.append(chip)
            self._chip_needs_station.append(needs_station)
        quick.addStretch()

        self.context_label = QLabel("")
        self.context_label.setObjectName("queryContext")
        quick.addWidget(self.context_label)

        self.status = QLabel("")
        self.status.setObjectName("queryStatus")
        self.status.setVisible(False)

        body.addWidget(heading)
        body.addWidget(caption)
        body.addLayout(line)
        body.addLayout(quick)
        body.addWidget(self.status)
        return card

    def _build_results(self) -> QScrollArea:
        self._scroll = QScrollArea()
        self._scroll.setObjectName("queryScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        body = QWidget()
        body.setObjectName("queryResult")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._build_empty())
        layout.addWidget(self._build_answer())
        layout.addWidget(self._build_evidence())
        layout.addWidget(self._build_gaps())
        layout.addStretch()
        self._scroll.setWidget(body)
        return self._scroll

    def _build_empty(self) -> QFrame:
        card = QFrame()
        card.setObjectName("queryEmptyCard")
        body = QVBoxLayout(card)
        body.setContentsMargins(24, 28, 24, 28)
        body.setSpacing(8)
        title = QLabel("아직 질의가 없습니다")
        title.setObjectName("queryEmptyTitle")
        hint = QLabel(
            "예측·비교·실측 조회·자료 상태를 한 줄로 물어보세요.\n"
            "예: 강남 내일 08시 예측 · 강남 2주 실측 · 데이터 최신 상태 점검"
        )
        hint.setObjectName("queryEmptyBody")
        hint.setWordWrap(True)
        body.addWidget(title)
        body.addWidget(hint)
        self.empty = card
        return card

    def _build_answer(self) -> QFrame:
        card = QFrame()
        card.setObjectName("queryAnswerCard")
        body = QVBoxLayout(card)
        body.setContentsMargins(20, 18, 20, 18)
        body.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        title = QLabel("답")
        title.setObjectName("queryCardTitle")
        self.meta = QLabel("")
        self.meta.setObjectName("queryMeta")
        head.addWidget(title)
        head.addStretch()
        head.addWidget(self.meta)

        self.answer = QLabel("")
        self.answer.setObjectName("queryAnswer")
        self.answer.setWordWrap(True)
        self.answer.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._conditions = QHBoxLayout()
        self._conditions.setSpacing(8)

        body.addLayout(head)
        body.addWidget(self.answer)
        body.addLayout(self._conditions)
        self.answer_card = card
        card.setVisible(False)
        return card

    def _build_evidence(self) -> QFrame:
        card = QFrame()
        card.setObjectName("queryEvidenceCard")
        body = QVBoxLayout(card)
        body.setContentsMargins(20, 18, 20, 18)
        body.setSpacing(8)
        title = QLabel("근거 요약")
        title.setObjectName("queryCardTitle")
        caption = QLabel("답에 쓴 자료와 조건입니다. 숫자는 자료에서 바로 계산한 값입니다.")
        caption.setObjectName("queryEvidenceCaption")
        caption.setWordWrap(True)
        # 표찰 폭이 저마다 달라도 글자가 같은 자리에서 시작하도록 칸을 나눠 담는다
        self._evidence_rows = QGridLayout()
        self._evidence_rows.setHorizontalSpacing(8)
        self._evidence_rows.setVerticalSpacing(8)
        self._evidence_rows.setColumnStretch(1, 1)
        body.addWidget(title)
        body.addWidget(caption)
        body.addLayout(self._evidence_rows)
        self.evidence_card = card
        card.setVisible(False)
        return card

    def _build_gaps(self) -> QFrame:
        card = QFrame()
        card.setObjectName("queryGapCard")
        body = QVBoxLayout(card)
        body.setContentsMargins(20, 18, 20, 18)
        body.setSpacing(8)
        title = QLabel("계산하지 못한 항목")
        title.setObjectName("queryGapTitle")
        self._gap_rows = QVBoxLayout()
        self._gap_rows.setSpacing(4)
        body.addWidget(title)
        body.addLayout(self._gap_rows)
        self.gap_card = card
        card.setVisible(False)
        return card

    # ---------- 고른 역 ----------

    def resolve(self, utterance: str) -> str:
        """'이 역' 같은 말과 {station} 자리를 지금 고른 역·호선 이름으로 바꾼다."""
        context = self._context() or {}
        station = context.get("station")
        line = context.get("line")
        text = str(utterance).replace("{station}", str(station or ""))
        text = text.replace("{line}", str(line or ""))
        if station:
            for word in STATION_WORDS:
                text = text.replace(word, str(station))
        if line:
            for word in LINE_WORDS:
                text = text.replace(word, str(line))
        return " ".join(text.split())

    def refresh_context(self) -> None:
        """고른 역·호선 표시와 빠른 질의 단추의 사용 여부를 맞춘다."""
        context = self._context() or {}
        station = context.get("station")
        line = context.get("line")
        if station and line:
            text = f"기준 역 {station} · {line}"
        elif station:
            text = f"기준 역 {station}"
        else:
            text = "기준 역 없음"
        self.context_label.setText(text)
        self.context_label.setToolTip(
            f"{station} 기준으로 계산합니다" if station else "노선도에서 역을 고르면 그 역으로 계산합니다"
        )
        for chip, needs_station in zip(self._chips, self._chip_needs_station, strict=True):
            chip.setEnabled(bool(station) or not needs_station)

    # ---------- 질의 ----------

    def submit_query(self) -> None:
        self.ask(self.input.text())

    def ask(self, utterance: str, wait: bool = False):
        """질의를 보낸다. wait=True 면 이 스레드에서 계산해 바로 그린다(검사·짧은 확인용)."""
        text = self.resolve(utterance)
        if not text:
            return None
        if self.input.text() != text:
            self.input.setText(text)
        context = dict(self._context() or {})
        self._set_busy(True)
        if wait:
            try:
                result = self._turn(text, context)
            except Exception as exc:  # noqa: BLE001 — 실패 사유를 화면에 남긴다
                self.show_failure(f"{type(exc).__name__}: {exc}")
                return None
            self.show_result(result)
            return result
        self._start(text, context)
        return None

    def _start(self, text: str, context: dict) -> None:
        self._request += 1
        self._thread = threading.Thread(
            target=self._run, args=(self._request, text, context), daemon=True
        )
        self._thread.start()

    def _run(self, request: int, text: str, context: dict) -> None:
        try:
            result = self._turn(text, context)
        except Exception as exc:  # noqa: BLE001
            if request == self._request:
                self.turn_failed.emit(f"{type(exc).__name__}: {exc}")
            return
        if request == self._request:  # 그 사이 다른 질의를 보냈으면 지난 결과는 버린다
            self.turn_done.emit(result)

    def _set_busy(self, busy: bool) -> None:
        self.submit.setEnabled(not busy)
        self.submit.setText("계산 중" if busy else "질의")
        self.status.setVisible(busy)
        set_flag(self.status, "error", False)
        self.status.setText("계산 중입니다. 끝나면 이 자리에 결과가 붙습니다." if busy else "")

    # ---------- 결과 ----------

    def show_result(self, result: dict | None) -> None:
        """질의 결과를 화면에 붙인다 (받은 값을 그대로 옮겨 적는다)."""
        self.empty.setVisible(False)
        self.answer_card.setVisible(True)
        if not result:
            self.show_failure("결과가 비어 있습니다.")
            return
        self._set_busy(False)
        self.answer.setText(str(result.get("answer") or ""))

        bundle = result.get("bundle") or {}
        facts = bundle.get("facts") or []
        parts = []
        latency = result.get("latency_ms")
        if latency is not None:
            parts.append(f"처리 {int(latency):,}ms")
        parts.append(f"근거 {len(facts)}건")
        self.meta.setText(" · ".join(parts))

        slots = (result.get("plan") or {}).get("slots") or {}
        self._fill_conditions(slots)
        self._set_evidence(facts)
        self._set_gaps(_unavailable_texts(bundle))
        self._scroll.verticalScrollBar().setValue(0)

    def show_failure(self, reason: str) -> None:
        """계산하지 못했을 때: 사유만 남기고 지난 결과는 지운다."""
        self._set_busy(False)
        self.empty.setVisible(False)
        self.answer_card.setVisible(True)
        self.answer.setText("질의를 처리하지 못했습니다.")
        self.meta.setText("")
        self._fill_conditions({})
        self._set_evidence([])
        self._set_gaps([f"사유: {reason}"])
        self._scroll.verticalScrollBar().setValue(0)

    def _fill_conditions(self, slots: dict) -> None:
        """해석한 조건(호선·역·날짜·시각)을 작은 표찰로 보여 준다."""
        self._clear(self._conditions)
        chips: list[str] = []
        for key in ("line", "station", "date"):
            if slots.get(key):
                chips.append(str(slots[key]))
        hour = slots.get("hour")
        if hour is not None:
            chips.append(f"{int(hour):02d}시")
        for text in chips:
            label = QLabel(text)
            label.setObjectName("queryCondition")
            self._conditions.addWidget(label)
        self._conditions.addStretch()
        self.conditions = chips

    def _set_evidence(self, facts: list[dict]) -> None:
        """근거 한 건을 한 줄씩. 근거가 없으면 그 사실을 적는다."""
        self._clear(self._evidence_rows)
        self.evidence_texts.clear()
        if not facts:
            self._add_evidence("안내", "이번 질의에는 남길 근거가 없습니다.")
        for fact in facts:
            kind = str(fact.get("kind") or "")
            self._add_evidence(
                KIND_LABELS.get(kind, kind or "근거"), fact_summary(fact) or "값이 없습니다."
            )
        self.evidence_card.setVisible(True)

    def _add_evidence(self, kind: str, text: str) -> None:
        row = self._evidence_rows.rowCount()
        tag = QLabel(kind)
        tag.setObjectName("queryEvidenceKind")
        tag.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body = QLabel(text)
        body.setObjectName("queryEvidenceText")
        body.setWordWrap(True)
        self._evidence_rows.addWidget(
            tag, row, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._evidence_rows.addWidget(body, row, 1, Qt.AlignmentFlag.AlignTop)
        self.evidence_texts.append(body)

    def _set_gaps(self, texts: list[str]) -> None:
        """계산하지 못한 항목. 없으면 카드를 숨긴다."""
        self._clear(self._gap_rows)
        self.gap_texts.clear()
        for text in dict.fromkeys(str(item).strip() for item in texts if str(item).strip()):
            label = QLabel(f"· {text}")
            label.setObjectName("queryGapText")
            label.setWordWrap(True)
            self._gap_rows.addWidget(label)
            self.gap_texts.append(label)
        self.gap_card.setVisible(bool(self.gap_texts))

    @staticmethod
    def _clear(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # 자리에서 바로 떼어 낸다. deleteLater 만 부르면 새 카드가 붙기 전까지
                # 옛 글자가 같은 자리에 남아 겹쳐 보인다.
                widget.setParent(None)
                widget.deleteLater()

    # ---------- 모드 ----------

    def apply_theme(self, mode: str = DEFAULT_MODE) -> None:
        """화면 모드를 기록한다 (카드·글자 색은 QSS 가 함께 바뀐다)."""
        self._mode = str(mode)
        self.update()

    @property
    def mode(self) -> str:
        return self._mode

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_context()  # 화면을 열 때마다 지금 고른 역을 다시 읽는다


def _unavailable_texts(bundle: dict) -> list[str]:
    """계산하지 못한 항목의 사유 문장들 (겹치면 한 번만)."""
    texts: list[str] = []
    for item in bundle.get("unavailable") or []:
        human = str(item.get("human") or "").strip()
        if human:
            texts.append(human)
    policy = bundle.get("policy") or {}
    if policy.get("de_missing"):
        texts.append("D·E 모델은 대상일 전 실측이 없어 계산할 수 없습니다.")
    return texts