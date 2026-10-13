"""화면과 모델 사이의 약속: 예측 조건 한 행을 받아 A~E 모델의 예상 승차 인원을 돌려준다.

학습한 모델을 화면에 연결하려면 Predictor 를 구현해 StationPanel(predictor=...) 에 넘긴다.
"""
from dataclasses import dataclass
from typing import Protocol

from metro_insight.model.evaluation import ModelScore

# 모델 이름 → 사용 데이터 (화면 표시 순서). A~C 는 머신러닝, D·E 는 시계열 딥러닝
MODELS = {
    "A": "과거 이용 패턴",
    "B": "과거 이용 패턴 + 시간/요일/공휴일",
    "C": "과거 이용 패턴 + 시간/요일/공휴일 + 날씨",
    "D": "과거 이용량 시계열 + 시간/요일/공휴일 + 날씨 (LSTM)",
    "E": "과거 이용량 시계열 + 시간/요일/공휴일 + 날씨 (GRU)",
}


@dataclass(frozen=True)
class PredictionInput:
    """역 정보 패널에서 고른 예측 조건 (모델 입력 한 행)."""

    date: str  # yyyy-mm-dd
    line: str  # "2호선"
    station: str
    hour: int
    weekday: int  # 0=월 ~ 6=일
    is_weekend: bool
    is_holiday: bool
    temperature: float | None  # ℃, 입력하지 않으면 None
    rainfall: bool  # 강수 여부, 눈도 강수에 포함
    snowfall: bool


class Predictor(Protocol):
    def predict(self, values: PredictionInput) -> dict[str, float]:
        """모델 이름(A~E) → 예상 승차 인원. 예측하지 못한 모델은 빼고 돌려준다."""
        ...

    def scores(self) -> dict[str, ModelScore]:
        """모델 이름(A~E) → Test 데이터 평가 결과."""
        ...
