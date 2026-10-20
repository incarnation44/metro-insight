"""화면과 모델 사이의 약속: 예측 조건 한 행을 받아 A~E 모델의 예상 승차 인원을 돌려준다.

학습한 모델을 화면에 연결하려면 Predictor 를 구현해 StationPanel(predictor=...) 에 넘긴다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from metro_insight.model.evaluation import ModelScore

# 모델 이름 → 사용 데이터 (화면 표시 순서). A~C 는 머신러닝, D·E 는 시계열 딥러닝
MODELS = {
    "A": "과거 이용 패턴",
    "B": "과거 이용 패턴 + 시간/요일/공휴일",
    "C": "과거 이용 패턴 + 시간/요일/공휴일 + 날씨",
    "D": "과거 이용량 시계열 + 시간/요일/공휴일 + 날씨 (LSTM)",
    "E": "과거 이용량 시계열 + 시간/요일/공휴일 + 날씨 (GRU)",
}

# 시간(0~24) → hour_code(0~19). 24시는 24_01 버킷(19, 다음 날 00~04시 4시간).
TIME_CODE_BY_HOUR = {
    0: 19, 1: 19, 2: 19, 3: 19, 4: 0, 5: 0,
    6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7, 13: 8, 14: 9,
    15: 10, 16: 11, 17: 12, 18: 13, 19: 14, 20: 15, 21: 16, 22: 17,
    23: 18, 24: 19,
}

FEATS = {
    "A": ["lag_1d", "lag_7d", "mean_prev7d"],
    "B": ["lag_1d", "lag_7d", "mean_prev7d", "hour_code", "weekday",
          "is_weekend", "is_holiday", "station_code", "line_code"],
    "C": ["lag_1d", "lag_7d", "mean_prev7d", "hour_code", "weekday",
          "is_weekend", "is_holiday", "station_code", "line_code",
          "temperature", "rainfall", "snowfall"],
}

HOURS_PER_DAY = 20
CACHE_NAME = "boarding_history.npz"
# 저장소 루트. 앱을 어느 폴더에서 실행해도 models/·data/processed 를 찾게 한다.
ROOT = Path(__file__).resolve().parents[3]


class PredictionUnavailable(Exception):
    """예측에 필요한 역·이력이 없을 때. 화면은 이 메시지를 그대로 보여준다."""


@dataclass(frozen=True)
class PredictionInput:
    """역 정보 패널에서 고른 예측 조건 (모델 입력 한 행)."""

    date: str  # yyyy-mm-dd
    line: str  # "2호선"
    station: str
    hour: int  # 시간대 시작 시: 4(04~06시), 6~23, 24(다음 날 00~04시 4시간)
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

    def has_data(self, line: str, station: str) -> bool:
        """그 호선·역 조합의 학습 데이터가 있는지. 화면에서 고를 수 있는 호선을 가른다."""
        ...


class ABCPredictor:
    """A/B/C HistGradientBoosting 모델을 화면에 연결하는 구현."""

    def __init__(
        self,
        models_dir: str | Path = ROOT / "models",
        data_dir: str | Path = ROOT / "data" / "processed",
        metrics_path: str | Path = ROOT / "models" / "metrics_abc.json",
    ):
        self._models_dir = Path(models_dir)
        self._models: dict[str, object] = {}
        for name in ("A", "B", "C"):
            p = self._models_dir / f"model_{name}.joblib"
            if p.exists():
                self._models[name] = joblib.load(p)

        # station/line 코드 매핑 + 과거 승차 조회용 데이터
        self._station_code: dict[tuple[str, str], int] = {}
        self._line_code: dict[str, int] = {}
        self._station_index: dict[int, int] = {}
        self._dates: list[str] = []
        self._date_index: dict[str, int] = {}
        self._history: np.ndarray | None = None  # (역, 시간대 20, 날짜), float32, NaN=없음
        self._load_data(Path(data_dir))

        # scores
        self._scores: dict[str, ModelScore] = {}
        mp = Path(metrics_path)
        if mp.exists():
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            for name in ("A", "B", "C"):
                if name in m and "test" in m[name]:
                    self._scores[name] = ModelScore(
                        mae=m[name]["test"]["mae"], rmse=m[name]["test"]["rmse"]
                    )

    @staticmethod
    def _station_key(name: str) -> str:
        """지도 이름과 데이터 이름을 맞추는 별칭. '잠실(송파구청)' → '잠실'."""
        return re.sub(r"\(.*?\)", "", str(name)).strip()

    def _add_aliases(self) -> None:
        """데이터 역명의 괄호 별칭을 함께 등록한다 (지도는 '잠실', 데이터는 '잠실(송파구청)')."""
        for (line, station), code in list(self._station_code.items()):
            alias = self._station_key(station)
            if alias and alias != station:
                self._station_code.setdefault((line, alias), code)

    def _load_data(self, data_dir: Path) -> None:
        cache = self._models_dir / CACHE_NAME
        if cache.exists():
            try:
                self._load_cache(cache)
                return
            except (KeyError, ValueError, OSError):
                # 캐시가 깨졌거나 예전 형식이면 CSV로 되돌아간다.
                self._station_code.clear()
                self._line_code.clear()
                self._station_index.clear()
                self._dates.clear()
                self._date_index.clear()
                self._history = None

        frames = []
        for fn in ("train.csv.gz", "valid.csv.gz", "test.csv.gz"):
            p = data_dir / fn
            if p.exists():
                frames.append(pd.read_csv(p, usecols=["date", "station", "line",
                                                      "station_code", "line_code",
                                                      "hour_code", "boarding"]))
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        for _, r in df[["line", "station", "station_code"]].drop_duplicates().iterrows():
            self._station_code[(str(r["line"]), str(r["station"]))] = int(r["station_code"])
        for _, r in df[["line", "line_code"]].drop_duplicates().iterrows():
            self._line_code[str(r["line"])] = int(r["line_code"])
        self._build_history(df)
        self._add_aliases()

    def _load_cache(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as z:
            history = z["boarding"]
            if history.ndim != 3 or history.shape[1] != HOURS_PER_DAY:
                raise ValueError("캐시 형식이 다르다")
            self._history = history
            station_codes = [int(c) for c in z["station_codes"]]
            self._station_index = {c: i for i, c in enumerate(station_codes)}
            self._dates = [str(d) for d in z["dates"]]
            self._date_index = {d: i for i, d in enumerate(self._dates)}
            for line, station, code in zip(z["pair_lines"], z["pair_stations"],
                                           z["pair_codes"], strict=True):
                self._station_code[(str(line), str(station))] = int(code)
            for line, code in zip(z["line_names"], z["line_codes"], strict=True):
                self._line_code[str(line)] = int(code)
        self._add_aliases()

    def _build_history(self, df: pd.DataFrame) -> None:
        station_codes = np.sort(df["station_code"].unique())
        self._station_index = {int(c): i for i, c in enumerate(station_codes)}
        self._dates = [str(d) for d in np.sort(df["date"].unique())]
        self._date_index = {d: i for i, d in enumerate(self._dates)}
        history = np.full((len(station_codes), HOURS_PER_DAY, len(self._dates)),
                          np.nan, dtype=np.float32)
        si = pd.Categorical(df["station_code"], categories=station_codes).codes
        di = pd.Categorical(df["date"], categories=self._dates).codes
        history[si, df["hour_code"].to_numpy(), di] = df["boarding"].to_numpy()
        self._history = history

    def _hour_code(self, hour: int) -> int:
        return TIME_CODE_BY_HOUR.get(int(hour), 19)

    def _lags(self, station_code: int, hour_code: int, date_str: str) -> tuple:
        if self._history is None:
            return (None, None, None)
        si = self._station_index.get(int(station_code))
        di = self._date_index.get(str(date_str))
        if si is None or di is None:
            return (None, None, None)

        def value(days_ago: int) -> float | None:
            j = di - days_ago
            if j < 0:
                return None
            v = self._history[si, hour_code, j]
            return None if np.isnan(v) else float(v)

        lag1 = value(1)
        lag7 = value(7)
        # 학습(전처리)과 같은 규칙: 직전 7일이 전부 있어야 평균을 낸다.
        vals = [value(i) for i in range(1, 8)]
        if any(v is None for v in vals):
            return (lag1, lag7, None)
        # 학습표(notebooks/preprocess.py)가 round(1) 한 값으로 학습했으므로 같은 자릿수로 맞춘다.
        return (lag1, lag7, round(float(np.mean(vals)), 1))

    def date_range(self) -> tuple[str, str] | None:
        """예측 가능한 날짜 범위 (데이터에 있는 날짜 기준)."""
        if not self._dates:
            return None
        return (self._dates[0], self._dates[-1])

    def _lookup(self, line: str, station: str) -> tuple[int | None, int | None]:
        """(station_code, line_code). 지도 이름('잠실')은 데이터 이름('잠실(송파구청)') 별칭으로도 찾는다."""
        sc = self._station_code.get((line, station))
        if sc is None:
            sc = self._station_code.get((line, self._station_key(station)))
        return sc, self._line_code.get(line)

    def has_data(self, line: str, station: str) -> bool:
        """그 호선·역 조합이 학습 데이터에 있는지 (호선 콤보 표시용)."""
        sc, lc = self._lookup(str(line), str(station))
        return sc is not None and lc is not None

    def predict(self, values: PredictionInput) -> dict[str, float]:
        line = str(values.line)
        station = str(values.station)
        sc, lc = self._lookup(line, station)
        if sc is None or lc is None:
            raise PredictionUnavailable(
                f"{station}({line}) 역은 학습 데이터에 없어 예측할 수 없습니다."
            )

        hc = self._hour_code(values.hour)
        lag1, lag7, mean7 = self._lags(sc, hc, values.date)
        if mean7 is None:
            if self._dates:
                period = f"데이터 기간({self._dates[0]} ~ {self._dates[-1]}) 안의 날짜를 골라 주세요."
            else:
                period = "데이터가 로드되지 않았습니다."
            raise PredictionUnavailable(
                f"{values.date} 기준 직전 7일 승차 이력이 없어 예측할 수 없습니다. {period}"
            )

        base_row = {
            "lag_1d": lag1,
            "lag_7d": lag7,
            "mean_prev7d": mean7,
            "hour_code": hc,
            "weekday": values.weekday,
            "is_weekend": bool(values.is_weekend),
            "is_holiday": bool(values.is_holiday),
            "station_code": sc,
            "line_code": lc,
            "temperature": values.temperature,
            "rainfall": int(bool(values.rainfall)),
            "snowfall": int(bool(values.snowfall)),
        }

        out: dict[str, float] = {}
        for name, model in self._models.items():
            if name == "C" and values.temperature is None:
                continue  # 기온이 없으면 날씨 모델 C는 표시하지 않는다.
            cols = FEATS[name]
            row = {c: (np.nan if base_row[c] is None else base_row[c]) for c in cols}
            X = pd.DataFrame([row], columns=cols)
            try:
                pred = model.predict(X)[0]
                out[name] = float(pred)
            except Exception:
                continue
        return out

    def scores(self) -> dict[str, ModelScore]:
        return dict(self._scores)
