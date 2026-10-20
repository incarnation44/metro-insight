"""화면과 모델 사이의 약속: 예측 조건 한 행을 받아 A~C 달력 모델의 예상 승차 인원을 돌려준다.

달력 모델은 '역·시간대·월 프로파일'(그 역·시간대의 월별 평균 승차 인원과 관측 수)과
달력 값(요일·주말·공휴일), 날씨 값으로 예측한다. 프로파일은 두 갈래로 만든다.

* 데이터 끝(2026-06-30) 이후 날짜 → models/cal_profiles.npz 의 전체 이력 프로파일을 그대로 쓴다.
* 데이터 안 날짜 → models/boarding_history.npz 에서 '그 날짜 이전' 자료만으로 다시 계산한다.
  (그 시점에 알 수 있었던 값만 쓴다 — 같은 날 값이 섞이면 성적이 실제보다 좋아진다)

날씨를 입력하지 않으면 기온은 월별 평년값, 비·눈은 '안 옴'으로 두고 C 모델도 항상 계산한다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from metro_insight.model.evaluation import ModelScore

# 모델 이름 → 사용 데이터 (화면 표시 순서). A~C 는 달력 모델, D·E 는 시계열 딥러닝
MODELS = {
    "A": "역·시간대·월 프로파일",
    "B": "역·시간대·월 프로파일 + 시간/요일/공휴일",
    "C": "역·시간대·월 프로파일 + 시간/요일/공휴일 + 날씨",
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

# 학습과 같은 feature 계약. 모델에 feature_names_in_ 이 있으면 그 순서를 먼저 믿는다.
FEATS = {
    "A": ["profile_month_mean", "profile_n", "station_code", "line_code"],
    "B": ["profile_month_mean", "profile_n", "station_code", "line_code",
          "profile_weekday_mean", "profile_weekday_n",
          "hour_code", "weekday", "is_weekend", "is_holiday"],
    "C": ["profile_month_mean", "profile_n", "station_code", "line_code",
          "profile_weekday_mean", "profile_weekday_n",
          "hour_code", "weekday", "is_weekend", "is_holiday",
          "temperature", "rainfall", "snowfall"],
}

HOURS_PER_DAY = 20
CACHE_NAME = "boarding_history.npz"
CACHE_VERSION = 2  # 캐시 형식 버전 (원본이 바뀌면 앱이 캐시를 버리고 CSV 를 읽는다)
PROFILES_NAME = "cal_profiles.npz"  # 서빙용 프로파일 (2024-01-08 ~ 2026-06-30 전체 이력)
MODEL_PREFIX = "cal_"  # cal_A.joblib · cal_B.joblib · cal_C.joblib
DATA_END = date(2026, 6, 30)  # 학습·프로파일 기준 마지막 날 — 이후는 장기 추정
SERVING_START = date(2024, 1, 8)
# 예측 가능 범위 = 오늘부터 6개월(182일) 뒤까지. 모듈을 읽는 시점에 계산한다.
SERVING_END = date.today() + timedelta(days=182)

# 프로파일 키 이름 (학습이 저장하는 npz 배열 이름 그대로)
PROFILE_MONTH_KEYS = ("station_code", "hour_code", "month", "mean", "count")
PROFILE_WEEKDAY_KEYS = ("station_code_wd", "hour_code_wd", "weekday_wd", "mean_wd", "count_wd")

# 예측 지평 구간 (데이터 끝 2026-06-30 기준 며칠 뒤인지). 마지막 구간은 끝이 없다.
HORIZON_BUCKETS = ("1-31", "32-91", "92-181", "182-366", "367+")
HORIZON_BUCKET_ENDS = (31, 91, 181, 366, None)
HORIZON_BUCKET_LABELS = {
    "1-31": "1~31일",
    "32-91": "32~91일",
    "92-181": "92~181일",
    "182-366": "182~366일",
    "367+": "367일 이상",
}
# Test 로 실제 오차를 재 본 구간. 182일 이후는 성적이 없어 ± 를 붙이지 않고 미검증으로 표시한다.
VERIFIED_HORIZON_BUCKETS = ("1-31", "32-91", "92-181")

# 서울 평년 기온(℃, 월별 · 기상청 1991~2020 평년값 근사). 관측 기온이 없을 때 쓴다.
MONTHLY_TEMPERATURE_NORMALS = {
    1: -1.9, 2: 0.6, 3: 6.1, 4: 12.6, 5: 18.2, 6: 22.7,
    7: 25.3, 8: 26.1, 9: 21.6, 10: 14.8, 11: 7.6, 12: 0.0,
}

# 저장소 루트. 앱을 어느 폴더에서 실행해도 models/·data/processed 를 찾게 한다.
ROOT = Path(__file__).resolve().parents[3]


class PredictionUnavailable(Exception):
    """예측에 필요한 역·이력이 없을 때. 화면은 이 메시지를 그대로 보여준다."""


def climatology_temperature(month: int) -> float:
    """그 달의 서울 평년 기온(℃). 미래 날짜에 관측 기온이 없을 때 쓴다."""
    try:
        return MONTHLY_TEMPERATURE_NORMALS[int(month)]
    except (KeyError, TypeError, ValueError):
        return MONTHLY_TEMPERATURE_NORMALS[1]


def horizon_bucket(when: date | str) -> str | None:
    """데이터 끝(2026-06-30)에서 며칠 뒤인지 → 지평 구간 이름 ('1-31', '367+' …).

    데이터 안(과거) 날짜는 이미 관측된 구간이라 지평이 없다 → None.
    """
    try:
        day = date.fromisoformat(str(when)[:10])
    except ValueError:
        return None
    days = (day - DATA_END).days
    if days < 1:
        return None
    for name, end in zip(HORIZON_BUCKETS, HORIZON_BUCKET_ENDS, strict=True):
        if end is None or days <= end:
            return name
    return HORIZON_BUCKETS[-1]


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
    temperature: float | None  # ℃, 입력하지 않으면 None → 평년값
    rainfall: bool | None  # 강수 여부, 눈도 강수에 포함 (None 은 '안 옴')
    snowfall: bool | None


class Predictor(Protocol):
    def predict(self, values: PredictionInput) -> dict[str, object]:
        """모델 이름(A~C) → 예상 승차 인원 + 지평 정보.

        예측하지 못한 모델은 빼고 돌려준다.
        함께 담는 값: ``horizon_bucket``(구간 이름) · ``horizon_scores``(모델 → 구간 MAE/RMSE).
        """
        ...

    def scores(self) -> dict[str, ModelScore]:
        """모델 이름(A~C) → Test 데이터 평가 결과."""
        ...

    def has_data(self, line: str, station: str) -> bool:
        """그 호선·역 조합의 학습 데이터가 있는지. 화면에서 고를 수 있는 호선을 가른다."""
        ...


def _to_score(value: object) -> ModelScore | None:
    """{'mae':…, 'rmse':…} 또는 [mae, rmse] → ModelScore. 형식이 다르면 None."""
    if isinstance(value, dict):
        mae, rmse = value.get("mae"), value.get("rmse")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        mae, rmse = value
    else:
        return None
    try:
        return ModelScore(mae=float(mae), rmse=float(rmse))
    except (TypeError, ValueError):
        return None


def _model_entries(data: dict) -> dict[str, dict]:
    """metric 파일에서 모델별 지표 dict 를 찾는다 (valid/test/overall 블록 또는 최상위 모델 키)."""
    found: dict[str, dict] = {}
    for block_name in ("test", "valid", "overall", "holdout"):
        block = data.get(block_name)
        if not isinstance(block, dict):
            continue
        for model in ("A", "B", "C"):
            entry = block.get(model)
            if isinstance(entry, dict):
                found.setdefault(model, entry)
    for model in ("A", "B", "C"):
        entry = data.get(model)
        if isinstance(entry, dict):
            found.setdefault(model, entry)
    return found


def _parse_overall_scores(data: dict) -> dict[str, ModelScore]:
    """metrics_calendar.json · metrics_abc.json 의 전체 평가 성적을 읽는다."""
    out: dict[str, ModelScore] = {}
    for model, entry in _model_entries(data).items():
        score = _to_score(entry)
        if score is None:
            for key in ("test", "valid", "overall", "holdout"):
                score = _to_score(entry.get(key))
                if score is not None:
                    break
        if score is not None:
            out[model] = score
    return out


_BUCKET_NAME = re.compile(r"^\d{1,3}(?:-\d{1,3})?\+?$")


def _is_bucket(name: object) -> bool:
    """지평 구간 이름인지 ('1-31' · '182-366' · '367+')."""
    return bool(_BUCKET_NAME.match(str(name).strip()))


def _collect_horizon_block(block: dict, out: dict[str, dict[str, ModelScore]]) -> None:
    """한 블록에서 지평 구간 점수를 모은다. {모델: {구간: 점수}} · {구간: {모델: 점수}} 둘 다 받는다."""
    for model in ("A", "B", "C"):
        entry = block.get(model)
        if not isinstance(entry, dict):
            continue
        nested = entry.get("horizon")
        table = nested if isinstance(nested, dict) else entry
        for bucket, item in table.items():
            if not _is_bucket(bucket):
                continue  # 'best'·'train' 처럼 구간이 아닌 키는 버린다
            score = _to_score(item)
            if score is not None:
                out.setdefault(str(bucket), {})[model] = score
    for bucket, entry in block.items():
        if not _is_bucket(bucket) or not isinstance(entry, dict):
            continue
        for model in ("A", "B", "C"):
            score = _to_score(entry.get(model))
            if score is not None:
                out.setdefault(str(bucket), {})[model] = score


def _parse_horizon_scores(data: dict) -> dict[str, dict[str, ModelScore]]:
    """지평 구간별 MAE/RMSE 를 읽는다.

    * evaluate_calendar.py 가 쓰는 모양: test.{모델}.horizon.{구간} = {mae, rmse, n}
    * train_calendar.py 가 쓰는 모양: valid.{모델}.horizon.{구간} = {mae, rmse, n}
    * 백테스트(notebooks/backtest_calendar.py)가 넣는 최상위 horizon 블록 (182-366 근사값)
    모두 같은 사전({구간: {모델: 점수}})으로 합친다.

    화면의 전체 ±MAE(``scores()``)가 test 기준이므로 구간 점수도 **test 우선**으로 읽고,
    test 에 없는 모델·구간만 valid 로 채운다(fallback). 뒤에 읽는 블록이 앞의 값을 덮어쓴다.
    """
    out: dict[str, dict[str, ModelScore]] = {}
    # 최상위가 곧 {모델: {구간: …}} 인 경우 — 우선순위 가장 낮음
    _collect_horizon_block(data, out)
    # 낮은 우선순위 → 높은 우선순위 순서
    for block_name in ("horizon_buckets", "horizon", "buckets", "horizon_scores",
                       "overall", "holdout", "valid", "test"):
        block = data.get(block_name)
        if isinstance(block, dict):
            _collect_horizon_block(block, out)
    return out


def _parse_feature_lists(data: dict) -> dict[str, list[str]]:
    """metric 파일에 적힌 feature 목록 (모델 객체에 feature_names_in_ 이 없을 때 대신 쓴다)."""
    raw = data.get("features") or data.get("feature_lists") or {}
    out: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for model in ("A", "B", "C"):
            cols = raw.get(model)
            if isinstance(cols, (list, tuple)) and cols:
                out[model] = [str(c) for c in cols]
    for model in ("A", "B", "C"):
        entry = data.get(model)
        if model in out or not isinstance(entry, dict):
            continue
        cols = entry.get("features")
        if isinstance(cols, (list, tuple)) and cols:
            out[model] = [str(c) for c in cols]
    return out


class ABCPredictor:
    """A/B/C 달력 모델(HistGradientBoosting)을 화면에 연결하는 구현."""

    def __init__(
        self,
        models_dir: str | Path = ROOT / "models",
        data_dir: str | Path = ROOT / "data" / "processed",
        metrics_path: str | Path | None = None,
    ):
        self._models_dir = Path(models_dir)
        self._models: dict[str, object] = {}
        self._last_errors: dict[str, str] = {}  # 최근 예측에서 계산에 실패한 모델과 이유
        for name in ("A", "B", "C"):
            p = self._models_dir / f"{MODEL_PREFIX}{name}.joblib"
            if p.exists():
                self._models[name] = joblib.load(p)

        # station/line 코드 매핑 + 과거 승차 조회용 데이터
        self._station_code: dict[tuple[str, str], int] = {}
        self._line_code: dict[str, int] = {}
        self._station_index: dict[int, int] = {}
        self._dates: list[str] = []
        self._month_cols: dict[int, np.ndarray] = {}  # 월 → 날짜 열 번호 (as-of 프로파일 계산용)
        self._weekday_cols: dict[int, np.ndarray] = {}  # 요일 → 날짜 열 번호
        self._weekday_array = np.asarray([], dtype=np.int64)  # self._dates 의 요일 (0=월)
        self._date_array = np.asarray([], dtype=str)
        self._history: np.ndarray | None = None  # (역, 시간대 20, 날짜), float32, NaN=없음
        self._load_data(Path(data_dir))

        # 서빙용 프로파일 (역, 시간대, 월) → (월평균, 관측 수) · (역, 시간대, 요일) → (요일평균, 관측 수)
        self._profiles: dict[tuple[int, int, int], tuple[float, int]] = {}
        self._profiles_wd: dict[tuple[int, int, int], tuple[float, int]] = {}
        self._load_profiles(self._models_dir / PROFILES_NAME)

        # scores
        self._scores: dict[str, ModelScore] = {}
        self._horizon_scores: dict[str, dict[str, ModelScore]] = {}
        self._metric_features: dict[str, list[str]] = {}
        self._load_metrics(metrics_path)

    # ---------- 파일 읽기 ----------

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

    def _csv_mtime(self) -> int:
        """원본 CSV 의 가장 최근 수정 시각(초). CSV 를 다시 만들면 값이 달라진다."""
        mtimes = []
        for name in ("train.csv.gz", "valid.csv.gz", "test.csv.gz"):
            p = self._data_dir / name
            if p.exists():
                mtimes.append(int(p.stat().st_mtime))
        return max(mtimes) if mtimes else 0

    def _load_profiles(self, path: Path) -> None:
        """달력 프로파일을 읽는다.

        * (역, 시간대, 월) → 월평균·관측 수          … A·B·C 의 profile_month_mean/profile_n
        * (역, 시간대, 요일) → 요일평균·관측 수      … B·C 의 profile_weekday_mean/profile_weekday_n
        """
        if not path.exists():
            return
        try:
            with np.load(path, allow_pickle=False) as z:
                self._profiles = self._read_profile_dict(z, PROFILE_MONTH_KEYS)
                self._profiles_wd = self._read_profile_dict(z, PROFILE_WEEKDAY_KEYS)
        except (KeyError, OSError, ValueError):
            self._profiles = {}
            self._profiles_wd = {}

    @staticmethod
    def _read_profile_dict(z, keys: tuple[str, ...]) -> dict[tuple[int, int, int], tuple[float, int]]:
        """npz 에서 (코드, 시간대, 몫) → (평균, 관측 수) 사전을 만든다. 평균이 NaN 인 키는 뺀다."""
        arrays = [np.asarray(z[k]) for k in keys]
        out: dict[tuple[int, int, int], tuple[float, int]] = {}
        for a, b, c, mean, count in zip(*arrays, strict=True):
            if np.isnan(mean) or count <= 0:
                continue
            out[(int(a), int(b), int(c))] = (float(mean), int(count))
        return out

    def _metrics_candidates(self, metrics_path: str | Path | None) -> list[Path]:
        """metrics 파일 후보: 지정 경로 → models/metrics_calendar.json → 예전 metrics_abc.json."""
        models_dir = self._models_dir
        given = (Path(metrics_path) if metrics_path is not None
                 else models_dir / "metrics_calendar.json")
        paths = [
            given,
            models_dir / "metrics_calendar.json",
            models_dir / "metrics_abc.json",
        ]
        # 기본 models/ 를 쓰는 앱 실행일 때만 저장소의 metrics 로 되돌아간다 (테스트 격리)
        if models_dir.resolve() == (ROOT / "models").resolve():
            paths += [ROOT / "models" / "metrics_calendar.json", ROOT / "models" / "metrics_abc.json"]
        seen: set[str] = set()
        out: list[Path] = []
        for path in paths:
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
        return out

    def _load_metrics(self, metrics_path: str | Path | None) -> None:
        for path in self._metrics_candidates(metrics_path):
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            self._scores = _parse_overall_scores(data)
            self._horizon_scores = _parse_horizon_scores(data)
            self._metric_features = _parse_feature_lists(data)
            if self._scores or self._horizon_scores:
                break

    def _load_data(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        cache = self._models_dir / CACHE_NAME
        if cache.exists():
            try:
                self._load_cache(cache)
                return
            except (KeyError, ValueError, OSError):
                # 캐시가 깨졌거나 예전 형식이면 CSV로 되돌아간다.
                self._reset_data()

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

    def _reset_data(self) -> None:
        self._station_code.clear()
        self._line_code.clear()
        self._station_index.clear()
        self._dates.clear()
        self._month_cols.clear()
        self._weekday_cols.clear()
        self._weekday_array = np.asarray([], dtype=np.int64)
        self._date_array = np.asarray([], dtype=str)
        self._history = None

    def _load_cache(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as z:
            history = z["boarding"]
            if history.ndim != 3 or history.shape[1] != HOURS_PER_DAY:
                raise ValueError("캐시 형식이 다르다")
            if int(z["version"]) != CACHE_VERSION:
                raise ValueError("캐시 버전이 다르다")
            if int(z["csv_mtime"]) != self._csv_mtime():
                raise ValueError("원본 CSV 가 바뀌었다 (캐시가 오래됨)")
            self._history = history
            station_codes = [int(c) for c in z["station_codes"]]
            self._station_index = {c: i for i, c in enumerate(station_codes)}
            self._dates = [str(d) for d in z["dates"]]
            for line, station, code in zip(z["pair_lines"], z["pair_stations"],
                                           z["pair_codes"], strict=True):
                self._station_code[(str(line), str(station))] = int(code)
            for line, code in zip(z["line_names"], z["line_codes"], strict=True):
                self._line_code[str(line)] = int(code)
        self._build_date_groups()
        self._add_aliases()

    def _build_history(self, df: pd.DataFrame) -> None:
        station_codes = np.sort(df["station_code"].unique())
        self._station_index = {int(c): i for i, c in enumerate(station_codes)}
        self._dates = [str(d) for d in np.sort(df["date"].unique())]
        history = np.full((len(station_codes), HOURS_PER_DAY, len(self._dates)),
                          np.nan, dtype=np.float32)
        si = pd.Categorical(df["station_code"], categories=station_codes).codes
        di = pd.Categorical(df["date"], categories=self._dates).codes
        history[si, df["hour_code"].to_numpy(), di] = df["boarding"].to_numpy()
        self._history = history
        self._build_date_groups()

    def _build_date_groups(self) -> None:
        """날짜 열을 월·요일별로 묶어 둔다. as-of 프로파일을 그 조건 자료로만 계산할 때 쓴다."""
        self._date_array = np.asarray(self._dates, dtype=str)
        self._weekday_array = np.full(len(self._dates), -1, dtype=np.int64)
        month_cols: dict[int, list[int]] = {}
        weekday_cols: dict[int, list[int]] = {}
        for i, d in enumerate(self._dates):
            try:
                day = date.fromisoformat(str(d)[:10])
            except ValueError:
                continue
            month_cols.setdefault(day.month, []).append(i)
            self._weekday_array[i] = day.weekday()
            weekday_cols.setdefault(day.weekday(), []).append(i)
        self._month_cols = {m: np.asarray(c, dtype=np.int64) for m, c in month_cols.items()}
        self._weekday_cols = {w: np.asarray(c, dtype=np.int64) for w, c in weekday_cols.items()}

    def _asof_values(self, station_code: int, hour_code: int, cols: np.ndarray | None,
                     limit: int) -> tuple[float, int]:
        """그 날짜 이전(열 < limit) 자료의 평균·표본수. 없으면 (NaN, 0) — 학습과 같은 결측 규칙."""
        si = self._station_index.get(int(station_code))
        if si is None or self._history is None or cols is None:
            return (float("nan"), 0)
        idx = cols[cols < limit]
        if not idx.size:
            return (float("nan"), 0)
        values = self._history[si, int(hour_code), idx]
        values = values[~np.isnan(values)]
        if not values.size:
            return (float("nan"), 0)
        return (float(values.mean()), int(values.size))

    # ---------- 프로파일 ----------

    def _profile(self, station_code: int, hour_code: int, month: int) -> tuple[float, int]:
        """(역, 시간대, 월) 서빙 프로파일 (데이터 끝 이후 날짜). 없으면 (NaN, 0) — 학습과 같은 결측 규칙."""
        hit = self._profiles.get((int(station_code), int(hour_code), int(month)))
        return hit if hit is not None else (float("nan"), 0)

    def _profile_weekday(self, station_code: int, hour_code: int, weekday: int) -> tuple[float, int]:
        """(역, 시간대, 요일) 서빙 프로파일 (데이터 끝 이후 날짜). 없으면 (NaN, 0)."""
        hit = self._profiles_wd.get((int(station_code), int(hour_code), int(weekday)))
        return hit if hit is not None else (float("nan"), 0)

    def _asof_limit(self, day: date) -> int:
        """그 날짜보다 앞선 날짜 열의 개수 (그 날짜 자신은 제외)."""
        if not self._dates:
            return 0
        return int(np.searchsorted(self._date_array, day.isoformat(), side="left"))

    def _asof_profile(self, station_code: int, hour_code: int, day: date) -> tuple[float, int]:
        """그 날짜 이전 + 같은 달 자료만으로 만든 프로파일 (데이터 안 날짜)."""
        return self._asof_values(station_code, hour_code,
                                 self._month_cols.get(day.month), self._asof_limit(day))

    def _asof_profile_weekday(self, station_code: int, hour_code: int, weekday: int,
                              day: date) -> tuple[float, int]:
        """그 날짜 이전 + 같은 요일 자료만으로 만든 프로파일 (데이터 안 날짜)."""
        return self._asof_values(station_code, hour_code,
                                 self._weekday_cols.get(int(weekday)), self._asof_limit(day))

    # ---------- 화면에 쓰는 값 ----------

    def _hour_code(self, hour: int) -> int:
        return TIME_CODE_BY_HOUR.get(int(hour), 19)

    def date_range(self) -> tuple[str, str] | None:
        """예측 가능한 날짜 범위. 데이터가 없어도 서빙 기간은 항상 같다 (오늘부터 6개월)."""
        return (SERVING_START.isoformat(), SERVING_END.isoformat())

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

    def _feature_columns(self, name: str, model: object) -> list[str]:
        """모델이 학습한 feature 순서. 없으면 metrics 파일 → FEATS 순서."""
        trained = getattr(model, "feature_names_in_", None)
        if trained is not None:
            return [str(c) for c in trained]
        return list(self._metric_features.get(name) or FEATS[name])

    def predict(self, values: PredictionInput) -> dict[str, object]:
        line = str(values.line)
        station = str(values.station)
        self._last_errors = {}
        sc, lc = self._lookup(line, station)
        if sc is None or lc is None:
            raise PredictionUnavailable(
                f"{station}({line}) 역은 학습 데이터에 없어 예측할 수 없습니다."
            )
        if not self._models:
            raise PredictionUnavailable(
                "학습된 달력 모델이 없어 예측할 수 없습니다. models/cal_*.joblib 을 확인하세요."
            )

        try:
            day = date.fromisoformat(str(values.date)[:10])
        except ValueError as exc:
            raise PredictionUnavailable(f"날짜 형식을 읽을 수 없습니다: {values.date}") from exc

        if day > SERVING_END:
            raise ValueError(
                f"예측 가능 범위는 {SERVING_END.isoformat()}까지입니다. (오늘부터 6개월)"
            )

        hc = self._hour_code(values.hour)
        weekday = int(values.weekday)
        future = day > DATA_END
        if future:
            if not self._profiles:
                raise PredictionUnavailable(
                    f"미래 날짜용 프로파일(models/{PROFILES_NAME})이 없어 예측할 수 없습니다."
                )
            profile_mean, profile_n = self._profile(sc, hc, day.month)
            wd_mean, wd_n = self._profile_weekday(sc, hc, weekday)
        else:
            if self._history is None or int(sc) not in self._station_index:
                raise PredictionUnavailable(
                    f"{station}({line}) {values.hour}시 기준 승차 이력이 없어 예측할 수 없습니다."
                )
            profile_mean, profile_n = self._asof_profile(sc, hc, day)
            wd_mean, wd_n = self._asof_profile_weekday(sc, hc, weekday, day)

        temperature = (float(values.temperature) if values.temperature is not None
                       else climatology_temperature(day.month))
        row = {
            "profile_month_mean": profile_mean,
            "profile_n": profile_n,
            "profile_weekday_mean": wd_mean,
            "profile_weekday_n": wd_n,
            "station_code": sc,
            "line_code": lc,
            "hour_code": hc,
            "weekday": weekday,
            "is_weekend": int(bool(values.is_weekend)),
            "is_holiday": int(bool(values.is_holiday)),
            "temperature": temperature,
            "rainfall": int(bool(values.rainfall)),
            "snowfall": int(bool(values.snowfall)),
        }

        out: dict[str, object] = {}
        for name, model in self._models.items():
            try:
                cols = self._feature_columns(name, model)
                rows = [{c: row[c] for c in cols}]
                X = pd.DataFrame(rows, columns=cols)
                out[name] = float(model.predict(X)[0])
            except Exception as exc:  # 실패한 모델은 조용히 넘어가지 않고 이유를 남긴다
                self._last_errors[name] = f"{type(exc).__name__}: {exc}"
                continue

        # 지평은 미래 날짜에만 있다 — 과거 날짜는 실측이 있으므로 지평 지표를 붙이지 않는다.
        bucket = None if not future else horizon_bucket(day)
        out["horizon_bucket"] = bucket
        out["horizon_scores"] = {} if bucket is None else dict(self._horizon_scores.get(bucket, {}))
        return out

    def horizon_scores(self, when: date | str) -> dict[str, ModelScore]:
        """그 날짜가 속한 지평 구간의 모델별 MAE/RMSE. 과거(데이터 안) 날짜면 빈 dict."""
        bucket = horizon_bucket(when)
        if bucket is None:
            return {}
        return dict(self._horizon_scores.get(bucket, {}))

    @property
    def last_errors(self) -> dict[str, str]:
        """최근 predict() 에서 계산에 실패한 모델 → 실패 이유. 화면이 이걸 보여 준다."""
        return dict(self._last_errors)

    def scores(self) -> dict[str, ModelScore]:
        return dict(self._scores)
