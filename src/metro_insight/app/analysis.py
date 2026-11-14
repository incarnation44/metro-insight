"""분석 화면에 보여 줄 숫자를 모델 평가 파일에서 읽는다.

여기서는 새로 계산하지 않는다. 학습할 때 만들어 둔 평가 기록(metrics_calendar.json ·
metrics_sequence.json · backtest_182_366.json)을 그대로 읽어 화면에 넘길 형태로 정리만 한다.
숫자를 코드에 적어 두면 모델을 다시 학습했을 때 화면만 옛 값으로 남기 때문이다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CALENDAR_NAME = "metrics_calendar.json"  # A·B·C (달력 피처 모델)
SEQUENCE_NAME = "metrics_sequence.json"  # D·E (시계열 모델)
BACKTEST_NAME = "backtest_182_366.json"  # 182일 이후 구간 백테스트

CALENDAR_MODELS = ("A", "B", "C")
SEQUENCE_MODELS = ("D", "E")
SHORT_HORIZON = "1-31"
LONG_HORIZON = "92-181"
FAR_HORIZON = "182-366"  # 백테스트로만 잰 구간 (182~362일 실측)
HORIZON_LABELS = {
    "1-31": "1~31일",
    "32-91": "32~91일",
    "92-181": "92~181일",
    FAR_HORIZON: "182~366일",
}
TERTILES = ("소", "중", "대")
HEAT_STEPS = (
    "#eef2f7",
    "#d7e0ec",
    "#b7c8dd",
    "#8fa8c6",
    "#6486ab",
    "#3d6288",
)  # 한 가지 색의 6단계 (옅음 → 진함)


@dataclass(frozen=True)
class ModelStat:
    """모델 하나의 Test 구간 성적."""

    name: str
    mae: float
    rmse: float


@dataclass(frozen=True)
class AnalysisData:
    """분석 화면이 쓰는 숫자 전부. 없는 값은 빈 dict 로 두고 화면에서 감춘다."""

    models: tuple[ModelStat, ...]  # MAE 가 작은 순
    horizons: tuple[str, ...]  # ("1-31", "32-91", "92-181", "182-366")
    horizon_mae: dict[str, dict[str, float]]  # 모델 → 지평 → MAE
    tertiles: dict[str, dict[str, float]]  # 모델(D·E) → 소/중/대 → MAE
    cut_dates: tuple[str, ...]  # 백테스트 기준일
    cut_mae: dict[str, dict[str, float]]  # 기준일 → 모델 → MAE
    pooled: dict[str, float]  # 182~366일 전체를 합친 모델별 MAE
    sources: dict[str, str]  # 읽은 파일 (안내용)

    def mae_of(self, model: str) -> float | None:
        for stat in self.models:
            if stat.name == model:
                return stat.mae
        return None

    def best(self) -> ModelStat | None:
        return self.models[0] if self.models else None

    def value_at(self, model: str, horizon: str) -> float | None:
        return self.horizon_mae.get(model, {}).get(horizon)

    def heat_range(self) -> tuple[float, float]:
        values = [value for row in self.horizon_mae.values() for value in row.values()]
        if not values:
            return 0.0, 0.0
        return min(values), max(values)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _horizon_rows(block: dict, model: str) -> dict[str, float]:
    """한 모델의 지평 구간별 MAE. 기록이 없는 구간은 넣지 않는다."""
    out: dict[str, float] = {}
    for horizon, entry in ((block.get(model) or {}).get("horizon") or {}).items():
        mae = entry.get("mae") if isinstance(entry, dict) else None
        if isinstance(mae, (int, float)):
            out[str(horizon)] = float(mae)
    return out


def load_analysis(models_dir: str | Path | None = None) -> AnalysisData:
    """세 평가 파일을 읽어 한 덩어리로 만든다. 파일이 없으면 그 부분만 비운다."""
    directory = Path(models_dir) if models_dir is not None else ROOT / "models"
    calendar_path = directory / CALENDAR_NAME
    sequence_path = directory / SEQUENCE_NAME
    backtest_path = directory / BACKTEST_NAME
    calendar = _read(calendar_path)
    sequence = _read(sequence_path)
    backtest = _read(backtest_path)

    calendar_test = calendar.get("test") or {}
    sequence_test = sequence.get("test") or {}

    stats: list[ModelStat] = []
    horizon_mae: dict[str, dict[str, float]] = {}
    for name in CALENDAR_MODELS:
        entry = calendar_test.get(name) or {}
        mae, rmse = entry.get("mae"), entry.get("rmse")
        if isinstance(mae, (int, float)) and isinstance(rmse, (int, float)):
            stats.append(ModelStat(name, float(mae), float(rmse)))
        horizon_mae[name] = _horizon_rows(calendar_test, name)
    for name in SEQUENCE_MODELS:
        entry = sequence_test.get(name) or {}
        mae, rmse = entry.get("mae"), entry.get("rmse")
        if isinstance(mae, (int, float)) and isinstance(rmse, (int, float)):
            stats.append(ModelStat(name, float(mae), float(rmse)))
        horizon_mae[name] = _horizon_rows(sequence_test, name)

    # 182~366일 구간은 백테스트(기준일 3개)로만 잰 값이라 지평 표에 마지막 줄로 붙인다
    pooled_block = backtest.get("pooled") or {}
    pooled: dict[str, float] = {}
    for name, entry in (pooled_block.get("models") or {}).items():
        mae = entry.get("mae_weighted") if isinstance(entry, dict) else None
        if isinstance(mae, (int, float)):
            pooled[str(name)] = float(mae)
    for name, mae in pooled.items():
        if name in horizon_mae or name in CALENDAR_MODELS:
            horizon_mae.setdefault(name, {})[FAR_HORIZON] = mae
    stats = [stat for stat in stats if stat.name in horizon_mae]
    stats.sort(key=lambda stat: stat.mae)

    horizons = tuple(
        horizon
        for horizon in ("1-31", "32-91", "92-181", FAR_HORIZON)
        if any(horizon in row for row in horizon_mae.values())
    )

    tertiles: dict[str, dict[str, float]] = {}
    for name in SEQUENCE_MODELS:
        entries = (sequence_test.get(name) or {}).get("tertile") or {}
        values = {
            label: float(entries[label]["mae"])
            for label in TERTILES
            if isinstance(entries.get(label), dict) and isinstance(entries[label].get("mae"), (int, float))
        }
        if values:
            tertiles[name] = values

    per_cut = backtest.get("per_cut") or {}
    cut_dates = tuple(sorted(per_cut))
    cut_mae: dict[str, dict[str, float]] = {}
    for cut in cut_dates:
        row = {}
        for name, entry in ((per_cut.get(cut) or {}).get("models") or {}).items():
            mae = entry.get("mae") if isinstance(entry, dict) else None
            if name in CALENDAR_MODELS and isinstance(mae, (int, float)):
                row[str(name)] = float(mae)
        if row:
            cut_mae[cut] = row

    return AnalysisData(
        models=tuple(stats),
        horizons=horizons,
        horizon_mae=horizon_mae,
        tertiles=tertiles,
        cut_dates=cut_dates,
        cut_mae=cut_mae,
        pooled=pooled,
        sources={
            "calendar": str(calendar_path),
            "sequence": str(sequence_path),
            "backtest": str(backtest_path),
        },
    )


def heat_step(value: float, low: float, high: float, steps: int = len(HEAT_STEPS)) -> int:
    """값이 6단계 중 어디에 드는지 (0 = 가장 옅음)."""
    if high <= low:
        return 0
    ratio = (value - low) / (high - low)
    return max(0, min(steps - 1, int(ratio * steps)))


def heat_color(value: float, low: float, high: float) -> str:
    return HEAT_STEPS[heat_step(value, low, high)]


def horizon_insight(data: AnalysisData) -> str:
    """지평이 길어질수록 오차가 줄어드는 모델과 늘어나는 모델을 다섯 모델 모두에서 찾아 한 문장으로 쓴다."""
    short, long = "", ""
    falling: list[str] = []
    rising: list[str] = []
    names = [name for name in (*CALENDAR_MODELS, *SEQUENCE_MODELS) if data.horizon_mae.get(name)]
    for name in names:
        near, far = data.value_at(name, SHORT_HORIZON), data.value_at(name, LONG_HORIZON)
        if near is None or far is None:
            continue
        if far < near:
            falling.append(f"{name}({near:,.0f}→{far:,.0f}명)")
        elif far > near:
            rising.append(f"{name}({near:,.0f}→{far:,.0f}명)")
    worst = None
    for name in names:
        value = data.value_at(name, SHORT_HORIZON)
        if value is None:
            continue
        if worst is None or value > worst[1]:
            worst = (name, value)
    parts = []
    if worst is not None:
        short = f"가까운 날짜(1~31일)에서 가장 부정확한 모델은 {worst[0]}입니다(MAE {worst[1]:,.0f}명)."
        parts.append(short)
    if falling:
        long = f"먼 날짜(92~181일)로 갈수록 오차가 줄어드는 모델은 {', '.join(falling)}입니다."
        parts.append(long)
    if rising:
        parts.append(f"반대로 오차가 늘어나는 모델은 {', '.join(rising)}입니다.")
    if not parts:
        return "지평 구간별 기록이 없습니다."
    return " ".join(parts)


def tertile_note(data: AnalysisData) -> str:
    """규모별 오차가 얼마나 벌어지는지 자료에서 계산해 적는다."""
    name = "D" if "D" in data.tertiles else next(iter(data.tertiles), "")
    values = data.tertiles.get(name) or {}
    if len(values) < 2:
        return ""
    small, large = values.get("소"), values.get("대")
    if small is None or large is None or small <= 0:
        return ""
    return (
        f"이용량이 적은 역·시간대는 오차 {small:,.1f}명, 많은 곳은 {large:,.1f}명입니다. "
        f"큰 역일수록 오차가 약 {large / small:.0f}배 커집니다."
    )


def backtest_note(data: AnalysisData) -> str:
    """182~366일 구간에서 모델 사이 차이가 얼마인지 적는다."""
    values = [data.pooled[name] for name in CALENDAR_MODELS if name in data.pooled]
    if not values:
        return ""
    best = min(data.pooled, key=data.pooled.get)
    return (
        f"기준일 {len(data.cut_dates)}개로 182~366일 앞을 다시 맞혀 본 결과, "
        f"가장 정확한 모델은 {best}(MAE {data.pooled[best]:,.1f}명)입니다. "
        "실제값은 학습·검증 구간(2026-02-14 이전)만 썼습니다."
    )