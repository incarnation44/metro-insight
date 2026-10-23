"""시계열(LSTM/GRU) 공통 모듈 — 14일 창, 시리즈별 정규화, 체크포인트.

D와 E는 같은 클래스를 쓰고 ``cell`` 인자만 다르다. 체크포인트에는 가중치뿐 아니라
feature 순서·시리즈별 scaler·날짜별 달력/날씨 배열을 함께 저장해, 앱이 CSV 를 다시
읽지 않고도 as-of 14일 창을 만들 수 있게 한다.

핵심 계약
    * 창은 대상일 **전날까지** 14일이다 (``target-WINDOW .. target-1``). 대상일 값은 들어가지 않는다.
    * 승차량은 (역, 시간대)별 train 평균으로 나눈다. 역 규모가 1.3~10,468명이라 전역 표준화는 쓰지 않는다.
    * 창 안에 NaN 이 하나라도 있으면 그 행은 쓰지 않는다 (학습·평가·서빙 공통).
    * 출력은 정규화된 값이므로, 지표는 반드시 ``pred * series_mean`` 으로 되돌린 뒤 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

WINDOW = 14
HOURS_PER_DAY = 20
HIDDEN = 64
DROPOUT = 0.1
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MAX_EPOCHS = 40
PATIENCE = 5
SEED = 42

SEQ_FEATURES = ("boarding_norm", "weekday", "is_holiday", "temperature", "rainfall", "snowfall")
CTX_FEATURES = (
    "weekday", "is_holiday", "temperature", "rainfall", "snowfall",
    "profile_month_ratio", "profile_month_n_log",
    "profile_weekday_ratio", "profile_weekday_n_log",
)
EMBEDDING_DIMS = {"station": 8, "line": 4, "hour": 4}
PROFILE_SLICE = slice(5, 9)


@dataclass(frozen=True)
class ProfileScaler:
    """프로파일 4개(ctx 의 5~8번) 표준화 통계. train 기간에서만 계산한다."""

    mean: np.ndarray
    std: np.ndarray


def series_mean(history: np.ndarray, train_end: int) -> np.ndarray:
    """(역, 시간대)별 train 기간 평균. 표본이 없거나 0 이면 1.0 으로 둔다."""
    part = history[:, :, :train_end]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(part, axis=2)
    mean = np.where(np.isfinite(mean) & (mean > 0), mean, 1.0)
    return mean.astype("float32")


def window_mask(history: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """대상일별로 '직전 window 일이 모두 채워져 있고 대상일도 NaN 이 아닌' 마스크.

    NaN 개수를 누적해 창 합이 0 인 곳만 True 로 만든다 (행별 루프 없음).
    """
    nan = np.isnan(history)
    n_days = history.shape[2]
    cum = np.concatenate(
        [np.zeros((*nan.shape[:2], 1), dtype=np.int32), np.cumsum(nan, axis=2, dtype=np.int32)],
        axis=2,
    )
    ok = np.zeros(nan.shape, dtype=bool)
    days = np.arange(window, n_days)
    if days.size:
        counts = cum[:, :, days] - cum[:, :, days - window]
        ok[:, :, days] = counts == 0
    return ok & ~nan


def sequence_tensor(
    history: np.ndarray,
    scale: np.ndarray,
    weekday: np.ndarray,
    is_holiday: np.ndarray,
    temperature: np.ndarray,
    rainfall: np.ndarray,
    snowfall: np.ndarray,
    temp_mean: float,
    temp_std: float,
) -> np.ndarray:
    """(역, 시간대, 날짜, 6) 시퀀스 feature. 날짜별 값은 그 날짜의 값이다."""
    n_st, n_hour, n_days = history.shape
    out = np.empty((n_st, n_hour, n_days, len(SEQ_FEATURES)), dtype="float32")
    out[..., 0] = history / scale[:, :, None]
    out[..., 1] = (weekday / 6.0)[None, None, :]
    out[..., 2] = is_holiday[None, None, :]
    out[..., 3] = ((temperature - temp_mean) / temp_std)[None, :, :]
    out[..., 4] = rainfall[None, :, :]
    out[..., 5] = snowfall[None, :, :]
    return out


def context_tensor(
    *,
    weekday: np.ndarray,
    is_holiday: np.ndarray,
    temperature: np.ndarray,
    rainfall: np.ndarray,
    snowfall: np.ndarray,
    profile_month_mean: np.ndarray,
    profile_n: np.ndarray,
    profile_weekday_mean: np.ndarray,
    profile_weekday_n: np.ndarray,
    scale: np.ndarray,
    temp_mean: float,
    temp_std: float,
) -> np.ndarray:
    """(역, 시간대, 날짜, 9) 대상일 조건. 프로파일은 시리즈 평균 대비 비율로 넣는다."""
    n_st, n_hour, n_days = profile_month_mean.shape
    out = np.empty((n_st, n_hour, n_days, len(CTX_FEATURES)), dtype="float32")
    scale3 = scale[:, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        month_ratio = profile_month_mean / scale3
        weekday_ratio = profile_weekday_mean / scale3
    out[..., 0] = (weekday / 6.0)[None, None, :]
    out[..., 1] = is_holiday[None, None, :]
    out[..., 2] = ((temperature - temp_mean) / temp_std)[None, :, :]
    out[..., 3] = rainfall[None, :, :]
    out[..., 4] = snowfall[None, :, :]
    out[..., 5] = np.where(np.isfinite(month_ratio), month_ratio, 1.0)
    out[..., 6] = np.log1p(np.where(np.isfinite(profile_n), profile_n, 0.0))
    out[..., 7] = np.where(np.isfinite(weekday_ratio), weekday_ratio, 1.0)
    out[..., 8] = np.log1p(np.where(np.isfinite(profile_weekday_n), profile_weekday_n, 0.0))
    return out


def fit_profile_scaler(ctx: np.ndarray, day_end: int) -> ProfileScaler:
    """train 날짜 구간의 프로파일 4개만 표준화한다. ctx 를 제자리에서 바꾼다."""
    part = ctx[:, :, :day_end, PROFILE_SLICE].reshape(-1, 4)
    mean = np.nanmean(part, axis=0)
    std = np.nanstd(part, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    ctx[..., PROFILE_SLICE] = (ctx[..., PROFILE_SLICE] - mean) / std
    return ProfileScaler(mean=mean.astype("float32"), std=std.astype("float32"))


def apply_profile_scaler(ctx: np.ndarray, scaler: ProfileScaler) -> None:
    ctx[..., PROFILE_SLICE] = (ctx[..., PROFILE_SLICE] - scaler.mean) / scaler.std


def split_index(mask: np.ndarray, day_start: int, day_end: int) -> tuple[np.ndarray, np.ndarray]:
    """마스크에서 (시리즈 번호, 날짜 번호) 짝을 뽑는다. 시리즈 = 역*HOURS_PER_DAY + 시간대."""
    flat = mask[:, :, day_start:day_end].reshape(mask.shape[0] * mask.shape[1], -1)
    series, day = np.nonzero(flat)
    return series.astype("int64"), (day + day_start).astype("int64")


def gather_batch(
    seq_flat: np.ndarray,
    ctx_flat: np.ndarray,
    target_flat: np.ndarray,
    series: np.ndarray,
    day: np.ndarray,
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """배치 하나를 (B,window,6)·(B,9)·(B,) 로 모은다. 창은 target-window..target-1."""
    offsets = day[:, None] - window + np.arange(window, dtype="int64")
    return seq_flat[series[:, None], offsets], ctx_flat[series, day], target_flat[series, day]


class SequenceRegressor(nn.Module):
    """14일 시퀀스 + 대상일 조건 + 임베딩 → 정규화된 승차량 1개."""

    def __init__(
        self,
        cell: str,
        n_stations: int,
        n_lines: int,
        n_hours: int = HOURS_PER_DAY,
        seq_features: int = len(SEQ_FEATURES),
        ctx_features: int = len(CTX_FEATURES),
        hidden: int = HIDDEN,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if cell not in ("LSTM", "GRU"):
            raise ValueError(f"cell 은 LSTM 또는 GRU 여야 한다: {cell}")
        self.cell = cell
        self.station_emb = nn.Embedding(n_stations, EMBEDDING_DIMS["station"])
        self.line_emb = nn.Embedding(n_lines, EMBEDDING_DIMS["line"])
        self.hour_emb = nn.Embedding(n_hours, EMBEDDING_DIMS["hour"])
        rnn_cls = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn_cls(seq_features, hidden, num_layers=1, batch_first=True)
        head_in = hidden + ctx_features + sum(EMBEDDING_DIMS.values())
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_ctx: torch.Tensor,
        station: torch.Tensor,
        line: torch.Tensor,
        hour: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.rnn(x_seq)
        last = out[:, -1, :]
        emb = torch.cat(
            [self.station_emb(station), self.line_emb(line), self.hour_emb(hour)],
            dim=1,
        )
        return self.head(torch.cat([last, x_ctx, emb], dim=1)).squeeze(1)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(path))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    # 체크포인트에는 numpy scaler 배열이 들어 있어 weights_only=False 가 필요하다.
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def build_model(payload: dict[str, Any]) -> SequenceRegressor:
    model = SequenceRegressor(
        cell=str(payload["cell"]),
        n_stations=int(payload["n_stations"]),
        n_lines=int(payload["n_lines"]),
        hidden=int(payload["hidden"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
