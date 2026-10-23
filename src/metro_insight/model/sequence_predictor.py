"""D/E(LSTM·GRU) 체크포인트를 화면 예측 계약에 연결한다.

torch·체크포인트·창 이력 중 하나라도 없으면 그 모델의 키만 빼고 한국어 사유를 돌려준다.
A/B/C 와 앱 전체는 계속 동작해야 하므로 이 모듈은 import 만으로 torch 를 부르지 않는다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

WINDOW_FALLBACK = 14


class SequencePredictor:
    """models/sequence_{D,E}.pt → (D·E 예상 승차 인원, 실패 사유)."""

    def __init__(self, models_dir: str | Path) -> None:
        self._models_dir = Path(models_dir)
        self._seq: Any = None
        self._torch: Any = None
        self._import_error: str | None = None
        self._models: dict[str, dict[str, Any]] = {}
        self._load_errors: dict[str, str] = {}
        self._loaded = False

    # ---------- 로딩 ----------

    def _ensure_import(self) -> bool:
        if self._seq is not None:
            return True
        if self._import_error is not None:
            return False
        try:
            import torch  # noqa: PLC0415 - torch 가 없어도 앱이 떠야 하므로 지연 import

            from metro_insight.model import sequence  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._import_error = f"torch 가 설치되어 있지 않아 D/E 예측을 건너뜁니다: {type(exc).__name__}"
            return False
        self._torch = torch
        self._seq = sequence
        return True

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._ensure_import():
            return
        for cell in ("D", "E"):
            path = self._models_dir / f"sequence_{cell}.pt"
            if not path.exists():
                self._load_errors[cell] = f"{path.name} 파일이 없어 {cell} 예측을 건너뜁니다."
                continue
            try:
                payload = self._seq.load_checkpoint(path)
                payload["model"] = self._seq.build_model(payload)
                self._models[cell] = payload
            except Exception as exc:  # noqa: BLE001
                self._load_errors[cell] = (
                    f"{path.name} 로드 실패: {type(exc).__name__}: {exc}"
                )

    # ---------- 예측 ----------

    def predict(
        self,
        *,
        history: np.ndarray | None,
        dates: np.ndarray | list[str] | None,
        station_code: int,
        hour_code: int,
        day: str,
        weekday: int,
        is_holiday: bool,
        temperature: float,
        rainfall: bool,
        snowfall: bool,
        profile_month_mean: float,
        profile_n: int,
        profile_weekday_mean: float,
        profile_weekday_n: int,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """성공한 모델 값(사람 수)과 실패한 모델 사유를 함께 돌려준다."""
        self._ensure_loaded()
        errors: dict[str, str] = {
            cell: msg for cell, msg in self._load_errors.items() if cell not in self._models
        }
        values: dict[str, float] = {}
        if not self._models:
            for cell in ("D", "E"):
                errors[cell] = self._load_errors.get(
                    cell, self._import_error or f"models/sequence_{cell}.pt 를 사용할 수 없습니다."
                )
            return values, errors
        if history is None or dates is None:
            for cell in self._models:
                errors[cell] = "승차 이력이 없어 D/E 예측을 건너뜁니다."
            return values, errors

        for cell, payload in self._models.items():
            try:
                value = self._predict_one(
                    payload, history, np.asarray(dates, dtype=str), station_code, hour_code,
                    day, weekday, is_holiday, temperature, rainfall, snowfall,
                    profile_month_mean, profile_n, profile_weekday_mean, profile_weekday_n,
                )
            except Exception as exc:  # noqa: BLE001
                errors[cell] = f"{type(exc).__name__}: {exc}"
                continue
            if value is not None:  # 데이터 기간 밖은 정상적으로 '-' (오류 아님)
                values[cell] = value
        return values, errors

    def _predict_one(
        self,
        payload: dict[str, Any],
        history: np.ndarray,
        dates: np.ndarray,
        station_code: int,
        hour_code: int,
        day: str,
        weekday: int,
        is_holiday: bool,
        temperature: float,
        rainfall: bool,
        snowfall: bool,
        profile_month_mean: float,
        profile_n: int,
        profile_weekday_mean: float,
        profile_weekday_n: int,
    ) -> float | None:
        window = int(payload.get("window", WINDOW_FALLBACK))
        ckpt_dates = np.asarray(payload["dates"], dtype=str)
        last = ckpt_dates[-1] if ckpt_dates.size else ""
        first = ckpt_dates[window] if ckpt_dates.size > window else ""
        d = int(np.searchsorted(ckpt_dates, day, side="left"))
        if day < first or day > last or d >= ckpt_dates.size or ckpt_dates[d] != day:
            return None  # D/E 예측 가능 범위 밖 — 키만 생략하고 오류로 남기지 않는다
        window_dates = ckpt_dates[d - window:d]
        if window_dates.size != window:
            return None

        # 앱의 승차 이력(역, 시간대, 날짜)에서 창 날짜 열을 찾는다.
        cols = np.searchsorted(dates, window_dates, side="left")
        if np.any(cols >= dates.size) or np.any(dates[np.minimum(cols, dates.size - 1)] != window_dates):
            raise ValueError("시계열 모델의 날짜 축과 앱 데이터가 맞지 않습니다.")
        station_row = self._station_row(payload, station_code)
        series = history[station_row, hour_code, cols].astype("float64")
        if np.isnan(series).any():
            raise ValueError("직전 14일 승차 이력에 결측이 있어 D/E 예측을 건너뜁니다.")

        scale = float(payload["series_mean"][station_row, hour_code])
        temp_mean = float(payload["temp_mean"])
        temp_std = float(payload["temp_std"])
        x_seq = np.empty((window, len(self._seq.SEQ_FEATURES)), dtype="float32")
        x_seq[:, 0] = series / scale
        x_seq[:, 1] = payload["weekday"][d - window:d] / 6.0
        x_seq[:, 2] = payload["is_holiday"][d - window:d]
        x_seq[:, 3] = ((payload["temperature"][hour_code, d - window:d] - temp_mean) / temp_std)
        x_seq[:, 4] = payload["rainfall"][hour_code, d - window:d]
        x_seq[:, 5] = payload["snowfall"][hour_code, d - window:d]

        x_ctx = np.empty(len(self._seq.CTX_FEATURES), dtype="float32")
        x_ctx[0] = weekday / 6.0
        x_ctx[1] = int(bool(is_holiday))
        x_ctx[2] = (float(temperature) - temp_mean) / temp_std
        x_ctx[3] = int(bool(rainfall))
        x_ctx[4] = int(bool(snowfall))
        x_ctx[5] = self._ratio(profile_month_mean, profile_n, scale)
        x_ctx[6] = np.log1p(max(int(profile_n), 0))
        x_ctx[7] = self._ratio(profile_weekday_mean, profile_weekday_n, scale)
        x_ctx[8] = np.log1p(max(int(profile_weekday_n), 0))
        prof = self._seq.PROFILE_SLICE
        x_ctx[prof] = (x_ctx[prof] - payload["profile_mean"]) / payload["profile_std"]

        model = payload["model"]
        torch = self._torch
        with torch.no_grad():
            pred = model(
                torch.from_numpy(x_seq[None, :, :]),
                torch.from_numpy(x_ctx[None, :]),
                torch.tensor([station_row], dtype=torch.long),
                torch.tensor([int(payload["line_of_station"][station_row])], dtype=torch.long),
                torch.tensor([int(hour_code)], dtype=torch.long),
            )
        return float(pred.item()) * scale

    @staticmethod
    def _station_row(payload: dict[str, Any], station_code: int) -> int:
        codes = np.asarray(payload["station_codes"], dtype=np.int64)
        hit = np.flatnonzero(codes == int(station_code))
        if hit.size != 1:
            raise ValueError(f"역 코드 {station_code} 가 시계열 체크포인트에 없습니다.")
        return int(hit[0])

    @staticmethod
    def _ratio(mean: float, n: int, scale: float) -> float:
        if n <= 0 or not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
            return 1.0
        return float(mean) / scale

    @property
    def load_errors(self) -> dict[str, str]:
        self._ensure_loaded()
        return dict(self._load_errors)
