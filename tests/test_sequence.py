"""D/E 시퀀스 데이터셋·모델·앱 연결 테스트. 실제 data/processed 를 읽지 않는다."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metro_insight.model.predictor import ABCPredictor, PredictionInput
from metro_insight.model.sequence_predictor import SequencePredictor


def _require_torch():
    pytest.importorskip("torch")
    from metro_insight.model import sequence as seq

    return seq


def values(day: str = "2024-01-25", station: str = "잠실(송파구청)") -> PredictionInput:
    d = date.fromisoformat(day)
    return PredictionInput(
        date=day, line="2호선", station=station, hour=8,
        weekday=d.weekday(), is_weekend=d.weekday() >= 5, is_holiday=False,
        temperature=10.0, rainfall=False, snowfall=False,
    )


def write_history(tmp_path: Path, days: int = 34, start: str = "2024-01-08") -> np.ndarray:
    dates = pd.date_range(start, periods=days).strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates, "line": "2호선", "station": "잠실(송파구청)",
        "station_code": 222, "line_code": 1, "hour_code": 3,
        "boarding": np.arange(100, 100 + days, dtype="float32"),
    })
    df.to_csv(tmp_path / "train.csv.gz", index=False, compression="gzip")
    return dates.to_numpy()


def write_profiles(tmp_path: Path) -> None:
    np.savez(
        tmp_path / "cal_profiles.npz",
        station_code=np.asarray([222]), hour_code=np.asarray([3]),
        month=np.asarray([12]), mean=np.asarray([120.0]), count=np.asarray([50]),
        station_code_wd=np.asarray([222]), hour_code_wd=np.asarray([3]),
        weekday_wd=np.asarray([1]), mean_wd=np.asarray([120.0]), count_wd=np.asarray([50]),
    )


def make_payload(dates: np.ndarray, *, cell: str = "LSTM", series_mean: float = 100.0) -> dict:
    seq = _require_torch()
    weekday = np.asarray([date.fromisoformat(d).weekday() for d in dates], dtype="int64")
    model = seq.SequenceRegressor(cell, 1, 2, hidden=8, dropout=0.0)
    return {
        "state_dict": model.state_dict(),
        "cell": cell, "hidden": 8, "dropout": 0.0, "window": seq.WINDOW,
        "seq_features": list(seq.SEQ_FEATURES), "ctx_features": list(seq.CTX_FEATURES),
        "embedding_dims": dict(seq.EMBEDDING_DIMS),
        "trained": {"seed": 42},
        "series_mean": np.full((1, seq.HOURS_PER_DAY), series_mean, dtype="float32"),
        "profile_mean": np.zeros(4, dtype="float32"),
        "profile_std": np.ones(4, dtype="float32"),
        "temp_mean": np.float32(15.0), "temp_std": np.float32(10.0),
        "dates": dates,
        "weekday": weekday,
        "is_holiday": np.zeros(len(dates), dtype="int64"),
        "temperature": np.full((seq.HOURS_PER_DAY, len(dates)), 15.0, dtype="float32"),
        "rainfall": np.zeros((seq.HOURS_PER_DAY, len(dates)), dtype="float32"),
        "snowfall": np.zeros((seq.HOURS_PER_DAY, len(dates)), dtype="float32"),
        "station_codes": np.asarray([222], dtype="int64"),
        "line_of_station": np.asarray([1], dtype="int64"),
        "n_stations": 1, "n_lines": 2,
    }


def write_checkpoint(tmp_path: Path, payload: dict, cell: str) -> Path:
    seq = _require_torch()
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    path = tmp_path / f"sequence_{cell}.pt"
    seq.save_checkpoint(path, clean)
    return path


def sp_inputs(dates: np.ndarray, day: str = "2024-01-25"):
    history = np.full((1, 20, len(dates)), 100.0, dtype="float32")
    return dict(
        history=history, dates=dates, station_code=222, hour_code=3, day=day,
        weekday=date.fromisoformat(day).weekday(), is_holiday=False, temperature=10.0,
        rainfall=False, snowfall=False, profile_month_mean=100.0, profile_n=17,
        profile_weekday_mean=100.0, profile_weekday_n=2,
    )


# ─────────── 1. 창 길이·feature 순서 ───────────
def test_checkpoint_window_and_feature_order():
    seq = _require_torch()
    dates = pd.date_range("2024-01-08", periods=34).strftime("%Y-%m-%d").to_numpy()
    payload = make_payload(dates)
    assert payload["window"] == 14
    assert payload["seq_features"] == ["boarding_norm", "weekday", "is_holiday",
                                       "temperature", "rainfall", "snowfall"]
    assert payload["ctx_features"] == ["weekday", "is_holiday", "temperature", "rainfall",
                                       "snowfall", "profile_month_ratio", "profile_month_n_log",
                                       "profile_weekday_ratio", "profile_weekday_n_log"]
    assert seq.PROFILE_SLICE == slice(5, 9)
    assert payload["embedding_dims"] == {"station": 8, "line": 4, "hour": 4}


# ─────────── 2. lag 교차검증 ───────────
def test_window_matches_lag_definitions():
    seq = _require_torch()
    n_days = 30
    boarding = np.arange(100, 100 + n_days, dtype="float32")
    df = pd.DataFrame({"boarding": boarding})
    df["lag_1d"] = df["boarding"].shift(1)
    df["lag_7d"] = df["boarding"].shift(7)
    df["mean_prev7d"] = df["boarding"].shift(1).rolling(7).mean()

    seq_flat = np.zeros((1, n_days, 6), dtype="float32")
    seq_flat[0, :, 0] = boarding
    ctx_flat = np.zeros((1, n_days, 9), dtype="float32")
    target_flat = boarding[None, :]
    day = 20
    xs, _, _ = seq.gather_batch(seq_flat, ctx_flat, target_flat, np.asarray([0]),
                                np.asarray([day]), window=14)
    window = xs[0, :, 0]
    assert window.shape == (14,)
    assert window[-1] == pytest.approx(df.loc[day, "lag_1d"])
    assert window[-7] == pytest.approx(df.loc[day, "lag_7d"])
    assert float(np.mean(window[-7:])) == pytest.approx(df.loc[day, "mean_prev7d"])


# ─────────── 3. 누수 3종 ───────────
def test_window_before_target_and_target_not_in_input():
    seq = _require_torch()
    n_days = 30
    seq_flat = np.arange(1, n_days + 1, dtype="float32")[None, :, None].repeat(6, axis=2)
    ctx_flat = np.zeros((1, n_days, 9), dtype="float32")
    day = 20
    offsets = day - 14 + np.arange(14)
    assert offsets.max() == day - 1
    assert day not in offsets
    xs, _, _ = seq.gather_batch(seq_flat, ctx_flat, seq_flat[..., 0], np.asarray([0]),
                                np.asarray([day]))
    assert np.array_equal(xs[0, :, 0], seq_flat[0, offsets, 0])


def test_future_target_shuffle_keeps_past_inputs():
    seq = _require_torch()
    n_days = 40
    seq_flat = np.arange(1, n_days + 1, dtype="float32")[None, :, None].repeat(6, axis=2)
    ctx_flat = np.zeros((1, n_days, 9), dtype="float32")
    target_flat = seq_flat[..., 0].copy()
    days = np.asarray([20, 25, 30])
    before, _, _ = seq.gather_batch(seq_flat, ctx_flat, target_flat, np.asarray([0, 0, 0]),
                                    days)
    changed = seq_flat.copy()
    changed[0, -1, :] = np.random.default_rng(0).permutation(changed[0, -1, :])
    after, _, _ = seq.gather_batch(changed, ctx_flat, target_flat, np.asarray([0, 0, 0]), days)
    assert np.array_equal(before, after)


def test_window_mask_excludes_nan_and_short_history():
    seq = _require_torch()
    history = np.ones((1, 1, 30), dtype="float32")
    history[0, 0, 10] = np.nan
    mask = seq.window_mask(history, 14)
    assert not mask[0, 0, :14].any()  # 창이 모자란 구간
    assert not mask[0, 0, 14]  # 창 0..13 에 NaN(10) 포함
    assert not mask[0, 0, 20]  # 창 6..19 에 NaN(10) 포함
    assert mask[0, 0, 25]  # 창 11..24 — NaN 없음
    history[0, 0, 29] = np.nan
    assert not seq.window_mask(history, 14)[0, 0, 29]  # 대상일 NaN 도 제외


# ─────────── 4. scaler 는 train 통계만 ───────────
def test_profile_scaler_uses_train_days_only():
    seq = _require_torch()
    ctx = np.zeros((1, 1, 20, 9), dtype="float32")
    ctx[..., 5] = np.arange(20, dtype="float32")
    original = ctx.copy()
    scaler = seq.fit_profile_scaler(ctx.copy(), day_end=10)
    assert scaler.mean[0] == pytest.approx(original[:, :, :10, 5].mean())
    assert scaler.std[0] == pytest.approx(original[:, :, :10, 5].std())

    changed = original.copy()
    changed[:, :, 10:, 5] = 999.0
    scaler2 = seq.fit_profile_scaler(changed, day_end=10)
    assert np.allclose(scaler.mean, scaler2.mean)
    assert np.allclose(scaler.std, scaler2.std)


# ─────────── 5. LSTM/GRU 동일 shape·head 재사용 ───────────
def test_lstm_and_gru_share_output_shape_and_head():
    seq = _require_torch()
    torch = pytest.importorskip("torch")
    x = torch.randn(4, 14, 6)
    c = torch.randn(4, 9)
    st = torch.zeros(4, dtype=torch.long)
    ln = torch.ones(4, dtype=torch.long)
    hc = torch.full((4,), 3, dtype=torch.long)
    out = []
    for cell in ("LSTM", "GRU"):
        model = seq.SequenceRegressor(cell, 1, 2, hidden=8, dropout=0.0)
        model.eval()
        out.append(model(x, c, st, ln, hc))
    assert out[0].shape == out[1].shape == (4,)
    names = set(dict(seq.SequenceRegressor("LSTM", 1, 2, hidden=8).head.named_parameters()))
    assert names == set(dict(seq.SequenceRegressor("GRU", 1, 2, hidden=8).head.named_parameters()))


# ─────────── 6. 체크포인트 저장→로드 동일 출력 ───────────
def test_checkpoint_roundtrip_same_output(tmp_path):
    seq = _require_torch()
    dates = pd.date_range("2024-01-08", periods=34).strftime("%Y-%m-%d").to_numpy()
    payload = make_payload(dates)
    path = write_checkpoint(tmp_path, payload, "D")
    loaded = seq.load_checkpoint(path)
    m1 = seq.build_model(payload)
    m2 = seq.build_model(loaded)
    x = np.random.default_rng(0).random((2, 14, 6), dtype="float32")
    c = np.random.default_rng(1).random((2, 9), dtype="float32")
    torch = pytest.importorskip("torch")
    args = (torch.from_numpy(x), torch.from_numpy(c), torch.tensor([0, 0]),
            torch.tensor([1, 1]), torch.tensor([3, 3]))
    assert torch.allclose(m1(*args), m2(*args))


# ─────────── 7. 실패 4종 — D/E 만 빠지고 사유가 남는다 ───────────
def test_missing_checkpoint_reports_reason(tmp_path):
    dates = pd.date_range("2024-01-08", periods=34).strftime("%Y-%m-%d").to_numpy()
    predictor = SequencePredictor(tmp_path)
    got, errors = predictor.predict(**sp_inputs(dates))
    assert got == {}
    assert "파일이 없어" in errors["D"] and "파일이 없어" in errors["E"]


def test_torch_missing_reports_reason(tmp_path):
    dates = pd.date_range("2024-01-08", periods=34).strftime("%Y-%m-%d").to_numpy()
    predictor = SequencePredictor(tmp_path)
    predictor._loaded = True
    predictor._import_error = "torch 가 설치되어 있지 않아 D/E 예측을 건너뜁니다"
    got, errors = predictor.predict(**sp_inputs(dates))
    assert got == {}
    assert "torch" in errors["D"] and "torch" in errors["E"]


def test_corrupt_checkpoint_reports_reason(tmp_path):
    dates = pd.date_range("2024-01-08", periods=34).strftime("%Y-%m-%d").to_numpy()
    (tmp_path / "sequence_D.pt").write_bytes(b"not a torch checkpoint")
    (tmp_path / "sequence_E.pt").write_bytes(b"not a torch checkpoint")
    got, errors = SequencePredictor(tmp_path).predict(**sp_inputs(dates))
    assert got == {}
    assert "로드 실패" in errors["D"] and "로드 실패" in errors["E"]


def test_nan_window_reports_reason(tmp_path):
    dates = pd.date_range("2024-01-08", periods=34).strftime("%Y-%m-%d").to_numpy()
    payload = make_payload(dates)
    write_checkpoint(tmp_path, payload, "D")
    inputs = sp_inputs(dates)
    inputs["history"][0, 3, 16] = np.nan  # 대상일 2024-01-25 의 직전 날
    got, errors = SequencePredictor(tmp_path).predict(**inputs)
    assert "D" not in got
    assert "결측" in errors["D"]


# ─────────── 8. 데이터 기간 밖 — D/E 만 '-' ───────────
def test_out_of_range_omits_only_sequence_models(tmp_path):
    dates = write_history(tmp_path)
    write_profiles(tmp_path)
    write_checkpoint(tmp_path, make_payload(dates), "D")
    write_checkpoint(tmp_path, make_payload(dates, cell="GRU"), "E")
    (tmp_path / "metrics.json").write_text(
        json.dumps({"test": {"A": {"mae": 10.0, "rmse": 20.0}}}), encoding="utf-8")
    (tmp_path / "metrics_sequence.json").write_text(
        json.dumps({"test": {"D": {"mae": 50.0, "rmse": 70.0},
                             "E": {"mae": 55.0, "rmse": 75.0}},
                    "models": {"D": {"headline": {"horizon": {"1-31": {"mae": 1.0, "rmse": 2.0}}}}}}),
        encoding="utf-8")

    p = ABCPredictor(models_dir=tmp_path, data_dir=tmp_path,
                     metrics_path=tmp_path / "metrics.json")
    p._models = {"A": _Capture(), "B": _Capture(), "C": _Capture()}

    inside = p.predict(values("2024-01-25"))
    assert {"A", "B", "C", "D", "E"} <= set(inside)
    assert p.last_errors == {}

    before = p.predict(values("2024-01-15"))  # 창 14일이 모자란 날
    assert {"A", "B", "C"} <= set(before)
    assert "D" not in before and "E" not in before
    assert p.last_errors == {}  # 정상 범위 밖은 오류가 아니다

    future = p.predict(values("2026-12-01"))
    assert {"A", "B", "C"} <= set(future)
    assert "D" not in future and "E" not in future

    assert p.scores()["A"].mae == 10.0
    assert p.scores()["D"].mae == 50.0
    assert p.scores()["E"].mae == 55.0
    assert "D" not in p._horizon_scores  # 지평 ± 에 D/E 를 붙이지 않는다


class _Capture:
    def predict(self, X):
        return [1.0]
