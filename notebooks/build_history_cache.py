"""models/boarding_history.npz 를 만든다.

앱이 시작할 때 train/valid/test CSV 4,936,960행을 다시 읽지 않도록,
(역, 시간대, 날짜) 승차 배열과 역·호선 매핑만 압축해 저장한다.

실행: python notebooks/build_history_cache.py
전제: data/processed/train.csv.gz · valid.csv.gz · test.csv.gz
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "models" / "boarding_history.npz"
HOURS_PER_DAY = 20
CACHE_VERSION = 2  # 앱이 같은 형식인지 확인한다 (1: 버전 표식 없음)


def csv_mtime() -> int:
    """원본 CSV 의 가장 최근 수정 시각 (초). CSV 를 다시 만들면 값이 달라진다."""
    mtimes = [p.stat().st_mtime for p in (DATA / f"{s}.csv.gz" for s in ("train", "valid", "test")) if p.exists()]
    return int(max(mtimes)) if mtimes else 0


def main() -> None:
    frames = [
        pd.read_csv(
            DATA / f"{split}.csv.gz",
            usecols=["date", "station", "line", "station_code", "line_code", "hour_code", "boarding"],
        )
        for split in ("train", "valid", "test")
    ]
    df = pd.concat(frames, ignore_index=True)

    station_codes = np.asarray(sorted(df["station_code"].unique()), dtype=np.int64)
    dates = np.asarray(sorted(df["date"].unique()), dtype=str)
    history = np.full((len(station_codes), HOURS_PER_DAY, len(dates)), np.nan, dtype=np.float32)
    si = pd.Categorical(df["station_code"], categories=station_codes).codes
    di = pd.Categorical(df["date"], categories=dates).codes
    history[si, df["hour_code"].to_numpy(), di] = df["boarding"].to_numpy()

    pairs = df[["line", "station", "station_code"]].drop_duplicates()
    lines = df[["line", "line_code"]].drop_duplicates()
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT,
        boarding=history,
        station_codes=station_codes,
        dates=dates,
        pair_lines=np.asarray(pairs["line"].tolist(), dtype=str),
        pair_stations=np.asarray(pairs["station"].tolist(), dtype=str),
        pair_codes=np.asarray(pairs["station_code"].tolist(), dtype=np.int32),
        line_names=np.asarray(lines["line"].tolist(), dtype=str),
        line_codes=np.asarray(lines["line_code"].tolist(), dtype=np.int32),
        # 원본이 바뀌었는지 앱이 확인할 수 있게 표식을 남긴다
        version=np.asarray(CACHE_VERSION, dtype=np.int64),
        csv_mtime=np.asarray(csv_mtime(), dtype=np.int64),
        source_rows=np.asarray(len(df), dtype=np.int64),
    )
    print(f"저장: {OUT} ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"역 {len(station_codes)}개 · 날짜 {len(dates)}일 · {len(df):,}행")


if __name__ == "__main__":
    main()
