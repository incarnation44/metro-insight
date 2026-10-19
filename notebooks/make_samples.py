"""샘플 CSV 재생성 — git 에 추적되는 *_sample.csv 를 현재 데이터에서 결정적으로 다시 뽑는다.

실행 (프로젝트 루트에서)
    python notebooks/make_samples.py

규칙: random_state=42 무작위 추출. split 샘플은 자기 split 파일에서만,
      그 split 에 존재하는 연도별로 균등하게 뽑는다 (36행 = 2개 연도면 연도별 18행, 1개 연도면 36행).
      행 수는 기존과 같게 유지 (train/valid/test 36 · weather 20 · 시간대별 30). 다시 실행해도 같은 결과가 나온다.
      data/raw/subway/*_sample.csv (원본 샘플)는 여기서 다루지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SEED = 42
SORT_KEYS_SPLIT = ["date", "line", "station_num", "hour"]

# split 샘플: 연도별 균등 추출 (합계 36행)
SPLIT_PLAN = [
    ("train_sample.csv", "train.csv.gz", 36),
    ("valid_sample.csv", "valid.csv.gz", 36),
    ("test_sample.csv", "test.csv.gz", 36),
]
# 날씨 샘플: 단순 무작위 추출
WEATHER_PLAN = [
    ("weather_sample.csv", "weather.csv", 20, ["date", "hour"]),
    ("weather_시간대별_sample.csv", "weather_시간대별.csv", 30, ["date", "hour"]),
]


def sample_by_year(df: pd.DataFrame, n: int) -> pd.DataFrame:
    years = sorted(df["date"].str[:4].unique())
    per = n // len(years)
    parts = [df[df["date"].str[:4] == y].sample(n=per, random_state=SEED) for y in years]
    return pd.concat(parts)


def main() -> None:
    for name, src, n in SPLIT_PLAN:
        df = pd.read_csv(PROCESSED / src, encoding="utf-8-sig")
        out = sample_by_year(df, n).sort_values(SORT_KEYS_SPLIT).reset_index(drop=True)
        out.to_csv(PROCESSED / name, index=False, encoding="utf-8-sig")
        years = out["date"].str[:4].value_counts().sort_index().to_dict()
        print(f"저장: {name:<30} {len(out):>3}행  ← {src} (seed {SEED}, 연도별 {years})")
    for name, src, n, keys in WEATHER_PLAN:
        df = pd.read_csv(PROCESSED / src, encoding="utf-8-sig")
        out = df.sample(n=n, random_state=SEED).sort_values(keys).reset_index(drop=True)
        out.to_csv(PROCESSED / name, index=False, encoding="utf-8-sig")
        print(f"저장: {name:<30} {len(out):>3}행  ← {src} (seed {SEED})")


if __name__ == "__main__":
    main()
