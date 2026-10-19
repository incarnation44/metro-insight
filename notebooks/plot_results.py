"""시각화: 실제 vs 예측, 모델별 MAE/RMSE 막대, A->B->C 개선."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(exist_ok=True)

FEATS = {
    "A": ["lag_1d", "lag_7d", "mean_prev7d"],
    "B": ["lag_1d", "lag_7d", "mean_prev7d", "hour_code", "weekday",
          "is_weekend", "is_holiday", "station_code", "line_code"],
    "C": ["lag_1d", "lag_7d", "mean_prev7d", "hour_code", "weekday",
          "is_weekend", "is_holiday", "station_code", "line_code",
          "temperature", "rainfall", "snowfall"],
}

te = pd.read_csv(DATA / "test.csv.gz")
with open(MODELS / "metrics_abc.json", encoding="utf-8") as f:
    metrics = json.load(f)

# 1) 모델별 MAE/RMSE 막대
names = ["baseline", "A", "B", "C"]
mae = [metrics[n]["test"]["mae"] for n in names]
rmse = [metrics[n]["test"]["rmse"] for n in names]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - 0.2, mae, 0.4, label="MAE")
ax.bar(x + 0.2, rmse, 0.4, label="RMSE")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("명")
ax.set_title("모델별 Test MAE / RMSE")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "metrics_bar.png", dpi=120)
plt.close(fig)

# 2) A->B->C 개선 폭 (MAE)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(names, mae, marker="o")
ax.set_ylabel("Test MAE (명)")
ax.set_title("A -> B -> C 개선 (MAE)")
for i, v in enumerate(mae):
    ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center")
fig.tight_layout()
fig.savefig(FIGS / "improvement.png", dpi=120)
plt.close(fig)

# 3) 실제 vs 예측 (B 모델, 샘플)
model_b = joblib.load(MODELS / "model_B.joblib")
sample = te.sample(n=min(2000, len(te)), random_state=42)
pred = model_b.predict(sample[FEATS["B"]])
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(sample.boarding, pred, s=5, alpha=0.3)
lim = max(sample.boarding.max(), pred.max())
ax.plot([0, lim], [0, lim], "r--", lw=1)
ax.set_xlabel("실제 승차")
ax.set_ylabel("예측 승차 (B)")
ax.set_title("실제 vs 예측 (B, 샘플 2000)")
fig.tight_layout()
fig.savefig(FIGS / "actual_vs_pred_B.png", dpi=120)
plt.close(fig)

print("saved figures to", FIGS)
