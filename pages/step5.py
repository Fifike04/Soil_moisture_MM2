import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Step 5 — Diagnostics & reporting", layout="wide")
st.title("Step 5 — Diagnostics & reporting (STRICT DISK MODE)")

OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
OUT5 = os.path.join("outputs", "step5")
os.makedirs(OUT5, exist_ok=True)

# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def require(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

def bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))

def rel_error_pct(y_true, y_pred):
    return float(np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100)

def split_by_date(meta_df, start, end):
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    return ((d >= pd.to_datetime(start)) & (d <= pd.to_datetime(end))).to_numpy()

def apply_saved_scaler_if_exists(X_3d, scaler_path):
    if not os.path.exists(scaler_path):
        return X_3d, None
    try:
        with open(scaler_path, "r", encoding="utf-8") as f:
            sc = json.load(f)
        mu = np.asarray(sc["mean"], dtype=np.float32)
        sd = np.asarray(sc["std"], dtype=np.float32)
        sd = np.where(sd == 0, 1.0, sd)
        if mu.shape[0] != X_3d.shape[-1]:
            return X_3d, None
        return ((X_3d - mu) / sd).astype(np.float32), sc
    except:
        return X_3d, None

def naive_persistence_baseline(X_test, target_dim):
    # Predict next = last value in window
    last_step = X_test[:, -1, :]
    return last_step[:, :target_dim]

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------

x_path = os.path.join(OUT2, "X.npy")
y_path = os.path.join(OUT2, "y.npy")
meta_path = os.path.join(OUT2, "meta.csv")
cfg_path = os.path.join(OUT2, "config.json")
model_path = os.path.join(OUT3, "model.keras")
scaler_path = os.path.join(OUT3, "x_scaler.json")

require(x_path, "X.npy")
require(y_path, "y.npy")
require(meta_path, "meta.csv")
require(model_path, "model.keras")

X = np.load(x_path)
y = np.load(y_path)
meta_df = pd.read_csv(meta_path)

if y.ndim == 1:
    y = y.reshape(-1, 1)

meta_df["target_date"] = pd.to_datetime(meta_df["target_date"], errors="coerce")

with open(cfg_path, "r") as f:
    step2_cfg = json.load(f)

target_names = step2_cfg.get("target_cols", [f"target{i}" for i in range(y.shape[1])])

st.success("Artifacts loaded successfully ✅")

# --------------------------------------------------
# Date-based split
# --------------------------------------------------

st.subheader("Date-based evaluation split")

c1, c2, c3 = st.columns(3)
with c1:
    train_start = st.text_input("Train start", "2016-01-01")
    train_end = st.text_input("Train end", "2022-12-31")
with c2:
    val_start = st.text_input("Val start", "2023-01-01")
    val_end = st.text_input("Val end", "2023-12-31")
with c3:
    test_start = st.text_input("Test start", "2024-01-01")
    test_end = st.text_input("Test end", "2025-12-31")

test_mask = split_by_date(meta_df, test_start, test_end)

if test_mask.sum() == 0:
    st.error("Test split has 0 samples.")
    st.stop()

# --------------------------------------------------
# Model + Scaling
# --------------------------------------------------

import tensorflow as tf
model = tf.keras.models.load_model(model_path)

apply_scaling = st.checkbox("Apply saved scaler (recommended)", True)

X_eval = X.copy()
scaler_used = None

if apply_scaling:
    X_eval, scaler_used = apply_saved_scaler_if_exists(X_eval, scaler_path)

X_test = X_eval[test_mask]
y_test = y[test_mask]
meta_test = meta_df[test_mask].reset_index(drop=True)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

with st.spinner("Predicting..."):
    y_pred = model.predict(X_test, verbose=0)

if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

# --------------------------------------------------
# Global metrics
# --------------------------------------------------

st.subheader("Global metrics (TEST)")

global_metrics = {
    "MAE": mae(y_test, y_pred),
    "RMSE": rmse(y_test, y_pred),
    "R2": r2(y_test, y_pred),
    "Bias": bias(y_test, y_pred),
    "Relative_Error_%": rel_error_pct(y_test, y_pred),
}

st.json(global_metrics)

# --------------------------------------------------
# Per-depth metrics
# --------------------------------------------------

st.subheader("Per-target metrics")

rows = []
for i in range(y_test.shape[1]):
    rows.append({
        "target": target_names[i],
        "MAE": mae(y_test[:, i], y_pred[:, i]),
        "RMSE": rmse(y_test[:, i], y_pred[:, i]),
        "R2": r2(y_test[:, i], y_pred[:, i]),
        "Bias": bias(y_test[:, i], y_pred[:, i]),
    })

depth_df = pd.DataFrame(rows)
st.dataframe(depth_df, use_container_width=True)

# --------------------------------------------------
# Baseline comparison
# --------------------------------------------------

st.subheader("Naive persistence baseline comparison")

baseline_pred = naive_persistence_baseline(X_test, y_test.shape[1])

baseline_metrics = {
    "Baseline_MAE": mae(y_test, baseline_pred),
    "Baseline_RMSE": rmse(y_test, baseline_pred),
    "Baseline_R2": r2(y_test, baseline_pred),
}

st.json(baseline_metrics)

# --------------------------------------------------
# Seasonal / monthly breakdown
# --------------------------------------------------

st.subheader("Monthly breakdown (TEST)")

meta_test["month"] = meta_test["target_date"].dt.month

month_rows = []
for m in sorted(meta_test["month"].dropna().unique()):
    idx = meta_test["month"] == m
    month_rows.append({
        "month": int(m),
        "MAE": mae(y_test[idx], y_pred[idx]),
        "RMSE": rmse(y_test[idx], y_pred[idx]),
        "R2": r2(y_test[idx], y_pred[idx]),
    })

month_df = pd.DataFrame(month_rows)
st.dataframe(month_df, use_container_width=True)

# --------------------------------------------------
# Error distribution
# --------------------------------------------------

st.subheader("Error distribution")

errors = (y_pred - y_test).flatten()

fig = plt.figure()
plt.hist(errors, bins=40)
plt.title("Signed error distribution")
st.pyplot(fig)

# --------------------------------------------------
# Worst stations
# --------------------------------------------------

st.subheader("Worst stations (by RMSE)")

station_rows = []
for stn, idx in meta_test.groupby("station").groups.items():
    idx = np.array(list(idx))
    station_rows.append({
        "station": stn,
        "RMSE": rmse(y_test[idx], y_pred[idx]),
        "MAE": mae(y_test[idx], y_pred[idx]),
    })

station_df = pd.DataFrame(station_rows).sort_values("RMSE", ascending=False)
st.dataframe(station_df, use_container_width=True)

# --------------------------------------------------
# Export
# --------------------------------------------------

if st.button("Export results to outputs/step5/"):
    depth_df.to_csv(os.path.join(OUT5, "metrics_by_depth_test.csv"), index=False)
    station_df.to_csv(os.path.join(OUT5, "metrics_by_station_test.csv"), index=False)
    month_df.to_csv(os.path.join(OUT5, "metrics_by_month_test.csv"), index=False)

    summary = {
        "global_metrics": global_metrics,
        "baseline_metrics": baseline_metrics,
        "scaler_applied": scaler_used is not None,
        "n_test_samples": int(test_mask.sum()),
    }

    with open(os.path.join(OUT5, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    st.success("Exported to outputs/step5/ ✅")