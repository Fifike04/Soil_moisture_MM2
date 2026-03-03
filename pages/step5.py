import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Step 5 — Diagnostics & reporting", layout="wide")
st.title("Step 5 — Diagnostics & reporting (DISK ONLY from outputs/)")

OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
OUT5 = os.path.join("outputs", "step5")
os.makedirs(OUT5, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def require(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true, y_pred):
    d = (y_true - y_pred)
    return float(np.mean(d * d))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2(y_true, y_pred):
    # R2 = 1 - SS_res/SS_tot
    yt = y_true
    yp = y_pred
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

def split_by_date(meta_df: pd.DataFrame, start: str, end: str) -> np.ndarray:
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    return ((d >= pd.to_datetime(start)) & (d <= pd.to_datetime(end))).to_numpy()

def metrics_by_depth(y_true_2d: np.ndarray, y_pred_2d: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    rows = []
    for j in range(y_true_2d.shape[1]):
        yt = y_true_2d[:, j]
        yp = y_pred_2d[:, j]
        rows.append({
            "target": target_names[j] if j < len(target_names) else f"target{j}",
            "MAE": mae(yt, yp),
            "RMSE": rmse(yt, yp),
            "R2": r2(yt, yp),
        })
    return pd.DataFrame(rows)

def metrics_by_station(meta_df: pd.DataFrame, y_true_2d: np.ndarray, y_pred_2d: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    if "station" not in meta_df.columns:
        return pd.DataFrame({"error": ["meta_df has no 'station' column"]})

    rows = []
    groups = meta_df.groupby("station").groups
    for stn, idx in groups.items():
        idx = np.array(list(idx), dtype=int)
        yt = y_true_2d[idx]
        yp = y_pred_2d[idx]

        maes = []
        rmses = []
        for j in range(yt.shape[1]):
            maes.append(mae(yt[:, j], yp[:, j]))
            rmses.append(rmse(yt[:, j], yp[:, j]))

        rows.append({
            "station": stn,
            "n_samples": int(len(idx)),
            "MAE_mean": float(np.mean(maes)),
            "RMSE_mean": float(np.mean(rmses)),
        })

    return pd.DataFrame(rows).sort_values("RMSE_mean", ascending=False)

def plot_error_hist(errors_1d: np.ndarray, title: str):
    fig = plt.figure()
    plt.hist(errors_1d, bins=40)
    plt.title(title)
    plt.xlabel("Error")
    plt.ylabel("Count")
    return fig

def plot_true_vs_pred(y_true_2d: np.ndarray, y_pred_2d: np.ndarray, target_names: list[str], max_points=600):
    figs = []
    n = min(len(y_true_2d), max_points)
    x = np.arange(n)
    for j in range(y_true_2d.shape[1]):
        fig = plt.figure()
        plt.plot(x, y_true_2d[:n, j], label="true")
        plt.plot(x, y_pred_2d[:n, j], label="pred")
        name = target_names[j] if j < len(target_names) else f"target{j}"
        plt.title(f"True vs Pred — {name} (first {n})")
        plt.xlabel("Sample index")
        plt.ylabel("Value")
        plt.legend()
        figs.append(fig)
    return figs

# -----------------------------
# Load artifacts from outputs/
# -----------------------------
x_path = os.path.join(OUT2, "X.npy")
y_path = os.path.join(OUT2, "y.npy")
meta_path = os.path.join(OUT2, "meta.csv")
cfg_path = os.path.join(OUT2, "config.json")  # optional

model_path = os.path.join(OUT3, "model.keras")
metrics3_path = os.path.join(OUT3, "metrics.json")  # optional

require(x_path, "Step2 X.npy (outputs/step2/X.npy)")
require(y_path, "Step2 y.npy (outputs/step2/y.npy)")
require(meta_path, "Step2 meta.csv (outputs/step2/meta.csv)")
require(model_path, "Step3 model.keras (outputs/step3/model.keras)")

X = np.load(x_path, allow_pickle=True)
y = np.load(y_path, allow_pickle=True)
meta_df = pd.read_csv(meta_path)

step2_cfg = {}
if os.path.exists(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        step2_cfg = json.load(f)

step3_metrics = {}
if os.path.exists(metrics3_path):
    with open(metrics3_path, "r", encoding="utf-8") as f:
        step3_metrics = json.load(f)

# Basic checks
if X.ndim != 3:
    st.error(f"Expected X as 3D (N,T,F). Got: {X.shape}")
    st.stop()

# y can be (N,) or (N,targets)
if y.ndim == 1:
    y2 = y.reshape(-1, 1)
elif y.ndim == 2:
    y2 = y
else:
    st.error(f"Expected y as 1D or 2D. Got: {y.shape}")
    st.stop()

if "target_date" not in meta_df.columns:
    st.error("meta.csv must contain 'target_date' column.")
    st.stop()

if len(meta_df) != len(y2):
    st.error(f"meta.csv length must match y length. meta={len(meta_df)}, y={len(y2)}")
    st.stop()

# target names (from step2 config if available)
target_names = step2_cfg.get("target_cols", [])
if not target_names:
    target_names = [f"target{i}" for i in range(y2.shape[1])]
if len(target_names) != y2.shape[1]:
    target_names = [f"target{i}" for i in range(y2.shape[1])]

st.success("Loaded Step2 + Step3 artifacts from outputs ✅")
with st.expander("Paths & configs"):
    st.write("Step2:")
    st.code("\n".join([x_path, y_path, meta_path]))
    if step2_cfg:
        st.write("Step2 config.json:")
        st.json(step2_cfg)
    st.write("Step3:")
    st.code(model_path)
    if step3_metrics:
        st.write("Step3 metrics.json:")
        st.json(step3_metrics)

# -----------------------------
# User split settings (by target_date)
# -----------------------------
st.subheader("Split ranges (by target_date)")

c1, c2, c3 = st.columns(3)
with c1:
    train_start = st.text_input("Train start", value="2016-01-01")
    train_end   = st.text_input("Train end",   value="2022-12-31")
with c2:
    val_start   = st.text_input("Val start",   value="2023-01-01")
    val_end     = st.text_input("Val end",     value="2023-12-31")
with c3:
    test_start  = st.text_input("Test start",  value="2024-01-01")
    test_end    = st.text_input("Test end",    value="2025-12-31")

train_idx = split_by_date(meta_df, train_start, train_end)
val_idx   = split_by_date(meta_df, val_start, val_end)
test_idx  = split_by_date(meta_df, test_start, test_end)

st.subheader("Dataset summary")
cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    st.metric("Samples", X.shape[0])
with cc2:
    st.metric("Timesteps", X.shape[1])
with cc3:
    st.metric("Features", X.shape[2])
with cc4:
    st.metric("Targets", y2.shape[1])

st.subheader("Split counts")
st.write({
    "train": int(train_idx.sum()),
    "val": int(val_idx.sum()),
    "test": int(test_idx.sum())
})

if int(test_idx.sum()) == 0:
    st.error("Test split has 0 samples — adjust date ranges.")
    st.stop()

# -----------------------------
# Load model and predict on TEST (and VAL optionally)
# -----------------------------
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error("Failed to load model.keras")
    st.exception(e)
    st.stop()

X_test = X[test_idx]
y_test = y2[test_idx]
meta_test = meta_df[test_idx].reset_index(drop=True)

with st.spinner("Predicting on TEST..."):
    y_pred_test = model.predict(X_test, verbose=0)

if y_pred_test.ndim == 1:
    y_pred_test = y_pred_test.reshape(-1, 1)

if y_pred_test.shape[1] != y_test.shape[1]:
    st.error(f"Prediction targets mismatch: y_pred={y_pred_test.shape}, y_true={y_test.shape}")
    st.stop()

# Optional VAL
do_val = int(val_idx.sum()) > 0
if do_val:
    X_val = X[val_idx]
    y_val = y2[val_idx]
    with st.spinner("Predicting on VAL..."):
        y_pred_val = model.predict(X_val, verbose=0)
    if y_pred_val.ndim == 1:
        y_pred_val = y_pred_val.reshape(-1, 1)
else:
    y_val = y_pred_val = None

# -----------------------------
# Metrics
# -----------------------------
st.subheader("Metrics — TEST set")

depth_df = metrics_by_depth(y_test, y_pred_test, target_names)
st.write("### Per-target metrics")
st.dataframe(depth_df, use_container_width=True)

station_df = metrics_by_station(meta_test, y_test, y_pred_test, target_names)
st.write("### Per-station metrics (mean across targets)")
st.dataframe(station_df, use_container_width=True)

# Error distribution
st.subheader("Error distribution (TEST)")
errors = (y_pred_test - y_test)
abs_errors = np.abs(errors)

st.pyplot(plot_error_hist(errors.flatten(), "Signed errors (all targets pooled)"))
st.pyplot(plot_error_hist(abs_errors.flatten(), "Absolute errors (all targets pooled)"))

# True vs Pred plots
st.subheader("True vs Pred (TEST, first samples)")
max_points = st.slider("Max points per plot", 100, 3000, 600, 100)
figs = plot_true_vs_pred(y_test, y_pred_test, target_names, max_points=max_points)
for fig in figs:
    st.pyplot(fig)

# Worst stations
st.subheader("Worst stations (highest RMSE_mean)")
st.dataframe(station_df.head(15), use_container_width=True)

# -----------------------------
# Export to outputs/step5
# -----------------------------
st.subheader("Export")

export_to_disk = st.checkbox("Export CSV metrics to outputs/step5/", value=True)

if export_to_disk:
    depth_path = os.path.join(OUT5, "metrics_by_depth_test.csv")
    station_path = os.path.join(OUT5, "metrics_by_station_test.csv")
    summary_path = os.path.join(OUT5, "step5_summary.json")

    depth_df.to_csv(depth_path, index=False)
    station_df.to_csv(station_path, index=False)

    summary = {
        "paths": {
            "X": x_path,
            "y": y_path,
            "meta": meta_path,
            "model": model_path,
        },
        "splits": {
            "train": {"start": train_start, "end": train_end, "n": int(train_idx.sum())},
            "val":   {"start": val_start,   "end": val_end,   "n": int(val_idx.sum())},
            "test":  {"start": test_start,  "end": test_end,  "n": int(test_idx.sum())},
        },
        "test_overall": {
            "MAE_mean_over_targets": float(depth_df["MAE"].mean()),
            "RMSE_mean_over_targets": float(depth_df["RMSE"].mean()),
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    st.success("Exported ✅")
    st.code("\n".join([depth_path, station_path, summary_path]))