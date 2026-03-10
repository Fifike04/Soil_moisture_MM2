import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Step 5 — Diagnostics & reporting", layout="wide")
st.title("Step 5 — Diagnostics & reporting")

OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
OUT5 = os.path.join("outputs", "step5")
os.makedirs(OUT5, exist_ok=True)

# --------------------------------------------------
# Paths
# --------------------------------------------------
X_PATH = os.path.join(OUT2, "X.npy")
Y_PATH = os.path.join(OUT2, "y.npy")
META_PATH = os.path.join(OUT2, "meta.csv")
CFG_PATH = os.path.join(OUT2, "config.json")

MODEL_PATH = os.path.join(OUT3, "model.keras")

DEPTH_OUT = os.path.join(OUT5, "metrics_by_depth_test.csv")
STATION_OUT = os.path.join(OUT5, "metrics_by_station_test.csv")
SUMMARY_OUT = os.path.join(OUT5, "step5_summary.json")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def require_file(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true, y_pred):
    d = y_true - y_pred
    return float(np.mean(d * d))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2_score_numpy(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
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
            "R2": r2_score_numpy(yt, yp),
        })
    return pd.DataFrame(rows)

def metrics_by_station(meta_df: pd.DataFrame, y_true_2d: np.ndarray, y_pred_2d: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    if "station" not in meta_df.columns:
        return pd.DataFrame({"error": ["meta.csv has no 'station' column"]})

    rows = []
    for stn, idx in meta_df.groupby("station").groups.items():
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

    return pd.DataFrame(rows).sort_values("RMSE_mean", ascending=False).reset_index(drop=True)

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

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# --------------------------------------------------
# Load Step2 + Step3 outputs
# --------------------------------------------------
st.subheader("1) Load required outputs")

require_file(X_PATH, "Step2 X.npy")
require_file(Y_PATH, "Step2 y.npy")
require_file(META_PATH, "Step2 meta.csv")
require_file(CFG_PATH, "Step2 config.json")
require_file(MODEL_PATH, "Step3 model.keras")

try:
    X = np.load(X_PATH, allow_pickle=True)
    y = np.load(Y_PATH, allow_pickle=True)
    meta_df = pd.read_csv(META_PATH)

    with open(CFG_PATH, "r", encoding="utf-8") as f:
        step2_cfg = json.load(f)

    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)

except Exception as e:
    st.error(f"Failed to load required outputs: {e}")
    st.stop()

if y.ndim == 1:
    y = y.reshape(-1, 1)

if X.ndim != 3:
    st.error(f"Expected X as 3D array, got: {X.shape}")
    st.stop()

if y.ndim != 2:
    st.error(f"Expected y as 2D array, got: {y.shape}")
    st.stop()

if len(meta_df) != len(y):
    st.error(f"meta.csv row count must match y length. meta={len(meta_df)}, y={len(y)}")
    st.stop()

if "target_date" not in meta_df.columns:
    st.error("meta.csv must contain 'target_date'.")
    st.stop()

meta_df["target_date"] = pd.to_datetime(meta_df["target_date"], errors="coerce")

target_names = step2_cfg.get("target_cols", [])
if not target_names or len(target_names) != y.shape[1]:
    target_names = [f"target{i}" for i in range(y.shape[1])]

st.success("Step2 + Step3 outputs loaded successfully ✅")
st.write({
    "X_shape": tuple(X.shape),
    "y_shape": tuple(y.shape),
    "meta_rows": int(len(meta_df)),
    "targets": target_names
})

with st.expander("Step2 config"):
    st.json(step2_cfg)

# --------------------------------------------------
# Test date range
# --------------------------------------------------
st.subheader("2) Select evaluation range")

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

st.write("Split counts:")
st.json({
    "train": int(train_idx.sum()),
    "val": int(val_idx.sum()),
    "test": int(test_idx.sum()),
})

if int(test_idx.sum()) == 0:
    st.error("Test split has 0 samples. Adjust the date range.")
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    if d.notna().any():
        st.info(f"Available target_date range: {d.min()} — {d.max()}")
    st.stop()

# --------------------------------------------------
# Predict on test set
# --------------------------------------------------
st.subheader("3) Run diagnostics on test set")

X_test = X[test_idx]
y_test = y[test_idx]
meta_test = meta_df.loc[test_idx].reset_index(drop=True)

with st.spinner("Predicting on test set..."):
    y_pred_test = model.predict(X_test, verbose=0)

if y_pred_test.ndim == 1:
    y_pred_test = y_pred_test.reshape(-1, 1)

if y_pred_test.shape[1] != y_test.shape[1]:
    st.error(f"Prediction shape mismatch: y_pred={y_pred_test.shape}, y_true={y_test.shape}")
    st.stop()

# --------------------------------------------------
# Metrics
# --------------------------------------------------
depth_df = metrics_by_depth(y_test, y_pred_test, target_names)
station_df = metrics_by_station(meta_test, y_test, y_pred_test, target_names)

st.write("### Per-target metrics")
st.dataframe(depth_df, use_container_width=True)

st.write("### Per-station metrics (mean across targets)")
st.dataframe(station_df, use_container_width=True)

# --------------------------------------------------
# Plots
# --------------------------------------------------
st.subheader("4) Diagnostic plots")

errors = y_pred_test - y_test
abs_errors = np.abs(errors)

st.write("#### Error distributions")
fig1 = plot_error_hist(errors.flatten(), "Signed errors (all targets pooled)")
st.pyplot(fig1)

fig2 = plot_error_hist(abs_errors.flatten(), "Absolute errors (all targets pooled)")
st.pyplot(fig2)

st.write("#### True vs Pred")
max_points = st.slider("Max points per target plot", 100, 3000, 600, 100)
figs = plot_true_vs_pred(y_test, y_pred_test, target_names, max_points=max_points)
for fig in figs:
    st.pyplot(fig)

st.write("#### Worst stations (highest RMSE_mean)")
st.dataframe(station_df.head(15), use_container_width=True)

# --------------------------------------------------
# Save outputs
# --------------------------------------------------
st.subheader("5) Save outputs")

if st.button("Save diagnostics to outputs/step5"):
    try:
        depth_df.to_csv(DEPTH_OUT, index=False)
        station_df.to_csv(STATION_OUT, index=False)

        summary = {
            "inputs": {
                "X": X_PATH,
                "y": Y_PATH,
                "meta": META_PATH,
                "config": CFG_PATH,
                "model": MODEL_PATH,
            },
            "split": {
                "train": {"start": train_start, "end": train_end, "n": int(train_idx.sum())},
                "val": {"start": val_start, "end": val_end, "n": int(val_idx.sum())},
                "test": {"start": test_start, "end": test_end, "n": int(test_idx.sum())},
            },
            "test_overall": {
                "MAE_mean_over_targets": float(depth_df["MAE"].mean()),
                "RMSE_mean_over_targets": float(depth_df["RMSE"].mean()),
            },
            "outputs": {
                "metrics_by_depth": DEPTH_OUT,
                "metrics_by_station": STATION_OUT,
                "summary": SUMMARY_OUT,
            }
        }

        with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(summary), f, ensure_ascii=False, indent=2)

        st.success("Step 5 saved successfully ✅")
        st.code("\n".join([DEPTH_OUT, STATION_OUT, SUMMARY_OUT]))

    except Exception as e:
        st.error(f"Failed to save Step5 outputs: {e}")