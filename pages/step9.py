import os
import json
import time
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

st.set_page_config(page_title="Step 9 — Experiment runner (reproducible)", layout="wide")
st.title("Step 9 — Experiment runner (reproducible run pack)")

OUT2 = os.path.join("outputs", "step2")
RUNS_ROOT_DEFAULT = os.path.join("outputs", "runs")

# --------------------------------------------------
# Paths
# --------------------------------------------------
X_PATH = os.path.join(OUT2, "X.npy")
Y_PATH = os.path.join(OUT2, "y.npy")
META_PATH = os.path.join(OUT2, "meta.csv")
CFG_PATH = os.path.join(OUT2, "config.json")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def require_file(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()

def now_run_id():
    return time.strftime("%Y%m%d_%H%M%S")

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def git_commit_hash():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return out
    except Exception:
        return None

def flatten_X(X):
    N, T, F = X.shape
    return X.reshape(N * T, F)

def unflatten_X(X_flat, N, T):
    F = X_flat.shape[1]
    return X_flat.reshape(N, T, F)

def make_scaler(name: str):
    if name == "StandardScaler":
        return StandardScaler()
    if name == "MinMaxScaler":
        return MinMaxScaler()
    return None

def build_lstm_model(T, F, out_dim, units1, units2, dropout, dense_units, lr):
    model = models.Sequential()
    model.add(layers.Input(shape=(T, F)))
    model.add(layers.LSTM(units1, return_sequences=(units2 > 0)))
    if dropout > 0:
        model.add(layers.Dropout(dropout))
    if units2 > 0:
        model.add(layers.LSTM(units2))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    if dense_units > 0:
        model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(out_dim, activation="linear"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )
    return model

def plot_history(hist):
    fig = plt.figure()
    plt.plot(hist.history.get("loss", []), label="train_loss")
    if "val_loss" in hist.history:
        plt.plot(hist.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    return fig

def plot_true_vs_pred(y_true, y_pred, target_names, max_points=600):
    figs = []
    n = min(len(y_true), max_points)
    x = np.arange(n)

    for j in range(y_true.shape[1]):
        fig = plt.figure()
        plt.plot(x, y_true[:n, j], label="true")
        plt.plot(x, y_pred[:n, j], label="pred")
        plt.title(f"True vs Pred — {target_names[j] if j < len(target_names) else f'target{j}'} (first {n})")
        plt.xlabel("Sample index")
        plt.ylabel("Value")
        plt.legend()
        figs.append((j, fig))

    return figs

def split_by_date(meta_df, start, end):
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    return ((d >= pd.to_datetime(start)) & (d <= pd.to_datetime(end))).to_numpy()

def per_target_metrics(y_true, y_pred, target_names):
    rows = []
    for j in range(y_true.shape[1]):
        rows.append({
            "target": target_names[j] if j < len(target_names) else f"target{j}",
            "MAE": float(mean_absolute_error(y_true[:, j], y_pred[:, j])),
            "RMSE": float(np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))),
            "R2": float(r2_score(y_true[:, j], y_pred[:, j]))
        })
    return pd.DataFrame(rows)

def pooled_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "MAE_pooled": float(np.mean(np.abs(err))),
        "RMSE_pooled": float(np.sqrt(np.mean(err ** 2))),
    }

def save_scalers_npz(path, x_scaler_type, y_scaler_type, x_scaler, y_scaler):
    np.savez(
        path,
        x_scaler_type=x_scaler_type,
        y_scaler_type=y_scaler_type,
        x_mean=getattr(x_scaler, "mean_", None),
        x_scale=getattr(x_scaler, "scale_", None),
        x_min=getattr(x_scaler, "min_", None),
        x_data_min=getattr(x_scaler, "data_min_", None),
        x_data_max=getattr(x_scaler, "data_max_", None),
        x_data_range=getattr(x_scaler, "data_range_", None),
        y_mean=getattr(y_scaler, "mean_", None),
        y_scale=getattr(y_scaler, "scale_", None),
        y_min=getattr(y_scaler, "min_", None),
        y_data_min=getattr(y_scaler, "data_min_", None),
        y_data_max=getattr(y_scaler, "data_max_", None),
        y_data_range=getattr(y_scaler, "data_range_", None),
    )

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
# Load Step2 outputs
# --------------------------------------------------
st.subheader("1) Load Step2 outputs")

require_file(X_PATH, "Step2 X.npy")
require_file(Y_PATH, "Step2 y.npy")
require_file(META_PATH, "Step2 meta.csv")
require_file(CFG_PATH, "Step2 config.json")

try:
    X = np.load(X_PATH, allow_pickle=True)
    y = np.load(Y_PATH, allow_pickle=True)
    meta_df = pd.read_csv(META_PATH)

    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

except Exception as e:
    st.error(f"Failed to load Step2 outputs: {e}")
    st.stop()

if y.ndim == 1:
    y = y.reshape(-1, 1)

if X.ndim != 3 or y.ndim != 2:
    st.error(f"Expected X 3D and y 2D. Got X={X.shape}, y={y.shape}")
    st.stop()

if "target_date" not in meta_df.columns:
    st.error("meta.csv must contain 'target_date'.")
    st.stop()

if len(meta_df) != len(y):
    st.error(f"meta.csv row count must match y length. meta={len(meta_df)}, y={len(y)}")
    st.stop()

meta_df["target_date"] = pd.to_datetime(meta_df["target_date"], errors="coerce")

target_names = cfg.get("target_cols", [f"target{i}" for i in range(y.shape[1])])
if len(target_names) != y.shape[1]:
    target_names = [f"target{i}" for i in range(y.shape[1])]

st.success("Step2 outputs loaded successfully ✅")
st.write({
    "X": tuple(X.shape),
    "y": tuple(y.shape),
    "meta_rows": int(len(meta_df)),
    "targets": target_names
})

with st.expander("Step2 config.json"):
    st.json(cfg)

# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------
with st.sidebar:
    st.header("Split by date (target_date)")
    train_start = st.text_input("Train start", value="2016-01-01")
    train_end   = st.text_input("Train end",   value="2022-12-31")
    val_start   = st.text_input("Val start",   value="2023-01-01")
    val_end     = st.text_input("Val end",     value="2023-12-31")
    test_start  = st.text_input("Test start",  value="2024-01-01")
    test_end    = st.text_input("Test end",    value="2025-12-31")

    st.divider()
    st.header("Scaling")
    x_scaler_type = st.selectbox("X scaler", ["StandardScaler", "MinMaxScaler", "No scaling"], index=0)
    y_scaler_type = st.selectbox("y scaler", ["StandardScaler", "MinMaxScaler", "No scaling"], index=0)

    st.divider()
    st.header("Model")
    units1 = st.number_input("LSTM units (layer 1)", 8, 512, 64, 8)
    units2 = st.number_input("LSTM units (layer 2) (0 = off)", 0, 512, 32, 8)
    dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
    dense_units = st.number_input("Dense units (0 = off)", 0, 512, 32, 8)

    st.divider()
    st.header("Training")
    lr = st.number_input("Learning rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f")
    batch_size = st.number_input("Batch size", 8, 2048, 64, 8)
    epochs = st.number_input("Epochs", 1, 400, 40, 1)
    patience = st.number_input("EarlyStopping patience", 1, 80, 8, 1)
    seed = st.number_input("Random seed", 0, 999999, 42, 1)

    st.divider()
    st.header("Output")
    run_root = st.text_input("Run output root", value=RUNS_ROOT_DEFAULT)
    save_model = st.checkbox("Save model + scalers", value=True)
    save_plots = st.checkbox("Save plots", value=True)
    save_predictions = st.checkbox("Save test predictions", value=True)

# --------------------------------------------------
# Split
# --------------------------------------------------
st.subheader("2) Build split")

train_idx = split_by_date(meta_df, train_start, train_end)
val_idx   = split_by_date(meta_df, val_start, val_end)
test_idx  = split_by_date(meta_df, test_start, test_end)

st.write("Split counts:")
st.json({
    "train": int(train_idx.sum()),
    "val": int(val_idx.sum()),
    "test": int(test_idx.sum())
})

if train_idx.sum() == 0 or val_idx.sum() == 0 or test_idx.sum() == 0:
    st.error("One split has 0 samples — adjust your date ranges.")
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    if d.notna().any():
        st.info(f"Available target_date range: {d.min()} — {d.max()}")
    st.stop()

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val     = X[val_idx], y[val_idx]
X_test, y_test   = X[test_idx], y[test_idx]

# --------------------------------------------------
# Run experiment
# --------------------------------------------------
st.subheader("3) Run experiment")

if st.button("Run (train + evaluate + export run pack)"):
    tf.random.set_seed(int(seed))
    np.random.seed(int(seed))

    run_id = now_run_id()
    safe_mkdir(run_root)
    out_dir = os.path.join(run_root, run_id)
    safe_mkdir(out_dir)

    if save_plots:
        safe_mkdir(os.path.join(out_dir, "plots"))

    # Scaling
    x_scaler = make_scaler(x_scaler_type)
    y_scaler = make_scaler(y_scaler_type)

    if x_scaler is not None:
        x_scaler.fit(flatten_X(X_train))
        X_train_s = unflatten_X(x_scaler.transform(flatten_X(X_train)), X_train.shape[0], X_train.shape[1])
        X_val_s   = unflatten_X(x_scaler.transform(flatten_X(X_val)),   X_val.shape[0],   X_val.shape[1])
        X_test_s  = unflatten_X(x_scaler.transform(flatten_X(X_test)),  X_test.shape[0],  X_test.shape[1])
    else:
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test

    if y_scaler is not None:
        y_scaler.fit(y_train)
        y_train_s = y_scaler.transform(y_train)
        y_val_s   = y_scaler.transform(y_val)
        y_test_s  = y_scaler.transform(y_test)
    else:
        y_train_s, y_val_s, y_test_s = y_train, y_val, y_test

    # Model
    model = build_lstm_model(
        T=X_train_s.shape[1],
        F=X_train_s.shape[2],
        out_dim=y_train.shape[1],
        units1=int(units1),
        units2=int(units2),
        dropout=float(dropout),
        dense_units=int(dense_units),
        lr=float(lr),
    )

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(patience),
        restore_best_weights=True
    )
    rlrop = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(2, int(patience // 2)),
        min_lr=1e-6
    )

    with st.spinner("Training..."):
        hist = model.fit(
            X_train_s, y_train_s,
            validation_data=(X_val_s, y_val_s),
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=[es, rlrop],
            verbose=0
        )

    # Predict on test
    y_pred_test_s = model.predict(X_test_s, verbose=0)
    y_pred_test = y_scaler.inverse_transform(y_pred_test_s) if y_scaler is not None else y_pred_test_s

    # Metrics
    df_depth = per_target_metrics(y_test, y_pred_test, target_names)
    pooled = pooled_metrics(y_test, y_pred_test)

    st.success(f"Run finished ✅  Run ID: {run_id}")
    st.write("### Test metrics (per target)")
    st.dataframe(df_depth, use_container_width=True)
    st.write("### Pooled test metrics")
    st.write(pooled)

    # Save config
    git_hash = git_commit_hash()
    run_cfg = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_hash,
        "inputs": {
            "X": X_PATH,
            "y": Y_PATH,
            "meta": META_PATH,
            "config_from_step2": CFG_PATH,
        },
        "data_config_from_step2": cfg,
        "split": {
            "train": [train_start, train_end],
            "val": [val_start, val_end],
            "test": [test_start, test_end],
        },
        "scaling": {"x": x_scaler_type, "y": y_scaler_type},
        "model": {
            "units1": int(units1),
            "units2": int(units2),
            "dropout": float(dropout),
            "dense_units": int(dense_units),
        },
        "training": {
            "lr": float(lr),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "patience": int(patience),
            "seed": int(seed),
        },
        "metrics_pooled_test": pooled,
    }

    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(run_cfg), f, ensure_ascii=False, indent=2)

    df_depth.to_csv(os.path.join(out_dir, "metrics_by_target_test.csv"), index=False)

    if save_model:
        model.save(os.path.join(out_dir, "model.keras"))
        save_scalers_npz(
            os.path.join(out_dir, "scalers.npz"),
            x_scaler_type, y_scaler_type,
            x_scaler, y_scaler
        )

    if save_predictions:
        pred_df = pd.DataFrame(y_pred_test, columns=target_names)
        pred_df.insert(0, "target_date", pd.to_datetime(meta_df.loc[test_idx, "target_date"]).to_numpy())
        if "station" in meta_df.columns:
            pred_df.insert(0, "station", meta_df.loc[test_idx, "station"].astype(str).to_numpy())
        pred_df.to_csv(os.path.join(out_dir, "predictions_test.csv"), index=False)

    if save_plots:
        fig_h = plot_history(hist)
        fig_h.savefig(os.path.join(out_dir, "plots", "loss_curve.png"), bbox_inches="tight")
        plt.close(fig_h)

        figs = plot_true_vs_pred(y_test, y_pred_test, target_names, max_points=600)
        for j, fig in figs:
            fig.savefig(os.path.join(out_dir, "plots", f"true_vs_pred_{j}.png"), bbox_inches="tight")
            plt.close(fig)

    st.info("Saved run pack to:")
    st.code(out_dir)