import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

st.set_page_config(page_title="Step 6 — Station holdout generalization", layout="wide")
st.title("Step 6 — Station holdout generalization (STRICT HOLDOUT)")

OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
OUT6 = os.path.join("outputs", "step6")
os.makedirs(OUT6, exist_ok=True)

# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------

def require(path, label):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))

def rel_error_pct(y_true, y_pred):
    return float(np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100)

def build_lstm(input_timesteps, input_features, output_dim,
               lstm1, lstm2, dropout, dense_units, lr):

    model = models.Sequential()
    model.add(layers.Input(shape=(input_timesteps, input_features)))
    model.add(layers.LSTM(lstm1, return_sequences=(lstm2 > 0)))
    if dropout > 0:
        model.add(layers.Dropout(dropout))
    if lstm2 > 0:
        model.add(layers.LSTM(lstm2))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    if dense_units > 0:
        model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(output_dim))
    model.compile(optimizer=optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model

def flatten_X(X):
    N, T, F = X.shape
    return X.reshape(N*T, F)

def unflatten_X(Xf, N, T):
    F = Xf.shape[1]
    return Xf.reshape(N, T, F)

def date_mask(meta_df, start, end):
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    return ((d >= pd.to_datetime(start)) & (d <= pd.to_datetime(end))).to_numpy()

# -------------------------------------------------------
# Load Step2 artifacts
# -------------------------------------------------------

x_path = os.path.join(OUT2, "X.npy")
y_path = os.path.join(OUT2, "y.npy")
meta_path = os.path.join(OUT2, "meta.csv")
cfg_path = os.path.join(OUT2, "config.json")

require(x_path, "X.npy")
require(y_path, "y.npy")
require(meta_path, "meta.csv")
require(cfg_path, "config.json")

X = np.load(x_path)
y = np.load(y_path)
meta_df = pd.read_csv(meta_path)

with open(cfg_path, "r") as f:
    config = json.load(f)

if y.ndim == 1:
    y = y.reshape(-1, 1)

meta_df["station"] = meta_df["station"].astype(str).str.lower().str.strip()

target_names = config.get("target_cols", [f"target{i}" for i in range(y.shape[1])])
feature_names = config.get("final_feature_cols", [])

st.success("Loaded Step2 dataset ✅")

# -------------------------------------------------------
# UI
# -------------------------------------------------------

stations = sorted(meta_df["station"].unique().tolist())

with st.sidebar:
    holdout_station = st.selectbox("Holdout station", stations)
    train_start = st.text_input("Train start", "2016-01-01")
    train_end   = st.text_input("Train end",   "2022-12-31")
    val_start   = st.text_input("Val start",   "2023-01-01")
    val_end     = st.text_input("Val end",     "2023-12-31")
    test_start  = st.text_input("Test start",  "2024-01-01")
    test_end    = st.text_input("Test end",    "2025-12-31")

    st.divider()
    lstm1 = st.number_input("LSTM units 1", 8, 512, 64, 8)
    lstm2 = st.number_input("LSTM units 2 (0=off)", 0, 512, 32, 8)
    dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
    dense_units = st.number_input("Dense units", 0, 512, 32, 8)

    lr = st.number_input("Learning rate", 1e-5, 1e-2, 1e-3, format="%.5f")
    batch_size = st.number_input("Batch size", 8, 1024, 64, 8)
    epochs = st.number_input("Epochs", 1, 300, 30)
    patience = st.number_input("Early stopping patience", 1, 50, 7)

    export_results = st.checkbox("Export results", True)

# -------------------------------------------------------
# Strict holdout masking
# -------------------------------------------------------

holdout_mask = meta_df["station"] == holdout_station
non_holdout = ~holdout_mask

train_mask = non_holdout & date_mask(meta_df, train_start, train_end)
val_mask   = non_holdout & date_mask(meta_df, val_start, val_end)
test_mask  = holdout_mask & date_mask(meta_df, test_start, test_end)

if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
    st.error("One of the splits has 0 samples.")
    st.stop()

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val     = X[val_mask], y[val_mask]
X_test, y_test   = X[test_mask], y[test_mask]

# -------------------------------------------------------
# Scaling (TRAIN ONLY)
# -------------------------------------------------------

x_scaler = StandardScaler()
x_scaler.fit(flatten_X(X_train))

X_train_s = unflatten_X(x_scaler.transform(flatten_X(X_train)), X_train.shape[0], X_train.shape[1])
X_val_s   = unflatten_X(x_scaler.transform(flatten_X(X_val)),   X_val.shape[0],   X_val.shape[1])
X_test_s  = unflatten_X(x_scaler.transform(flatten_X(X_test)),  X_test.shape[0],  X_test.shape[1])

# -------------------------------------------------------
# Training
# -------------------------------------------------------

if st.button("Train holdout model"):

    tf.random.set_seed(42)
    np.random.seed(42)

    model = build_lstm(
        X_train_s.shape[1],
        X_train_s.shape[2],
        y_train.shape[1],
        lstm1, lstm2, dropout, dense_units, lr
    )

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(patience),
        restore_best_weights=True
    )

    model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=[es],
        verbose=0
    )

    st.success("Training complete ✅")

    # -------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------

    y_pred = model.predict(X_test_s, verbose=0)

    global_metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": rmse(y_test, y_pred),
        "R2": float(r2_score(y_test, y_pred)),
        "Bias": bias(y_test, y_pred),
        "Relative_Error_%": rel_error_pct(y_test, y_pred),
    }

    st.subheader("Global holdout metrics")
    st.json(global_metrics)

    # Per-depth
    rows = []
    for i in range(y_test.shape[1]):
        rows.append({
            "target": target_names[i],
            "MAE": mean_absolute_error(y_test[:, i], y_pred[:, i]),
            "RMSE": rmse(y_test[:, i], y_pred[:, i]),
            "R2": r2_score(y_test[:, i], y_pred[:, i]),
        })

    depth_df = pd.DataFrame(rows)
    st.subheader("Per-depth metrics")
    st.dataframe(depth_df)

    # -------------------------------------------------------
    # Plot
    # -------------------------------------------------------

    st.subheader("True vs Pred (first 500 samples)")
    n = min(500, len(y_test))
    for i in range(y_test.shape[1]):
        fig = plt.figure()
        plt.plot(y_test[:n, i], label="true")
        plt.plot(y_pred[:n, i], label="pred")
        plt.title(target_names[i])
        plt.legend()
        st.pyplot(fig)

    # -------------------------------------------------------
    # Export
    # -------------------------------------------------------

    if export_results:
        depth_df.to_csv(os.path.join(OUT6, f"holdout_depth_{holdout_station}.csv"), index=False)
        with open(os.path.join(OUT6, f"holdout_summary_{holdout_station}.json"), "w") as f:
            json.dump(global_metrics, f, indent=2)

        st.success("Exported to outputs/step6/ ✅")