import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

st.set_page_config(page_title="Step 6 — Station holdout generalization", layout="wide")
st.title("Step 6 — Station holdout generalization test (DISK ONLY from outputs/)")

OUT2 = os.path.join("outputs", "step2")
OUT6 = os.path.join("outputs", "step6")
os.makedirs(OUT6, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def require(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def build_lstm_model(input_timesteps, input_features, output_dim,
                     lstm_units1, lstm_units2, dropout, dense_units, learning_rate):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_timesteps, input_features)))
    model.add(layers.LSTM(lstm_units1, return_sequences=(lstm_units2 > 0)))
    if dropout > 0:
        model.add(layers.Dropout(dropout))
    if lstm_units2 > 0:
        model.add(layers.LSTM(lstm_units2))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    if dense_units > 0:
        model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(output_dim, activation="linear"))
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model

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

def fit_transform_scalers(X_train, X_val, X_test, y_train, y_val, y_test, x_scaler_type, y_scaler_type):
    x_scaler = make_scaler(x_scaler_type)
    y_scaler = make_scaler(y_scaler_type)

    # X scaling
    if x_scaler is not None:
        x_scaler.fit(flatten_X(X_train))
        X_train_s = unflatten_X(x_scaler.transform(flatten_X(X_train)), X_train.shape[0], X_train.shape[1])
        X_val_s   = unflatten_X(x_scaler.transform(flatten_X(X_val)),   X_val.shape[0],   X_val.shape[1])
        X_test_s  = unflatten_X(x_scaler.transform(flatten_X(X_test)),  X_test.shape[0],  X_test.shape[1])
    else:
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test

    # y scaling
    if y_scaler is not None:
        y_scaler.fit(y_train)
        y_train_s = y_scaler.transform(y_train)
        y_val_s   = y_scaler.transform(y_val)
        y_test_s  = y_scaler.transform(y_test)
    else:
        y_train_s, y_val_s, y_test_s = y_train, y_val, y_test

    return (X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, x_scaler, y_scaler)

def inverse_y(y_scaled, y_scaler):
    if y_scaler is None:
        return y_scaled
    return y_scaler.inverse_transform(y_scaled)

def metrics_per_target(y_true, y_pred, target_names):
    rows = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        rows.append({
            "target": target_names[j] if j < len(target_names) else f"target{j}",
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "R2": float(r2_score(yt, yp))
        })
    return pd.DataFrame(rows)

def naive_baseline_last_step(X_test, target_dim, feature_names, target_names):
    idxs = []
    for t in target_names:
        if t in feature_names:
            idxs.append(feature_names.index(t))
        else:
            return None
    last_step = X_test[:, -1, :]
    y_pred = last_step[:, idxs]
    return y_pred

def date_mask(meta_df, start, end):
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    return ((d >= pd.to_datetime(start)) & (d <= pd.to_datetime(end))).to_numpy()

def plot_true_vs_pred(y_true, y_pred, target_names, max_points=400):
    figs = []
    n = min(len(y_true), max_points)
    x = np.arange(n)
    for j in range(y_true.shape[1]):
        fig = plt.figure()
        plt.plot(x, y_true[:n, j], label="true")
        plt.plot(x, y_pred[:n, j], label="pred")
        plt.title(f"Holdout True vs Pred — {target_names[j] if j < len(target_names) else f'target{j}'}")
        plt.xlabel("Sample index")
        plt.ylabel("Value")
        plt.legend()
        figs.append(fig)
    return figs

# -----------------------------
# Load from outputs/step2
# -----------------------------
x_path = os.path.join(OUT2, "X.npy")
y_path = os.path.join(OUT2, "y.npy")
meta_path = os.path.join(OUT2, "meta.csv")
cfg_path = os.path.join(OUT2, "config.json")

require(x_path, "Step2 X.npy (outputs/step2/X.npy)")
require(y_path, "Step2 y.npy (outputs/step2/y.npy)")
require(meta_path, "Step2 meta.csv (outputs/step2/meta.csv)")
require(cfg_path, "Step2 config.json (outputs/step2/config.json)")

X = np.load(x_path, allow_pickle=True)
y = np.load(y_path, allow_pickle=True)
meta_df = pd.read_csv(meta_path)
with open(cfg_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Checks
if "station" not in meta_df.columns or "target_date" not in meta_df.columns:
    st.error("meta.csv must contain columns: 'station' and 'target_date'.")
    st.stop()

meta_df["station"] = meta_df["station"].astype(str).str.strip().str.lower()

if y.ndim == 1:
    y = y.reshape(-1, 1)
if X.ndim != 3 or y.ndim != 2:
    st.error(f"Expected X 3D and y 2D. Got X={X.shape}, y={y.shape}")
    st.stop()

feature_names = config.get("final_feature_cols", [])
target_names = config.get("target_cols", [f"target{i}" for i in range(y.shape[1])])
if len(target_names) != y.shape[1]:
    target_names = [f"target{i}" for i in range(y.shape[1])]

st.subheader("Loaded dataset (from outputs/step2)")
st.write({
    "samples": int(X.shape[0]),
    "timesteps": int(X.shape[1]),
    "features": int(X.shape[2]),
    "targets": int(y.shape[1]),
})

with st.expander("Step2 config.json"):
    st.json(config)

# -----------------------------
# UI (holdout + split + model)
# -----------------------------
with st.sidebar:
    st.header("Holdout station")
    holdout_station = st.text_input("Holdout station name (lowercase)", value="csengele").strip().lower()

    st.divider()
    st.header("Date ranges (applied to holdout test)")
    test_start = st.text_input("Test start", value="2024-01-01")
    test_end   = st.text_input("Test end",   value="2025-12-31")

    st.divider()
    st.header("Train/Val split (non-holdout stations)")
    val_start = st.text_input("Val start", value="2023-01-01")
    val_end   = st.text_input("Val end",   value="2023-12-31")
    train_start = st.text_input("Train start", value="2016-01-01")
    train_end   = st.text_input("Train end",   value="2022-12-31")

    st.divider()
    st.header("Scaling")
    x_scaler_type = st.selectbox("X scaler", ["StandardScaler", "MinMaxScaler", "No scaling"], index=0)
    y_scaler_type = st.selectbox("y scaler", ["StandardScaler", "MinMaxScaler", "No scaling"], index=0)

    st.divider()
    st.header("Model hyperparameters")
    lstm_units1 = st.number_input("LSTM units (layer 1)", min_value=8, max_value=512, value=64, step=8)
    lstm_units2 = st.number_input("LSTM units (layer 2) (0 = off)", min_value=0, max_value=512, value=32, step=8)
    dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
    dense_units = st.number_input("Dense units (0 = off)", min_value=0, max_value=512, value=32, step=8)

    st.divider()
    st.header("Training")
    learning_rate = st.number_input("Learning rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f")
    batch_size = st.number_input("Batch size", min_value=8, max_value=1024, value=64, step=8)
    epochs = st.number_input("Epochs", min_value=1, max_value=300, value=30, step=1)
    patience = st.number_input("EarlyStopping patience", min_value=1, max_value=50, value=7, step=1)

    st.divider()
    st.header("Export")
    export_to_disk = st.checkbox("Export results to outputs/step6/", value=True)
    save_model = st.checkbox("Also save trained holdout model", value=False)

# -----------------------------
# Build masks
# -----------------------------
holdout_mask = (meta_df["station"] == holdout_station).to_numpy()

test_mask = holdout_mask & date_mask(meta_df, test_start, test_end)
non_holdout = ~holdout_mask
train_mask = non_holdout & date_mask(meta_df, train_start, train_end)
val_mask   = non_holdout & date_mask(meta_df, val_start, val_end)

# ---- DEBUG: why splits are empty? ----
st.subheader("DEBUG — split diagnostics")

# station distribution
st.write("Unique stations:", int(meta_df["station"].nunique()))
st.write("Top stations:", meta_df["station"].value_counts().head(15).to_dict())

# does holdout exist?
n_holdout_total = int((meta_df["station"] == holdout_station).sum())
st.write(f"Holdout station '{holdout_station}' samples (ALL dates):", n_holdout_total)

# date parsing + range
d = pd.to_datetime(meta_df["target_date"], errors="coerce")
st.write("target_date NaT count:", int(d.isna().sum()))
if d.notna().any():
    st.write("target_date min:", str(d.min()))
    st.write("target_date max:", str(d.max()))

# counts per mask component
test_date_only = date_mask(meta_df, test_start, test_end)
train_date_only = date_mask(meta_df, train_start, train_end)
val_date_only = date_mask(meta_df, val_start, val_end)

st.write({
    "test_date_only": int(test_date_only.sum()),
    "train_date_only": int(train_date_only.sum()),
    "val_date_only": int(val_date_only.sum()),
    "holdout_mask_only": int(holdout_mask.sum()),
    "non_holdout_only": int((~holdout_mask).sum()),
})

# final masks
st.write({
    "train_mask": int(train_mask.sum()),
    "val_mask": int(val_mask.sum()),
    "test_mask": int(test_mask.sum()),
})

# show a few rows for holdout station (to verify dates)
if n_holdout_total > 0:
    st.write("Sample rows for holdout station (first 20):")
    st.dataframe(
        meta_df.loc[meta_df["station"] == holdout_station, ["station", "target_date"]].head(20),
        use_container_width=True
    )
else:
    st.warning("Holdout station not found. Try picking one from the station list above.")
    
st.subheader("Split counts")
st.write({
    "train (non-holdout)": int(train_mask.sum()),
    "val (non-holdout)": int(val_mask.sum()),
    "test (holdout)": int(test_mask.sum())
})

if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0 or int(test_mask.sum()) == 0:
    st.error("One split has 0 samples. Check station name and date ranges.")
    st.stop()

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val     = X[val_mask], y[val_mask]
X_test, y_test   = X[test_mask], y[test_mask]
meta_test = meta_df[test_mask].reset_index(drop=True)

# -----------------------------
# Scaling (fit on TRAIN only)
# -----------------------------
(X_train_s, X_val_s, X_test_s,
 y_train_s, y_val_s, y_test_s,
 x_scaler, y_scaler) = fit_transform_scalers(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    x_scaler_type, y_scaler_type
)

# -----------------------------
# Train + Evaluate
# -----------------------------
st.subheader("Train on non-holdout stations and test on holdout")

if st.button("Train & evaluate holdout"):
    tf.random.set_seed(42)
    np.random.seed(42)

    model = build_lstm_model(
        input_timesteps=X_train_s.shape[1],
        input_features=X_train_s.shape[2],
        output_dim=y_train.shape[1],
        lstm_units1=int(lstm_units1),
        lstm_units2=int(lstm_units2),
        dropout=float(dropout),
        dense_units=int(dense_units),
        learning_rate=float(learning_rate),
    )

    es = callbacks.EarlyStopping(monitor="val_loss", patience=int(patience), restore_best_weights=True)

    with st.spinner("Training..."):
        hist = model.fit(
            X_train_s, y_train_s,
            validation_data=(X_val_s, y_val_s),
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=[es],
            verbose=0
        )

    st.success("Training finished ✅")

    # Predict on holdout test
    y_pred_test_s = model.predict(X_test_s, verbose=0)
    y_pred_test = inverse_y(y_pred_test_s, y_scaler)

    # Metrics
    df_depth = metrics_per_target(y_test, y_pred_test, target_names)
    st.write("### Holdout test metrics (per target/depth)")
    st.dataframe(df_depth, use_container_width=True)

    # Baseline if possible
    st.write("### Baseline comparison")
    baseline = None
    if feature_names:
        baseline = naive_baseline_last_step(X_test, y_test.shape[1], feature_names, target_names)

    if baseline is None:
        st.info("Naive baseline not available (targets are not present among features).")
        df_cmp = None
    else:
        df_base = metrics_per_target(y_test, baseline, target_names)
        df_cmp = df_depth.merge(df_base, on="target", suffixes=("_model", "_baseline"))
        st.dataframe(df_cmp, use_container_width=True)

    # Plots
    st.subheader("Holdout True vs Pred plots (first samples)")
    max_points = st.slider("Max points per plot", 100, 3000, 400, 100)
    figs = plot_true_vs_pred(y_test, y_pred_test, target_names, max_points=max_points)
    for fig in figs:
        st.pyplot(fig)

    # Export
    if export_to_disk:
        out_metrics = os.path.join(OUT6, f"holdout_metrics_{holdout_station}.csv")
        df_depth.to_csv(out_metrics, index=False)

        out_cmp = None
        if df_cmp is not None:
            out_cmp = os.path.join(OUT6, f"holdout_compare_{holdout_station}.csv")
            df_cmp.to_csv(out_cmp, index=False)

        run_json = os.path.join(OUT6, f"step6_run_{holdout_station}.json")
        run_info = {
            "inputs": {
                "X": os.path.join(OUT2, "X.npy"),
                "y": os.path.join(OUT2, "y.npy"),
                "meta": os.path.join(OUT2, "meta.csv"),
                "config": os.path.join(OUT2, "config.json"),
            },
            "holdout_station": holdout_station,
            "splits": {
                "train": {"start": train_start, "end": train_end, "n": int(train_mask.sum())},
                "val": {"start": val_start, "end": val_end, "n": int(val_mask.sum())},
                "test": {"start": test_start, "end": test_end, "n": int(test_mask.sum())},
            },
            "scaling": {"x_scaler": x_scaler_type, "y_scaler": y_scaler_type},
            "hyperparams": {
                "lstm_units1": int(lstm_units1),
                "lstm_units2": int(lstm_units2),
                "dropout": float(dropout),
                "dense_units": int(dense_units),
                "learning_rate": float(learning_rate),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "patience": int(patience),
            },
            "outputs": {
                "metrics": out_metrics,
                "compare": out_cmp,
            }
        }
        with open(run_json, "w", encoding="utf-8") as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2)

        if save_model:
            out_model = os.path.join(OUT6, f"model_holdout_{holdout_station}.keras")
            model.save(out_model)
            st.success(f"Saved model: {out_model}")
            st.code(out_model)

        st.success("Exported ✅")
        st.code("\n".join([p for p in [out_metrics, out_cmp, run_json] if p]))

    # Also store in session_state (optional, harmless)
    st.session_state["step6_holdout_metrics"] = df_depth
    if df_cmp is not None:
        st.session_state["step6_holdout_compare"] = df_cmp