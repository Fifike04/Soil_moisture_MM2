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
st.title("Step 6 — Station holdout generalization test")

OUT2 = os.path.join("outputs", "step2")
OUT6 = os.path.join("outputs", "step6")
os.makedirs(OUT6, exist_ok=True)

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


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))


def rel_error_pct(y_true, y_pred):
    return float(np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100.0)


def build_lstm_model(
    input_timesteps,
    input_features,
    output_dim,
    lstm_units1,
    lstm_units2,
    dropout,
    dense_units,
    learning_rate,
):
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
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


def flatten_X(X):
    n, t, f = X.shape
    return X.reshape(n * t, f)


def unflatten_X(X_flat, n, t):
    f = X_flat.shape[1]
    return X_flat.reshape(n, t, f)


def make_scaler(name: str):
    if name == "StandardScaler":
        return StandardScaler()
    if name == "MinMaxScaler":
        return MinMaxScaler()
    return None


def fit_transform_scalers(X_train, X_val, X_test, y_train, y_val, y_test, x_scaler_type, y_scaler_type):
    x_scaler = make_scaler(x_scaler_type)
    y_scaler = make_scaler(y_scaler_type)

    # X scaling (fit TRAIN only)
    if x_scaler is not None:
        x_scaler.fit(flatten_X(X_train))
        X_train_s = unflatten_X(x_scaler.transform(flatten_X(X_train)), X_train.shape[0], X_train.shape[1])
        X_val_s   = unflatten_X(x_scaler.transform(flatten_X(X_val)),   X_val.shape[0],   X_val.shape[1])
        X_test_s  = unflatten_X(x_scaler.transform(flatten_X(X_test)),  X_test.shape[0],  X_test.shape[1])
    else:
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test

    # y scaling (fit TRAIN only)
    if y_scaler is not None:
        y_scaler.fit(y_train)
        y_train_s = y_scaler.transform(y_train)
        y_val_s   = y_scaler.transform(y_val)
        y_test_s  = y_scaler.transform(y_test)
    else:
        y_train_s, y_val_s, y_test_s = y_train, y_val, y_test

    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, x_scaler, y_scaler


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
            "R2": float(r2_score(yt, yp)),
            "Bias": bias(yt, yp),
            "Relative_Error_%": rel_error_pct(yt, yp),
        })
    return pd.DataFrame(rows)


def naive_baseline_last_step(X_test, feature_names, target_names):
    """
    Baseline = predict target(t) ~= same variable at last timestep of input window.
    Only works if target cols are present among final_feature_cols.
    """
    idxs = []
    for t in target_names:
        if t in feature_names:
            idxs.append(feature_names.index(t))
        else:
            return None

    last_step = X_test[:, -1, :]
    return last_step[:, idxs]


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
        config = json.load(f)

except Exception as e:
    st.error(f"Failed to load Step2 outputs: {e}")
    st.stop()

if y.ndim == 1:
    y = y.reshape(-1, 1)

if X.ndim != 3 or y.ndim != 2:
    st.error(f"Expected X 3D and y 2D. Got X={X.shape}, y={y.shape}")
    st.stop()

if "station" not in meta_df.columns or "target_date" not in meta_df.columns:
    st.error("meta.csv must contain columns: 'station' and 'target_date'.")
    st.stop()

if len(meta_df) != len(y):
    st.error(f"meta.csv row count must match y length. meta={len(meta_df)}, y={len(y)}")
    st.stop()

meta_df["station"] = meta_df["station"].astype(str).str.strip().str.lower()
meta_df["target_date"] = pd.to_datetime(meta_df["target_date"], errors="coerce")

feature_names = config.get("final_feature_cols", [])
target_names = config.get("target_cols", [f"target{i}" for i in range(y.shape[1])])

if len(target_names) != y.shape[1]:
    target_names = [f"target{i}" for i in range(y.shape[1])]

st.success("Step2 outputs loaded successfully ✅")
st.write({
    "samples": int(X.shape[0]),
    "timesteps": int(X.shape[1]),
    "features": int(X.shape[2]),
    "targets": int(y.shape[1]),
})

with st.expander("Step2 config.json"):
    st.json(config)

# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------
with st.sidebar:
    st.header("Holdout station")
    unique_stations = sorted(meta_df["station"].dropna().unique().tolist())
    default_station = "csengele" if "csengele" in unique_stations else (unique_stations[0] if unique_stations else "")
    holdout_station = st.selectbox(
        "Holdout station",
        options=unique_stations,
        index=unique_stations.index(default_station) if default_station in unique_stations else 0
    )

    st.divider()
    st.header("Date ranges")
    test_start = st.text_input("Test start (holdout)", value="2024-01-01")
    test_end   = st.text_input("Test end (holdout)",   value="2025-12-31")

    val_start = st.text_input("Val start (non-holdout)", value="2023-01-01")
    val_end   = st.text_input("Val end (non-holdout)",   value="2023-12-31")

    train_start = st.text_input("Train start (non-holdout)", value="2016-01-01")
    train_end   = st.text_input("Train end (non-holdout)",   value="2022-12-31")

    st.divider()
    st.header("Scaling")
    x_scaler_type = st.selectbox("X scaler", ["StandardScaler", "MinMaxScaler", "No scaling"], index=0)
    y_scaler_type = st.selectbox("y scaler", ["No scaling", "StandardScaler", "MinMaxScaler"], index=0)

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

# --------------------------------------------------
# Build masks
# --------------------------------------------------
st.subheader("2) Build holdout split")

holdout_mask = (meta_df["station"] == holdout_station).to_numpy()

test_mask = holdout_mask & date_mask(meta_df, test_start, test_end)
non_holdout = ~holdout_mask
train_mask = non_holdout & date_mask(meta_df, train_start, train_end)
val_mask   = non_holdout & date_mask(meta_df, val_start, val_end)

st.write("Split counts:")
st.json({
    "train (non-holdout)": int(train_mask.sum()),
    "val (non-holdout)": int(val_mask.sum()),
    "test (holdout)": int(test_mask.sum())
})

if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0 or int(test_mask.sum()) == 0:
    st.error("One split has 0 samples. Check station name and date ranges.")
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    if d.notna().any():
        st.info(f"Available target_date range: {d.min()} — {d.max()}")
    st.write("Station counts:")
    station_counts = meta_df["station"].value_counts().reset_index()
    station_counts.columns = ["station", "count"]
    st.dataframe(station_counts, use_container_width=True)
    st.stop()

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val     = X[val_mask], y[val_mask]
X_test, y_test   = X[test_mask], y[test_mask]
meta_test = meta_df[test_mask].reset_index(drop=True)

# --------------------------------------------------
# Scale
# --------------------------------------------------
st.subheader("3) Scaling")

(
    X_train_s, X_val_s, X_test_s,
    y_train_s, y_val_s, y_test_s,
    x_scaler, y_scaler
) = fit_transform_scalers(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    x_scaler_type,
    y_scaler_type if y_scaler_type != "No scaling" else "No scaling",
)

st.success("Scaling prepared ✅")

# --------------------------------------------------
# Train + Evaluate
# --------------------------------------------------
st.subheader("4) Train on non-holdout stations and test on holdout")

if st.button("Train & evaluate holdout"):
    tf.keras.backend.clear_session()
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

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(patience),
        restore_best_weights=True
    )

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

    # Predict on holdout
    y_pred_test_s = model.predict(X_test_s, verbose=0)
    y_pred_test = inverse_y(y_pred_test_s, y_scaler)

    # Metrics
    df_depth = metrics_per_target(y_test, y_pred_test, target_names)
    st.write("### Holdout test metrics (per target)")
    st.dataframe(df_depth, use_container_width=True)

    global_summary = {
        "MAE_mean_over_targets": float(df_depth["MAE"].mean()),
        "RMSE_mean_over_targets": float(df_depth["RMSE"].mean()),
        "R2_mean_over_targets": float(df_depth["R2"].mean()),
        "Bias_mean_over_targets": float(df_depth["Bias"].mean()),
        "Relative_Error_%_mean_over_targets": float(df_depth["Relative_Error_%"].mean()),
    }
    st.write("### Global summary")
    st.json(global_summary)

    # Baseline
    st.write("### Baseline comparison")
    baseline = None
    if feature_names:
        baseline = naive_baseline_last_step(X_test, feature_names, target_names)

    if baseline is None:
        st.info("Naive baseline not available (targets are not present among features).")
        df_cmp = None
    else:
        df_base = metrics_per_target(y_test, baseline, target_names)
        df_cmp = df_depth.merge(df_base, on="target", suffixes=("_model", "_baseline"))
        st.dataframe(df_cmp, use_container_width=True)

    # Plots
    st.subheader("Holdout True vs Pred plots")
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
                "X": X_PATH,
                "y": Y_PATH,
                "meta": META_PATH,
                "config": CFG_PATH,
            },
            "holdout_station": holdout_station,
            "splits": {
                "train": {"start": train_start, "end": train_end, "n": int(train_mask.sum())},
                "val": {"start": val_start, "end": val_end, "n": int(val_mask.sum())},
                "test": {"start": test_start, "end": test_end, "n": int(test_mask.sum())},
            },
            "scaling": {
                "x_scaler": x_scaler_type,
                "y_scaler": y_scaler_type
            },
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
            "test_overall": global_summary,
            "outputs": {
                "metrics": out_metrics,
                "compare": out_cmp,
            }
        }

        with open(run_json, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(run_info), f, ensure_ascii=False, indent=2)

        if save_model:
            out_model = os.path.join(OUT6, f"model_holdout_{holdout_station}.keras")
            model.save(out_model)
            st.success(f"Saved model: {out_model}")
            st.code(out_model)

        st.success("Step 6 saved successfully ✅")
        paths = [out_metrics, run_json]
        if out_cmp is not None:
            paths.insert(1, out_cmp)
        st.code("\n".join(paths))