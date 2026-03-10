import os
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Step 3 — Train & Evaluate", layout="wide")
st.title("Step 3 — Train an LSTM model and evaluate")

# --------------------------------------------------
# Paths
# --------------------------------------------------
STEP2_DIR = os.path.join("outputs", "step2")
STEP3_DIR = os.path.join("outputs", "step3")
os.makedirs(STEP3_DIR, exist_ok=True)

X_PATH = os.path.join(STEP2_DIR, "X.npy")
Y_PATH = os.path.join(STEP2_DIR, "y.npy")
META_PATH = os.path.join(STEP2_DIR, "meta.csv")
CONFIG_PATH = os.path.join(STEP2_DIR, "config.json")

MODEL_PATH = os.path.join(STEP3_DIR, "model.keras")
HISTORY_PATH = os.path.join(STEP3_DIR, "history.json")
METRICS_PATH = os.path.join(STEP3_DIR, "metrics.json")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def require_file(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()

def load_step2_outputs():
    require_file(X_PATH, "Step2 X.npy")
    require_file(Y_PATH, "Step2 y.npy")
    require_file(META_PATH, "Step2 meta.csv")
    require_file(CONFIG_PATH, "Step2 config.json")

    X = np.load(X_PATH, allow_pickle=True)
    y = np.load(Y_PATH, allow_pickle=True)
    meta_df = pd.read_csv(META_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return X, y, meta_df, config

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

try:
    X, y, meta_df, step2_config = load_step2_outputs()
except Exception as e:
    st.error(f"Failed to load Step2 outputs: {e}")
    st.stop()

if y.ndim == 1:
    y = y.reshape(-1, 1)

if X.ndim != 3:
    st.error(f"Expected X to be 3D, got shape: {X.shape}")
    st.stop()

if y.ndim != 2:
    st.error(f"Expected y to be 2D, got shape: {y.shape}")
    st.stop()

if len(meta_df) != len(y):
    st.error(f"meta.csv row count must match y length. meta={len(meta_df)}, y={len(y)}")
    st.stop()

if "target_date" not in meta_df.columns:
    st.error("meta.csv must contain 'target_date' column.")
    st.stop()

meta_df["target_date"] = pd.to_datetime(meta_df["target_date"], errors="coerce")

st.success("Step2 outputs loaded successfully ✅")
st.write({
    "X_shape": tuple(X.shape),
    "y_shape": tuple(y.shape),
    "meta_rows": int(len(meta_df)),
    "targets": step2_config.get("target_cols", []),
})
with st.expander("Step2 config.json"):
    st.json(step2_config)

# --------------------------------------------------
# Split settings
# --------------------------------------------------
st.subheader("2) Split settings")

split_mode = st.radio(
    "Split method",
    ["Chronological ratio split", "Date-based split"],
    index=0
)

if split_mode == "Chronological ratio split":
    c1, c2 = st.columns(2)
    with c1:
        test_ratio = st.slider("Test ratio", 0.05, 0.30, 0.15, 0.01)
    with c2:
        val_ratio = st.slider("Validation ratio", 0.05, 0.30, 0.15, 0.01)

    n = X.shape[0]
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test

    if n_train <= 0:
        st.error("Split ratios are too large for the sample count.")
        st.stop()

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    split_info = {
        "mode": "chronological_ratio",
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "test_ratio": float(test_ratio),
        "val_ratio": float(val_ratio),
    }

else:
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

    d = pd.to_datetime(meta_df["target_date"], errors="coerce")

    train_idx = ((d >= pd.to_datetime(train_start)) & (d <= pd.to_datetime(train_end))).to_numpy()
    val_idx   = ((d >= pd.to_datetime(val_start)) & (d <= pd.to_datetime(val_end))).to_numpy()
    test_idx  = ((d >= pd.to_datetime(test_start)) & (d <= pd.to_datetime(test_end))).to_numpy()

    if int(train_idx.sum()) == 0 or int(val_idx.sum()) == 0 or int(test_idx.sum()) == 0:
        st.error("One split has 0 samples. Check your date ranges.")
        if d.notna().any():
            st.info(f"Available target_date range: {d.min()} — {d.max()}")
        st.stop()

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx], y[val_idx]
    X_test, y_test   = X[test_idx], y[test_idx]

    split_info = {
        "mode": "date_based",
        "train": {"start": train_start, "end": train_end, "n": int(train_idx.sum())},
        "val":   {"start": val_start,   "end": val_end,   "n": int(val_idx.sum())},
        "test":  {"start": test_start,  "end": test_end,  "n": int(test_idx.sum())},
    }

st.write("Split summary:")
st.json(split_info)

# --------------------------------------------------
# Model settings
# --------------------------------------------------
st.subheader("3) Model settings")

c1, c2, c3, c4 = st.columns(4)
with c1:
    units = st.number_input("LSTM units", min_value=8, max_value=512, value=64, step=8)
with c2:
    dropout = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05)
with c3:
    lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
with c4:
    batch_size = st.selectbox("Batch size", options=[16, 32, 64, 128, 256], index=1)

epochs = st.number_input("Epochs", min_value=1, max_value=500, value=30, step=1)
patience = st.number_input("EarlyStopping patience", min_value=2, max_value=50, value=8, step=1)

# --------------------------------------------------
# Train
# --------------------------------------------------
st.subheader("4) Train model")

if st.button("Train LSTM + save to outputs/step3"):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers

        tf.keras.backend.clear_session()

        model = tf.keras.Sequential([
            layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
            layers.LSTM(int(units), return_sequences=False),
            layers.Dropout(float(dropout)),
            layers.Dense(y_train.shape[1]),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
            loss="mse",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
        )

        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(patience),
                restore_best_weights=True
            )
        ]

        with st.spinner("Training model..."):
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=int(epochs),
                batch_size=int(batch_size),
                verbose=0,
                callbacks=cb
            )

        # Evaluate
        test_res = model.evaluate(X_test, y_test, verbose=0)
        test_mse = float(test_res[0])
        test_mae = float(test_res[1]) if len(test_res) > 1 else None

        # Predict sample preview
        preds = model.predict(X_test, verbose=0)

        st.success("Training finished ✅")
        st.write(f"Test MSE: {test_mse:.6f}")
        if test_mae is not None:
            st.write(f"Test MAE: {test_mae:.6f}")

        st.write("Prediction preview (first 10 rows):")
        preview_df = pd.DataFrame({
            "y_true": y_test[:10].reshape(-1).tolist()[:10],
            "y_pred": preds[:10].reshape(-1).tolist()[:10],
        })
        st.dataframe(preview_df, use_container_width=True)

        # Save outputs
        model.save(MODEL_PATH)

        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(history.history), f, ensure_ascii=False, indent=2)

        metrics = {
            "test_mse": test_mse,
            "test_mae": test_mae,
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "split_info": split_info,
            "step2_config": step2_config,
            "model_config": {
                "lstm_units": int(units),
                "dropout": float(dropout),
                "learning_rate": float(lr),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "patience": int(patience),
            }
        }

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(metrics), f, ensure_ascii=False, indent=2)

        # Optional session_state
        st.session_state["step3_metrics"] = metrics
        st.session_state["step3_history"] = history.history

        st.success("Step 3 saved successfully ✅")
        st.code(STEP3_DIR)

        with st.expander("Training history"):
            st.json(history.history)

    except ModuleNotFoundError:
        st.error("TensorFlow is not installed in this environment.")
    except Exception as e:
        st.error(f"Training failed: {e}")