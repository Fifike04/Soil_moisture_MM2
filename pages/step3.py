import os
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Step 3 — Train & Evaluate (date split)", layout="wide")
st.title("Step 3 — Train an LSTM model and evaluate (DISK ONLY, date-based split)")

OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
os.makedirs(OUT3, exist_ok=True)

# -----------------------------
# Load Step2 artifacts (DISK ONLY)
# -----------------------------
def require(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()

x_path = os.path.join(OUT2, "X.npy")
y_path = os.path.join(OUT2, "y.npy")
cfg_path = os.path.join(OUT2, "config.json")
meta_csv = os.path.join(OUT2, "meta.csv")
diag_csv = os.path.join(OUT2, "diagnostics.csv")

require(x_path, "Step2 X.npy (outputs/step2/X.npy)")
require(y_path, "Step2 y.npy (outputs/step2/y.npy)")
require(cfg_path, "Step2 config.json (outputs/step2/config.json)")
require(meta_csv, "Step2 meta.csv (outputs/step2/meta.csv)")

X = np.load(x_path, allow_pickle=True)
y = np.load(y_path, allow_pickle=True)
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
meta_df = pd.read_csv(meta_csv)
diag_df = pd.read_csv(diag_csv) if os.path.exists(diag_csv) else pd.DataFrame()

# normalize y shape
if y.ndim == 1:
    y = y.reshape(-1, 1)

if X.ndim != 3 or y.ndim != 2:
    st.error(f"Bad shapes. Expected X 3D and y 2D. Got X={X.shape}, y={y.shape}")
    st.stop()

if "target_date" not in meta_df.columns or "station" not in meta_df.columns:
    st.error("meta.csv must contain columns: station, target_date")
    st.stop()

meta_df["target_date"] = pd.to_datetime(meta_df["target_date"], errors="coerce")
meta_df["station"] = meta_df["station"].astype(str).str.strip().str.lower()

st.success("Loaded Step2 artifacts ✅")
st.write("X:", X.shape, "y:", y.shape)

with st.expander("Step2 config.json"):
    st.json(cfg)

with st.expander("Step2 diagnostics"):
    if not diag_df.empty:
        st.dataframe(diag_df, use_container_width=True)
    else:
        st.info("No diagnostics.csv found.")

# -----------------------------
# Date-based split (cleanest)
# -----------------------------
st.subheader("1) Date-based split (by meta.target_date)")

with st.sidebar:
    st.header("Split ranges")
    train_start = st.text_input("Train start", value="2016-01-01")
    train_end   = st.text_input("Train end",   value="2022-12-31")
    val_start   = st.text_input("Val start",   value="2023-01-01")
    val_end     = st.text_input("Val end",     value="2023-12-31")
    test_start  = st.text_input("Test start",  value="2024-01-01")
    test_end    = st.text_input("Test end",    value="2025-12-31")

def mask_range(start: str, end: str) -> np.ndarray:
    d = meta_df["target_date"]
    return ((d >= pd.to_datetime(start)) & (d <= pd.to_datetime(end))).to_numpy()

train_mask = mask_range(train_start, train_end)
val_mask   = mask_range(val_start, val_end)
test_mask  = mask_range(test_start, test_end)

st.write({
    "train_samples": int(train_mask.sum()),
    "val_samples": int(val_mask.sum()),
    "test_samples": int(test_mask.sum())
})

if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0 or int(test_mask.sum()) == 0:
    st.error("One split has 0 samples. Adjust date ranges to match your dataset.")
    st.stop()

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val     = X[val_mask], y[val_mask]
X_test, y_test   = X[test_mask], y[test_mask]

# -----------------------------
# Scaling (X only, fit on TRAIN)
# -----------------------------
st.subheader("2) Scaling (X only, fit on TRAIN)")

with st.sidebar:
    st.header("Scaling")
    do_x_scaling = st.checkbox("Standardize X (recommended)", value=True)

def flatten_X(X3):
    n, t, f = X3.shape
    return X3.reshape(n * t, f)

def unflatten_X(Xflat, n, t):
    f = Xflat.shape[1]
    return Xflat.reshape(n, t, f)

x_scaler = None
if do_x_scaling:
    Xtr = flatten_X(X_train).astype(np.float32)
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd = np.where(sd == 0, 1.0, sd)

    def scale(X3):
        Xf = flatten_X(X3).astype(np.float32)
        Xf = (Xf - mu) / sd
        return unflatten_X(Xf, X3.shape[0], X3.shape[1]).astype(np.float32)

    X_train_s = scale(X_train)
    X_val_s   = scale(X_val)
    X_test_s  = scale(X_test)

    x_scaler = {"mean": mu.tolist(), "std": sd.tolist()}
    with open(os.path.join(OUT3, "x_scaler.json"), "w", encoding="utf-8") as f:
        json.dump(x_scaler, f, ensure_ascii=False, indent=2)

    st.success("Saved X scaler ✅ outputs/step3/x_scaler.json")
else:
    X_train_s, X_val_s, X_test_s = X_train, X_val, X_test

# -----------------------------
# Model config
# -----------------------------
st.subheader("3) Model settings")

with st.sidebar:
    st.header("Model")
    units1 = st.number_input("LSTM units (layer 1)", min_value=8, max_value=512, value=64, step=8)
    units2 = st.number_input("LSTM units (layer 2) (0=off)", min_value=0, max_value=512, value=32, step=8)
    dropout = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05)
    lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
    batch_size = st.selectbox("Batch size", options=[16, 32, 64, 128, 256], index=2)
    epochs = st.number_input("Epochs", min_value=1, max_value=500, value=30, step=1)
    patience = st.number_input("EarlyStopping patience", min_value=2, max_value=50, value=8, step=1)

# -----------------------------
# Train
# -----------------------------
st.subheader("4) Train")

if st.button("Train LSTM (SAVE to outputs/step3/)"):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers

        tf.random.set_seed(42)
        np.random.seed(42)
        tf.keras.backend.clear_session()

        out_dim = y_train.shape[1]

        model_layers = [
            layers.Input(shape=(X_train_s.shape[1], X_train_s.shape[2])),
            layers.LSTM(int(units1), return_sequences=(int(units2) > 0)),
            layers.Dropout(float(dropout)),
        ]
        if int(units2) > 0:
            model_layers += [
                layers.LSTM(int(units2), return_sequences=False),
                layers.Dropout(float(dropout)),
            ]
        model_layers += [layers.Dense(int(out_dim))]

        model = tf.keras.Sequential(model_layers)
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

        with st.spinner("Training..."):
            history = model.fit(
                X_train_s, y_train,
                validation_data=(X_val_s, y_val),
                epochs=int(epochs),
                batch_size=int(batch_size),
                verbose=0,
                callbacks=cb
            )

        # Evaluate
        test_res = model.evaluate(X_test_s, y_test, verbose=0)
        test_loss = float(test_res[0])
        test_mae = float(test_res[1]) if len(test_res) > 1 else None

        st.success("Training done ✅")
        st.write(f"Test MSE: {test_loss:.6f}")
        if test_mae is not None:
            st.write(f"Test MAE: {test_mae:.6f}")

        # Save model + history + metrics
        model_path = os.path.join(OUT3, "model.keras")
        model.save(model_path)

        with open(os.path.join(OUT3, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history.history, f, ensure_ascii=False, indent=2)

        metrics = {
            "test_loss_mse": test_loss,
            "test_mae": test_mae,
            "n_train": int(X_train_s.shape[0]),
            "n_val": int(X_val_s.shape[0]),
            "n_test": int(X_test_s.shape[0]),
            "date_split": {
                "train": {"start": train_start, "end": train_end},
                "val": {"start": val_start, "end": val_end},
                "test": {"start": test_start, "end": test_end},
            },
            "input_scaling": bool(do_x_scaling),
            "step2_config": cfg,
            "step2_source": OUT2,
        }
        with open(os.path.join(OUT3, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        st.success("Saved ✅ outputs/step3/model.keras + history.json + metrics.json (+ x_scaler.json if enabled)")
        st.code(OUT3)

        # quick preview
        preds = model.predict(X_test_s, verbose=0)
        st.write("Predictions preview (first 10 rows):")
        df_prev = pd.DataFrame({
            "y_true_first_target": y_test[:10, 0].tolist(),
            "y_pred_first_target": preds[:10, 0].tolist(),
        })
        st.dataframe(df_prev, use_container_width=True)

    except ModuleNotFoundError:
        st.error("TensorFlow is not installed in this environment.")
    except Exception as e:
        st.error(f"Training error: {e}")