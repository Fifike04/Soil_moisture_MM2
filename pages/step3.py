import os
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Step 3 — Train & Evaluate", layout="wide")
st.title("Step 3 — Train an LSTM model and evaluate")

OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
os.makedirs(OUT3, exist_ok=True)

# -----------------------------
# Load Step 2 artifacts (DISK ONLY)
# -----------------------------
def load_step2_from_outputs(out_dir: str):
    x_path = os.path.join(out_dir, "X.npy")
    y_path = os.path.join(out_dir, "y.npy")
    cfg_path = os.path.join(out_dir, "config.json")
    meta_csv = os.path.join(out_dir, "meta.csv")
    diag_csv = os.path.join(out_dir, "diagnostics.csv")
    ready_path = os.path.join(out_dir, "_READY.txt")

    missing = [p for p in [x_path, y_path, cfg_path] if not os.path.exists(p)]
    if missing:
        return None, None, None, None, None, missing

    # Optional: ready marker check (not required)
    ready_ok = os.path.exists(ready_path)

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    meta_df = pd.read_csv(meta_csv) if os.path.exists(meta_csv) else pd.DataFrame()
    diag_df = pd.read_csv(diag_csv) if os.path.exists(diag_csv) else pd.DataFrame()

    extra = {"ready_marker": ready_ok, "paths": {"X": x_path, "y": y_path, "config": cfg_path}}
    return X, y, cfg, meta_df, diag_df, extra


X, y, cfg, meta_df, diag_df, extra_or_missing = load_step2_from_outputs(OUT2)

if X is None:
    st.error("Nem találom a Step 2 kimenetét az outputs/step2/ mappában.")
    st.write("Hiányzó fájl(ok):")
    for p in extra_or_missing:
        st.code(p)
    st.info("Futtasd le a Step 2-t úgy, hogy kimentsen: X.npy, y.npy, config.json az outputs/step2/ mappába.")
    st.stop()

st.success("Step 2 adat betöltve az outputs/step2/ mappából ✅")
st.write("X shape:", X.shape)
st.write("y shape:", y.shape)

with st.expander("Step 2 config (config.json)"):
    st.json(cfg)

with st.expander("Step 2 diagnostics/meta (ha létezik)"):
    if not diag_df.empty:
        st.write("diagnostics.csv")
        st.dataframe(diag_df, use_container_width=True)
    else:
        st.info("diagnostics.csv nincs vagy üres.")
    if not meta_df.empty:
        st.write("meta.csv")
        st.dataframe(meta_df.head(30), use_container_width=True)
    else:
        st.info("meta.csv nincs vagy üres.")

# -----------------------------
# Train/val/test split
# -----------------------------
st.subheader("1) Split settings")

n = X.shape[0]
test_ratio = st.slider("Test ratio", 0.05, 0.30, 0.15, 0.01)
val_ratio = st.slider("Validation ratio", 0.05, 0.30, 0.15, 0.01)

# deterministic split (no shuffle): safer for timeseries
n_test = int(round(n * test_ratio))
n_val = int(round(n * val_ratio))
n_train = n - n_val - n_test
if n_train <= 0:
    st.error("Split arányok túl nagyok ehhez a mintaszámhoz.")
    st.stop()

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

st.write(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# -----------------------------
# Model config
# -----------------------------
st.subheader("2) Model settings")

units = st.number_input("LSTM units", min_value=8, max_value=512, value=64, step=8)
dropout = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05)
lr = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
batch_size = st.selectbox("Batch size", options=[16, 32, 64, 128, 256], index=1)
epochs = st.number_input("Epochs", min_value=1, max_value=500, value=30, step=1)
patience = st.number_input("EarlyStopping patience", min_value=2, max_value=50, value=8, step=1)

# -----------------------------
# Train
# -----------------------------
st.subheader("3) Train")

do_train = st.button("Train LSTM")

if do_train:
    try:
        import tensorflow as tf
        from tensorflow.keras import layers

        # output dim: if y is (N,) -> 1, if (N,h) -> h
        out_dim = 1 if y_train.ndim == 1 else y_train.shape[1]

        # Ensure y shapes compatible with keras
        y_train_fit = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        y_val_fit = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
        y_test_fit = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test

        tf.keras.backend.clear_session()

        model = tf.keras.Sequential([
            layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
            layers.LSTM(int(units), return_sequences=False),
            layers.Dropout(float(dropout)),
            layers.Dense(int(out_dim)),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
            loss="mse",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(patience),
                restore_best_weights=True
            )
        ]

        with st.spinner("Training..."):
            history = model.fit(
                X_train, y_train_fit,
                validation_data=(X_val, y_val_fit),
                epochs=int(epochs),
                batch_size=int(batch_size),
                verbose=0,
                callbacks=callbacks
            )

        # Evaluate
        test_res = model.evaluate(X_test, y_test_fit, verbose=0)
        test_loss = float(test_res[0])
        test_mae = float(test_res[1]) if len(test_res) > 1 else None

        st.success("Training kész ✅")
        st.write(f"Test MSE (loss): {test_loss:.6f}")
        if test_mae is not None:
            st.write(f"Test MAE: {test_mae:.6f}")

        # Predictions sample
        preds = model.predict(X_test, verbose=0)
        st.write("Predictions preview (first 10):")
        st.dataframe(
            {
                "y_true": (y_test_fit[:10].flatten().tolist()),
                "y_pred": (preds[:10].flatten().tolist())
            },
            use_container_width=True
        )

        # Save artifacts
        model_path = os.path.join(OUT3, "model.keras")
        model.save(model_path)

        hist_path = os.path.join(OUT3, "history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history.history, f, ensure_ascii=False, indent=2)

        metrics_path = os.path.join(OUT3, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_loss_mse": test_loss,
                    "test_mae": test_mae,
                    "n_train": int(X_train.shape[0]),
                    "n_val": int(X_val.shape[0]),
                    "n_test": int(X_test.shape[0]),
                    "step2_config": cfg,
                    "step2_source": OUT2,
                    "step2_ready_marker": bool(extra_or_missing.get("ready_marker", False)),
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        st.success("Mentve ✅ outputs/step3/model.keras + history.json + metrics.json")
        st.code(OUT3)

        with st.expander("History (loss/mae)"):
            st.json(history.history)

    except ModuleNotFoundError:
        st.error("TensorFlow nincs telepítve ebben a környezetben. Telepítsd: pip install tensorflow (vagy a környezetednek megfelelőt).")
    except Exception as e:
        st.error(f"Hiba training közben: {e}")