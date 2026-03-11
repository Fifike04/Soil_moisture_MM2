import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Step 7 — Seasonality & regime report", layout="wide")
st.title("Step 7 — Seasonality & regime performance report")

OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
OUT7 = os.path.join("outputs", "step7")
os.makedirs(OUT7, exist_ok=True)

# --------------------------------------------------
# Paths
# --------------------------------------------------
X_PATH = os.path.join(OUT2, "X.npy")
Y_PATH = os.path.join(OUT2, "y.npy")
META_PATH = os.path.join(OUT2, "meta.csv")
CFG_PATH = os.path.join(OUT2, "config.json")
MODEL_PATH = os.path.join(OUT3, "model.keras")
X_SCALER_PATH = os.path.join(OUT3, "x_scaler.json")

MONTH_OUT = os.path.join(OUT7, "monthly_pooled_metrics.csv")
SEASON_OUT = os.path.join(OUT7, "seasonal_pooled_metrics.csv")
PER_TARGET_SEASON_OUT = os.path.join(OUT7, "per_target_by_season.csv")
REGIME_OUT = os.path.join(OUT7, "regime_metrics.csv")
SUMMARY_OUT = os.path.join(OUT7, "step7_summary.json")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def require_file(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()


def season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def date_mask(meta_df: pd.DataFrame, start: str, end: str) -> np.ndarray:
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    return ((d >= pd.to_datetime(start)) & (d <= pd.to_datetime(end))).to_numpy()


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "MAE_pooled": float(np.mean(np.abs(err))),
        "RMSE_pooled": float(np.sqrt(np.mean(err ** 2))),
    }


def per_target_metrics_numpy(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> pd.DataFrame:
    rows = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        err = yp - yt

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))

        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

        rows.append({
            "target": target_names[j] if j < len(target_names) else f"target{j}",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
    return pd.DataFrame(rows)


def plot_month_bar(df_month: pd.DataFrame, metric_col: str, title: str):
    fig = plt.figure()
    x = df_month["month"].astype(int).to_numpy()
    y = df_month[metric_col].to_numpy()
    plt.bar(x, y)
    plt.xticks(range(1, 13))
    plt.xlabel("Month")
    plt.ylabel(metric_col)
    plt.title(title)
    return fig


def plot_season_bar(df_season: pd.DataFrame, metric_col: str, title: str):
    order = ["DJF", "MAM", "JJA", "SON"]
    fig = plt.figure()
    df = df_season.set_index("season").reindex(order).reset_index()
    plt.bar(df["season"], df[metric_col])
    plt.xlabel("Season")
    plt.ylabel(metric_col)
    plt.title(title)
    return fig


def flatten_X(X: np.ndarray) -> np.ndarray:
    n, t, f = X.shape
    return X.reshape(n * t, f)


def unflatten_X(X_flat: np.ndarray, n: int, t: int) -> np.ndarray:
    f = X_flat.shape[1]
    return X_flat.reshape(n, t, f)


def apply_x_scaler(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Xf = flatten_X(X).astype(np.float32)
    Xf = (Xf - mu) / sd
    return unflatten_X(Xf, X.shape[0], X.shape[1]).astype(np.float32)


def load_x_scaler(path: str):
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            sc = json.load(f)
        mu = np.asarray(sc["mean"], dtype=np.float32)
        sd = np.asarray(sc["std"], dtype=np.float32)
        sd = np.where(sd == 0, 1.0, sd).astype(np.float32)
        return mu, sd
    except Exception:
        return None, None


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

if X.ndim != 3 or y.ndim != 2:
    st.error(f"Expected X 3D and y 2D. Got X={X.shape}, y={y.shape}")
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
    "targets": target_names,
    "n_features": int(X.shape[2]),
})

with st.expander("Step2 config"):
    st.json(step2_cfg)

# --------------------------------------------------
# Load Step3 scaler
# --------------------------------------------------
st.subheader("2) Load Step3 input scaler")

mu, sd = load_x_scaler(X_SCALER_PATH)
if mu is not None and sd is not None:
    st.success("Step3 x_scaler.json loaded ✅")
    st.write(f"Scaler feature dimension: {len(mu)}")
else:
    st.warning("No valid x_scaler.json found. Step 7 will run without scaling.")
    mu, sd = None, None

if mu is not None and len(mu) != X.shape[2]:
    st.warning(
        f"Scaler dimension ({len(mu)}) does not match X feature dimension ({X.shape[2]}). "
        "Scaling will be skipped."
    )
    mu, sd = None, None

# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------
with st.sidebar:
    st.header("Test date range (by target_date)")
    test_start = st.text_input("Test start", value="2024-01-01")
    test_end = st.text_input("Test end", value="2025-12-31")

    st.divider()
    st.header("Scaling")
    apply_scaling = st.checkbox("Apply Step3 X scaling", value=True)

    st.divider()
    st.header("Wet/Dry regime (optional)")
    regime_on = st.checkbox("Enable wet/dry regime split", value=False)
    regime_proxy = st.text_input("Regime proxy column name (must exist in meta.csv)", value="_NapiCsapadek_sum7")
    wet_quantile = st.slider("Wet threshold quantile", 0.5, 0.95, 0.8, 0.05)

    st.divider()
    st.header("Export")
    export_to_disk = st.checkbox("Export CSVs to outputs/step7/", value=True)

# --------------------------------------------------
# Build test subset
# --------------------------------------------------
st.subheader("3) Build test subset")

test_idx = date_mask(meta_df, test_start, test_end)
n_test = int(test_idx.sum())

st.write({
    "test_samples": n_test,
    "test_range": [test_start, test_end]
})

if n_test == 0:
    st.error("No samples in test range. Adjust test_start/test_end.")
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    if d.notna().any():
        st.info(f"Available target_date range: {d.min()} — {d.max()}")
    st.stop()

X_test = X[test_idx]
y_test = y[test_idx]
meta_test = meta_df.loc[test_idx].reset_index(drop=True)

meta_test["month"] = meta_test["target_date"].dt.month.astype("Int64")
meta_test["season"] = meta_test["month"].astype(int).apply(season_from_month)

# --------------------------------------------------
# Predict
# --------------------------------------------------
st.subheader("4) Predict on test subset")

if apply_scaling and mu is not None and sd is not None:
    X_test_used = apply_x_scaler(X_test, mu, sd)
    st.info("Applied Step3 scaling to X_test.")
else:
    X_test_used = X_test
    st.info("No scaling applied to X_test.")

with st.spinner("Predicting..."):
    y_pred = model.predict(X_test_used, verbose=0)

if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

if y_pred.shape != y_test.shape:
    st.error(f"Prediction shape mismatch: y_pred={y_pred.shape}, y_true={y_test.shape}")
    st.stop()

# --------------------------------------------------
# Monthly report
# --------------------------------------------------
st.subheader("5) Monthly performance (pooled across targets)")

month_rows = []
for m in range(1, 13):
    idx = (meta_test["month"] == m).to_numpy()
    if idx.sum() == 0:
        month_rows.append({
            "month": m,
            "n": 0,
            "MAE_pooled": np.nan,
            "RMSE_pooled": np.nan
        })
        continue

    met = pooled_metrics(y_test[idx], y_pred[idx])
    month_rows.append({
        "month": m,
        "n": int(idx.sum()),
        **met
    })

df_month = pd.DataFrame(month_rows)
st.dataframe(df_month, use_container_width=True)
st.pyplot(plot_month_bar(df_month.dropna(), "RMSE_pooled", "Monthly RMSE (pooled)"))

# --------------------------------------------------
# Seasonal report
# --------------------------------------------------
st.subheader("6) Seasonal performance (pooled across targets)")

season_rows = []
for s in ["DJF", "MAM", "JJA", "SON"]:
    idx = (meta_test["season"] == s).to_numpy()
    if idx.sum() == 0:
        season_rows.append({
            "season": s,
            "n": 0,
            "MAE_pooled": np.nan,
            "RMSE_pooled": np.nan
        })
        continue

    met = pooled_metrics(y_test[idx], y_pred[idx])
    season_rows.append({
        "season": s,
        "n": int(idx.sum()),
        **met
    })

df_season = pd.DataFrame(season_rows)
st.dataframe(df_season, use_container_width=True)
st.pyplot(plot_season_bar(df_season.dropna(), "RMSE_pooled", "Seasonal RMSE (pooled)"))

# --------------------------------------------------
# Per-target by season
# --------------------------------------------------
st.subheader("7) Per-target metrics by season")

blocks = []
for s in ["DJF", "MAM", "JJA", "SON"]:
    idx = (meta_test["season"] == s).to_numpy()
    if idx.sum() == 0:
        continue

    df_t = per_target_metrics_numpy(y_test[idx], y_pred[idx], target_names)
    df_t.insert(0, "season", s)
    blocks.append(df_t)

if blocks:
    df_depth_season = pd.concat(blocks, ignore_index=True)
    st.dataframe(df_depth_season, use_container_width=True)
else:
    df_depth_season = None
    st.info("No samples for seasonal per-target breakdown.")

# --------------------------------------------------
# Wet/Dry regime
# --------------------------------------------------
st.subheader("8) Wet/Dry regime report (optional)")

df_reg = None
if regime_on and regime_proxy in meta_test.columns:
    proxy = pd.to_numeric(meta_test[regime_proxy], errors="coerce")
    thr = proxy.quantile(wet_quantile)

    is_wet = (proxy >= thr).to_numpy()
    is_dry = (proxy < thr).to_numpy()

    wet_metrics = pooled_metrics(y_test[is_wet], y_pred[is_wet]) if is_wet.sum() > 0 else {"MAE_pooled": np.nan, "RMSE_pooled": np.nan}
    dry_metrics = pooled_metrics(y_test[is_dry], y_pred[is_dry]) if is_dry.sum() > 0 else {"MAE_pooled": np.nan, "RMSE_pooled": np.nan}

    df_reg = pd.DataFrame([
        {"regime": "wet", "n": int(is_wet.sum()), **wet_metrics},
        {"regime": "dry", "n": int(is_dry.sum()), **dry_metrics},
    ])

    st.write({
        "proxy": regime_proxy,
        "wet_quantile": wet_quantile,
        "threshold": float(thr)
    })
    st.dataframe(df_reg, use_container_width=True)

elif regime_on:
    st.info(f"Regime proxy '{regime_proxy}' not found in meta.csv.")

# --------------------------------------------------
# Save outputs
# --------------------------------------------------
st.subheader("9) Save outputs")

if st.button("Save report to outputs/step7"):
    try:
        saved_paths = []

        df_month.to_csv(MONTH_OUT, index=False)
        saved_paths.append(MONTH_OUT)

        df_season.to_csv(SEASON_OUT, index=False)
        saved_paths.append(SEASON_OUT)

        if df_depth_season is not None:
            df_depth_season.to_csv(PER_TARGET_SEASON_OUT, index=False)
            saved_paths.append(PER_TARGET_SEASON_OUT)

        if df_reg is not None:
            df_reg.to_csv(REGIME_OUT, index=False)
            saved_paths.append(REGIME_OUT)

        summary = {
            "inputs": {
                "X": X_PATH,
                "y": Y_PATH,
                "meta": META_PATH,
                "config": CFG_PATH,
                "model": MODEL_PATH,
                "x_scaler": X_SCALER_PATH if (apply_scaling and mu is not None) else None,
            },
            "test_range": {
                "start": test_start,
                "end": test_end,
                "n": int(n_test)
            },
            "outputs": {
                "monthly": MONTH_OUT,
                "seasonal": SEASON_OUT,
                "per_target_by_season": PER_TARGET_SEASON_OUT if df_depth_season is not None else None,
                "regime": REGIME_OUT if df_reg is not None else None,
                "summary": SUMMARY_OUT
            }
        }

        with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(summary), f, ensure_ascii=False, indent=2)
        saved_paths.append(SUMMARY_OUT)

        st.success("Step 7 saved successfully ✅")
        st.code("\n".join(saved_paths))

    except Exception as e:
        st.error(f"Failed to save Step7 outputs: {e}")