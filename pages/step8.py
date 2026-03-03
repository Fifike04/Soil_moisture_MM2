import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Step 8 — Multi-step forecasting", layout="wide")
st.title("Step 8 — Multi-step forecasting (DISK ONLY from outputs/)")

STATION_COL = "station"
DATE_COL = "date"

OUT1 = os.path.join("outputs", "step1")
OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
OUT8 = os.path.join("outputs", "step8")
os.makedirs(OUT8, exist_ok=True)

DEFAULT_TARGETS = ["_SM10", "_SM20", "_SM30", "_SM45", "_SM60", "_SM75"]

# -----------------------------
# Helpers
# -----------------------------
def require(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()

def load_step1_clean():
    cand_csv = os.path.join(OUT1, "combined_stations_clean_step1.csv")
    cand_xlsx = os.path.join(OUT1, "combined_stations_clean_step1.xlsx")
    if os.path.exists(cand_csv):
        return pd.read_csv(cand_csv), cand_csv
    if os.path.exists(cand_xlsx):
        return pd.read_excel(cand_xlsx), cand_xlsx
    return None, ""

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\t", "") for c in df.columns]
    if STATION_COL not in df.columns or DATE_COL not in df.columns:
        raise ValueError(f"Missing required columns: {STATION_COL}, {DATE_COL}")
    df[STATION_COL] = df[STATION_COL].astype(str).str.strip().str.lower()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([STATION_COL, DATE_COL])
    return df

def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if out[c].dtype == object:
            s = (
                out[c].astype(str).str.strip().str.replace(",", ".", regex=False)
                .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            )
            out[c] = pd.to_numeric(s, errors="coerce")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def apply_saved_scaler_if_exists(X_3d: np.ndarray, scaler_path: str) -> tuple[np.ndarray, dict | None]:
    if not os.path.exists(scaler_path):
        return X_3d, None
    try:
        with open(scaler_path, "r", encoding="utf-8") as f:
            sc = json.load(f)
        mu = np.asarray(sc["mean"], dtype=np.float32)
        sd = np.asarray(sc["std"], dtype=np.float32)
        sd = np.where(sd == 0, 1.0, sd).astype(np.float32)

        if mu.shape[0] != X_3d.shape[-1]:
            return X_3d, None

        return ((X_3d.astype(np.float32) - mu) / sd).astype(np.float32), sc
    except Exception:
        return X_3d, None

def update_feature_vector_autoregressive(
    last_feat_vec: np.ndarray,
    feature_cols: list[str],
    target_cols: list[str],
    y_pred: np.ndarray,
    max_lag: int = 60,
):
    """
    Updates ONLY what is safe:
    - base target features: _SM10 etc (if they exist among features)
    - target lags: _SM10_lag1.. etc (if they exist)
    Everything else persists.
    """
    next_vec = last_feat_vec.copy()
    idx = {c: j for j, c in enumerate(feature_cols)}
    pred_map = {t: float(y_pred[i]) for i, t in enumerate(target_cols) if i < len(y_pred)}

    # base target features updated to predicted
    for t, val in pred_map.items():
        if t in idx:
            next_vec[idx[t]] = val

    # shift lags for each target, if present
    for t in target_cols:
        if t not in idx:
            continue

        prev_base = float(last_feat_vec[idx[t]])

        # lag1 <= previous base
        n1 = f"{t}_lag1"
        if n1 in idx:
            next_vec[idx[n1]] = prev_base

        # lagk <= previous lag(k-1)
        for k in range(2, max_lag + 1):
            nk = f"{t}_lag{k}"
            nkm1 = f"{t}_lag{k-1}"
            if nk in idx and nkm1 in idx:
                next_vec[idx[nk]] = float(last_feat_vec[idx[nkm1]])

    return next_vec

# -----------------------------
# Load from outputs/
# -----------------------------
cfg_path = os.path.join(OUT2, "config.json")
model_path = os.path.join(OUT3, "model.keras")
scaler_path = os.path.join(OUT3, "x_scaler.json")

require(cfg_path, "Step2 config.json")
require(model_path, "Step3 model.keras")

with open(cfg_path, "r", encoding="utf-8") as f:
    step2_cfg = json.load(f)

window = int(step2_cfg.get("window", 30))
final_feature_cols = step2_cfg.get("final_feature_cols", [])
target_cols = step2_cfg.get("target_cols", DEFAULT_TARGETS)

if not final_feature_cols:
    st.error("Step2 config.json missing 'final_feature_cols'.")
    st.stop()

# Load model
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error("Failed to load model.keras")
    st.exception(e)
    st.stop()

st.subheader("Config summary (Step2)")
st.write({
    "window": window,
    "targets": target_cols,
    "n_feature_cols": len(final_feature_cols),
    "model": model_path,
    "scaler": scaler_path if os.path.exists(scaler_path) else None,
})
with st.expander("Step2 config.json"):
    st.json(step2_cfg)

# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("Forecast settings")
    n_steps = st.slider("Forecast steps (days ahead)", 2, 30, 7, 1)

    method = st.selectbox(
        "Multi-step method",
        ["Autoregressive (recommended)", "Repeat last window (fallback)"],
        index=0
    )

    st.divider()
    st.header("Scaling")
    apply_scaling = st.checkbox("Apply Step3 scaling (x_scaler.json) if available", value=True)

    st.divider()
    st.header("Feature source")
    feat_parquet = os.path.join(OUT2, "features_clean_step2.parquet")
    feat_csv = os.path.join(OUT2, "features_clean_step2.csv")
    auto_use_saved = os.path.exists(feat_parquet) or os.path.exists(feat_csv)
    use_saved_feat = st.checkbox("Use engineered features saved by Step2", value=auto_use_saved)

# -----------------------------
# Load engineered features (recommended)
# -----------------------------
df_feat_source = None
if use_saved_feat and (os.path.exists(feat_parquet) or os.path.exists(feat_csv)):
    df_feat = pd.read_parquet(feat_parquet) if os.path.exists(feat_parquet) else pd.read_csv(feat_csv)
    df_feat_source = feat_parquet if os.path.exists(feat_parquet) else feat_csv
else:
    df_raw, step1_path = load_step1_clean()
    if df_raw is None:
        st.error("Step1 cleaned data not found in outputs/step1/.")
        st.stop()
    df_feat = df_raw.copy()
    df_feat_source = step1_path
    st.warning("Using Step1 only (no engineered features). This may be incomplete vs Step2 features.")

df_feat = normalize_df(df_feat)
df_feat = to_numeric_safe(df_feat, final_feature_cols + target_cols)

st.subheader("Feature source")
st.write(df_feat_source)
st.dataframe(df_feat.tail(15), use_container_width=True)

missing_feat_cols = [c for c in final_feature_cols if c not in df_feat.columns]
if missing_feat_cols:
    st.error(f"Missing feature columns in df_feat (first 30): {missing_feat_cols[:30]}")
    st.info("Fix: ensure Step2 saved features_clean_step2.parquet/csv and enable the checkbox.")
    st.stop()

targets_in_features = all(t in final_feature_cols for t in target_cols)
if method.startswith("Autoregressive") and not targets_in_features:
    st.warning(
        "Autoregressive update needs ALL target columns present in final_feature_cols. "
        "Otherwise forecasts drift badly. Switch to fallback or add targets as features in Step2."
    )

# -----------------------------
# Build last window per station (unscaled feature space)
# -----------------------------
stations, last_dates, X0_list, skipped = [], [], [], []
for stn, g in df_feat.groupby(STATION_COL):
    g = g.sort_values(DATE_COL)
    if len(g) < window:
        skipped.append((stn, f"too few rows ({len(g)} < {window})"))
        continue
    w = g.tail(window)
    Xw = w[final_feature_cols].to_numpy(dtype=np.float32)

    if np.isnan(Xw).any():
        col_means = np.nanmean(Xw, axis=0)
        inds = np.where(np.isnan(Xw))
        Xw[inds] = np.take(col_means, inds[1])
        Xw = np.nan_to_num(Xw, nan=0.0)

    X0_list.append(Xw)
    stations.append(stn)
    last_dates.append(pd.to_datetime(w[DATE_COL].iloc[-1]))

if not X0_list:
    st.error("No station has enough rows to build the last window.")
    if skipped:
        st.dataframe(pd.DataFrame(skipped, columns=["station", "reason"]).head(30), use_container_width=True)
    st.stop()

if skipped:
    with st.expander("Skipped stations"):
        st.dataframe(pd.DataFrame(skipped, columns=["station", "reason"]), use_container_width=True)

X_window = np.stack(X0_list, axis=0)  # (S, window, F)

# -----------------------------
# Forecast loop
# -----------------------------
all_preds = []
forecast_dates = []

sc_used = None
if apply_scaling:
    _, sc_used = apply_saved_scaler_if_exists(X_window[:1], scaler_path)

for step in range(1, n_steps + 1):
    # scale just for prediction (keep X_window in original feature units)
    X_in = X_window
    if apply_scaling and sc_used is not None:
        X_in, _ = apply_saved_scaler_if_exists(X_window, scaler_path)

    with st.spinner(f"Predicting step {step}/{n_steps}..."):
        y_pred = model.predict(X_in, verbose=0)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    all_preds.append(y_pred)
    forecast_dates.append([d + pd.Timedelta(days=step) for d in last_dates])

    if method.startswith("Repeat"):
        continue

    if method.startswith("Autoregressive") and targets_in_features:
        # update last row, shift window
        for i in range(X_window.shape[0]):
            last_vec = X_window[i, -1, :].copy()
            next_vec = update_feature_vector_autoregressive(
                last_feat_vec=last_vec,
                feature_cols=final_feature_cols,
                target_cols=target_cols,
                y_pred=y_pred[i, :],
                max_lag=60
            )
            X_window[i, :-1, :] = X_window[i, 1:, :]
            X_window[i, -1, :] = next_vec

# -----------------------------
# Output table
# -----------------------------
pred_stack = np.stack(all_preds, axis=1)  # (S, steps, out_dim)
out_dim = pred_stack.shape[2]
out_targets = target_cols[:out_dim] if len(target_cols) >= out_dim else [f"y{i}" for i in range(out_dim)]

rows = []
for si, stn in enumerate(stations):
    for step_i in range(n_steps):
        row = {
            "station": stn,
            "base_date": last_dates[si],
            "forecast_date": forecast_dates[step_i][si],
            "step_ahead_days": step_i + 1,
        }
        for tj, tname in enumerate(out_targets):
            row[tname] = float(pred_stack[si, step_i, tj])
        rows.append(row)

out_df = pd.DataFrame(rows)
st.success("Multi-step forecast finished ✅")
st.dataframe(out_df.head(50), use_container_width=True)

# -----------------------------
# Plot
# -----------------------------
st.subheader("Plots")
selected_station = st.selectbox("Select station to plot", options=stations, index=0)
station_df = out_df[out_df["station"] == selected_station].sort_values("step_ahead_days")
for t in out_targets:
    fig = plt.figure()
    plt.plot(station_df["step_ahead_days"], station_df[t], marker="o")
    plt.title(f"{selected_station} — Forecast for {t}")
    plt.xlabel("Days ahead")
    plt.ylabel("Predicted value")
    st.pyplot(fig)

# -----------------------------
# Save
# -----------------------------
st.subheader("Save outputs")
if st.button("Save forecast to outputs/step8/"):
    out_csv = os.path.join(OUT8, "step8_multistep_forecast.csv")
    out_json = os.path.join(OUT8, "step8_summary.json")

    out_df.to_csv(out_csv, index=False)
    summary = {
        "inputs": {
            "step2_config": cfg_path,
            "model": model_path,
            "x_scaler": scaler_path if (apply_scaling and os.path.exists(scaler_path)) else None,
            "feature_source": df_feat_source,
        },
        "settings": {
            "n_steps": int(n_steps),
            "method": method,
            "window": int(window),
            "targets_in_features": bool(targets_in_features),
        },
        "outputs": {"csv": out_csv, "summary": out_json}
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    st.success("Saved ✅")
    st.code("\n".join([out_csv, out_json]))

st.divider()
csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button("Download forecasts (CSV)", data=csv_bytes, file_name="step8_multistep_forecast.csv", mime="text/csv")

with st.expander("Notes / limitations", expanded=False):
    st.write("""
- Step8 uses a 1-step model recursively.
- Only target and lag features can be updated safely without future meteorological drivers.
- If you want high-quality multi-step forecasts, best practice is:
  (1) direct multi-horizon model, OR
  (2) provide future exogenous predictors (rain/temp/ET forecasts).
""")