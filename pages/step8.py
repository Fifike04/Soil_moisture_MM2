import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Step 8 — Multi-step forecasting", layout="wide")

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

def add_lags_and_rollings(
    df: pd.DataFrame,
    group_col: str,
    lag_cols: list[str],
    lags: list[int],
    roll_sum_cols: list[str],
    roll_sum_windows: list[int],
    roll_mean_cols: list[str],
    roll_mean_windows: list[int],
):
    out = df.copy()
    g = out.groupby(group_col, group_keys=False)

    for c in lag_cols:
        if c in out.columns:
            for L in lags:
                out[f"{c}_lag{L}"] = g[c].shift(L)

    for c in roll_sum_cols:
        if c in out.columns:
            for w in roll_sum_windows:
                out[f"{c}_sum{w}"] = g[c].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)

    for c in roll_mean_cols:
        if c in out.columns:
            for w in roll_mean_windows:
                out[f"{c}_mean{w}"] = g[c].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)

    return out

def apply_nan_strategy(df_feat: pd.DataFrame, strategy: str):
    if strategy == "None (strict)":
        return df_feat
    if strategy == "Forward fill within station":
        return df_feat.groupby(STATION_COL, group_keys=False).apply(lambda g: g.ffill())

    def interp_group(g):
        g = g.copy().sort_values(DATE_COL).set_index(DATE_COL)
        num_cols = g.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
        g[num_cols] = g[num_cols].interpolate(method="time", limit_direction="both")
        return g.reset_index()

    out = df_feat.groupby(STATION_COL, group_keys=False).apply(interp_group)
    return out.sort_values([STATION_COL, DATE_COL])

def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if out[c].dtype == object:
            s = out[c].astype(str).str.replace(",", ".", regex=False)
            out[c] = pd.to_numeric(s, errors="coerce")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def update_feature_vector_autoregressive(
    last_feat_vec: np.ndarray,
    feature_cols: list[str],
    target_cols: list[str],
    y_pred: np.ndarray,
    max_lag: int = 30,
):
    """
    Next-step feature vector from last timestep feature vector.

    - If feature is exactly a target col (e.g. _SM10), replace with predicted value.
    - Shift lags for targets: lag1 <- prev base, lag2 <- prev lag1, ...
    - Other features persist (unchanged).
    """
    next_vec = last_feat_vec.copy()
    idx = {c: j for j, c in enumerate(feature_cols)}
    pred_map = {t: float(y_pred[i]) for i, t in enumerate(target_cols) if i < len(y_pred)}

    # Update base target features
    for t, val in pred_map.items():
        if t in idx:
            next_vec[idx[t]] = val

    # Shift target lags
    for t in target_cols:
        base_j = idx.get(t, None)
        if base_j is None:
            continue

        prev_base = float(last_feat_vec[base_j])

        lag1_name = f"{t}_lag1"
        if lag1_name in idx:
            next_vec[idx[lag1_name]] = prev_base

        for k in range(2, max_lag + 1):
            name_k = f"{t}_lag{k}"
            name_km1 = f"{t}_lag{k-1}"
            if name_k in idx and name_km1 in idx:
                next_vec[idx[name_k]] = float(last_feat_vec[idx[name_km1]])

    return next_vec

# -----------------------------
# Load from outputs/
# -----------------------------
cfg_path = os.path.join(OUT2, "config.json")
model_path = os.path.join(OUT3, "model.keras")
require(cfg_path, "Step2 config.json (outputs/step2/config.json)")
require(model_path, "Step3 model.keras (outputs/step3/model.keras)")

with open(cfg_path, "r", encoding="utf-8") as f:
    step2_cfg = json.load(f)

# Feature cols / targets / window
window = int(step2_cfg.get("window", 30))
final_feature_cols = step2_cfg.get("final_feature_cols", [])
target_cols = step2_cfg.get("target_cols", DEFAULT_TARGETS)
nan_strategy = step2_cfg.get("nan_strategy", "Time-based interpolation (numeric) within station")

if not final_feature_cols:
    st.error("Step2 config.json missing 'final_feature_cols'.")
    st.stop()

# Optional: if you decide to store these in Step2 config later
lag_cols = step2_cfg.get("lag_cols", [])
lags = step2_cfg.get("lags", [])
roll_sum_cols = step2_cfg.get("roll_sum_cols", [])
roll_sum_windows = step2_cfg.get("roll_sum_windows", [])
roll_mean_cols = step2_cfg.get("roll_mean_cols", [])
roll_mean_windows = step2_cfg.get("roll_mean_windows", [])

# Load model
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error("Failed to load outputs/step3/model.keras")
    st.exception(e)
    st.stop()

st.subheader("Config summary (from outputs/step2/config.json)")
st.write({
    "window": window,
    "targets": target_cols,
    "n_feature_cols": len(final_feature_cols),
    "nan_strategy": nan_strategy,
    "model_path": model_path
})
with st.expander("Step2 config.json"):
    st.json(step2_cfg)

# -----------------------------
# UI
# -----------------------------
st.title("Step 8 — Multi-step forecasting (autoregressive, DISK ONLY)")

with st.sidebar:
    st.header("Forecast settings")
    n_steps = st.slider("Forecast steps (days ahead)", 2, 30, 7, 1)
    method = st.selectbox(
        "Multi-step method",
        ["Autoregressive (recommended)", "Repeat last window (fallback)"],
        index=0
    )

    st.divider()
    st.header("Feature source")
    # Best option: reuse engineered features if saved by Step2
    feat_parquet = os.path.join(OUT2, "features_clean_step2.parquet")
    feat_csv = os.path.join(OUT2, "features_clean_step2.csv")
    use_saved_feat = os.path.exists(feat_parquet) or os.path.exists(feat_csv)
    use_saved_feat = st.checkbox("Use engineered features saved by Step2 (recommended if exists)", value=use_saved_feat)

# -----------------------------
# Load data / features
# -----------------------------
df_feat_source = None

if use_saved_feat:
    if os.path.exists(feat_parquet):
        df_feat = pd.read_parquet(feat_parquet)
        df_feat_source = feat_parquet
    elif os.path.exists(feat_csv):
        df_feat = pd.read_csv(feat_csv)
        df_feat_source = feat_csv
    else:
        # fallback if user forced checkbox but files not there
        use_saved_feat = False

if not use_saved_feat:
    df_raw, step1_path = load_step1_clean()
    if df_raw is None:
        st.error("Step1 cleaned data not found in outputs/step1/.")
        st.code(os.path.join(OUT1, "combined_stations_clean_step1.csv"))
        st.code(os.path.join(OUT1, "combined_stations_clean_step1.xlsx"))
        st.stop()
    df = normalize_df(df_raw)
    df_feat = df.copy()

    # If Step2 config doesn't store lag/rolling params, we cannot perfectly rebuild them.
    if not (lag_cols and lags) and not (roll_sum_cols and roll_sum_windows) and not (roll_mean_cols and roll_mean_windows):
        st.warning(
            "Step2 config.json does not contain lag/rolling settings (lag_cols/lags/roll_*). "
            "So Step8 will NOT regenerate engineered features. "
            "Recommendation: in Step2, also save df_feat to outputs/step2/features_clean_step2.parquet."
        )
    else:
        with st.spinner("Rebuilding engineered features from Step2 config..."):
            df_feat = add_lags_and_rollings(
                df=df,
                group_col=STATION_COL,
                lag_cols=lag_cols,
                lags=list(map(int, lags)),
                roll_sum_cols=roll_sum_cols,
                roll_sum_windows=list(map(int, roll_sum_windows)),
                roll_mean_cols=roll_mean_cols,
                roll_mean_windows=list(map(int, roll_mean_windows)),
            )
            df_feat = apply_nan_strategy(df_feat, nan_strategy)

df_feat = normalize_df(df_feat)

st.subheader("Feature source")
st.write(df_feat_source if df_feat_source else "recomputed from outputs/step1 (may be incomplete)")
st.subheader("Engineered data preview (tail)")
st.dataframe(df_feat.tail(20), use_container_width=True)

# Ensure required feature columns exist
missing_feat_cols = [c for c in final_feature_cols if c not in df_feat.columns]
if missing_feat_cols:
    st.error(f"Missing feature columns in df_feat (first 30): {missing_feat_cols[:30]}")
    st.info(
        "Fix: save engineered features in Step2 to outputs/step2/features_clean_step2.parquet "
        "and enable 'Use engineered features saved by Step2'."
    )
    st.stop()

# Numeric coercion for feature cols
df_feat = to_numeric_safe(df_feat, final_feature_cols + target_cols)

# Check whether autoregressive is possible (targets must be in feature set)
targets_in_features = all(t in final_feature_cols for t in target_cols)
if method.startswith("Autoregressive") and not targets_in_features:
    st.warning(
        "Autoregressive forecasting requires ALL target columns to be included among final_feature_cols. "
        "Otherwise the window can't be updated meaningfully. Switching to fallback method is recommended."
    )

# -----------------------------
# Build last window per station
# -----------------------------
st.subheader("Multi-step forecast")

stations = []
last_dates = []
X0_list = []
skipped = []

for stn, g in df_feat.groupby(STATION_COL):
    g = g.sort_values(DATE_COL)
    if len(g) < window:
        skipped.append((stn, f"too few rows ({len(g)} < {window})"))
        continue
    w = g.tail(window)
    Xw = w[final_feature_cols].to_numpy(dtype=np.float32)

    if np.isnan(Xw).any():
        # Fill NaNs (similar spirit as Step2)
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

X_unscaled = np.stack(X0_list, axis=0)  # (S, window, F)

# -----------------------------
# Forecast loop (no scaling)
# -----------------------------
all_preds = []  # list of (S, targets) per step
forecast_dates = []

for step in range(1, n_steps + 1):
    with st.spinner(f"Predicting step {step}/{n_steps}..."):
        y_pred = model.predict(X_unscaled, verbose=0)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    all_preds.append(y_pred)
    forecast_dates.append([d + pd.Timedelta(days=step) for d in last_dates])

    if method.startswith("Repeat") or (method.startswith("Autoregressive") and not targets_in_features):
        continue

    # Autoregressive update (in feature space)
    for i in range(X_unscaled.shape[0]):
        last_vec = X_unscaled[i, -1, :].copy()
        next_vec = update_feature_vector_autoregressive(
            last_feat_vec=last_vec,
            feature_cols=final_feature_cols,
            target_cols=target_cols,
            y_pred=y_pred[i, :],
            max_lag=30
        )

        # shift left and append
        X_unscaled[i, :-1, :] = X_unscaled[i, 1:, :]
        X_unscaled[i, -1, :] = next_vec

# -----------------------------
# Build output table
# -----------------------------
pred_stack = np.stack(all_preds, axis=1)  # (S, steps, out_dim)

# column names for outputs
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
# Plot per station (lines)
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
# Save to outputs/step8
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
            "feature_source": df_feat_source if df_feat_source else "recomputed_from_outputs/step1",
        },
        "settings": {
            "n_steps": int(n_steps),
            "method": method,
            "window": int(window),
            "targets_in_features": bool(targets_in_features),
        },
        "outputs": {
            "csv": out_csv,
            "summary": out_json
        }
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    st.success("Saved ✅")
    st.code("\n".join([out_csv, out_json]))

# Download button (optional)
st.divider()
csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button("Download forecasts (CSV)", data=csv_bytes, file_name="step8_multistep_forecast.csv", mime="text/csv")

with st.expander("Notes / limitations", expanded=False):
    st.write("""
- This app performs multi-step forecasting using a 1-step model.
- Autoregressive updating is reliable only if target columns (e.g., _SM10.._SM75) are present among feature columns.
- Rolling features and exogenous predictors for future days are kept as persistence (unchanged) unless you provide future exogenous data.
- Best practice for multi-step is a direct multi-horizon model, or provide future meteorological predictors.
""")