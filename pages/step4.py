import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Step 4 — Inference & forecasting", layout="wide")
st.title("Step 4 — Inference & forecasting (DISK ONLY from outputs/)")

STATION_COL = "station"
DATE_COL = "date"

OUT1 = os.path.join("outputs", "step1")
OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
OUT4 = os.path.join("outputs", "step4")
os.makedirs(OUT4, exist_ok=True)

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


def _parse_hu_datetime_series(s: pd.Series) -> pd.Series:
    """
    Handles:
      - '2019. 5. 8. 2:00'
      - '2019. 05. 08. 02:00'
      - '2019-05-08 02:00'
    """
    s0 = s.astype(str).str.strip()
    dt = pd.to_datetime(s0, errors="coerce", infer_datetime_format=True)

    if dt.isna().mean() > 0.05:
        for fmt in ("%Y. %m. %d. %H:%M", "%Y.%m.%d. %H:%M", "%Y. %m. %d. %H:%M:%S", "%Y.%m.%d. %H:%M:%S"):
            dt2 = pd.to_datetime(s0, format=fmt, errors="coerce")
            if dt2.notna().sum() > dt.notna().sum():
                dt = dt2

        if dt.isna().mean() > 0.05:
            dt3 = pd.to_datetime(s0.str.replace(r"\s+", " ", regex=True), errors="coerce")
            if dt3.notna().sum() > dt.notna().sum():
                dt = dt3

    return dt


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\t", "") for c in df.columns]

    if STATION_COL not in df.columns or DATE_COL not in df.columns:
        raise ValueError(f"Missing required columns: {STATION_COL}, {DATE_COL}")

    df[STATION_COL] = df[STATION_COL].astype(str).str.strip().str.lower()
    df[DATE_COL] = _parse_hu_datetime_series(df[DATE_COL])

    before = len(df)
    df = df.dropna(subset=[DATE_COL]).sort_values([STATION_COL, DATE_COL]).reset_index(drop=True)
    after = len(df)
    if after < before:
        st.info(f"Dropped {before - after} rows with invalid dates during normalization.")

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
        if c not in out.columns:
            continue
        for L in lags:
            out[f"{c}_lag{L}"] = g[c].shift(L)

    for c in roll_sum_cols:
        if c not in out.columns:
            continue
        for w in roll_sum_windows:
            out[f"{c}_sum{w}"] = g[c].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)

    for c in roll_mean_cols:
        if c not in out.columns:
            continue
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
    out = out.sort_values([STATION_COL, DATE_COL])
    return out


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if out[c].dtype == object:
            s = (
                out[c]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
                .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            )
            out[c] = pd.to_numeric(s, errors="coerce")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def apply_saved_scaler_if_exists(X_3d: np.ndarray, scaler_path: str) -> tuple[np.ndarray, dict | None]:
    """
    If Step3 saved x_scaler.json, apply it here so inference uses the same scaling as training.
    Returns scaled X and scaler dict (or None).
    """
    if not os.path.exists(scaler_path):
        return X_3d, None

    try:
        with open(scaler_path, "r", encoding="utf-8") as f:
            sc = json.load(f)

        if "mean" not in sc or "std" not in sc:
            st.warning("x_scaler.json exists but missing 'mean'/'std'. Skipping scaling.")
            return X_3d, None

        mu = np.asarray(sc["mean"], dtype=np.float32)
        sd = np.asarray(sc["std"], dtype=np.float32)
        sd = np.where(sd == 0, 1.0, sd).astype(np.float32)

        if mu.shape[0] != X_3d.shape[-1] or sd.shape[0] != X_3d.shape[-1]:
            st.warning(
                f"Scaler feature dim mismatch. scaler={mu.shape[0]}, X={X_3d.shape[-1]}. "
                "Skipping scaling to avoid incorrect inference."
            )
            return X_3d, None

        Xs = (X_3d - mu) / sd
        return Xs.astype(np.float32), sc

    except Exception as e:
        st.warning(f"Failed to load/apply x_scaler.json: {e}. Proceeding without scaling.")
        return X_3d, None


def check_feature_compatibility(final_feature_cols: list[str], df_feat: pd.DataFrame):
    missing = [c for c in final_feature_cols if c not in df_feat.columns]
    if missing:
        st.error("Feature mismatch: some Step2 features are missing in df_feat.")
        st.write("Missing features (first 50):")
        st.code(", ".join(missing[:50]) + (" ..." if len(missing) > 50 else ""))
        st.write("Fix: enable 'Use saved engineered features' OR re-run Step2 with saving df_feat.")
        st.stop()


# -----------------------------
# Paths (DISK ONLY)
# -----------------------------
step2_config_path = os.path.join(OUT2, "config.json")
model_path = os.path.join(OUT3, "model.keras")
scaler_path = os.path.join(OUT3, "x_scaler.json")

require(step2_config_path, "Step 2 config.json (outputs/step2/config.json)")
require(model_path, "Step 3 model.keras (outputs/step3/model.keras)")

# load step2 config
with open(step2_config_path, "r", encoding="utf-8") as f:
    step2_cfg = json.load(f)

# load step1 input data
df_raw, step1_path = load_step1_clean()
if df_raw is None:
    st.error("Could not find Step 1 cleaned data in outputs/step1/")
    st.code(os.path.join(OUT1, "combined_stations_clean_step1.csv"))
    st.code(os.path.join(OUT1, "combined_stations_clean_step1.xlsx"))
    st.stop()

df = normalize_df(df_raw)

st.success("Loaded inputs from outputs ✅")
st.write("Step1 source:", step1_path)
st.write("Step2 config:", step2_config_path)
st.write("Model:", model_path)
st.write("Scaler (optional):", scaler_path)

with st.expander("Step 2 config preview"):
    st.json(step2_cfg)

st.subheader("Input data preview (tail)")
st.dataframe(df.tail(20), use_container_width=True)

# -----------------------------
# Step2 settings
# -----------------------------
window = int(step2_cfg.get("window", 7))
horizon = int(step2_cfg.get("horizon", 1))
nan_strategy = step2_cfg.get("nan_strategy", "Time-based interpolation (numeric) within station")

final_feature_cols = step2_cfg.get("final_feature_cols", [])
target_cols = step2_cfg.get("target_cols", [])

if not final_feature_cols:
    st.error("Step2 config.json is missing: final_feature_cols")
    st.stop()

st.subheader("Step2 feature set info")
st.write(f"Window: {window} | Horizon: {horizon} | Features: {len(final_feature_cols)} | Targets: {len(target_cols)}")

# pF check
pf_like = [c for c in final_feature_cols if "pf" in str(c).lower()]
st.write(f"Detected pF-like features in inputs: {len(pf_like)}")
if pf_like:
    st.code(", ".join(pf_like[:50]) + (" ..." if len(pf_like) > 50 else ""))
else:
    st.warning(
        "No pF-like features detected in Step2 final_feature_cols. "
        "If you want pF as inputs, ensure Step2 included them in base_feature_cols."
    )

# Engineered feature regeneration params (if Step2 saved them)
lags = step2_cfg.get("lags", [])
roll_sum_windows = step2_cfg.get("roll_sum_windows", [])
roll_mean_windows = step2_cfg.get("roll_mean_windows", [])

lag_cols = step2_cfg.get("lag_cols", [])
roll_sum_cols = step2_cfg.get("roll_sum_cols", [])
roll_mean_cols = step2_cfg.get("roll_mean_cols", [])

# Prefer using saved df_feat (best match)
feat_parquet = os.path.join(OUT2, "features_clean_step2.parquet")
feat_csv = os.path.join(OUT2, "features_clean_step2.csv")

auto_use_saved = os.path.exists(feat_parquet) or os.path.exists(feat_csv)
use_saved_feat = st.checkbox(
    "Use saved engineered features from outputs/step2/ (recommended if exists)",
    value=auto_use_saved,
)

if use_saved_feat:
    if os.path.exists(feat_parquet):
        df_feat = pd.read_parquet(feat_parquet)
        df_feat_source = feat_parquet
    elif os.path.exists(feat_csv):
        df_feat = pd.read_csv(feat_csv)
        df_feat_source = feat_csv
    else:
        st.error("Checkbox enabled, but no saved features file found in outputs/step2/.")
        st.stop()

    df_feat = normalize_df(df_feat)
    st.success("Loaded engineered df_feat from Step2 ✅")
    st.write("df_feat source:", df_feat_source)
else:
    st.warning(
        "Rebuilding engineered features on the fly. This is safe ONLY if Step2 config includes "
        "lag/rolling parameters AND the raw Step1 file is identical.\n\n"
        "Best practice: enable saved df_feat from Step2."
    )

    # If no params are available, warn strongly
    if (step2_cfg.get("use_lags", False) and (not lag_cols or not lags)) or (step2_cfg.get("use_roll", False) and (not roll_sum_cols and not roll_mean_cols)):
        st.warning(
            "Step2 config suggests engineered features were used, but lag/rolling parameters are missing. "
            "Recomputed df_feat may not match training features."
        )

    with st.spinner("Building engineered features..."):
        df_feat = df.copy()
        df_feat = add_lags_and_rollings(
            df=df_feat,
            group_col=STATION_COL,
            lag_cols=[c for c in lag_cols if c],
            lags=list(map(int, lags)) if lags else [],
            roll_sum_cols=[c for c in roll_sum_cols if c],
            roll_sum_windows=list(map(int, roll_sum_windows)) if roll_sum_windows else [],
            roll_mean_cols=[c for c in roll_mean_cols if c],
            roll_mean_windows=list(map(int, roll_mean_windows)) if roll_mean_windows else [],
        )
        df_feat = apply_nan_strategy(df_feat, nan_strategy)
        df_feat = normalize_df(df_feat)

# Ensure df_feat contains the exact features expected by the model
check_feature_compatibility(final_feature_cols, df_feat)

# Ensure numeric conversion (handles comma decimals, including pF)
df_feat = to_numeric_safe(df_feat, final_feature_cols)

# -----------------------------
# Load model
# -----------------------------
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error("Failed to load model from outputs/step3/model.keras")
    st.exception(e)
    st.stop()

# -----------------------------
# Build latest window per station
# -----------------------------
st.subheader("Build latest window per station for prediction")

X_list, stations, last_dates, skipped = [], [], [], []

for station, g in df_feat.groupby(STATION_COL):
    g = g.sort_values(DATE_COL)
    if len(g) < window:
        skipped.append((station, f"too few rows ({len(g)} < {window})"))
        continue

    last_window = g.tail(window)
    Xw = last_window[final_feature_cols].to_numpy(dtype=np.float32)

    # Fill NaNs with col-means in the window (same logic as Step2)
    if np.isnan(Xw).any():
        col_means = np.nanmean(Xw, axis=0)
        inds = np.where(np.isnan(Xw))
        Xw[inds] = np.take(col_means, inds[1])
        Xw = np.nan_to_num(Xw, nan=0.0)

    X_list.append(Xw)
    stations.append(station)
    last_dates.append(pd.to_datetime(last_window[DATE_COL].iloc[-1]))

if not X_list:
    st.error("No station has enough valid rows to build a prediction window.")
    if skipped:
        st.write("Skipped stations (first 20):")
        st.dataframe(pd.DataFrame(skipped, columns=["station", "reason"]).head(20), use_container_width=True)
    st.stop()

X_latest = np.stack(X_list, axis=0).astype(np.float32)
st.write("X_latest shape:", X_latest.shape)

if skipped:
    with st.expander("Skipped stations"):
        st.dataframe(pd.DataFrame(skipped, columns=["station", "reason"]), use_container_width=True)

# -----------------------------
# Apply SAME scaling as Step3 (if saved)
# -----------------------------
X_latest_scaled, scaler_used = apply_saved_scaler_if_exists(X_latest, scaler_path)
if scaler_used is not None:
    st.success("Applied saved input scaling from outputs/step3/x_scaler.json ✅")
else:
    st.info("No saved scaler applied (x_scaler.json not found or incompatible).")

# -----------------------------
# Predict
# -----------------------------
with st.spinner("Running prediction..."):
    y_pred = model.predict(X_latest_scaled, verbose=0)

if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

# Column names for output
target_names = target_cols if target_cols else [f"y{i}" for i in range(y_pred.shape[1])]
if len(target_names) != y_pred.shape[1]:
    target_names = [f"y{i}" for i in range(y_pred.shape[1])]

# -----------------------------
# Output dataframe
# -----------------------------
res_df = pd.DataFrame(y_pred, columns=target_names)
res_df.insert(0, "last_input_date", last_dates)
res_df.insert(0, "station", stations)

st.subheader("Predictions (model output)")
st.dataframe(res_df, use_container_width=True)

# Plot per station
st.subheader("Predictions per station (bar)")
max_plots = st.slider("Max stations to plot", 1, min(50, len(stations)), min(12, len(stations)))
for i, stn in enumerate(stations[:max_plots]):
    fig = plt.figure()
    plt.bar(target_names, res_df.loc[i, target_names].values)
    plt.title(f"Predicted — {stn}")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -----------------------------
# Save outputs/step4
# -----------------------------
st.subheader("Save outputs")

if st.button("Save predictions to outputs/step4/"):
    try:
        out_csv = os.path.join(OUT4, "predictions_step4.csv")
        out_parq = os.path.join(OUT4, "predictions_step4.parquet")
        out_info = os.path.join(OUT4, "step4_run.json")

        res_df.to_csv(out_csv, index=False)
        try:
            res_df.to_parquet(out_parq, index=False)
        except Exception:
            out_parq = None

        info = {
            "step1_source": step1_path,
            "step2_config": step2_config_path,
            "model_path": model_path,
            "scaler_path_used": scaler_path if scaler_used is not None else None,
            "window": window,
            "horizon": horizon,
            "nan_strategy": nan_strategy,
            "n_stations_predicted": int(len(stations)),
            "feature_source": ("outputs/step2/features_clean_step2.*" if use_saved_feat else "recomputed_in_step4"),
            "n_features": int(len(final_feature_cols)),
            "pf_features_detected": pf_like,
            "saved_files": {
                "csv": out_csv,
                "parquet": out_parq,
                "info": out_info,
            },
        }
        with open(out_info, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        st.success("Saved ✅ outputs/step4/")
        st.code(OUT4)

    except Exception as e:
        st.error("Save failed.")
        st.exception(e)

# Download button
st.divider()
csv_bytes = res_df.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions (CSV)", data=csv_bytes, file_name="predictions_step4.csv", mime="text/csv")