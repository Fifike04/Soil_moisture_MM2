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

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if STATION_COL not in df.columns or DATE_COL not in df.columns:
        raise ValueError(f"Missing required columns: {STATION_COL}, {DATE_COL}")
    df = df.copy()
    df.columns = [str(c).strip().replace("\t", "") for c in df.columns]
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
        for L in lags:
            if c in out.columns:
                out[f"{c}_lag{L}"] = g[c].shift(L)

    for c in roll_sum_cols:
        for w in roll_sum_windows:
            if c in out.columns:
                out[f"{c}_sum{w}"] = g[c].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)

    for c in roll_mean_cols:
        for w in roll_mean_windows:
            if c in out.columns:
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
            s = out[c].astype(str).str.replace(",", ".", regex=False)
            out[c] = pd.to_numeric(s, errors="coerce")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# -----------------------------
# Paths (DISK ONLY)
# -----------------------------
step2_config_path = os.path.join(OUT2, "config.json")
model_path = os.path.join(OUT3, "model.keras")

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

with st.expander("Step 2 config preview"):
    st.json(step2_cfg)

st.subheader("Input data preview (tail)")
st.dataframe(df.tail(20), use_container_width=True)

# -----------------------------
# Get Step2 settings (feature list + window)
# -----------------------------
window = int(step2_cfg.get("window", 30))
horizon = int(step2_cfg.get("horizon", 1))
nan_strategy = step2_cfg.get("nan_strategy", "Time-based interpolation (numeric) within station")

final_feature_cols = step2_cfg.get("final_feature_cols", [])
target_cols = step2_cfg.get("target_cols", [])

if not final_feature_cols:
    st.error("Step2 config.json is missing: final_feature_cols")
    st.stop()

# Optional: if you saved engineered_feature_cols etc.
base_feature_cols = step2_cfg.get("base_feature_cols", [])
engineered_feature_cols = step2_cfg.get("engineered_feature_cols", [])

# If engineered cols were created by lags/rollings, rebuild them using same rules.
# We infer lag/rolling params from config if you stored them (if not, we keep defaults).
lags = step2_cfg.get("lags", [1, 3, 7])
roll_sum_windows = step2_cfg.get("roll_sum_windows", [7, 14])
roll_mean_windows = step2_cfg.get("roll_mean_windows", [7, 14])

lag_cols = step2_cfg.get("lag_cols", [])          # if not saved, empty -> no lag regeneration
roll_sum_cols = step2_cfg.get("roll_sum_cols", [])  # if not saved, empty -> no roll regeneration
roll_mean_cols = step2_cfg.get("roll_mean_cols", []) # if not saved, empty -> no roll regeneration

st.info(
    "Megjegyzés: ha a Step2 config nem tartalmazza a lag/rolling beállításokat (lag_cols, roll_*), "
    "akkor Step4 nem tudja 100%-osan újragenerálni az engineered oszlopokat. "
    "Ilyenkor két opció: (1) mentsd el ezeket Step2-ben configba, vagy (2) Step4 használja a Step2-ben elmentett df_feat fájlt."
)

# Prefer using saved df_feat if exists (best match!)
feat_parquet = os.path.join(OUT2, "features_clean_step2.parquet")
feat_csv = os.path.join(OUT2, "features_clean_step2.csv")

use_saved_feat = os.path.exists(feat_parquet) or os.path.exists(feat_csv)
use_saved_feat = st.checkbox(
    "Use saved engineered features from outputs/step2/ (recommended if exists)",
    value=use_saved_feat
)

if use_saved_feat:
    if os.path.exists(feat_parquet):
        df_feat = pd.read_parquet(feat_parquet)
        df_feat_source = feat_parquet
    else:
        df_feat = pd.read_csv(feat_csv)
        df_feat_source = feat_csv

    df_feat = normalize_df(df_feat)
    st.success("Loaded engineered df_feat from Step2 ✅")
    st.write("df_feat source:", df_feat_source)
else:
    # Rebuild features on the fly (may mismatch if config lacks parameters)
    st.warning("Rebuilding engineered features on the fly (may mismatch Step 2 if params differ).")

    with st.spinner("Building engineered features..."):
        df_feat = add_lags_and_rollings(
            df=df,
            group_col=STATION_COL,
            lag_cols=[c for c in lag_cols if c],
            lags=list(map(int, lags)) if lags else [],
            roll_sum_cols=[c for c in roll_sum_cols if c],
            roll_sum_windows=list(map(int, roll_sum_windows)) if roll_sum_windows else [],
            roll_mean_cols=[c for c in roll_mean_cols if c],
            roll_mean_windows=list(map(int, roll_mean_windows)) if roll_mean_windows else [],
        )
        df_feat = apply_nan_strategy(df_feat, nan_strategy)

# Ensure numeric for feature cols
df_feat = to_numeric_safe(df_feat, final_feature_cols)

# -----------------------------
# Load model (disk)
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

X_list = []
stations = []
last_dates = []
skipped = []

for station, g in df_feat.groupby(STATION_COL):
    g = g.sort_values(DATE_COL)

    if len(g) < window:
        skipped.append((station, f"too few rows ({len(g)} < {window})"))
        continue

    last_window = g.tail(window)

    missing_cols = [c for c in final_feature_cols if c not in last_window.columns]
    if missing_cols:
        skipped.append((station, f"missing cols: {missing_cols[:5]}{'...' if len(missing_cols)>5 else ''}"))
        continue

    Xw = last_window[final_feature_cols].to_numpy(dtype=np.float32)

    # If still NaNs, fill simple (consistent with your Step2 sequence builder behaviour)
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

X_latest = np.stack(X_list, axis=0)
st.write("X_latest shape:", X_latest.shape)

if skipped:
    with st.expander("Skipped stations"):
        st.dataframe(pd.DataFrame(skipped, columns=["station", "reason"]), use_container_width=True)

# -----------------------------
# Predict
# -----------------------------
with st.spinner("Running prediction..."):
    y_pred = model.predict(X_latest, verbose=0)

# Ensure 2D preds
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

# Decide column names
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

# Plot per station (optional)
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
            # parquet optional
            out_parq = None

        info = {
            "step1_source": step1_path,
            "step2_config": step2_config_path,
            "model_path": model_path,
            "window": window,
            "horizon": horizon,
            "nan_strategy": nan_strategy,
            "n_stations_predicted": int(len(stations)),
            "feature_source": ("outputs/step2/features_clean_step2.*" if use_saved_feat else "recomputed_in_step4"),
            "saved_files": {
                "csv": out_csv,
                "parquet": out_parq,
                "info": out_info
            }
        }
        with open(out_info, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        st.success("Mentve ✅ outputs/step4/")
        st.code(OUT4)
    except Exception as e:
        st.error("Save failed.")
        st.exception(e)

# Download button (optional)
st.divider()
csv_bytes = res_df.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions (CSV)", data=csv_bytes, file_name="predictions_step4.csv", mime="text/csv")