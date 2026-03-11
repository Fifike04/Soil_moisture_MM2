import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Step 4 — Inference & forecasting", layout="wide")
st.title("Step 4 — Inference & forecasting")

STATION_COL = "station"
DATE_COL = "date"

OUT1 = os.path.join("outputs", "step1")
OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
OUT4 = os.path.join("outputs", "step4")
os.makedirs(OUT4, exist_ok=True)

# --------------------------------------------------
# Paths
# --------------------------------------------------
STEP1_CSV = os.path.join(OUT1, "combined_stations_clean_step1.csv")
STEP1_XLSX = os.path.join(OUT1, "combined_stations_clean_step1.xlsx")

STEP2_CFG = os.path.join(OUT2, "config.json")
STEP2_FEAT_PARQUET = os.path.join(OUT2, "features_clean_step2.parquet")
STEP2_FEAT_CSV = os.path.join(OUT2, "features_clean_step2.csv")

MODEL_PATH = os.path.join(OUT3, "model.keras")
X_SCALER_PATH = os.path.join(OUT3, "x_scaler.json")

OUT_CSV = os.path.join(OUT4, "predictions_step4.csv")
OUT_PARQUET = os.path.join(OUT4, "predictions_step4.parquet")
OUT_JSON = os.path.join(OUT4, "step4_run.json")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def require_file(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()


def load_step1_output():
    if os.path.exists(STEP1_CSV):
        df = pd.read_csv(STEP1_CSV)
        src = STEP1_CSV
    elif os.path.exists(STEP1_XLSX):
        df = pd.read_excel(STEP1_XLSX)
        src = STEP1_XLSX
    else:
        raise FileNotFoundError("Step1 output not found.")
    return df, src


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\t", "") for c in df.columns]

    if STATION_COL not in df.columns or DATE_COL not in df.columns:
        raise ValueError(f"Missing required columns: {STATION_COL}, {DATE_COL}")

    df[STATION_COL] = df[STATION_COL].astype(str).str.strip().str.lower()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[DATE_COL]).sort_values([STATION_COL, DATE_COL]).reset_index(drop=True)
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
        return (
            df_feat.groupby(STATION_COL, group_keys=False)
            .apply(lambda g: g.ffill())
            .reset_index(drop=True)
        )

    def interp_group(g):
        g = g.copy().sort_values(DATE_COL).set_index(DATE_COL)
        num_cols = g.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
        g[num_cols] = g[num_cols].interpolate(method="time", limit_direction="both")
        return g.reset_index()

    out = df_feat.groupby(STATION_COL, group_keys=False).apply(interp_group)
    return out.sort_values([STATION_COL, DATE_COL]).reset_index(drop=True)


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue

        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            s = (
                out[c]
                .astype(str)
                .str.strip()
                .replace({"": np.nan, "nan": np.nan, "None": np.nan, "none": np.nan, "-": np.nan})
                .str.replace(",", ".", regex=False)
            )
            out[c] = pd.to_numeric(s, errors="coerce")
    return out


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
# Load config + model
# --------------------------------------------------
st.subheader("1) Load required outputs")

require_file(STEP2_CFG, "Step2 config.json")
require_file(MODEL_PATH, "Step3 model.keras")

with open(STEP2_CFG, "r", encoding="utf-8") as f:
    step2_cfg = json.load(f)

try:
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

window = int(step2_cfg.get("window", 7))
target_cols = step2_cfg.get("target_cols", [])
final_feature_cols = step2_cfg.get("final_feature_cols", [])
nan_strategy = step2_cfg.get("nan_strategy", "Time-based interpolation (numeric) within station")

lag_cols = step2_cfg.get("lag_cols", [])
lags = step2_cfg.get("lags", [])
roll_sum_cols = step2_cfg.get("roll_sum_cols", [])
roll_sum_windows = step2_cfg.get("roll_sum_windows", [])
roll_mean_cols = step2_cfg.get("roll_mean_cols", [])
roll_mean_windows = step2_cfg.get("roll_mean_windows", [])

st.success("Step2 config + Step3 model loaded ✅")
with st.expander("Step2 config"):
    st.json(step2_cfg)

if not final_feature_cols:
    st.error("Step2 config does not contain final_feature_cols.")
    st.stop()

# --------------------------------------------------
# Load scaler
# --------------------------------------------------
st.subheader("2) Load Step3 input scaler")

mu, sd = load_x_scaler(X_SCALER_PATH)
if mu is not None and sd is not None:
    st.success("Step3 x_scaler.json loaded ✅")
    st.write(f"Scaler feature dimension: {len(mu)}")
else:
    st.warning("No valid x_scaler.json found. Inference will run without scaling.")
    mu, sd = None, None

# --------------------------------------------------
# Feature source
# --------------------------------------------------
st.subheader("3) Load feature source")

use_saved_feat_default = os.path.exists(STEP2_FEAT_PARQUET) or os.path.exists(STEP2_FEAT_CSV)
use_saved_feat = st.checkbox("Use Step2 saved engineered features (recommended)", value=use_saved_feat_default)

df_feat_source = None

try:
    if use_saved_feat and os.path.exists(STEP2_FEAT_PARQUET):
        df_feat = pd.read_parquet(STEP2_FEAT_PARQUET)
        df_feat_source = STEP2_FEAT_PARQUET
    elif use_saved_feat and os.path.exists(STEP2_FEAT_CSV):
        df_feat = pd.read_csv(STEP2_FEAT_CSV)
        df_feat_source = STEP2_FEAT_CSV
    else:
        df_raw, step1_source = load_step1_output()
        df = normalize_df(df_raw)

        if not ((lag_cols and lags) or (roll_sum_cols and roll_sum_windows) or (roll_mean_cols and roll_mean_windows)):
            st.warning(
                "No lag/rolling settings found in Step2 config. "
                "Feature regeneration may be incomplete. Prefer using saved Step2 engineered features."
            )
            df_feat = df.copy()
            df_feat_source = step1_source
        else:
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
            df_feat_source = f"recomputed from {step1_source}"

    df_feat = normalize_df(df_feat)

except Exception as e:
    st.error(f"Failed to load/build engineered features: {e}")
    st.stop()

st.success("Feature source ready ✅")
st.write("Feature source:", df_feat_source)
st.dataframe(df_feat.tail(20), use_container_width=True)

# --------------------------------------------------
# Validate features
# --------------------------------------------------
st.subheader("4) Validate feature columns")

missing_feat_cols = [c for c in final_feature_cols if c not in df_feat.columns]
if missing_feat_cols:
    st.error(f"Missing feature columns (first 30): {missing_feat_cols[:30]}")
    st.info("Use the Step2 saved engineered dataframe for the best match.")
    st.stop()

df_feat = to_numeric_safe(df_feat, final_feature_cols + target_cols)

if mu is not None and len(mu) != len(final_feature_cols):
    st.warning(
        f"Scaler dimension ({len(mu)}) does not match final_feature_cols ({len(final_feature_cols)}). "
        "Scaling will be skipped to avoid incorrect inference."
    )
    mu, sd = None, None

# --------------------------------------------------
# Build latest windows
# --------------------------------------------------
st.subheader("5) Build latest input window per station")

stations = []
last_dates = []
X_list = []
skipped = []

for stn, g in df_feat.groupby(STATION_COL):
    g = g.sort_values(DATE_COL)

    if len(g) < window:
        skipped.append((stn, f"too few rows ({len(g)} < {window})"))
        continue

    last_window = g.tail(window)
    Xw = last_window[final_feature_cols].to_numpy(dtype=np.float32)

    if np.isnan(Xw).any():
        col_means = np.nanmean(Xw, axis=0)
        inds = np.where(np.isnan(Xw))
        Xw[inds] = np.take(col_means, inds[1])
        Xw = np.nan_to_num(Xw, nan=0.0)

    X_list.append(Xw)
    stations.append(stn)
    last_dates.append(pd.to_datetime(last_window[DATE_COL].iloc[-1]))

if not X_list:
    st.error("No station has enough rows to build a prediction window.")
    if skipped:
        st.dataframe(pd.DataFrame(skipped, columns=["station", "reason"]), use_container_width=True)
    st.stop()

X_latest = np.stack(X_list, axis=0)

st.write("X_latest shape:", X_latest.shape)

if skipped:
    with st.expander("Skipped stations"):
        st.dataframe(pd.DataFrame(skipped, columns=["station", "reason"]), use_container_width=True)

# --------------------------------------------------
# Apply scaling
# --------------------------------------------------
st.subheader("6) Apply Step3 scaling")

if mu is not None and sd is not None:
    X_latest_scaled = apply_x_scaler(X_latest, mu, sd)
    st.success("Input windows scaled with Step3 x_scaler.json ✅")
else:
    X_latest_scaled = X_latest
    st.info("No scaling applied.")

# --------------------------------------------------
# Predict
# --------------------------------------------------
st.subheader("7) Predict")

if st.button("Run inference + save to outputs/step4"):
    try:
        with st.spinner("Predicting..."):
            y_pred = model.predict(X_latest_scaled, verbose=0)

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        out_dim = y_pred.shape[1]
        out_targets = target_cols[:out_dim] if len(target_cols) >= out_dim else [f"y{i}" for i in range(out_dim)]

        res_df = pd.DataFrame(y_pred, columns=out_targets)
        res_df.insert(0, "last_input_date", last_dates)
        res_df.insert(0, "station", stations)

        st.success("Prediction finished ✅")
        st.dataframe(res_df, use_container_width=True)

        # plots
        st.subheader("Predictions per station")
        max_plots = st.slider("Max stations to plot", 1, min(50, len(stations)), min(12, len(stations)))
        for i, stn in enumerate(stations[:max_plots]):
            fig = plt.figure()
            plt.bar(out_targets, res_df.loc[i, out_targets].values)
            plt.title(f"Predicted values — {stn}")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # save outputs
        res_df.to_csv(OUT_CSV, index=False)

        saved_parquet = None
        try:
            res_df.to_parquet(OUT_PARQUET, index=False)
            saved_parquet = OUT_PARQUET
        except Exception:
            saved_parquet = None

        run_info = {
            "inputs": {
                "step2_config": STEP2_CFG,
                "step3_model": MODEL_PATH,
                "step3_x_scaler": X_SCALER_PATH if mu is not None else None,
                "feature_source": df_feat_source,
            },
            "settings": {
                "window": int(window),
                "target_cols": target_cols,
                "n_features": int(len(final_feature_cols)),
                "n_stations_predicted": int(len(stations)),
            },
            "outputs": {
                "csv": OUT_CSV,
                "parquet": saved_parquet,
                "json": OUT_JSON,
            }
        }

        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(run_info), f, ensure_ascii=False, indent=2)

        st.success("Step 4 saved successfully ✅")
        if saved_parquet:
            st.code("\n".join([OUT_CSV, saved_parquet, OUT_JSON]))
        else:
            st.code("\n".join([OUT_CSV, OUT_JSON]))

    except Exception as e:
        st.error(f"Inference failed: {e}")