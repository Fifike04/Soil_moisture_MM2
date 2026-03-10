import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Step 2 — Feature engineering & LSTM sequences", layout="wide")
st.title("Step 2 — Feature engineering & LSTM sequences")

STATION_COL = "station"
DATE_COL = "date"

SM_COLS = ["_SM10", "_SM20", "_SM30", "_SM45", "_SM60", "_SM75"]
PF_COLS = ["pF2.5_10", "pF2.5_20", "pF2.5_30", "pF2.5_45", "pF2.5_60", "pF2.5_70"]

# --------------------------------------------------
# Paths
# --------------------------------------------------
STEP1_DIR = os.path.join("outputs", "step1")
STEP2_DIR = os.path.join("outputs", "step2")
os.makedirs(STEP2_DIR, exist_ok=True)

STEP1_CSV = os.path.join(STEP1_DIR, "combined_stations_clean_step1.csv")
STEP1_XLSX = os.path.join(STEP1_DIR, "combined_stations_clean_step1.xlsx")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def find_step1_output() -> str:
    if os.path.exists(STEP1_CSV):
        return STEP1_CSV
    if os.path.exists(STEP1_XLSX):
        return STEP1_XLSX
    return ""

def load_step1_output(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported Step1 file type.")

    df.columns = [str(c).strip().replace("\t", "") for c in df.columns]

    if STATION_COL not in df.columns or DATE_COL not in df.columns:
        raise ValueError(f"Missing required columns: {STATION_COL}, {DATE_COL}")

    df[STATION_COL] = df[STATION_COL].astype(str).str.strip().str.lower()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
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
) -> pd.DataFrame:
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

def apply_nan_strategy(df_feat: pd.DataFrame, strategy: str) -> pd.DataFrame:
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

def build_sequences(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    feature_cols: list[str],
    target_cols: list[str],
    window: int,
    horizon: int,
    allow_nan_ratio: float = 0.0
):
    """
    Returns:
        X: (samples, window, features)
        y: (samples, targets)
        meta_df: station, target_date
        diag_df: per-station diagnostics
    """
    X_list, y_list, meta_rows = [], [], []
    diag_rows = []

    work = df.copy()

    # convert selected columns to numeric safely
    for c in feature_cols + target_cols:
        if c in work.columns:
            if work[c].dtype == object:
                s = work[c].astype(str).str.replace(",", ".", regex=False)
                work[c] = pd.to_numeric(s, errors="coerce")
            else:
                work[c] = pd.to_numeric(work[c], errors="coerce")

    for station, g in work.groupby(group_col):
        g = g.sort_values(date_col).reset_index(drop=True)

        feat = g[feature_cols].to_numpy(dtype=np.float32)
        targ = g[target_cols].to_numpy(dtype=np.float32)
        dates = g[date_col].to_numpy()

        n = len(g)
        last_start = n - window - horizon + 1

        if last_start <= 0:
            diag_rows.append({
                "station": station,
                "rows": n,
                "possible_windows": 0,
                "kept_windows": 0,
                "dropped_target_nan": 0,
                "dropped_nan_ratio": 0,
                "reason": "too few rows for window+horizon"
            })
            continue

        possible = 0
        kept = 0
        dropped_target_nan = 0
        dropped_nan_ratio = 0

        for start in range(last_start):
            possible += 1
            end = start + window
            y_idx = end + horizon - 1

            x_win = feat[start:end, :].copy()
            y_val = targ[y_idx, :].copy()

            if np.isnan(y_val).any():
                dropped_target_nan += 1
                continue

            nan_count = np.isnan(x_win).sum()
            nan_ratio = nan_count / x_win.size if x_win.size > 0 else 1.0

            if nan_ratio > allow_nan_ratio:
                dropped_nan_ratio += 1
                continue

            if nan_count > 0:
                col_means = np.nanmean(x_win, axis=0)
                inds = np.where(np.isnan(x_win))
                x_win[inds] = np.take(col_means, inds[1])
                x_win = np.nan_to_num(x_win, nan=0.0)

            X_list.append(x_win)
            y_list.append(y_val)
            meta_rows.append({
                "station": station,
                "target_date": pd.to_datetime(dates[y_idx])
            })
            kept += 1

        diag_rows.append({
            "station": station,
            "rows": n,
            "possible_windows": possible,
            "kept_windows": kept,
            "dropped_target_nan": dropped_target_nan,
            "dropped_nan_ratio": dropped_nan_ratio,
            "reason": "ok" if kept > 0 else "all windows dropped"
        })

    if not X_list:
        raise ValueError(
            "No sequences were generated. Try smaller window, weaker feature engineering, "
            "or a different NaN strategy."
        )

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    meta_df = pd.DataFrame(meta_rows)
    diag_df = pd.DataFrame(diag_rows).sort_values(["kept_windows", "possible_windows"]).reset_index(drop=True)
    return X, y, meta_df, diag_df

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

def save_step2_outputs(
    X: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    config: dict,
    df_feat: pd.DataFrame | None = None,
    features_format: str = "Parquet (recommended)",
):
    np.save(os.path.join(STEP2_DIR, "X.npy"), X)
    np.save(os.path.join(STEP2_DIR, "y.npy"), y)

    meta_df.to_csv(os.path.join(STEP2_DIR, "meta.csv"), index=False)
    diag_df.to_csv(os.path.join(STEP2_DIR, "diagnostics.csv"), index=False)

    with open(os.path.join(STEP2_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(make_json_safe(config), f, ensure_ascii=False, indent=2)

    if df_feat is not None:
        if features_format.startswith("CSV"):
            df_feat.to_csv(os.path.join(STEP2_DIR, "features_clean_step2.csv"), index=False)
        else:
            try:
                df_feat.to_parquet(os.path.join(STEP2_DIR, "features_clean_step2.parquet"), index=False)
            except Exception:
                df_feat.to_csv(os.path.join(STEP2_DIR, "features_clean_step2.csv"), index=False)

    with open(os.path.join(STEP2_DIR, "_READY.txt"), "w", encoding="utf-8") as f:
        f.write("ok")

# --------------------------------------------------
# Load Step1 output
# --------------------------------------------------
st.subheader("1) Load Step1 output")

step1_path = find_step1_output()
if not step1_path:
    st.error("Step1 output not found.")
    st.code(STEP1_CSV)
    st.code(STEP1_XLSX)
    st.info("Futtasd le először a Step1-et.")
    st.stop()

try:
    df = load_step1_output(step1_path)
except Exception as e:
    st.error(f"Failed to load Step1 output: {e}")
    st.stop()

st.success("Step1 output loaded successfully.")
st.write("Source:", step1_path)
st.write("Shape:", df.shape)
st.dataframe(df.head(20), use_container_width=True)

# --------------------------------------------------
# Targets
# --------------------------------------------------
st.subheader("2) Select targets")

existing_sm = [c for c in SM_COLS if c in df.columns]
existing_pf = [c for c in PF_COLS if c in df.columns]

target_mode = st.radio(
    "Target set",
    ["Soil moisture (_SM*)", "pF2.5 (pF2.5_*)"],
    index=0
)

if target_mode == "Soil moisture (_SM*)":
    if not existing_sm:
        st.error("No _SM* columns found in Step1 output.")
        st.stop()
    target_cols = st.multiselect("Target columns", options=existing_sm, default=existing_sm)
else:
    if not existing_pf:
        st.error("No pF2.5_* columns found in Step1 output.")
        st.stop()
    target_cols = st.multiselect("Target columns", options=existing_pf, default=existing_pf)

if not target_cols:
    st.warning("Select at least one target.")
    st.stop()

# --------------------------------------------------
# Feature selection
# --------------------------------------------------
st.subheader("3) Base feature selection")

exclude = {STATION_COL, DATE_COL} | set(target_cols)

numeric_candidates = []
for c in df.columns:
    if c in exclude:
        continue

    if pd.api.types.is_numeric_dtype(df[c]):
        numeric_candidates.append(c)
    else:
        sample = pd.to_numeric(
            df[c].dropna().astype(str).head(50).str.replace(",", ".", regex=False),
            errors="coerce"
        )
        if len(sample) > 0 and sample.notna().mean() >= 0.6:
            numeric_candidates.append(c)

default_inputs = [c for c in numeric_candidates if any(k in c for k in ["_Napi", "_ET", "_HD", "_Talajhom", "_Viz"])]
if not default_inputs:
    default_inputs = numeric_candidates[:20]

base_feature_cols = st.multiselect(
    "Base feature columns",
    options=sorted(numeric_candidates),
    default=sorted(set(default_inputs)),
)

if not base_feature_cols:
    st.warning("Select at least one input feature.")
    st.stop()

# --------------------------------------------------
# Sequence + feature engineering settings
# --------------------------------------------------
st.subheader("4) Sequence and feature engineering settings")

c1, c2 = st.columns(2)
with c1:
    window = st.number_input("Window length (days)", min_value=3, max_value=365, value=30, step=1)
with c2:
    horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=1, step=1)

nan_strategy = st.selectbox(
    "Missing value strategy",
    ["None (strict)", "Forward fill within station", "Time-based interpolation (numeric) within station"],
    index=0
)
allow_nan_ratio = st.slider(
    "Allowed NaN ratio inside X window",
    min_value=0.0, max_value=0.5, value=0.0, step=0.05
)

use_lags = st.checkbox("Add lag features", value=True)
lags = st.multiselect("Lag days", options=[1, 2, 3, 5, 7, 10, 14, 21, 30], default=[1, 3, 7])

use_roll = st.checkbox("Add rolling features", value=True)
roll_sum_windows = st.multiselect("Rolling SUM windows", options=[3, 5, 7, 10, 14, 21, 30], default=[7, 14])
roll_mean_windows = st.multiselect("Rolling MEAN windows", options=[3, 5, 7, 10, 14, 21, 30], default=[7, 14])

lag_cols = []
roll_sum_cols = []
roll_mean_cols = []

if use_lags:
    st.write("### Lag features")
    lag_cols = st.multiselect(
        "Columns to lag",
        options=sorted(set(base_feature_cols + target_cols)),
        default=sorted(set([c for c in base_feature_cols if "_NapiCsapadek" in c or "_ET" in c] + target_cols))[:12],
    )

if use_roll:
    st.write("### Rolling features")
    roll_sum_cols = st.multiselect(
        "Columns for rolling SUM",
        options=sorted(base_feature_cols),
        default=[c for c in base_feature_cols if "_NapiCsapadek" in c][:1],
    )
    roll_mean_cols = st.multiselect(
        "Columns for rolling MEAN",
        options=sorted(base_feature_cols),
        default=[c for c in base_feature_cols if "_NapiLeghomersekletAtl" in c or "_Talajhom" in c][:3],
    )

save_features_df = st.checkbox("Also save engineered dataframe to outputs/step2/", value=True)
features_format = st.selectbox("Engineered dataframe format", ["Parquet (recommended)", "CSV"], index=0)

with st.expander("Quick tips if you get 0 sequences", expanded=True):
    st.write("- Reduce window (e.g. 30 → 14 or 7)")
    st.write("- Temporarily disable lag/rolling")
    st.write("- Use interpolation or allow a small NaN ratio")
    st.write("- Check diagnostics.csv for which station drops windows")

# --------------------------------------------------
# Run Step2
# --------------------------------------------------
st.subheader("5) Run")

if st.button("Build features + generate sequences + save to outputs/step2"):
    with st.spinner("Building engineered features..."):
        df_feat = add_lags_and_rollings(
            df=df,
            group_col=STATION_COL,
            lag_cols=lag_cols if (use_lags and lags) else [],
            lags=list(map(int, lags)) if (use_lags and lags) else [],
            roll_sum_cols=roll_sum_cols if (use_roll and roll_sum_windows) else [],
            roll_sum_windows=list(map(int, roll_sum_windows)) if (use_roll and roll_sum_windows) else [],
            roll_mean_cols=roll_mean_cols if (use_roll and roll_mean_windows) else [],
            roll_mean_windows=list(map(int, roll_mean_windows)) if (use_roll and roll_mean_windows) else [],
        )

    with st.spinner("Applying NaN strategy..."):
        df_feat = apply_nan_strategy(df_feat, nan_strategy)

    engineered_cols = [c for c in df_feat.columns if c not in df.columns]
    final_feature_cols = base_feature_cols + engineered_cols

    st.success("Feature engineering finished ✅")
    st.write(f"Engineered columns: {len(engineered_cols)}")
    st.dataframe(df_feat.head(20), use_container_width=True)

    with st.spinner("Building sequences..."):
        X, y, meta_df, diag_df = build_sequences(
            df=df_feat,
            group_col=STATION_COL,
            date_col=DATE_COL,
            feature_cols=final_feature_cols,
            target_cols=target_cols,
            window=int(window),
            horizon=int(horizon),
            allow_nan_ratio=float(allow_nan_ratio),
        )

    st.success("Sequences generated ✅")
    st.write(f"X shape: {X.shape}")
    st.write(f"y shape: {y.shape}")

    st.write("### Diagnostics")
    st.dataframe(diag_df, use_container_width=True)

    st.write("### Meta preview")
    st.dataframe(meta_df.head(20), use_container_width=True)

    config = {
        "input_source": step1_path,
        "window": int(window),
        "horizon": int(horizon),
        "nan_strategy": nan_strategy,
        "allow_nan_ratio": float(allow_nan_ratio),
        "target_cols": target_cols,
        "base_feature_cols": base_feature_cols,
        "engineered_feature_cols": engineered_cols,
        "final_feature_cols": final_feature_cols,
        "lag_cols": lag_cols,
        "lags": list(map(int, lags)) if lags else [],
        "roll_sum_cols": roll_sum_cols,
        "roll_sum_windows": list(map(int, roll_sum_windows)) if roll_sum_windows else [],
        "roll_mean_cols": roll_mean_cols,
        "roll_mean_windows": list(map(int, roll_mean_windows)) if roll_mean_windows else [],
        "save_features_df": bool(save_features_df),
        "features_format": features_format,
    }

    # optional session_state too
    st.session_state["step2_X"] = X
    st.session_state["step2_y"] = y
    st.session_state["step2_meta"] = meta_df
    st.session_state["step2_diag"] = diag_df
    st.session_state["step2_config"] = config

    try:
        save_step2_outputs(
            X=X,
            y=y,
            meta_df=meta_df,
            diag_df=diag_df,
            config=config,
            df_feat=(df_feat if save_features_df else None),
            features_format=features_format,
        )
        st.success("Step 2 saved successfully ✅")
        st.code(STEP2_DIR)
    except Exception as e:
        st.error(f"Saving Step2 outputs failed: {e}")

    st.divider()
    st.write("### Optional downloads")
    if st.checkbox("Enable X/y downloads"):
        x_buf = io.BytesIO()
        y_buf = io.BytesIO()
        np.save(x_buf, X)
        np.save(y_buf, y)
        st.download_button("Download X.npy", data=x_buf.getvalue(), file_name="X.npy")
        st.download_button("Download y.npy", data=y_buf.getvalue(), file_name="y.npy")