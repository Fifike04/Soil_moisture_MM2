import os
import io
import glob
import json
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Step 2 — Feature engineering & LSTM sequences", layout="wide")

STATION_COL = "station"
DATE_COL = "date"

SM_COLS = ["_SM10", "_SM20", "_SM30", "_SM45", "_SM60", "_SM75"]
PF_COLS = ["pF2.5_10", "pF2.5_20", "pF2.5_30", "pF2.5_45", "pF2.5_60", "pF2.5_70"]

# --- Output dir ---
STEP2_OUT_DIR = os.path.join("outputs", "step2")
os.makedirs(STEP2_OUT_DIR, exist_ok=True)

def save_step2_artifacts(X, y, meta_df, diag_df, config: dict, df_feat: pd.DataFrame | None = None):
    # Core artifacts
    np.save(os.path.join(STEP2_OUT_DIR, "X.npy"), X)
    np.save(os.path.join(STEP2_OUT_DIR, "y.npy"), y)

    meta_df.to_csv(os.path.join(STEP2_OUT_DIR, "meta.csv"), index=False)
    diag_df.to_csv(os.path.join(STEP2_OUT_DIR, "diagnostics.csv"), index=False)

    # ensure JSON-serializable
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_json_safe(x) for x in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    config_safe = make_json_safe(config)

    with open(os.path.join(STEP2_OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_safe, f, ensure_ascii=False, indent=2)

    # Optional: save engineered dataframe
    if df_feat is not None:
        # Parquet is smaller & faster (recommended)
        try:
            df_feat.to_parquet(os.path.join(STEP2_OUT_DIR, "features_clean_step2.parquet"), index=False)
        except Exception:
            # fallback to CSV if parquet engine missing
            df_feat.to_csv(os.path.join(STEP2_OUT_DIR, "features_clean_step2.csv"), index=False)

    # Ready marker
    with open(os.path.join(STEP2_OUT_DIR, "_READY.txt"), "w", encoding="utf-8") as f:
        f.write("ok")


# -----------------------------
# Auto-detect Step 1 cleaned output
# -----------------------------
def find_step1_clean_auto() -> str:
    preferred_csv = os.path.join("outputs", "step1", "combined_stations_clean_step1.csv")
    preferred_xlsx = os.path.join("outputs", "step1", "combined_stations_clean_step1.xlsx")

    if os.path.exists(preferred_csv):
        return preferred_csv
    if os.path.exists(preferred_xlsx):
        return preferred_xlsx

    hits = glob.glob("**/*clean_step1*.csv", recursive=True) + glob.glob("**/*clean_step1*.xlsx", recursive=True)
    hits = [h for h in hits if os.path.isfile(h) and ".git" not in h and ".venv" not in h]
    if hits:
        hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return hits[0]

    all_files = glob.glob("**/*.csv", recursive=True) + glob.glob("**/*.xlsx", recursive=True)
    all_files = [p for p in all_files if os.path.isfile(p) and ".git" not in p and ".venv" not in p]
    if all_files:
        all_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return all_files[0]

    return ""


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\t", "") for c in df.columns]

    if STATION_COL not in df.columns or DATE_COL not in df.columns:
        raise ValueError(f"Missing required columns: {STATION_COL}, {DATE_COL}")

    df[STATION_COL] = df[STATION_COL].astype(str).str.strip().str.lower()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([STATION_COL, DATE_COL])
    return df


def load_data_from_path(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx.")
    return normalize_df(df)


def load_data_from_upload(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx.")
    return normalize_df(df)


# -----------------------------
# Feature engineering
# -----------------------------
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

    # Lags
    for c in lag_cols:
        if c not in out.columns:
            continue
        for L in lags:
            out[f"{c}_lag{L}"] = g[c].shift(L)

    # Rolling SUM
    for c in roll_sum_cols:
        if c not in out.columns:
            continue
        for w in roll_sum_windows:
            out[f"{c}_sum{w}"] = g[c].rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)

    # Rolling MEAN
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
        return df_feat.groupby(STATION_COL, group_keys=False).apply(lambda g: g.ffill())

    # Time-based interpolation for numeric columns
    def interp_group(g):
        g = g.copy().sort_values(DATE_COL).set_index(DATE_COL)
        num_cols = g.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
        g[num_cols] = g[num_cols].interpolate(method="time", limit_direction="both")
        return g.reset_index()

    out = df_feat.groupby(STATION_COL, group_keys=False).apply(interp_group)
    out = out.sort_values([STATION_COL, DATE_COL])
    return out


# -----------------------------
# Sequences + diagnostics
# -----------------------------
def build_sequences(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    feature_cols: list[str],
    target_cols: list[str],
    window: int,
    horizon: int,
    allow_nan_ratio: float = 0.0
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    X_list, y_list, meta_rows = [], [], []
    diag_rows = []

    work = df.copy()

    # Coerce numeric for features/targets
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
                "reason": "too few rows for window+horizon"
            })
            continue

        possible = 0
        kept = 0
        dropped_target_nan = 0
        dropped_nan_ratio = 0

        for start in range(0, last_start):
            possible += 1
            end = start + window
            y_idx = end + horizon - 1

            x_win = feat[start:end, :].copy()
            y_val = targ[y_idx, :].copy()

            if np.isnan(y_val).any():
                dropped_target_nan += 1
                continue

            nan_count = np.isnan(x_win).sum()
            nan_ratio = nan_count / x_win.size

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
            meta_rows.append({"station": station, "target_date": pd.to_datetime(dates[y_idx])})
            kept += 1

        reason = "ok" if kept > 0 else "all windows dropped (NaNs/targets missing)"
        diag_rows.append({
            "station": station,
            "rows": n,
            "possible_windows": possible,
            "kept_windows": kept,
            "dropped_target_nan": dropped_target_nan,
            "dropped_nan_ratio": dropped_nan_ratio,
            "reason": reason
        })

    diag_df = pd.DataFrame(diag_rows).sort_values(["kept_windows", "possible_windows"])

    if not X_list:
        raise ValueError(
            "No sequences were generated. Tips: reduce window, disable lag/rolling, "
            "choose a NaN strategy, or increase allow_nan_ratio (e.g. 0.05)."
        )

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    meta_df = pd.DataFrame(meta_rows)
    return X, y, meta_df, diag_df


# -----------------------------
# UI
# -----------------------------
st.title("Step 2 — Feature engineering & LSTM sequences (with diagnostics)")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Source:", ["Auto-detect (repo)", "Upload"], index=0)

    auto_path = find_step1_clean_auto()
    uploaded = None

    if mode == "Auto-detect (repo)":
        st.write("Detected file:")
        st.code(auto_path if auto_path else "(no file found)")
    else:
        uploaded = st.file_uploader("Upload cleaned Step 1 file (CSV/XLSX)", type=["csv", "xlsx"])

    st.divider()
    st.header("Sequence settings")
    window = st.number_input("Window length (days)", min_value=3, max_value=365, value=30, step=1)
    horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=1, step=1)

    st.divider()
    st.header("NaN handling before sequences")
    nan_strategy = st.selectbox(
        "Missing value strategy",
        ["None (strict)", "Forward fill within station", "Time-based interpolation (numeric) within station"],
        index=0
    )
    allow_nan_ratio = st.slider(
        "Allowed NaN ratio inside X window",
        min_value=0.0, max_value=0.5, value=0.0, step=0.05
    )

    st.divider()
    st.header("Feature engineering")
    use_lags = st.checkbox("Add lag features", value=True)
    lags = st.multiselect("Lag days", options=[1, 2, 3, 5, 7, 10, 14, 21, 30], default=[1, 3, 7])

    use_roll = st.checkbox("Add rolling features", value=True)
    roll_sum_windows = st.multiselect("Rolling SUM windows", options=[3, 5, 7, 10, 14, 21, 30], default=[7, 14])
    roll_mean_windows = st.multiselect("Rolling MEAN windows", options=[3, 5, 7, 10, 14, 21, 30], default=[7, 14])

    st.divider()
    save_features_df = st.checkbox("Also save engineered df_feat to outputs/step2/", value=True)
    features_format = st.selectbox("df_feat format", ["Parquet (recommended)", "CSV"], index=0)

# Load data
try:
    if mode == "Auto-detect (repo)":
        if not auto_path:
            st.error("No Step 1 cleaned file found. Expected: outputs/step1/combined_stations_clean_step1.csv")
            st.stop()
        df = load_data_from_path(auto_path)
        input_label = auto_path
    else:
        if uploaded is None:
            st.info("Upload a cleaned CSV/XLSX file to continue.")
            st.stop()
        df = load_data_from_upload(uploaded)
        input_label = uploaded.name
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

st.subheader("Loaded data preview")
st.caption(f"Source: {input_label}")
st.dataframe(df.head(25), use_container_width=True)

# Targets
st.subheader("Targets (what you want to predict)")
existing_sm = [c for c in SM_COLS if c in df.columns]
existing_pf = [c for c in PF_COLS if c in df.columns]

target_mode = st.radio("Target set", ["Soil moisture (_SM*)", "pF2.5 (pF2.5_*)"], index=0)
if target_mode == "Soil moisture (_SM*)":
    if not existing_sm:
        st.error("No _SM* columns found in this file.")
        st.stop()
    target_cols = st.multiselect("Target columns", options=existing_sm, default=existing_sm)
else:
    if not existing_pf:
        st.error("No pF2.5_* columns found in this file.")
        st.stop()
    target_cols = st.multiselect("Target columns", options=existing_pf, default=existing_pf)

# Base features
st.subheader("Input features (base features)")
exclude = set([STATION_COL, DATE_COL]) | set(target_cols)

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
    st.warning("Select at least 1 input feature.")
    st.stop()

# Feature engineering column selection
lag_cols, roll_sum_cols, roll_mean_cols = [], [], []
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

with st.expander("Quick tips if you get 0 sequences", expanded=True):
    st.write("- Reduce window (e.g. 30 → 14 or 7).")
    st.write("- Temporarily disable lag/rolling.")
    st.write("- Choose a NaN strategy (interpolation) or increase allowed NaN ratio (e.g. 0.05).")
    st.write("- Use the diagnostics table to see which station drops windows.")

# Run
st.subheader("Run")
if st.button("Build features + generate sequences (SAVE to outputs)"):
    with st.spinner("Feature engineering..."):
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

    st.success("Feature engineering done ✅")
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
    st.write(f"X shape: {X.shape}  (samples, window, features)")
    st.write(f"y shape: {y.shape}  (samples, targets)")

    st.write("### Diagnostics (per station)")
    st.dataframe(diag_df, use_container_width=True)

    st.write("### Meta (first 20 samples)")
    st.dataframe(meta_df.head(20), use_container_width=True)

    # Store in session_state for Step 3
    st.session_state["step2_X"] = X
    st.session_state["step2_y"] = y
    st.session_state["step2_meta"] = meta_df
    st.session_state["step2_diag"] = diag_df
    st.session_state["step2_config"] = {
        "input_source": input_label,
        "window": int(window),
        "horizon": int(horizon),
        "nan_strategy": nan_strategy,
        "allow_nan_ratio": float(allow_nan_ratio),
        "target_cols": target_cols,
        "base_feature_cols": base_feature_cols,
        "engineered_feature_cols": engineered_cols,
        "final_feature_cols": final_feature_cols,
        "save_features_df": bool(save_features_df),
        "features_format": features_format,
    }

    # --- SAVE TO outputs/step2 ---
    try:
        df_to_save = None
        if save_features_df:
            df_to_save = df_feat.copy()
            if features_format.startswith("CSV"):
                df_to_save.to_csv(os.path.join(STEP2_OUT_DIR, "features_clean_step2.csv"), index=False)
            else:
                # parquet recommended; if fails, fallback in save_step2_artifacts
                try:
                    df_to_save.to_parquet(os.path.join(STEP2_OUT_DIR, "features_clean_step2.parquet"), index=False)
                except Exception:
                    pass

        save_step2_artifacts(
            X=X,
            y=y,
            meta_df=meta_df,
            diag_df=diag_df,
            config=st.session_state["step2_config"],
            df_feat=(df_feat if save_features_df else None)
        )

        st.success("Mentve ✅ Minden kimenet az outputs/step2/ mappába került.")
        st.code(STEP2_OUT_DIR)
        st.write("Saved files (typical): X.npy, y.npy, meta.csv, diagnostics.csv, config.json (+ optional features_clean_step2.*)")
    except Exception as e:
        st.error(f"Mentés sikertelen: {e}")

    st.divider()
    st.write("### Optional downloads (X/y .npy)")
    if st.checkbox("Enable downloads (X/y .npy)"):
        x_buf = io.BytesIO()
        y_buf = io.BytesIO()
        np.save(x_buf, X)
        np.save(y_buf, y)
        st.download_button("Download X.npy", data=x_buf.getvalue(), file_name="X.npy")
        st.download_button("Download y.npy", data=y_buf.getvalue(), file_name="y.npy")