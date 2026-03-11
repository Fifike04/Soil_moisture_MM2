import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Step 10 — Visual diagnostics", layout="wide")
st.title("Step 10 — Visual diagnostics")

STATION_COL = "station"
DATE_COL = "date"

OUT1 = os.path.join("outputs", "step1")
OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
OUT10 = os.path.join("outputs", "step10")
os.makedirs(OUT10, exist_ok=True)

# --------------------------------------------------
# Paths
# --------------------------------------------------
STEP1_CSV = os.path.join(OUT1, "combined_stations_clean_step1.csv")
STEP1_XLSX = os.path.join(OUT1, "combined_stations_clean_step1.xlsx")

X_PATH = os.path.join(OUT2, "X.npy")
Y_PATH = os.path.join(OUT2, "y.npy")
META_PATH = os.path.join(OUT2, "meta.csv")
CFG_PATH = os.path.join(OUT2, "config.json")
STEP2_FEAT_PARQUET = os.path.join(OUT2, "features_clean_step2.parquet")
STEP2_FEAT_CSV = os.path.join(OUT2, "features_clean_step2.csv")

MODEL_PATH = os.path.join(OUT3, "model.keras")
X_SCALER_PATH = os.path.join(OUT3, "x_scaler.json")

SUMMARY_JSON = os.path.join(OUT10, "step10_summary.json")
SCATTER_METRICS_CSV = os.path.join(OUT10, "scatter_metrics.csv")

DEFAULT_TARGETS = ["_SM10", "_SM20", "_SM30", "_SM45", "_SM60", "_SM75"]

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


def load_step2_feat():
    if os.path.exists(STEP2_FEAT_PARQUET):
        return pd.read_parquet(STEP2_FEAT_PARQUET), STEP2_FEAT_PARQUET
    if os.path.exists(STEP2_FEAT_CSV):
        return pd.read_csv(STEP2_FEAT_CSV), STEP2_FEAT_CSV
    return None, None


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\t", "") for c in df.columns]

    if STATION_COL not in df.columns or DATE_COL not in df.columns:
        raise ValueError(f"Missing required columns: {STATION_COL}, {DATE_COL}")

    df[STATION_COL] = df[STATION_COL].astype(str).str.strip().str.lower()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[DATE_COL]).sort_values([STATION_COL, DATE_COL]).reset_index(drop=True)
    return df


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

        scaler_type = sc.get("scaler_type", "StandardScaler")

        if scaler_type == "StandardScaler" and "mean" in sc and "std" in sc:
            mu = np.asarray(sc["mean"], dtype=np.float32)
            sd = np.asarray(sc["std"], dtype=np.float32)
            sd = np.where(sd == 0, 1.0, sd).astype(np.float32)
            return mu, sd

        return None, None
    except Exception:
        return None, None


def split_by_date(meta_df: pd.DataFrame, start: str, end: str) -> np.ndarray:
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    return ((d >= pd.to_datetime(start)) & (d <= pd.to_datetime(end))).to_numpy()


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def scatter_plot(y_true, y_pred, target_name: str):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]

    if len(yt) == 0:
        return None, None

    r2 = float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan")
    err_rmse = rmse(yt, yp)
    err_mae = mae(yt, yp)

    xy_min = float(min(np.min(yt), np.min(yp)))
    xy_max = float(max(np.max(yt), np.max(yp)))

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, alpha=0.5, s=18)

    # 1:1 line
    plt.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--")

    # regression line
    if len(yt) >= 2:
        a, b = np.polyfit(yt, yp, 1)
        xline = np.array([xy_min, xy_max], dtype=float)
        yline = a * xline + b
        plt.plot(xline, yline)
        eq_text = f"y = {a:.3f}x + {b:.3f}"
    else:
        eq_text = "y = n/a"

    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Observed vs Predicted — {target_name}")

    text = f"R² = {r2:.4f}\nRMSE = {err_rmse:.4f}\nMAE = {err_mae:.4f}\n{eq_text}"
    plt.text(
        0.05, 0.95, text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.2)
    )

    plt.tight_layout()

    metrics = {
        "target": target_name,
        "R2": r2,
        "RMSE": err_rmse,
        "MAE": err_mae,
        "n_points": int(len(yt)),
    }
    return fig, metrics


def cleaning_plot(
    df_raw: pd.DataFrame,
    df_feat: pd.DataFrame,
    station: str,
    value_col: str,
    start_date=None,
    end_date=None,
    smooth_window: int = 7,
):
    raw_s = df_raw[df_raw[STATION_COL] == station].copy()
    feat_s = df_feat[df_feat[STATION_COL] == station].copy()

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        raw_s = raw_s[raw_s[DATE_COL] >= start_date]
        feat_s = feat_s[feat_s[DATE_COL] >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        raw_s = raw_s[raw_s[DATE_COL] <= end_date]
        feat_s = feat_s[feat_s[DATE_COL] <= end_date]

    if value_col not in raw_s.columns or value_col not in feat_s.columns:
        return None

    raw_s = raw_s[[DATE_COL, value_col]].copy()
    feat_s = feat_s[[DATE_COL, value_col]].copy()

    raw_s[value_col] = pd.to_numeric(raw_s[value_col], errors="coerce")
    feat_s[value_col] = pd.to_numeric(feat_s[value_col], errors="coerce")

    feat_s[f"{value_col}_smooth"] = feat_s[value_col].rolling(
        window=max(1, int(smooth_window)),
        min_periods=1
    ).mean()

    fig = plt.figure(figsize=(12, 5))
    plt.plot(raw_s[DATE_COL], raw_s[value_col], label="Step1 raw/cleaned base", alpha=0.6)
    plt.plot(feat_s[DATE_COL], feat_s[value_col], label="Step2 engineered/filled", alpha=0.85)
    plt.plot(feat_s[DATE_COL], feat_s[f'{value_col}_smooth'], label=f"Smoothed ({smooth_window})", linewidth=2)

    plt.title(f"Cleaning / smoothing view — {station} — {value_col}")
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    return fig


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
# Load artifacts
# --------------------------------------------------
st.subheader("1) Load Step1 + Step2 + Step3 outputs")

require_file(X_PATH, "Step2 X.npy")
require_file(Y_PATH, "Step2 y.npy")
require_file(META_PATH, "Step2 meta.csv")
require_file(CFG_PATH, "Step2 config.json")
require_file(MODEL_PATH, "Step3 model.keras")

try:
    df_step1, step1_src = load_step1_output()
    df_step1 = normalize_df(df_step1)

    df_feat, feat_src = load_step2_feat()
    if df_feat is not None:
        df_feat = normalize_df(df_feat)

    X = np.load(X_PATH, allow_pickle=True)
    y = np.load(Y_PATH, allow_pickle=True)
    meta_df = pd.read_csv(META_PATH)

    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)

except Exception as e:
    st.error(f"Failed to load required files: {e}")
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

target_names = cfg.get("target_cols", DEFAULT_TARGETS)
if len(target_names) != y.shape[1]:
    target_names = [f"target{i}" for i in range(y.shape[1])]

st.success("Loaded successfully ✅")
st.write({
    "Step1_source": step1_src,
    "Step2_feature_source": feat_src,
    "X_shape": tuple(X.shape),
    "y_shape": tuple(y.shape),
    "targets": target_names
})

# --------------------------------------------------
# Load scaler
# --------------------------------------------------
st.subheader("2) Load Step3 scaler")

mu, sd = load_x_scaler(X_SCALER_PATH)
if mu is not None and sd is not None and len(mu) == X.shape[2]:
    st.success("Step3 x_scaler.json loaded ✅")
else:
    mu, sd = None, None
    st.info("No compatible x_scaler.json found. Scatter plots will use raw X.")

# --------------------------------------------------
# Controls
# --------------------------------------------------
st.subheader("3) Controls")

c1, c2, c3 = st.columns(3)
with c1:
    train_start = st.text_input("Train start", value="2016-01-01")
    train_end = st.text_input("Train end", value="2022-12-31")
with c2:
    val_start = st.text_input("Val start", value="2023-01-01")
    val_end = st.text_input("Val end", value="2023-12-31")
with c3:
    test_start = st.text_input("Test start", value="2024-01-01")
    test_end = st.text_input("Test end", value="2025-12-31")

subset_name = st.selectbox(
    "Which subset to use for scatter plots?",
    ["Test", "Validation", "Train"],
    index=0
)

apply_scaling = st.checkbox("Apply Step3 X scaling to scatter prediction", value=True)

st.divider()
st.subheader("4) Cleaning / smoothing plot settings")

station_list = sorted(df_step1[STATION_COL].dropna().unique().tolist())
default_station = station_list[0] if station_list else None

selected_station = st.selectbox("Station for cleaning plot", options=station_list, index=0 if station_list else None)

available_plot_cols = [c for c in df_step1.columns if c not in [STATION_COL, DATE_COL]]
default_plot_cols = [c for c in target_names if c in available_plot_cols]
if not default_plot_cols:
    default_plot_cols = available_plot_cols[:3]

selected_plot_cols = st.multiselect(
    "Columns to plot for cleaning / smoothing",
    options=available_plot_cols,
    default=default_plot_cols[:3]
)

cc1, cc2, cc3 = st.columns(3)
with cc1:
    plot_start_date = st.text_input("Plot start date", value="")
with cc2:
    plot_end_date = st.text_input("Plot end date", value="")
with cc3:
    smooth_window = st.number_input("Smoothing window", min_value=1, max_value=60, value=7, step=1)

# --------------------------------------------------
# Build subset for scatter
# --------------------------------------------------
st.subheader("5) Build subset for scatter plots")

train_idx = split_by_date(meta_df, train_start, train_end)
val_idx = split_by_date(meta_df, val_start, val_end)
test_idx = split_by_date(meta_df, test_start, test_end)

subset_map = {
    "Train": train_idx,
    "Validation": val_idx,
    "Test": test_idx,
}
mask = subset_map[subset_name]

st.write({
    "train_n": int(train_idx.sum()),
    "val_n": int(val_idx.sum()),
    "test_n": int(test_idx.sum()),
    "selected_subset": subset_name,
    "selected_n": int(mask.sum()),
})

if int(mask.sum()) == 0:
    st.error(f"{subset_name} subset has 0 samples.")
    st.stop()

X_sub = X[mask]
y_sub = y[mask]

if apply_scaling and mu is not None and sd is not None:
    X_used = apply_x_scaler(X_sub, mu, sd)
    st.info("Applied Step3 scaling to scatter subset.")
else:
    X_used = X_sub
    st.info("No scaling applied to scatter subset.")

# --------------------------------------------------
# Predict for scatter
# --------------------------------------------------
st.subheader("6) Scatter plots by soil-moisture depth")

with st.spinner("Predicting for scatter plots..."):
    y_pred = model.predict(X_used, verbose=0)

if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

if y_pred.shape != y_sub.shape:
    st.error(f"Prediction shape mismatch: y_pred={y_pred.shape}, y_true={y_sub.shape}")
    st.stop()

metrics_rows = []
for j in range(y_sub.shape[1]):
    target_name = target_names[j] if j < len(target_names) else f"target{j}"
    fig, metrics = scatter_plot(y_sub[:, j], y_pred[:, j], target_name)

    if fig is None:
        st.warning(f"No valid data to plot for {target_name}")
        continue

    st.pyplot(fig)
    metrics_rows.append(metrics)

metrics_df = pd.DataFrame(metrics_rows)
st.write("### Scatter metrics summary")
st.dataframe(metrics_df, use_container_width=True)

# --------------------------------------------------
# Cleaning / smoothing plots
# --------------------------------------------------
st.subheader("7) Cleaning / smoothing inspection plots")

if df_feat is None:
    st.warning("No Step2 engineered feature file found. Cleaning / smoothing plots need outputs/step2/features_clean_step2.*")
else:
    plot_cols_numeric = list(selected_plot_cols)
    df_step1_num = to_numeric_safe(df_step1, plot_cols_numeric)
    df_feat_num = to_numeric_safe(df_feat, plot_cols_numeric)

    for col in plot_cols_numeric:
        fig = cleaning_plot(
            df_raw=df_step1_num,
            df_feat=df_feat_num,
            station=selected_station,
            value_col=col,
            start_date=plot_start_date if str(plot_start_date).strip() else None,
            end_date=plot_end_date if str(plot_end_date).strip() else None,
            smooth_window=int(smooth_window),
        )
        if fig is None:
            st.warning(f"Could not plot {col} for station {selected_station}")
            continue
        st.pyplot(fig)

# --------------------------------------------------
# Save outputs
# --------------------------------------------------
st.subheader("8) Save outputs")

if st.button("Save Step 10 outputs to outputs/step10"):
    try:
        saved_files = []

        # Save scatter plots
        plot_files = []
        for j in range(y_sub.shape[1]):
            target_name = target_names[j] if j < len(target_names) else f"target{j}"
            fig, _ = scatter_plot(y_sub[:, j], y_pred[:, j], target_name)
            if fig is None:
                continue

            safe_name = str(target_name).replace("/", "_").replace("\\", "_")
            out_png = os.path.join(OUT10, f"scatter_{safe_name}.png")
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close(fig)
            plot_files.append(out_png)
            saved_files.append(out_png)

        # Save cleaning plots
        cleaning_files = []
        if df_feat is not None:
            plot_cols_numeric = list(selected_plot_cols)
            df_step1_num = to_numeric_safe(df_step1, plot_cols_numeric)
            df_feat_num = to_numeric_safe(df_feat, plot_cols_numeric)

            for col in plot_cols_numeric:
                fig = cleaning_plot(
                    df_raw=df_step1_num,
                    df_feat=df_feat_num,
                    station=selected_station,
                    value_col=col,
                    start_date=plot_start_date if str(plot_start_date).strip() else None,
                    end_date=plot_end_date if str(plot_end_date).strip() else None,
                    smooth_window=int(smooth_window),
                )
                if fig is None:
                    continue

                safe_station = str(selected_station).replace("/", "_").replace("\\", "_")
                safe_col = str(col).replace("/", "_").replace("\\", "_")
                out_png = os.path.join(OUT10, f"cleaning_{safe_station}_{safe_col}.png")
                fig.savefig(out_png, dpi=300, bbox_inches="tight")
                plt.close(fig)
                cleaning_files.append(out_png)
                saved_files.append(out_png)

        metrics_df.to_csv(SCATTER_METRICS_CSV, index=False)
        saved_files.append(SCATTER_METRICS_CSV)

        summary = {
            "inputs": {
                "step1_source": step1_src,
                "step2_feature_source": feat_src,
                "X": X_PATH,
                "y": Y_PATH,
                "meta": META_PATH,
                "config": CFG_PATH,
                "model": MODEL_PATH,
                "x_scaler": X_SCALER_PATH if (apply_scaling and mu is not None) else None,
            },
            "scatter_subset": {
                "name": subset_name,
                "train_range": [train_start, train_end],
                "val_range": [val_start, val_end],
                "test_range": [test_start, test_end],
                "n_samples": int(mask.sum()),
            },
            "cleaning_plot": {
                "station": selected_station,
                "columns": selected_plot_cols,
                "plot_start_date": plot_start_date,
                "plot_end_date": plot_end_date,
                "smooth_window": int(smooth_window),
            },
            "outputs": {
                "scatter_metrics_csv": SCATTER_METRICS_CSV,
                "scatter_plot_files": plot_files,
                "cleaning_plot_files": cleaning_files,
                "summary_json": SUMMARY_JSON,
            }
        }

        with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(make_json_safe(summary), f, ensure_ascii=False, indent=2)
        saved_files.append(SUMMARY_JSON)

        st.success("Step 10 saved successfully ✅")
        st.code("\n".join(saved_files))

    except Exception as e:
        st.error(f"Failed to save Step10 outputs: {e}")