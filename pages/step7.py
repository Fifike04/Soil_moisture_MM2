import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Step 7 — Seasonality & regime report", layout="wide")
st.title("Step 7 — Seasonality & regime performance report (DISK ONLY from outputs/)")

OUT2 = os.path.join("outputs", "step2")
OUT3 = os.path.join("outputs", "step3")
OUT7 = os.path.join("outputs", "step7")
os.makedirs(OUT7, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def require(path: str, label: str):
    if not os.path.exists(path):
        st.error(f"Missing required file: {label}")
        st.code(path)
        st.stop()

def season_from_month(m: int) -> str:
    if m in (12, 1, 2): return "DJF"
    if m in (3, 4, 5):  return "MAM"
    if m in (6, 7, 8):  return "JJA"
    return "SON"

def date_mask(meta_df: pd.DataFrame, start: str, end: str) -> np.ndarray:
    d = pd.to_datetime(meta_df["target_date"], errors="coerce")
    return ((d >= pd.to_datetime(start)) & (d <= pd.to_datetime(end))).to_numpy()

def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = (y_pred - y_true)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"MAE_pooled": mae, "RMSE_pooled": rmse}

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
        rows.append({"target": target_names[j], "MAE": mae, "RMSE": rmse, "R2": r2})
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

def apply_saved_scaler_if_exists(X_3d: np.ndarray, scaler_path: str) -> tuple[np.ndarray, dict | None]:
    """
    Apply Step3 x_scaler.json (mean/std per feature) to X.
    """
    if not os.path.exists(scaler_path):
        return X_3d, None
    try:
        with open(scaler_path, "r", encoding="utf-8") as f:
            sc = json.load(f)
        mu = np.asarray(sc["mean"], dtype=np.float32)
        sd = np.asarray(sc["std"], dtype=np.float32)
        sd = np.where(sd == 0, 1.0, sd).astype(np.float32)

        if X_3d.shape[-1] != mu.shape[0]:
            st.warning("x_scaler.json exists but feature dimension mismatch. Skipping scaling.")
            return X_3d, None

        return ((X_3d.astype(np.float32) - mu) / sd).astype(np.float32), sc
    except Exception:
        st.warning("Failed to load/apply x_scaler.json. Proceeding without scaling.")
        return X_3d, None

# -----------------------------
# Load disk artifacts
# -----------------------------
x_path = os.path.join(OUT2, "X.npy")
y_path = os.path.join(OUT2, "y.npy")
meta_path = os.path.join(OUT2, "meta.csv")
cfg_path = os.path.join(OUT2, "config.json")
model_path = os.path.join(OUT3, "model.keras")
scaler_path = os.path.join(OUT3, "x_scaler.json")

require(x_path, "Step2 X.npy")
require(y_path, "Step2 y.npy")
require(meta_path, "Step2 meta.csv")
require(cfg_path, "Step2 config.json")
require(model_path, "Step3 model.keras")

X = np.load(x_path, allow_pickle=True)
y = np.load(y_path, allow_pickle=True)
meta_df = pd.read_csv(meta_path)
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

if y.ndim == 1:
    y = y.reshape(-1, 1)
if X.ndim != 3 or y.ndim != 2:
    st.error(f"Expected X 3D and y 2D. Got X={X.shape}, y={y.shape}")
    st.stop()

if "target_date" not in meta_df.columns:
    st.error("meta.csv must contain 'target_date'.")
    st.stop()

meta_df["target_date"] = pd.to_datetime(meta_df["target_date"], errors="coerce")
if meta_df["target_date"].isna().all():
    st.error("All target_date are NaT. Fix Step2 meta.csv export.")
    st.stop()

target_names = cfg.get("target_cols", [f"target{i}" for i in range(y.shape[1])])
if len(target_names) != y.shape[1]:
    target_names = [f"target{i}" for i in range(y.shape[1])]

st.success("Loaded Step2 + Step3 artifacts ✅")
with st.expander("Config preview"):
    st.json(cfg)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Test date range (by target_date)")
    test_start = st.text_input("Test start", value="2024-01-01")
    test_end = st.text_input("Test end", value="2025-12-31")

    st.divider()
    st.header("Scaling")
    apply_scaling = st.checkbox("Apply Step3 input scaling (x_scaler.json) if available", value=True)

    st.divider()
    st.header("Optional: Wet/Dry regime split")
    regime_on = st.checkbox("Enable wet/dry regime split", value=False)
    regime_proxy = st.text_input("Proxy column (meta.csv OR joinable from features df)", value="_NapiCsapadek_sum7")
    wet_quantile = st.slider("Wet threshold quantile", 0.5, 0.95, 0.8, 0.05)

    st.divider()
    st.header("Export")
    export_to_disk = st.checkbox("Export CSV to outputs/step7/", value=True)

# -----------------------------
# Subset
# -----------------------------
test_idx = date_mask(meta_df, test_start, test_end)
n_test = int(test_idx.sum())

st.subheader("Dataset summary")
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Samples", int(X.shape[0]))
with c2: st.metric("Timesteps", int(X.shape[1]))
with c3: st.metric("Features", int(X.shape[2]))
with c4: st.metric("Targets", int(y.shape[1]))

st.subheader("Test subset")
st.write({"test_samples": n_test, "test_range": [test_start, test_end]})
if n_test == 0:
    st.error("No samples in test range. Adjust dates.")
    st.info(f"Available target_date range: {meta_df['target_date'].min()} — {meta_df['target_date'].max()}")
    st.stop()

X_test = X[test_idx]
y_test = y[test_idx]
meta_test = meta_df.loc[test_idx].reset_index(drop=True)

meta_test["month"] = meta_test["target_date"].dt.month.astype("Int64")
meta_test["season"] = meta_test["month"].astype(int).apply(season_from_month)

# -----------------------------
# Load model + predict (with Step3 scaling)
# -----------------------------
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error("Failed to load model.keras")
    st.exception(e)
    st.stop()

X_eval = X_test
sc_used = None
if apply_scaling:
    X_eval, sc_used = apply_saved_scaler_if_exists(X_eval, scaler_path)
    if sc_used is not None:
        st.success("Applied Step3 scaling ✅")
    else:
        st.info("No compatible scaler applied.")

with st.spinner("Predicting on test subset..."):
    y_pred = model.predict(X_eval, verbose=0)

if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

if y_pred.shape != y_test.shape:
    st.error(f"Prediction shape mismatch: y_pred={y_pred.shape}, y_true={y_test.shape}")
    st.stop()

# -----------------------------
# Monthly (pooled)
# -----------------------------
st.subheader("Monthly performance (pooled across targets)")
month_rows = []
for m in range(1, 13):
    idx = (meta_test["month"] == m).to_numpy()
    if idx.sum() == 0:
        month_rows.append({"month": m, "n": 0, "MAE_pooled": np.nan, "RMSE_pooled": np.nan})
        continue
    met = pooled_metrics(y_test[idx], y_pred[idx])
    month_rows.append({"month": m, "n": int(idx.sum()), **met})

df_month = pd.DataFrame(month_rows)
st.dataframe(df_month, use_container_width=True)
st.pyplot(plot_month_bar(df_month.dropna(), "RMSE_pooled", "Monthly RMSE (pooled)"))

# -----------------------------
# Seasonal (pooled)
# -----------------------------
st.subheader("Seasonal performance (pooled across targets)")
season_rows = []
for s in ["DJF", "MAM", "JJA", "SON"]:
    idx = (meta_test["season"] == s).to_numpy()
    if idx.sum() == 0:
        season_rows.append({"season": s, "n": 0, "MAE_pooled": np.nan, "RMSE_pooled": np.nan})
        continue
    met = pooled_metrics(y_test[idx], y_pred[idx])
    season_rows.append({"season": s, "n": int(idx.sum()), **met})

df_season = pd.DataFrame(season_rows)
st.dataframe(df_season, use_container_width=True)
st.pyplot(plot_season_bar(df_season.dropna(), "RMSE_pooled", "Seasonal RMSE (pooled)"))

# -----------------------------
# Per-target by season
# -----------------------------
with st.expander("Per-target metrics by season (table)", expanded=False):
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
        st.info("No samples for seasonal breakdown.")

# -----------------------------
# Wet/Dry regime (optional)
# -----------------------------
st.subheader("Wet/Dry regime report (optional)")

df_reg = None
proxy_series = None

# 1) Try meta.csv directly
if regime_on and regime_proxy in meta_test.columns:
    proxy_series = pd.to_numeric(meta_test[regime_proxy], errors="coerce")

# 2) If not in meta, try join from engineered df if exists
if regime_on and proxy_series is None:
    feat_parquet = os.path.join(OUT2, "features_clean_step2.parquet")
    feat_csv = os.path.join(OUT2, "features_clean_step2.csv")

    if os.path.exists(feat_parquet) or os.path.exists(feat_csv):
        df_feat = pd.read_parquet(feat_parquet) if os.path.exists(feat_parquet) else pd.read_csv(feat_csv)
        # normalize join keys
        if "station" in meta_test.columns and "date" in df_feat.columns:
            df_feat = df_feat.copy()
            df_feat["station"] = df_feat["station"].astype(str).str.strip().str.lower()
            df_feat["date"] = pd.to_datetime(df_feat["date"], errors="coerce")
            tmp = meta_test.copy()
            tmp["station"] = tmp.get("station", "").astype(str).str.strip().str.lower()

            joined = tmp.merge(
                df_feat[["station", "date", regime_proxy]].dropna(subset=["date"]),
                left_on=["station", "target_date"],
                right_on=["station", "date"],
                how="left"
            )
            if regime_proxy in joined.columns:
                proxy_series = pd.to_numeric(joined[regime_proxy], errors="coerce")
                st.info("Regime proxy loaded by join (meta.target_date == features.date).")
    else:
        st.info("No engineered features file found to join regime proxy.")

if regime_on:
    if proxy_series is None or proxy_series.isna().all():
        st.warning(f"Regime proxy '{regime_proxy}' not found / not joinable. Disable regime split or export proxy into meta.csv in Step2.")
    else:
        thr = proxy_series.quantile(wet_quantile)
        is_wet = (proxy_series >= thr).to_numpy()
        is_dry = (proxy_series < thr).to_numpy()

        st.write({"proxy": regime_proxy, "wet_quantile": wet_quantile, "threshold": float(thr)})

        wet_metrics = pooled_metrics(y_test[is_wet], y_pred[is_wet]) if is_wet.sum() > 0 else {"MAE_pooled": np.nan, "RMSE_pooled": np.nan}
        dry_metrics = pooled_metrics(y_test[is_dry], y_pred[is_dry]) if is_dry.sum() > 0 else {"MAE_pooled": np.nan, "RMSE_pooled": np.nan}

        df_reg = pd.DataFrame([
            {"regime": "wet", "n": int(is_wet.sum()), **wet_metrics},
            {"regime": "dry", "n": int(is_dry.sum()), **dry_metrics},
        ])
        st.dataframe(df_reg, use_container_width=True)

# -----------------------------
# Export
# -----------------------------
if export_to_disk:
    out_month = os.path.join(OUT7, "monthly_pooled_metrics.csv")
    out_season = os.path.join(OUT7, "seasonal_pooled_metrics.csv")
    df_month.to_csv(out_month, index=False)
    df_season.to_csv(out_season, index=False)

    out_depth_season = None
    blocks = []
    for s in ["DJF", "MAM", "JJA", "SON"]:
        idx = (meta_test["season"] == s).to_numpy()
        if idx.sum() == 0:
            continue
        df_t = per_target_metrics_numpy(y_test[idx], y_pred[idx], target_names)
        df_t.insert(0, "season", s)
        blocks.append(df_t)
    if blocks:
        out_depth_season = os.path.join(OUT7, "per_target_by_season.csv")
        pd.concat(blocks, ignore_index=True).to_csv(out_depth_season, index=False)

    out_reg = None
    if df_reg is not None:
        out_reg = os.path.join(OUT7, "regime_metrics.csv")
        df_reg.to_csv(out_reg, index=False)

    summary_path = os.path.join(OUT7, "step7_summary.json")
    summary = {
        "inputs": {
            "X": x_path,
            "y": y_path,
            "meta": meta_path,
            "config": cfg_path,
            "model": model_path,
            "x_scaler": scaler_path if sc_used is not None else None,
        },
        "test_range": {"start": test_start, "end": test_end, "n": int(n_test)},
        "exports": {
            "monthly": out_month,
            "seasonal": out_season,
            "per_target_by_season": out_depth_season,
            "regime": out_reg,
            "summary": summary_path,
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    st.success("Exported ✅")
    st.code("\n".join([p for p in [out_month, out_season, out_depth_season, out_reg, summary_path] if p]))