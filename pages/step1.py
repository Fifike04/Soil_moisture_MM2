import os
import glob
import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Step 1 — Data validation & cleaning", layout="wide")

DEFAULT_DATE_COL = "date"
DEFAULT_STATION_COL = "station"

SM_COLS = ["_SM10", "_SM20", "_SM30", "_SM45", "_SM60", "_SM75"]
PF_COLS = ["pF2.5_10", "pF2.5_20", "pF2.5_30", "pF2.5_45", "pF2.5_60", "pF2.5_70"]

# -----------------------------
# Auto-detect Excel in repo
# -----------------------------
def find_excel_auto() -> str:
    """
    Prefer: data/combined_stations.xlsx
    Else: search recursively for combined_stations.xlsx
    Else: pick the most recently modified .xlsx
    """
    preferred = os.path.join("data", "combined_stations.xlsx")
    if os.path.exists(preferred):
        return preferred

    hits = glob.glob("**/combined_stations.xlsx", recursive=True)
    hits = [h for h in hits if os.path.isfile(h)]
    if hits:
        hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return hits[0]

    all_xlsx = glob.glob("**/*.xlsx", recursive=True)
    all_xlsx = [p for p in all_xlsx if os.path.isfile(p) and ".git" not in p and ".venv" not in p]
    if all_xlsx:
        all_xlsx.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return all_xlsx[0]

    return ""

def sanitize_colnames(cols):
    return [str(c).strip().replace("\t", "") for c in cols]

def to_float_safe(series: pd.Series) -> pd.Series:
    """Convert messy numeric columns to float (supports comma decimals)."""
    if series.dtype.kind in "if":
        return series.astype(float)

    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaT": np.nan})

    # European decimal comma -> dot
    s = s.str.replace(r"(?<=\d),(?=\d)", ".", regex=True)

    # keep only digits, dot, minus
    s = s.str.replace(r"[^0-9\.\-]+", "", regex=True)

    return pd.to_numeric(s, errors="coerce")

def parse_date_col(series: pd.Series) -> pd.Series:
    """
    Parse dates like '2019. 5. 8. 2:00' robustly.
    """
    s = series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"(\d{4})\.\s*", r"\1-", regex=True)
    s = s.str.replace(r"(\d{1,2})\.\s*", r"\1-", regex=True)
    s = s.str.replace(r"-\s*", "-", regex=True)
    return pd.to_datetime(s, errors="coerce")

def summarize_missing(df: pd.DataFrame, top_n=30):
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    return miss.head(top_n)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# UI
# -----------------------------
st.title("Step 1 — Data validation & cleaning (auto-detect input)")

with st.sidebar:
    st.header("Auto input detection")
    auto_path = find_excel_auto()
    st.write("Detected Excel file:")
    st.code(auto_path if auto_path else "(no .xlsx found)")

    station_col = st.text_input("Station column name", value=DEFAULT_STATION_COL)
    date_col = st.text_input("Date column name", value=DEFAULT_DATE_COL)

    st.divider()
    st.header("Output (save to folder)")
    out_dir = st.text_input("Output folder", value=os.path.join("outputs", "step1"))
    out_csv_name = st.text_input("Output CSV filename", value="combined_stations_clean_step1.csv")
    out_xlsx_name = st.text_input("Output XLSX filename (optional)", value="combined_stations_clean_step1.xlsx")

    st.divider()
    st.header("Missing value handling (optional)")
    fill_choice = st.selectbox(
        "Basic strategy",
        ["Do not fill", "Forward fill within station", "Time-based interpolation (numeric) within station"],
        index=0,
    )

if not auto_path:
    st.error("No Excel file found in the repo. Put it here: data/combined_stations.xlsx")
    st.stop()

try:
    df = pd.read_excel(auto_path)
except Exception as e:
    st.error(f"Failed to load Excel: {e}")
    st.stop()

df.columns = sanitize_colnames(df.columns)

missing_required = [c for c in [station_col, date_col] if c not in df.columns]
if missing_required:
    st.error(f"Missing required columns: {missing_required}")
    st.write("Columns found:", list(df.columns))
    st.stop()

st.subheader("Raw data preview")
st.dataframe(df.head(30), use_container_width=True)

# -----------------------------
# Cleaning
# -----------------------------
st.subheader("Cleaning & type conversion")

df[date_col] = parse_date_col(df[date_col])
df[station_col] = df[station_col].astype(str).str.strip().str.lower()

convert_candidates = [c for c in df.columns if c not in [station_col, date_col]]
numeric_cols = []
for c in convert_candidates:
    sample = df[c].dropna().astype(str).head(50)
    if len(sample) == 0:
        continue
    digit_ratio = sample.str.contains(r"\d").mean()
    if digit_ratio >= 0.6:
        numeric_cols.append(c)

for c in numeric_cols:
    df[c] = to_float_safe(df[c])

df = df.sort_values([station_col, date_col])

dup_count_before = int(df.duplicated().sum())

# Optional missing fill
if fill_choice != "Do not fill":
    if fill_choice == "Forward fill within station":
        df = df.groupby(station_col, group_keys=False).apply(lambda g: g.ffill())
    else:
        def interp_group(g):
            g = g.copy().sort_values(date_col).set_index(date_col)
            num_cols = g.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
            g[num_cols] = g[num_cols].interpolate(method="time", limit_direction="both")
            return g.reset_index()

        df = df.groupby(station_col, group_keys=False).apply(interp_group)
        df = df.sort_values([station_col, date_col])

# Reports
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", len(df))
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    st.metric("Invalid dates (NaT)", int(df[date_col].isna().sum()))
with col4:
    st.metric("Duplicate rows (before drop)", dup_count_before)

st.write("### Missing values (top)")
miss = summarize_missing(df, top_n=40)
if len(miss) == 0:
    st.success("No missing values detected (or filled).")
else:
    st.dataframe(miss.to_frame("missing_count"), use_container_width=True)

st.write("### Column dtypes")
st.dataframe(
    pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]}),
    use_container_width=True
)

existing_sm = [c for c in SM_COLS if c in df.columns]
existing_pf = [c for c in PF_COLS if c in df.columns]
with st.expander("Detected target columns", expanded=True):
    st.write("Soil moisture (_SM*) columns:", existing_sm if existing_sm else "None detected")
    st.write("pF2.5 columns:", existing_pf if existing_pf else "None detected")

st.subheader("Cleaned data preview")
st.dataframe(df.head(30), use_container_width=True)

# -----------------------------
# Save outputs to folder
# -----------------------------
st.subheader("Save cleaned dataset to folder")

ensure_dir(out_dir)
out_csv_path = os.path.join(out_dir, out_csv_name)
out_xlsx_path = os.path.join(out_dir, out_xlsx_name)

save_csv = st.button("Save cleaned CSV to output folder")
save_xlsx = st.button("Save cleaned XLSX to output folder (optional)")

if save_csv:
    try:
        df.to_csv(out_csv_path, index=False, encoding="utf-8")
        st.success(f"Saved: {out_csv_path}")
    except Exception as e:
        st.error(f"CSV save error: {e}")

if save_xlsx:
    try:
        with pd.ExcelWriter(out_xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="clean_step1")
        st.success(f"Saved: {out_xlsx_path}")
    except Exception as e:
        st.error(f"XLSX save error: {e}")

# Optional downloads
st.divider()
st.write("### Optional downloads")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download cleaned CSV", data=csv_bytes, file_name=out_csv_name, mime="text/csv")

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="clean_step1")
st.download_button(
    "Download cleaned XLSX",
    data=buf.getvalue(),
    file_name=out_xlsx_name,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)