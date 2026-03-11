import os
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Step 1 — Data cleaning", layout="wide")
st.title("Step 1 — Data cleaning")

# --------------------------------------------------
# Paths
# --------------------------------------------------
INPUT_FILE = "combined_stations.xlsx"
OUT_DIR = os.path.join("outputs", "step1")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "combined_stations_clean_step1.csv")
OUT_XLSX = os.path.join(OUT_DIR, "combined_stations_clean_step1.xlsx")

REQUIRED_COLS = ["station", "date"]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def clean_column_name(col: str) -> str:
    """
    Normalize column names while preserving meaning.
    """
    c = str(col).strip()
    c = c.replace("\t", "")
    c = c.replace("\n", "_")
    c = c.replace(" ", "_")
    c = c.replace("__", "_")
    while "__" in c:
        c = c.replace("__", "_")
    return c


def normalize_station(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def parse_date_column(series: pd.Series) -> pd.Series:
    """
    Robust parser for mixed date formats such as:
    - 2019. 5. 8. 2:00
    - 08/05/2019 02:00
    - 2019-05-08 02:00
    """
    s = series.astype(str).str.strip()

    # First try general parsing
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)

    # If many values fail, try dotted Hungarian-like format
    if dt.notna().sum() < max(1, int(len(s) * 0.7)):
        candidates = [
            "%Y. %m. %d. %H:%M",
            "%Y.%m.%d. %H:%M",
            "%d/%m/%Y %H:%M",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%d/%m/%Y",
        ]
        best = dt
        best_count = dt.notna().sum()

        for fmt in candidates:
            trial = pd.to_datetime(s, format=fmt, errors="coerce")
            count = trial.notna().sum()
            if count > best_count:
                best = trial
                best_count = count

        dt = best

    return dt


def try_convert_numeric(series: pd.Series) -> pd.Series:
    """
    Convert to numeric if enough values are numeric-like.
    Handles comma decimals safely.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()

    s = s.replace({
        "": np.nan,
        "nan": np.nan,
        "None": np.nan,
        "none": np.nan,
        "NA": np.nan,
        "N/A": np.nan,
        "-": np.nan
    })

    # convert decimal commas to decimal points
    s2 = s.str.replace(",", ".", regex=False)

    converted = pd.to_numeric(s2, errors="coerce")

    non_na_original = s.notna().sum()
    non_na_converted = converted.notna().sum()

    if non_na_original == 0:
        return converted

    ratio = non_na_converted / max(non_na_original, 1)

    # Use numeric version if most non-empty values are parseable
    if ratio >= 0.6:
        return converted

    return series


def validate_required_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def report_column_types(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        rows.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "missing_ratio": float(df[c].isna().mean())
        })
    return pd.DataFrame(rows)


# --------------------------------------------------
# Load
# --------------------------------------------------
st.subheader("1) Load input file")

if not os.path.exists(INPUT_FILE):
    st.error(f"Input file not found: {INPUT_FILE}")
    st.info("Place the file in the project folder with this exact name: combined_stations.xlsx")
    st.stop()

try:
    df = pd.read_excel(INPUT_FILE)
except Exception as e:
    st.error(f"Failed to read Excel file: {e}")
    st.stop()

st.success("Excel file loaded successfully.")
st.write("Original shape:", df.shape)
st.dataframe(df.head(20), use_container_width=True)

# --------------------------------------------------
# Clean
# --------------------------------------------------
st.subheader("2) Clean dataset")

try:
    df = df.copy()

    # Clean column names
    original_cols = list(df.columns)
    df.columns = [clean_column_name(c) for c in df.columns]

    # Validate required columns
    validate_required_columns(df)

    # Normalize station and parse date
    df["station"] = normalize_station(df["station"])
    df["date"] = parse_date_column(df["date"])

    # Drop rows with invalid date
    before_drop_date = len(df)
    df = df.dropna(subset=["date"])
    dropped_date_rows = before_drop_date - len(df)

    # Convert all non-key columns to numeric where possible
    for col in df.columns:
        if col in ["station", "date"]:
            continue
        df[col] = try_convert_numeric(df[col])

    # Sort
    df = df.sort_values(["station", "date"]).reset_index(drop=True)

except Exception as e:
    st.error(f"Cleaning failed: {e}")
    st.stop()

st.success("Cleaning finished.")
st.write("Cleaned shape:", df.shape)
st.write("Dropped rows with invalid/missing date:", dropped_date_rows)

with st.expander("Cleaned column names"):
    st.write(list(df.columns))

with st.expander("Column type summary"):
    st.dataframe(report_column_types(df), use_container_width=True)

st.dataframe(df.head(20), use_container_width=True)

# --------------------------------------------------
# Save
# --------------------------------------------------
st.subheader("3) Save outputs")

try:
    df.to_csv(OUT_CSV, index=False)
    df.to_excel(OUT_XLSX, index=False)

    st.success("Step 1 saved successfully ✅")
    st.write("Saved files:")
    st.code(OUT_CSV)
    st.code(OUT_XLSX)

except Exception as e:
    st.error(f"Saving failed: {e}")
    st.stop()

# --------------------------------------------------
# Summary
# --------------------------------------------------
st.subheader("4) Summary")

summary = {
    "input_file": INPUT_FILE,
    "output_folder": OUT_DIR,
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "stations": int(df["station"].nunique()) if "station" in df.columns else None,
    "date_min": str(df["date"].min()) if "date" in df.columns and not df.empty else None,
    "date_max": str(df["date"].max()) if "date" in df.columns and not df.empty else None,
}

st.json(summary)