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
    Normalize column names:
    - strip spaces
    - replace tabs with nothing
    - replace spaces with underscores
    - replace commas with dots
    - remove duplicated underscores
    """
    c = str(col).strip()
    c = c.replace("\t", "")
    c = c.replace(" ", "_")
    c = c.replace(",", ".")
    while "__" in c:
        c = c.replace("__", "_")
    return c


def normalize_station(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
    )


def parse_date_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def try_convert_numeric(series: pd.Series) -> pd.Series:
    """
    Try to convert object columns to numeric if possible.
    Handles comma decimal separators.
    Keeps original if conversion would fail for most values.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    s = series.astype(str).str.strip()

    # common placeholders -> NaN
    s = s.replace({
        "": np.nan,
        "nan": np.nan,
        "None": np.nan,
        "none": np.nan,
        "NA": np.nan,
        "N/A": np.nan,
        "-": np.nan
    })

    # decimal comma -> decimal point
    s2 = s.str.replace(",", ".", regex=False)

    converted = pd.to_numeric(s2, errors="coerce")

    # if enough values can be parsed, use numeric version
    non_na_original = s.notna().sum()
    non_na_converted = converted.notna().sum()

    if non_na_original == 0:
        return converted

    ratio = non_na_converted / max(non_na_original, 1)

    # threshold can be adjusted; 0.6 is a safe compromise
    if ratio >= 0.6:
        return converted

    return series


def validate_required_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# --------------------------------------------------
# Load
# --------------------------------------------------
st.subheader("1) Load input file")

if not os.path.exists(INPUT_FILE):
    st.error(f"Input file not found: {INPUT_FILE}")
    st.info("Tedd a fájlt a projekt mappájába ezzel a névvel: combined_stations.xlsx")
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
    # Clean column names
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    # Check required columns
    validate_required_columns(df)

    # Normalize station/date
    df["station"] = normalize_station(df["station"])
    df["date"] = parse_date_column(df["date"])

    # Drop rows with missing date
    before_drop_date = len(df)
    df = df.dropna(subset=["date"])
    dropped_date_rows = before_drop_date - len(df)

    # Convert non-key columns to numeric where possible
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