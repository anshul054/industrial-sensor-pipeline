"""
ingest.py
----------
Loads and lightly validates raw sensor CSV data before it's handed to
transform.py. "Validated" here means: correct dtypes, required columns
present, and timestamps parseable -- it does NOT mean the data is clean
(missing values, duplicates, and out-of-range readings are expected and
are transform.py's / quality_checks.py's job to handle).
"""

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "machine_id", "temperature_c", "vibration_mm_s", "pressure_psi", "rpm"]


def load_raw_readings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad_ts = df["timestamp"].isna().sum()
    if n_bad_ts:
        print(f"  Warning: dropping {n_bad_ts} rows with unparseable timestamps")
        df = df.dropna(subset=["timestamp"])

    df["machine_id"] = df["machine_id"].astype(str)

    for col in ["temperature_c", "vibration_mm_s", "pressure_psi", "rpm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df):,} rows from {path}")
    return df
