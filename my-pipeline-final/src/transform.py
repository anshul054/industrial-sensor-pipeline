"""
Takes validated raw sensor readings and turns them into a clean dataset
with some basic features on top, ready to be used for analytics or as
ML input.

Rough steps:
- drop exact duplicate rows
- resample each machine onto a consistent 5-min grid
- forward-fill short gaps (<=15 min), leave longer gaps as null
- add rolling mean/std per sensor
- flag anomalies with a rolling z-score
- bolt on a fake "hours since last maintenance" column, just to show
  how you'd join in operational metadata for a real feature pipeline
"""

import numpy as np
import pandas as pd

RESAMPLE_FREQ = "5min"
SHORT_GAP_LIMIT = pd.Timedelta(minutes=15)
ROLLING_WINDOW = 12  # 12 * 5min = 1 hour rolling window
Z_SCORE_THRESHOLD = 3.0

SENSOR_COLS = ["temperature_c", "vibration_mm_s", "pressure_psi", "rpm"]


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["machine_id", "timestamp"]).copy()
    removed = before - len(df)
    if removed:
        print(f"  Dropped {removed} duplicate rows")
    return df


def resample_machine(group: pd.DataFrame, machine_id: str) -> pd.DataFrame:
    """Resample a single machine's readings onto a fixed time grid."""
    group = group.set_index("timestamp").sort_index()
    resampled = group[SENSOR_COLS].resample(RESAMPLE_FREQ).mean()

    # Forward-fill only short gaps; longer gaps stay NaN (flagged, not fabricated)
    filled = resampled.copy()
    for col in SENSOR_COLS:
        filled[col] = _fill_short_gaps(filled[col], SHORT_GAP_LIMIT, RESAMPLE_FREQ)

    filled["machine_id"] = machine_id
    return filled.reset_index()


def _fill_short_gaps(series: pd.Series, limit: pd.Timedelta, freq: str) -> pd.Series:
    """Forward-fill NaN runs shorter than limit; leave longer runs untouched."""
    max_periods = int(limit / pd.Timedelta(freq))
    return series.ffill(limit=max_periods)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean/std per machine per sensor column."""
    df = df.sort_values(["machine_id", "timestamp"]).copy()
    for col in SENSOR_COLS:
        df[f"{col}_roll_mean"] = (
            df.groupby("machine_id")[col]
            .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=3).mean())
        )
        df[f"{col}_roll_std"] = (
            df.groupby("machine_id")[col]
            .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=3).std())
        )
    return df


def add_anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag readings that deviate > Z_SCORE_THRESHOLD std devs from rolling mean."""
    df = df.copy()
    any_anomaly = pd.Series(False, index=df.index)

    for col in SENSOR_COLS:
        mean_col, std_col = f"{col}_roll_mean", f"{col}_roll_std"
        z = (df[col] - df[mean_col]) / df[std_col].replace(0, np.nan)
        flag_col = f"{col}_anomaly"
        df[flag_col] = (z.abs() > Z_SCORE_THRESHOLD).fillna(False)
        any_anomaly = any_anomaly | df[flag_col]

    df["any_anomaly"] = any_anomaly
    return df


def add_maintenance_feature(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """
    Simulate a 'time since last maintenance' feature per machine, the
    kind of operational metadata a real ML feature pipeline would join
    in from a separate maintenance-log table.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["hours_since_maintenance"] = np.nan

    for machine_id, group in df.groupby("machine_id"):
        idx = group.index
        n = len(idx)
        # Simulate maintenance events every ~4-8 days
        maintenance_gap_hours = rng.uniform(96, 192)
        hours_elapsed = np.arange(n) * (5 / 60)  # 5-min steps in hours
        hours_since = hours_elapsed % maintenance_gap_hours
        df.loc[idx, "hours_since_maintenance"] = hours_since

    return df


def run_transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full transform pipeline: dedupe -> resample -> features -> anomalies."""
    print("Running transform pipeline...")
    df = drop_duplicates(raw_df)

    print("  Resampling per machine onto 5-min grid...")
    resampled_frames = [
        resample_machine(group, machine_id)
        for machine_id, group in df.groupby("machine_id")
    ]
    resampled = pd.concat(resampled_frames, ignore_index=True)

    print("  Computing rolling features...")
    featured = add_rolling_features(resampled)

    print("  Flagging anomalies...")
    flagged = add_anomaly_flags(featured)

    print("  Adding maintenance feature...")
    final = add_maintenance_feature(flagged)

    print(f"  Transform complete: {len(final):,} rows, {final['any_anomaly'].sum():,} anomalies flagged")
    return final


if __name__ == "__main__":
    import os
    from ingest import load_raw_readings

    path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sensor_readings.csv")
    raw = load_raw_readings(path)
    result = run_transform(raw)
    print(result.head(10))
