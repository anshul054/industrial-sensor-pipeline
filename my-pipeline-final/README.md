# Industrial Sensor ETL Pipeline

A Python feature pipeline that ingests messy multi-machine sensor data (temperature, vibration, pressure, RPM), resamples it onto a consistent time grid, engineers rolling features, flags anomalies with a rolling z-score, and joins in a simulated maintenance-log feature — the kind of pipeline that would feed a predictive maintenance model.

**Live demo:** [add your Streamlit Cloud link here after deploying] — upload your own sensor CSV or explore the built-in sample data, see anomalies flagged per machine in real time.

## About this project

In real industrial settings, machines constantly stream sensor data — temperature, vibration, pressure, RPM — and that data is almost never clean. Readings arrive at slightly irregular intervals, sensors occasionally send duplicate or dropped readings, and short network hiccups create small gaps. Before any of that data is useful for monitoring or machine learning, it has to be put into a consistent, trustworthy shape.

This project simulates that real-world problem end to end:

- **Multiple machines** streaming four correlated sensor readings each
- **A two-stage pipeline** (`ingest.py` → `transform.py`) that separates "load and validate the shape of the data" from "clean, engineer features, and flag anomalies" — mirroring how a real data engineering pipeline is usually split into distinct, testable stages
- **Time-series-aware cleaning**, not naive cleaning — gaps are only filled if they're short enough to trust, rolling statistics are only computed after the data is on a consistent time grid, and anomaly detection is scoped per machine rather than judged against one global threshold
- **A stand-in for real operational metadata** (`hours_since_maintenance`), included to show how a feature pipeline like this would eventually join in data from other systems (e.g. a maintenance-log database) rather than working off sensor data alone

**The end goal** is a feature-ready DataFrame that could plug directly into a predictive maintenance model — the kind of model that tries to answer "is this machine likely to fail soon?" using recent sensor trends and time-since-service as inputs.

**Why this project exists:** most beginner data-cleaning projects use a single, mostly-clean CSV. This one is deliberately built around a messier, more realistic scenario — multiple entities (machines) with correlated time-series data — because that's closer to what an actual data/ML engineering role involves, and it gives more to talk through in a technical interview than "I filled in some missing values with the mean."

## What it actually does

```
raw CSV (data/raw/sensor_readings.csv)
        │
        ▼
ingest.py        → load_raw_readings()
        │           validates required columns exist, parses timestamps,
        │           coerces sensor columns to numeric. Does NOT clean the
        │           data — missing values, duplicates, and out-of-range
        │           readings are expected to still be present here.
        ▼
transform.py      → run_transform()
        │
        ├─ drop_duplicates()        exact (machine_id, timestamp) duplicates removed
        ├─ resample_machine()       each machine resampled onto a 5-min grid
        │                           short gaps (≤15 min) forward-filled;
        │                           longer gaps left as NaN, not fabricated
        ├─ add_rolling_features()   1-hour rolling mean/std per sensor, per machine
        ├─ add_anomaly_flags()      rolling z-score per sensor; any_anomaly = True
        │                           if any sensor exceeds ±3 std devs from its
        │                           own rolling mean
        └─ add_maintenance_feature() simulated hours-since-last-maintenance,
                                     standing in for what would normally be a
                                     join against a real maintenance-log table
        ▼
final feature-ready DataFrame (20 columns: raw readings + rolling stats +
anomaly flags + maintenance feature)
```

## Why it's built this way

- **Resample before feature engineering, not after.** Rolling mean/std only means something on a regular time grid — if you compute a "1-hour rolling window" on irregularly-spaced raw timestamps, the window doesn't actually represent 1 hour of data. Resampling first makes every downstream feature interpretable.
- **Short gaps get forward-filled, long gaps don't.** A sensor missing one reading (5–15 min) is almost certainly a transient blip, safe to carry forward. A gap longer than 15 minutes is more likely a real outage, and forward-filling across it would fabricate readings that never happened — so it's left as NaN instead, to be handled explicitly downstream rather than silently smoothed over.
- **Z-score is computed per machine, not globally.** Each machine's rolling mean/std comes from its own history, so a vibration level that's normal for one machine but abnormal for another is judged against the right baseline.
- **The maintenance feature is intentionally simulated**, not fabricated to look real. In production this would be a join against an actual maintenance-log table (machine_id, timestamp of last service). It's included here to show the shape of that feature and how it would slot into the pipeline, since predictive maintenance models typically rely heavily on "time since last service" as a signal.

## Project structure

```
industrial-sensor-pipeline/
├── src/
│   ├── ingest.py       # load + lightly validate raw CSV
│   └── transform.py    # dedupe, resample, feature engineering, anomaly flags
├── data/
│   └── raw/             # place sensor_readings.csv here
└── README.md
```

## Getting started

```bash
git clone https://github.com/anshul054/industrial-sensor-pipeline.git
cd industrial-sensor-pipeline
pip install pandas numpy
```

Place a CSV at `data/raw/sensor_readings.csv` with these columns:
`timestamp, machine_id, temperature_c, vibration_mm_s, pressure_psi, rpm`

Then run:

```bash
cd src
python transform.py
```

This loads the raw CSV, runs the full transform, and prints a preview of the resulting feature-ready DataFrame along with a summary line (row count, anomalies flagged).

### Example output

```
Loaded 50,650 rows from ../data/raw/sensor_readings.csv
Running transform pipeline...
  Dropped 250 duplicate rows
  Resampling per machine onto 5-min grid...
  Computing rolling features...
  Flagging anomalies...
  Adding maintenance feature...
  Transform complete: 10,080 rows, 197 anomalies flagged
```

## Output columns

| Column | Description |
|---|---|
| `timestamp`, `machine_id` | Resampled 5-min grid index |
| `temperature_c`, `vibration_mm_s`, `pressure_psi`, `rpm` | Resampled sensor readings |
| `{sensor}_roll_mean`, `{sensor}_roll_std` | 1-hour rolling mean/std per sensor |
| `{sensor}_anomaly` | Boolean, True if that sensor's z-score exceeds ±3 |
| `any_anomaly` | True if any sensor was flagged for that row |
| `hours_since_maintenance` | Simulated time since last maintenance event |

## Running the interactive dashboard locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This opens a browser dashboard where you can upload a CSV (or use built-in sample data), pick a machine and sensor, and see the resampled reading plotted against its rolling mean with flagged anomalies marked.

## Deploying the live demo (Streamlit Community Cloud, free)

1. Push this repo to GitHub (see commands above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo, branch `main`, and set the main file path to `streamlit_app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically and gives you a public URL like `https://your-app-name.streamlit.app`.
5. Add that URL to the "Live demo" line at the top of this README, and use it in your resume/portfolio.

Redeploys happen automatically on every push to `main`.

## Skills this project demonstrates

- Pandas time-series handling: `resample`, `groupby` + `transform`, rolling windows, gap-aware forward-filling
- Structuring a pipeline into separate, single-responsibility stages instead of one monolithic script
- Statistical anomaly detection (rolling z-score) and the reasoning for why it's computed per-entity rather than globally
- Thinking about a feature pipeline from the perspective of "what would this feed into" (a predictive maintenance model), not just "clean the data"

## Possible extensions

- Replace the simulated maintenance feature with a real join against a maintenance-log table
- Persist output to a feature store instead of returning an in-memory DataFrame
- Swap the z-score anomaly rule for an isolation forest or a supervised failure-prediction model trained on labeled failure events
- Add unit tests for `_fill_short_gaps` edge cases (gap exactly at the 15-min boundary) and for multi-machine grouping correctness

## License

MIT
