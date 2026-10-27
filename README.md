# Probabilistic and Statistical Analysis of Weather Patterns

## Project introduction

This is a small Flask web application for exploring weather observations in a CSV file. It previews the selected dataset, calculates descriptive statistics and historical rainfall probabilities, and generates charts from the uploaded values. It is intended for an MCA Mathematics project demonstration.

## Objectives

- Provide a simple interface for managing a weather CSV dataset.
- Summarize temperature, rainfall, humidity, and rainfall occurrence.
- Group observations into monthly and yearly summaries where applicable.
- Visualize the uploaded observations without inventing values.
- Keep the implementation understandable and database-free.

## Features

- CSV upload, validation, replacement, reset, and a 10-row preview.
- Record count, start date, and end date.
- Temperature mean, median, extrema, range, population variance, and population standard deviation.
- Rainfall and humidity descriptive summaries.
- Monthly temperature, rainfall, humidity, and rainy-day probability summaries.
- Yearly rainy-day probabilities when multiple years are present.
- Six Matplotlib charts generated from the active dataset.
- Explicit **Historical data** and **Sample / demo data** labels.
- Per-browser-session, in-memory dataset workspaces; generated chart URLs are limited to the session that owns them.
- Helpful upload errors, a processing indicator, and responsive laptop/mobile layout.

## Technology stack

- Python and Flask
- HTML, CSS, and JavaScript
- Pandas and NumPy
- Matplotlib (non-interactive `Agg` backend)
- Python `unittest` for automated tests

## Installation

Use Python 3.9 or newer. In PowerShell, open the project folder and create an environment:

```powershell
cd "C:\Users\hp\Desktop\Weather-Pattern-Analayis"
python -m venv .venv
```

Install packages without activating the environment (this avoids PowerShell execution-policy issues):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The app generates a random Flask session secret at startup if `FLASK_SECRET_KEY` is not set. For a longer-running deployment, set `FLASK_SECRET_KEY` to a private, high-entropy value in the environment; do not commit it to the project.

## Execution

From the project folder:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000/>. Select a CSV, choose **Historical data** or **Sample / demo data**, and press **Upload dataset**. Use **Replace dataset** to refresh all results or **Reset dataset** to clear the active dashboard. Stop the server with `Ctrl+C`.

The Flask debugger is disabled by default. `FLASK_DEBUG=1` enables it for local development only.

## CSV format

The CSV requires these columns:

```csv
Date,Temperature,Rainfall,Humidity
2024-01-01,27,0,60
2024-01-02,28,2.5,62
```

Common aliases are accepted: `temp` for `Temperature`, `precipitation` for `Rainfall`, and `relative_humidity` for `Humidity`. Dates must be valid. Numeric cells may be blank; missing values are excluded from the corresponding calculation and chart. Rainfall must be nonnegative. Empty files, unsupported extensions, invalid numeric text, and files over 5 MB are rejected.

## Statistical formulas

For valid finite observations `x₁, …, xₙ` in the relevant column:

- Mean: `x̄ = Σxᵢ / n`
- Median: middle ordered observation (or the mean of the two middle values)
- Range: `maximum − minimum`
- Temperature population variance: `σ² = Σ(xᵢ − x̄)² / n`
- Temperature population standard deviation: `σ = √σ²`
- Rainfall total: `Σ rainfallᵢ`
- Monthly temperature and humidity means use that month’s valid values; monthly rainfall total sums its valid values.

Each measure has its own valid-observation count. Missing or non-finite numeric observations are excluded. A measure with no valid observations is shown as unavailable (`—`).

## Probability formulas

A rainy day has rainfall `> 0 mm`; a non-rainy day has rainfall `= 0 mm`. Blank rainfall values are excluded from both counts and the denominator.

- Valid rainfall observations: `n = rainy days + non-rainy days`
- `P(Rain) = rainy days / n × 100%`
- `P(No Rain) = non-rainy days / n × 100%`

Monthly and yearly probabilities use the same formulas within each calendar group. They are historical empirical frequencies for the selected data, not forecasts.

## Testing

Run the automated test suite from the project folder:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover homepage/assets, upload validation and size limits, empty and missing data, expected seven-row statistics and probabilities, monthly/yearly summaries, chart generation and delivery, dataset replacement, reset, session isolation, chart access control, filename safety, dashboard button wiring, and upload-content handling. The suite uses Flask’s test client and temporary chart directories; it does not require a browser.

## Known limitations

- Datasets are kept in process memory and expire after 24 hours of inactivity; restarting the app clears active datasets. Re-upload after a restart.
- This in-memory design is for a single-process local demonstration. Multiple production workers would need a shared server-side session/data store.
- The app does not forecast weather, connect to a live weather service, or persist uploads in a database.
- Browser layout responsiveness is implemented with CSS breakpoints; the automated suite checks the breakpoint rules but does not perform screenshot-based visual regression.
- The development server is not intended for public production deployment.
