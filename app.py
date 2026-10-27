"""Flask application for the weather patterns project dashboard."""

from io import BytesIO
import os
from pathlib import Path
import secrets
import time
from threading import RLock
from datetime import timedelta
from uuid import uuid4

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from flask import send_from_directory

WORKSPACE_TTL_SECONDS = 24 * 60 * 60
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,  # 5 MB, including multipart overhead
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(seconds=WORKSPACE_TTL_SECONDS),
)

# This simple in-memory store isolates visitors by signed Flask session. Entries
# expire after 24 hours and are lost when the demonstration server restarts.
workspaces = {}
workspace_lock = RLock()
CHART_DIRECTORY = Path(app.root_path) / "graphs"

REQUIRED_COLUMNS = ("Date", "Temperature", "Rainfall", "Humidity")
COLUMN_ALIASES = {
    "date": "Date",
    "temperature": "Temperature",
    "temp": "Temperature",
    "rainfall": "Rainfall",
    "precipitation": "Rainfall",
    "humidity": "Humidity",
    "relative_humidity": "Humidity",
}


def _remove_chart_files(chart_files):
    """Remove only chart filenames generated and tracked by this application."""
    for filename in (chart_files or {}).values():
        try:
            (CHART_DIRECTORY / filename).unlink(missing_ok=True)
        except OSError:
            # An active response may still hold the file open on Windows.
            pass


def get_workspace(create=False):
    """Return the active visitor's server-side workspace, if one exists."""
    now = time.monotonic()
    with workspace_lock:
        expired_ids = [
            workspace_id
            for workspace_id, workspace in workspaces.items()
            if now - workspace["last_access"] > WORKSPACE_TTL_SECONDS
        ]
        for workspace_id in expired_ids:
            expired = workspaces.pop(workspace_id)
            _remove_chart_files(expired.get("chart_files"))

        workspace_id = session.get("workspace_id")
        workspace = workspaces.get(workspace_id) if workspace_id else None
        if workspace is not None:
            workspace["last_access"] = now
            return workspace
        if workspace_id:
            session.pop("workspace_id", None)
        if not create:
            return None

        workspace_id = secrets.token_urlsafe(32)
        session["workspace_id"] = workspace_id
        session.permanent = True
        workspace = {"last_access": now, "weather_data": None, "weather_filename": None, "chart_files": None, "dataset_kind": None}
        workspaces[workspace_id] = workspace
        return workspace


def validate_weather_csv(file_storage):
    """Read and normalize an uploaded CSV, raising ValueError on bad data."""
    try:
        frame = pd.read_csv(BytesIO(file_storage.read()))
    except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ValueError("The file could not be read as a valid CSV. Check its contents and encoding.") from error

    if frame.empty:
        raise ValueError("The CSV contains no data rows. Add at least one weather record.")

    # Normalize common spellings and harmless whitespace/case differences.
    normalized = {}
    for original in frame.columns:
        key = str(original).strip().lower().replace(" ", "_").replace("-", "_")
        canonical = COLUMN_ALIASES.get(key)
        if canonical:
            if canonical in normalized.values():
                raise ValueError(f"Multiple columns map to {canonical}. Keep only one equivalent column.")
            normalized[original] = canonical
    frame = frame.rename(columns=normalized)

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing) + ".")

    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    if dates.isna().any():
        rows = ", ".join(str(index + 2) for index in frame.index[dates.isna()][:5])
        raise ValueError(f"Date values are missing or invalid (CSV row(s): {rows}).")
    frame["Date"] = dates.dt.strftime("%Y-%m-%d")

    for column in REQUIRED_COLUMNS[1:]:
        raw_values = frame[column]
        values = pd.to_numeric(raw_values, errors="coerce")
        missing = raw_values.isna() | raw_values.astype(str).str.strip().eq("")
        invalid = (values.isna() & ~missing) | values.map(lambda value: pd.notna(value) and abs(value) == float("inf"))
        if column == "Rainfall":
            invalid |= values < 0
        if invalid.any():
            rows = ", ".join(str(index + 2) for index in frame.index[invalid][:5])
            raise ValueError(f"{column} contains missing or invalid numeric values (CSV row(s): {rows}).")
        frame[column] = values

    return frame


def analyze_weather_data(frame):
    """Calculate descriptive summaries from valid observations in a dataset."""
    try:
        analysis_frame = frame.copy()
        analysis_frame["Date"] = pd.to_datetime(analysis_frame["Date"], errors="coerce")
        for column in REQUIRED_COLUMNS[1:]:
            analysis_frame[column] = pd.to_numeric(analysis_frame[column], errors="coerce")

        # Each measure uses its own non-missing, finite observations. Dates are
        # required for monthly grouping, while ungrouped statistics can still
        # use rows whose date is unavailable if called independently.
        valid = {}
        for column in REQUIRED_COLUMNS[1:]:
            values = analysis_frame[column].to_numpy(dtype=float)
            valid[column] = values[np.isfinite(values)]

        temperature = valid["Temperature"]
        rainfall = valid["Rainfall"]
        humidity = valid["Humidity"]

        monthly_frame = analysis_frame.loc[analysis_frame["Date"].notna()].copy()
        monthly_frame["Month"] = monthly_frame["Date"].dt.to_period("M")
        monthly = []
        monthly_record_count = 0
        for month, group in monthly_frame.groupby("Month", sort=True):
            temperature_month = group["Temperature"].to_numpy(dtype=float)
            rainfall_month = group["Rainfall"].to_numpy(dtype=float)
            humidity_month = group["Humidity"].to_numpy(dtype=float)
            temperature_month = temperature_month[np.isfinite(temperature_month)]
            rainfall_month = rainfall_month[np.isfinite(rainfall_month)]
            humidity_month = humidity_month[np.isfinite(humidity_month)]
            monthly_record_count += max(temperature_month.size, rainfall_month.size, humidity_month.size)
            monthly.append({
                "month": str(month),
                "temperature_count": int(temperature_month.size),
                "temperature_mean": float(np.mean(temperature_month)) if temperature_month.size else None,
                "rainfall_count": int(rainfall_month.size),
                "rainfall_total": float(np.sum(rainfall_month)) if rainfall_month.size else None,
                "humidity_count": int(humidity_month.size),
                "humidity_mean": float(np.mean(humidity_month)) if humidity_month.size else None,
            })

        return {
            "temperature": {
                "count": int(temperature.size),
                "mean": float(np.mean(temperature)) if temperature.size else None,
                "median": float(np.median(temperature)) if temperature.size else None,
                "minimum": float(np.min(temperature)) if temperature.size else None,
                "maximum": float(np.max(temperature)) if temperature.size else None,
                "range": float(np.ptp(temperature)) if temperature.size else None,
                "variance": float(np.var(temperature, ddof=0)) if temperature.size else None,
                "standard_deviation": float(np.std(temperature, ddof=0)) if temperature.size else None,
            },
            "rainfall": {
                "count": int(rainfall.size),
                "mean": float(np.mean(rainfall)) if rainfall.size else None,
                "median": float(np.median(rainfall)) if rainfall.size else None,
                "maximum": float(np.max(rainfall)) if rainfall.size else None,
                "total": float(np.sum(rainfall)) if rainfall.size else None,
            },
            "humidity": {
                "count": int(humidity.size),
                "mean": float(np.mean(humidity)) if humidity.size else None,
                "median": float(np.median(humidity)) if humidity.size else None,
                "minimum": float(np.min(humidity)) if humidity.size else None,
                "maximum": float(np.max(humidity)) if humidity.size else None,
            },
            "monthly": monthly,
            "monthly_record_count": monthly_record_count,
            "probability": calculate_rain_probabilities(analysis_frame),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, ValueError) and str(error).startswith("No valid "):
            raise
        raise ValueError("The uploaded dataset could not be analyzed. Check that its dates and numeric values are valid.") from error


def calculate_rain_probabilities(frame):
    """Return historical empirical rainy-day probabilities for valid rainfall rows."""
    data = frame.loc[:, ["Date", "Rainfall"]].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    rainfall = pd.to_numeric(data["Rainfall"], errors="coerce").to_numpy(dtype=float)
    data["Rainfall"] = rainfall
    valid = data.loc[data["Date"].notna() & np.isfinite(data["Rainfall"])].copy()

    def summarize(group, label):
        values = group["Rainfall"].to_numpy(dtype=float)
        rainy = int(np.count_nonzero(values > 0))
        non_rainy = int(np.count_nonzero(values == 0))
        total = int(values.size)
        return {
            label: str(group[label].iloc[0]) if total else "",
            "valid_count": total,
            "rainy_days": rainy,
            "non_rainy_days": non_rainy,
            "rain_probability": rainy / total * 100 if total else None,
            "no_rain_probability": non_rainy / total * 100 if total else None,
        }

    total = len(valid)
    rainy = int(np.count_nonzero(valid["Rainfall"].to_numpy(dtype=float) > 0))
    non_rainy = int(np.count_nonzero(valid["Rainfall"].to_numpy(dtype=float) == 0))
    valid["Month"] = valid["Date"].dt.to_period("M")
    monthly = [summarize(group, "Month") for _, group in valid.groupby("Month", sort=True)]
    valid["Year"] = valid["Date"].dt.year.astype(str)
    yearly = []
    if valid["Year"].nunique() > 1:
        yearly = [summarize(group, "Year") for _, group in valid.groupby("Year", sort=True)]

    return {
        "valid_count": total,
        "rainy_days": rainy,
        "non_rainy_days": non_rainy,
        "rain_probability": rainy / total * 100 if total else None,
        "no_rain_probability": non_rainy / total * 100 if total else None,
        "monthly": monthly,
        "yearly": yearly,
    }


def generate_weather_charts(frame):
    """Save six web-ready charts from the supplied data and return their names."""
    chart_id = uuid4().hex
    CHART_DIRECTORY.mkdir(parents=True, exist_ok=True)
    prepared = frame.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
    for column in REQUIRED_COLUMNS[1:]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared.loc[~np.isfinite(prepared[column]), column] = np.nan
    prepared = prepared.sort_values("Date")
    names = {
        key: f"{key}_{chart_id}.png"
        for key in (
            "temperature_over_time", "rainfall_over_time", "humidity_over_time",
            "rainy_days", "monthly_temperature", "monthly_rainfall",
        )
    }

    def save_chart(key, title, ylabel, plotter, color):
        fig, axis = plt.subplots(figsize=(7.2, 3.5), dpi=120)
        try:
            axis.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#20394d", pad=13)
            axis.set_ylabel(ylabel, fontsize=10, color="#4d6474")
            axis.tick_params(axis="both", labelsize=9, colors="#536a7a")
            axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#dce6eb")
            has_data = plotter(axis)
            if not has_data:
                axis.text(.5, .5, "No valid observations available", ha="center", va="center", transform=axis.transAxes, fontsize=11, color="#536a7a")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.grid(False)
            else:
                axis.tick_params(axis="x", labelrotation=30)
                for tick in axis.get_xticklabels():
                    tick.set_ha("right")
            fig.tight_layout(pad=1.5)
            fig.savefig(CHART_DIRECTORY / names[key], format="png", facecolor="white", bbox_inches="tight")
        finally:
            plt.close(fig)

    def time_series(column, kind, color, label):
        def draw(axis):
            values = prepared.loc[prepared["Date"].notna() & prepared[column].notna(), ["Date", column]]
            if values.empty:
                return False
            if kind == "line":
                axis.plot(values["Date"], values[column], color=color, marker="o", markersize=4, linewidth=2, label=label)
            else:
                axis.bar(values["Date"], values[column], color=color, width=0.7, label=label)
            axis.set_xlabel("Date", fontsize=10, color="#4d6474")
            return True
        return draw

    try:
        save_chart("temperature_over_time", "Temperature over time", "Temperature (°C)", time_series("Temperature", "line", "#4f9fc3", "Temperature"), "#4f9fc3")
        save_chart("rainfall_over_time", "Rainfall over time", "Rainfall (mm)", time_series("Rainfall", "bar", "#54b9aa", "Rainfall"), "#54b9aa")
        save_chart("humidity_over_time", "Humidity over time", "Humidity (%)", time_series("Humidity", "line", "#8a83c1", "Humidity"), "#8a83c1")

        def draw_rain_days(axis):
            counts = calculate_rain_probabilities(prepared)
            if not counts["valid_count"]:
                return False
            bars = axis.bar(["Rainy", "Non-rainy"], [counts["rainy_days"], counts["non_rainy_days"]], color=["#54b9aa", "#78a9cf"], width=.55)
            axis.bar_label(bars, padding=3, fontsize=10, color="#40596c")
            axis.set_ylabel("Valid days", fontsize=10, color="#4d6474")
            axis.set_xlabel("Rainfall observation type", fontsize=10, color="#4d6474")
            axis.set_ylim(0, max(counts["rainy_days"], counts["non_rainy_days"]) * 1.18 + .5)
            return True
        save_chart("rainy_days", "Rainy versus non-rainy days", "Number of days", draw_rain_days, "#54b9aa")

        def monthly_series(column, aggregation, title_axis, color):
            def draw(axis):
                rows = prepared.loc[prepared["Date"].notna() & prepared[column].notna(), ["Date", column]].copy()
                if rows.empty:
                    return False
                rows["Month"] = rows["Date"].dt.to_period("M")
                grouped = rows.groupby("Month")[column].agg(aggregation)
                bars = axis.bar([str(month) for month in grouped.index], grouped.to_numpy(), color=color, width=.65)
                axis.set_xlabel("Month", fontsize=10, color="#4d6474")
                if len(grouped) <= 1:
                    axis.set_xlim(-.7, .7)
                return bool(len(bars))
            return draw
        save_chart("monthly_temperature", "Monthly average temperature", "Average temperature (°C)", monthly_series("Temperature", "mean", "Month", "#4f9fc3"), "#4f9fc3")
        save_chart("monthly_rainfall", "Monthly total rainfall", "Total rainfall (mm)", monthly_series("Rainfall", "sum", "Month", "#54b9aa"), "#54b9aa")
    except Exception:
        for filename in names.values():
            (CHART_DIRECTORY / filename).unlink(missing_ok=True)
        raise
    return names


@app.route("/")
def index():
    """Render the dashboard and, when available, a small upload preview."""
    preview = []
    summary = None
    analysis = None
    workspace = get_workspace()
    weather_data = workspace.get("weather_data") if workspace else None
    if weather_data is not None:
        preview_frame = weather_data.head(10).copy()
        for column in REQUIRED_COLUMNS[1:]:
            preview_frame[column] = preview_frame[column].round(2)
        preview_frame = preview_frame.astype(object).where(pd.notna(preview_frame), "—")
        preview = preview_frame.to_dict(orient="records")
        summary = {
            "record_count": len(weather_data),
            "start_date": weather_data["Date"].min(),
            "end_date": weather_data["Date"].max(),
            "filename": workspace["weather_filename"],
            "dataset_kind": workspace["dataset_kind"],
        }
        try:
            analysis = analyze_weather_data(weather_data)
        except ValueError as error:
            flash(f"Analysis error: {error}", "error")
    return render_template("index.html", preview=preview, summary=summary, analysis=analysis, charts=workspace.get("chart_files") if workspace else None)


@app.route("/graphs/<path:filename>")
def generated_graph(filename):
    """Serve generated chart images from the application-owned graphs folder."""
    workspace = get_workspace()
    if not workspace or filename not in (workspace.get("chart_files") or {}).values():
        abort(404)
    return send_from_directory(CHART_DIRECTORY, filename)


@app.route("/upload", methods=["POST"])
def upload():
    """Validate and replace the in-memory weather dataset."""
    uploaded_file = request.files.get("weather_file")
    if not uploaded_file or not uploaded_file.filename:
        flash("Choose a CSV file before uploading.", "error")
        return redirect(url_for("index") + "#upload")
    if not uploaded_file.filename.lower().endswith(".csv"):
        flash("Unsupported file type. Please choose a .csv file.", "error")
        return redirect(url_for("index") + "#upload")

    try:
        frame = validate_weather_csv(uploaded_file)
    except ValueError as error:
        flash(str(error), "error")
        return redirect(url_for("index") + "#upload")

    selected_kind = request.form.get("dataset_kind", "historical")
    if selected_kind not in {"historical", "sample"}:
        flash("Choose whether this is historical weather data or sample/demo data.", "error")
        return redirect(url_for("index") + "#upload")

    try:
        analysis = analyze_weather_data(frame)
    except ValueError as error:
        flash(f"Analysis error: {error}", "error")
        return redirect(url_for("index") + "#upload")

    try:
        new_chart_files = generate_weather_charts(frame)
    except Exception as error:
        app.logger.exception("Chart generation failed")
        flash(f"The dataset was not replaced because charts could not be generated: {error}", "error")
        return redirect(url_for("index") + "#upload")

    workspace = get_workspace(create=True)
    previous_chart_files = workspace.get("chart_files")
    workspace["weather_data"] = frame
    # Use the name for display only; it is never used in a filesystem path.
    workspace["weather_filename"] = uploaded_file.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    workspace["dataset_kind"] = selected_kind
    # Unique chart names let active browser image requests finish safely on
    # Windows while the dashboard immediately switches to this upload's set.
    workspace["chart_files"] = new_chart_files
    _remove_chart_files(previous_chart_files)
    flash(f"Dataset uploaded and analyzed successfully ({analysis['temperature']['count']} valid temperature observations, {analysis['probability']['valid_count']} valid rainfall observations, and six charts generated).", "success")
    return redirect(url_for("index") + "#dataset-preview")


@app.route("/reset", methods=["POST"])
def reset_dataset():
    """Clear the active in-memory dataset and its generated chart links."""
    workspace_id = session.pop("workspace_id", None)
    with workspace_lock:
        workspace = workspaces.pop(workspace_id, None) if workspace_id else None
    if workspace:
        _remove_chart_files(workspace.get("chart_files"))
    flash("The active dataset and dashboard results have been cleared.", "success")
    return redirect(url_for("index") + "#upload")


@app.errorhandler(413)
def file_too_large(_error):
    flash("The file is too large. The maximum upload size is 5 MB.", "error")
    return redirect(url_for("index") + "#upload")


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
