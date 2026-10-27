"""Automated checks for the Weather Pattern Analysis demonstration app."""

from io import BytesIO
from pathlib import Path
import re
import tempfile
import unittest

import matplotlib.pyplot as plt
import pandas as pd

import app as weather_app


def csv_bytes(rows, columns=("Date", "Temperature", "Rainfall", "Humidity")):
    return pd.DataFrame(rows, columns=columns).to_csv(index=False).encode("utf-8")


def seven_row_sample():
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=7).strftime("%Y-%m-%d"),
        "Temperature": [27, 28, 29, 30, 30, 31, 32],
        "Rainfall": [0, 2.5, 1, 0, 4, 3, 0],
        "Humidity": [60, 62, 58, 65, 66, 64, 61],
    })


class WeatherDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_chart_directory = weather_app.CHART_DIRECTORY
        weather_app.CHART_DIRECTORY = Path(self.temp_dir.name)
        with weather_app.workspace_lock:
            weather_app.workspaces.clear()
        weather_app.app.config.update(TESTING=True)
        self.client = weather_app.app.test_client()

    def tearDown(self):
        with weather_app.workspace_lock:
            weather_app.workspaces.clear()
        weather_app.CHART_DIRECTORY = self.original_chart_directory
        self.temp_dir.cleanup()

    def upload(self, content, filename="weather.csv", client=None, dataset_kind="historical"):
        client = client or self.client
        return client.post(
            "/upload",
            data={
                "weather_file": (BytesIO(content), filename),
                "dataset_kind": dataset_kind,
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )

    def workspace_for(self, client=None):
        client = client or self.client
        with client.session_transaction() as browser_session:
            workspace_id = browser_session.get("workspace_id")
        return weather_app.workspaces.get(workspace_id)

    def test_homepage_and_assets_load(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Ready for upload", response.data)
        self.assertIn(b"Charts will appear here", response.data)
        with self.client.get("/static/script.js") as script:
            self.assertEqual(script.status_code, 200)
        with self.client.get("/static/style.css") as stylesheet:
            self.assertEqual(stylesheet.status_code, 200)

    def test_upload_valid_dataset_preview_and_expected_statistics(self):
        frame = seven_row_sample()
        response = self.upload(frame.to_csv(index=False).encode(), "sample.csv", dataset_kind="sample")
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("sample.csv", page)
        self.assertIn("SAMPLE / DEMO DATA", page)
        self.assertIn("7 records", page)
        self.assertIn("2024-01-01", page)
        self.assertIn("2024-01-07", page)
        self.assertIn("29.57", page)
        self.assertIn("2.53 °C²", page)
        self.assertIn("1.59 °C", page)
        self.assertIn("57.14%", page)
        self.assertIn("42.86%", page)
        result = weather_app.analyze_weather_data(frame)
        self.assertAlmostEqual(result["temperature"]["mean"], 29.57142857)
        self.assertAlmostEqual(result["temperature"]["variance"], 2.53061224)
        self.assertEqual(result["probability"]["rainy_days"], 4)
        self.assertEqual(result["probability"]["non_rainy_days"], 3)
        self.assertIn("27.0", page)
        self.assertIn("2.5", page)

    def test_alias_columns_are_accepted(self):
        payload = b"Date,temp,precipitation,relative_humidity\n2024-01-01,20,1,55\n"
        response = self.upload(payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Dataset preview", response.data)
        self.assertIn(b"20.00", response.data)

    def test_unsupported_file_type_is_rejected(self):
        response = self.upload(b"not a csv", "weather.txt")
        self.assertIn(b"Unsupported file type", response.data)
        self.assertIsNone(self.workspace_for())

    def test_missing_columns_are_reported(self):
        response = self.upload(b"Date,Temperature\n2024-01-01,20\n")
        self.assertIn(b"Missing required column(s): Rainfall, Humidity", response.data)

    def test_empty_dataset_is_reported(self):
        response = self.upload(b"Date,Temperature,Rainfall,Humidity\n")
        self.assertIn(b"no data rows", response.data)

    def test_file_size_limit_is_reported(self):
        payload = b"x" * (5 * 1024 * 1024 + 1024)
        response = self.upload(payload, "large.csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"maximum upload size is 5 MB", response.data)

    def test_malformed_numeric_value_is_reported(self):
        response = self.upload(b"Date,Temperature,Rainfall,Humidity\n2024-01-01,hot,0,50\n")
        self.assertIn(b"Temperature contains missing or invalid numeric values", response.data)

    def test_missing_values_are_excluded_consistently(self):
        payload = (
            b"Date,Temperature,Rainfall,Humidity\n"
            b"2024-01-01,27,0,60\n"
            b"2024-01-02,, ,\n"
            b"2024-01-03,29,2,\n"
        )
        response = self.upload(payload)
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("2 valid temperature observations", page)
        self.assertIn("2 valid rainfall observations", page)
        self.assertIn("50.00%", page)
        self.assertIn("—", page)
        workspace = self.workspace_for()
        analysis = weather_app.analyze_weather_data(workspace["weather_data"])
        self.assertEqual(analysis["temperature"]["count"], 2)
        self.assertEqual(analysis["humidity"]["count"], 1)
        self.assertEqual(analysis["probability"]["valid_count"], 2)

    def test_all_numeric_values_missing_render_without_crashing(self):
        payload = b"Date,Temperature,Rainfall,Humidity\n2024-01-01,,,\n"
        response = self.upload(payload)
        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("No valid rainfall observations", page)
        self.assertIn("No valid observations", page)
        self.assertEqual(len(re.findall(r'<img src="[^"]+"', page)), 6)

    def test_probability_formula_and_monthly_table(self):
        response = self.upload(seven_row_sample().to_csv(index=False).encode())
        page = response.get_data(as_text=True)
        self.assertIn("2024-01", page)
        self.assertIn("57.14%", page)
        self.assertIn("42.86%", page)
        probability = self.workspace_for()["weather_data"]
        result = weather_app.calculate_rain_probabilities(probability)
        self.assertEqual(result["valid_count"], 7)
        self.assertAlmostEqual(result["rain_probability"], 4 / 7 * 100)
        self.assertAlmostEqual(result["no_rain_probability"], 3 / 7 * 100)

    def test_yearly_probabilities_appear_for_multiple_years(self):
        frame = seven_row_sample()
        frame.loc[6, "Date"] = "2025-01-01"
        response = self.upload(frame.to_csv(index=False).encode())
        page = response.get_data(as_text=True)
        self.assertIn("Yearly rainy-day probabilities", page)
        self.assertIn("2024", page)
        self.assertIn("2025", page)

    def test_upload_replacement_refreshes_results_and_invalidates_old_chart_urls(self):
        first = seven_row_sample()
        self.upload(first.to_csv(index=False).encode(), "first.csv")
        first_charts = dict(self.workspace_for()["chart_files"])
        first_temperature_chart = (weather_app.CHART_DIRECTORY / first_charts["temperature_over_time"]).read_bytes()
        first_rainfall_chart = (weather_app.CHART_DIRECTORY / first_charts["rainfall_over_time"]).read_bytes()
        second = first.assign(Temperature=[10] * 7, Rainfall=[1] * 7)
        response = self.upload(second.to_csv(index=False).encode(), "second.csv")
        page = response.get_data(as_text=True)
        self.assertIn("second.csv", page)
        self.assertNotIn("first.csv", page)
        self.assertIn("10.00", page)
        self.assertIn("100.00%", page)
        second_charts = self.workspace_for()["chart_files"]
        self.assertNotEqual(first_charts, second_charts)
        self.assertNotEqual(first_temperature_chart, (weather_app.CHART_DIRECTORY / second_charts["temperature_over_time"]).read_bytes())
        self.assertNotEqual(first_rainfall_chart, (weather_app.CHART_DIRECTORY / second_charts["rainfall_over_time"]).read_bytes())
        for old_filename in first_charts.values():
            self.assertEqual(self.client.get(f"/graphs/{old_filename}").status_code, 404)
        for filename in second_charts.values():
            with self.client.get(f"/graphs/{filename}") as image:
                self.assertEqual(image.status_code, 200)
                self.assertTrue(image.data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_chart_generation_closes_figures_and_uses_six_outputs(self):
        response = self.upload(seven_row_sample().to_csv(index=False).encode())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.workspace_for()["chart_files"]), 6)
        self.assertEqual(plt.get_fignums(), [])
        for filename in self.workspace_for()["chart_files"].values():
            self.assertGreater((weather_app.CHART_DIRECTORY / filename).stat().st_size, 1000)

    def test_sessions_do_not_share_dataset_or_chart_access(self):
        client_a = weather_app.app.test_client()
        client_b = weather_app.app.test_client()
        self.upload(seven_row_sample().to_csv(index=False).encode(), "private.csv", client=client_a)
        page_b = client_b.get("/").get_data(as_text=True)
        self.assertIn("Ready for upload", page_b)
        self.assertNotIn("private.csv", page_b)
        cookie = client_a.get_cookie("session")
        self.assertIsNotNone(cookie)
        self.assertNotIn("private.csv", cookie.value)
        filename = next(iter(self.workspace_for(client_a)["chart_files"].values()))
        self.assertEqual(client_b.get(f"/graphs/{filename}").status_code, 404)
        with client_a.get(f"/graphs/{filename}") as image:
            self.assertEqual(image.status_code, 200)

    def test_reset_clears_only_current_session(self):
        client_a = weather_app.app.test_client()
        client_b = weather_app.app.test_client()
        self.upload(seven_row_sample().to_csv(index=False).encode(), "a.csv", client=client_a)
        self.upload(seven_row_sample().assign(Temperature=[5] * 7).to_csv(index=False).encode(), "b.csv", client=client_b)
        response = client_a.post("/reset", follow_redirects=True)
        self.assertIn(b"dashboard results have been cleared", response.data)
        self.assertIn(b"Ready for upload", client_a.get("/").data)
        self.assertIn(b"b.csv", client_b.get("/").data)

    def test_chart_urls_cannot_read_arbitrary_files(self):
        response = self.client.get("/graphs/../app.py")
        self.assertEqual(response.status_code, 404)

    def test_uploaded_name_is_never_used_as_a_filesystem_path(self):
        before = set(Path(weather_app.CHART_DIRECTORY).glob("*"))
        response = self.upload(seven_row_sample().to_csv(index=False).encode(), r"..\..\outside.csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("outside.csv", response.get_data(as_text=True))
        after = set(Path(weather_app.CHART_DIRECTORY).glob("*"))
        self.assertEqual(len(after - before), 6)

    def test_uploaded_csv_content_is_not_executed(self):
        payload = b'Date,Temperature,Rainfall,Humidity\n2024-01-01,"__import__(\'os\').system(\'whoami\')",0,50\n'
        response = self.upload(payload)
        self.assertIn(b"Temperature contains missing or invalid numeric values", response.data)
        self.assertIsNone(self.workspace_for())

    def test_debugger_is_off_by_default_and_secret_is_not_hardcoded_demo_value(self):
        self.assertFalse(weather_app.app.debug)
        self.assertNotEqual(weather_app.app.secret_key, "weather-pattern-analysis-local-demo")
        self.assertTrue(weather_app.app.config["SESSION_COOKIE_HTTPONLY"])
        self.assertEqual(weather_app.app.config["SESSION_COOKIE_SAMESITE"], "Lax")

    def test_responsive_layout_rules_are_present(self):
        css = (Path(weather_app.app.root_path) / "static" / "style.css").read_text(encoding="utf-8")
        self.assertIn("@media(max-width:700px)", css)
        self.assertIn(".generated-chart-grid{grid-template-columns:1fr}", css)

    def test_all_dashboard_buttons_have_an_action_or_script_handler(self):
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="menu-toggle"', page)
        self.assertIn('id="upload-button"', page)
        script = (Path(weather_app.app.root_path) / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn('menuToggle?.addEventListener("click"', script)
        self.assertIn('uploadForm?.addEventListener("submit"', script)
        with self.client.session_transaction() as browser_session:
            self.assertNotIn("weather_data", browser_session)


if __name__ == "__main__":
    unittest.main()
