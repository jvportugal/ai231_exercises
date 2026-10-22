"""Deterministic tests for live weather parsing and offline fallback."""
import datetime as dt
import json
import unittest
import urllib.error
from zoneinfo import ZoneInfo

from vcm_action_runner.actions import weather


SAMPLE = {
    "latitude": 14.657293,
    "longitude": 121.0316,
    "timezone": "Asia/Manila",
    "current": {
        "time": "2026-10-23T01:15",
        "interval": 900,
        "temperature_2m": 29.1,
        "relative_humidity_2m": 71,
        "apparent_temperature": 32.4,
        "precipitation": 0.0,
        "weather_code": 3,
        "cloud_cover": 96,
        "wind_speed_10m": 15.8,
        "wind_direction_10m": 269,
        "is_day": 0,
    },
}


class _Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, _limit):
        return json.dumps(self.value).encode("utf-8")


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.now = dt.datetime(
            2026, 10, 23, 1, 17, tzinfo=ZoneInfo("Asia/Manila"))

    def test_live_response_has_api_and_retrieval_times(self):
        result = weather.fetch_live_weather(
            opener=lambda _request, timeout: _Response(SAMPLE), now=self.now)
        self.assertTrue(result["live"])
        self.assertEqual(result["condition"], "OVERCAST")
        self.assertEqual(result["wind_direction"], "W")
        self.assertEqual(result["interval_minutes"], 15)
        self.assertEqual(result["observed_at"], "23 OCT 2026  01:15 AM PHT")
        self.assertEqual(result["retrieved_at"], "23 OCT 2026  01:17 AM PHT")

    def test_invalid_timezone_is_rejected(self):
        payload = dict(SAMPLE)
        payload["timezone"] = "UTC"
        with self.assertRaises(weather.WeatherError):
            weather.parse_live_response(payload, retrieved_at=self.now)

    def test_network_failure_is_safe(self):
        def fail(_request, timeout):
            raise urllib.error.URLError("offline")

        with self.assertRaisesRegex(weather.WeatherError, "internet unavailable"):
            weather.fetch_live_weather(opener=fail, now=self.now)

    def test_simulation_is_unambiguously_labelled(self):
        result = weather.simulated_weather("internet unavailable", now=self.now)
        self.assertFalse(result["live"])
        self.assertEqual(result["source"], "LOCAL SIMULATOR")
        self.assertIn("SIMULATION", result["observed_at"])
        self.assertEqual(result["location"], "QUEZON CITY, PH")


if __name__ == "__main__":
    unittest.main()
