"""Live Quezon City weather with a deterministic offline fallback.

Only the Python standard library is used so the Raspberry Pi bundle does not
gain another dependency.  Open-Meteo requires no API key for this endpoint.
The caller runs ``get_weather`` on an action worker thread; tkinter's main
thread therefore remains responsive while the HTTPS request is in flight.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo


LOCATION = "QUEZON CITY, PH"
LATITUDE = 14.6488
LONGITUDE = 121.0509
TIMEZONE = "Asia/Manila"
_PHT = ZoneInfo(TIMEZONE)

_CURRENT_FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "is_day",
)

_QUERY = urllib.parse.urlencode({
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "current": ",".join(_CURRENT_FIELDS),
    "timezone": TIMEZONE,
    "forecast_days": 1,
})
API_URL = "https://api.open-meteo.com/v1/forecast?" + _QUERY


# WMO interpretation codes returned by Open-Meteo.
_CONDITIONS = {
    0: "CLEAR SKY",
    1: "MAINLY CLEAR",
    2: "PARTLY CLOUDY",
    3: "OVERCAST",
    45: "FOG",
    48: "RIME FOG",
    51: "LIGHT DRIZZLE",
    53: "DRIZZLE",
    55: "HEAVY DRIZZLE",
    56: "FREEZING DRIZZLE",
    57: "HEAVY FREEZING DRIZZLE",
    61: "LIGHT RAIN",
    63: "RAIN",
    65: "HEAVY RAIN",
    66: "FREEZING RAIN",
    67: "HEAVY FREEZING RAIN",
    71: "LIGHT SNOW",
    73: "SNOW",
    75: "HEAVY SNOW",
    77: "SNOW GRAINS",
    80: "LIGHT SHOWERS",
    81: "SHOWERS",
    82: "HEAVY SHOWERS",
    85: "SNOW SHOWERS",
    86: "HEAVY SNOW SHOWERS",
    95: "THUNDERSTORM",
    96: "THUNDERSTORM + HAIL",
    99: "SEVERE STORM + HAIL",
}


class WeatherError(RuntimeError):
    """A safe, user-displayable weather failure category."""


def _number(mapping, name):
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WeatherError("invalid %s" % name)
    value = float(value)
    if not math.isfinite(value):
        raise WeatherError("invalid %s" % name)
    return value


def _format_time(value):
    """Format an API-local or aware datetime as an unambiguous PHT time."""
    if isinstance(value, str):
        try:
            value = dt.datetime.fromisoformat(value)
        except ValueError as exc:
            raise WeatherError("invalid observation time") from exc
    if not isinstance(value, dt.datetime):
        raise WeatherError("invalid observation time")
    if value.tzinfo is None:
        value = value.replace(tzinfo=_PHT)
    else:
        value = value.astimezone(_PHT)
    return value.strftime("%d %b %Y  %I:%M %p PHT").upper()


def _compass(degrees):
    points = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
    return points[int((degrees % 360.0) / 22.5 + 0.5) % 16]


def parse_live_response(payload, retrieved_at=None):
    """Validate an Open-Meteo response and return renderer-friendly values."""
    if not isinstance(payload, dict):
        raise WeatherError("invalid API response")
    current = payload.get("current")
    if not isinstance(current, dict):
        raise WeatherError("current conditions missing")
    timezone = payload.get("timezone")
    if timezone != TIMEZONE:
        raise WeatherError("unexpected timezone")

    code = int(_number(current, "weather_code"))
    observed = _format_time(current.get("time"))
    retrieved_at = retrieved_at or dt.datetime.now(_PHT)
    retrieved = _format_time(retrieved_at)
    wind_direction = _number(current, "wind_direction_10m")
    interval = int(_number(current, "interval"))
    if interval <= 0 or interval > 3600:
        raise WeatherError("invalid observation interval")

    return {
        "status": "live",
        "live": True,
        "location": LOCATION,
        "condition": _CONDITIONS.get(code, "WEATHER CODE %d" % code),
        "weather_code": code,
        "temperature": _number(current, "temperature_2m"),
        "apparent_temperature": _number(current, "apparent_temperature"),
        "humidity": int(round(_number(current, "relative_humidity_2m"))),
        "precipitation": _number(current, "precipitation"),
        "cloud_cover": int(round(_number(current, "cloud_cover"))),
        "wind_speed": _number(current, "wind_speed_10m"),
        "wind_direction": _compass(wind_direction),
        "wind_degrees": int(round(wind_direction)) % 360,
        "is_day": bool(int(_number(current, "is_day"))),
        "observed_at": observed,
        "retrieved_at": retrieved,
        "interval_minutes": max(1, interval // 60),
        "source": "OPEN-METEO",
        "error": None,
    }


def fetch_live_weather(timeout=3.5, opener=None, now=None):
    """Fetch and validate current weather; raise WeatherError on any failure."""
    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "VCM-Weather/1.0 (+Raspberry-Pi)"})
    try:
        with opener(request, timeout=timeout) as response:
            raw = response.read(65537)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError):
            raise WeatherError("request timed out") from exc
        raise WeatherError("internet unavailable") from exc
    if len(raw) > 65536:
        raise WeatherError("API response too large")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WeatherError("invalid API response") from exc
    return parse_live_response(payload, retrieved_at=now)


def simulated_weather(reason=None, now=None):
    """Return believable but explicitly labelled non-live Quezon City data."""
    now = now or dt.datetime.now(_PHT)
    # Stable values derived from the hour make repeat renders consistent while
    # retaining the previous simulator's small amount of variation.
    choices = (
        ("PARTLY CLOUDY", 30.0, 34.0, 72, 0.0, 74, 11.0, "E"),
        ("OVERCAST", 29.0, 33.0, 77, 0.0, 92, 9.0, "ENE"),
        ("LIGHT RAIN", 27.0, 31.0, 84, 0.6, 96, 14.0, "SW"),
        ("CLEAR NIGHT", 26.0, 29.0, 79, 0.0, 18, 7.0, "NE"),
    )
    cond, temp, feels, humidity, rain, cloud, wind, direction = choices[now.hour % len(choices)]
    return {
        "status": "simulated",
        "live": False,
        "location": LOCATION,
        "condition": cond,
        "weather_code": None,
        "temperature": temp,
        "apparent_temperature": feels,
        "humidity": humidity,
        "precipitation": rain,
        "cloud_cover": cloud,
        "wind_speed": wind,
        "wind_direction": direction,
        "wind_degrees": None,
        "is_day": 6 <= now.hour < 18,
        "observed_at": "NOT AVAILABLE — SIMULATION",
        "retrieved_at": _format_time(now),
        "interval_minutes": None,
        "source": "LOCAL SIMULATOR",
        "error": str(reason or "internet unavailable").upper(),
    }


def get_weather(timeout=3.5):
    """Return live current weather, or simulation after any safe failure."""
    try:
        return fetch_live_weather(timeout=timeout)
    except Exception as exc:
        # This is intentionally broad at the fallback boundary: malformed API
        # changes must never terminate the action runner.
        return simulated_weather(exc)
