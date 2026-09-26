"""The workshop's single weather tool. Live calls use Open-Meteo; evals use a fixture."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
TOOL_NAME = "get_forecast"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Get the daily forecast for a city and an ISO date within the next 16 days. Ask for clarification if the city is ambiguous.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City with country when known, e.g. Berlin, Germany"},
                "date": {"type": "string", "description": "Forecast date in YYYY-MM-DD format"},
            },
            "required": ["city", "date"],
            "additionalProperties": False,
        },
    },
}


def get_forecast(city: str, date: str, *, fixture: bool = False) -> dict:
    """Return one daily forecast. Tool errors are data so the agent can explain them."""
    try:
        target = __import__("datetime").date.fromisoformat(date)
    except ValueError:
        return {"error": "date must be YYYY-MM-DD"}
    days_ahead = (target - __import__("datetime").date.today()).days
    if days_ahead < 0 or days_ahead > 15:
        return {"error": "date is outside the available 16-day forecast horizon"}
    city = city.strip()
    if not city:
        return {"error": "city is required"}
    if fixture:
        cases = json.loads((ROOT / "fixtures" / "forecast.json").read_text())
        record = cases.get(city.casefold())
        if record is None:
            return {"error": f"no recorded forecast for {city}"}
        return {"city": city, "date": date, "source": "recorded illustrative fixture", **record}
    try:
        with httpx.Client(timeout=10) as client:
            geo = client.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 5})
            geo.raise_for_status()
            matches = geo.json().get("results", [])
            if not matches:
                return {"error": f"location not found: {city}"}
            if "," not in city and len({m.get("country_code") for m in matches if m.get("name", "").casefold() == city.casefold()}) > 1:
                return {"error": f"ambiguous city: {city}; include the country"}
            location = matches[0]
            response = client.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": location["latitude"], "longitude": location["longitude"],
                "daily": "temperature_2m_max,precipitation_sum,precipitation_probability_max",
                "temperature_unit": "celsius", "timezone": location["timezone"], "forecast_days": 16,
            })
            response.raise_for_status()
            daily = response.json()["daily"]
            index = daily["time"].index(date)
            return {
                "city": f"{location['name']}, {location['country']}", "date": date,
                "temperature_max_c": daily["temperature_2m_max"][index],
                "precipitation_mm": daily["precipitation_sum"][index],
                "precipitation_probability_pct": daily["precipitation_probability_max"][index],
                "source": "Open-Meteo",
            }
    except (httpx.HTTPError, KeyError, ValueError, IndexError) as exc:
        return {"error": f"forecast unavailable: {type(exc).__name__}"}
