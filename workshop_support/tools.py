"""The agent's only tool: get_forecast(city, date).

By default it reads recorded data from forecast.json, so every run gives the
same result. With live=True it calls the Open-Meteo API.
"""

import datetime
import json
from pathlib import Path

import httpx

RECORDED_FORECASTS = Path(__file__).parent / "forecast.json"

# The tool description a real LLM sees (OpenAI function-calling format).
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_forecast",
        "description": "Get the daily forecast for a city and an ISO date within the next 16 days. "
                       "Ask for clarification if the city is ambiguous.",
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


def get_forecast(city, date, live=False):
    """Return one day's forecast. Errors are returned as data so the agent can explain them."""
    try:
        target = datetime.date.fromisoformat(date)
    except ValueError:
        return {"error": "date must be YYYY-MM-DD"}
    days_ahead = (target - datetime.date.today()).days
    if days_ahead < 0 or days_ahead > 15:
        return {"error": "date is outside the available 16-day forecast horizon"}

    if live:
        return fetch_open_meteo(city, date)

    recorded = json.loads(RECORDED_FORECASTS.read_text())
    key = city.strip().lower()
    if key not in recorded:
        return {"error": f"no recorded forecast for {city}"}
    forecast = recorded[key]
    return {
        "city": city,
        "date": date,
        "temperature_max_c": forecast["temperature_max_c"],
        "precipitation_mm": forecast["precipitation_mm"],
        "precipitation_probability_pct": forecast["precipitation_probability_pct"],
        "source": "recorded illustrative fixture",
    }


def fetch_open_meteo(city, date):
    """Look up the city, then its daily forecast, with the free Open-Meteo APIs."""
    try:
        with httpx.Client(timeout=10) as client:
            geo = client.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 5})
            geo.raise_for_status()
            matches = geo.json().get("results", [])
            if not matches:
                return {"error": f"location not found: {city}"}

            if "," not in city:
                countries = set()
                for match in matches:
                    if match["name"].lower() == city.lower():
                        countries.add(match.get("country_code"))
                if len(countries) > 1:
                    return {"error": f"ambiguous city: {city}; include the country"}

            location = matches[0]
            response = client.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "daily": "temperature_2m_max,precipitation_sum,precipitation_probability_max",
                "temperature_unit": "celsius",
                "timezone": location["timezone"],
                "forecast_days": 16,
            })
            response.raise_for_status()
            daily = response.json()["daily"]
            index = daily["time"].index(date)
            return {
                "city": f"{location['name']}, {location['country']}",
                "date": date,
                "temperature_max_c": daily["temperature_2m_max"][index],
                "precipitation_mm": daily["precipitation_sum"][index],
                "precipitation_probability_pct": daily["precipitation_probability_max"][index],
                "source": "Open-Meteo",
            }
    except (httpx.HTTPError, KeyError, ValueError) as error:
        return {"error": f"forecast unavailable: {type(error).__name__}"}
