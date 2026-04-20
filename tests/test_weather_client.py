"""Tests for the Open-Meteo weather client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.weather.client import WeatherClient, _raise_for_openmeteo_error
from src.data.weather.geocoding import GeoLocation, geocode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_HOURLY_RESPONSE = {
    "latitude": 33.45,
    "longitude": -112.07,
    "timezone": "America/Phoenix",
    "hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
        "temperature_2m": [12.3, 11.8],
        "relative_humidity_2m": [55.0, 57.0],
        "shortwave_radiation": [0.0, 0.0],
        "direct_normal_irradiance": [0.0, 0.0],
        "diffuse_radiation": [0.0, 0.0],
        "terrestrial_radiation": [0.0, 0.0],
        "wind_speed_10m": [3.1, 2.9],
        "precipitation": [0.0, 0.0],
        "surface_pressure": [950.0, 950.2],
        "et0_fao_evapotranspiration": [0.01, 0.01],
    },
}

MOCK_GEOCODING_RESPONSE = {
    "results": [
        {
            "name": "Phoenix",
            "latitude": 33.4484,
            "longitude": -112.074,
            "country": "United States",
            "admin1": "Arizona",
        }
    ]
}


# ---------------------------------------------------------------------------
# WeatherClient tests
# ---------------------------------------------------------------------------


class TestWeatherClient:
    @patch("src.data.weather.client.requests.Session.get")
    def test_get_historical_returns_dataframe(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_HOURLY_RESPONSE
        mock_get.return_value = mock_resp

        client = WeatherClient()
        df = client.get_historical(33.45, -112.07, "2024-01-01", "2024-01-01")

        assert isinstance(df, pd.DataFrame)
        assert "temperature_2m" in df.columns
        assert "relative_humidity_2m" in df.columns
        assert "shortwave_radiation" in df.columns
        assert len(df) == 2

    @patch("src.data.weather.client.requests.Session.get")
    def test_get_forecast_returns_dataframe(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_HOURLY_RESPONSE
        mock_get.return_value = mock_resp

        client = WeatherClient()
        df = client.get_forecast(33.45, -112.07, days=1)

        assert isinstance(df, pd.DataFrame)
        assert "temperature_2m" in df.columns

    def test_get_forecast_invalid_days_raises(self):
        client = WeatherClient()
        with pytest.raises(ValueError, match="days must be between"):
            client.get_forecast(0, 0, days=20)

    @patch("src.data.weather.client.requests.Session.get")
    def test_api_error_raises_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"reason": "Invalid date range"}
        mock_get.return_value = mock_resp

        client = WeatherClient()
        with pytest.raises(Exception, match="Open-Meteo API error 400"):
            client.get_historical(0, 0, "2024-01-01", "2024-01-01")

    @patch("src.data.weather.geocoding.requests.get")
    @patch("src.data.weather.client.requests.Session.get")
    def test_get_historical_by_name(self, mock_weather_get, mock_geo_get):
        geo_resp = MagicMock()
        geo_resp.status_code = 200
        geo_resp.json.return_value = MOCK_GEOCODING_RESPONSE
        mock_geo_get.return_value = geo_resp

        weather_resp = MagicMock()
        weather_resp.status_code = 200
        weather_resp.json.return_value = MOCK_HOURLY_RESPONSE
        mock_weather_get.return_value = weather_resp

        client = WeatherClient()
        df = client.get_historical_by_name("Phoenix, AZ", "2024-01-01", "2024-01-01")

        assert "location_name" in df.columns
        assert "Phoenix" in df["location_name"].iloc[0]

    def test_variable_aliases_resolved(self):
        resolved = WeatherClient._resolve_variables(["rh", "temp", "ghi"])
        assert "relative_humidity_2m" in resolved
        assert "temperature_2m" in resolved
        assert "shortwave_radiation" in resolved


# ---------------------------------------------------------------------------
# Geocoding tests
# ---------------------------------------------------------------------------


class TestGeocoding:
    @patch("src.data.weather.geocoding.requests.get")
    def test_geocode_returns_geolocation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_GEOCODING_RESPONSE
        mock_get.return_value = mock_resp

        results = geocode("Phoenix, AZ")
        assert len(results) == 1
        loc = results[0]
        assert isinstance(loc, GeoLocation)
        assert loc.name == "Phoenix"
        assert abs(loc.latitude - 33.4484) < 0.001

    @patch("src.data.weather.geocoding.requests.get")
    def test_geocode_no_results_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        with pytest.raises(ValueError, match="No geocoding results"):
            geocode("zzz_nonexistent_place_zzz")

    def test_geolocation_str(self):
        loc = GeoLocation("Riyadh", 24.69, 46.72, "Saudi Arabia", admin1="Riyadh Region")
        assert "Riyadh" in str(loc)
        assert "Saudi Arabia" in str(loc)
