"""Open-Meteo weather client for global AWH-relevant climate data.

Fetches hourly time-series of relative humidity, temperature, and solar
irradiance for any location on Earth.  Supports both historical reanalysis
(ERA5 archive, back to 1940) and short-range forecasts (16 days).

No API key is required for either endpoint.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from .geocoding import GeoLocation, geocode

# ---------------------------------------------------------------------------
# Endpoint constants
# ---------------------------------------------------------------------------

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# ---------------------------------------------------------------------------
# Variables available from the API that are relevant to AWH modelling
# ---------------------------------------------------------------------------

#: Default set of hourly variables fetched when none are specified.
DEFAULT_VARIABLES: tuple[str, ...] = (
    "temperature_2m",           # °C  — dry-bulb air temperature
    "relative_humidity_2m",     # %   — relative humidity
    "shortwave_radiation",       # W/m² — global horizontal irradiance (GHI)
    "direct_normal_irradiance",  # W/m² — DNI (beam radiation)
    "diffuse_radiation",         # W/m² — diffuse horizontal irradiance
    "terrestrial_radiation",     # W/m² — top-of-atmosphere (clear-sky ref)
    "wind_speed_10m",            # m/s
    "precipitation",             # mm/h
    "surface_pressure",          # hPa
    "et0_fao_evapotranspiration",  # mm/h
)

#: Mapping of friendly aliases → Open-Meteo variable names.
VARIABLE_ALIASES: dict[str, str] = {
    "rh": "relative_humidity_2m",
    "temp": "temperature_2m",
    "ghi": "shortwave_radiation",
    "dni": "direct_normal_irradiance",
    "dhi": "diffuse_radiation",
    "wind": "wind_speed_10m",
    "precip": "precipitation",
    "pressure": "surface_pressure",
}

# ---------------------------------------------------------------------------
# Main client class
# ---------------------------------------------------------------------------


class WeatherClient:
    """Client for retrieving AWH-relevant weather data via Open-Meteo.

    Parameters
    ----------
    cache_dir:
        Optional directory where HTTP responses are cached to avoid
        redundant network requests.  If ``None`` caching is disabled.
    session_timeout:
        Seconds to wait for a server response before raising
        ``requests.exceptions.Timeout``.

    Examples
    --------
    >>> client = WeatherClient()
    >>> df = client.get_historical(latitude=33.45, longitude=-112.07,
    ...                            start="2023-01-01", end="2023-12-31")
    >>> df.head()
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        session_timeout: int = 30,
    ) -> None:
        self._timeout = session_timeout
        self._session = self._build_session(cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_historical(
        self,
        latitude: float,
        longitude: float,
        start: str | date,
        end: str | date,
        variables: tuple[str, ...] | list[str] | None = None,
        timezone: str = "auto",
    ) -> pd.DataFrame:
        """Fetch hourly historical weather data from the ERA5 archive.

        Data is available globally from 1940-01-01 to ~5 days before today.

        Parameters
        ----------
        latitude:
            Decimal degrees north (−90 to 90).
        longitude:
            Decimal degrees east (−180 to 180).
        start:
            First day of the requested period (inclusive), as ``"YYYY-MM-DD"``
            or a :class:`datetime.date` object.
        end:
            Last day of the requested period (inclusive).
        variables:
            Open-Meteo variable names or friendly aliases (see
            :data:`VARIABLE_ALIASES`). Defaults to :data:`DEFAULT_VARIABLES`.
        timezone:
            IANA timezone string (e.g. ``"America/Phoenix"``). ``"auto"``
            resolves to the timezone of the requested coordinates.

        Returns
        -------
        pd.DataFrame
            Hourly time-series indexed by a timezone-aware
            :class:`pandas.DatetimeIndex`.  Each requested variable is a
            column, plus metadata columns ``latitude``, ``longitude``.
        """
        params = self._build_params(
            latitude=latitude,
            longitude=longitude,
            start=start,
            end=end,
            variables=variables,
            timezone=timezone,
        )
        return self._fetch(_ARCHIVE_URL, params, latitude, longitude)

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 7,
        variables: tuple[str, ...] | list[str] | None = None,
        timezone: str = "auto",
    ) -> pd.DataFrame:
        """Fetch hourly forecast weather data (up to 16 days ahead).

        Parameters
        ----------
        latitude:
            Decimal degrees north (−90 to 90).
        longitude:
            Decimal degrees east (−180 to 180).
        days:
            Number of forecast days (1–16).
        variables:
            Variable names or aliases. Defaults to :data:`DEFAULT_VARIABLES`.
        timezone:
            IANA timezone string or ``"auto"``.

        Returns
        -------
        pd.DataFrame
            Same structure as :meth:`get_historical`.
        """
        if not 1 <= days <= 16:
            raise ValueError(f"days must be between 1 and 16, got {days}")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": self._resolve_variables(variables),
            "forecast_days": days,
            "timezone": timezone,
        }
        return self._fetch(_FORECAST_URL, params, latitude, longitude)

    def get_historical_by_name(
        self,
        location_name: str,
        start: str | date,
        end: str | date,
        variables: tuple[str, ...] | list[str] | None = None,
        timezone: str = "auto",
    ) -> pd.DataFrame:
        """Like :meth:`get_historical` but accepts a place name.

        Resolves the name to coordinates using the Open-Meteo geocoding API.

        Parameters
        ----------
        location_name:
            Human-readable place name, e.g. ``"Atacama Desert"`` or
            ``"Dubai, UAE"``.
        start, end:
            Date range (see :meth:`get_historical`).
        variables:
            Variable names or aliases.
        timezone:
            IANA timezone string or ``"auto"``.

        Returns
        -------
        pd.DataFrame
            Hourly time-series with an additional ``location_name`` column.
        """
        loc: GeoLocation = geocode(location_name)[0]
        df = self.get_historical(
            latitude=loc.latitude,
            longitude=loc.longitude,
            start=start,
            end=end,
            variables=variables,
            timezone=timezone,
        )
        df["location_name"] = str(loc)
        return df

    def get_forecast_by_name(
        self,
        location_name: str,
        days: int = 7,
        variables: tuple[str, ...] | list[str] | None = None,
        timezone: str = "auto",
    ) -> pd.DataFrame:
        """Like :meth:`get_forecast` but accepts a place name."""
        loc: GeoLocation = geocode(location_name)[0]
        df = self.get_forecast(
            latitude=loc.latitude,
            longitude=loc.longitude,
            days=days,
            variables=variables,
            timezone=timezone,
        )
        df["location_name"] = str(loc)
        return df

    def get_climate_summary(
        self,
        latitude: float,
        longitude: float,
        start: str | date,
        end: str | date,
        freq: Literal["D", "W", "ME", "YE"] = "ME",
    ) -> pd.DataFrame:
        """Return aggregated climate statistics over a date range.

        Computes mean, min, and max of each variable resampled at the
        requested frequency.  Useful for quickly characterising a site's
        AWH potential without wrangling hourly data.

        Parameters
        ----------
        latitude, longitude:
            Site coordinates.
        start, end:
            Date range.
        freq:
            Resample frequency:  ``"D"`` (daily), ``"W"`` (weekly),
            ``"ME"`` (month-end), or ``"YE"`` (year-end).

        Returns
        -------
        pd.DataFrame
            Multi-level columns ``(stat, variable)`` where *stat* is one of
            ``mean``, ``min``, ``max``.
        """
        df = self.get_historical(latitude=latitude, longitude=longitude, start=start, end=end)
        numeric = df.select_dtypes("number").drop(columns=["latitude", "longitude"], errors="ignore")
        agg = numeric.resample(freq).agg(["mean", "min", "max"])
        return agg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session(self, cache_dir: str | Path | None) -> requests.Session:
        """Return a (possibly cached) requests Session."""
        if cache_dir is not None:
            try:
                import requests_cache  # type: ignore[import]

                session = requests_cache.CachedSession(
                    cache_name=str(Path(cache_dir) / "openmeteo_cache"),
                    backend="sqlite",
                    expire_after=timedelta(hours=6),
                )
                return session
            except ImportError:
                warnings.warn(
                    "requests-cache is not installed; caching is disabled. "
                    "Install it with: pip install requests-cache",
                    stacklevel=3,
                )
        return requests.Session()

    @staticmethod
    def _resolve_variables(
        variables: tuple[str, ...] | list[str] | None,
    ) -> list[str]:
        """Resolve user-supplied variable names (or aliases) to API names."""
        if variables is None:
            return list(DEFAULT_VARIABLES)
        resolved = []
        for v in variables:
            resolved.append(VARIABLE_ALIASES.get(v, v))
        return resolved

    @staticmethod
    def _build_params(
        latitude: float,
        longitude: float,
        start: str | date,
        end: str | date,
        variables: tuple[str, ...] | list[str] | None,
        timezone: str,
    ) -> dict:
        start_str = str(start) if isinstance(start, date) else start
        end_str = str(end) if isinstance(end, date) else end
        return {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": WeatherClient._resolve_variables(variables),
            "start_date": start_str,
            "end_date": end_str,
            "timezone": timezone,
        }

    def _fetch(
        self,
        url: str,
        params: dict,
        latitude: float,
        longitude: float,
    ) -> pd.DataFrame:
        """Execute the API request and parse the response into a DataFrame."""
        # Open-Meteo expects comma-joined variable lists
        params = dict(params)
        params["hourly"] = ",".join(params["hourly"])

        response = self._session.get(url, params=params, timeout=self._timeout)
        _raise_for_openmeteo_error(response)
        data = response.json()

        hourly = dict(data.get("hourly", {}))
        if not hourly:
            raise ValueError("API returned no hourly data. Check your date range and coordinates.")

        times = pd.to_datetime(hourly.pop("time"))
        df = pd.DataFrame(hourly, index=times)
        df.index.name = "time"

        # Attach timezone info if present in response
        tz = data.get("timezone")
        if tz and tz != "UTC":
            try:
                df.index = df.index.tz_localize(tz)
            except Exception:
                pass  # leave as naive if localisation fails

        df["latitude"] = data.get("latitude", latitude)
        df["longitude"] = data.get("longitude", longitude)
        return df


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _raise_for_openmeteo_error(response: requests.Response) -> None:
    """Raise a descriptive error for Open-Meteo API failures."""
    if response.status_code == 200:
        return
    try:
        detail = response.json().get("reason", response.text)
    except Exception:
        detail = response.text
    raise requests.HTTPError(
        f"Open-Meteo API error {response.status_code}: {detail}",
        response=response,
    )
