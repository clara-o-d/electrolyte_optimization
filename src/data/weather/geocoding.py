"""Geocoding helper using the Open-Meteo geocoding API."""

from __future__ import annotations

from dataclasses import dataclass

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


@dataclass
class GeoLocation:
    name: str
    latitude: float
    longitude: float
    country: str
    admin1: str | None = None  # state / province

    def __str__(self) -> str:
        parts = [self.name]
        if self.admin1:
            parts.append(self.admin1)
        parts.append(self.country)
        return ", ".join(parts)


def geocode(location_name: str, count: int = 1) -> list[GeoLocation]:
    """Resolve a place name to geographic coordinates.

    Uses the Open-Meteo geocoding API (no API key required).

    Parameters
    ----------
    location_name:
        Human-readable place name, e.g. ``"Phoenix, AZ"`` or ``"Riyadh"``.
    count:
        Number of candidate results to return. The first result is the
        best match.

    Returns
    -------
    list[GeoLocation]
        Matching locations, ordered by relevance. Raises ``ValueError`` if
        no results are found.

    Examples
    --------
    >>> from src.data.weather.geocoding import geocode
    >>> loc = geocode("Phoenix, AZ")[0]
    >>> print(loc.latitude, loc.longitude)
    33.4484 -112.074
    """
    params = {"name": location_name, "count": count, "language": "en", "format": "json"}
    response = requests.get(GEOCODING_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = data.get("results")
    if not results:
        raise ValueError(
            f"No geocoding results found for '{location_name}'. "
            "Try a more specific name or use coordinates directly."
        )

    return [
        GeoLocation(
            name=r["name"],
            latitude=r["latitude"],
            longitude=r["longitude"],
            country=r.get("country", ""),
            admin1=r.get("admin1"),
        )
        for r in results
    ]
