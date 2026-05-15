"""Aggregate relative humidity for SAWH (diurnal extrema from hourly weather data)."""

from __future__ import annotations

import pandas as pd


def site_row_from_hourly(df: pd.DataFrame) -> dict[str, float]:
    """Return ``{"rh_high_frac", "rh_low_frac"}`` for one location's hourly frame."""
    hi, lo = diurnal_rh_from_hourly(df)
    return {"rh_high_frac": hi, "rh_low_frac": lo}


def diurnal_rh_from_hourly(df: pd.DataFrame) -> tuple[float, float]:
    """Mean of daily max RH and mean of daily min RH (fractions 0–1).

    Expects ``relative_humidity_2m`` in % (Open-Meteo) and a datetime index.
    """
    s = "relative_humidity_2m"
    if s not in df.columns:
        raise KeyError(f"DataFrame must contain column {s!r}")
    r = df[s].resample("D")
    daily_max = r.max()
    daily_min = r.min()
    return (float(daily_max.mean() / 100.0), float(daily_min.mean() / 100.0))
