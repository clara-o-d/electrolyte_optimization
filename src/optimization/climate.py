"""Aggregate hourly weather into the diurnal extrema needed by the SAWH model."""

from __future__ import annotations

import pandas as pd


def site_row_from_hourly(df: pd.DataFrame) -> dict[str, float]:
    """Return mean daily diurnal extrema for the SAWH site model.

    Keys returned:
        ``rh_high_frac``                mean of daily max relative humidity (fraction 0-1)
        ``rh_low_frac``                 mean of daily min relative humidity (fraction 0-1)
        ``temperature_high_c``          mean of daily max ambient temperature (deg C);
                                        used as the condenser/ambient temperature in
                                        the sun-driven desorption equilibrium
        ``temperature_low_c``           mean of daily min ambient temperature (deg C),
                                        diagnostic only
        ``solar_irradiance_w_per_m2``   mean of daily max shortwave (GHI) irradiance
                                        (W/m^2); drives the gel temperature in the
                                        steady-state heat-transfer model
    """
    rh_high, rh_low = diurnal_rh_from_hourly(df)
    out: dict[str, float] = {"rh_high_frac": rh_high, "rh_low_frac": rh_low}
    if "temperature_2m" in df.columns:
        t_high, t_low = diurnal_temperature_from_hourly(df)
        out["temperature_high_c"] = t_high
        out["temperature_low_c"] = t_low
    if "shortwave_radiation" in df.columns:
        out["solar_irradiance_w_per_m2"] = mean_daily_max_irradiance_from_hourly(df)
    return out


def diurnal_rh_from_hourly(df: pd.DataFrame) -> tuple[float, float]:
    """Mean of daily max RH and mean of daily min RH (fractions 0-1).

    Expects ``relative_humidity_2m`` in % (Open-Meteo) and a datetime index.
    """
    s = "relative_humidity_2m"
    if s not in df.columns:
        raise KeyError(f"DataFrame must contain column {s!r}")
    r = df[s].resample("D")
    daily_max = r.max()
    daily_min = r.min()
    return (float(daily_max.mean() / 100.0), float(daily_min.mean() / 100.0))


def diurnal_temperature_from_hourly(df: pd.DataFrame) -> tuple[float, float]:
    """Mean of daily max temperature and mean of daily min temperature (deg C).

    Expects ``temperature_2m`` in deg C (Open-Meteo) and a datetime index.
    """
    s = "temperature_2m"
    if s not in df.columns:
        raise KeyError(f"DataFrame must contain column {s!r}")
    r = df[s].resample("D")
    daily_max = r.max()
    daily_min = r.min()
    return (float(daily_max.mean()), float(daily_min.mean()))


def mean_daily_max_irradiance_from_hourly(df: pd.DataFrame) -> float:
    """Mean of daily peak shortwave (GHI) irradiance in W/m^2.

    Expects ``shortwave_radiation`` in W/m^2 (Open-Meteo) and a datetime index.
    """
    s = "shortwave_radiation"
    if s not in df.columns:
        raise KeyError(f"DataFrame must contain column {s!r}")
    return float(df[s].resample("D").max().mean())
