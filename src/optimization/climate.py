"""Aggregate relative humidity for AWH (diurnal extrema, world-grid LCOW mean)."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.materials.salts import get_salt
from src.optimization.economics import (
    DRY_COMPOSITE_MASS_KG,
    LCOEconomicParams,
    annual_operating_plus_capital_usd,
    hydrogel_cost_usd_per_kg_composite,
    lcow_usd_per_kg_water,
)
from src.optimization.sorption import delta_U_half_swing, gross_annual_water_kg


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


def _cell_lcow(
    salt_name: str,
    sl: float,
    rh_high: float,
    rh_low: float,
    p: LCOEconomicParams,
) -> float:
    rec = get_salt(salt_name)
    c_h = hydrogel_cost_usd_per_kg_composite(rec.c_salt_usd_per_kg, sl, c_acrylamide=p.c_acrylamide_usd_per_kg, c_add=p.c_additives_usd_per_kg_composite)
    num = annual_operating_plus_capital_usd(
        p, c_hyd_usd_per_kg=c_h, dry_mass_kg=DRY_COMPOSITE_MASS_KG
    )
    du = delta_U_half_swing(sl, rh_high, rh_low, rec)
    y0 = gross_annual_water_kg(du, DRY_COMPOSITE_MASS_KG)
    return lcow_usd_per_kg_water(num, p.f_util, y0)


def mean_lcow_for_grid(
    salt_name: str,
    sl: float,
    site_rows: Iterable[dict],
    p: LCOEconomicParams | None = None,
) -> float:
    """Unweighted mean LCOW ($/kg water) over sites with precomputed ``rh_high_frac``, ``rh_low_frac``."""
    p = p or LCOEconomicParams()
    vals: list[float] = []
    for row in site_rows:
        lo = float(row["rh_low_frac"])
        hi = float(row["rh_high_frac"])
        c = _cell_lcow(salt_name, sl, hi, lo, p)
        if c < float("inf") and c == c:
            vals.append(c)
    if not vals:
        return float("inf")
    return float(sum(vals) / len(vals))
