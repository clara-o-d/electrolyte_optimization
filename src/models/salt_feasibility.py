"""Filter candidate salts to those physically capable of a half-cycle at a given site."""

from __future__ import annotations

import math

from src.materials.salts import get_salt
from src.models.zsr_lcow_model import SiteClimate
from src.optimization.brine_equilibrium import equilibrate_salt_mf
from src.optimization.brine_uptake import (
    WATER_MOLAR_MASS_G_MOL,
    desorption_water_activity,
    water_sorption_factor,
)


def feasible_salts_for_site(
    site: SiteClimate, candidates: tuple[str, ...] | list[str]
) -> tuple[str, ...]:
    """Return the subset of candidate salts that have a positive half-swing at this site.

    A salt passes if:
    - A valid equilibrium brine composition exists at the night-time uptake state
      (humidity_high, ambient temperature) and at the sun-driven desorption state
      (effective water activity at gel_temperature_c).
    - The uptake coefficient at the night-time uptake state is strictly higher than
      at the desorption state, meaning the composite can absorb water at night and
      release it during the day.
    """
    ambient_temperature_c = site.temperature_c
    gel_temperature_c = site.gel_temperature_c
    desorption_a_w = desorption_water_activity(ambient_temperature_c, gel_temperature_c)
    if not math.isfinite(desorption_a_w) or desorption_a_w <= 0.0 or desorption_a_w >= 1.0:
        return tuple()
    passing: list[str] = []
    for salt_name in candidates:
        rec = get_salt(salt_name)
        if salt_name not in ("NaCl", "LiCl", "CaCl2", "MgCl2"):
            continue
        brine_fraction_uptake = equilibrate_salt_mf(
            salt_name, site.humidity_high, ambient_temperature_c
        )
        brine_fraction_desorption = equilibrate_salt_mf(
            salt_name, desorption_a_w, gel_temperature_c
        )
        if not (
            math.isfinite(brine_fraction_uptake) and math.isfinite(brine_fraction_desorption)
        ):
            continue
        if (
            brine_fraction_uptake <= 0.0
            or brine_fraction_desorption <= 0.0
            or brine_fraction_uptake >= 1.0
            or brine_fraction_desorption >= 1.0
        ):
            continue
        sorption_uptake = water_sorption_factor(
            site.humidity_high, brine_fraction_uptake, rec.nu, rec.mw
        )
        sorption_desorption = water_sorption_factor(
            desorption_a_w, brine_fraction_desorption, rec.nu, rec.mw
        )
        if not (math.isfinite(sorption_uptake) and math.isfinite(sorption_desorption)):
            continue
        uptake_coeff_uptake = sorption_uptake * (WATER_MOLAR_MASS_G_MOL / rec.mw)
        uptake_coeff_desorption = sorption_desorption * (WATER_MOLAR_MASS_G_MOL / rec.mw)
        if not (
            math.isfinite(uptake_coeff_uptake)
            and math.isfinite(uptake_coeff_desorption)
            and uptake_coeff_uptake > uptake_coeff_desorption + 1e-15
        ):
            continue
        passing.append(salt_name)
    return tuple(passing)
