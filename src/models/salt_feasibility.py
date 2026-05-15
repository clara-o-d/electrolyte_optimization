"""Filter candidate salts to those physically capable of a half-cycle at a given site."""

from __future__ import annotations

import math

from src.materials.salts import get_salt
from src.models.zsr_lcow_model import SiteClimate
from src.optimization.brine_equilibrium import equilibrate_salt_mf
from src.optimization.brine_uptake import WATER_MOLAR_MASS_G_MOL, water_sorption_factor


def feasible_salts_for_site(
    site: SiteClimate, candidates: tuple[str, ...] | list[str]
) -> tuple[str, ...]:
    """Return the subset of candidate salts that have a positive half-swing at this site.

    A salt passes if:
    - A valid equilibrium brine composition exists at both the high and low humidity.
    - The uptake coefficient is strictly higher at high humidity than at low humidity,
      meaning the composite can absorb water at high humidity and release it at low humidity.
    """
    temperature_c = site.temperature_c
    passing: list[str] = []
    for salt_name in candidates:
        rec = get_salt(salt_name)
        if salt_name not in ("NaCl", "LiCl", "CaCl2", "MgCl2"):
            continue
        brine_fraction_high = equilibrate_salt_mf(salt_name, site.humidity_high, temperature_c)
        brine_fraction_low = equilibrate_salt_mf(salt_name, site.humidity_low, temperature_c)
        if not (math.isfinite(brine_fraction_high) and math.isfinite(brine_fraction_low)):
            continue
        if (
            brine_fraction_high <= 0.0
            or brine_fraction_low <= 0.0
            or brine_fraction_high >= 1.0
            or brine_fraction_low >= 1.0
        ):
            continue
        sorption_high = water_sorption_factor(
            site.humidity_high, brine_fraction_high, rec.nu, rec.mw
        )
        sorption_low = water_sorption_factor(
            site.humidity_low, brine_fraction_low, rec.nu, rec.mw
        )
        if not (math.isfinite(sorption_high) and math.isfinite(sorption_low)):
            continue
        uptake_coeff_high = sorption_high * (WATER_MOLAR_MASS_G_MOL / rec.mw)
        uptake_coeff_low = sorption_low * (WATER_MOLAR_MASS_G_MOL / rec.mw)
        if not (
            math.isfinite(uptake_coeff_high)
            and math.isfinite(uptake_coeff_low)
            and uptake_coeff_high > uptake_coeff_low + 1e-15
        ):
            continue
        passing.append(salt_name)
    return tuple(passing)
