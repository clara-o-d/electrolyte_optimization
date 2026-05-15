"""Water uptake physics: mole fraction, sorption factor, and desorption equilibrium."""

from __future__ import annotations

import math

WATER_MOLAR_MASS_G_MOL: float = 18.015  # g/mol


def _water_mole_fraction(
    brine_salt_fraction: float,
    ions_per_formula: int | float,
    salt_formula_weight_g_per_mol: float,
) -> float:
    """Water mole fraction in a salt-water brine (colligative particle basis).

    Counts each dissolved ion separately: moles of water / (moles water + ions * moles salt).
    """
    if brine_salt_fraction <= 0.0 or brine_salt_fraction >= 1.0 or not math.isfinite(brine_salt_fraction):
        return float("nan")
    moles_water = (1.0 - brine_salt_fraction) / WATER_MOLAR_MASS_G_MOL
    moles_salt = brine_salt_fraction / float(salt_formula_weight_g_per_mol)
    denominator = moles_water + float(ions_per_formula) * moles_salt
    if denominator <= 0.0:
        return float("nan")
    return moles_water / denominator


def water_sorption_factor(
    relative_humidity: float,
    brine_salt_fraction: float,
    ions_per_formula: int | float,
    salt_formula_weight_g_per_mol: float,
) -> float:
    """Dimensionless sorption factor: (x_w * ions_per_formula) / (1 - x_w).

    At equilibrium the water activity equals relative humidity, which collapses the
    activity coefficient and leaves the mole-fraction grouping above.
    """
    x_w = _water_mole_fraction(brine_salt_fraction, ions_per_formula, salt_formula_weight_g_per_mol)
    if not math.isfinite(x_w) or x_w <= 0.0 or x_w >= 1.0:
        return float("nan")
    return (x_w * float(ions_per_formula)) / (1.0 - x_w)


def saturation_vapor_pressure_pa(temperature_c: float) -> float:
    """Saturation vapor pressure of liquid water (Pa) using the Tetens (Magnus) formula.

    Accurate to roughly ±0.3% from 0 to 100 C, which is sufficient for the 25-70 C
    range used in the SAWH desorption equilibrium.
    """
    return 1000.0 * 0.61078 * math.exp(17.27 * temperature_c / (temperature_c + 237.3))


def desorption_water_activity(
    condenser_temperature_c: float,
    gel_temperature_c: float,
) -> float:
    """Effective water activity in the gel at desorption equilibrium with a sealed condenser.

    During desorption the hydrogel is cut off from the ambient air. Sunlight heats it
    to ``gel_temperature_c`` (commonly modeled as ``T_amb + DEFAULT_GEL_TEMPERATURE_RISE_C``)
    while a condenser sits at ``condenser_temperature_c`` (taken to be the local
    ambient temperature). Equating molar vapor concentration (P / RT) above the gel
    and the condenser gives:

        x_w * gamma_w * P_sat(T_gel) / T_gel = P_sat(T_cond) / T_cond

    so the brine water activity satisfies

        a_w = x_w * gamma_w = P_sat(T_cond) * T_gel / (P_sat(T_gel) * T_cond)

    with absolute (Kelvin) temperatures. The salt brine in the gel reaches the
    composition that produces this water activity at ``gel_temperature_c``.
    """
    p_sat_cond = saturation_vapor_pressure_pa(condenser_temperature_c)
    p_sat_gel = saturation_vapor_pressure_pa(gel_temperature_c)
    if p_sat_gel <= 0.0 or not math.isfinite(p_sat_gel) or not math.isfinite(p_sat_cond):
        return float("nan")
    t_cond_k = condenser_temperature_c + 273.15
    t_gel_k = gel_temperature_c + 273.15
    if t_cond_k <= 0.0 or t_gel_k <= 0.0:
        return float("nan")
    return p_sat_cond * t_gel_k / (p_sat_gel * t_cond_k)
