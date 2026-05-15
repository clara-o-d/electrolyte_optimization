"""Water uptake physics: mole fraction and sorption factor for a salt brine."""

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
