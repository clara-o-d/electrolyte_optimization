"""Equilibrium brine salt mass fraction as a function of relative humidity.

Each function solves for the mass fraction of salt in a saturated aqueous solution
that is in equilibrium with air at the given relative humidity (and temperature for
temperature-dependent salts). Implemented as Python ports of the MATLAB
``calculate_mf_*.m`` isotherm fits.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.optimize import brentq

from src.materials.salts import get_salt

_BRACKET_LO = 0.01
_BRACKET_HI = 0.75


def _find_root(
    f: Callable[[float], float],
    x_min: float = _BRACKET_LO,
    x_max: float = _BRACKET_HI,
) -> float:
    """Find a root of f on (x_min, x_max) using Brent's method; returns nan if no bracket."""
    fa, fb = f(x_min), f(x_max)
    if not (fa * fb < 0) or (math.isnan(fa) or math.isnan(fb)):
        return float("nan")
    return float(brentq(f, x_min, x_max, maxiter=200))


def mf_NaCl(relative_humidity: float) -> float:
    """Equilibrium brine salt fraction for NaCl at 25°C. Valid for RH ≈ 0.762–0.993."""
    if not (0.0 < relative_humidity < 1.0):
        return float("nan")
    A_4, A_3, A_2, A_1, A_0 = 5.863, -5.545, -0.332, -0.5597, 0.9998

    def residual(salt_fraction: float) -> float:
        return (
            relative_humidity
            - A_0 - A_1 * salt_fraction - A_2 * salt_fraction**2
            - A_3 * salt_fraction**3 - A_4 * salt_fraction**4
        )

    return _find_root(residual, 0.0116, 0.2596)


def _mf_LiCl_CaCl2_style(
    relative_humidity: float,
    temperature_c: float,
    p0: float, p1: float, p2: float,
    p3: float, p4: float, p5: float,
    p6: float, p7: float, p8: float, p9: float,
) -> float:
    """Shared isotherm template for LiCl and CaCl2 (thermal science paper form)."""
    if not (0.0 < relative_humidity < 1.0) or temperature_c > 100.0:
        return float("nan")
    reduced_temperature = (temperature_c + 273.15) / 647.0

    def residual(salt_fraction: float) -> float:
        concentration_term = (
            1.0
            - (1.0 + (salt_fraction / p6) ** p7) ** p8
            - p9 * math.exp(-((salt_fraction - 0.1) ** 2) / 0.005)
        )
        temperature_term = (
            2.0
            - (1.0 + (salt_fraction / p0) ** p1) ** p2
            + ((1.0 + (salt_fraction / p3) ** p4) ** p5 - 1.0) * reduced_temperature
        )
        return relative_humidity - concentration_term * temperature_term

    return _find_root(residual, _BRACKET_LO, _BRACKET_HI)


def mf_LiCl(relative_humidity: float, temperature_c: float = 25.0) -> float:
    """Equilibrium brine salt fraction for LiCl."""
    return _mf_LiCl_CaCl2_style(
        relative_humidity, temperature_c,
        0.28, 4.3, 0.60, 0.21, 5.10, 0.49, 0.362, -4.75, -0.40, 0.03,
    )


def mf_CaCl2(relative_humidity: float, temperature_c: float = 25.0) -> float:
    """Equilibrium brine salt fraction for CaCl2."""
    return _mf_LiCl_CaCl2_style(
        relative_humidity, temperature_c,
        0.31, 3.698, 0.60, 0.231, 4.584, 0.49, 0.478, -5.20, -0.40, 0.018,
    )


def mf_MgCl2(relative_humidity: float) -> float:
    """Equilibrium brine salt fraction for MgCl2 (polynomial fit)."""
    if not (0.0 < relative_humidity < 1.0):
        return float("nan")
    A_4, A_3, A_2, A_1, A_0 = 186.32487108, -153.67496570, 38.21982328, -4.86704441, 1.16231287

    def residual(salt_fraction: float) -> float:
        return (
            relative_humidity
            - A_0 - A_1 * salt_fraction - A_2 * salt_fraction**2
            - A_3 * salt_fraction**3 - A_4 * salt_fraction**4
        )

    return _find_root(residual, 0.01, 0.75)


_isotherm_by_salt: dict[str, Callable[[float, float], float]] = {
    "LiCl": lambda rh, t: mf_LiCl(rh, t),
    "NaCl": lambda rh, t: mf_NaCl(rh),
    "CaCl2": lambda rh, t: mf_CaCl2(rh, t),
    "MgCl2": lambda rh, t: mf_MgCl2(rh),
}


def equilibrate_salt_mf(
    salt_name: str, relative_humidity: float, temperature_c: float = 25.0
) -> float:
    """Return equilibrium brine salt mass fraction, or nan if outside the salt's RH range."""
    rec = get_salt(salt_name)
    if rec.name not in _isotherm_by_salt:
        return float("nan")
    if not (rec.rh_min <= relative_humidity <= rec.rh_max):
        return float("nan")
    return float(_isotherm_by_salt[rec.name](relative_humidity, temperature_c))


def salt_fraction_to_molality(brine_salt_fraction: float, formula_weight_g_per_mol: float) -> float:
    """Convert salt mass fraction to binary brine molality (mol / kg water)."""
    if not (0.0 < brine_salt_fraction < 1.0) or not math.isfinite(brine_salt_fraction):
        return float("nan")
    return 1000.0 * brine_salt_fraction / (formula_weight_g_per_mol * (1.0 - brine_salt_fraction))


def molality_to_salt_fraction(molality: float, formula_weight_g_per_mol: float) -> float:
    """Convert binary brine molality (mol / kg water) to salt mass fraction."""
    if molality < 0.0 or not math.isfinite(molality):
        return float("nan")
    return (molality * formula_weight_g_per_mol) / (1000.0 + molality * formula_weight_g_per_mol)


def binary_molality_at_rh(
    salt_name: str, relative_humidity: float, temperature_c: float
) -> float:
    """Molality of a pure salt-water brine in equilibrium at the given relative humidity."""
    brine_fraction = equilibrate_salt_mf(salt_name, relative_humidity, temperature_c)
    if not (math.isfinite(brine_fraction) and 0.0 < brine_fraction < 1.0):
        return float("nan")
    rec = get_salt(salt_name)
    return salt_fraction_to_molality(brine_fraction, rec.mw)
