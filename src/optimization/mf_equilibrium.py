"""Equilibrium salt mass fraction mf(RH) from the same relations as ``calculate_mf_*.m`` (Python port)."""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.optimize import brentq

from src.materials.salts import get_salt

# Brackets from MATLAB `robust_fzero` common usage
_BRACKET_LO = 0.01
_BRACKET_HI = 0.75


def robust_fzero_brentq(
    f: Callable[[float], float],
    x_min: float = _BRACKET_LO,
    x_max: float = _BRACKET_HI,
) -> float:
    """Root of ``f`` on (x_min, x_max) using Brent; same spirit as ``robust_fzero.m``."""
    fa, fb = f(x_min), f(x_max)
    if not (fa * fb < 0) or (math.isnan(fa) or math.isnan(fb)):
        return float("nan")
    return float(brentq(f, x_min, x_max, maxiter=200))


def mf_NaCl(rh: float) -> float:
    """``calculate_mf_NaCl`` (25°C). Valid ~0.762 -- 0.9934."""
    if not (0.0 < rh < 1.0):
        return float("nan")
    A_4 = 5.863
    A_3 = -5.545
    A_2 = -0.332
    A_1 = -0.5597
    A_0 = 0.9998

    def f(xi: float) -> float:
        return (
            rh
            - A_0
            - A_1 * xi
            - A_2 * xi**2
            - A_3 * xi**3
            - A_4 * xi**4
        )

    return robust_fzero_brentq(f, 0.0116, 0.2596)


def _mf_Li_Ca_style(
    rh: float,
    t_c: float,
    p0: float,
    p1: float,
    p2: float,
    p3: float,
    p4: float,
    p5: float,
    p6: float,
    p7: float,
    p8: float,
    p9: float,
) -> float:
    """Shared template for LiCl / CaCl2 isotherm (dilute crystal paper form)."""
    if not (0.0 < rh < 1.0) or t_c > 100.0:
        return float("nan")
    theta = (t_c + 273.15) / 647.0

    def f(xi: float) -> float:
        t1 = 1.0 - (1.0 + (xi / p6) ** p7) ** p8 - p9 * math.exp(-((xi - 0.1) ** 2) / 0.005)
        t2 = 2.0 - (1.0 + (xi / p0) ** p1) ** p2 + ((1.0 + (xi / p3) ** p4) ** p5 - 1.0) * theta
        return rh - t1 * t2

    return robust_fzero_brentq(f, _BRACKET_LO, _BRACKET_HI)


def mf_LiCl(rh: float, t_c: float = 25.0) -> float:
    """``calculate_mf_LiCl`` (IH thermal sci form)."""
    p_0, p_1, p_2, p_3, p_4, p_5 = 0.28, 4.3, 0.60, 0.21, 5.10, 0.49
    p_6, p_7, p_8, p_9 = 0.362, -4.75, -0.40, 0.03
    return _mf_Li_Ca_style(rh, t_c, p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9)


def mf_CaCl2(rh: float, t_c: float = 25.0) -> float:
    """``calculate_mf_CaCl`` / CaCl2 (IH thermal sci form, q_* parameters in MATLAB)."""
    q_0, q_1, q_2, q_3, q_4, q_5 = 0.31, 3.698, 0.60, 0.231, 4.584, 0.49
    q_6, q_7, q_8, q_9 = 0.478, -5.20, -0.40, 0.018
    return _mf_Li_Ca_style(rh, t_c, q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8, q_9)


def mf_MgCl2(rh: float) -> float:
    """``calculate_mf_MgCl`` / MgCl2 polynomial form."""
    if not (0.0 < rh < 1.0):
        return float("nan")
    A_4 = 186.32487108
    A_3 = -153.67496570
    A_2 = 38.21982328
    A_1 = -4.86704441
    A_0 = 1.16231287

    def f(x: float) -> float:
        return rh - A_0 - A_1 * x - A_2 * x**2 - A_3 * x**3 - A_4 * x**4

    return robust_fzero_brentq(f, 0.01, 0.75)


_salt_mf: dict[str, Callable[[float, float], float]] = {
    "LiCl": lambda rh, T: mf_LiCl(rh, T),
    "NaCl": lambda rh, T: mf_NaCl(rh),
    "CaCl2": lambda rh, T: mf_CaCl2(rh, T),
    "MgCl2": lambda rh, T: mf_MgCl2(rh),
}


def equilibrate_salt_mf(name: str, rh: float, t_c: float = 25.0) -> float:
    """Return mf, or ``nan`` if out of the salt's RH range or not implemented."""
    rec = get_salt(name)
    if rec.name not in _salt_mf:
        return float("nan")
    if not (rec.rh_min <= rh <= rec.rh_max):
        return float("nan")
    return float(_salt_mf[rec.name](rh, t_c))
