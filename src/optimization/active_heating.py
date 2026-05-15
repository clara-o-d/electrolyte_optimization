"""Polynomial surrogate of desorption reference molality versus gel temperature.

The desorption brine state at a given gel temperature ``T_gel`` depends on a
pure-salt isotherm root-find (see :func:`src.optimization.brine_equilibrium.binary_molality_at_rh`),
which cannot be embedded directly in the Pyomo NLP. Instead we precompute the
root-find on a grid of ``T_gel`` values and fit a smooth low-degree polynomial
that the NLP can differentiate. The fit captures the implicit
``a_w = desorption_water_activity(T_amb, T_gel)`` coupling along the sealed
gel/condenser equilibrium curve, so the surrogate only needs ``T_gel`` as input
(``T_amb`` is fixed per site).

The polynomial coefficients are returned in Horner order (highest-degree first)
so they can be evaluated via :func:`pyomo_poly_expr` in the model.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.optimization.brine_equilibrium import binary_molality_at_rh
from src.optimization.brine_uptake import desorption_water_activity


def default_t_gel_grid_c(
    t_amb_c: float,
    *,
    t_max_c: float = 120.0,
    n: int = 40,
) -> NDArray[np.float64]:
    """Default ``T_gel`` grid (deg C) for surrogate fits.

    Starts just above ambient (since the gel cannot cool below ambient with
    only solar + electric heating) and extends to ``t_max_c``. ``n`` evenly
    spaced points; the default of 40 over a 30-90 K span gives the salts with
    narrow deliquescence ranges (CaCl2, MgCl2) enough finite samples to fit a
    higher-degree polynomial without ill-conditioning.
    """
    if not math.isfinite(t_amb_c) or not math.isfinite(t_max_c) or t_max_c <= t_amb_c:
        raise ValueError("t_max_c must be finite and strictly above t_amb_c.")
    if n < 4:
        raise ValueError("n must be >= 4 for a cubic surrogate fit.")
    return np.linspace(t_amb_c + 0.5, t_max_c, n, dtype=np.float64)


def fit_desorption_molality_polynomial(
    salt_name: str,
    t_amb_c: float,
    t_gel_grid_c: NDArray[np.float64],
    *,
    degree: int = 5,
    min_degree: int = 2,
) -> NDArray[np.float64] | None:
    """Fit a polynomial surrogate ``m_ref(T_gel)`` for one salt at fixed ``T_amb``.

    For each ``T_gel`` on the grid, evaluates the sealed gel/condenser
    desorption water activity ``a_w = desorption_water_activity(t_amb_c, T_gel)``
    and the corresponding pure-salt molality
    ``binary_molality_at_rh(salt_name, a_w, T_gel)``. Infeasible points (where
    ``a_w`` falls outside the salt's deliquescence range and the root-find
    returns NaN) are dropped before fitting.

    The fit automatically steps the polynomial degree down from ``degree`` to
    ``min_degree`` if there are too few finite samples for the highest degree
    (we always require at least ``degree + 2`` points to keep the fit well-
    conditioned). Returns ``None`` if even ``min_degree`` cannot be fit, in
    which case the salt is effectively infeasible at this site even with
    maximum active heating. Otherwise returns coefficients in Horner order
    (highest power first) for evaluation via :func:`pyomo_poly_expr`.
    """
    if degree < 1 or min_degree < 1 or min_degree > degree:
        raise ValueError("degree/min_degree must satisfy 1 <= min_degree <= degree.")
    t_grid = np.asarray(t_gel_grid_c, dtype=np.float64)
    molalities = np.empty_like(t_grid)
    for i, t_gel in enumerate(t_grid):
        a_w = desorption_water_activity(t_amb_c, float(t_gel))
        if not math.isfinite(a_w) or a_w <= 0.0 or a_w >= 1.0:
            molalities[i] = np.nan
            continue
        molalities[i] = binary_molality_at_rh(salt_name, a_w, float(t_gel))
    mask = np.isfinite(molalities) & (molalities >= 0.0)
    n_finite = int(np.sum(mask))
    for d in range(degree, min_degree - 1, -1):
        if n_finite >= d + 2:
            coeffs = np.polyfit(t_grid[mask], molalities[mask], d)
            return np.asarray(coeffs, dtype=np.float64)
    return None


def evaluate_poly(coeffs: NDArray[np.float64], t_gel_c: float) -> float:
    """Scalar Horner evaluation of a polynomial surrogate (highest power first)."""
    acc = 0.0
    for c in coeffs:
        acc = acc * t_gel_c + float(c)
    return float(acc)


def pyomo_poly_expr(coeffs: NDArray[np.float64], t_gel_var):  # noqa: ANN001 - pyomo expr
    """Horner-form Pyomo expression of a polynomial surrogate (highest power first).

    Accepts a Pyomo ``Var`` (or any expression that supports ``+`` / ``*``) and
    returns a Pyomo expression evaluating the polynomial at it. Using Horner's
    rule keeps the expression tree shallow and avoids redundant powers.
    """
    if coeffs.size == 0:
        raise ValueError("coeffs must be non-empty.")
    acc = float(coeffs[0])
    for c in coeffs[1:]:
        acc = acc * t_gel_var + float(c)
    return acc
