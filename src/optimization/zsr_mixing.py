"""Zdanovskii isopiestic (ZSR) mixing rule for salt brines.

At fixed water activity (= relative humidity in this model), the molality of each salt
in the mixture is:

    m_i_blend = blend_weight_i * m_i_reference

where m_i_reference is the molality of a pure salt-i brine in equilibrium at that humidity.
The blend weights must sum to 1 (simplex constraint).

Optimization runs via a Pyomo NLP (Ipopt) when available; falls back to SciPy SLSQP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pyomo.opt import SolverFactory

from src.materials.salts import get_salt
from src.models.zsr_lcow_model import SiteClimate, HalfSwingCoefficients
from src.optimization.economics import C_DEVICE_USD, DRY_COMPOSITE_MASS_KG, LCOEconomicParams
from src.optimization.brine_equilibrium import binary_molality_at_rh, equilibrate_salt_mf
from src.optimization.brine_uptake import WATER_MOLAR_MASS_G_MOL, water_sorption_factor


def ipopt_available() -> bool:
    """Return True if the Ipopt executable is available on PATH."""
    return SolverFactory("ipopt", validate=False).available(False)


def _normalize_blend_weights(
    weights: NDArray[np.float64], salt_names: tuple[str, ...]
) -> NDArray[np.float64]:
    total = float(np.sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Blend weights must be non-negative and not all zero.")
    normalized = np.clip(weights / total, 0.0, 1.0)
    if normalized.shape[0] != len(salt_names):
        raise ValueError("Number of blend weights must match number of salt names.")
    return normalized


def zsr_brine_state(
    blend_weights: NDArray[np.float64],
    salt_names: tuple[str, ...],
    relative_humidity: float,
    temperature_c: float,
) -> tuple[float, float, float]:
    """Compute mixed brine state from ZSR rule at a given humidity.

    Returns (brine_salt_fraction, effective_ions_per_formula, effective_formula_weight_g_per_mol),
    or (nan, nan, nan) if the brine state is unphysical.
    """
    w = _normalize_blend_weights(blend_weights, salt_names)
    reference_molality = np.empty(len(salt_names), dtype=np.float64)
    formula_weight = np.empty(len(salt_names), dtype=np.float64)
    ions_per_formula = np.empty(len(salt_names), dtype=np.float64)

    for i, name in enumerate(salt_names):
        m_ref = binary_molality_at_rh(name, relative_humidity, temperature_c)
        reference_molality[i] = m_ref if w[i] > 1e-15 else 0.0
        rec = get_salt(name)
        formula_weight[i] = rec.mw
        ions_per_formula[i] = float(rec.nu)

    blended_molality = np.where(w > 1e-15, w * reference_molality, 0.0)
    total_molality = float(np.sum(blended_molality))
    if total_molality <= 0.0 or not math.isfinite(total_molality):
        return float("nan"), float("nan"), float("nan")

    dissolved_salt_g_per_kg_water = float(np.sum(blended_molality * formula_weight))
    brine_salt_fraction = dissolved_salt_g_per_kg_water / (dissolved_salt_g_per_kg_water + 1000.0)
    effective_ions = float(np.sum(blended_molality * ions_per_formula) / total_molality)
    effective_formula_weight = dissolved_salt_g_per_kg_water / total_molality

    if not (
        math.isfinite(brine_salt_fraction) and 0.0 < brine_salt_fraction < 1.0
        and math.isfinite(effective_ions) and math.isfinite(effective_formula_weight)
    ):
        return float("nan"), float("nan"), float("nan")
    return float(brine_salt_fraction), float(effective_ions), float(effective_formula_weight)


def uptake_B_zsr(
    blend_weights: NDArray[np.float64],
    salt_names: tuple[str, ...],
    site: SiteClimate,
) -> HalfSwingCoefficients | None:
    """Compute uptake coefficients for a ZSR brine blend at both humidity setpoints.

    Returns None if the brine state is unphysical at either setpoint.
    """
    temperature_c = site.temperature_c
    fraction_high, ions_high, weight_high = zsr_brine_state(
        blend_weights, salt_names, site.humidity_high, temperature_c
    )
    fraction_low, ions_low, weight_low = zsr_brine_state(
        blend_weights, salt_names, site.humidity_low, temperature_c
    )
    if not all(map(math.isfinite, (fraction_high, fraction_low, weight_high, weight_low, ions_high, ions_low))):
        return None
    sorption_high = water_sorption_factor(site.humidity_high, fraction_high, ions_high, weight_high)
    sorption_low = water_sorption_factor(site.humidity_low, fraction_low, ions_low, weight_low)
    if not (math.isfinite(sorption_high) and math.isfinite(sorption_low)):
        return None
    uptake_high = sorption_high * (WATER_MOLAR_MASS_G_MOL / weight_high)
    uptake_low = sorption_low * (WATER_MOLAR_MASS_G_MOL / weight_low)
    if not (uptake_high > uptake_low + 1e-15 and math.isfinite(uptake_high) and math.isfinite(uptake_low)):
        return None
    return HalfSwingCoefficients(at_high_humidity=uptake_high, at_low_humidity=uptake_low)


def _blend_salt_price(
    blend_weights: NDArray[np.float64],
    salt_names: tuple[str, ...],
) -> float:
    w = _normalize_blend_weights(blend_weights, salt_names)
    return float(sum(w[i] * get_salt(name).c_salt_usd_per_kg for i, name in enumerate(salt_names)))


def lcow_zsr_at_sl(
    blend_weights: NDArray[np.float64],
    salt_names: tuple[str, ...],
    site: SiteClimate,
    econ: LCOEconomicParams,
    salt_to_polymer_ratio: float,
    *,
    cycles_per_year: int = 365,
) -> float:
    """Scalar LCOW (USD/kg water) for a ZSR brine blend at a fixed salt-to-polymer ratio."""
    if salt_to_polymer_ratio <= 0.0 or not math.isfinite(salt_to_polymer_ratio):
        return 1e30
    half_swing = uptake_B_zsr(blend_weights, salt_names, site)
    if half_swing is None or half_swing.at_high_humidity <= half_swing.at_low_humidity + 1e-15:
        return 1e30
    blend_price = _blend_salt_price(blend_weights, salt_names)
    sl = salt_to_polymer_ratio
    salt_fraction_in_composite = sl / (1.0 + sl)
    uptake_swing = 0.5 * salt_fraction_in_composite * (
        half_swing.at_high_humidity - half_swing.at_low_humidity
    )
    annual_water_yield_kg = cycles_per_year * uptake_swing * DRY_COMPOSITE_MASS_KG
    hydrogel_cost_per_kg = (
        (blend_price * sl + econ.c_acrylamide_usd_per_kg) / (1.0 + sl)
        + econ.c_additives_usd_per_kg_composite
    )
    hydrogel_replacement = hydrogel_cost_per_kg * DRY_COMPOSITE_MASS_KG / econ.hydrogel_lifetime_years
    annual_cost_usd = (
        econ.capital_recovery_factor() * econ.total_investment_factor * C_DEVICE_USD
        + hydrogel_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * C_DEVICE_USD
        + econ.energy_cost_usd_per_year
    )
    if annual_water_yield_kg <= 0.0 or not math.isfinite(annual_water_yield_kg) or not math.isfinite(annual_cost_usd):
        return 1e30
    return float(annual_cost_usd / (econ.utilization_factor * (annual_water_yield_kg + 1e-9)))


@dataclass(slots=True)
class ZSROptResult:
    """Result of :func:`optimize_zsr_blend_and_sl`."""

    best_sl: float
    best_f: NDArray[np.float64]
    best_lcow: float
    names: tuple[str, ...]
    nfev: int
    success: bool
    message: str
    backend: str = "unknown"


def optimize_zsr_blend_and_sl(
    site: SiteClimate,
    salt_names: tuple[str, ...],
    econ: LCOEconomicParams | None = None,
    *,
    sl_lo: float = 0.05,
    sl_hi: float = 16.0,
    ipopt_tee: bool = False,
    ipopt_print_level: int | None = None,
) -> ZSROptResult:
    """Optimize salt blend weights and salt-to-polymer ratio to minimize LCOW.

    Uses Pyomo + Ipopt when available; falls back to SciPy SLSQP.
    """
    from pyomo.opt import TerminationCondition
    from src.models.zsr_lcow_model import build_lcow_model, extract_solution

    if len(salt_names) < 1:
        raise ValueError("At least one salt name is required.")
    econ = econ or LCOEconomicParams()
    num_salts = len(salt_names)
    for name in salt_names:
        get_salt(name)

    pyomo_model = build_lcow_model(site, salt_names, econ, salt_to_polymer_ratio_min=sl_lo, salt_to_polymer_ratio_max=sl_hi)
    if getattr(pyomo_model, "infeasible", False):
        return ZSROptResult(
            float("nan"),
            np.zeros(num_salts, dtype=np.float64),
            1e30,
            salt_names,
            0,
            False,
            "No finite binary molalities for this site/salt list.",
            backend="none",
        )

    if ipopt_available():
        opt = SolverFactory("ipopt", validate=False)
        opt.options["max_iter"] = 500
        opt.options["tol"] = 1e-7
        if ipopt_print_level is not None:
            opt.options["print_level"] = int(ipopt_print_level)
        res = opt.solve(pyomo_model, tee=ipopt_tee, load_solutions=True)
        solver_result = res.solver
        s0 = solver_result[0] if isinstance(solver_result, (list, tuple)) and solver_result else solver_result
        termination_condition = getattr(s0, "termination_condition", None) if s0 is not None else None
        converged = termination_condition in (
            TerminationCondition.optimal, TerminationCondition.locallyOptimal
        )
        num_iterations = int(getattr(s0, "iterations", None) or 0) if s0 is not None else 0
        solver_message = str(getattr(s0, "message", "") or getattr(s0, "Message", "") or "")
        if converged:
            opt_sl, opt_weights, opt_lcow = extract_solution(pyomo_model)
            if math.isfinite(opt_sl) and math.isfinite(opt_lcow) and opt_lcow < 0.99 * 1e30:
                return ZSROptResult(
                    opt_sl,
                    _normalize_blend_weights(np.array(opt_weights, dtype=np.float64), salt_names),
                    float(opt_lcow),
                    salt_names,
                    num_iterations,
                    True,
                    solver_message or "ipopt",
                    backend="ipopt",
                )

    # SciPy SLSQP fallback
    from scipy.optimize import minimize

    def unpack(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        sl = float(x[0])
        if num_salts == 1:
            return sl, np.ones(1, dtype=np.float64)
        rest = x[1:num_salts]
        weights = np.append(rest, max(0.0, 1.0 - float(np.sum(rest))))
        return sl, weights

    def objective(x: NDArray[np.float64]) -> float:
        sl, weights = unpack(x)
        if sl < sl_lo or sl > sl_hi or np.any(weights < 0) or not math.isfinite(sl):
            return 1e12
        return lcow_zsr_at_sl(weights, salt_names, site, econ, sl)

    x0 = np.empty(num_salts, dtype=np.float64)
    x0[0] = 4.0
    if num_salts > 1:
        x0[1:] = 1.0 / num_salts

    bounds = [(sl_lo, sl_hi)] + [(0.0, 1.0)] * (num_salts - 1)

    constraints = []
    if num_salts > 1:
        constraints.append({
            "type": "ineq",
            "fun": lambda x: 1.0 - float(np.sum(x[1:num_salts])),
        })

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=tuple(bounds),
        constraints=constraints or (),
        options={"maxiter": 400, "ftol": 1e-9},
    )
    num_func_evals = int(getattr(res, "nfev", 0) or 0)
    if not res.success or res.x is None:
        return ZSROptResult(
            float("nan"),
            np.zeros(num_salts, dtype=np.float64),
            1e30,
            salt_names,
            num_func_evals,
            False,
            str(res.message),
            backend="scipy_slsqp",
        )
    result_sl, result_weights = unpack(np.array(res.x, dtype=np.float64))
    result_lcow = lcow_zsr_at_sl(result_weights, salt_names, site, econ, result_sl)
    if not (math.isfinite(result_lcow) and result_lcow < 0.99 * 1e30):
        return ZSROptResult(
            result_sl,
            result_weights,
            1e30,
            salt_names,
            num_func_evals,
            False,
            f"unbounded or failed LCOW: {res.message!s}",
            backend="scipy_slsqp",
        )
    return ZSROptResult(
        result_sl,
        _normalize_blend_weights(result_weights, salt_names),
        float(result_lcow),
        salt_names,
        num_func_evals,
        bool(res.success),
        str(res.message),
        backend="scipy_slsqp",
    )
