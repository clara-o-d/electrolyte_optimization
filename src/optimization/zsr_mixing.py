"""Zdanovskii isopiestic (ZSR) mixing rule for salt brines.

At a fixed water activity, the molality of each salt in the mixture is:

    m_i_blend = blend_weight_i * m_i_reference

where m_i_reference is the molality of a pure salt-i brine in equilibrium at the
same water activity (and temperature). The blend weights must sum to 1 (simplex
constraint).

The two operating points used by the SAWH model are:

* Uptake (humid night air): water activity = humidity_high, temperature = ambient.
* Desorption (sun-heated, sealed gel): water activity =
  ``desorption_water_activity(T_cond, T_gel)``, temperature = gel_temperature_c.

Optimization runs via a Pyomo NLP (Ipopt) when available; falls back to SciPy SLSQP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pyomo.opt import SolverFactory

from src.materials.salts import get_salt
from src.models.zsr_lcow_model import SiteClimate, HalfSwingCoefficients, ZSRObjective
from src.optimization.economics import (
    C_DEVICE_USD,
    DRY_COMPOSITE_MASS_KG,
    LCOEconomicParams,
    WATER_DENSITY_KG_PER_L,
)
from src.optimization.brine_equilibrium import binary_molality_at_rh, equilibrate_salt_mf
from src.optimization.brine_uptake import (
    WATER_MOLAR_MASS_G_MOL,
    desorption_water_activity,
    water_sorption_factor,
)


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
    water_activity: float,
    temperature_c: float,
) -> tuple[float, float, float]:
    """Compute mixed brine state from the ZSR rule at a given water activity / T.

    For uptake the water activity equals the ambient relative humidity. For
    desorption it equals :func:`desorption_water_activity` at the gel temperature.

    Returns (brine_salt_fraction, effective_ions_per_formula, effective_formula_weight_g_per_mol),
    or (nan, nan, nan) if the brine state is unphysical.
    """
    w = _normalize_blend_weights(blend_weights, salt_names)
    reference_molality = np.empty(len(salt_names), dtype=np.float64)
    formula_weight = np.empty(len(salt_names), dtype=np.float64)
    ions_per_formula = np.empty(len(salt_names), dtype=np.float64)

    for i, name in enumerate(salt_names):
        m_ref = binary_molality_at_rh(name, water_activity, temperature_c)
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
    *,
    electric_heat_w_per_m2: float = 0.0,
) -> HalfSwingCoefficients | None:
    """Compute uptake coefficients for a ZSR brine blend at the uptake and desorption states.

    With ``electric_heat_w_per_m2 > 0`` the gel temperature is solved from the
    energy balance with the extra electrical heat input, shifting the desorption
    water activity. Returns None if the brine state is unphysical at either state.
    """
    from src.optimization.heat_transfer import gel_steady_state_temperature_c

    ambient_temperature_c = site.temperature_c
    if electric_heat_w_per_m2 > 0.0:
        gel_temperature_c = gel_steady_state_temperature_c(
            site.solar_irradiance_w_per_m2,
            ambient_temperature_c,
            electric_heat_w_per_m2=electric_heat_w_per_m2,
        )
    else:
        gel_temperature_c = site.gel_temperature_c
    if not math.isfinite(gel_temperature_c):
        return None
    desorption_a_w = desorption_water_activity(ambient_temperature_c, gel_temperature_c)
    if not math.isfinite(desorption_a_w) or desorption_a_w <= 0.0 or desorption_a_w >= 1.0:
        return None

    fraction_uptake, ions_uptake, weight_uptake = zsr_brine_state(
        blend_weights, salt_names, site.humidity_high, ambient_temperature_c
    )
    fraction_desorption, ions_desorption, weight_desorption = zsr_brine_state(
        blend_weights, salt_names, desorption_a_w, gel_temperature_c
    )
    if not all(
        map(
            math.isfinite,
            (
                fraction_uptake, fraction_desorption,
                weight_uptake, weight_desorption,
                ions_uptake, ions_desorption,
            ),
        )
    ):
        return None
    sorption_uptake = water_sorption_factor(
        site.humidity_high, fraction_uptake, ions_uptake, weight_uptake
    )
    sorption_desorption = water_sorption_factor(
        desorption_a_w, fraction_desorption, ions_desorption, weight_desorption
    )
    if not (math.isfinite(sorption_uptake) and math.isfinite(sorption_desorption)):
        return None
    uptake_at_uptake = sorption_uptake * (WATER_MOLAR_MASS_G_MOL / weight_uptake)
    uptake_at_desorption = sorption_desorption * (WATER_MOLAR_MASS_G_MOL / weight_desorption)
    if not (
        uptake_at_uptake > uptake_at_desorption + 1e-15
        and math.isfinite(uptake_at_uptake)
        and math.isfinite(uptake_at_desorption)
    ):
        return None
    return HalfSwingCoefficients(
        at_uptake=uptake_at_uptake, at_desorption=uptake_at_desorption
    )


def _blend_salt_price(
    blend_weights: NDArray[np.float64],
    salt_names: tuple[str, ...],
) -> float:
    w = _normalize_blend_weights(blend_weights, salt_names)
    return float(sum(w[i] * get_salt(name).c_salt_usd_per_kg for i, name in enumerate(salt_names)))


def daily_water_yield_L_per_m2_at_sl(
    blend_weights: NDArray[np.float64],
    salt_names: tuple[str, ...],
    site: SiteClimate,
    salt_to_polymer_ratio: float,
    *,
    electric_heat_w_per_m2: float = 0.0,
) -> float:
    """Scalar daily water yield (L water / m² / day) for a ZSR brine blend at fixed SL.

    Second objective for the LCOW–yield Pareto sweep. ``DRY_COMPOSITE_MASS_KG`` is
    per 1 m² footprint, so ``uptake_swing * DRY_COMPOSITE_MASS_KG`` already gives
    kg water / m² / cycle; with one cycle/day and ρ_water ≈ 1 kg/L it is L/m²/day.
    Returns 0.0 when the blend is infeasible at this site (no positive half-swing).
    """
    if salt_to_polymer_ratio <= 0.0 or not math.isfinite(salt_to_polymer_ratio):
        return 0.0
    half_swing = uptake_B_zsr(
        blend_weights, salt_names, site,
        electric_heat_w_per_m2=electric_heat_w_per_m2,
    )
    if half_swing is None or half_swing.at_uptake <= half_swing.at_desorption + 1e-15:
        return 0.0
    sl = salt_to_polymer_ratio
    salt_fraction_in_composite = sl / (1.0 + sl)
    uptake_swing = 0.5 * salt_fraction_in_composite * (
        half_swing.at_uptake - half_swing.at_desorption
    )
    daily_yield_kg_per_m2 = uptake_swing * DRY_COMPOSITE_MASS_KG
    if not math.isfinite(daily_yield_kg_per_m2) or daily_yield_kg_per_m2 < 0.0:
        return 0.0
    return float(daily_yield_kg_per_m2 / WATER_DENSITY_KG_PER_L)


def lcow_zsr_at_sl(
    blend_weights: NDArray[np.float64],
    salt_names: tuple[str, ...],
    site: SiteClimate,
    econ: LCOEconomicParams,
    salt_to_polymer_ratio: float,
    *,
    cycles_per_year: int = 365,
    electric_heat_w_per_m2: float = 0.0,
) -> float:
    """Scalar LCOW (USD/kg water) for a ZSR brine blend at a fixed salt-to-polymer ratio.

    With ``electric_heat_w_per_m2 > 0`` the desorption state uses the actively
    heated gel temperature and the LCOW annual cost includes the corresponding
    electricity expense at ``econ.electricity_price_usd_per_kwh``.
    """
    if salt_to_polymer_ratio <= 0.0 or not math.isfinite(salt_to_polymer_ratio):
        return 1e30
    half_swing = uptake_B_zsr(
        blend_weights, salt_names, site,
        electric_heat_w_per_m2=electric_heat_w_per_m2,
    )
    if half_swing is None or half_swing.at_uptake <= half_swing.at_desorption + 1e-15:
        return 1e30
    blend_price = _blend_salt_price(blend_weights, salt_names)
    sl = salt_to_polymer_ratio
    salt_fraction_in_composite = sl / (1.0 + sl)
    uptake_swing = 0.5 * salt_fraction_in_composite * (
        half_swing.at_uptake - half_swing.at_desorption
    )
    annual_water_yield_kg = cycles_per_year * uptake_swing * DRY_COMPOSITE_MASS_KG
    hydrogel_cost_per_kg = (
        (blend_price * sl + econ.c_acrylamide_usd_per_kg) / (1.0 + sl)
        + econ.c_additives_usd_per_kg_composite
    )
    hydrogel_replacement = hydrogel_cost_per_kg * DRY_COMPOSITE_MASS_KG / econ.hydrogel_lifetime_years
    annual_electricity_cost = (
        econ.electricity_price_usd_per_kwh
        * float(electric_heat_w_per_m2)
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
    )
    annual_cost_usd = (
        econ.capital_recovery_factor() * econ.total_investment_factor * C_DEVICE_USD
        + hydrogel_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * C_DEVICE_USD
        + econ.energy_cost_usd_per_year
        + annual_electricity_cost
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
    best_daily_yield_L_per_m2: float
    names: tuple[str, ...]
    nfev: int
    success: bool
    message: str
    backend: str = "unknown"
    best_electric_heat_w_per_m2: float = 0.0
    """Optimal electrical heat power density (W/m^2); 0.0 in passive mode."""
    best_gel_temperature_c: float = float("nan")
    """Optimal gel temperature (deg C); NaN when active heating is disabled."""


def _failed_result(
    salt_names: tuple[str, ...],
    message: str,
    *,
    backend: str,
    nfev: int = 0,
) -> ZSROptResult:
    return ZSROptResult(
        float("nan"),
        np.zeros(len(salt_names), dtype=np.float64),
        1e30,
        0.0,
        salt_names,
        nfev,
        False,
        message,
        backend=backend,
    )


def optimize_zsr_blend_and_sl(
    site: SiteClimate,
    salt_names: tuple[str, ...],
    econ: LCOEconomicParams | None = None,
    *,
    sl_lo: float = 0.05,
    sl_hi: float = 16.0,
    ipopt_tee: bool = False,
    ipopt_print_level: int | None = None,
    objective: ZSRObjective = "lcow",
    min_daily_yield_L_per_m2: float | None = None,
) -> ZSROptResult:
    """Optimize salt blend weights and salt-to-polymer ratio.

    Parameters
    ----------
    objective:
        ``"lcow"`` (default) minimizes LCOW; ``"yield"`` maximizes daily water
        yield (L/m²/day). Both objectives use the same brine + cost model; only
        the active Pyomo objective differs.
    min_daily_yield_L_per_m2:
        Optional lower bound on daily water yield, used to trace the Pareto
        front by epsilon-constraining yield while minimizing LCOW.

    Uses Pyomo + Ipopt when available; falls back to SciPy SLSQP.
    """
    from pyomo.opt import TerminationCondition
    from src.models.zsr_lcow_model import (
        build_lcow_model,
        extract_active_heating_solution,
        extract_solution_with_yield,
    )

    if len(salt_names) < 1:
        raise ValueError("At least one salt name is required.")
    econ = econ or LCOEconomicParams()
    num_salts = len(salt_names)
    for name in salt_names:
        get_salt(name)
    active_heating = econ.max_electric_heat_w_per_m2 > 0.0

    pyomo_model = build_lcow_model(
        site,
        salt_names,
        econ,
        salt_to_polymer_ratio_min=sl_lo,
        salt_to_polymer_ratio_max=sl_hi,
        objective=objective,
        min_daily_yield_L_per_m2=min_daily_yield_L_per_m2,
    )
    if getattr(pyomo_model, "infeasible", False):
        return _failed_result(
            salt_names,
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
            opt_sl, opt_weights, opt_lcow, opt_yield = extract_solution_with_yield(pyomo_model)
            opt_q_elec, opt_t_gel = extract_active_heating_solution(pyomo_model)
            if (
                math.isfinite(opt_sl)
                and math.isfinite(opt_lcow)
                and opt_lcow < 0.99 * 1e30
                and math.isfinite(opt_yield)
            ):
                return ZSROptResult(
                    opt_sl,
                    _normalize_blend_weights(np.array(opt_weights, dtype=np.float64), salt_names),
                    float(opt_lcow),
                    float(opt_yield),
                    salt_names,
                    num_iterations,
                    True,
                    solver_message or "ipopt",
                    backend="ipopt",
                    best_electric_heat_w_per_m2=float(opt_q_elec),
                    best_gel_temperature_c=float(opt_t_gel),
                )

    # SciPy SLSQP fallback. Active heating adds Q_elec and T_gel as decision
    # variables coupled by a nonlinear (T^4) energy-balance equality; the
    # surrogate molality also couples T_gel into the brine state. We do not
    # re-derive that machinery in pure NumPy here. If the user requested active
    # heating without Ipopt, we fall back to the passive LCOW (Q_elec = 0) and
    # surface a clear message so it's not silently dropped.
    if active_heating:
        import warnings

        warnings.warn(
            "Active electrical heating requested but Ipopt is unavailable; "
            "SciPy SLSQP fallback does not implement the energy balance. "
            "Falling back to the passive (Q_elec = 0) optimum.",
            RuntimeWarning,
            stacklevel=2,
        )
        passive_econ = LCOEconomicParams(
            discount_rate=econ.discount_rate,
            device_lifetime_years=econ.device_lifetime_years,
            total_investment_factor=econ.total_investment_factor,
            maintenance_cost_fraction=econ.maintenance_cost_fraction,
            utilization_factor=econ.utilization_factor,
            hydrogel_lifetime_years=econ.hydrogel_lifetime_years,
            energy_cost_usd_per_year=econ.energy_cost_usd_per_year,
            c_acrylamide_usd_per_kg=econ.c_acrylamide_usd_per_kg,
            c_additives_usd_per_kg_composite=econ.c_additives_usd_per_kg_composite,
            electricity_price_usd_per_kwh=econ.electricity_price_usd_per_kwh,
            desorption_hours_per_day=econ.desorption_hours_per_day,
            max_electric_heat_w_per_m2=0.0,
        )
        passive_result = optimize_zsr_blend_and_sl(
            site, salt_names, passive_econ,
            sl_lo=sl_lo, sl_hi=sl_hi,
            ipopt_tee=ipopt_tee, ipopt_print_level=ipopt_print_level,
            objective=objective,
            min_daily_yield_L_per_m2=min_daily_yield_L_per_m2,
        )
        passive_result.message = (
            f"active heating skipped (no Ipopt); {passive_result.message}"
        )
        return passive_result

    from scipy.optimize import minimize

    def unpack(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        sl = float(x[0])
        if num_salts == 1:
            return sl, np.ones(1, dtype=np.float64)
        rest = x[1:num_salts]
        weights = np.append(rest, max(0.0, 1.0 - float(np.sum(rest))))
        return sl, weights

    def scalar_lcow(sl: float, weights: NDArray[np.float64]) -> float:
        return lcow_zsr_at_sl(weights, salt_names, site, econ, sl)

    def scalar_yield(sl: float, weights: NDArray[np.float64]) -> float:
        return daily_water_yield_L_per_m2_at_sl(weights, salt_names, site, sl)

    def scipy_objective(x: NDArray[np.float64]) -> float:
        sl, weights = unpack(x)
        if sl < sl_lo or sl > sl_hi or np.any(weights < 0) or not math.isfinite(sl):
            return 1e12
        if objective == "yield":
            # Maximize → minimize negative yield. Infeasible blends return 0,
            # so the objective is bounded above by zero from below.
            return -scalar_yield(sl, weights)
        return scalar_lcow(sl, weights)

    x0 = np.empty(num_salts, dtype=np.float64)
    x0[0] = 4.0
    if num_salts > 1:
        x0[1:] = 1.0 / num_salts

    bounds = [(sl_lo, sl_hi)] + [(0.0, 1.0)] * (num_salts - 1)

    constraints: list[dict] = []
    if num_salts > 1:
        constraints.append({
            "type": "ineq",
            "fun": lambda x: 1.0 - float(np.sum(x[1:num_salts])),
        })
    if min_daily_yield_L_per_m2 is not None:
        eps_yield = float(min_daily_yield_L_per_m2)
        constraints.append({
            "type": "ineq",
            "fun": lambda x: scalar_yield(*unpack(x)) - eps_yield,
        })

    res = minimize(
        scipy_objective,
        x0,
        method="SLSQP",
        bounds=tuple(bounds),
        constraints=constraints or (),
        options={"maxiter": 400, "ftol": 1e-9},
    )
    num_func_evals = int(getattr(res, "nfev", 0) or 0)
    if not res.success or res.x is None:
        return _failed_result(
            salt_names,
            str(res.message),
            backend="scipy_slsqp",
            nfev=num_func_evals,
        )
    result_sl, result_weights = unpack(np.array(res.x, dtype=np.float64))
    result_lcow = scalar_lcow(result_sl, result_weights)
    result_yield = scalar_yield(result_sl, result_weights)
    if not (math.isfinite(result_lcow) and result_lcow < 0.99 * 1e30):
        return ZSROptResult(
            result_sl,
            result_weights,
            1e30,
            float(result_yield),
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
        float(result_yield),
        salt_names,
        num_func_evals,
        bool(res.success),
        str(res.message),
        backend="scipy_slsqp",
    )


@dataclass(slots=True)
class ParetoPoint:
    """One Pareto-optimal (LCOW, daily-yield) solution from the sweep."""

    salt_to_polymer_ratio: float
    blend_weights: NDArray[np.float64]
    lcow_usd_per_kg: float
    daily_water_yield_L_per_m2: float
    backend: str
    success: bool
    message: str
    is_anchor_min_lcow: bool = False
    is_anchor_max_yield: bool = False
    electric_heat_w_per_m2: float = 0.0
    gel_temperature_c: float = float("nan")


@dataclass(slots=True)
class ParetoFrontResult:
    """Result of :func:`optimize_zsr_pareto_front`.

    ``points`` is sorted by increasing ``daily_water_yield_L_per_m2`` and
    therefore (for a well-behaved problem) increasing ``lcow_usd_per_kg``. The
    two anchors are also exposed separately for convenience.
    """

    points: list[ParetoPoint]
    names: tuple[str, ...]
    min_lcow_point: ParetoPoint
    max_yield_point: ParetoPoint


def optimize_zsr_pareto_front(
    site: SiteClimate,
    salt_names: tuple[str, ...],
    econ: LCOEconomicParams | None = None,
    *,
    num_points: int = 20,
    sl_lo: float = 0.05,
    sl_hi: float = 16.0,
    yield_relative_tolerance: float = 1e-4,
    ipopt_tee: bool = False,
    ipopt_print_level: int | None = None,
) -> ParetoFrontResult:
    """Trace the LCOW–daily-yield Pareto front by epsilon-constraining yield.

    Method (standard epsilon-constraint scalarization):

    1. Solve min LCOW (unconstrained) → anchor with the lowest LCOW and the
       associated yield ``y_lo``.
    2. Solve max daily yield (unconstrained) → anchor with the highest yield
       ``y_hi`` and the associated LCOW.
    3. For ``num_points - 2`` levels evenly spaced in ``(y_lo, y_hi)``, solve
       min LCOW subject to ``daily_water_yield_L_per_m2 >= y_k``. Each gives a
       Pareto-optimal trade-off point.

    Successful points are sorted by increasing daily yield. Infeasible / failed
    sub-solves are dropped from ``points`` but still surfaced if either anchor
    fails.
    """
    if num_points < 2:
        raise ValueError("num_points must be >= 2 (need at least the two anchors).")
    econ = econ or LCOEconomicParams()

    def _to_point(
        result: ZSROptResult,
        *,
        anchor_min_lcow: bool = False,
        anchor_max_yield: bool = False,
    ) -> ParetoPoint:
        return ParetoPoint(
            salt_to_polymer_ratio=float(result.best_sl),
            blend_weights=np.array(result.best_f, dtype=np.float64),
            lcow_usd_per_kg=float(result.best_lcow),
            daily_water_yield_L_per_m2=float(result.best_daily_yield_L_per_m2),
            backend=result.backend,
            success=bool(result.success),
            message=result.message,
            is_anchor_min_lcow=anchor_min_lcow,
            is_anchor_max_yield=anchor_max_yield,
            electric_heat_w_per_m2=float(result.best_electric_heat_w_per_m2),
            gel_temperature_c=float(result.best_gel_temperature_c),
        )

    min_lcow_solve = optimize_zsr_blend_and_sl(
        site, salt_names, econ,
        sl_lo=sl_lo, sl_hi=sl_hi,
        ipopt_tee=ipopt_tee, ipopt_print_level=ipopt_print_level,
        objective="lcow",
    )
    max_yield_solve = optimize_zsr_blend_and_sl(
        site, salt_names, econ,
        sl_lo=sl_lo, sl_hi=sl_hi,
        ipopt_tee=ipopt_tee, ipopt_print_level=ipopt_print_level,
        objective="yield",
    )
    min_lcow_point = _to_point(min_lcow_solve, anchor_min_lcow=True)
    max_yield_point = _to_point(max_yield_solve, anchor_max_yield=True)

    points: list[ParetoPoint] = []
    if min_lcow_point.success:
        points.append(min_lcow_point)
    if max_yield_point.success:
        points.append(max_yield_point)

    y_lo = min_lcow_point.daily_water_yield_L_per_m2 if min_lcow_point.success else float("nan")
    y_hi = max_yield_point.daily_water_yield_L_per_m2 if max_yield_point.success else float("nan")
    # Skip interior sweep if either anchor is missing or the band is degenerate
    # (e.g. min-LCOW happens to coincide with max-yield).
    band_ok = (
        math.isfinite(y_lo) and math.isfinite(y_hi)
        and (y_hi - y_lo) > yield_relative_tolerance * max(abs(y_hi), 1e-12)
    )
    num_interior = max(0, num_points - 2)
    if band_ok and num_interior > 0:
        for k in range(1, num_interior + 1):
            frac = k / float(num_interior + 1)
            y_k = y_lo + frac * (y_hi - y_lo)
            sub = optimize_zsr_blend_and_sl(
                site, salt_names, econ,
                sl_lo=sl_lo, sl_hi=sl_hi,
                ipopt_tee=ipopt_tee, ipopt_print_level=ipopt_print_level,
                objective="lcow",
                min_daily_yield_L_per_m2=y_k,
            )
            if sub.success and math.isfinite(sub.best_lcow) and sub.best_lcow < 0.99 * 1e30:
                points.append(_to_point(sub))

    points.sort(key=lambda p: p.daily_water_yield_L_per_m2)
    return ParetoFrontResult(
        points=points,
        names=salt_names,
        min_lcow_point=min_lcow_point,
        max_yield_point=max_yield_point,
    )
