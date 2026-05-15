"""Pyomo NLP model: minimize levelized cost of water for a salt-hydrogel composite.

Decision variables:
  - salt_to_polymer_ratio : mass of salt per mass of polymer in the dry composite
  - blend_weight[i]       : fraction of the mixture attributed to salt i (sums to 1)

The brine state is computed at two operating points and the half-cycle yield comes
from their difference:

* Uptake (humid night air): the gel is open to the atmosphere and equilibrates
  with the ambient relative humidity at ambient temperature.
* Desorption (sun-heated day): the gel is sealed off from the atmosphere and
  reaches vapor equilibrium with a condenser at ambient temperature. The brine
  composition is set by the effective water activity
  ``a_w = P_sat(T_cond) * T_gel / (P_sat(T_gel) * T_cond)`` at the gel temperature
  (see :func:`src.optimization.brine_uptake.desorption_water_activity`).

At both operating points the multi-salt brine state is built with the ZSR
isopiestic mixing rule: each salt contributes molality proportional to its blend
weight and its binary reference molality at that operating point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyomo.environ as pe
from numpy.typing import NDArray

from src.materials.salts import get_salt
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
from src.optimization.active_heating import (
    default_t_gel_grid_c,
    fit_desorption_molality_polynomial,
    pyomo_poly_expr,
)
from src.optimization.heat_transfer import (
    DEFAULT_GEL_CONVECTION_W_M2_K,
    DEFAULT_GEL_EMISSIVITY,
    DEFAULT_SOLAR_ABSORPTIVITY,
    DEFAULT_SOLAR_IRRADIANCE_W_M2,
    STEFAN_BOLTZMANN_W_M2_K4,
    gel_steady_state_temperature_c,
)


ZSRObjective = Literal["lcow", "yield"]


@dataclass(frozen=True, slots=True)
class SiteClimate:
    """Operating conditions for a SAWH site.

    The night-time uptake humidity and ambient/condenser temperature drive
    the uptake state. The desorption state is set by the sun-driven gel
    temperature, which is derived from a steady-state energy balance between
    absorbed solar irradiance and convective + radiative losses to ambient
    (see :func:`src.optimization.heat_transfer.gel_steady_state_temperature_c`).
    The condenser is assumed to track ``temperature_c``.
    """

    humidity_high: float
    temperature_c: float = 25.0
    solar_irradiance_w_per_m2: float = DEFAULT_SOLAR_IRRADIANCE_W_M2

    @property
    def gel_temperature_c(self) -> float:
        """Steady-state gel temperature during sun-driven desorption (deg C)."""
        return gel_steady_state_temperature_c(
            self.solar_irradiance_w_per_m2,
            self.temperature_c,
        )


@dataclass(frozen=True, slots=True)
class HalfSwingCoefficients:
    """Uptake coefficient B at the uptake (night) and desorption (day) operating points.

    B = (water_mole_fraction * ions_per_formula / (1 - water_mole_fraction))
        * (water_MW / salt_MW)

    The difference (at_uptake - at_desorption) drives the half-cycle water yield.
    """

    at_uptake: float
    at_desorption: float


def _desorption_brine_fraction(salt_name: str, site: SiteClimate) -> float:
    """Equilibrium brine salt fraction for a single salt at the desorption state.

    Solves the salt isotherm at the gel temperature using the effective water
    activity set by the sealed gel/condenser equilibrium.
    """
    a_w = desorption_water_activity(site.temperature_c, site.gel_temperature_c)
    if not math.isfinite(a_w) or a_w <= 0.0 or a_w >= 1.0:
        return float("nan")
    return equilibrate_salt_mf(salt_name, a_w, site.gel_temperature_c)


def single_salt_half_swing_coefficients(
    salt_name: str, site: SiteClimate
) -> HalfSwingCoefficients | None:
    """Compute uptake coefficients for one pure salt at both operating points.

    Returns None if the equilibrium brine composition is unphysical at either point.
    """
    rec = get_salt(salt_name)
    brine_fraction_uptake = equilibrate_salt_mf(salt_name, site.humidity_high, site.temperature_c)
    brine_fraction_desorption = _desorption_brine_fraction(salt_name, site)
    if (
        not (math.isfinite(brine_fraction_uptake) and math.isfinite(brine_fraction_desorption))
        or brine_fraction_uptake <= 0.0
        or brine_fraction_desorption <= 0.0
        or brine_fraction_uptake >= 1.0
        or brine_fraction_desorption >= 1.0
    ):
        return None
    sorption_uptake = water_sorption_factor(
        site.humidity_high, brine_fraction_uptake, rec.nu, rec.mw
    )
    a_w_desorption = desorption_water_activity(site.temperature_c, site.gel_temperature_c)
    sorption_desorption = water_sorption_factor(
        a_w_desorption, brine_fraction_desorption, rec.nu, rec.mw
    )
    if not (math.isfinite(sorption_uptake) and math.isfinite(sorption_desorption)):
        return None
    water_to_salt_mw_ratio = WATER_MOLAR_MASS_G_MOL / rec.mw
    return HalfSwingCoefficients(
        at_uptake=sorption_uptake * water_to_salt_mw_ratio,
        at_desorption=sorption_desorption * water_to_salt_mw_ratio,
    )


def single_salt_lcow_at_loading(
    salt_name: str,
    site: SiteClimate,
    econ: LCOEconomicParams,
    salt_to_polymer_ratio: float,
    *,
    cycles_per_year: int = 365,
) -> float:
    """Scalar LCOW (USD/kg water) for one pure salt at a fixed salt-to-polymer ratio.

    Used as a reference and cross-check against the Pyomo ZSR model.
    """
    rec = get_salt(salt_name)
    coeffs = single_salt_half_swing_coefficients(salt_name, site)
    if coeffs is None or coeffs.at_uptake <= coeffs.at_desorption + 1e-15:
        return 1e30
    sl = salt_to_polymer_ratio
    salt_fraction_in_composite = sl / (1.0 + sl)
    uptake_swing = 0.5 * salt_fraction_in_composite * (
        coeffs.at_uptake - coeffs.at_desorption
    )
    annual_water_yield_kg = cycles_per_year * uptake_swing * DRY_COMPOSITE_MASS_KG
    if annual_water_yield_kg <= 0.0 or not math.isfinite(annual_water_yield_kg):
        return 1e30
    hydrogel_cost_per_kg = (
        (rec.c_salt_usd_per_kg * sl + econ.c_acrylamide_usd_per_kg) / (1.0 + sl)
        + econ.c_additives_usd_per_kg_composite
    )
    hydrogel_replacement = hydrogel_cost_per_kg * DRY_COMPOSITE_MASS_KG / econ.hydrogel_lifetime_years
    annual_cost_usd = (
        econ.capital_recovery_factor() * econ.total_investment_factor * C_DEVICE_USD
        + hydrogel_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * C_DEVICE_USD
        + econ.energy_cost_usd_per_year
    )
    return float(annual_cost_usd / (econ.utilization_factor * (annual_water_yield_kg + 1e-9)))


def single_salt_daily_yield_at_loading(
    salt_name: str,
    site: SiteClimate,
    salt_to_polymer_ratio: float,
) -> float:
    """Scalar daily water yield (L water / m² / day) for one pure salt at a fixed SL.

    Companion to :func:`single_salt_lcow_at_loading` for the second objective. The
    device footprint convention (``DRY_COMPOSITE_MASS_KG`` is per 1 m²) means this
    quantity is intrinsically area-normalized.
    """
    coeffs = single_salt_half_swing_coefficients(salt_name, site)
    if coeffs is None or coeffs.at_uptake <= coeffs.at_desorption + 1e-15:
        return 0.0
    sl = salt_to_polymer_ratio
    salt_fraction_in_composite = sl / (1.0 + sl)
    uptake_swing = 0.5 * salt_fraction_in_composite * (
        coeffs.at_uptake - coeffs.at_desorption
    )
    daily_yield_kg_per_m2 = uptake_swing * DRY_COMPOSITE_MASS_KG
    if not math.isfinite(daily_yield_kg_per_m2) or daily_yield_kg_per_m2 < 0.0:
        return 0.0
    return float(daily_yield_kg_per_m2 / WATER_DENSITY_KG_PER_L)


def _annual_electricity_cost_usd(
    econ: LCOEconomicParams,
    electric_heat_w_per_m2,  # noqa: ANN001 - Pyomo Var or float
) -> pe.Expression | float:
    """Annual electricity cost (USD per m^2 of gel footprint) for active heating.

    Q [W/m^2] * desorption_hours_per_day [h/day] * 365 [day/yr] / 1000 [W/kW]
    gives kWh / m^2 / yr; multiplied by ``electricity_price_usd_per_kwh`` gives
    USD / m^2 / yr.
    """
    return (
        econ.electricity_price_usd_per_kwh
        * electric_heat_w_per_m2
        * econ.desorption_hours_per_day
        * 365.0
        / 1000.0
    )


def _annual_cost_expr(
    capital_recovery_factor: float,
    econ: LCOEconomicParams,
    hydrogel_cost_per_kg: pe.Expression,
    device_cost_usd: float,
    composite_dry_mass_kg: float,
    annual_electricity_cost_usd: pe.Expression | float = 0.0,
) -> pe.Expression:
    hydrogel_replacement = hydrogel_cost_per_kg * composite_dry_mass_kg / econ.hydrogel_lifetime_years
    return (
        capital_recovery_factor * econ.total_investment_factor * device_cost_usd
        + hydrogel_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * device_cost_usd
        + econ.energy_cost_usd_per_year
        + annual_electricity_cost_usd
    )


def build_lcow_model(
    site: SiteClimate,
    salt_names: tuple[str, ...],
    econ: LCOEconomicParams,
    *,
    salt_to_polymer_ratio_min: float = 0.05,
    salt_to_polymer_ratio_max: float = 16.0,
    cycles_per_year: int = 365,
    numerical_floor: float = 1e-14,
    objective: ZSRObjective = "lcow",
    min_daily_yield_L_per_m2: float | None = None,
) -> pe.ConcreteModel:
    """Build a Pyomo NLP over blend weights and salt-to-polymer ratio.

    Parameters
    ----------
    objective:
        ``"lcow"`` (default) → minimize LCOW; ``"yield"`` → maximize daily water
        yield in L/m²/day. Both objectives use the same decision variables; only
        the active ``model.objective`` differs.
    min_daily_yield_L_per_m2:
        Optional epsilon-constraint forcing ``daily_water_yield_L_per_m2 >= eps``.
        Used by the Pareto-front sweep to trace the LCOW–yield trade-off curve.

    Returns a model with ``infeasible=True`` if any salt lacks a valid equilibrium
    brine state.
    """
    humidity_uptake = site.humidity_high
    ambient_temperature_c = site.temperature_c
    passive_gel_temperature_c = site.gel_temperature_c
    passive_desorption_a_w = desorption_water_activity(
        ambient_temperature_c, passive_gel_temperature_c
    )
    n = len(salt_names)
    if n < 1:
        raise ValueError("At least one salt name is required.")

    active_heating = econ.max_electric_heat_w_per_m2 > 0.0

    # Precompute per-salt reference molalities at each operating point (scalar Python).
    # Uptake: equilibrium with humid night air at ambient T (T_gel-independent).
    # Desorption (passive): brine in the sealed, sun-heated gel (T_gel = passive
    # solar-only solution) in vapor equilibrium with a condenser at ambient T.
    # In active-heating mode the desorption molality becomes a Pyomo Expression
    # in T_gel via the per-salt polynomial surrogate fit below.
    ref_molality_uptake: list[float] = []
    ref_molality_desorption_passive: list[float] = []
    formula_weight_g_per_mol: list[float] = []
    ions_per_formula_unit: list[float] = []
    salt_price_usd_per_kg: list[float] = []
    desorption_surrogate_coeffs: list[NDArray[np.float64]] = []

    t_gel_max_c = 120.0  # upper bound on the active gel temperature

    if active_heating:
        t_grid = default_t_gel_grid_c(ambient_temperature_c, t_max_c=t_gel_max_c)

    for name in salt_names:
        m_uptake = binary_molality_at_rh(name, humidity_uptake, ambient_temperature_c)
        m_desorption_passive = (
            binary_molality_at_rh(name, passive_desorption_a_w, passive_gel_temperature_c)
            if math.isfinite(passive_desorption_a_w) and 0.0 < passive_desorption_a_w < 1.0
            else float("nan")
        )
        rec = get_salt(name)
        passive_ok = (
            math.isfinite(m_uptake) and math.isfinite(m_desorption_passive)
            and m_uptake >= 0.0 and m_desorption_passive >= 0.0
        )
        if active_heating:
            coeffs = fit_desorption_molality_polynomial(
                name, ambient_temperature_c, t_grid
            )
            # Salt is feasible if (a) uptake state is valid AND (b) either the
            # passive desorption point is valid OR the surrogate fit succeeded
            # somewhere on the grid (so active heating can rescue it).
            if not (math.isfinite(m_uptake) and m_uptake >= 0.0) or coeffs is None:
                stub = pe.ConcreteModel("lcow_infeasible")
                stub.infeasible = True
                stub.objective = pe.Objective(expr=1e30, sense=pe.minimize)
                return stub
            desorption_surrogate_coeffs.append(coeffs)
            # If passive desorption is invalid, seed the Param to a small sentinel
            # value so the (unused) passive Param remains finite; the model only
            # consults the surrogate Expression in active mode.
            if not passive_ok:
                m_desorption_passive = 0.0
        else:
            if not passive_ok:
                stub = pe.ConcreteModel("lcow_infeasible")
                stub.infeasible = True
                stub.objective = pe.Objective(expr=1e30, sense=pe.minimize)
                return stub
        ref_molality_uptake.append(float(m_uptake))
        ref_molality_desorption_passive.append(float(m_desorption_passive))
        formula_weight_g_per_mol.append(float(rec.mw))
        ions_per_formula_unit.append(float(rec.nu))
        salt_price_usd_per_kg.append(float(rec.c_salt_usd_per_kg))

    model = pe.ConcreteModel("lcow_zsr")
    model.infeasible = False
    model.active_heating = active_heating

    def _as_param_dict(values: list[float]) -> dict[int, float]:
        return {i: values[i] for i in range(n)}

    model.salt_idx = pe.RangeSet(0, n - 1)

    # Fixed parameters (computed from binary brine equilibrium before the solve)
    model.ref_molality_at_uptake = pe.Param(
        model.salt_idx, within=pe.NonNegativeReals, initialize=_as_param_dict(ref_molality_uptake)
    )
    model.formula_weight_g_per_mol = pe.Param(
        model.salt_idx, within=pe.PositiveReals, initialize=_as_param_dict(formula_weight_g_per_mol)
    )
    model.ions_per_formula_unit = pe.Param(
        model.salt_idx, within=pe.PositiveReals, initialize=_as_param_dict(ions_per_formula_unit)
    )
    model.salt_price_usd_per_kg = pe.Param(
        model.salt_idx, within=pe.NonNegativeReals, initialize=_as_param_dict(salt_price_usd_per_kg)
    )

    # Decision variables
    model.blend_weight = pe.Var(
        model.salt_idx, bounds=(0.0, 1.0), initialize=1.0 / float(n)
    )
    model.salt_to_polymer_ratio = pe.Var(
        bounds=(salt_to_polymer_ratio_min, salt_to_polymer_ratio_max), initialize=4.0
    )

    # --- Active electrical heating (optional) ---
    # When max_electric_heat_w_per_m2 > 0 we expose Q_elec (W/m^2) and T_gel as
    # decision variables coupled by the gel energy balance. The desorption
    # reference molality of each salt is then evaluated from a polynomial
    # surrogate in T_gel rather than from a precomputed scalar Param.
    if active_heating:
        model.electric_heat_w_per_m2 = pe.Var(
            bounds=(0.0, float(econ.max_electric_heat_w_per_m2)),
            initialize=0.0,
        )
        model.gel_temperature_c = pe.Var(
            bounds=(float(ambient_temperature_c), float(t_gel_max_c)),
            initialize=float(passive_gel_temperature_c),
        )
        t_amb_k = float(ambient_temperature_c) + 273.15
        i_solar = float(site.solar_irradiance_w_per_m2)
        model.gel_energy_balance = pe.Constraint(
            expr=(
                DEFAULT_SOLAR_ABSORPTIVITY * i_solar + model.electric_heat_w_per_m2
                == DEFAULT_GEL_CONVECTION_W_M2_K
                * (model.gel_temperature_c - float(ambient_temperature_c))
                + DEFAULT_GEL_EMISSIVITY
                * STEFAN_BOLTZMANN_W_M2_K4
                * ((model.gel_temperature_c + 273.15) ** 4 - t_amb_k**4)
            )
        )
        # Per-salt polynomial surrogate Expressions: m_ref_desorption_i(T_gel).
        # Coefficients are precomputed by fit_desorption_molality_polynomial.
        # Use a dict to avoid Pyomo's automatic Param storage of numpy arrays.
        model._desorption_surrogate_coeffs = desorption_surrogate_coeffs

        def _ref_desorption_rule(m, i):
            return pyomo_poly_expr(
                m._desorption_surrogate_coeffs[i], m.gel_temperature_c
            )

        model.ref_molality_at_desorption = pe.Expression(
            model.salt_idx, rule=_ref_desorption_rule
        )
    else:
        # Passive mode: scalar Param precomputed at the solar-only T_gel.
        model.ref_molality_at_desorption = pe.Param(
            model.salt_idx,
            within=pe.NonNegativeReals,
            initialize=_as_param_dict(ref_molality_desorption_passive),
        )

    # Constraint: blend weights must sum to 1 (proper mixture)
    model.blend_weights_sum_to_one = pe.Constraint(
        expr=pe.quicksum(model.blend_weight[i] for i in model.salt_idx) == 1.0
    )

    # --- ZSR brine state at each operating point ---
    # Each salt contributes molality = blend_weight[i] * reference_molality[i]
    model.contrib_molality_at_uptake = pe.Expression(
        model.salt_idx,
        rule=lambda m, i: m.blend_weight[i] * m.ref_molality_at_uptake[i],
    )
    model.contrib_molality_at_desorption = pe.Expression(
        model.salt_idx,
        rule=lambda m, i: m.blend_weight[i] * m.ref_molality_at_desorption[i],
    )

    # Total mixture molality (mol / kg water), summed across all salts
    model.total_molality_uptake = pe.Expression(
        expr=pe.quicksum(model.contrib_molality_at_uptake[i] for i in model.salt_idx)
    )
    model.total_molality_desorption = pe.Expression(
        expr=pe.quicksum(model.contrib_molality_at_desorption[i] for i in model.salt_idx)
    )

    # Total dissolved salt mass per kg of water (g/mol * mol/kg_water = g salt / kg water)
    model.dissolved_salt_g_per_kg_water_uptake = pe.Expression(
        expr=pe.quicksum(
            model.contrib_molality_at_uptake[i] * model.formula_weight_g_per_mol[i]
            for i in model.salt_idx
        )
    )
    model.dissolved_salt_g_per_kg_water_desorption = pe.Expression(
        expr=pe.quicksum(
            model.contrib_molality_at_desorption[i] * model.formula_weight_g_per_mol[i]
            for i in model.salt_idx
        )
    )

    # Brine salt mass fraction: dissolved salt / (dissolved salt + 1000 g water per kg)
    model.brine_salt_fraction_uptake = pe.Expression(
        expr=model.dissolved_salt_g_per_kg_water_uptake
        / (model.dissolved_salt_g_per_kg_water_uptake + 1000.0 + numerical_floor)
    )
    model.brine_salt_fraction_desorption = pe.Expression(
        expr=model.dissolved_salt_g_per_kg_water_desorption
        / (model.dissolved_salt_g_per_kg_water_desorption + 1000.0 + numerical_floor)
    )

    # Mixture-averaged ion count per formula unit (molality-weighted)
    model.effective_ions_per_formula_uptake = pe.Expression(
        expr=pe.quicksum(
            model.contrib_molality_at_uptake[i] * model.ions_per_formula_unit[i]
            for i in model.salt_idx
        )
        / (model.total_molality_uptake + numerical_floor)
    )
    model.effective_ions_per_formula_desorption = pe.Expression(
        expr=pe.quicksum(
            model.contrib_molality_at_desorption[i] * model.ions_per_formula_unit[i]
            for i in model.salt_idx
        )
        / (model.total_molality_desorption + numerical_floor)
    )

    # Mixture-averaged formula weight (g/mol), molality-weighted
    model.effective_formula_weight_uptake = pe.Expression(
        expr=model.dissolved_salt_g_per_kg_water_uptake
        / (model.total_molality_uptake + numerical_floor)
    )
    model.effective_formula_weight_desorption = pe.Expression(
        expr=model.dissolved_salt_g_per_kg_water_desorption
        / (model.total_molality_desorption + numerical_floor)
    )

    # Water mole fraction in the brine (colligative basis)
    # x_w = moles_water / (moles_water + ions_per_formula * moles_salt)
    model.moles_water_uptake = pe.Expression(
        expr=(1.0 - model.brine_salt_fraction_uptake) / WATER_MOLAR_MASS_G_MOL
    )
    model.moles_salt_uptake = pe.Expression(
        expr=model.brine_salt_fraction_uptake
        / (model.effective_formula_weight_uptake + numerical_floor)
    )
    model.water_mole_fraction_uptake = pe.Expression(
        expr=model.moles_water_uptake
        / (
            model.moles_water_uptake
            + model.effective_ions_per_formula_uptake * model.moles_salt_uptake
            + numerical_floor
        )
    )

    model.moles_water_desorption = pe.Expression(
        expr=(1.0 - model.brine_salt_fraction_desorption) / WATER_MOLAR_MASS_G_MOL
    )
    model.moles_salt_desorption = pe.Expression(
        expr=model.brine_salt_fraction_desorption
        / (model.effective_formula_weight_desorption + numerical_floor)
    )
    model.water_mole_fraction_desorption = pe.Expression(
        expr=model.moles_water_desorption
        / (
            model.moles_water_desorption
            + model.effective_ions_per_formula_desorption * model.moles_salt_desorption
            + numerical_floor
        )
    )

    # Sorption factor: (x_w * ions_per_formula) / (1 - x_w)
    model.sorption_factor_uptake = pe.Expression(
        expr=(model.water_mole_fraction_uptake * model.effective_ions_per_formula_uptake)
        / (1.0 - model.water_mole_fraction_uptake + numerical_floor)
    )
    model.sorption_factor_desorption = pe.Expression(
        expr=(model.water_mole_fraction_desorption * model.effective_ions_per_formula_desorption)
        / (1.0 - model.water_mole_fraction_desorption + numerical_floor)
    )

    # Uptake coefficient B = sorption_factor * (water_MW / effective_salt_MW) [kg water / kg salt]
    model.uptake_coeff_uptake = pe.Expression(
        expr=model.sorption_factor_uptake
        * WATER_MOLAR_MASS_G_MOL
        / (model.effective_formula_weight_uptake + numerical_floor)
    )
    model.uptake_coeff_desorption = pe.Expression(
        expr=model.sorption_factor_desorption
        * WATER_MOLAR_MASS_G_MOL
        / (model.effective_formula_weight_desorption + numerical_floor)
    )

    # Constraint: uptake (night) must exceed desorption (day) loading (positive half-cycle swing)
    model.positive_half_swing = pe.Constraint(
        expr=model.uptake_coeff_uptake >= model.uptake_coeff_desorption + 1e-9
    )

    # --- Water yield ---
    # Salt fraction of dry composite by mass: SL / (1 + SL)
    model.salt_fraction_in_composite = pe.Expression(
        expr=model.salt_to_polymer_ratio / (1.0 + model.salt_to_polymer_ratio)
    )
    model.uptake_swing = pe.Expression(
        expr=0.5
        * model.salt_fraction_in_composite
        * (model.uptake_coeff_uptake - model.uptake_coeff_desorption)
    )
    model.annual_water_yield_kg = pe.Expression(
        expr=float(cycles_per_year) * model.uptake_swing * DRY_COMPOSITE_MASS_KG
    )
    # Daily water yield in L per m² per day (second objective). DRY_COMPOSITE_MASS_KG
    # is per 1 m² footprint by construction, so uptake_swing * DRY_COMPOSITE_MASS_KG
    # is the per-cycle kg-water-per-m². With one cycle/day and ρ_water ≈ 1 kg/L this
    # equals L water per m² per day.
    model.daily_water_yield_L_per_m2 = pe.Expression(
        expr=model.uptake_swing * DRY_COMPOSITE_MASS_KG / WATER_DENSITY_KG_PER_L
    )

    # Optional epsilon-constraint used by Pareto-front sweep.
    if min_daily_yield_L_per_m2 is not None:
        model.min_daily_yield_constraint = pe.Constraint(
            expr=model.daily_water_yield_L_per_m2 >= float(min_daily_yield_L_per_m2)
        )

    # --- Cost ---
    model.blend_salt_price = pe.Expression(
        expr=pe.quicksum(
            model.blend_weight[i] * model.salt_price_usd_per_kg[i]
            for i in model.salt_idx
        )
    )
    model.hydrogel_cost_per_kg = pe.Expression(
        expr=(model.blend_salt_price * model.salt_to_polymer_ratio + econ.c_acrylamide_usd_per_kg)
        / (1.0 + model.salt_to_polymer_ratio)
        + econ.c_additives_usd_per_kg_composite
    )
    if active_heating:
        model.annual_electricity_cost_usd = pe.Expression(
            expr=_annual_electricity_cost_usd(econ, model.electric_heat_w_per_m2)
        )
        annual_electricity_cost_term = model.annual_electricity_cost_usd
    else:
        annual_electricity_cost_term = 0.0
    model.annual_cost_usd = pe.Expression(
        expr=_annual_cost_expr(
            econ.capital_recovery_factor(), econ, model.hydrogel_cost_per_kg,
            C_DEVICE_USD, DRY_COMPOSITE_MASS_KG,
            annual_electricity_cost_usd=annual_electricity_cost_term,
        )
    )
    model.lcow_usd_per_kg_water = pe.Expression(
        expr=model.annual_cost_usd / (econ.utilization_factor * (model.annual_water_yield_kg + 1e-9))
    )
    if objective == "yield":
        model.objective = pe.Objective(
            expr=model.daily_water_yield_L_per_m2, sense=pe.maximize
        )
    elif objective == "lcow":
        model.objective = pe.Objective(expr=model.lcow_usd_per_kg_water, sense=pe.minimize)
    else:
        raise ValueError(f"Unknown objective {objective!r}; expected 'lcow' or 'yield'.")
    return model


def extract_solution(model: pe.ConcreteModel) -> tuple[float, list[float], float]:
    """Read optimized values from a solved model: (salt_to_polymer_ratio, blend_weights, lcow)."""
    sl = float(pe.value(model.salt_to_polymer_ratio))
    weights = [float(pe.value(model.blend_weight[i])) for i in model.salt_idx]
    lcow = float(pe.value(model.lcow_usd_per_kg_water))
    return sl, weights, lcow


def extract_solution_with_yield(
    model: pe.ConcreteModel,
) -> tuple[float, list[float], float, float]:
    """Like :func:`extract_solution` but also returns daily yield (L water / m² / day)."""
    sl, weights, lcow = extract_solution(model)
    daily_yield = float(pe.value(model.daily_water_yield_L_per_m2))
    return sl, weights, lcow, daily_yield


def extract_active_heating_solution(model: pe.ConcreteModel) -> tuple[float, float]:
    """Return (electric_heat_w_per_m2, gel_temperature_c) from a solved active-heating model.

    For passive models (``model.active_heating == False``) returns
    ``(0.0, float(pe.value(...)) if available else nan)`` to give a uniform
    interface to the optimizer wrappers.
    """
    if getattr(model, "active_heating", False):
        q_elec = float(pe.value(model.electric_heat_w_per_m2))
        t_gel = float(pe.value(model.gel_temperature_c))
        return q_elec, t_gel
    return 0.0, float("nan")
