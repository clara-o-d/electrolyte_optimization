"""Pyomo NLP model: minimize levelized cost of water for a salt-hydrogel composite.

Decision variables:
  - salt_to_polymer_ratio : mass of salt per mass of polymer in the dry composite
  - blend_weight[i]       : fraction of the mixture attributed to salt i (sums to 1)

The brine state at each humidity setpoint is computed via the ZSR isopiestic mixing rule:
each salt contributes molality proportional to its blend weight and its binary reference
molality (molality of a pure salt-water solution at the same humidity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pyomo.environ as pe

from src.materials.salts import get_salt
from src.optimization.economics import C_DEVICE_USD, DRY_COMPOSITE_MASS_KG, LCOEconomicParams
from src.optimization.brine_equilibrium import binary_molality_at_rh, equilibrate_salt_mf
from src.optimization.brine_uptake import WATER_MOLAR_MASS_G_MOL, water_sorption_factor


@dataclass(frozen=True, slots=True)
class SiteClimate:
    """Mean diurnal high and low relative humidity (0–1 fraction) and temperature for a site."""

    humidity_high: float
    humidity_low: float
    temperature_c: float = 25.0


@dataclass(frozen=True, slots=True)
class HalfSwingCoefficients:
    """Uptake coefficient B at the high- and low-humidity setpoints.

    B = (water_mole_fraction * ions_per_formula / (1 - water_mole_fraction))
        * (water_MW / salt_MW)

    The difference (at_high_humidity - at_low_humidity) drives the half-cycle water yield.
    """

    at_high_humidity: float
    at_low_humidity: float


def single_salt_half_swing_coefficients(
    salt_name: str, site: SiteClimate
) -> HalfSwingCoefficients | None:
    """Compute uptake coefficients for one pure salt at both humidity setpoints.

    Returns None if the equilibrium brine composition is unphysical at either setpoint.
    """
    rec = get_salt(salt_name)
    brine_fraction_high = equilibrate_salt_mf(salt_name, site.humidity_high, site.temperature_c)
    brine_fraction_low = equilibrate_salt_mf(salt_name, site.humidity_low, site.temperature_c)
    if (
        not (math.isfinite(brine_fraction_high) and math.isfinite(brine_fraction_low))
        or brine_fraction_high <= 0.0
        or brine_fraction_low <= 0.0
        or brine_fraction_high >= 1.0
        or brine_fraction_low >= 1.0
    ):
        return None
    sorption_high = water_sorption_factor(
        site.humidity_high, brine_fraction_high, rec.nu, rec.mw
    )
    sorption_low = water_sorption_factor(
        site.humidity_low, brine_fraction_low, rec.nu, rec.mw
    )
    if not (math.isfinite(sorption_high) and math.isfinite(sorption_low)):
        return None
    water_to_salt_mw_ratio = WATER_MOLAR_MASS_G_MOL / rec.mw
    return HalfSwingCoefficients(
        at_high_humidity=sorption_high * water_to_salt_mw_ratio,
        at_low_humidity=sorption_low * water_to_salt_mw_ratio,
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
    if coeffs is None or coeffs.at_high_humidity <= coeffs.at_low_humidity + 1e-15:
        return 1e30
    sl = salt_to_polymer_ratio
    salt_fraction_in_composite = sl / (1.0 + sl)
    uptake_swing = 0.5 * salt_fraction_in_composite * (
        coeffs.at_high_humidity - coeffs.at_low_humidity
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


def _annual_cost_expr(
    capital_recovery_factor: float,
    econ: LCOEconomicParams,
    hydrogel_cost_per_kg: pe.Expression,
    device_cost_usd: float,
    composite_dry_mass_kg: float,
) -> pe.Expression:
    hydrogel_replacement = hydrogel_cost_per_kg * composite_dry_mass_kg / econ.hydrogel_lifetime_years
    return (
        capital_recovery_factor * econ.total_investment_factor * device_cost_usd
        + hydrogel_replacement
        + econ.maintenance_cost_fraction * econ.total_investment_factor * device_cost_usd
        + econ.energy_cost_usd_per_year
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
) -> pe.ConcreteModel:
    """Build a Pyomo NLP that minimizes LCOW over blend weights and salt-to-polymer ratio.

    Returns a model with infeasible=True if any salt lacks a valid equilibrium brine state.
    """
    humidity_high = site.humidity_high
    humidity_low = site.humidity_low
    temperature_c = site.temperature_c
    n = len(salt_names)
    if n < 1:
        raise ValueError("At least one salt name is required.")

    # Precompute per-salt reference molalities from binary brine equilibrium (scalar Python)
    ref_molality_high: list[float] = []
    ref_molality_low: list[float] = []
    formula_weight_g_per_mol: list[float] = []
    ions_per_formula_unit: list[float] = []
    salt_price_usd_per_kg: list[float] = []

    for name in salt_names:
        m_high = binary_molality_at_rh(name, humidity_high, temperature_c)
        m_low = binary_molality_at_rh(name, humidity_low, temperature_c)
        rec = get_salt(name)
        if not (
            math.isfinite(m_high) and math.isfinite(m_low)
            and m_high >= 0.0 and m_low >= 0.0
        ):
            stub = pe.ConcreteModel("lcow_infeasible")
            stub.infeasible = True
            stub.objective = pe.Objective(expr=1e30, sense=pe.minimize)
            return stub
        ref_molality_high.append(float(m_high))
        ref_molality_low.append(float(m_low))
        formula_weight_g_per_mol.append(float(rec.mw))
        ions_per_formula_unit.append(float(rec.nu))
        salt_price_usd_per_kg.append(float(rec.c_salt_usd_per_kg))

    model = pe.ConcreteModel("lcow_zsr")
    model.infeasible = False

    def _as_param_dict(values: list[float]) -> dict[int, float]:
        return {i: values[i] for i in range(n)}

    model.salt_idx = pe.RangeSet(0, n - 1)

    # Fixed parameters (computed from binary brine equilibrium before the solve)
    model.ref_molality_at_high_rh = pe.Param(
        model.salt_idx, within=pe.NonNegativeReals, initialize=_as_param_dict(ref_molality_high)
    )
    model.ref_molality_at_low_rh = pe.Param(
        model.salt_idx, within=pe.NonNegativeReals, initialize=_as_param_dict(ref_molality_low)
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

    # Constraint: blend weights must sum to 1 (proper mixture)
    model.blend_weights_sum_to_one = pe.Constraint(
        expr=pe.quicksum(model.blend_weight[i] for i in model.salt_idx) == 1.0
    )

    # --- ZSR brine state at each humidity setpoint ---
    # Each salt contributes molality = blend_weight[i] * reference_molality[i]
    model.contrib_molality_at_high_rh = pe.Expression(
        model.salt_idx,
        rule=lambda m, i: m.blend_weight[i] * m.ref_molality_at_high_rh[i],
    )
    model.contrib_molality_at_low_rh = pe.Expression(
        model.salt_idx,
        rule=lambda m, i: m.blend_weight[i] * m.ref_molality_at_low_rh[i],
    )

    # Total mixture molality (mol / kg water), summed across all salts
    model.total_molality_high_rh = pe.Expression(
        expr=pe.quicksum(model.contrib_molality_at_high_rh[i] for i in model.salt_idx)
    )
    model.total_molality_low_rh = pe.Expression(
        expr=pe.quicksum(model.contrib_molality_at_low_rh[i] for i in model.salt_idx)
    )

    # Total dissolved salt mass per kg of water (g/mol * mol/kg_water = g salt / kg water)
    model.dissolved_salt_g_per_kg_water_high_rh = pe.Expression(
        expr=pe.quicksum(
            model.contrib_molality_at_high_rh[i] * model.formula_weight_g_per_mol[i]
            for i in model.salt_idx
        )
    )
    model.dissolved_salt_g_per_kg_water_low_rh = pe.Expression(
        expr=pe.quicksum(
            model.contrib_molality_at_low_rh[i] * model.formula_weight_g_per_mol[i]
            for i in model.salt_idx
        )
    )

    # Brine salt mass fraction: dissolved salt / (dissolved salt + 1000 g water per kg)
    model.brine_salt_fraction_high_rh = pe.Expression(
        expr=model.dissolved_salt_g_per_kg_water_high_rh
        / (model.dissolved_salt_g_per_kg_water_high_rh + 1000.0 + numerical_floor)
    )
    model.brine_salt_fraction_low_rh = pe.Expression(
        expr=model.dissolved_salt_g_per_kg_water_low_rh
        / (model.dissolved_salt_g_per_kg_water_low_rh + 1000.0 + numerical_floor)
    )

    # Mixture-averaged ion count per formula unit (molality-weighted)
    model.effective_ions_per_formula_high_rh = pe.Expression(
        expr=pe.quicksum(
            model.contrib_molality_at_high_rh[i] * model.ions_per_formula_unit[i]
            for i in model.salt_idx
        )
        / (model.total_molality_high_rh + numerical_floor)
    )
    model.effective_ions_per_formula_low_rh = pe.Expression(
        expr=pe.quicksum(
            model.contrib_molality_at_low_rh[i] * model.ions_per_formula_unit[i]
            for i in model.salt_idx
        )
        / (model.total_molality_low_rh + numerical_floor)
    )

    # Mixture-averaged formula weight (g/mol), molality-weighted
    model.effective_formula_weight_high_rh = pe.Expression(
        expr=model.dissolved_salt_g_per_kg_water_high_rh
        / (model.total_molality_high_rh + numerical_floor)
    )
    model.effective_formula_weight_low_rh = pe.Expression(
        expr=model.dissolved_salt_g_per_kg_water_low_rh
        / (model.total_molality_low_rh + numerical_floor)
    )

    # Water mole fraction in the brine (colligative basis)
    # x_w = moles_water / (moles_water + ions_per_formula * moles_salt)
    model.moles_water_high_rh = pe.Expression(
        expr=(1.0 - model.brine_salt_fraction_high_rh) / WATER_MOLAR_MASS_G_MOL
    )
    model.moles_salt_high_rh = pe.Expression(
        expr=model.brine_salt_fraction_high_rh
        / (model.effective_formula_weight_high_rh + numerical_floor)
    )
    model.water_mole_fraction_high_rh = pe.Expression(
        expr=model.moles_water_high_rh
        / (
            model.moles_water_high_rh
            + model.effective_ions_per_formula_high_rh * model.moles_salt_high_rh
            + numerical_floor
        )
    )

    model.moles_water_low_rh = pe.Expression(
        expr=(1.0 - model.brine_salt_fraction_low_rh) / WATER_MOLAR_MASS_G_MOL
    )
    model.moles_salt_low_rh = pe.Expression(
        expr=model.brine_salt_fraction_low_rh
        / (model.effective_formula_weight_low_rh + numerical_floor)
    )
    model.water_mole_fraction_low_rh = pe.Expression(
        expr=model.moles_water_low_rh
        / (
            model.moles_water_low_rh
            + model.effective_ions_per_formula_low_rh * model.moles_salt_low_rh
            + numerical_floor
        )
    )

    # Sorption factor: (x_w * ions_per_formula) / (1 - x_w)
    model.sorption_factor_high_rh = pe.Expression(
        expr=(model.water_mole_fraction_high_rh * model.effective_ions_per_formula_high_rh)
        / (1.0 - model.water_mole_fraction_high_rh + numerical_floor)
    )
    model.sorption_factor_low_rh = pe.Expression(
        expr=(model.water_mole_fraction_low_rh * model.effective_ions_per_formula_low_rh)
        / (1.0 - model.water_mole_fraction_low_rh + numerical_floor)
    )

    # Uptake coefficient B = sorption_factor * (water_MW / effective_salt_MW) [kg water / kg salt]
    model.uptake_coeff_high_rh = pe.Expression(
        expr=model.sorption_factor_high_rh
        * WATER_MOLAR_MASS_G_MOL
        / (model.effective_formula_weight_high_rh + numerical_floor)
    )
    model.uptake_coeff_low_rh = pe.Expression(
        expr=model.sorption_factor_low_rh
        * WATER_MOLAR_MASS_G_MOL
        / (model.effective_formula_weight_low_rh + numerical_floor)
    )

    # Constraint: uptake at high humidity must exceed low humidity (positive half-cycle swing)
    model.positive_half_swing = pe.Constraint(
        expr=model.uptake_coeff_high_rh >= model.uptake_coeff_low_rh + 1e-9
    )

    # --- Water yield ---
    # Salt fraction of dry composite by mass: SL / (1 + SL)
    model.salt_fraction_in_composite = pe.Expression(
        expr=model.salt_to_polymer_ratio / (1.0 + model.salt_to_polymer_ratio)
    )
    model.uptake_swing = pe.Expression(
        expr=0.5
        * model.salt_fraction_in_composite
        * (model.uptake_coeff_high_rh - model.uptake_coeff_low_rh)
    )
    model.annual_water_yield_kg = pe.Expression(
        expr=float(cycles_per_year) * model.uptake_swing * DRY_COMPOSITE_MASS_KG
    )

    # --- Cost ---
    # Blend-averaged salt price, weighted by blend fractions
    model.blend_salt_price = pe.Expression(
        expr=pe.quicksum(
            model.blend_weight[i] * model.salt_price_usd_per_kg[i]
            for i in model.salt_idx
        )
    )
    # Hydrogel material cost per kg dry composite: salt + acrylamide (polymer) + fixed additives
    model.hydrogel_cost_per_kg = pe.Expression(
        expr=(model.blend_salt_price * model.salt_to_polymer_ratio + econ.c_acrylamide_usd_per_kg)
        / (1.0 + model.salt_to_polymer_ratio)
        + econ.c_additives_usd_per_kg_composite
    )
    model.annual_cost_usd = pe.Expression(
        expr=_annual_cost_expr(
            econ.capital_recovery_factor(), econ, model.hydrogel_cost_per_kg,
            C_DEVICE_USD, DRY_COMPOSITE_MASS_KG,
        )
    )
    model.lcow_usd_per_kg_water = pe.Expression(
        expr=model.annual_cost_usd / (econ.utilization_factor * (model.annual_water_yield_kg + 1e-9))
    )
    model.objective = pe.Objective(expr=model.lcow_usd_per_kg_water, sense=pe.minimize)
    return model


def extract_solution(model: pe.ConcreteModel) -> tuple[float, list[float], float]:
    """Read optimized values from a solved model: (salt_to_polymer_ratio, blend_weights, lcow)."""
    sl = float(pe.value(model.salt_to_polymer_ratio))
    weights = [float(pe.value(model.blend_weight[i])) for i in model.salt_idx]
    lcow = float(pe.value(model.lcow_usd_per_kg_water))
    return sl, weights, lcow
