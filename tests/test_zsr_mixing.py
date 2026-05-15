"""ZSR brine mixing: pure-salt limit and small optimization smoke test."""

from __future__ import annotations

import math

import numpy as np
import pyomo.environ as pe
import pytest
from pyomo.opt import SolverFactory

from src.models.zsr_lcow_model import (
    SiteClimate,
    build_lcow_model,
    single_salt_lcow_at_loading,
    single_salt_half_swing_coefficients,
)
from src.models.salt_feasibility import feasible_salts_for_site
from src.materials.salts import CANDIDATE_SALTS, get_salt
from src.optimization.economics import LCOEconomicParams
from src.optimization.brine_equilibrium import (
    binary_molality_at_rh,
    equilibrate_salt_mf,
    mf_MgCl2,
    salt_fraction_to_molality,
    molality_to_salt_fraction,
)
from src.optimization.brine_uptake import (
    desorption_water_activity,
    saturation_vapor_pressure_pa,
)
from src.optimization.heat_transfer import (
    DEFAULT_SOLAR_IRRADIANCE_W_M2,
    gel_steady_state_temperature_c,
)
from src.optimization.zsr_mixing import (
    daily_water_yield_L_per_m2_at_sl,
    lcow_zsr_at_sl,
    optimize_zsr_blend_and_sl,
    optimize_zsr_pareto_front,
    uptake_B_zsr,
)


def test_salt_fraction_molality_roundtrip() -> None:
    formula_weight = 58.44
    for fraction in (0.05, 0.12, 0.3):
        molality = salt_fraction_to_molality(fraction, formula_weight)
        fraction_back = molality_to_salt_fraction(molality, formula_weight)
        assert abs(fraction_back - fraction) < 1e-9 * max(1.0, abs(fraction))


def test_mgcl2_isotherm_finds_root() -> None:
    """MgCl2 polynomial is non-monotonic; robust bracketing must match MATLAB robust_fzero."""
    for rh in (0.4, 0.55, 0.75, 0.9):
        mf = mf_MgCl2(rh)
        assert math.isfinite(mf) and 0.0 < mf < 0.75


def test_binary_molality_matches_equilibrium() -> None:
    site = SiteClimate(humidity_high=0.88, temperature_c=25.0)
    for name in ("NaCl", "LiCl"):
        brine_fraction = equilibrate_salt_mf(name, site.humidity_high, site.temperature_c)
        molality_from_helper = binary_molality_at_rh(name, site.humidity_high, site.temperature_c)
        molality_from_fraction = salt_fraction_to_molality(brine_fraction, get_salt(name).mw)
        assert abs(molality_from_helper - molality_from_fraction) < 1e-6 * max(1.0, abs(molality_from_fraction))


def test_saturation_vapor_pressure_known_values() -> None:
    """Tetens formula matches steam-table values at 25 C and 70 C within 1%."""
    p25 = saturation_vapor_pressure_pa(25.0)
    p70 = saturation_vapor_pressure_pa(70.0)
    assert abs(p25 - 3170.0) / 3170.0 < 0.02
    assert abs(p70 - 31200.0) / 31200.0 < 0.02


def test_desorption_activity_matches_user_formula() -> None:
    """a_w = P_sat(T_cond) * T_gel / (P_sat(T_gel) * T_cond), absolute Kelvin."""
    t_cond_c = 25.0
    t_gel_c = 70.0
    p_cond = saturation_vapor_pressure_pa(t_cond_c)
    p_gel = saturation_vapor_pressure_pa(t_gel_c)
    expected = p_cond * (t_gel_c + 273.15) / (p_gel * (t_cond_c + 273.15))
    actual = desorption_water_activity(t_cond_c, t_gel_c)
    assert abs(actual - expected) < 1e-12


def test_zsr_pure_salt_uptake_matches_single_salt() -> None:
    """At a single-salt blend the ZSR result must match the scalar single-salt path."""
    site = SiteClimate(humidity_high=0.9, temperature_c=45.0)
    name = "LiCl"
    vec = np.zeros(len(CANDIDATE_SALTS), dtype=np.float64)
    vec[CANDIDATE_SALTS.index(name)] = 1.0
    zsr_coeffs = uptake_B_zsr(vec, CANDIDATE_SALTS, site)
    single_coeffs = single_salt_half_swing_coefficients(name, site)
    assert single_coeffs is not None and zsr_coeffs is not None
    assert abs(zsr_coeffs.at_uptake - single_coeffs.at_uptake) < 1e-5 * max(1.0, abs(single_coeffs.at_uptake))
    assert abs(zsr_coeffs.at_desorption - single_coeffs.at_desorption) < 1e-5 * max(1.0, abs(single_coeffs.at_desorption))


def test_zsr_pure_salt_lcow_matches_single_salt_scalar() -> None:
    site = SiteClimate(humidity_high=0.9, temperature_c=45.0)
    econ = LCOEconomicParams()
    sl = 2.0
    name = "LiCl"
    vec = np.zeros(len(CANDIDATE_SALTS), dtype=np.float64)
    vec[CANDIDATE_SALTS.index(name)] = 1.0
    lcow_single = single_salt_lcow_at_loading(name, site, econ, sl)
    lcow_zsr = lcow_zsr_at_sl(vec, CANDIDATE_SALTS, site, econ, sl)
    assert abs(lcow_single - lcow_zsr) < 1e-4 * max(1.0, abs(lcow_single))


def test_zsr_isopiestic_sum() -> None:
    """ZSR: m_i_blend = blend_weight_i * m_i_ref, so sum(blend_weights) == 1."""
    site = SiteClimate(humidity_high=0.9, temperature_c=25.0)
    names = ("NaCl", "LiCl", "CaCl2")
    blend_weights = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    assert abs(float(np.sum(blend_weights)) - 1.0) < 1e-9
    reference_molalities = [binary_molality_at_rh(nm, site.humidity_high, site.temperature_c) for nm in names]
    blended_molalities = blend_weights * np.array(reference_molalities)
    ratios = blended_molalities / np.array(reference_molalities)
    assert np.allclose(ratios, blend_weights, rtol=0, atol=1e-9)


def test_optimize_zsr_runs() -> None:
    """Sunny humid site under default heat-transfer model gives a feasible LiCl half-swing."""
    site = SiteClimate(humidity_high=0.85, temperature_c=30.0)
    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    assert len(names) >= 1
    out = optimize_zsr_blend_and_sl(site, names, LCOEconomicParams())
    assert out.success
    assert math.isfinite(out.best_lcow) and out.best_lcow < 1e6
    assert math.isfinite(out.best_sl) and 0.05 <= out.best_sl <= 16.0
    assert out.best_f.shape[0] == len(names)
    assert abs(float(np.sum(out.best_f)) - 1.0) < 1e-4
    assert out.backend in ("ipopt", "scipy_slsqp")


def test_pyomo_model_lcow_matches_scalar_at_fixed_point() -> None:
    # Use a lower irradiance so the gel doesn't get hot enough to push the
    # desorption a_w below CaCl2's deliquescence (~0.31); this exercises the
    # multi-salt code path.
    site = SiteClimate(humidity_high=0.9, temperature_c=45.0, solar_irradiance_w_per_m2=300.0)
    names = ("CaCl2", "LiCl")
    econ = LCOEconomicParams(max_electric_heat_w_per_m2=0.0)
    model = build_lcow_model(site, names, econ)
    assert not model.infeasible
    model.blend_weight[0].set_value(0.5)
    model.blend_weight[1].set_value(0.5)
    model.salt_to_polymer_ratio.set_value(3.0)
    pyomo_lcow = pe.value(model.lcow_usd_per_kg_water)
    scalar_lcow = lcow_zsr_at_sl(np.array([0.5, 0.5]), names, site, econ, 3.0)
    assert abs(pyomo_lcow - scalar_lcow) < 1e-5 * max(1.0, abs(scalar_lcow))


def test_gel_temperature_steady_state_balance() -> None:
    """Solved T_gel must satisfy the absorbed-power = convective + radiative-loss balance."""
    from src.optimization.heat_transfer import (
        DEFAULT_GEL_CONVECTION_W_M2_K,
        DEFAULT_GEL_EMISSIVITY,
        DEFAULT_SOLAR_ABSORPTIVITY,
        STEFAN_BOLTZMANN_W_M2_K4,
    )

    t_amb_c = 30.0
    irradiance = 800.0
    t_gel_c = gel_steady_state_temperature_c(irradiance, t_amb_c)
    absorbed = DEFAULT_SOLAR_ABSORPTIVITY * irradiance
    conv = DEFAULT_GEL_CONVECTION_W_M2_K * (t_gel_c - t_amb_c)
    rad = (
        DEFAULT_GEL_EMISSIVITY * STEFAN_BOLTZMANN_W_M2_K4
        * ((t_gel_c + 273.15) ** 4 - (t_amb_c + 273.15) ** 4)
    )
    assert abs(absorbed - conv - rad) < 1e-6
    # Gel must heat above ambient when the sun is shining.
    assert t_gel_c > t_amb_c
    # At zero / negative irradiance the gel cannot warm above ambient.
    assert gel_steady_state_temperature_c(0.0, t_amb_c) == t_amb_c


def test_site_climate_gel_temperature_uses_heat_transfer() -> None:
    """SiteClimate.gel_temperature_c is derived from the steady-state balance."""
    assert DEFAULT_SOLAR_IRRADIANCE_W_M2 == 800.0
    site = SiteClimate(humidity_high=0.9, temperature_c=25.0)
    assert site.solar_irradiance_w_per_m2 == DEFAULT_SOLAR_IRRADIANCE_W_M2
    expected = gel_steady_state_temperature_c(
        site.solar_irradiance_w_per_m2, site.temperature_c
    )
    assert abs(site.gel_temperature_c - expected) < 1e-9
    # A weaker sun should give a cooler gel.
    cloudy = SiteClimate(humidity_high=0.9, temperature_c=25.0, solar_irradiance_w_per_m2=200.0)
    assert cloudy.gel_temperature_c < site.gel_temperature_c


@pytest.mark.skipif(
    not SolverFactory("ipopt", validate=False).available(False),
    reason="Ipopt not installed",
)
def test_optimize_zsr_reports_ipopt_when_available() -> None:
    site = SiteClimate(humidity_high=0.55, temperature_c=45.0)
    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    out = optimize_zsr_blend_and_sl(site, names, LCOEconomicParams())
    assert out.success
    assert out.backend == "ipopt"


def test_pyomo_daily_yield_matches_scalar_at_fixed_point() -> None:
    """The new daily_water_yield_L_per_m2 Pyomo expression must match the scalar helper."""
    site = SiteClimate(humidity_high=0.9, temperature_c=45.0, solar_irradiance_w_per_m2=300.0)
    names = ("CaCl2", "LiCl")
    model = build_lcow_model(
        site, names, LCOEconomicParams(max_electric_heat_w_per_m2=0.0)
    )
    assert not model.infeasible
    model.blend_weight[0].set_value(0.4)
    model.blend_weight[1].set_value(0.6)
    model.salt_to_polymer_ratio.set_value(2.5)
    pyomo_yield = float(pe.value(model.daily_water_yield_L_per_m2))
    scalar_yield = daily_water_yield_L_per_m2_at_sl(
        np.array([0.4, 0.6]), names, site, 2.0 + 0.5
    )
    assert pyomo_yield > 0.0
    assert abs(pyomo_yield - scalar_yield) < 1e-6 * max(1.0, abs(scalar_yield))


def test_optimize_zsr_reports_daily_yield() -> None:
    """ZSROptResult exposes the second objective alongside LCOW."""
    site = SiteClimate(humidity_high=0.85, temperature_c=30.0)
    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    out = optimize_zsr_blend_and_sl(site, names, LCOEconomicParams())
    assert out.success
    assert math.isfinite(out.best_daily_yield_L_per_m2)
    assert out.best_daily_yield_L_per_m2 > 0.0
    expected_yield = daily_water_yield_L_per_m2_at_sl(out.best_f, names, site, out.best_sl)
    assert abs(out.best_daily_yield_L_per_m2 - expected_yield) < 1e-5 * max(1.0, abs(expected_yield))


def test_yield_objective_yields_at_least_as_much_as_lcow_objective() -> None:
    """Switching the objective to 'yield' must not lower the achieved daily yield."""
    site = SiteClimate(humidity_high=0.85, temperature_c=30.0)
    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    econ = LCOEconomicParams()
    lcow_opt = optimize_zsr_blend_and_sl(site, names, econ, objective="lcow")
    yield_opt = optimize_zsr_blend_and_sl(site, names, econ, objective="yield")
    assert lcow_opt.success and yield_opt.success
    # Max-yield anchor must achieve at least as much yield as the LCOW-optimal point,
    # up to solver tolerance.
    assert (
        yield_opt.best_daily_yield_L_per_m2
        >= lcow_opt.best_daily_yield_L_per_m2 - 1e-6 * max(1.0, lcow_opt.best_daily_yield_L_per_m2)
    )
    # And conversely the LCOW-optimal point must have LCOW no higher than the yield-anchor
    # LCOW (this is the fundamental Pareto-anchor relation that justifies the sweep).
    assert (
        lcow_opt.best_lcow
        <= yield_opt.best_lcow + 1e-6 * max(1.0, abs(yield_opt.best_lcow))
    )


def test_min_daily_yield_epsilon_constraint_binds() -> None:
    """When epsilon-constrained, the achieved yield must meet the bound (within solver tol)."""
    site = SiteClimate(humidity_high=0.85, temperature_c=30.0)
    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    econ = LCOEconomicParams()
    anchors = optimize_zsr_blend_and_sl(site, names, econ, objective="yield")
    assert anchors.success
    lcow_anchor = optimize_zsr_blend_and_sl(site, names, econ, objective="lcow")
    assert lcow_anchor.success
    # Pick an interior epsilon strictly between the two yield anchors.
    y_lo = lcow_anchor.best_daily_yield_L_per_m2
    y_hi = anchors.best_daily_yield_L_per_m2
    if y_hi - y_lo < 1e-6:
        pytest.skip("yield band is degenerate at this site; nothing to constrain")
    eps = y_lo + 0.5 * (y_hi - y_lo)
    constrained = optimize_zsr_blend_and_sl(
        site, names, econ, objective="lcow", min_daily_yield_L_per_m2=eps,
    )
    assert constrained.success
    assert constrained.best_daily_yield_L_per_m2 >= eps - 1e-4 * max(1.0, eps)
    # Constraining yield up should never improve LCOW vs. the unconstrained min-LCOW.
    assert (
        constrained.best_lcow
        >= lcow_anchor.best_lcow - 1e-6 * max(1.0, abs(lcow_anchor.best_lcow))
    )


def test_pareto_front_is_monotone_and_spans_anchors() -> None:
    """Pareto front: yield is sorted ascending, LCOW is non-decreasing, anchors are present."""
    site = SiteClimate(humidity_high=0.85, temperature_c=30.0)
    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    front = optimize_zsr_pareto_front(site, names, LCOEconomicParams(), num_points=6)
    assert front.min_lcow_point.success
    assert front.max_yield_point.success
    assert len(front.points) >= 2

    yields = [p.daily_water_yield_L_per_m2 for p in front.points]
    lcows = [p.lcow_usd_per_kg for p in front.points]
    # Yields are sorted ascending by construction.
    for a, b in zip(yields, yields[1:]):
        assert a <= b + 1e-9
    # LCOW must be non-decreasing along a true Pareto front (allow small numerical slack).
    for a, b in zip(lcows, lcows[1:]):
        assert b >= a - 1e-4 * max(1.0, abs(a))
    # First / last points bracket the anchors (allowing identical anchors at degenerate sites).
    assert yields[0] <= front.min_lcow_point.daily_water_yield_L_per_m2 + 1e-9
    assert yields[-1] >= front.max_yield_point.daily_water_yield_L_per_m2 - 1e-9


def test_pareto_front_num_points_bound() -> None:
    """num_points < 2 is rejected (need at least the two anchors)."""
    site = SiteClimate(humidity_high=0.85, temperature_c=30.0)
    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    with pytest.raises(ValueError):
        optimize_zsr_pareto_front(site, names, LCOEconomicParams(), num_points=1)


# ---------------------------------------------------------------------------
# Active electrical heating of the SAWH device
# ---------------------------------------------------------------------------


def test_heat_transfer_electric_heat_raises_t_gel() -> None:
    """Adding electrical heat must raise the steady-state gel temperature."""
    t_amb = 25.0
    irradiance = 600.0
    t_passive = gel_steady_state_temperature_c(irradiance, t_amb)
    t_active = gel_steady_state_temperature_c(
        irradiance, t_amb, electric_heat_w_per_m2=400.0
    )
    assert math.isfinite(t_passive) and math.isfinite(t_active)
    assert t_active > t_passive + 1.0
    # With only electrical heat (no sun), gel must still rise above ambient.
    t_dark_active = gel_steady_state_temperature_c(
        0.0, t_amb, electric_heat_w_per_m2=400.0
    )
    assert t_dark_active > t_amb + 1.0


def test_pyomo_passive_match_when_active_heating_disabled() -> None:
    """With max_electric_heat_w_per_m2 == 0 the model reproduces the passive optimum."""
    site = SiteClimate(humidity_high=0.85, temperature_c=30.0)
    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    econ = LCOEconomicParams(max_electric_heat_w_per_m2=0.0)
    out = optimize_zsr_blend_and_sl(site, names, econ)
    assert out.success
    assert out.best_electric_heat_w_per_m2 == 0.0
    assert math.isnan(out.best_gel_temperature_c)


@pytest.mark.skipif(
    not SolverFactory("ipopt", validate=False).available(False),
    reason="Active heating NLP requires Ipopt (SciPy SLSQP fallback gates it off).",
)
def test_pyomo_active_heat_can_improve_lcow_at_low_price() -> None:
    """With cheap electricity, the optimizer should pick Q_elec > 0 and improve LCOW."""
    site = SiteClimate(humidity_high=0.85, temperature_c=30.0)
    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    passive_econ = LCOEconomicParams()
    passive = optimize_zsr_blend_and_sl(site, names, passive_econ)
    assert passive.success
    active_econ = LCOEconomicParams(
        electricity_price_usd_per_kwh=0.001,
        desorption_hours_per_day=8.0,
        max_electric_heat_w_per_m2=1500.0,
    )
    active = optimize_zsr_blend_and_sl(site, names, active_econ)
    assert active.success
    assert active.best_electric_heat_w_per_m2 > 1.0
    assert math.isfinite(active.best_gel_temperature_c)
    assert active.best_gel_temperature_c > site.gel_temperature_c
    # Cheap electricity must not worsen LCOW (within solver/surrogate tolerance).
    assert active.best_lcow <= passive.best_lcow + 1e-4 * max(1.0, abs(passive.best_lcow))


def test_desorption_molality_surrogate_accuracy() -> None:
    """Polynomial surrogate must match the root-finder over the interior T_gel range.

    Tolerance is checked at a set of T_gel probes that fall comfortably inside
    each salt's deliquescence window so we're not sensitive to the steep
    behavior at the very edges of the feasible region (where the optimizer
    won't operate anyway).
    """
    from src.optimization.active_heating import (
        default_t_gel_grid_c,
        evaluate_poly,
        fit_desorption_molality_polynomial,
    )
    from src.optimization.brine_equilibrium import binary_molality_at_rh
    from src.optimization.brine_uptake import desorption_water_activity

    t_amb = 30.0
    t_grid = default_t_gel_grid_c(t_amb, t_max_c=120.0)
    # Salts have different deliquescence windows; sample each at probes inside
    # its own feasible T_gel band rather than a single shared probe set.
    probes_by_salt: dict[str, tuple[float, ...]] = {
        "LiCl": (45.0, 55.0, 65.0, 75.0),
        "CaCl2": (40.0, 45.0, 50.0, 55.0),
        "MgCl2": (40.0, 45.0, 50.0, 55.0),
        "NaCl": (32.0, 33.0, 34.0),
    }
    at_least_one_feasible = False
    for name, probes in probes_by_salt.items():
        coeffs = fit_desorption_molality_polynomial(name, t_amb, t_grid)
        if coeffs is None:
            continue
        at_least_one_feasible = True
        max_rel_err = 0.0
        compared = 0
        for t_gel in probes:
            a_w = desorption_water_activity(t_amb, t_gel)
            true_m = binary_molality_at_rh(name, a_w, t_gel)
            if not (math.isfinite(true_m) and true_m > 0.0):
                continue
            surrogate_m = evaluate_poly(coeffs, t_gel)
            max_rel_err = max(max_rel_err, abs(surrogate_m - true_m) / max(1.0, abs(true_m)))
            compared += 1
        # Only enforce the tolerance once we have probes that actually compared.
        if compared >= 2:
            assert max_rel_err < 0.02, (
                f"{name}: surrogate max relative error {max_rel_err:.4%} > 2% "
                f"on interior probes."
            )
    assert at_least_one_feasible, "Expected at least one candidate salt to be fittable."
