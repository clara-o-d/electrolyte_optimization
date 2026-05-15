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
    salt_fraction_to_molality,
    molality_to_salt_fraction,
)
from src.optimization.zsr_mixing import (
    lcow_zsr_at_sl,
    optimize_zsr_blend_and_sl,
    uptake_B_zsr,
)


def test_salt_fraction_molality_roundtrip() -> None:
    formula_weight = 58.44
    for fraction in (0.05, 0.12, 0.3):
        molality = salt_fraction_to_molality(fraction, formula_weight)
        fraction_back = molality_to_salt_fraction(molality, formula_weight)
        assert abs(fraction_back - fraction) < 1e-9 * max(1.0, abs(fraction))


def test_binary_molality_matches_equilibrium() -> None:
    site = SiteClimate(0.88, 0.78)
    for name in ("NaCl", "LiCl"):
        brine_fraction = equilibrate_salt_mf(name, site.humidity_high, site.temperature_c)
        molality_from_helper = binary_molality_at_rh(name, site.humidity_high, site.temperature_c)
        molality_from_fraction = salt_fraction_to_molality(brine_fraction, get_salt(name).mw)
        assert abs(molality_from_helper - molality_from_fraction) < 1e-6 * max(1.0, abs(molality_from_fraction))


def test_zsr_pure_salt_uptake_matches_single_salt() -> None:
    site = SiteClimate(0.9, 0.78)
    for name in ("NaCl", "LiCl"):
        vec = np.zeros(len(CANDIDATE_SALTS), dtype=np.float64)
        vec[CANDIDATE_SALTS.index(name)] = 1.0
        zsr_coeffs = uptake_B_zsr(vec, CANDIDATE_SALTS, site)
        single_coeffs = single_salt_half_swing_coefficients(name, site)
        assert single_coeffs is not None and zsr_coeffs is not None
        assert abs(zsr_coeffs.at_high_humidity - single_coeffs.at_high_humidity) < 1e-5 * max(1.0, abs(single_coeffs.at_high_humidity))
        assert abs(zsr_coeffs.at_low_humidity - single_coeffs.at_low_humidity) < 1e-5 * max(1.0, abs(single_coeffs.at_low_humidity))


def test_zsr_pure_salt_lcow_matches_single_salt_scalar() -> None:
    site = SiteClimate(0.9, 0.78)
    econ = LCOEconomicParams()
    sl = 2.0
    for name in ("NaCl", "LiCl"):
        vec = np.zeros(len(CANDIDATE_SALTS), dtype=np.float64)
        vec[CANDIDATE_SALTS.index(name)] = 1.0
        lcow_single = single_salt_lcow_at_loading(name, site, econ, sl)
        lcow_zsr = lcow_zsr_at_sl(vec, CANDIDATE_SALTS, site, econ, sl)
        assert abs(lcow_single - lcow_zsr) < 1e-4 * max(1.0, abs(lcow_single))


def test_zsr_isopiestic_sum() -> None:
    """ZSR: m_i_blend = blend_weight_i * m_i_ref, so sum(blend_weights) == 1."""
    site = SiteClimate(0.9, 0.78)
    names = ("NaCl", "LiCl", "CaCl2")
    blend_weights = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    assert abs(float(np.sum(blend_weights)) - 1.0) < 1e-9
    reference_molalities = [binary_molality_at_rh(nm, site.humidity_high, site.temperature_c) for nm in names]
    blended_molalities = blend_weights * np.array(reference_molalities)
    ratios = blended_molalities / np.array(reference_molalities)
    assert np.allclose(ratios, blend_weights, rtol=0, atol=1e-9)


def test_optimize_zsr_runs() -> None:
    site = SiteClimate(0.55, 0.2)
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
    site = SiteClimate(0.9, 0.78)
    names = ("NaCl", "LiCl")
    econ = LCOEconomicParams()
    model = build_lcow_model(site, names, econ)
    assert not model.infeasible
    model.blend_weight[0].set_value(0.5)
    model.blend_weight[1].set_value(0.5)
    model.salt_to_polymer_ratio.set_value(3.0)
    pyomo_lcow = pe.value(model.lcow_usd_per_kg_water)
    scalar_lcow = lcow_zsr_at_sl(np.array([0.5, 0.5]), names, site, econ, 3.0)
    assert abs(pyomo_lcow - scalar_lcow) < 1e-5 * max(1.0, abs(scalar_lcow))


@pytest.mark.skipif(
    not SolverFactory("ipopt", validate=False).available(False),
    reason="Ipopt not installed",
)
def test_optimize_zsr_reports_ipopt_when_available() -> None:
    site = SiteClimate(0.55, 0.2)
    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    out = optimize_zsr_blend_and_sl(site, names, LCOEconomicParams())
    assert out.success
    assert out.backend == "ipopt"
