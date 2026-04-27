"""Tests for LCOW economics, mf, sorption, and Pyomo model (optional Ipopt)."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

import pyomo.environ as pe

from src.models.lcow_sawh import (
    SiteClimate,
    build_lcow_sawh_model,
    lcow_at_sl,
    uptake_B_coefficients,
)
from src.optimization.economics import (
    LCOEconomicParams,
    hydrogel_cost_usd_per_kg_composite,
    lcow_usd_per_kg_water,
)
from src.models.salt_unified_model import feasible_salts_for_site
from src.materials.salts import CANDIDATE_SALTS
from src.optimization.solve import (
    _best_lcow_among_feasible_scipy,
    _min_sl_lcow_one_salt_scipy,
    optimize_salt_and_sl,
    solve_lcow_nlp,
)
from src.optimization.mf_equilibrium import mf_NaCl
from src.optimization.sorption import water_mole_fraction_from_mf, water_sorption_factor, salt_uptake_U
from src.materials.salt_prices import load_salt_prices_usd_per_kg, salt_price_data_path
from src.materials.salts import get_salt


def test_salt_price_workbook_loads():
    p = load_salt_prices_usd_per_kg()
    assert salt_price_data_path().is_file()
    assert abs(p["NaCl"] - 0.045) < 1e-6
    assert abs(p["CaCl2"] - 0.17) < 1e-6
    assert abs(p["MgCl2"] - 0.01985) < 1e-8
    assert abs(p["LiCl"] - 8.5) < 1e-6


def test_salt_table_matches_workbook_prices():
    assert abs(get_salt("NaCl").c_salt_usd_per_kg - 0.045) < 1e-6
    assert abs(get_salt("LiCl").c_salt_usd_per_kg - 8.5) < 1e-6


def test_f_crf_numeric():
    p = LCOEconomicParams(f_wacc=0.1, L_years=10)
    f = p.f_crf()
    # Annuity: i(1+i)^L / ((1+i)^L - 1) for 10% / 10yr ≈ 0.1627
    assert 0.16 < f < 0.17


def test_mf_nacl_bracket() -> None:
    # Mid-range isotherm (quartic fit)
    m = mf_NaCl(0.80)
    assert 0.05 < m < 0.35


def test_lcow_denominator_convention() -> None:
    p = LCOEconomicParams(f_util=0.8)
    assert (
        abs(
            lcow_usd_per_kg_water(100.0, p.f_util, 50.0)
            - 100.0 / (0.8 * 50.0)
        )
        < 1e-9
    )


def test_sorption_factor_identity() -> None:
    # At high dilution x_w ~ 1, t2 = nu * x_w / (1-x_w) can blow; use mid mf
    mf = 0.1
    nu = 2
    mw = 58.44
    xw = water_mole_fraction_from_mf(mf, nu, mw)
    rh = 0.8
    g = water_sorption_factor(rh, mf, nu, mw)
    assert g > 0.0
    g2 = (xw * float(nu)) / (1.0 - xw)
    assert abs(g - g2) < 1e-9


def test_uptake_B_coefficients_for_valid_site() -> None:
    site = SiteClimate(0.9, 0.78)
    uc = uptake_B_coefficients("NaCl", site)
    assert uc is not None
    assert uc.b_high > uc.b_low


def test_hydrogel_cost_more_salt_when_sl_higher() -> None:
    # With cheap salt ($0.5) vs pricier acrylamide, higher SL lowers $/kg composite
    c_lo_sl = hydrogel_cost_usd_per_kg_composite(0.5, 1.0)
    c_hi_sl = hydrogel_cost_usd_per_kg_composite(0.5, 8.0)
    assert c_hi_sl < c_lo_sl
    # If salt is more expensive than acrylamide, higher SL raises material cost
    c_lo_sl2 = hydrogel_cost_usd_per_kg_composite(2.0, 1.0)
    c_hi_sl2 = hydrogel_cost_usd_per_kg_composite(2.0, 8.0)
    assert c_hi_sl2 > c_lo_sl2


def test_salt_uptake_NaCl() -> None:
    rec = get_salt("NaCl")
    u = salt_uptake_U(4.0, 0.85, rec)
    assert math.isfinite(u) and u > 0.0


# Climate valid for all v1 candidate salts (intersection of RH windows)
_MULTI_SALT_CLIMATE = SiteClimate(0.9, 0.78)
# For end-to-end `optimize_salt_and_sl` without a mandatory ASL MINLP (CI): only one feasible salt
_SINGLE_SALT_UNIFIED_CLIMATE = SiteClimate(0.55, 0.2)


def test_lcow_at_sl_matches_pyomo_expression() -> None:
    site = SiteClimate(0.9, 0.78)
    econ = LCOEconomicParams()
    m = build_lcow_sawh_model("NaCl", site, econ)
    assert not m.infeasible
    m.SL.set_value(2.0)
    pyo = pe.value(m.lcow_expr)
    sc = lcow_at_sl("NaCl", site, econ, 2.0)
    assert abs(pyo - sc) < 1e-5 * max(1.0, abs(pyo))


def test_solve_lcow_nlp_converges_with_reasonable_lcow() -> None:
    site = _MULTI_SALT_CLIMATE
    econ = LCOEconomicParams()
    lc, sl, m, sinfo = solve_lcow_nlp("NaCl", site, econ)
    assert "backend" in sinfo
    assert not m.infeasible
    assert math.isfinite(lc) and math.isfinite(sl)
    assert 0.05 <= sl <= 16.0
    # Finite, positive LCOW; upper bound guards only against numerical blow-up
    assert 0.0 < lc < 100_000.0
    # Scalar objective at returned optimum should agree with solve
    assert abs(lcow_at_sl("NaCl", site, econ, sl) - lc) < 1e-3 * max(1.0, abs(lc))


def test_optimize_salt_and_sl_end_to_end() -> None:
    """One unified solve (NLP: single feasible salt); does not require Bonmin/Couenne."""
    out = optimize_salt_and_sl(_SINGLE_SALT_UNIFIED_CLIMATE, econ=LCOEconomicParams())
    assert out.solved_unified and out.unified_mode == "nlp"
    assert math.isfinite(out.best_lcow) and out.best_lcow < 1e6
    assert math.isfinite(out.best_sl) and 0.05 <= out.best_sl <= 16.0
    assert out.best_salt
    assert abs(out.per_salt[out.best_salt]["lcow"] - out.best_lcow) < 1e-6 * max(1.0, abs(out.best_lcow))
    for name, d in out.per_salt.items():
        if d["infeasible"]:
            continue
        assert math.isfinite(d["sl"]) and 0.05 <= d["sl"] <= 16.0
        if math.isfinite(d["lcow"]):
            assert 0.0 < d["lcow"] < 100_000.0, name


@pytest.mark.skipif(
    not __import__("pyomo.opt", fromlist=["SolverFactory"]).SolverFactory("ipopt").available(
        False
    ),
    reason="Ipopt not installed",
)
def test_best_lcow_among_feasible_matches_independent_per_salt_minima() -> None:
    """Reference: min over feasible salts of (1D optimum LCOW) == shared helper."""
    site = _MULTI_SALT_CLIMATE
    econ = LCOEconomicParams()
    feasible = frozenset(feasible_salts_for_site(site, CANDIDATE_SALTS))
    assert len(feasible) >= 2
    trials: list[tuple[str, float, float]] = []
    for s in sorted(feasible):
        lc, sl, ok, _msg = _min_sl_lcow_one_salt_scipy(s, site, econ)
        assert ok
        trials.append((s, sl, lc))
    s_ref, sl_ref, lc_ref = min(trials, key=lambda t: t[2])
    s_h, sl_h, lc_h, _info = _best_lcow_among_feasible_scipy(site, econ, feasible)
    assert s_ref == s_h
    assert abs(sl_ref - sl_h) < 1e-6 * max(1.0, abs(sl_ref))
    assert abs(lc_ref - lc_h) < 1e-5 * max(1.0, abs(lc_ref))


def test_multisalt_uses_exact_scipy_enum_when_asl_minlp_fails() -> None:
    """Simulate ApplicationError from Bonmin; optimization must still return the nested min LCOW."""
    site = _MULTI_SALT_CLIMATE
    econ = LCOEconomicParams()
    feasible = frozenset(feasible_salts_for_site(site, CANDIDATE_SALTS))
    assert len(feasible) >= 2
    ref_lc = min(_min_sl_lcow_one_salt_scipy(s, site, econ)[0] for s in feasible)
    with patch(
        "src.optimization.solve._first_available_minlp_solver",
        return_value="bonmin",
    ), patch(
        "src.optimization.solve._all_available_minlp_solvers",
        return_value=("bonmin",),
    ), patch(
        "src.optimization.solve._solve_unified_minlp",
        return_value=(
            None,
            {
                "unified": "minlp",
                "last_error": "ApplicationError('Solver (asl) did not exit normally')",
            },
        ),
    ):
        out = optimize_salt_and_sl(site, econ=econ)
    assert out.solved_unified
    assert out.unified_mode == "minlp_scipy_enum"
    assert abs(out.best_lcow - ref_lc) < 1e-5 * max(1.0, abs(ref_lc))
    assert out.best_salt in feasible
    assert "minlp_asl_attempts" in (out.per_salt[out.best_salt].get("solver") or {})


@pytest.mark.skipif(
    not __import__("pyomo.opt", fromlist=["SolverFactory"]).SolverFactory("ipopt").available(
        False
    ),
    reason="Ipopt not installed",
)
def test_ipopt_solves_lcow() -> None:
    from pyomo.opt import SolverFactory

    m = build_lcow_sawh_model(
        "NaCl", SiteClimate(0.9, 0.78), LCOEconomicParams()
    )
    assert not m.infeasible
    res = SolverFactory("ipopt").solve(m, tee=False)
    from pyomo.opt import TerminationCondition

    assert res.solver.termination_condition in (
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
    )
    assert math.isfinite(pe.value(m.SL))
    v = pe.value(m.lcow_expr)
    assert 0.0 < v < 1e6
