"""ZSR brine mixing: pure-salt limit and small optimization smoke test."""

from __future__ import annotations

import math

import numpy as np
import pyomo.environ as pe
import pytest
from pyomo.opt import SolverFactory

from src.models.lcow_zsr_pyomo import build_lcow_zsr_pyomo_model
from src.models.lcow_sawh import SiteClimate, lcow_at_sl, uptake_B_coefficients
from src.models.salt_unified_model import feasible_salts_for_site
from src.materials.salts import CANDIDATE_SALTS
from src.optimization.economics import LCOEconomicParams
from src.optimization.mf_equilibrium import equilibrate_salt_mf
from src.optimization.zsr_mixing import (
    binary_molality_at_rh,
    lcow_zsr_at_sl,
    mf_to_molality_kg_mol,
    molality_to_mf,
    optimize_zsr_blend_and_sl,
    uptake_B_zsr,
)
from src.materials.salts import get_salt


def test_mf_molality_roundtrip() -> None:
    mw = 58.44
    for _mf in (0.05, 0.12, 0.3):
        m = mf_to_molality_kg_mol(_mf, mw)
        mf2 = molality_to_mf(m, mw)
        assert abs(mf2 - _mf) < 1e-9 * max(1.0, abs(_mf))


def test_binary_molality_matches_mf() -> None:
    site = SiteClimate(0.88, 0.78)
    for name in ("NaCl", "LiCl"):
        mf = equilibrate_salt_mf(name, site.rh_high, site.t_c)
        m_code = binary_molality_at_rh(name, site.rh_high, site.t_c)
        m_ref = mf_to_molality_kg_mol(mf, get_salt(name).mw)
        assert abs(m_code - m_ref) < 1e-6 * max(1.0, abs(m_ref))


def test_zsr_pure_salt_uptake_matches_single_salt() -> None:
    site = SiteClimate(0.9, 0.78)
    for name in ("NaCl", "LiCl"):
        vec = np.zeros(len(CANDIDATE_SALTS), dtype=np.float64)
        idx = CANDIDATE_SALTS.index(name)
        vec[idx] = 1.0
        u_z = uptake_B_zsr(vec, CANDIDATE_SALTS, site)
        u0 = uptake_B_coefficients(name, site)
        assert u0 is not None and u_z is not None
        assert abs(u_z.b_high - u0.b_high) < 1e-5 * max(1.0, abs(u0.b_high))
        assert abs(u_z.b_low - u0.b_low) < 1e-5 * max(1.0, abs(u0.b_low))


def test_zsr_pure_salt_lcow_matches_lcow_at_sl() -> None:
    site = SiteClimate(0.9, 0.78)
    econ = LCOEconomicParams()
    sl = 2.0
    for name in ("NaCl", "LiCl"):
        vec = np.zeros(len(CANDIDATE_SALTS), dtype=np.float64)
        vec[CANDIDATE_SALTS.index(name)] = 1.0
        a = lcow_at_sl(name, site, econ, sl)
        b = lcow_zsr_at_sl(vec, CANDIDATE_SALTS, site, econ, sl)
        assert abs(a - b) < 1e-4 * max(1.0, abs(a))


def test_zsr_isopiestic_sum() -> None:
    """ZSR: m_i,blend = f_i m_i* => m_i,blend / m_i* = f_i, sum f_i = 1."""
    site = SiteClimate(0.9, 0.78)
    names = ("NaCl", "LiCl", "CaCl2")
    f = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    t_c = site.t_c
    s = 0.0
    for i, nm in enumerate(names):
        m_star = binary_molality_at_rh(nm, site.rh_high, t_c)
        s += f[i]  # f_i
    assert abs(s - 1.0) < 1e-9
    m_star = [binary_molality_at_rh(nm, site.rh_high, t_c) for nm in names]
    m_blend = f * np.array(m_star)
    ratios = m_blend / np.array(m_star)
    assert np.allclose(ratios, f, rtol=0, atol=1e-9)


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


def test_pyomo_zsr_lcow_matches_scalar_at_fixed_point() -> None:
    site = SiteClimate(0.9, 0.78)
    names = ("NaCl", "LiCl")
    econ = LCOEconomicParams()
    m = build_lcow_zsr_pyomo_model(site, names, econ)
    assert not m.infeasible
    m.f[0].set_value(0.5)
    m.f[1].set_value(0.5)
    m.SL.set_value(3.0)
    pyv = pe.value(m.lcow_expr)
    sc = lcow_zsr_at_sl(np.array([0.5, 0.5]), names, site, econ, 3.0)
    assert abs(pyv - sc) < 1e-5 * max(1.0, abs(sc))


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
