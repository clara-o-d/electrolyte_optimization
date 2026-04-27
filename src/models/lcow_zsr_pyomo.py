"""Pyomo NLP for LCOW with ZSR mixed brine: variables ``SL``, simplex blend ``f``, Ipopt-friendly."""

from __future__ import annotations

import math

import pyomo.environ as pe

from src.materials.salts import get_salt
from src.models.lcow_sawh import SiteClimate, annual_op_expr
from src.optimization.economics import C_DEVICE_USD, DRY_COMPOSITE_MASS_KG, LCOEconomicParams
from src.optimization.sorption import MOLAR_MASS_H2O
from src.optimization.zsr_mixing import binary_molality_at_rh


def build_lcow_zsr_pyomo_model(
    site: SiteClimate,
    names: tuple[str, ...],
    econ: LCOEconomicParams,
    *,
    sl_lo: float = 0.05,
    sl_hi: float = 16.0,
    cycles_per_year: int = 365,
    eps: float = 1e-14,
) -> pe.ConcreteModel:
    """Minimal LCOW objective under ZSR mixing (same algebra as :mod:`src.optimization.zsr_mixing`).

    * **Variables:** ``SL``, ``f[i]`` with ``sum_i f[i] == 1``, ``f[i] \\in [0,1]``.
    * **Parameters:** binary reference molalities ``m*_j(rh_high)``, ``m*_j(rh_low)``, ``MW``, ``nu``, salt price.
    * **Expressions:** ``mf``, ``nu_e``, ``mw_e`` at high/low RH, ``t2`` via ``x_w(mf, nu_e, mw_e)``,
      ``B = t2 * M_w / mw_e``, ``yield0``, blend ``c_salt``, ``annual``, ``lcow``.
    * **Constraint:** ``B_high >= B_low + 1e-9`` to exclude inverted half-swing (mirrors scalar ``1e30`` guards).

    If binary molalities are non-finite for any salt in ``names``, returns a model marked ``infeasible``.
    """
    t_c = site.t_c
    rh_h, rh_l = site.rh_high, site.rh_low
    n = len(names)
    if n < 1:
        raise ValueError("names must be non-empty.")

    mstar_h: list[float] = []
    mstar_l: list[float] = []
    mw_l: list[float] = []
    nu_l: list[float] = []
    cs_l: list[float] = []
    for nm in names:
        get_salt(nm)
        msh = binary_molality_at_rh(nm, rh_h, t_c)
        msl = binary_molality_at_rh(nm, rh_l, t_c)
        rec = get_salt(nm)
        if not (
            math.isfinite(msh)
            and math.isfinite(msl)
            and msh >= 0.0
            and msl >= 0.0
        ):
            m = pe.ConcreteModel("lcow_zsr_infeasible")
            m.infeasible = True
            m.o = pe.Objective(expr=1e30, sense=pe.minimize)
            return m
        mstar_h.append(float(msh))
        mstar_l.append(float(msl))
        mw_l.append(float(rec.mw))
        nu_l.append(float(rec.nu))
        cs_l.append(float(rec.c_salt_usd_per_kg))

    m = pe.ConcreteModel("lcow_zsr")
    m.infeasible = False
    idx = range(n)
    init_f = 1.0 / float(n)

    def _idx_dict(xs: list[float]) -> dict[int, float]:
        return {i: xs[i] for i in idx}

    m.I = pe.RangeSet(0, n - 1)
    m.MSTAR_H = pe.Param(m.I, within=pe.NonNegativeReals, initialize=_idx_dict(mstar_h))
    m.MSTAR_L = pe.Param(m.I, within=pe.NonNegativeReals, initialize=_idx_dict(mstar_l))
    m.MW = pe.Param(m.I, within=pe.PositiveReals, initialize=_idx_dict(mw_l))
    m.NU = pe.Param(m.I, within=pe.PositiveReals, initialize=_idx_dict(nu_l))
    m.CSALT = pe.Param(m.I, within=pe.NonNegativeReals, initialize=_idx_dict(cs_l))

    m.f = pe.Var(m.I, bounds=(0.0, 1.0), initialize=init_f)
    m.SL = pe.Var(bounds=(sl_lo, sl_hi), initialize=4.0)

    def _simplex_rule(mod: pe.ConcreteModel):
        return pe.quicksum(mod.f[i] for i in mod.I) == 1.0

    m.con_simplex = pe.Constraint(rule=_simplex_rule)

    def _m_i(mod: pe.ConcreteModel, i: int, *, hi: bool) -> pe.Expression:
        ms = mod.MSTAR_H[i] if hi else mod.MSTAR_L[i]
        return mod.f[i] * ms

    m.m_i_h = pe.Expression(m.I, rule=lambda mod, i: _m_i(mod, int(i), hi=True))
    m.m_i_l = pe.Expression(m.I, rule=lambda mod, i: _m_i(mod, int(i), hi=False))

    m.s_m_h = pe.Expression(expr=pe.quicksum(m.m_i_h[i] for i in m.I))
    m.s_m_l = pe.Expression(expr=pe.quicksum(m.m_i_l[i] for i in m.I))

    m.m_s_g_h = pe.Expression(expr=pe.quicksum(m.m_i_h[i] * m.MW[i] for i in m.I))
    m.m_s_g_l = pe.Expression(expr=pe.quicksum(m.m_i_l[i] * m.MW[i] for i in m.I))

    m.mf_h = pe.Expression(expr=m.m_s_g_h / (m.m_s_g_h + 1000.0 + eps))
    m.mf_l = pe.Expression(expr=m.m_s_g_l / (m.m_s_g_l + 1000.0 + eps))

    m.nu_e_h = pe.Expression(
        expr=pe.quicksum(m.m_i_h[i] * m.NU[i] for i in m.I) / (m.s_m_h + eps)
    )
    m.nu_e_l = pe.Expression(
        expr=pe.quicksum(m.m_i_l[i] * m.NU[i] for i in m.I) / (m.s_m_l + eps)
    )

    m.mw_e_h = pe.Expression(expr=m.m_s_g_h / (m.s_m_h + eps))
    m.mw_e_l = pe.Expression(expr=m.m_s_g_l / (m.s_m_l + eps))

    m.n_w_h = pe.Expression(expr=(1.0 - m.mf_h) / MOLAR_MASS_H2O)
    m.n_s_h = pe.Expression(expr=m.mf_h / (m.mw_e_h + eps))
    m.den_h = pe.Expression(expr=m.n_w_h + m.nu_e_h * m.n_s_h)
    m.x_w_h = pe.Expression(expr=m.n_w_h / (m.den_h + eps))

    m.n_w_l = pe.Expression(expr=(1.0 - m.mf_l) / MOLAR_MASS_H2O)
    m.n_s_l = pe.Expression(expr=m.mf_l / (m.mw_e_l + eps))
    m.den_l = pe.Expression(expr=m.n_w_l + m.nu_e_l * m.n_s_l)
    m.x_w_l = pe.Expression(expr=m.n_w_l / (m.den_l + eps))

    m.t2_h = pe.Expression(expr=(m.x_w_h * m.nu_e_h) / (1.0 - m.x_w_h + eps))
    m.t2_l = pe.Expression(expr=(m.x_w_l * m.nu_e_l) / (1.0 - m.x_w_l + eps))

    m.B_h = pe.Expression(expr=m.t2_h * MOLAR_MASS_H2O / (m.mw_e_h + eps))
    m.B_l = pe.Expression(expr=m.t2_l * MOLAR_MASS_H2O / (m.mw_e_l + eps))

    m.con_swing = pe.Constraint(expr=m.B_h >= m.B_l + 1e-9)

    m.term1 = pe.Expression(expr=m.SL / (1.0 + m.SL))
    m.dB = pe.Expression(expr=m.B_h - m.B_l)
    m.dU = pe.Expression(expr=0.5 * m.term1 * m.dB)
    dm = DRY_COMPOSITE_MASS_KG
    m.yield0 = pe.Expression(expr=float(cycles_per_year) * m.dU * dm)

    m.c_blend = pe.Expression(expr=pe.quicksum(m.f[i] * m.CSALT[i] for i in m.I))
    c_a = econ.c_acrylamide_usd_per_kg
    c_add = econ.c_additives_usd_per_kg_composite
    m.c_hyd = pe.Expression(expr=(m.c_blend * m.SL + c_a) / (1.0 + m.SL) + c_add)

    f_c = econ.f_crf()
    c_dev = C_DEVICE_USD
    m.annual = pe.Expression(expr=annual_op_expr(f_c, econ, m.c_hyd, c_dev, dm))
    m.lcow_expr = pe.Expression(expr=m.annual / (econ.f_util * (m.yield0 + 1e-9)))
    m.o = pe.Objective(expr=m.lcow_expr, sense=pe.minimize)
    return m


def extract_zsr_solution(m: pe.ConcreteModel, names: tuple[str, ...]) -> tuple[float, list[float], float]:
    """After solve: ``(sl, f_list, lcow)``."""
    slv = float(pe.value(m.SL))
    fv = [float(pe.value(m.f[i])) for i in m.I]
    lv = float(pe.value(m.lcow_expr))
    return slv, fv, lv
