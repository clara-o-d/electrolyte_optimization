"""Pyomo model: minimize LCOW over ``SL`` for a fixed (salt, site climate) pair."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pyomo.environ as pe

from src.materials.salts import get_salt
from src.optimization.economics import C_DEVICE_USD, DRY_COMPOSITE_MASS_KG, LCOEconomicParams
from src.optimization.mf_equilibrium import equilibrate_salt_mf
from src.optimization.sorption import MOLAR_MASS_H2O, water_sorption_factor


@dataclass(frozen=True, slots=True)
class SiteClimate:
    """Relative humidity bounds (0–1): high = uptake, low = desorption reference."""

    rh_high: float
    rh_low: float
    t_c: float = 25.0


@dataclass(frozen=True, slots=True)
class UptakeCoefficients:
    """``B = t2 * MW_w / MW_s`` at each branch RH so ``U = (SL/(1+SL)) * B``."""

    b_high: float
    b_low: float


def uptake_B_coefficients(name: str, site: SiteClimate) -> UptakeCoefficients | None:
    rec = get_salt(name)
    mfh = equilibrate_salt_mf(name, site.rh_high, site.t_c)
    mfl = equilibrate_salt_mf(name, site.rh_low, site.t_c)
    if not (math.isfinite(mfh) and math.isfinite(mfl)) or mfh <= 0.0 or mfl <= 0.0 or mfh >= 1.0 or mfl >= 1.0:
        return None
    t2h = water_sorption_factor(site.rh_high, mfh, rec.nu, rec.mw)
    t2l = water_sorption_factor(site.rh_low, mfl, rec.nu, rec.mw)
    if not (math.isfinite(t2h) and math.isfinite(t2l)):
        return None
    f = MOLAR_MASS_H2O / rec.mw
    return UptakeCoefficients(b_high=t2h * f, b_low=t2l * f)


def build_lcow_sawh_model(
    salt_name: str,
    site: SiteClimate,
    econ: LCOEconomicParams,
    *,
    sl_lo: float = 0.05,
    sl_hi: float = 16.0,
    cycles_per_year: int = 365,
) -> pe.ConcreteModel:
    """:return: model with ``Objective`` to minimize; infeasible site/salt => huge objective."""
    m = pe.ConcreteModel("lcow_sawh")
    rec = get_salt(salt_name)
    c_s = rec.c_salt_usd_per_kg
    c_a = econ.c_acrylamide_usd_per_kg
    c_add = econ.c_additives_usd_per_kg_composite
    uc = uptake_B_coefficients(salt_name, site)

    m.SL = pe.Var(bounds=(sl_lo, sl_hi), initialize=4.0)
    m.term1 = pe.Expression(expr=m.SL / (1.0 + m.SL))
    f_c = econ.f_crf()
    c_dev = C_DEVICE_USD
    dm = DRY_COMPOSITE_MASS_KG

    if uc is None or uc.b_high <= uc.b_low + 1e-15:
        m.lcow_expr = pe.Param(initialize=1e30)
        m.o = pe.Objective(expr=m.lcow_expr, sense=pe.minimize)
        m.infeasible = True
        return m

    dB = float(uc.b_high - uc.b_low)
    m.dU = pe.Expression(expr=0.5 * m.term1 * dB)
    m.yield0 = pe.Expression(expr=cycles_per_year * m.dU * dm)

    m.c_hyd = pe.Expression(expr=(c_s * m.SL + c_a) / (1.0 + m.SL) + c_add)
    m.annual = pe.Expression(
        expr=annual_op_expr(f_c, econ, m.c_hyd, c_dev, dm)
    )
    m.lcow_expr = pe.Expression(
        expr=m.annual
        / (econ.f_util * (m.yield0 + 1e-9))
    )
    m.o = pe.Objective(expr=m.lcow_expr, sense=pe.minimize)
    m.infeasible = False
    return m


def annual_op_expr(
    f_c: float,
    econ: LCOEconomicParams,
    c_hyd: pe.Expression,
    c_device: float,
    dry_mass: float,
) -> pe.Expression:
    rep = c_hyd * dry_mass / econ.tau_hyd_years
    return (
        f_c * econ.f_toti * c_device
        + rep
        + econ.f_mlc * econ.f_toti * c_device
        + econ.C_energy_usd_per_year
    )


def lcow_at_sl(
    salt_name: str,
    site: SiteClimate,
    econ: LCOEconomicParams,
    sl: float,
    *,
    cycles_per_year: int = 365,
) -> float:
    """Scalar LCOW ($/kg water) at ``sl``; mirrors :func:`build_lcow_sawh_model` algebra."""
    rec = get_salt(salt_name)
    uc = uptake_B_coefficients(salt_name, site)
    if uc is None or uc.b_high <= uc.b_low + 1e-15:
        return 1e30
    c_s, c_a, c_add = rec.c_salt_usd_per_kg, econ.c_acrylamide_usd_per_kg, econ.c_additives_usd_per_kg_composite
    f_c = econ.f_crf()
    c_dev = C_DEVICE_USD
    dm = DRY_COMPOSITE_MASS_KG
    dB = float(uc.b_high - uc.b_low)
    term1 = sl / (1.0 + sl)
    dU = 0.5 * term1 * dB
    y0 = cycles_per_year * dU * dm
    c_h = (c_s * sl + c_a) / (1.0 + sl) + c_add
    rep = c_h * dm / econ.tau_hyd_years
    ann = f_c * econ.f_toti * c_dev + rep + econ.f_mlc * econ.f_toti * c_dev + econ.C_energy_usd_per_year
    if y0 <= 0.0 or not math.isfinite(y0):
        return 1e30
    return float(ann / (econ.f_util * (y0 + 1e-9)))
