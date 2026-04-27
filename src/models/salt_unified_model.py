"""Single MINLP: pick salt (binary) and SL, with isotherm + uptake encoded as Pyomo constraints.

Equilibrium: ``RH = P_s(mf)`` (isotherm root), water activity, ``B = t2 * (M_w / MW_s)``,
half-swing water yield, and LCOW = annual / (f_util × gross_annual) — same as
:mod:`src.models.lcow_sawh`, but *all* in Pyomo so future inequality constraints can join.

*Salt choice* uses binary indicators and big-M to select one branch. Requires a
MINLP solver (e.g. Bonmin, Couenne) or a MindtPy run (Ipopt + MILP for OA).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pyomo.environ as pe

from src.materials.salts import SaltRecord, get_salt
from src.models.lcow_sawh import SiteClimate, annual_op_expr
from src.optimization.economics import C_DEVICE_USD, DRY_COMPOSITE_MASS_KG, LCOEconomicParams
from src.optimization.mf_equilibrium import equilibrate_salt_mf
from src.optimization.sorption import MOLAR_MASS_H2O

# Big-M: |R_iso| in [-1,1] at worst for normalized RH; add slack for P(mf) shape
M_ISOTHERM = 2.0
# |f_s - lcow| when the salt is not selected: keep ≥ max plausible LCOW spread
M_LCOW = 200.0

_SL_LO, _SL_HI = 0.05, 16.0


@dataclass(frozen=True, slots=True)
class UnifiedBuildResult:
    model: pe.ConcreteModel
    """Pyomo model with ``SL``, per-salt ``mf_h``/``mf_l``; may include binary ``y`` (multi-salt)."""

    salt_list: tuple[str, ...]
    """Feasible salts, fixed order (subset of candidates)."""

    mode: str
    """``\"nlp\"`` = one feasible branch (Ipopt). ``\"minlp\"`` = salt binaries + big-M reification."""


def _t2_of_mf(mf: pe.Var | pe.Param, rec: SaltRecord) -> pe.Expression:
    """``t2 = (x_w * nu) / (1 - x_w)`` with mole fraction ``x_w`` from ``mf``."""
    n_w = (1.0 - mf) / MOLAR_MASS_H2O
    n_s = mf / rec.mw
    den = n_w + float(rec.nu) * n_s
    x_w = n_w / den
    return (x_w * float(rec.nu)) / (1.0 - x_w)


def _B_from_mf(mf, rec: SaltRecord) -> pe.Expression:
    return _t2_of_mf(mf, rec) * (MOLAR_MASS_H2O / rec.mw)


# --- Isotherm residual R(mf) = 0  <=>  rh_target = P(mf) (same as ``mf_equilibrium``). ---


def isotherm_residual_nacl_expr(mf, rh: float) -> pe.Expression:
    A_0 = 0.9998
    A_1 = -0.5597
    A_2 = -0.332
    A_3 = -5.545
    A_4 = 5.863
    p = A_0 + A_1 * mf + A_2 * mf**2 + A_3 * mf**3 + A_4 * mf**4
    return rh - p


def isotherm_residual_mgcl2_expr(mf, rh: float) -> pe.Expression:
    A_4 = 186.32487108
    A_3 = -153.67496570
    A_2 = 38.21982328
    A_1 = -4.86704441
    A_0 = 1.16231287
    p = A_0 + A_1 * mf + A_2 * mf**2 + A_3 * mf**3 + A_4 * mf**4
    return rh - p


def isotherm_residual_li_ca_expr(
    mf, rh: float, t_c: float, p0: float, p1: float, p2: float, p3: float, p4: float, p5: float, p6: float, p7: float, p8: float, p9: float
) -> pe.Expression:
    """LiCl / CaCl2 template: ``rh = t1 * t2`` (IH thermal form)."""
    theta = (t_c + 273.15) / 647.0
    t1 = (
        1.0
        - (1.0 + (mf / p6) ** p7) ** p8
        - p9 * pe.exp(-((mf - 0.1) ** 2) / 0.005)
    )
    t2 = (
        2.0
        - (1.0 + (mf / p0) ** p1) ** p2
        + ((1.0 + (mf / p3) ** p4) ** p5 - 1.0) * theta
    )
    return rh - t1 * t2


def _li_params() -> tuple[float, ...]:
    return 0.28, 4.3, 0.60, 0.21, 5.10, 0.49, 0.362, -4.75, -0.40, 0.03


def _ca_params() -> tuple[float, ...]:
    return 0.31, 3.698, 0.60, 0.231, 4.584, 0.49, 0.478, -5.20, -0.40, 0.018


def _isotherm_pick(name: str, mf, rh: float, t_c: float) -> pe.Expression:
    if name == "NaCl":
        return isotherm_residual_nacl_expr(mf, rh)
    if name == "MgCl2":
        return isotherm_residual_mgcl2_expr(mf, rh)
    if name == "LiCl":
        return isotherm_residual_li_ca_expr(mf, rh, t_c, *_li_params())
    if name == "CaCl2":
        return isotherm_residual_li_ca_expr(mf, rh, t_c, *_ca_params())
    raise ValueError(f"Unified model has no isotherm for {name!r}")


def _mf_tight_bounds(name: str, rh: float, t_c: float) -> tuple[float, float] | None:
    m0 = equilibrate_salt_mf(name, rh, t_c)
    if not (math.isfinite(m0) and 0.0 < m0 < 1.0):
        return None
    lo = max(0.01, m0 - 0.2)
    hi = min(0.9, m0 + 0.2)
    if lo >= hi - 1e-8:
        return None
    return (lo, hi)


def feasible_salts_for_site(
    site: SiteClimate, candidates: tuple[str, ...] | list[str]
) -> tuple[str, ...]:
    """Salts for which isotherm roots exist at *both* RHs and ``dB>0`` at those roots."""
    t_c = site.t_c
    ok: list[str] = []
    for s in candidates:
        rec = get_salt(s)
        if s not in ("NaCl", "LiCl", "CaCl2", "MgCl2"):
            continue
        mfh0 = equilibrate_salt_mf(s, site.rh_high, t_c)
        mfl0 = equilibrate_salt_mf(s, site.rh_low, t_c)
        if not (math.isfinite(mfh0) and math.isfinite(mfl0)):
            continue
        if mfh0 <= 0.0 or mfl0 <= 0.0 or mfh0 >= 1.0 or mfl0 >= 1.0:
            continue
        from src.optimization.sorption import water_sorption_factor

        # dB = B(mfh) - B(mfl) in scalars; must be positive (half-swing)

        t2h_ = water_sorption_factor(site.rh_high, mfh0, rec.nu, rec.mw)
        t2l_ = water_sorption_factor(site.rh_low, mfl0, rec.nu, rec.mw)
        if not (math.isfinite(t2h_) and math.isfinite(t2l_)):
            continue
        Bh = t2h_ * (MOLAR_MASS_H2O / rec.mw)
        Bl = t2l_ * (MOLAR_MASS_H2O / rec.mw)
        if not (math.isfinite(Bh) and math.isfinite(Bl) and Bh > Bl + 1e-15):
            continue
        ok.append(s)
    return tuple(ok)


def build_unified_lcow_model(
    site: SiteClimate,
    econ: LCOEconomicParams,
    salt_list: tuple[str, ...] | list[str] | None = None,
    *,
    cycles_per_year: int = 365,
) -> UnifiedBuildResult | None:
    """:return: Model with binary salt, SL, and Pyomo isotherm + LCOW, or ``None`` if no salt works."""
    if salt_list is None:
        from src.materials.salts import CANDIDATE_SALTS

        salt_list = CANDIDATE_SALTS
    feasible = feasible_salts_for_site(site, tuple(salt_list))
    if not feasible:
        return None
    fset = set(feasible)
    # Preserve *salt_list* order when provided
    SS = tuple(s for s in salt_list if s in fset)
    t_c = site.t_c
    rh_h, rh_l = site.rh_high, site.rh_low
    m = pe.ConcreteModel("lcow_sawh_unified")
    m.SS = pe.Set(initialize=SS, ordered=True)
    m.SL = pe.Var(bounds=(_SL_LO, _SL_HI), initialize=4.0)

    # mf bounds: tight box around the equilibrium so inactive branches stay physical enough
    mb_h: dict[str, tuple[float, float]] = {}
    mb_l: dict[str, tuple[float, float]] = {}
    for s in m.SS:
        b_h = _mf_tight_bounds(s, rh_h, t_c)
        b_l = _mf_tight_bounds(s, rh_l, t_c)
        if b_h is None or b_l is None:
            return None
        mb_h[s], mb_l[s] = b_h, b_l

    m.mf_h = pe.Var(m.SS, bounds=mb_h)
    m.mf_l = pe.Var(m.SS, bounds=mb_l)

    def R_h(mod, s):
        return _isotherm_pick(s, mod.mf_h[s], rh_h, t_c)

    def R_l(mod, s):
        return _isotherm_pick(s, mod.mf_l[s], rh_l, t_c)

    m.term1 = pe.Expression(expr=m.SL / (1.0 + m.SL))
    f_c = econ.f_crf()
    c_dev = C_DEVICE_USD
    dm = DRY_COMPOSITE_MASS_KG

    def lcow_s_expr(s_name: str) -> pe.Expression:
        rec = get_salt(s_name)
        c_s = rec.c_salt_usd_per_kg
        c_a = econ.c_acrylamide_usd_per_kg
        c_add = econ.c_additives_usd_per_kg_composite
        dB = _B_from_mf(m.mf_h[s_name], rec) - _B_from_mf(m.mf_l[s_name], rec)
        dU = 0.5 * m.term1 * dB
        y0 = cycles_per_year * dU * dm
        c_h = (c_s * m.SL + c_a) / (1.0 + m.SL) + c_add
        ann = annual_op_expr(f_c, econ, c_h, c_dev, dm)
        return ann / (econ.f_util * (y0 + 1e-9))

    m.f_branch = pe.Expression(m.SS, rule=lambda mod, s: lcow_s_expr(s))

    if len(SS) == 1:
        s0 = SS[0]

        def iso_h_eq(mod, s):
            return R_h(mod, s) == 0.0

        def iso_l_eq(mod, s):
            return R_l(mod, s) == 0.0

        m.iso_h_eq = pe.Constraint(m.SS, rule=iso_h_eq)
        m.iso_l_eq = pe.Constraint(m.SS, rule=iso_l_eq)
        m.o = pe.Objective(expr=m.f_branch[s0], sense=pe.minimize)
        m.minlp_meta: dict[str, Any] = {"mode": "nlp", "salt_set": list(SS)}
        return UnifiedBuildResult(model=m, salt_list=SS, mode="nlp")

    m.lcow = pe.Var(bounds=(0.0, None), initialize=0.1)
    m.y = pe.Var(m.SS, within=pe.Binary)
    m.sum_one_salt = pe.Constraint(expr=sum(m.y[s] for s in m.SS) == 1)

    def iso_h_lo(mod, s):
        r = R_h(mod, s)
        return r >= -M_ISOTHERM * (1.0 - mod.y[s])

    def iso_h_hi(mod, s):
        r = R_h(mod, s)
        return r <= M_ISOTHERM * (1.0 - mod.y[s])

    def iso_l_lo(mod, s):
        r = R_l(mod, s)
        return r >= -M_ISOTHERM * (1.0 - mod.y[s])

    def iso_l_hi(mod, s):
        r = R_l(mod, s)
        return r <= M_ISOTHERM * (1.0 - mod.y[s])

    m.iso_h_lo = pe.Constraint(m.SS, rule=iso_h_lo)
    m.iso_h_hi = pe.Constraint(m.SS, rule=iso_h_hi)
    m.iso_l_lo = pe.Constraint(m.SS, rule=iso_l_lo)
    m.iso_l_hi = pe.Constraint(m.SS, rule=iso_l_hi)

    def lcow_link_lo(mod, s):
        return mod.f_branch[s] - mod.lcow <= M_LCOW * (1.0 - mod.y[s])

    def lcow_link_hi(mod, s):
        return mod.lcow - mod.f_branch[s] <= M_LCOW * (1.0 - mod.y[s])

    m.lc_lo = pe.Constraint(m.SS, rule=lcow_link_lo)
    m.lc_hi = pe.Constraint(m.SS, rule=lcow_link_hi)
    m.o = pe.Objective(expr=m.lcow, sense=pe.minimize)
    m.minlp_meta: dict[str, Any] = {"mode": "minlp", "salt_set": list(SS)}
    return UnifiedBuildResult(model=m, salt_list=SS, mode="minlp")
