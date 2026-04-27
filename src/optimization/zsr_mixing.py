"""Zdanovskii isopiestic (ZSR) mixing for brine: at fixed water activity, binary molalities mix linearly.

For each solute *i* at the mixture water activity, let :math:`m_i^*\\!` be the molality of
binary *i*--water in equilibrium. The Zdanovskii (isopiestic) relations use

  ``m_i,blend = f_i * m_i^*``,   with ``f_i >= 0``, ``sum_i f_i = 1``.

Then ``m_i,blend / m_i^* = f_i`` and ``sum_i m_i,blend / m_i^* = 1`` (one common statement of the rule).

**Optimization:** full **Pyomo** NLP in :mod:`src.models.lcow_zsr_pyomo` with **Ipopt** when
available; otherwise :func:`optimize_zsr_blend_and_sl` uses SciPy **SLSQP** on :func:`lcow_zsr_at_sl`.

Stokes--Robinson extensions (hydration numbers) are *not* added here: we use the same brine
mass--fraction construction as the single-salt path and blend the resulting t2/sorption terms
using one effective van't-Hoff and formula weight for the brine, consistent with
:mod:`src.optimization.sorption` at the mixed molality.

References: isopiestic mixing (Zdanovskii), Stokes--Robinson water--electrolyte (activity); see
e.g. Clegg, Brimblecombe, Wexler, *Atmospheric Environment* and AIM / E-AIM style notes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from src.materials.salts import SaltRecord, get_salt
from src.models.lcow_sawh import SiteClimate, UptakeCoefficients
from src.optimization.economics import C_DEVICE_USD, DRY_COMPOSITE_MASS_KG, LCOEconomicParams
from src.optimization.mf_equilibrium import equilibrate_salt_mf
from src.optimization.sorption import MOLAR_MASS_H2O, water_sorption_factor


def mf_to_molality_kg_mol(mf: float, mw_salt: float) -> float:
    """Binary molality (mol solute / kg H2O) from salt mass fraction ``mf`` and formula weight (g/mol)."""
    if not (0.0 < mf < 1.0) or not math.isfinite(mf):
        return float("nan")
    return 1000.0 * mf / (mw_salt * (1.0 - mf))


def molality_to_mf(m: float, mw_salt: float) -> float:
    """Salt mass fraction from binary molality and formula weight (g/mol)."""
    if m < 0.0 or not math.isfinite(m):
        return float("nan")
    return (m * mw_salt) / (1000.0 + m * mw_salt)


def binary_molality_at_rh(name: str, rh: float, t_c: float) -> float:
    """:func:`equilibrate_salt_mf` then ``mf_to_molality_kg_mol``."""
    mf = equilibrate_salt_mf(name, rh, t_c)
    if not (math.isfinite(mf) and 0.0 < mf < 1.0):
        return float("nan")
    rec = get_salt(name)
    return mf_to_molality_kg_mol(mf, rec.mw)


def _normalize_f(f: NDArray[np.float64], names: tuple[str, ...]) -> NDArray[np.float64]:
    s = float(np.sum(f))
    if s <= 0.0 or not math.isfinite(s):
        raise ValueError("Blend weights must be non-negative and not all zero.")
    out = np.clip(f / s, 0.0, 1.0)
    if out.shape[0] != len(names):
        raise ValueError("f length must match names.")
    return out


def zsr_brine_state(
    f: NDArray[np.float64],
    names: tuple[str, ...],
    rh: float,
    t_c: float,
) -> tuple[float, float, float]:
    """
    :return: ``(mf_mix, nu_effective, mw_effective)`` for the ZSR brine at ``rh`` (same a_w = RH in this model).
    """
    w = _normalize_f(f, names)
    m_star = np.empty(len(names), dtype=np.float64)
    mw_arr = np.empty(len(names), dtype=np.float64)
    nu_arr = np.empty(len(names), dtype=np.float64)
    for i, nm in enumerate(names):
        raw = binary_molality_at_rh(nm, rh, t_c)
        m_star[i] = raw if w[i] > 1e-15 else 0.0
        rec = get_salt(nm)
        mw_arr[i] = rec.mw
        nu_arr[i] = float(rec.nu)
    m_i = np.empty_like(w)
    for i in range(len(names)):
        m_i[i] = w[i] * m_star[i] if w[i] > 1e-15 else 0.0
    s_m = float(np.sum(m_i))
    if s_m <= 0.0 or not math.isfinite(s_m):
        return float("nan"), float("nan"), float("nan")
    m_s_g = float(np.sum(m_i * mw_arr))
    mf = m_s_g / (m_s_g + 1000.0)
    nu_e = float(np.sum(m_i * nu_arr) / s_m)
    mw_e = m_s_g / s_m
    if not (math.isfinite(mf) and 0.0 < mf < 1.0 and math.isfinite(nu_e) and math.isfinite(mw_e)):
        return float("nan"), float("nan"), float("nan")
    return float(mf), float(nu_e), float(mw_e)


def uptake_B_zsr(
    f: NDArray[np.float64],
    names: tuple[str, ...],
    site: SiteClimate,
) -> UptakeCoefficients | None:
    """``B = t2 * M_w / MW`` at each ``site`` RH, using the ZSR brine (different ``mf``, ``nu``, ``MW`` at high vs low)."""
    t_c = site.t_c
    mf_h, nuh, mwh = zsr_brine_state(f, names, site.rh_high, t_c)
    mf_l, nul, mwl = zsr_brine_state(f, names, site.rh_low, t_c)
    if not all(map(math.isfinite, (mf_h, mf_l, mwh, mwl, nuh, nul))):
        return None
    t2h = water_sorption_factor(site.rh_high, mf_h, nuh, mwh)
    t2l = water_sorption_factor(site.rh_low, mf_l, nul, mwl)
    if not (math.isfinite(t2h) and math.isfinite(t2l)):
        return None
    b_high = t2h * (MOLAR_MASS_H2O / mwh)
    b_low = t2l * (MOLAR_MASS_H2O / mwl)
    if not (b_high > b_low + 1e-15 and math.isfinite(b_high) and math.isfinite(b_low)):
        return None
    return UptakeCoefficients(b_high=b_high, b_low=b_low)


def blend_salt_price_usd_per_kg(
    f: NDArray[np.float64],
    names: tuple[str, ...],
) -> float:
    w = _normalize_f(f, names)
    total = 0.0
    for i, nm in enumerate(names):
        rec: SaltRecord = get_salt(nm)
        total += w[i] * rec.c_salt_usd_per_kg
    return float(total)


def lcow_zsr_at_sl(
    f: NDArray[np.float64],
    names: tuple[str, ...],
    site: SiteClimate,
    econ: LCOEconomicParams,
    sl: float,
    *,
    cycles_per_year: int = 365,
) -> float:
    """Levelized $/kg water for a ZSR brine with blend vector ``f`` and salt-to-polymer ratio ``sl``."""
    if sl <= 0.0 or not math.isfinite(sl):
        return 1e30
    uc = uptake_B_zsr(f, names, site)
    if uc is None or uc.b_high <= uc.b_low + 1e-15:
        return 1e30
    c_blend = blend_salt_price_usd_per_kg(f, names)
    c_a, c_add = econ.c_acrylamide_usd_per_kg, econ.c_additives_usd_per_kg_composite
    f_c = econ.f_crf()
    c_dev, dm = C_DEVICE_USD, DRY_COMPOSITE_MASS_KG
    dB = float(uc.b_high - uc.b_low)
    term1 = sl / (1.0 + sl)
    dU = 0.5 * term1 * dB
    y0 = cycles_per_year * dU * dm
    c_h = (c_blend * sl + c_a) / (1.0 + sl) + c_add
    rep = c_h * dm / econ.tau_hyd_years
    ann = f_c * econ.f_toti * c_dev + rep + econ.f_mlc * econ.f_toti * c_dev + econ.C_energy_usd_per_year
    if y0 <= 0.0 or not math.isfinite(y0) or not math.isfinite(ann):
        return 1e30
    return float(ann / (econ.f_util * (y0 + 1e-9)))


@dataclass(slots=True)
class ZSROptResult:
    """Result of :func:`optimize_zsr_blend_and_sl` (Ipopt or SciPy)."""

    best_sl: float
    best_f: NDArray[np.float64]
    best_lcow: float
    names: tuple[str, ...]
    nfev: int
    success: bool
    message: str
    backend: str = "unknown"


def optimize_zsr_blend_and_sl(
    site: SiteClimate,
    names: tuple[str, ...],
    econ: LCOEconomicParams | None = None,
    *,
    sl_lo: float = 0.05,
    sl_hi: float = 16.0,
    ipopt_tee: bool = False,
    ipopt_print_level: int | None = None,
) -> ZSROptResult:
    """Optimize ``(SL, f)`` on the ZSR simplex.

    **Primary:** full Pyomo model (:func:`src.models.lcow_zsr_pyomo.build_lcow_zsr_pyomo_model`) + **Ipopt**
    when the executable is available.

    **Fallback:** SciPy **SLSQP** on :func:`lcow_zsr_at_sl` if Ipopt is missing or does not terminate optimally.
    """
    from pyomo.opt import SolverFactory, TerminationCondition

    from src.models.lcow_zsr_pyomo import build_lcow_zsr_pyomo_model, extract_zsr_solution
    from src.optimization.solve import ipopt_available

    if len(names) < 1:
        raise ValueError("Need at least one salt name for ZSR optimization.")
    econ = econ or LCOEconomicParams()
    n = len(names)
    for nm in names:
        get_salt(nm)

    m = build_lcow_zsr_pyomo_model(site, names, econ, sl_lo=sl_lo, sl_hi=sl_hi)
    if getattr(m, "infeasible", False):
        return ZSROptResult(
            float("nan"),
            np.zeros(n, dtype=np.float64),
            1e30,
            names,
            0,
            False,
            "No finite binary molalities for this site/salt list.",
            backend="none",
        )

    if ipopt_available():
        opt = SolverFactory("ipopt", validate=False)
        opt.options["max_iter"] = 500
        opt.options["tol"] = 1e-7
        if ipopt_print_level is not None:
            opt.options["print_level"] = int(ipopt_print_level)
        res = opt.solve(m, tee=ipopt_tee, load_solutions=True)
        sol = res.solver
        s0 = sol[0] if isinstance(sol, (list, tuple)) and sol else sol
        tc = getattr(s0, "termination_condition", None) if s0 is not None else None
        ok = tc in (TerminationCondition.optimal, TerminationCondition.locallyOptimal)
        niter = int(getattr(s0, "iterations", None) or 0) if s0 is not None else 0
        msg = str(getattr(s0, "message", "") or getattr(s0, "Message", "") or "")
        if ok:
            slv, fv, lv = extract_zsr_solution(m, names)
            if math.isfinite(slv) and math.isfinite(lv) and lv < 0.99 * 1e30:
                return ZSROptResult(
                    slv,
                    _normalize_f(np.array(fv, dtype=np.float64), names),
                    float(lv),
                    names,
                    niter,
                    True,
                    msg or "ipopt",
                    backend="ipopt",
                )

    # --- SciPy fallback ---
    from scipy.optimize import minimize

    def pack(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        sl = float(x[0])
        if n == 1:
            return sl, np.ones(1, dtype=np.float64)
        rest = x[1:n]
        last = 1.0 - float(np.sum(rest))
        f = np.append(rest, max(0.0, last))
        return sl, f

    def fun(x: NDArray[np.float64]) -> float:
        sl, f = pack(x)
        if sl < sl_lo or sl > sl_hi or np.any(f < 0) or not math.isfinite(sl):
            return 1e12
        return lcow_zsr_at_sl(f, names, site, econ, sl)

    x0 = np.empty(n, dtype=np.float64)
    x0[0] = 4.0
    if n > 1:
        x0[1:] = 1.0 / n

    bounds = [(sl_lo, sl_hi)]
    for _ in range(1, n):
        bounds.append((0.0, 1.0))

    cons = []
    if n > 1:
        def _simplex_ineq(x: NDArray[np.float64]) -> float:
            return 1.0 - float(np.sum(x[1:n]))

        cons.append({"type": "ineq", "fun": _simplex_ineq})

    res = minimize(
        fun,
        x0,
        method="SLSQP",
        bounds=tuple(bounds),
        constraints=cons or (),
        options={"maxiter": 400, "ftol": 1e-9},
    )
    nfev = int(getattr(res, "nfev", 0) or 0)
    if not res.success or res.x is None:
        return ZSROptResult(
            float("nan"),
            np.zeros(n, dtype=np.float64),
            1e30,
            names,
            nfev,
            False,
            str(res.message),
            backend="scipy_slsqp",
        )
    rsl, rf = pack(np.array(res.x, dtype=np.float64))
    lcv = lcow_zsr_at_sl(rf, names, site, econ, rsl)
    if not (math.isfinite(lcv) and lcv < 0.99 * 1e30):
        return ZSROptResult(
            rsl,
            rf,
            1e30,
            names,
            nfev,
            False,
            f"unbounded or failed LCOW: {res.message!s}",
            backend="scipy_slsqp",
        )
    return ZSROptResult(
        rsl,
        _normalize_f(rf, names),
        float(lcv),
        names,
        nfev,
        bool(res.success),
        str(res.message),
        backend="scipy_slsqp",
    )
