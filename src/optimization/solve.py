"""Unified Pyomo NLP or ASL MINLP; if every ASL MINLP fails, exact per-feasible-salt 1-D minimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pyomo.environ as pe
from pyomo.opt import SolverFactory, TerminationCondition
from scipy import optimize

from src.materials.salts import CANDIDATE_SALTS, get_salt
from src.models.lcow_sawh import (
    SiteClimate,
    build_lcow_sawh_model,
    lcow_at_sl,
)
from src.models.salt_unified_model import build_unified_lcow_model, feasible_salts_for_site
from src.optimization.economics import LCOEconomicParams

# Must match :func:`build_lcow_sawh_model` defaults
_SL_LO, _SL_HI = 0.05, 16.0


def ipopt_available() -> bool:
    # validate=False: avoid Pyomo log spam when the executable is missing (common in CI)
    return SolverFactory("ipopt", validate=False).available(False)


@lru_cache(maxsize=1)
def _first_available_minlp_solver() -> str | None:
    """``\"bonmin\"`` or ``\"couenne\"`` on ``PATH``, else ``None`` (cached for the process)."""
    for name in _MINLP_CANDIDATES:
        opt = SolverFactory(name, validate=False)
        if opt.available(False):
            return name
    return None


_MINLP_CANDIDATES: tuple[str, ...] = ("bonmin", "couenne")


def _all_available_minlp_solvers() -> tuple[str, ...]:
    """Every ASL MINLP on ``PATH``, stable order (Bonmin first, then Couenne)."""
    return tuple(
        n
        for n in _MINLP_CANDIDATES
        if SolverFactory(n, validate=False).available(False)
    )


@dataclass(slots=True)
class LCOOPTResult:
    best_salt: str
    best_sl: float
    best_lcow: float
    per_salt: dict[str, dict[str, Any]]
    solved_unified: bool = False
    """``True`` if :func:`build_unified_lcow_model` + one MIP/NLP run produced the result."""

    unified_mode: str | None = None
    """``\"nlp\"``, ``\"minlp\"`` (Bonmin/Couenne), or ``\"minlp_scipy_enum\"`` (see docs)."""


def _extract_unified(
    m: pe.ConcreteModel,
    salt_list: tuple[str, ...],
    *,
    mode: str,
) -> tuple[str, float, float]:
    s_opt: str | None = None
    if mode == "minlp" and hasattr(m, "y"):
        for s in salt_list:
            v = pe.value(m.y[s])
            if v is not None and float(v) > 0.5:
                s_opt = s
                break
    elif mode == "nlp" and len(salt_list) == 1:
        s_opt = salt_list[0]
    if s_opt is None and salt_list:
        s_opt = salt_list[0]
    slv = float(pe.value(m.SL))
    if mode == "nlp" and s_opt is not None:
        lv = float(pe.value(m.f_branch[s_opt]))
    else:
        lv = float(pe.value(m.lcow))
    return s_opt or "none", slv, lv


def _per_salt_table_joint_sl(
    site: SiteClimate,
    econ: LCOEconomicParams,
    salts: tuple[str, ...],
    joint_sl: float,
    joint_lcow: float,
    best: str,
    feasible: frozenset[str],
) -> dict[str, dict[str, Any]]:
    per: dict[str, dict[str, Any]] = {}
    for s in salts:
        if s not in feasible:
            per[s] = {
                "lcow": 1e30,
                "sl": float("nan"),
                "infeasible": True,
                "at_joint_sl": True,
            }
        else:
            klc = float(lcow_at_sl(s, site, econ, joint_sl))
            per[s] = {
                "lcow": klc,
                "sl": joint_sl,
                "infeasible": False,
                "at_joint_sl": True,
                "solver": {
                    "note": "LCOW at unified SL; best row = solver objective"
                },
            }
    if best in per and math.isfinite(joint_lcow):
        per[best]["lcow"] = float(joint_lcow)
    return per


def _unified_termination_ok(res: Any) -> bool:
    s0 = _first_solver_subresult(res)
    if s0 is None:
        return False
    tc = getattr(s0, "termination_condition", None)
    return tc in (TerminationCondition.optimal, TerminationCondition.locallyOptimal)


def _solve_unified_nlp(
    m: pe.ConcreteModel,
    *,
    ipopt_tee: bool,
    ipopt_print_level: int | None,
) -> tuple[Any, dict[str, Any]]:
    opt = SolverFactory("ipopt", validate=False)
    opt.options["max_iter"] = 500
    opt.options["tol"] = 1e-7
    if ipopt_print_level is not None:
        opt.options["print_level"] = int(ipopt_print_level)
    res = opt.solve(m, tee=ipopt_tee, load_solutions=True)
    return res, {**_ipopt_solve_info(res), "unified": "nlp"}


def _solve_unified_minlp(
    m: pe.ConcreteModel, *, tee: bool, solver_name: str
) -> tuple[Any | None, dict[str, Any]]:
    """ASL MINLP (Bonmin / Couenne). Call only when :func:`_first_available_minlp_solver` is non-``None``."""
    info: dict[str, Any] = {"unified": "minlp", "minlp_solver": solver_name}
    opt = SolverFactory(solver_name, validate=False)
    if not opt.available(False):
        info["note"] = "solver was available at discovery but not now"
        return None, info
    if solver_name == "couenne" and hasattr(opt, "options"):
        opt.options["print_level"] = 5
    if solver_name == "bonmin" and hasattr(opt, "options"):
        # Some conda-forge / macOS ASL builds are touchy; keep logs quiet to reduce I/O.
        try:
            b = opt.options.get("bonmin")
            if not isinstance(b, dict):
                opt.options["bonmin"] = b = {}
            b.setdefault("bb_log_level", 0)
        except (TypeError, KeyError, AttributeError):
            pass
    try:
        res = opt.solve(m, tee=tee, load_solutions=True)
    except Exception as exc:
        info["last_error"] = repr(exc)
        return None, info
    info["backend"] = solver_name
    return res, info


def _min_sl_lcow_one_salt_scipy(
    salt: str,
    site: SiteClimate,
    econ: LCOEconomicParams,
) -> tuple[float, float, bool, str]:
    r = optimize.minimize_scalar(
        lambda sl: lcow_at_sl(salt, site, econ, sl),
        bounds=(_SL_LO, _SL_HI),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    ok = bool(r.success and math.isfinite(r.x) and math.isfinite(r.fun))
    msg = str(getattr(r, "message", ""))
    if ok:
        return float(r.fun), float(r.x), True, msg
    return float("inf"), float("nan"), False, msg


def _best_lcow_among_feasible_scipy(
    site: SiteClimate,
    econ: LCOEconomicParams,
    feasible: frozenset[str],
) -> tuple[str, float, float, dict[str, Any]]:
    """``min`` over feasible salts of ``min`` SL LCOW — same optimum as the big-M MINLP here.

    Used when every ASL MINLP (Bonmin/Couenne) throws or returns a non-optimal status.
    """
    best_s = "none"
    best_sl = float("nan")
    best_lc = float("inf")
    per_s: dict[str, Any] = {}
    for s in sorted(feasible):
        lc, sl, ok, msg = _min_sl_lcow_one_salt_scipy(s, site, econ)
        per_s[s] = {"lcow": lc, "sl": sl, "ok": ok, "message": msg}
        if ok and lc < best_lc - 1e-15 * max(1.0, abs(lc)):
            best_lc, best_sl, best_s = lc, sl, s
    if best_s == "none" or not math.isfinite(best_lc):
        raise RuntimeError(
            "Multi-salt fallback: bounded minimization failed for every feasible salt. "
            f"Trials: {per_s!r}"
        )
    return best_s, best_sl, best_lc, {
        "backend": "scipy_per_feasible_salt",
        "per_salt_trials": per_s,
        "unified": "minlp_scipy_enum",
        "note": "Exact min_s min_sl LCOW; used when ASL MINLP is unavailable to finish optimally.",
    }


def _unified_nlp_scipy(
    s0: str,
    site: SiteClimate,
    econ: LCOEconomicParams,
    salts: tuple[str, ...],
    feasible: frozenset[str],
) -> LCOOPTResult:
    """One ``minimize_scalar`` on :func:`lcow_at_sl` for the *only* feasible salt (unified NLP path)."""
    r = optimize.minimize_scalar(
        lambda sl: lcow_at_sl(s0, site, econ, sl),
        bounds=(_SL_LO, _SL_HI),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    sinfo: dict[str, Any] = {
        "backend": "scipy_minimize_scalar_bounded",
        "success": bool(r.success),
        "nfev": int(getattr(r, "nfev", 0) or 0),
        "unified": "nlp",
        "note": "Ipopt not used; single 1D solve for the one feasible salt",
    }
    if not (r.success and math.isfinite(r.x) and math.isfinite(r.fun)):
        raise RuntimeError(
            "Unified single-salt NLP: Ipopt is unavailable and SciPy 1D minimization did not "
            f"succeed: {r.message!s}"
        )
    sl, lc = float(r.x), float(r.fun)
    per = _per_salt_table_joint_sl(site, econ, salts, sl, lc, s0, feasible)
    if s0 in per:
        per[s0]["solver"] = sinfo
    return LCOOPTResult(
        s0,
        sl,
        lc,
        per,
        solved_unified=True,
        unified_mode="nlp",
    )


def _solve_lcow_scipy(
    m: pe.ConcreteModel,
    salt_name: str,
    site: SiteClimate,
    econ: LCOEconomicParams,
) -> tuple[float, float, dict[str, Any]] | None:
    """1-D bounded minimization; return ``(lcow, sl, info)`` on success."""
    r = optimize.minimize_scalar(
        lambda sl: lcow_at_sl(salt_name, site, econ, sl),
        bounds=(_SL_LO, _SL_HI),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    nfev = int(getattr(r, "nfev", 0) or 0)
    nit = int(getattr(r, "nit", 0) or 0)
    info: dict[str, Any] = {
        "backend": "scipy_minimize_scalar_bounded",
        "success": bool(r.success),
        "nfev": nfev,
        "nit": nit,
        "message": str(getattr(r, "message", "")),
    }
    if not r.success or not math.isfinite(r.x) or not math.isfinite(r.fun):
        return None
    sl_opt = float(r.x)
    m.SL.set_value(sl_opt)
    lv = lcow_at_sl(salt_name, site, econ, sl_opt)
    if not math.isfinite(lv):
        return None
    return (lv, sl_opt, info)


def _first_solver_subresult(res: Any) -> Any:
    if res is None or not getattr(res, "solver", None):
        return None
    s = res.solver
    if isinstance(s, (list, tuple)) and s:
        return s[0]
    return s


def _ipopt_solve_info(res: Any) -> dict[str, Any]:
    """Extract a small, printable subset of a Pyomo ``SolverResults`` object."""
    out: dict[str, Any] = {"backend": "ipopt"}
    s0 = _first_solver_subresult(res)
    if s0 is None:
        out["message"] = "no solver block in results"
        return out
    tc = getattr(s0, "termination_condition", None)
    out["termination"] = str(tc if tc is not None else "unknown")
    for key in ("message", "Message"):
        m = getattr(s0, key, None)
        if m is not None and str(m).strip():
            out["message"] = str(m)
            break
    rs = getattr(s0, "return_status", None)
    if rs is not None and str(rs).strip():
        out["return_status"] = str(rs)
    t = getattr(s0, "time", None) or getattr(s0, "Time", None)
    if t is not None and t == t:
        out["time_s"] = float(t)
    return out


def solve_lcow_nlp(
    salt_name: str,
    site: SiteClimate,
    econ: LCOEconomicParams | None = None,
    *,
    ipopt_tee: bool = False,
    ipopt_print_level: int | None = None,
) -> tuple[float, float, pe.ConcreteModel, dict[str, Any]]:
    """Return (best LCOW, SL, model, solver_info) for one salt and one site.

    Uses **Ipopt** when available, else **SciPy** ``minimize_scalar``. Pass ``ipopt_tee=True``
    to stream Ipopt’s iteration log to the process stdout (useful with ``--ipopt-tee`` in CLIs).
    """
    info: dict[str, Any] = {"backend": "none", "salt": salt_name}
    econ = econ or LCOEconomicParams()
    m = build_lcow_sawh_model(salt_name, site, econ)
    if getattr(m, "infeasible", False):
        info["reason"] = "model_infeasible (RH/mf or uptake)"
        return 1e30, float("nan"), m, info
    if ipopt_available():
        opt = SolverFactory("ipopt", validate=False)
        opt.options["max_iter"] = 500
        opt.options["tol"] = 1e-7
        if ipopt_print_level is not None:
            opt.options["print_level"] = int(ipopt_print_level)
        res = opt.solve(m, tee=ipopt_tee, load_solutions=True)
        info = _ipopt_solve_info(res)
        info["salt"] = salt_name
        s0 = _first_solver_subresult(res)
        tc = getattr(s0, "termination_condition", None) if s0 is not None else None
        if tc not in (
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
        ):
            lcv, slv, m, sc_info = _scipy_or_fail(salt_name, site, econ, m)
            info = {**info, "ipopt_failed": True, "fallback": sc_info}
            return lcv, slv, m, info
        slv = pe.value(m.SL)
        lv = pe.value(m.lcow_expr)
        if not (math.isfinite(slv) and math.isfinite(lv)):
            lcv, slv, m, sc_info = _scipy_or_fail(salt_name, site, econ, m)
            info = {**info, "ipopt_incomplete_values": True, "fallback": sc_info}
            return lcv, slv, m, info
        lv_ref = lcow_at_sl(salt_name, site, econ, float(slv))
        if math.isfinite(lv_ref) and abs(float(lv) - lv_ref) > max(1e-4, abs(lv_ref) * 1e-5):
            lv = lv_ref
        return float(lv), float(slv), m, info
    lcv, slv, m, sc_info = _scipy_or_fail(salt_name, site, econ, m)
    return lcv, slv, m, {**sc_info, "salt": salt_name}


def _scipy_or_fail(
    salt_name: str,
    site: SiteClimate,
    econ: LCOEconomicParams,
    m: pe.ConcreteModel,
) -> tuple[float, float, pe.ConcreteModel, dict[str, Any]]:
    out = _solve_lcow_scipy(m, salt_name, site, econ)
    if out is None:
        return 1e30, float("nan"), m, {
            "backend": "scipy",
            "success": False,
            "message": "minimize_scalar failed or non-finite objective",
        }
    lcv, slv, sinfo = out
    m.SL.set_value(slv)
    return float(lcv), float(slv), m, sinfo


def optimize_salt_and_sl(
    site: SiteClimate,
    salts: tuple[str, ...] = CANDIDATE_SALTS,
    econ: LCOEconomicParams | None = None,
    *,
    ipopt_tee: bool = False,
    ipopt_print_level: int | None = None,
) -> LCOOPTResult:
    """One Pyomo model, **one** outer solve: NLP if one feasible salt, else MINLP.

    * **NLP (single feasible salt)**: prefer Ipopt on the unified isotherm + LCOW model; if Ipopt
      is missing or not optimal, one SciPy 1D minimization (never one solve per *candidate* salt).
    * **MINLP (several feasible)**: tries each of **Bonmin** and **Couenne** on ``PATH`` (if
      present). If every ASL run fails or is non-optimal (common: Bonmin/ASL exit abnormally on
      some platforms), falls back to **one bounded 1-D minimization per feasible salt** and takes
      the minimum — the same as nested min over salt then SL for this model, not a heuristic.

    If multiple salts are feasible and **no** ASL MINLP is installed, raises ``RuntimeError`` with
    install hints.

    For a single-salt per-site API (tests, debugging), use :func:`solve_lcow_nlp`.
    """
    econ = econ or LCOEconomicParams()
    feasible = frozenset(feasible_salts_for_site(site, salts))
    if len(feasible) >= 2 and _first_available_minlp_solver() is None:
        raise RuntimeError(
            "Several salts are thermodynamically feasible for this site, so the model requires "
            "a single MINLP solve (bonmin or couenne on PATH). None were found. Install e.g. "
            "'conda install -c conda-forge coinbonmin', or use a smaller salt list / pre-filter "
            "candidates. Feasible here: "
            f"{sorted(feasible)}"
        )

    ubr = build_unified_lcow_model(site, econ, salt_list=salts)
    if ubr is None:
        per0: dict[str, dict[str, Any]] = {}
        for s in salts:
            get_salt(s)
            ok = s in feasible
            per0[s] = {
                "lcow": 1e30,
                "sl": float("nan"),
                "infeasible": not ok,
                "solver": {"reason": "no salt passes isotherm / swing for this site"},
            }
        return LCOOPTResult("none", float("nan"), float("inf"), per0, False, None)

    m = ubr.model
    mode = ubr.mode
    s_list = ubr.salt_list

    if mode == "nlp":
        if ipopt_available():
            res, sinfo = _solve_unified_nlp(
                m, ipopt_tee=ipopt_tee, ipopt_print_level=ipopt_print_level
            )
            if _unified_termination_ok(res):
                best, sl, lc = _extract_unified(m, s_list, mode="nlp")
                if math.isfinite(sl) and math.isfinite(lc):
                    per = _per_salt_table_joint_sl(site, econ, salts, sl, lc, best, feasible)
                    if best in per:
                        per[best]["solver"] = sinfo
                    return LCOOPTResult(
                        best, sl, lc, per, solved_unified=True, unified_mode="nlp"
                    )
        return _unified_nlp_scipy(s_list[0], site, econ, salts, feasible)

    if mode == "minlp":
        asl = _all_available_minlp_solvers()
        if not asl:
            raise RuntimeError("MINLP mode but no bonmin/couenne; this should not happen.")
        attempts: list[dict[str, Any]] = []
        for mn in asl:
            res, sinfo = _solve_unified_minlp(m, tee=ipopt_tee, solver_name=mn)
            attempts.append(dict(sinfo))
            if res is not None and _unified_termination_ok(res):
                best, sl, lc = _extract_unified(m, s_list, mode="minlp")
                if math.isfinite(sl) and math.isfinite(lc):
                    per = _per_salt_table_joint_sl(site, econ, salts, sl, lc, best, feasible)
                    if best in per:
                        per[best]["solver"] = {**sinfo, "minlp_tried": list(asl)}
                    return LCOOPTResult(
                        best, sl, lc, per, solved_unified=True, unified_mode="minlp"
                    )
        best, sl, lc, enum_info = _best_lcow_among_feasible_scipy(site, econ, feasible)
        enum_info["minlp_asl_attempts"] = attempts
        per = _per_salt_table_joint_sl(site, econ, salts, sl, lc, best, feasible)
        if best in per:
            per[best]["solver"] = enum_info
        return LCOOPTResult(
            best, sl, lc, per, solved_unified=True, unified_mode="minlp_scipy_enum"
        )

    raise RuntimeError(f"Unknown unified model mode: {mode!r}")
