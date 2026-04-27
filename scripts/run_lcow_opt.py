#!/usr/bin/env python3
"""Example: solve LCOW for one site (lat/lon + year).

Default: discrete salt choice via :func:`optimize_salt_and_sl` (unified Pyomo).
Use ``--zsr`` for continuous blend optimization (Zdanovskii isopiestic mixing + SLSQP).
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.weather import WeatherClient
from src.models.lcow_sawh import SiteClimate
from src.optimization.climate import site_row_from_hourly
from src.optimization.economics import LCOEconomicParams
from src.materials.salts import CANDIDATE_SALTS
from src.models.salt_unified_model import feasible_salts_for_site
from src.optimization.solve import ipopt_available, optimize_salt_and_sl
from src.optimization.zsr_mixing import optimize_zsr_blend_and_sl


def _print_solve_info(title: str, d: object) -> None:
    if not isinstance(d, dict):
        return
    skip = {"salt", "fallback"}
    parts = [
        f"{kk}={d[kk]!r}"
        for kk in sorted(d)
        if kk not in skip and d.get(kk) is not None
    ]
    if parts:
        print(f"    [{title}] " + " ".join(parts))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, default=33.45)
    p.add_argument("--lon", type=float, default=-112.07)
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--cache", type=Path, default=_REPO / ".cache" / "openmeteo")
    p.add_argument(
        "--ipopt-tee",
        action="store_true",
        help="Stream Ipopt iteration log to stdout (unified model solve).",
    )
    p.add_argument(
        "--ipopt-print-level",
        type=int,
        default=None,
        metavar="N",
        help="Set Ipopt print_level 0-12 (default: omit, or 5 when --ipopt-tee is set).",
    )
    p.add_argument(
        "--zsr",
        action="store_true",
        help="Continuous ZSR salt blend + SL: Pyomo NLP + Ipopt if available, else SciPy SLSQP.",
    )
    args = p.parse_args()
    ipopt_pl = args.ipopt_print_level
    if args.ipopt_tee and ipopt_pl is None:
        ipopt_pl = 5
    start = date(args.year, 1, 1)
    end = date(args.year, 12, 31)
    client = WeatherClient(cache_dir=args.cache)
    df = client.get_historical(
        args.lat,
        args.lon,
        start,
        end,
        variables=("relative_humidity_2m",),
    )
    row = site_row_from_hourly(df)
    site = SiteClimate(rh_high=row["rh_high_frac"], rh_low=row["rh_low_frac"])
    econ = LCOEconomicParams()
    print(f"Site ({args.lat}, {args.lon}) {args.year}: rh_high={site.rh_high:.3f} rh_low={site.rh_low:.3f}")

    if args.zsr:
        names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
        if not names:
            print("No thermodynamically feasible salts at this site for ZSR.", file=sys.stderr)
            raise SystemExit(1)
        z = optimize_zsr_blend_and_sl(
            site,
            names,
            econ,
            ipopt_tee=args.ipopt_tee,
            ipopt_print_level=ipopt_pl,
        )
        if not z.success or not math.isfinite(z.best_lcow):
            print(f"ZSR optimize failed: {z.message}", file=sys.stderr)
            raise SystemExit(1)
        print(
            f"Solve: mode=zsr (Pyomo+Ipopt when available; else SciPy)  backend={z.backend!r}"
        )
        print(f"Best SL={z.best_sl:.4f}  LCOW=${z.best_lcow:.6f}/kg")
        for nm, fi in zip(z.names, z.best_f, strict=True):
            if fi >= 1e-5:
                print(f"  {nm}: f={fi:.5f}")
        return

    if not ipopt_available():
        print(
            "Warning: Ipopt not found; unified NLP falls back to SciPy 1D minimization for "
            "single-feasible-salt sites. If several salts are feasible, install coinbonmin "
            "(or couenne) for one MINLP solve.",
            file=sys.stderr,
        )
    else:
        pl_msg = f"print_level={ipopt_pl}" if ipopt_pl is not None else "print_level=default"
        print(
            f"Ipopt: tee={'on' if args.ipopt_tee else 'off'}  {pl_msg}",
            file=sys.stderr,
        )
    try:
        out = optimize_salt_and_sl(
            site,
            econ=econ,
            ipopt_tee=args.ipopt_tee,
            ipopt_print_level=ipopt_pl,
        )
    except RuntimeError as exc:
        print(f"LCOW solve failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Solve: unified={out.solved_unified}  mode={out.unified_mode!r}")
    print(f"Best salt: {out.best_salt}  SL={out.best_sl:.4f}  LCOW=${out.best_lcow:.4f}/kg")
    for k, v in out.per_salt.items():
        si = v.get("solver") or {}
        infeas = v.get("infeasible", False)
        print(
            f"  {k}: lcow={v['lcow']:.4f} sl={v['sl']}  infeasible={infeas}"
        )
        _print_solve_info("solver", {x: y for x, y in si.items() if x != "fallback"})
        _print_solve_info("scipy_fallback", si.get("fallback"))


if __name__ == "__main__":
    main()
