#!/usr/bin/env python3
"""Solve LCOW for one site using ZSR continuous blend optimization.

Fetches one year of hourly RH from Open-Meteo, derives mean diurnal RH extrema,
then runs :func:`optimize_zsr_blend_and_sl` (Pyomo + Ipopt when available, else
SciPy SLSQP) to find the optimal salt blend fractions and salt-to-polymer ratio.
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
from src.models.zsr_lcow_model import SiteClimate
from src.models.salt_feasibility import feasible_salts_for_site
from src.optimization.climate import site_row_from_hourly
from src.optimization.economics import LCOEconomicParams
from src.materials.salts import CANDIDATE_SALTS
from src.optimization.zsr_mixing import optimize_zsr_blend_and_sl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, default=33.45)
    p.add_argument("--lon", type=float, default=-112.07)
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--cache", type=Path, default=_REPO / ".cache" / "openmeteo")
    p.add_argument(
        "--ipopt-tee",
        action="store_true",
        help="Stream Ipopt iteration log to stdout.",
    )
    p.add_argument(
        "--ipopt-print-level",
        type=int,
        default=None,
        metavar="N",
        help="Set Ipopt print_level 0-12 (default: omit, or 5 when --ipopt-tee is set).",
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
    site = SiteClimate(humidity_high=row["rh_high_frac"], humidity_low=row["rh_low_frac"])
    econ = LCOEconomicParams()
    print(f"Site ({args.lat}, {args.lon}) {args.year}: humidity_high={site.humidity_high:.3f} humidity_low={site.humidity_low:.3f}")

    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    if not names:
        print("No thermodynamically feasible salts at this site.", file=sys.stderr)
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

    print(f"backend={z.backend!r}")
    print(f"Best SL={z.best_sl:.4f}  LCOW=${z.best_lcow:.6f}/kg")
    for nm, fi in zip(z.names, z.best_f, strict=True):
        if fi >= 1e-5:
            print(f"  {nm}: f={fi:.5f}")


if __name__ == "__main__":
    main()
