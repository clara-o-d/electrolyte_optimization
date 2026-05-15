#!/usr/bin/env python3
"""Solve the bi-objective (LCOW, daily water yield) problem for one site.

Fetches one year of hourly weather from Open-Meteo, derives mean diurnal RH /
temperature / irradiance, then either:

* (default) minimizes LCOW with :func:`optimize_zsr_blend_and_sl` and also
  reports the daily water yield at that optimum; or
* (``--pareto-points N``) traces the full LCOW–yield Pareto front via the
  epsilon-constraint sweep in :func:`optimize_zsr_pareto_front`.
"""

from __future__ import annotations

import argparse
import csv
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
from src.optimization.zsr_mixing import (
    ParetoPoint,
    optimize_zsr_blend_and_sl,
    optimize_zsr_pareto_front,
)


def _format_blend(names: tuple[str, ...], weights, *, threshold: float = 1e-5) -> str:
    return ", ".join(
        f"{nm}={float(w):.4f}"
        for nm, w in zip(names, weights, strict=True)
        if float(w) >= threshold
    )


def _print_point(label: str, names: tuple[str, ...], point: ParetoPoint) -> None:
    blend = _format_blend(names, point.blend_weights)
    print(
        f"{label}  SL={point.salt_to_polymer_ratio:.4f}  "
        f"LCOW=${point.lcow_usd_per_kg:.6f}/kg  "
        f"yield={point.daily_water_yield_L_per_m2:.4f} L/m²/day  "
        f"[{blend}]"
    )


def _write_pareto_csv(path: Path, names: tuple[str, ...], points: list[ParetoPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lcow_usd_per_kg",
        "daily_water_yield_L_per_m2",
        "salt_to_polymer_ratio",
        "anchor_min_lcow",
        "anchor_max_yield",
        "backend",
        *(f"f_{nm}" for nm in names),
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in points:
            row = {
                "lcow_usd_per_kg": p.lcow_usd_per_kg,
                "daily_water_yield_L_per_m2": p.daily_water_yield_L_per_m2,
                "salt_to_polymer_ratio": p.salt_to_polymer_ratio,
                "anchor_min_lcow": int(p.is_anchor_min_lcow),
                "anchor_max_yield": int(p.is_anchor_max_yield),
                "backend": p.backend,
            }
            for nm, weight in zip(names, p.blend_weights, strict=True):
                row[f"f_{nm}"] = float(weight)
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p.add_argument(
        "--pareto-points",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If N>=2, trace the LCOW–daily-yield Pareto front with N points "
            "(2 anchors + N-2 interior epsilon-constraint solves). Default 0: "
            "single LCOW-minimizing solve."
        ),
    )
    p.add_argument(
        "--pareto-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the Pareto front to this CSV (used only with --pareto-points).",
    )
    p.add_argument(
        "--hydrogel-lifetime-months",
        type=float,
        default=None,
        metavar="M",
        help="Hydrogel replacement interval in months (default: 12). Use 1 for monthly.",
    )
    p.add_argument(
        "--max-electric-heat-w-per-m2",
        type=float,
        default=None,
        metavar="Q",
        help=(
            "Enable active electrical gel heating with this upper bound on the "
            "Q_elec decision variable (W/m^2 of gel footprint). 0 (default) keeps "
            "the device passive (sun-only)."
        ),
    )
    p.add_argument(
        "--electricity-price-usd-per-kwh",
        type=float,
        default=None,
        metavar="P",
        help="Override electricity price for active-heating LCOW (USD/kWh).",
    )
    p.add_argument(
        "--desorption-hours-per-day",
        type=float,
        default=None,
        metavar="H",
        help="Active electrical heating duty (hours per day) for the energy cost.",
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
        variables=("relative_humidity_2m", "temperature_2m", "shortwave_radiation"),
    )
    row = site_row_from_hourly(df)
    site = SiteClimate(
        humidity_high=row["rh_high_frac"],
        temperature_c=row["temperature_high_c"],
        solar_irradiance_w_per_m2=row["solar_irradiance_w_per_m2"],
    )
    econ = LCOEconomicParams()
    if args.hydrogel_lifetime_months is not None and args.hydrogel_lifetime_months <= 0.0:
        print("--hydrogel-lifetime-months must be positive.", file=sys.stderr)
        raise SystemExit(1)
    if args.max_electric_heat_w_per_m2 is not None and args.max_electric_heat_w_per_m2 < 0.0:
        print("--max-electric-heat-w-per-m2 must be >= 0.", file=sys.stderr)
        raise SystemExit(1)
    hydrogel_years = (
        args.hydrogel_lifetime_months / 12.0
        if args.hydrogel_lifetime_months is not None
        else econ.hydrogel_lifetime_years
    )
    max_q = (
        float(args.max_electric_heat_w_per_m2)
        if args.max_electric_heat_w_per_m2 is not None
        else econ.max_electric_heat_w_per_m2
    )
    elec_price = (
        float(args.electricity_price_usd_per_kwh)
        if args.electricity_price_usd_per_kwh is not None
        else econ.electricity_price_usd_per_kwh
    )
    desorp_hours = (
        float(args.desorption_hours_per_day)
        if args.desorption_hours_per_day is not None
        else econ.desorption_hours_per_day
    )
    econ = LCOEconomicParams(
        discount_rate=econ.discount_rate,
        device_lifetime_years=econ.device_lifetime_years,
        total_investment_factor=econ.total_investment_factor,
        maintenance_cost_fraction=econ.maintenance_cost_fraction,
        utilization_factor=econ.utilization_factor,
        hydrogel_lifetime_years=hydrogel_years,
        energy_cost_usd_per_year=econ.energy_cost_usd_per_year,
        c_acrylamide_usd_per_kg=econ.c_acrylamide_usd_per_kg,
        c_additives_usd_per_kg_composite=econ.c_additives_usd_per_kg_composite,
        electricity_price_usd_per_kwh=elec_price,
        desorption_hours_per_day=desorp_hours,
        max_electric_heat_w_per_m2=max_q,
    )
    print(
        f"Site ({args.lat}, {args.lon}) {args.year}:  humidity_high={site.humidity_high:.3f} "
        f"(diag min RH: {row['rh_low_frac']:.3f})  "
        f"T_amb={site.temperature_c:.1f}C (diag min: {row['temperature_low_c']:.1f}C)  "
        f"I_solar={site.solar_irradiance_w_per_m2:.0f} W/m^2  "
        f"T_gel={site.gel_temperature_c:.1f}C  "
        f"hydrogel_lifetime={econ.hydrogel_lifetime_years * 12.0:.4g} months"
    )

    names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
    if not names:
        print("No thermodynamically feasible salts at this site.", file=sys.stderr)
        raise SystemExit(1)

    if args.pareto_points and args.pareto_points >= 2:
        front = optimize_zsr_pareto_front(
            site,
            names,
            econ,
            num_points=args.pareto_points,
            ipopt_tee=args.ipopt_tee,
            ipopt_print_level=ipopt_pl,
        )
        if not front.min_lcow_point.success:
            print(f"Min-LCOW anchor failed: {front.min_lcow_point.message}", file=sys.stderr)
            raise SystemExit(1)
        if not front.max_yield_point.success:
            print(f"Max-yield anchor failed: {front.max_yield_point.message}", file=sys.stderr)
            raise SystemExit(1)
        print(
            f"Pareto front ({len(front.points)} successful points, "
            f"requested {args.pareto_points}):"
        )
        _print_point("  [min LCOW    ]", front.names, front.min_lcow_point)
        _print_point("  [max yield   ]", front.names, front.max_yield_point)
        print("  ── sorted by daily yield ──")
        for pt in front.points:
            tag = "anchor" if (pt.is_anchor_min_lcow or pt.is_anchor_max_yield) else "interior"
            _print_point(f"  [{tag:<8}]", front.names, pt)
        if args.pareto_csv is not None:
            _write_pareto_csv(args.pareto_csv, front.names, front.points)
            print(f"Wrote {args.pareto_csv}")
        return

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
    print(
        f"Best SL={z.best_sl:.4f}  LCOW=${z.best_lcow:.6f}/kg  "
        f"daily yield={z.best_daily_yield_L_per_m2:.4f} L/m²/day"
    )
    if econ.max_electric_heat_w_per_m2 > 0.0:
        annual_electricity_usd = (
            econ.electricity_price_usd_per_kwh
            * z.best_electric_heat_w_per_m2
            * econ.desorption_hours_per_day
            * 365.0
            / 1000.0
        )
        print(
            f"  Active heating: Q_elec={z.best_electric_heat_w_per_m2:.2f} W/m² "
            f" T_gel={z.best_gel_temperature_c:.2f} °C "
            f" annual electricity=${annual_electricity_usd:.4f}/yr/m²"
        )
    for nm, fi in zip(z.names, z.best_f, strict=True):
        if fi >= 1e-5:
            print(f"  {nm}: f={fi:.5f}")


if __name__ == "__main__":
    main()
