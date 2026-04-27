#!/usr/bin/env python3
"""Sample random land locations, run LCOW optimization, plot results on a world map.

Requires optional deps:  pip install -e ".[maps]"

By default each site uses **ZSR continuous salt mixture** optimization
(:func:`optimize_zsr_blend_and_sl`: Pyomo + Ipopt when available). Pass ``--discrete`` to use
the legacy discrete-salt path (:func:`optimize_salt_and_sl`).

Uses Natural Earth (via Cartopy) for coastlines/land and Shapely for land-only point
sampling. Open-Meteo is queried once per point (see ``--sleep`` for polite pacing).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.weather import WeatherClient
from src.materials.salts import CANDIDATE_SALTS
from src.models.lcow_sawh import SiteClimate
from src.models.salt_unified_model import feasible_salts_for_site
from src.optimization.climate import site_row_from_hourly
from src.materials.salt_prices import salt_price_data_path
from src.optimization.economics import LCOEconomicParams
from src.optimization.solve import ipopt_available, optimize_salt_and_sl
from src.optimization.zsr_mixing import optimize_zsr_blend_and_sl

_FAIL_LCO: float = 1e30

_SALT_MARKERS: dict[str, str] = {
    "LiCl": "o",
    "NaCl": "s",
    "CaCl2": "^",
    "MgCl2": "D",
    "none": "x",
}
_DEFAULT_MARK = "P"


@dataclass(slots=True)
class SiteResult:
    lat: float
    lon: float
    rh_high: float
    rh_low: float
    best_salt: str
    """For ZSR maps, the **dominant** salt (largest blend fraction) so markers stay meaningful."""
    best_sl: float
    best_lcow: float
    infeasible: bool
    optimize_mode: str = "zsr"
    """``\"zsr\"`` = continuous mixture NLP; ``\"discrete\"`` = :func:`optimize_salt_and_sl`."""
    zsr_backend: str = ""
    """``ipopt`` / ``scipy_slsqp`` when ``optimize_mode == \"zsr\"``."""
    blend: str = ""
    """Semicolon-separated ``Salt=weight`` for ZSR (empty for discrete)."""


def _try_import_map_stack():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    return plt, LogNorm, ccrs, cfeature


def _prepared_land_union():
    """Shapely prepared geometry of merged Natural Earth land polygons."""
    from shapely import geometry as sh_geom
    from shapely.ops import unary_union
    from shapely.prepared import prep

    import cartopy.io.shapereader as shpreader

    path = shpreader.natural_earth(resolution="110m", category="physical", name="land")
    geoms = list(shpreader.Reader(path).geometries())
    u = prep(unary_union(geoms))
    return u, sh_geom


def sample_land_points(
    n: int,
    seed: int,
    *,
    lat_lo: float = -56.0,
    lat_hi: float = 72.0,
    max_tries: int = 200_000,
) -> list[tuple[float, float]]:
    """Rejection sample ``n`` (lat, lon) degrees on land (WGS84)."""
    print("  Loading Natural Earth land polygons (first run may download shapefiles)…", flush=True)
    t0 = time.perf_counter()
    land, sh_geom = _prepared_land_union()
    print(f"  Land geometry ready in {time.perf_counter() - t0:.2f}s; sampling (lat ∈ [{lat_lo}, {lat_hi}])…", flush=True)
    rng = np.random.default_rng(seed)
    out: list[tuple[float, float]] = []
    attempts = 0
    for _ in range(max_tries):
        if len(out) >= n:
            break
        attempts += 1
        lat = float(rng.uniform(lat_lo, lat_hi))
        lon = float(rng.uniform(-180.0, 180.0))
        p = sh_geom.Point(lon, lat)
        if land.contains(p):
            out.append((lat, lon))
    if len(out) < n:
        raise RuntimeError(
            f"Only collected {len(out)}/{n} land points after {max_tries} attempts; "
            "try different seed or bounds."
        )
    print(
        f"  Picked {n} land point(s) in {attempts} random draws (seed={seed}, "
        f"acceptance ≈ {100.0 * n / max(attempts, 1):.1f}%).",
        flush=True,
    )
    return out


def run_sites(
    lats: list[float],
    lons: list[float],
    *,
    year: int,
    sleep_s: float,
    cache_dir: Path | None,
    discrete_salt: bool = False,
) -> list[SiteResult]:
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    n = len(lats)
    print(f"Open-Meteo: hourly RH for {year} ({start} → {end}), {n} site(s).", flush=True)
    if cache_dir is not None:
        print(f"  Weather cache: {cache_dir}", flush=True)
    client = WeatherClient(cache_dir=cache_dir)
    econ = LCOEconomicParams()
    results: list[SiteResult] = []
    t_batch = time.perf_counter()
    for i, (lat, lon) in enumerate(zip(lats, lons, strict=True), start=1):
        t_site = time.perf_counter()
        print(f"  [{i}/{n}] ({lat:+.4f}, {lon:+.4f})  fetching ERA5 archive…", end="", flush=True)
        df = client.get_historical(
            lat,
            lon,
            start,
            end,
            variables=("relative_humidity_2m",),
        )
        row = site_row_from_hourly(df)
        site = SiteClimate(
            rh_high=row["rh_high_frac"],
            rh_low=row["rh_low_frac"],
        )
        print(
            f"  mean daily max RH={site.rh_high:.3f}  min RH={site.rh_low:.3f}  optimizing…",
            end="",
            flush=True,
        )
        err_msg = ""
        out = None
        z_out = None
        try:
            if discrete_salt:
                out = optimize_salt_and_sl(site, econ=econ)
            else:
                names = tuple(feasible_salts_for_site(site, CANDIDATE_SALTS))
                if not names:
                    err_msg = "no feasible salts for site"
                else:
                    z_out = optimize_zsr_blend_and_sl(site, names, econ)
                    if not z_out.success or not math.isfinite(z_out.best_lcow):
                        err_msg = (z_out.message or "ZSR failed")[:240]
                        z_out = None
        except Exception as exc:
            err_msg = str(exc).split("\n", 1)[0][:240]
            out = None
            z_out = None

        if discrete_salt and out is None:
            print(f"  → skipped ({err_msg})  ({time.perf_counter() - t_site:.1f}s)", flush=True)
            results.append(
                SiteResult(
                    lat=lat,
                    lon=lon,
                    rh_high=site.rh_high,
                    rh_low=site.rh_low,
                    best_salt="none",
                    best_sl=float("nan"),
                    best_lcow=_FAIL_LCO,
                    infeasible=True,
                    optimize_mode="discrete",
                )
            )
            if sleep_s > 0.0 and i < n:
                time.sleep(sleep_s)
            continue
        if not discrete_salt and z_out is None:
            print(f"  → skipped ({err_msg})  ({time.perf_counter() - t_site:.1f}s)", flush=True)
            results.append(
                SiteResult(
                    lat=lat,
                    lon=lon,
                    rh_high=site.rh_high,
                    rh_low=site.rh_low,
                    best_salt="none",
                    best_sl=float("nan"),
                    best_lcow=_FAIL_LCO,
                    infeasible=True,
                    optimize_mode="zsr",
                )
            )
            if sleep_s > 0.0 and i < n:
                time.sleep(sleep_s)
            continue

        dt = time.perf_counter() - t_site
        if discrete_salt and out is not None:
            bad = not math.isfinite(out.best_lcow) or out.best_lcow >= 0.99 * _FAIL_LCO
            results.append(
                SiteResult(
                    lat=lat,
                    lon=lon,
                    rh_high=site.rh_high,
                    rh_low=site.rh_low,
                    best_salt=out.best_salt,
                    best_sl=float(out.best_sl) if out.best_sl == out.best_sl else float("nan"),
                    best_lcow=float(out.best_lcow),
                    infeasible=bad,
                    optimize_mode="discrete",
                )
            )
            if bad:
                print(f"  → infeasible / placeholder LCOW  ({dt:.1f}s)", flush=True)
            else:
                print(
                    f"  → discrete {out.best_salt}  SL={out.best_sl:.4f}  "
                    f"LCOW=${out.best_lcow:.6f}/kg  ({dt:.1f}s)",
                    flush=True,
                )
        elif z_out is not None:
            bad = not math.isfinite(z_out.best_lcow) or z_out.best_lcow >= 0.99 * _FAIL_LCO
            dom_i = int(np.argmax(z_out.best_f))
            dom = z_out.names[dom_i]
            blend = ";".join(
                f"{nm}={float(w):.4f}"
                for nm, w in zip(z_out.names, z_out.best_f, strict=True)
                if float(w) >= 1e-4
            )
            results.append(
                SiteResult(
                    lat=lat,
                    lon=lon,
                    rh_high=site.rh_high,
                    rh_low=site.rh_low,
                    best_salt=dom,
                    best_sl=float(z_out.best_sl),
                    best_lcow=float(z_out.best_lcow),
                    infeasible=bad,
                    optimize_mode="zsr",
                    zsr_backend=z_out.backend,
                    blend=blend,
                )
            )
            if bad:
                print(f"  → infeasible / placeholder LCOW  ({dt:.1f}s)", flush=True)
            else:
                short = blend if len(blend) <= 56 else blend[:53] + "…"
                print(
                    f"  → ZSR[{z_out.backend}]  dom={dom}  SL={z_out.best_sl:.4f}  "
                    f"LCOW=${z_out.best_lcow:.6f}/kg  [{short}]  ({dt:.1f}s)",
                    flush=True,
                )
        if sleep_s > 0.0 and i < n:
            time.sleep(sleep_s)
    print(f"  All sites done in {time.perf_counter() - t_batch:.1f}s (avg {((time.perf_counter() - t_batch) / max(n,1)):.1f}s / site, incl. sleep).", flush=True)
    return results


def plot_map(
    results: list[SiteResult],
    out_path: Path,
    *,
    title: str,
) -> None:
    plt, LogNorm, ccrs, cfeature = _try_import_map_stack()
    lats = np.array([r.lat for r in results])
    lons = np.array([r.lon for r in results])
    lc = np.array([r.best_lcow for r in results])
    salts = [r.best_salt for r in results]
    feas = np.array([not r.infeasible for r in results])

    ok = feas & np.isfinite(lc) & (lc < 0.99 * _FAIL_LCO) & (lc > 0.0)
    if np.any(ok):
        fvals = np.clip(lc[ok], 1e-9, None)
        vmin = float(np.nanmin(fvals) * 0.7)
        vmax = float(np.nanmax(fvals) * 1.4)
        vmin = max(vmin, 1e-6)
        vmax = max(vmax, vmin * 10)
    else:
        vmin, vmax = 1e-4, 1.0
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "110m", facecolor="0.88", edgecolor="0.4", linewidth=0.3, zorder=0),
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="0.92", zorder=0),
    )
    ax.coastlines(resolution="110m", color="0.3", linewidth=0.4, zorder=1)
    ax.gridlines(draw_labels=True, alpha=0.3, dms=True, x_inline=False, y_inline=False)
    sc_last = None
    for salt in sorted(set(salts)):
        idx = np.array(
            [i for i, s in enumerate(salts) if s == salt and ok[i]],
            dtype=int,
        )
        if idx.size == 0:
            continue
        sub = np.clip(lc[idx], 1e-9, None)
        sc_last = ax.scatter(
            lons[idx],
            lats[idx],
            c=sub,
            s=64,
            marker=_SALT_MARKERS.get(salt, _DEFAULT_MARK),
            transform=ccrs.PlateCarree(),
            zorder=4,
            cmap="viridis",
            norm=norm,
            edgecolors="0.1",
            linewidths=0.4,
        )
    bad_idx = np.where(~ok)[0]
    if bad_idx.size:
        ax.scatter(
            lons[bad_idx],
            lats[bad_idx],
            c="0.3",
            s=50,
            marker="x",
            transform=ccrs.PlateCarree(),
            zorder=5,
            label="infeasible or failed",
        )
    mappable = sc_last if sc_last is not None else plt.matplotlib.cm.ScalarMappable(
        norm=norm, cmap="viridis"
    )
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=0.03,
        pad=0.04,
    )
    cbar.set_label(
        "Best LCOW (USD/kg water) — log scale; marker = dominant salt in blend",
        fontsize=9,
    )
    ax.set_title(title, fontsize=12)
    leg_items = [plt.Line2D([0], [0], marker=m, color="k", linestyle="", label=n, ms=7) for n, m in _SALT_MARKERS.items() if n != "none"]
    ax.legend(
        handles=leg_items,
        loc="lower left",
        title="Marker = dominant salt",
        framealpha=0.9,
        fontsize=8,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=60, help="Number of random land points")
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep", type=float, default=0.35, help="Seconds between Open-Meteo calls")
    p.add_argument(
        "--out-png",
        type=Path,
        default=_REPO / "outputs" / "lcow_global" / "lcow_random_sites.png",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=_REPO / "outputs" / "lcow_global" / "lcow_random_sites.csv",
    )
    p.add_argument("--cache", type=Path, default=_REPO / ".cache" / "openmeteo")
    p.add_argument(
        "--discrete",
        action="store_true",
        help="Use discrete salt choice (optimize_salt_and_sl / MINLP path) instead of default ZSR mixture NLP.",
    )
    args = p.parse_args()
    print("=== lcow_random_global_map.py ===", flush=True)
    print(
        f"  points={args.n}  year={args.year}  seed={args.seed}  "
        f"sleep={args.sleep}s  Open-Meteo cache={args.cache}",
        flush=True,
    )
    xlsx = salt_price_data_path()
    print(f"  Salt $/kg: {xlsx.name}  (exists={xlsx.is_file()})", flush=True)
    if args.discrete:
        print(
            f"  Optimize: **discrete** (unified Pyomo); Ipopt={ipopt_available()}  "
            f"(multi-feasible sites may need Bonmin/Couenne or scipy enum fallback).",
            flush=True,
        )
    else:
        print(
            f"  Optimize: **ZSR mixture** (Pyomo+Ipopt if available, else SciPy); Ipopt={ipopt_available()}",
            flush=True,
        )
    t_main = time.perf_counter()
    try:
        _try_import_map_stack()
    except ImportError as e:
        print("Install map dependencies:  pip install -e '.[maps]'\n", e, file=sys.stderr)
        return 1
    print("Map/plot stack import OK (matplotlib + cartopy).", flush=True)
    print(f"--- Step 1: sample {args.n} random land point(s) ---", flush=True)
    lats, lons = zip(*sample_land_points(args.n, args.seed), strict=True)
    lats, lons = list(lats), list(lons)
    for i, (la, lo) in enumerate(zip(lats, lons, strict=True), start=1):
        print(f"    site {i}: ({la:+.4f}, {lo:+.4f})", flush=True)
    print("--- Step 2: weather + optimize each site ---", flush=True)
    res = run_sites(
        lats,
        lons,
        year=args.year,
        sleep_s=args.sleep,
        cache_dir=args.cache,
        discrete_salt=args.discrete,
    )
    # Summary
    feas = [r for r in res if r.infeasible is False]
    n_bad = len(res) - len(feas)
    lcs = [r.best_lcow for r in feas if math.isfinite(r.best_lcow) and r.best_lcow < 0.99 * _FAIL_LCO]
    salt_wins = Counter(r.best_salt for r in res)
    print("--- Step 3: summary ---", flush=True)
    print(f"  Feasible LCOW: {len(lcs)}/{len(res)}  (infeasible/placeholder: {n_bad})", flush=True)
    if lcs:
        print(
            f"  LCOW ($/kg) min={min(lcs):.6f}  max={max(lcs):.6f}  "
            f"median={float(np.median(np.asarray(lcs))):.6f}",
            flush=True,
        )
    print(f"  Best salt counts: {dict(salt_wins)}", flush=True)
    df = pd.DataFrame([asdict(r) for r in res])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}", flush=True)
    print("--- Step 4: render map (Natural Earth + scatter) ---", flush=True)
    mode_tag = "ZSR blend" if not args.discrete else "discrete salt"
    plot_map(
        res,
        args.out_png,
        title=(
            f"SAWH LCOW min ($/kg), {mode_tag}, random land sites, {args.year} "
            f"(mean diurnal RH extrema)"
        ),
    )
    print(f"Wrote {args.out_png}", flush=True)
    print(f"Done in {time.perf_counter() - t_main:.1f}s total.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
