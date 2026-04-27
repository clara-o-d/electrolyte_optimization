"""Generate global maps of AWH-relevant climate statistics.

Samples a regular lat/lon grid using :class:`src.data.weather.WeatherClient`
(Open-Meteo ERA5 archive), aggregates one year of hourly data at each grid
point, and renders five global maps:

    1. Annual-average relative humidity                 (%)
    2. Annual-average shortwave (GHI) solar irradiance  (W/m^2)
    3. Annual-average 2 m air temperature                (degC)
    4. Maximum daily relative-humidity swing             (%-points)
    5. Mean diurnal temperature swing (day - night)      (degC)

Usage
-----
    python scripts/global_weather_maps.py                     # 10 deg grid, 1 year
    python scripts/global_weather_maps.py --resolution 20     # coarser, faster
    python scripts/global_weather_maps.py --year 2022 \
        --calls-per-minute 20 --workers 2                     # extra-polite pacing

Notes
-----
Each grid point triggers one Open-Meteo archive request.  Responses are
cached under ``.cache/openmeteo`` so re-runs skip already-fetched cells.
The script defaults to 2 workers and ~25 calls/minute to stay under
Open-Meteo's archive-API rate limit; expect the first full 10 deg run
(648 cells) to take ~25-30 minutes. 429s are handled with backoff and
minutely-reset sleeps, so the script should not silently drop cells.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# Allow running `python scripts/global_weather_maps.py` from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.weather import WeatherClient  # noqa: E402

LOGGER = logging.getLogger("global_weather_maps")

# Variables we actually need; keeping the list small speeds up the API.
_VARIABLES = (
    "relative_humidity_2m",
    "temperature_2m",
    "shortwave_radiation",
)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def build_grid(resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-D arrays of latitude and longitude cell centres."""
    half = resolution_deg / 2.0
    lats = np.arange(-90.0 + half, 90.0, resolution_deg)
    lons = np.arange(-180.0 + half, 180.0, resolution_deg)
    return lats, lons


def summarize_point(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-location AWH statistics from an hourly DataFrame."""
    rh = df["relative_humidity_2m"]
    temp = df["temperature_2m"]
    ghi = df["shortwave_radiation"]

    daily = df[["relative_humidity_2m", "temperature_2m"]].resample("D").agg(["min", "max"])
    rh_daily_range = daily[("relative_humidity_2m", "max")] - daily[("relative_humidity_2m", "min")]
    temp_daily_range = daily[("temperature_2m", "max")] - daily[("temperature_2m", "min")]

    return {
        "rh_mean": float(rh.mean()),
        "temp_mean": float(temp.mean()),
        "ghi_mean": float(ghi.mean()),
        "rh_swing_max": float(rh_daily_range.max()),
        "temp_swing_mean": float(temp_daily_range.mean()),
    }


class RateLimiter:
    """Simple sliding-window limiter: at most ``max_calls`` per ``period`` seconds.

    Shared across threads. Callers invoke :meth:`acquire` right before
    making a request; it blocks until the call fits under the window.
    """

    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0])
            time.sleep(max(wait, 0.05))


def fetch_point(
    client: WeatherClient,
    lat: float,
    lon: float,
    start: str,
    end: str,
    limiter: RateLimiter,
    max_retries: int = 8,
) -> dict[str, float] | None:
    """Fetch + summarize one grid cell with retry/backoff on 429s.

    Returns ``None`` for non-retryable failures or when retries are
    exhausted.  Successful responses are cached by the WeatherClient,
    so re-running resumes quickly.
    """
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        limiter.acquire()
        try:
            df = client.get_historical(
                latitude=float(lat),
                longitude=float(lon),
                start=start,
                end=end,
                variables=_VARIABLES,
                timezone="UTC",
            )
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            msg = str(exc)
            if status == 429:
                if "Minutely" in msg or "minute" in msg.lower():
                    wait = 65.0
                elif "Hourly" in msg or "hour" in msg.lower():
                    wait = 60.0 * 10
                elif "Daily" in msg or "daily" in msg.lower():
                    LOGGER.error("Hit DAILY API limit; stopping further retries.")
                    return None
                else:
                    wait = backoff + random.uniform(0, 1.0)
                    backoff = min(backoff * 2, 30.0)
                LOGGER.info(
                    "429 at (%.1f, %.1f) attempt %d/%d; sleeping %.1fs",
                    lat, lon, attempt, max_retries, wait,
                )
                time.sleep(wait)
                continue
            LOGGER.warning("skip (%.1f, %.1f): %s", lat, lon, exc)
            return None
        except requests.ConnectionError as exc:
            wait = backoff + random.uniform(0, 1.0)
            backoff = min(backoff * 2, 30.0)
            LOGGER.info("conn error at (%.1f, %.1f) attempt %d: %s; sleep %.1fs",
                        lat, lon, attempt, exc, wait)
            time.sleep(wait)
            continue
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("skip (%.1f, %.1f): %s", lat, lon, exc)
            return None

        if df.empty:
            return None
        return summarize_point(df)

    LOGGER.warning("skip (%.1f, %.1f): retries exhausted", lat, lon)
    return None


def collect_global_stats(
    resolution_deg: float,
    year: int,
    workers: int,
    cache_dir: Path,
    calls_per_minute: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Fetch and summarize the full global grid, returning stat arrays."""
    lats, lons = build_grid(resolution_deg)
    total_cells = len(lats) * len(lons)
    LOGGER.info("Grid: %d x %d = %d points", len(lats), len(lons), total_cells)
    LOGGER.info("Workers=%d, target pace=%d calls/min", workers, calls_per_minute)

    client = WeatherClient(cache_dir=cache_dir)
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    limiter = RateLimiter(max_calls=calls_per_minute, period=60.0)

    stat_keys = ("rh_mean", "temp_mean", "ghi_mean", "rh_swing_max", "temp_swing_mean")
    grids = {k: np.full((len(lats), len(lons)), np.nan, dtype=float) for k in stat_keys}

    jobs: list[tuple[int, int, float, float]] = [
        (i, j, float(lat), float(lon))
        for i, lat in enumerate(lats)
        for j, lon in enumerate(lons)
    ]

    t0 = time.time()
    total = len(jobs)
    done = 0
    succeeded = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_point, client, lat, lon, start, end, limiter): (i, j)
            for i, j, lat, lon in jobs
        }
        for fut in as_completed(futures):
            i, j = futures[fut]
            result = fut.result()
            done += 1
            if result is not None:
                succeeded += 1
                for k, v in result.items():
                    grids[k][i, j] = v
            if done % max(1, total // 20) == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                LOGGER.info(
                    "  %4d / %d points  (ok=%d, %.1f pts/s, elapsed %.0fs)",
                    done, total, succeeded, rate, elapsed,
                )

    return lats, lons, grids


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _draw_map(
    ax: plt.Axes,
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Draw one global map onto ``ax`` using pcolormesh."""
    res_lat = lats[1] - lats[0] if len(lats) > 1 else 1.0
    res_lon = lons[1] - lons[0] if len(lons) > 1 else 1.0
    lat_edges = np.concatenate(([lats[0] - res_lat / 2], lats + res_lat / 2))
    lon_edges = np.concatenate(([lons[0] - res_lon / 2], lons + res_lon / 2))

    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(color="white", alpha=0.25, linewidth=0.5)
    ax.set_title(title, fontsize=11)

    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)


_MAP_SPECS: tuple[dict[str, object], ...] = (
    {
        "key": "rh_mean",
        "title": "Average relative humidity",
        "cbar": "RH (%)",
        "cmap": "Blues",
        "vmin": 0,
        "vmax": 100,
        "file": "global_rh_mean.png",
    },
    {
        "key": "ghi_mean",
        "title": "Average shortwave solar irradiance (GHI)",
        "cbar": "GHI (W/m$^2$)",
        "cmap": "inferno",
        "vmin": 0,
        "vmax": None,
        "file": "global_ghi_mean.png",
    },
    {
        "key": "temp_mean",
        "title": "Average 2 m air temperature",
        "cbar": "T (deg C)",
        "cmap": "RdYlBu_r",
        "vmin": -30,
        "vmax": 35,
        "file": "global_temp_mean.png",
    },
    {
        "key": "rh_swing_max",
        "title": "Maximum daily RH swing (max - min)",
        "cbar": "RH swing (%-pts)",
        "cmap": "viridis",
        "vmin": 0,
        "vmax": 100,
        "file": "global_rh_swing_max.png",
    },
    {
        "key": "temp_swing_mean",
        "title": "Mean diurnal temperature swing (day - night)",
        "cbar": "dT (deg C)",
        "cmap": "magma",
        "vmin": 0,
        "vmax": None,
        "file": "global_temp_swing_mean.png",
    },
)


def plot_all_maps(
    lats: np.ndarray,
    lons: np.ndarray,
    grids: dict[str, np.ndarray],
    out_dir: Path,
    year: int,
) -> None:
    """Save one PNG per statistic plus a combined 5-panel figure."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in _MAP_SPECS:
        fig, ax = plt.subplots(figsize=(12, 6))
        _draw_map(
            ax,
            lats,
            lons,
            grids[spec["key"]],
            title=f"{spec['title']}  ({year})",
            cbar_label=spec["cbar"],
            cmap=spec["cmap"],
            vmin=spec["vmin"],
            vmax=spec["vmax"],
        )
        fig.tight_layout()
        out_path = out_dir / spec["file"]
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("saved %s", out_path)

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    axes = axes.flatten()
    for ax, spec in zip(axes, _MAP_SPECS):
        _draw_map(
            ax,
            lats,
            lons,
            grids[spec["key"]],
            title=spec["title"],
            cbar_label=spec["cbar"],
            cmap=spec["cmap"],
            vmin=spec["vmin"],
            vmax=spec["vmax"],
        )
    axes[-1].axis("off")
    fig.suptitle(f"Global AWH climate summary ({year})", fontsize=15, y=0.995)
    fig.tight_layout()
    combined = out_dir / "global_summary.png"
    fig.savefig(combined, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("saved %s", combined)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolution", type=float, default=10.0,
                   help="Grid spacing in degrees (default 10)")
    p.add_argument("--year", type=int, default=2023,
                   help="Year of ERA5 history to summarize (default 2023)")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel HTTP workers (default 2; Open-Meteo rejects bursts)")
    p.add_argument("--calls-per-minute", type=int, default=25,
                   help="Global rate limit across all workers (default 25/min)")
    p.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "outputs" / "global_maps",
                   help="Where to save PNGs and the cached CSV")
    p.add_argument("--cache-dir", type=Path, default=_REPO_ROOT / ".cache" / "openmeteo",
                   help="Open-Meteo HTTP cache directory")
    p.add_argument("--stats-csv", type=Path, default=None,
                   help="Optional path to save the raw per-cell stats as CSV")
    return p.parse_args(argv)


def stats_to_dataframe(
    lats: np.ndarray,
    lons: np.ndarray,
    grids: dict[str, np.ndarray],
) -> pd.DataFrame:
    records = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            row = {"latitude": float(lat), "longitude": float(lon)}
            for key, grid in grids.items():
                row[key] = float(grid[i, j])
            records.append(row)
    return pd.DataFrame.from_records(records)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Collecting global stats for %d at %.1f deg resolution", args.year, args.resolution)
    lats, lons, grids = collect_global_stats(
        resolution_deg=args.resolution,
        year=args.year,
        workers=args.workers,
        cache_dir=args.cache_dir,
        calls_per_minute=args.calls_per_minute,
    )

    csv_path = args.stats_csv or (args.out_dir / f"global_stats_{args.year}_{int(args.resolution)}deg.csv")
    stats_df = stats_to_dataframe(lats, lons, grids)
    stats_df.to_csv(csv_path, index=False)
    LOGGER.info("Wrote per-cell stats to %s", csv_path)

    plot_all_maps(lats, lons, grids, args.out_dir, args.year)

    coverage = {k: float(np.isfinite(v).mean()) for k, v in grids.items()}
    LOGGER.info("Coverage (fraction of cells with data): %s", coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
