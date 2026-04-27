# Sorbent AWH Optimization

A Pyomo-based optimization framework for **electrolyte and hydrogel design** for sorbent-based **atmospheric water harvesting (AWH)** — anywhere in the world.

## Overview

This package provides tools to:

- **Optimize sorbent material formulations** (electrolytes, hydrogels) for AWH performance under real-world climate conditions
- **Query global weather data** (relative humidity, solar irradiance, temperature) via the [Open-Meteo](https://open-meteo.com/) API
- **Run Pyomo optimization models** for sorbent design, cost, and water yield
- **Analyze sensitivity** of system performance to material and climate parameters

## Repository Structure

```
electrolyte_optimization/
├── src/
│   ├── data/
│   │   └── weather/        # Open-Meteo API client for global climate data
│   ├── models/             # Pyomo optimization model definitions
│   ├── optimization/       # Solvers, problem formulations, and workflows
│   ├── materials/          # Electrolyte and hydrogel property databases
│   ├── analysis/           # Post-processing, sensitivity analysis, visualization
│   └── utils/              # Shared utilities (unit conversion, I/O, logging)
├── tests/                  # Unit and integration tests
├── notebooks/              # Jupyter notebooks for exploration and results
├── docs/                   # Documentation
└── configs/                # Configuration files (solver settings, material params)
```

## Installation

```bash
git clone https://github.com/<your-org>/electrolyte_optimization.git
cd electrolyte_optimization
pip install -e ".[dev]"
```

> **Requirements:** Python ≥ 3.10, Pyomo, SciPy, pandas, requests. **Ipopt** (nonlinear solver) is required for the LCOW NLP; install via Conda (`conda install -c conda-forge ipopt`), Homebrew (`brew install ipopt`), or your platform package manager. Optional: IDAES-PSE bundles solvers in some environments.

## Quick Start

### Fetch weather data for a location

```python
from src.data.weather import WeatherClient

client = WeatherClient()

# By coordinates
df = client.get_historical(latitude=28.6, longitude=77.2, start="2024-01-01", end="2024-12-31")

# By place name (uses built-in geocoding)
df = client.get_historical_by_name("Phoenix, AZ", start="2024-01-01", end="2024-12-31")

print(df[["temperature_2m", "relative_humidity_2m", "shortwave_radiation"]].describe())
```

### Run LCOW optimization (salt + salt-to-polymer ratio)

The model minimizes **levelized cost of water** ($/kg) over **salt** and **SL = m_s / m_p** in **one** unified Pyomo solve: NLP (Ipopt, or SciPy 1D if only one salt is feasible and Ipopt is missing) or MINLP (Bonmin and/or Couenne, each tried if installed) when several salts are thermodynamically feasible. If every ASL MINLP run fails or is non-optimal (e.g. “Solver (asl) did not exit normally”), the code uses **one bounded 1-D solve per feasible salt** and takes the minimum—the same nested minimum for this model. Economics use supplement-style BOM constants; sorption uses equilibrium **mf(RH)** ported from the project’s `calculate_mf_*.m` logic; default active layer is **1 mm** at **2000 kg/m³** (2 kg dry composite per m²).

```python
from src.models.lcow_sawh import SiteClimate
from src.optimization.economics import LCOEconomicParams
from src.optimization.solve import optimize_salt_and_sl

# Representative diurnal RH (fractions 0–1): high for absorption, low for desorption.
site = SiteClimate(rh_high=0.88, rh_low=0.78)
econ = LCOEconomicParams(f_wacc=0.08, L_years=10, f_util=0.9)
result = optimize_salt_and_sl(site, econ=econ)
print(result.best_salt, result.best_sl, result.best_lcow)
```

CLI example (fetches one year of hourly RH via Open-Meteo, then optimizes):

```bash
python scripts/run_lcow_opt.py --lat 33.45 --lon -112.07 --year 2023
```

### Random land sites → world map (LCOW)

Install map dependencies, then sample random **land** locations (Natural Earth + Shapely), run the same optimization at each site, and save a PNG + CSV:

```bash
pip install -e ".[maps]"
python scripts/lcow_random_global_map.py --n 24 --seed 0 --year 2023
```

Outputs default to `outputs/lcow_global/lcow_random_sites.png` and `.csv`. Use `--sleep` to pace Open-Meteo requests (seconds between sites).

**Convention:** `LCOEconomicParams` keeps **one** utilization factor in the LCOW denominator: annual cost ÷ (`f_util` × gross annual water yield), with gross yield before `f_util` (see `src/optimization/economics.py`).

**Salt bulk cost:** Candidate-salt **$/kg** comes from [`src/data/salt_pricing/salt_price_data.xlsx`](src/data/salt_pricing/salt_price_data.xlsx) (`$/kg` column) when present; otherwise a small default (see `src/materials/salts.py`). Requires `openpyxl` (listed in project dependencies).

## Data Sources

| Variable | Source | Notes |
|---|---|---|
| Relative Humidity (%) | Open-Meteo Archive / Forecast | ERA5 reanalysis |
| Temperature (°C) | Open-Meteo Archive / Forecast | ERA5 reanalysis |
| Shortwave Radiation (W/m²) | Open-Meteo Archive / Forecast | Global horizontal |
| Direct Normal Irradiance (W/m²) | Open-Meteo Archive / Forecast | DNI |
| Diffuse Radiation (W/m²) | Open-Meteo Archive / Forecast | |

## License

MIT
