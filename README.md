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

> **Requirements:** Python ≥ 3.10, Pyomo, IDAES-PSE (optional), pandas, requests

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

### Run the optimization (coming soon)

```python
from src.optimization import AWHOptimizer

opt = AWHOptimizer(weather_df=df)
result = opt.solve()
```

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
