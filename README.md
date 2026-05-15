# Electrolyte Optimization for Sorbent Atmospheric Water Harvesting

This repo models and optimizes salt-hydrogel composites for **sorbent atmospheric water harvesting (SAWH)** — devices that passively absorb water vapor from humid air and release it as liquid water when heated. The goal is to find the salt blend and composite formulation that minimizes the **levelized cost of water (LCOW, $/kg)** at a given geographic site.

## Background

A SAWH device cycles between two humidity states each day:

- **High humidity (uptake):** the hydrogel composite absorbs water vapor from the air. The salt dissolved in the hydrogel lowers water activity, creating a driving force for vapor absorption.
- **Low humidity (desorption):** the composite is heated (e.g., by sunlight), driving the water out as vapor, which condenses and is collected.

The amount of water produced per cycle depends on the **difference in water uptake** between the high- and low-humidity states. The four candidate salts — **LiCl, NaCl, CaCl2, MgCl2** — each have different vapor pressure isotherms and prices, so blending them can improve yield or reduce cost compared to any single salt.

## What the optimization does

For a site described by its mean daily high and low relative humidity, the optimizer finds:

1. **Blend weights** `f_i` for each salt (fraction of the brine attributed to salt `i`, summing to 1)
2. **Salt-to-polymer ratio** (mass of salt per mass of acrylamide polymer in the dry composite)

...to minimize LCOW:

```
LCOW = annual_cost_USD / (utilization_factor × annual_water_yield_kg)
```

The brine state at each humidity is computed using the **ZSR (Zdanovskii) isopiestic mixing rule**: at fixed humidity, each salt contributes molality in proportion to its blend weight and its binary reference molality (the molality of a pure salt solution at that humidity).

## Repository structure

```
src/
  materials/
    salts.py              Salt records: formula weight, ion count, RH range, price
    salt_prices.py        Loads bulk salt prices from salt_price_data.xlsx

  optimization/
    brine_equilibrium.py  Isotherm fits: equilibrium brine salt fraction at a given RH
                          Also converts between mass fraction and molality
    brine_uptake.py       Sorption factor physics: water mole fraction, uptake coefficient B
    economics.py          LCOEconomicParams dataclass + device/material cost constants
    climate.py            Aggregates hourly weather data to mean daily RH high/low
    zsr_mixing.py         ZSR mixing rule, scalar LCOW function, optimizer entry point

  models/
    zsr_lcow_model.py     Pyomo NLP model (decision variables, constraints, objective)
                          Also contains SiteClimate, HalfSwingCoefficients, and scalar helpers
    salt_feasibility.py   Pre-filters candidate salts to those with a positive half-swing

  data/
    weather/              Open-Meteo weather client (fetches/caches hourly RH)
    salt_pricing/         salt_price_data.xlsx

configs/
  default_lcow.yaml       Default economic parameters

scripts/
  run_lcow_opt.py         Optimize a single site (fetches weather, prints result)
  lcow_random_global_map.py  Sample random land sites, optimize each, plot a world map

tests/
  test_zsr_mixing.py      Unit + integration tests for the ZSR model
```

## Key concepts

**Equilibrium brine salt fraction** (`brine_equilibrium.py`): at a given relative humidity, a salt solution reaches thermodynamic equilibrium at a specific salt mass fraction. This is solved numerically from isotherm fits (polynomial and implicit forms, ported from MATLAB).

**Uptake coefficient B** (`brine_uptake.py`): a dimensionless group that connects the brine composition to water uptake. Defined as:

```
B = (x_w × ions_per_formula / (1 − x_w)) × (water_MW / salt_MW)
```

where `x_w` is the water mole fraction in the brine (colligative basis). The half-cycle water yield per kg of composite is proportional to `(B_high − B_low)`.

**ZSR mixing** (`zsr_mixing.py`): at fixed humidity, the mixture molality of salt `i` is `blend_weight_i × reference_molality_i`. Effective mixture properties (ion count, formula weight) are molality-weighted averages across salts.

**Pyomo NLP** (`zsr_lcow_model.py`): the blend weights and salt-to-polymer ratio are continuous decision variables. Constraints enforce that weights sum to 1 and that the high-humidity uptake exceeds the low-humidity uptake (positive half-swing). The objective is LCOW.

**Solvers**: Ipopt (interior-point NLP solver) is used when available. SciPy SLSQP is the fallback.

## Installation

```bash
pip install -e .
```

For the global map script:
```bash
pip install -e ".[maps]"
```

Ipopt can be installed via conda:
```bash
conda install -c conda-forge ipopt
```

## Usage

**Optimize a single site:**
```bash
python scripts/run_lcow_opt.py --lat 33.45 --lon -112.07 --year 2023
```

**Run across random land sites and plot:**
```bash
python scripts/lcow_random_global_map.py --n-sites 50 --year 2023
```

**Use the optimizer directly:**
```python
from src.models.zsr_lcow_model import SiteClimate
from src.models.salt_feasibility import feasible_salts_for_site
from src.optimization.economics import LCOEconomicParams
from src.optimization.zsr_mixing import optimize_zsr_blend_and_sl
from src.materials.salts import CANDIDATE_SALTS

site = SiteClimate(humidity_high=0.9, humidity_low=0.35, temperature_c=25.0)
econ = LCOEconomicParams()
names = feasible_salts_for_site(site, CANDIDATE_SALTS)
result = optimize_zsr_blend_and_sl(site, names, econ)

print(f"LCOW: ${result.best_lcow:.4f}/kg water")
print(f"Salt-to-polymer ratio: {result.best_sl:.3f}")
for name, weight in zip(result.names, result.best_f):
    if weight > 0.01:
        print(f"  {name}: {weight:.3f}")
```

## Economic parameters

All parameters are in `LCOEconomicParams` (see `src/optimization/economics.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `discount_rate` | 0.08 | WACC / annual discount rate |
| `device_lifetime_years` | 10 | Device economic lifetime |
| `total_investment_factor` | 1.0 | Scales capital cost for indirect costs |
| `maintenance_cost_fraction` | 0.05 | Annual maintenance as fraction of capital |
| `utilization_factor` | 0.9 | Fraction of days the device operates |
| `hydrogel_lifetime_years` | 1.0 | Hydrogel replacement interval |
| `energy_cost_usd_per_year` | 0.0 | Auxiliary energy cost (e.g. pumping) |

## Running tests

```bash
pytest
```
