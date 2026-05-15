# Electrolyte Optimization for Sorbent Atmospheric Water Harvesting

This repo models and optimizes salt-hydrogel composites for **sorbent atmospheric water harvesting (SAWH)** — devices that passively absorb water vapor from humid air and release it as liquid water when heated. The goal is to find the salt blend and composite formulation that minimizes the **levelized cost of water (LCOW, $/kg)** at a given geographic site.

## Background

A SAWH device cycles between two operating states each day:

- **Night uptake:** the hydrogel is open to humid night air at ambient temperature. The salt dissolved in the hydrogel lowers water activity, driving vapor absorption until the brine reaches equilibrium with the ambient relative humidity.
- **Day desorption:** the hydrogel is **sealed off from the atmosphere** while sunlight heats it. The gel temperature is computed from a steady-state energy balance — absorbed solar irradiance against convective + radiative losses to ambient:

  ```
  α · I_solar = h_conv · (T_gel − T_amb) + ε · σ · (T_gel⁴ − T_amb⁴)
  ```

  (desorption enthalpy and the gel/condenser radiative coupling are neglected; the condenser is taken to track ambient). Water vapor then leaves the gel toward the condenser. Equating the molar vapor concentrations above the gel and the condenser gives an effective water activity in the brine of

  ```
  a_w = P_sat(T_cond) · T_gel / (P_sat(T_gel) · T_cond)        (T in K)
  ```

  The brine in the gel reaches the salt concentration whose isotherm produces this water activity at `T_gel` (i.e. `x_w · γ_w(c_salt, T_gel) = a_w`).

The amount of water produced per cycle is the **difference in water uptake** between the night uptake and day desorption states. The four candidate salts — **LiCl, NaCl, CaCl2, MgCl2** — each have different vapor pressure isotherms, deliquescence ranges, and prices, so blending them can improve yield or reduce cost compared to any single salt.

## What the optimization does

For a site described by its mean nightly maximum relative humidity and ambient temperature, the optimizer finds:

1. **Blend weights** `f_i` for each salt (fraction of the brine attributed to salt `i`, summing to 1)
2. **Salt-to-polymer ratio** (mass of salt per mass of acrylamide polymer in the dry composite)

...to minimize LCOW:

```
LCOW = annual_cost_USD / (utilization_factor × annual_water_yield_kg)
```

The brine state at each operating point is computed using the **ZSR (Zdanovskii) isopiestic mixing rule**: at fixed water activity, each salt contributes molality in proportion to its blend weight and its binary reference molality (the molality of a pure salt solution that is in equilibrium at that water activity and temperature). The two operating points are uptake (`a_w = humidity_high`, `T = T_amb`) and desorption (`a_w` from the sealed gel/condenser equation above, `T = T_gel`).

## Repository structure

```
src/
  materials/
    salts.py              Salt records: formula weight, ion count, RH range, price
    salt_prices.py        Loads bulk salt prices from salt_price_data.xlsx

  optimization/
    brine_equilibrium.py  Isotherm fits: equilibrium brine salt fraction at a given RH
                          Also converts between mass fraction and molality
    brine_uptake.py       Sorption factor physics: water mole fraction, uptake coefficient B,
                          saturation vapor pressure, sealed-gel desorption water activity
    heat_transfer.py      Steady-state gel temperature from absorbed solar irradiance
                          balanced against convective + radiative losses to ambient
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

where `x_w` is the water mole fraction in the brine (colligative basis). The half-cycle water yield per kg of composite is proportional to `(B_uptake − B_desorption)`.

**Sealed-gel desorption** (`brine_uptake.py`): during the day the gel is closed off and heated by the sun. Equating molar vapor concentrations above the gel and the condenser gives an effective water activity `a_w = P_sat(T_cond) · T_gel / (P_sat(T_gel) · T_cond)` (absolute Kelvin). The brine in the gel reaches the salt concentration whose isotherm at `T_gel` matches this `a_w`; i.e. `x_w · γ_w = a_w`, where `γ_w` is the activity coefficient of water at that salt concentration and temperature (encoded by the salt isotherm fits).

**ZSR mixing** (`zsr_mixing.py`): at fixed water activity, the mixture molality of salt `i` is `blend_weight_i × reference_molality_i`. Effective mixture properties (ion count, formula weight) are molality-weighted averages across salts.

**Pyomo NLP** (`zsr_lcow_model.py`): the blend weights and salt-to-polymer ratio are continuous decision variables. Constraints enforce that weights sum to 1 and that the night-time uptake loading exceeds the day-time desorption loading (positive half-swing). The objective is LCOW.

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

site = SiteClimate(humidity_high=0.9, temperature_c=25.0)  # gel_temperature_c defaults to 70 C
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
| `device_lifetime_years` | 20 | Device economic lifetime |
| `total_investment_factor` | 1.0 | Scales capital cost for indirect costs |
| `maintenance_cost_fraction` | 0.05 | Annual maintenance as fraction of capital |
| `utilization_factor` | 0.9 | Fraction of days the device operates |
| `hydrogel_lifetime_years` | 1.0 | Hydrogel replacement interval |
| `c_acrylamide_usd_per_kg` | 1.1 | Acrylamide (polymer matrix) cost ($/kg polymer) |
| `c_additives_usd_per_kg_composite` | ~0.0069 | APS + MBAA + TEMED per kg composite |
| `energy_cost_usd_per_year` | 0.0 | Auxiliary energy cost (e.g. pumping) |

## Running tests

```bash
pytest
```
