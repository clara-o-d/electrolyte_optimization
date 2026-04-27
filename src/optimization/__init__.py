"""SAWH workflow: climate aggregation, solvers, sorption, costs.

Avoid importing the Ipopt/solve path at package import to prevent circular
imports with :mod:`src.models`. Use ``from src.optimization.solve import ...``.
"""

from .climate import diurnal_rh_from_hourly, mean_lcow_for_grid, site_row_from_hourly
from .economics import (
    C_DEVICE_USD,
    DRY_COMPOSITE_MASS_KG,
    HYDROGEL_DENSITY_KG_M3,
    HYDROGEL_THICKNESS_M,
    LCOEconomicParams,
    annual_operating_plus_capital_usd,
    hydrogel_cost_usd_per_kg_composite,
    lcow_usd_per_kg_water,
)
from .mf_equilibrium import equilibrate_salt_mf, mf_CaCl2, mf_LiCl, mf_MgCl2, mf_NaCl
from .sorption import (
    MOLAR_MASS_H2O,
    delta_U_half_swing,
    gross_annual_water_kg,
    salt_uptake_U,
    water_mole_fraction_from_mf,
    water_sorption_factor,
)

__all__ = [
    "C_DEVICE_USD",
    "DRY_COMPOSITE_MASS_KG",
    "HYDROGEL_DENSITY_KG_M3",
    "HYDROGEL_THICKNESS_M",
    "LCOEconomicParams",
    "MOLAR_MASS_H2O",
    "annual_operating_plus_capital_usd",
    "diurnal_rh_from_hourly",
    "delta_U_half_swing",
    "equilibrate_salt_mf",
    "gross_annual_water_kg",
    "hydrogel_cost_usd_per_kg_composite",
    "lcow_usd_per_kg_water",
    "mean_lcow_for_grid",
    "mf_CaCl2",
    "mf_LiCl",
    "mf_MgCl2",
    "mf_NaCl",
    "site_row_from_hourly",
    "salt_uptake_U",
    "water_mole_fraction_from_mf",
    "water_sorption_factor",
]
