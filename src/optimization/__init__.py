"""SAWH workflow: climate aggregation, ZSR mixing, brine equilibrium, and costs."""

from .brine_equilibrium import (
    binary_molality_at_rh,
    equilibrate_salt_mf,
    mf_CaCl2,
    mf_LiCl,
    mf_MgCl2,
    mf_NaCl,
    molality_to_salt_fraction,
    salt_fraction_to_molality,
)
from .brine_uptake import WATER_MOLAR_MASS_G_MOL, water_sorption_factor
from .climate import diurnal_rh_from_hourly, site_row_from_hourly
from .economics import (
    C_DEVICE_USD,
    DRY_COMPOSITE_MASS_KG,
    HYDROGEL_DENSITY_KG_M3,
    HYDROGEL_THICKNESS_M,
    LCOEconomicParams,
)

__all__ = [
    "C_DEVICE_USD",
    "DRY_COMPOSITE_MASS_KG",
    "HYDROGEL_DENSITY_KG_M3",
    "HYDROGEL_THICKNESS_M",
    "LCOEconomicParams",
    "WATER_MOLAR_MASS_G_MOL",
    "binary_molality_at_rh",
    "diurnal_rh_from_hourly",
    "equilibrate_salt_mf",
    "mf_CaCl2",
    "mf_LiCl",
    "mf_MgCl2",
    "mf_NaCl",
    "molality_to_salt_fraction",
    "salt_fraction_to_molality",
    "site_row_from_hourly",
    "water_sorption_factor",
]
