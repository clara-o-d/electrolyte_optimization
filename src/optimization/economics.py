"""Levelized cost of water (LCOW) economics: capital recovery, device and hydrogel BOM."""

from __future__ import annotations

from dataclasses import dataclass

# --- Physical scale (active hydrogel–salt layer, 1 m²) ---
# Thickness 1 mm, density 2000 kg/m³ => 2.0 kg dry composite per m²
HYDROGEL_THICKNESS_M: float = 0.001
HYDROGEL_DENSITY_KG_M3: float = 2000.0
DRY_COMPOSITE_MASS_KG: float = HYDROGEL_THICKNESS_M * 1.0 * HYDROGEL_DENSITY_KG_M3

# --- Passive SAWH device bill of materials, USD per m² footprint (Table S2) ---
# Aluminum heater 5.42 kg @ 2.1 $/kg, condenser 13.55 @ 2.1, acrylic 0.004 m³ @ 1487.5,
# insulation 4 m² @ 0.5, clamps 1.93 @ 1.8, fixtures 100 @ 0.01
C_DEVICE_USD: float = (
    5.42 * 2.1
    + 13.55 * 2.1
    + 0.004 * 1487.5
    + 4.0 * 0.5
    + 1.93 * 1.8
    + 100.0 * 0.01
)

# Small composite additives per kg (Table S1): APS, MBAA, TEMED
_ADDITIVE_USD_PER_KG: float = 0.00068 * 0.61 + 0.0024 * 1.0 + 0.00044 * 10.0
_ACRYLAMIDE_USD_PER_KG: float = 1.6


@dataclass(frozen=True, slots=True)
class LCOEconomicParams:
    """Economic and utilization parameters for LCOW (all money in USD unless noted).

    LCOW = annual_cost / (utilization_factor * gross_annual_water_kg)

    gross_annual_water_kg is nameplate yield (365 × half_swing × dry_mass) without
    applying utilization_factor; it appears only in the denominator.
    """

    discount_rate: float = 0.08
    device_lifetime_years: int = 10
    total_investment_factor: float = 1.0
    maintenance_cost_fraction: float = 0.05
    utilization_factor: float = 0.9
    hydrogel_lifetime_years: float = 1.0
    energy_cost_usd_per_year: float = 0.0
    c_acrylamide_usd_per_kg: float = _ACRYLAMIDE_USD_PER_KG
    c_additives_usd_per_kg_composite: float = _ADDITIVE_USD_PER_KG

    def capital_recovery_factor(self) -> float:
        """Annualized fraction of capital cost: i(1+i)^L / ((1+i)^L - 1)."""
        i = self.discount_rate
        L = self.device_lifetime_years
        if i <= 0.0 or L < 1:
            raise ValueError("discount_rate must be > 0 and device_lifetime_years >= 1")
        return (i * (1.0 + i) ** L) / ((1.0 + i) ** L - 1.0)
