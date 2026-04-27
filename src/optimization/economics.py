"""Levelized cost of water (LCOW) economics: capital recovery, device and hydrogel BOM."""

from __future__ import annotations

from dataclasses import dataclass

# --- Physical scale (active hydrogel–salt layer, 1 m²) ---
# Thickness 1 mm, density 2000 kg/m³ => mass per m²
HYDROGEL_THICKNESS_M: float = 0.001
HYDROGEL_DENSITY_KG_M3: float = 2000.0
DRY_COMPOSITE_MASS_KG: float = HYDROGEL_THICKNESS_M * 1.0 * HYDROGEL_DENSITY_KG_M3  # 2.0

# --- Table S2. Passive SAWH device, $/m² footprint (BOM) ---
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
# = 52.261 USD (rounded in docstrings; keep full precision in code)

# Table S1 small components (per kg of composite) — g/mL converted to kg
# APS, MBAA, TEMED from supplement (fixed; do not double-count with acrylamide)
_ADDITIVE_USD_PER_KG: float = 0.00068 * 0.61 + 0.0024 * 1.0 + 0.00044 * 10.0
_ACRYLAMIDE_USD_PER_KG: float = 1.6


@dataclass(frozen=True, slots=True)
class LCOEconomicParams:
    """Economic and utilization parameters for LCOW (all money USD unless noted).

    **Yield convention (no double counting of f_util):** ``gross_annual_water_kg`` is
    “nameplate” water per year (365 × ΔU × dry mass) *without* applying ``f_util``.
    LCOW uses::

        LCOW = annual_sum_costs / (f_util * gross_annual_water_kg)

    so ``f_util`` appears only in the denominator, matching the supplement figure.
    """

    f_wacc: float = 0.08
    L_years: int = 10
    f_toti: float = 1.0
    f_mlc: float = 0.05
    f_util: float = 0.9
    tau_hyd_years: float = 1.0
    C_energy_usd_per_year: float = 0.0
    c_acrylamide_usd_per_kg: float = _ACRYLAMIDE_USD_PER_KG
    c_additives_usd_per_kg_composite: float = _ADDITIVE_USD_PER_KG

    def f_crf(self) -> float:
        """Capital recovery factor from WACC and device lifetime."""
        i = self.f_wacc
        L = self.L_years
        if i <= 0.0 or L < 1:
            raise ValueError("f_wacc must be > 0 and L_years >= 1 for f_crf")
        return (i * (1.0 + i) ** L) / ((1.0 + i) ** L - 1.0)


def hydrogel_cost_usd_per_kg_composite(
    c_salt_usd_per_kg: float,
    sl: float,
    *,
    c_acrylamide: float = _ACRYLAMIDE_USD_PER_KG,
    c_add: float = _ADDITIVE_USD_PER_KG,
) -> float:
    """$ per kg dry (salt + polymer) composite from Table S1-style split.

    For 1 kg total dry, m_s = SL/(1+SL), m_p = 1/(1+SL); acrylamide is charged on m_p;
    ``c_add`` is a fixed $/kg of final composite (small components).
    """
    if sl <= 0.0:
        raise ValueError("sl must be positive")
    inv = 1.0 + sl
    m_s = sl / inv
    m_p = 1.0 / inv
    return c_salt_usd_per_kg * m_s + c_acrylamide * m_p + c_add


def annual_operating_plus_capital_usd(
    p: LCOEconomicParams,
    *,
    c_hyd_usd_per_kg: float,
    dry_mass_kg: float = DRY_COMPOSITE_MASS_KG,
) -> float:
    """Numerator of LCOW: ann. capital + replacement + fixed O&M + variable O&M."""
    c_dev = C_DEVICE_USD
    f_c = p.f_crf()
    rep = c_hyd_usd_per_kg * dry_mass_kg / p.tau_hyd_years
    return f_c * p.f_toti * c_dev + rep + p.f_mlc * p.f_toti * c_dev + p.C_energy_usd_per_year


def lcow_usd_per_kg_water(
    annual_sum_costs: float,
    f_util: float,
    gross_annual_water_kg: float,
) -> float:
    """Levelized $/kg water. Uses gross water × f_util in denominator (single f_util)."""
    if f_util <= 0.0 or gross_annual_water_kg <= 0.0:
        return float("inf")
    return annual_sum_costs / (f_util * gross_annual_water_kg)
