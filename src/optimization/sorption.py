"""Salt uptake U(RH, SL) and diurnal half-swing; water activity from equilibrium mf."""

from __future__ import annotations

import math

from src.materials.salts import SaltRecord, get_salt
from src.optimization.mf_equilibrium import equilibrate_salt_mf

MOLAR_MASS_H2O: float = 18.015  # g/mol


def water_mole_fraction_from_mf(
    mf: float,
    nu: int,
    mw_salt_g_mol: float,
    mw_water: float = MOLAR_MASS_H2O,
) -> float:
    """Mole-based water fraction in a binary salt+water brine.

    n_w = (1-mf)/M_w, n_s = mf/MW_s; effective ionic count nu per formula; we use
    x_w = n_w / (n_w + nu * n_s) (colligative particle basis).
    """
    if mf <= 0.0 or mf >= 1.0 or not math.isfinite(mf):
        return float("nan")
    mw = mw_water
    ms = float(mw_salt_g_mol)
    n_w = (1.0 - mf) / mw
    n_s = mf / ms
    den = n_w + float(nu) * n_s
    if den <= 0.0:
        return float("nan")
    return n_w / den


def water_sorption_factor(rh: float, mf: float, nu: int, mw_salt_g_mol: float) -> float:
    """The dimensionless group ( RH/gamma_w * nu ) / (1 - RH/gamma_w) with a_w=RH, gamma_w=RH/x_w.

    If a_w = x_w * gamma_w and at equilibrium a_w = RH, then gamma_w = RH / x_w (x_w>0) and
    RH/gamma_w = x_w, so the group reduces to (x_w * nu) / (1 - x_w).
    """
    x_w = water_mole_fraction_from_mf(mf, nu, mw_salt_g_mol)
    if not math.isfinite(x_w) or x_w <= 0.0 or x_w >= 1.0:
        return float("nan")
    return (x_w * float(nu)) / (1.0 - x_w)


def salt_uptake_U(
    sl: float,
    rh: float,
    rec: SaltRecord,
    t_c: float = 25.0,
) -> float:
    """U = m_w / (m_s + m_p) from manuscript (one equilibrium RH)."""
    if sl <= 0.0:
        return float("nan")
    mf = equilibrate_salt_mf(rec.name, rh, t_c)
    if not math.isfinite(mf) or mf <= 0.0 or mf >= 1.0:
        return float("nan")
    term1 = 1.0 / (1.0 + 1.0 / sl)
    t2 = water_sorption_factor(rh, mf, rec.nu, rec.mw)
    if not math.isfinite(t2):
        return float("nan")
    term3 = MOLAR_MASS_H2O / rec.mw
    return float(term1 * t2 * term3)


def delta_U_half_swing(
    sl: float,
    rh_high: float,
    rh_low: float,
    rec: SaltRecord,
    t_c: float = 25.0,
) -> float:
    """(1/2) * (U(rh_high) - U(rh_low)) with non-negativity."""
    uh = salt_uptake_U(sl, rh_high, rec, t_c)
    ul = salt_uptake_U(sl, rh_low, rec, t_c)
    if not (math.isfinite(uh) and math.isfinite(ul)):
        return 0.0
    d = 0.5 * (uh - ul)
    return float(max(0.0, d))


def gross_annual_water_kg(
    delta_u: float,
    dry_mass_kg: float,
    cycles_per_year: int = 365,
) -> float:
    """Nameplate water mass per year; f_util is applied only in LCOW denominator."""
    return float(max(0.0, delta_u * dry_mass_kg * cycles_per_year))
