"""Sun-driven gel temperature from a steady-state energy balance.

Models the sealed hydrogel as a single thermal node that absorbs solar radiation
(and optional electrical resistive heating) and loses heat to ambient via
convection and thermal radiation. This is the gel analog of the condenser
energy balance in the SAWH device modeling literature
(e.g. ``(rho cp L)_cond dT_cond/dt = h_conv,g (T_gel - T_cond) - h_conv,cond
(T_cond - T_amb) + m_des h_fg + eps sigma (T_gel^4 - T_cond^4)``).

Simplifications applied here:

* Steady state: ``dT_gel/dt = 0`` (the cycle period is much longer than the
  gel's thermal time constant).
* Desorption enthalpy is neglected (water release is energetically small
  compared to the absorbed solar flux for typical bulk-salt hydrogels).
* The condenser is taken to track ambient temperature, so the gel radiates and
  convects ultimately to ambient rather than via a separately solved condenser
  node.

The resulting balance is::

    alpha * I_solar + Q_elec = h_conv * (T_gel - T_amb) + epsilon * sigma * (T_gel^4 - T_amb^4)

where ``Q_elec`` (W/m^2) is an optional electrical heating power density that
lets the device be actively driven rather than purely sun-driven. The balance
is solved for ``T_gel`` with a 1-D bracketed root finder.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

STEFAN_BOLTZMANN_W_M2_K4: float = 5.670374419e-8
"""Stefan-Boltzmann constant (W / m^2 / K^4)."""

DEFAULT_SOLAR_ABSORPTIVITY: float = 0.9
"""Fraction of incident plane-of-gel solar irradiance absorbed by the gel surface."""

DEFAULT_GEL_CONVECTION_W_M2_K: float = 10.0
"""Combined convective heat-transfer coefficient between the gel and ambient air.

Natural convection from a heated horizontal surface in still air is typically
5-25 W/m^2/K; 10 W/m^2/K is a reasonable still-air estimate. Increase for windy
conditions.
"""

DEFAULT_GEL_EMISSIVITY: float = 1.0
"""Thermal emissivity of the gel surface (dimensionless); 1.0 matches liquid water."""

DEFAULT_SOLAR_IRRADIANCE_W_M2: float = 800.0
"""Default plane-of-gel solar irradiance (W/m^2).

Roughly clear-sky midday GHI on a sunny day at moderate latitudes. Override
with site-specific irradiance via ``SiteClimate.solar_irradiance_w_per_m2``.
"""


def gel_steady_state_temperature_c(
    solar_irradiance_w_per_m2: float,
    ambient_temperature_c: float,
    *,
    electric_heat_w_per_m2: float = 0.0,
    absorptivity: float = DEFAULT_SOLAR_ABSORPTIVITY,
    convection_coefficient_w_m2_k: float = DEFAULT_GEL_CONVECTION_W_M2_K,
    emissivity: float = DEFAULT_GEL_EMISSIVITY,
) -> float:
    """Steady-state hydrogel temperature (deg C) from a single-node energy balance.

    Sets the absorbed solar power plus any electrical heating power equal to
    convective + radiative losses to ambient:

        alpha * I + Q_elec = h_conv * (T_gel - T_amb) + epsilon * sigma * (T_gel^4 - T_amb^4)

    With both ``solar_irradiance_w_per_m2 <= 0`` and ``electric_heat_w_per_m2 <= 0``
    (e.g. passive at night), the gel cannot heat above ambient and the function
    returns ``ambient_temperature_c``.
    """
    solar = solar_irradiance_w_per_m2 if math.isfinite(solar_irradiance_w_per_m2) else 0.0
    solar = max(0.0, solar)
    q_elec = electric_heat_w_per_m2 if math.isfinite(electric_heat_w_per_m2) else 0.0
    q_elec = max(0.0, q_elec)
    absorbed = absorptivity * solar + q_elec
    if absorbed <= 0.0:
        return float(ambient_temperature_c)
    t_amb_k = ambient_temperature_c + 273.15
    if t_amb_k <= 0.0 or not math.isfinite(t_amb_k):
        return float("nan")

    def residual(t_gel_c: float) -> float:
        t_gel_k = t_gel_c + 273.15
        conv = convection_coefficient_w_m2_k * (t_gel_c - ambient_temperature_c)
        rad = emissivity * STEFAN_BOLTZMANN_W_M2_K4 * (t_gel_k**4 - t_amb_k**4)
        return absorbed - conv - rad

    # T_gel >= T_amb (no other heating sources); upper bound is when pure
    # convection alone would dissipate all the absorbed power.
    lo = ambient_temperature_c
    h = max(convection_coefficient_w_m2_k, 1e-9)
    hi = ambient_temperature_c + absorbed / h + 1.0
    f_lo = residual(lo)
    f_hi = residual(hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_lo * f_hi > 0.0:
        return float("nan")
    return float(brentq(residual, lo, hi, maxiter=200))
