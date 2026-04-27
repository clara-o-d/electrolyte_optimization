"""Salt / sorbent records (from ``load_salt_data.m``-style fields; Python only)."""

from __future__ import annotations

from dataclasses import dataclass

from src.materials.salt_prices import load_salt_prices_usd_per_kg


@dataclass(frozen=True, slots=True)
class SaltRecord:
    name: str
    mw: float
    """Formula weight (g/mol), same convention as the MATLAB table."""
    rh_min: float
    rh_max: float
    nu: int
    """Ions (van't Hoff factor) per formula unit, e.g. 2 for NaCl, 3 for CaCl2."""
    needs_temperature: bool
    c_salt_usd_per_kg: float
    """Bulk salt $/kg (sparse data; use mid LiCl 0.55 as default where unknown)."""

    @property
    def mw_kg_mol(self) -> float:
        return self.mw * 1e-3


# Fallback $/kg if a salt is missing from ``salt_price_data.xlsx`` or price ≤ 0
_DEFAULT_SALT_USD: float = 0.55
_DEFAULT_BY_SALT: dict[str, float] = {
    "LiCl": _DEFAULT_SALT_USD,
    "NaCl": 0.2,
    "CaCl2": 0.15,
    "MgCl2": 0.15,
}

_PRICES_XLSX: dict[str, float] = load_salt_prices_usd_per_kg()


def _c_salt(name: str) -> float:
    """$/kg: workbook first, else :data:`_DEFAULT_BY_SALT`."""
    if name in _PRICES_XLSX:
        return _PRICES_XLSX[name]
    return _DEFAULT_BY_SALT.get(name, _DEFAULT_SALT_USD)


SALT_TABLE: dict[str, SaltRecord] = {
    "LiCl": SaltRecord(
        name="LiCl",
        mw=42.4,
        rh_min=0.12,
        rh_max=0.97,
        nu=2,
        needs_temperature=True,
        c_salt_usd_per_kg=_c_salt("LiCl"),
    ),
    "NaCl": SaltRecord(
        name="NaCl",
        mw=58.443,
        rh_min=0.765,
        rh_max=0.99,
        nu=2,
        needs_temperature=False,
        c_salt_usd_per_kg=_c_salt("NaCl"),
    ),
    "CaCl2": SaltRecord(
        name="CaCl2",
        mw=111.0,
        rh_min=0.31,
        rh_max=0.97,
        nu=3,
        needs_temperature=True,
        c_salt_usd_per_kg=_c_salt("CaCl2"),
    ),
    "MgCl2": SaltRecord(
        name="MgCl2",
        mw=95.2,
        rh_min=0.33,
        rh_max=0.97,
        nu=3,
        needs_temperature=False,
        c_salt_usd_per_kg=_c_salt("MgCl2"),
    ),
}

CANDIDATE_SALTS: tuple[str, ...] = tuple(SALT_TABLE.keys())


def get_salt(name: str) -> SaltRecord:
    if name not in SALT_TABLE:
        raise KeyError(f"Unknown salt: {name!r}; known: {sorted(SALT_TABLE)}")
    return SALT_TABLE[name]
