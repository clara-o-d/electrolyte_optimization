"""Load bulk salt $/kg from ``src/data/salt_pricing/salt_price_data.xlsx``."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

_COL_SALT = "Salt Formula"
_COL_USD_KG = "$/kg"


def salt_price_data_path() -> Path:
    """Path to the project’s ``salt_price_data.xlsx`` (under ``src/data/salt_pricing/``)."""
    return Path(__file__).resolve().parent.parent / "data" / "salt_pricing" / "salt_price_data.xlsx"


def load_salt_prices_usd_per_kg(
    xlsx: Path | str | None = None,
) -> dict[str, float]:
    """Return mapping **salt name → bulk price (USD per kg)** from the workbook.

    Uses the ``$/kg`` column. Rows with missing or non-positive $/kg are skipped.

    If the file is missing, ``openpyxl`` is not installed, or read fails, returns
    an empty dict (callers should keep their defaults).
    """
    path = Path(xlsx) if xlsx is not None else salt_price_data_path()
    if not path.is_file():
        warnings.warn(f"Salt price data not found at {path}; using code defaults.", stacklevel=2)
        return {}
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        warnings.warn(f"pandas not available for salt prices: {e}", stacklevel=2)
        return {}
    try:
        df: Any = pd.read_excel(path, engine="openpyxl")
    except ImportError:
        warnings.warn("Install openpyxl to read salt_price_data.xlsx: pip install openpyxl", stacklevel=2)
        return {}
    except Exception as e:
        warnings.warn(f"Could not read {path}: {e}", stacklevel=2)
        return {}
    if _COL_SALT not in df.columns or _COL_USD_KG not in df.columns:
        warnings.warn(
            f"Expected columns {_COL_SALT!r} and {_COL_USD_KG!r} in {path}.",
            stacklevel=2,
        )
        return {}
    out: dict[str, float] = {}
    for _, srow in df.iterrows():
        raw = srow.get(_COL_SALT)
        if raw is None or (isinstance(raw, float) and raw != raw):
            continue
        name = str(raw).strip()
        if not name:
            continue
        p = srow.get(_COL_USD_KG, float("nan"))
        try:
            price = float(p)
        except (TypeError, ValueError):
            continue
        if not (price == price) or price <= 0.0:
            continue
        out[name] = float(price)
    return out
