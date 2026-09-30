"""Load the transformed PUBLIC Treasury panels from parquet (no network).

These loaders read the public parquet files shipped in ``data/`` and return
pandas DataFrames: buyback operations, per-CUSIP operation details, auction
results, and the daily Treasury par-yield curve. They never reach the network,
and their directories come from :mod:`src.common.paths`. See ``data/README.md``
for the layout and the source of each file.
"""
from __future__ import annotations

import pandas as pd

from src.common import paths


def _read(directory, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; this public layer ships with the repository, "
            "see data/README.md for its expected layout."
        )
    return pd.read_parquet(path)


def load_buyback_operations() -> pd.DataFrame:
    """One row per buyback operation (Fiscal Data buybacks_operations)."""
    return _read(paths.DATA_BUYBACKS, "operations.parquet")


def load_buyback_security_details() -> pd.DataFrame:
    """One row per eligible security per operation (buybacks_security_details)."""
    return _read(paths.DATA_BUYBACKS, "security_details.parquet")


def load_auctions() -> pd.DataFrame:
    """One row per auction (Fiscal Data auctions_query)."""
    return _read(paths.DATA_AUCTIONS, "auctions.parquet")


def load_par_yields() -> pd.DataFrame:
    """Daily Treasury par yield curve, stacked across covered years."""
    return _read(paths.DATA_CURVES, "par_yields.parquet")
