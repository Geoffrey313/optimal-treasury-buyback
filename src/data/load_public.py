"""Load the transformed PUBLIC Treasury panels from parquet (no network).

These loaders read the parquet files written by
``src.data.acquire.public_sources`` and return pandas DataFrames. They never
reach the network; regenerate the parquet with the acquisition module if a
file is missing. Output directories come from ``src.common.paths``.
"""
from __future__ import annotations

import pandas as pd

from src.common import paths


def _read(directory, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run "
            "`python -m src.data.acquire.public_sources` to regenerate it."
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
