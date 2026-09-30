"""Load the locally cached CRSP Treasury tables (no network access).

These loaders read the parquet files written by
``src.data.acquire.crsp`` into pandas DataFrames. They perform no WRDS access,
so they run without credentials once the licensed layer has been regenerated
under ``data/crsp/`` (gitignored). Callers downstream (panel assembly) import
from here rather than reaching for the parquet paths directly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.paths import DATA_CRSP

TFZ_DLY_PARQUET: Path = DATA_CRSP / "tfz_dly.parquet"
TFZ_ISS_PARQUET: Path = DATA_CRSP / "tfz_iss.parquet"


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Regenerate the CRSP layer first: "
            f"python -m src.data.acquire.crsp"
        )
    return pd.read_parquet(path)


def load_tfz_dly() -> pd.DataFrame:
    """Return the per-CUSIP daily CRSP Treasury panel (``tfz_dly``)."""
    return _load(TFZ_DLY_PARQUET)


def load_tfz_iss() -> pd.DataFrame:
    """Return the CRSP Treasury issue crosswalk (``tfz_iss``)."""
    return _load(TFZ_ISS_PARQUET)
