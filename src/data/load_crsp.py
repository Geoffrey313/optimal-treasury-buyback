"""Load the locally cached CRSP Treasury tables (no network access).

These loaders read the parquet files of the licensed CRSP layer into pandas
DataFrames. They perform no WRDS access, so they run without credentials once a
subscriber has regenerated that layer under ``data/crsp/`` (not redistributed;
see ``data/crsp/README.md`` for the library and tables to pull). Callers
downstream (panel assembly) import from here rather than reaching for the
parquet paths directly.
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
            f"{path} not found. The CRSP layer is licensed and not "
            f"redistributed; regenerate it as described in data/crsp/README.md."
        )
    return pd.read_parquet(path)


def load_tfz_dly() -> pd.DataFrame:
    """Return the per-CUSIP daily CRSP Treasury panel (``tfz_dly``)."""
    return _load(TFZ_DLY_PARQUET)


def load_tfz_iss() -> pd.DataFrame:
    """Return the CRSP Treasury issue crosswalk (``tfz_iss``)."""
    return _load(TFZ_ISS_PARQUET)
