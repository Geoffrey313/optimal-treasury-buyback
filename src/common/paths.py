"""Portable path resolution for the reproduction package.

All paths resolve relative to the repository root, discovered from this file's
location. No machine-specific absolute path appears anywhere in the code base.
"""
from __future__ import annotations

from pathlib import Path

# repo_root/src/common/paths.py -> repo_root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = REPO_ROOT / "data"
DATA_BUYBACKS: Path = DATA_DIR / "buybacks"
DATA_AUCTIONS: Path = DATA_DIR / "auctions"
DATA_CURVES: Path = DATA_DIR / "curves"
DATA_CRSP: Path = DATA_DIR / "crsp"

# Generated numeric outputs (reproducible; gitignored, never shipped).
RESULTS_DIR: Path = REPO_ROOT / "results"

# Manuscript figure targets (language-specific), per research-repo-rules.
MANUSCRIPT_DIR: Path = REPO_ROOT / "manuscript"
FIGURES_EN: Path = MANUSCRIPT_DIR / "en" / "ssrn" / "figures"
FIGURES_FR: Path = MANUSCRIPT_DIR / "fr" / "ssrn" / "figures"


def figures_dir(language: str) -> Path:
    """Return the manuscript figure directory for a language ('en' or 'fr')."""
    if language not in {"en", "fr"}:
        raise ValueError(f"language must be 'en' or 'fr', got {language!r}")
    return FIGURES_EN if language == "en" else FIGURES_FR
