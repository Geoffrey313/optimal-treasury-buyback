"""Term-premium context and the buyback inversion (US-03 S4).

Anchors the pivot thesis on the ACM decomposition and turns it into the headline
inversion. Three guardrails are baked in and must survive into the manuscript:

1. The ACM decomposition shows the 10-year move over the program was a term-
   premium phenomenon (term premium up, expected-rate component down). It shows
   THAT the term premium rose, not WHY. This module never attributes the rise to
   supply; that would need the horse-race, and decomposing the term premium into
   supply / inflation / foreign demand is the macro paper, out of scope.
2. The term premium is used as motivation and context, not decomposed.
3. The buyback effect is TRIVIAL, not zero: the buyback acts on the term premium,
   so it moves the level, by only a few basis points. The inversion makes the
   "too small to matter" precise without claiming "no effect".

The inversion uses a term-premium elasticity of duration removal taken from the
quantitative-easing literature (a range, labeled as a calibration): removing one
hundred billion dollars of ten-year equivalents compresses the ten-year term
premium by roughly this many basis points. Our own issuance slope is not
distinguishable from zero, so it cannot pin the elasticity; the literature range
is used and clearly flagged.
"""
from __future__ import annotations

import pandas as pd

from src.common.config import SAMPLE_START
from src.common.paths import DATA_CURVES

WINDOW_END = "2026-09-30"
# QE-literature term-premium elasticity of duration removal, basis points of the
# 10-year term premium per $100bn of 10-year equivalents. A calibrated RANGE, not
# an estimate from this paper's data.
ELASTICITY_BP_PER_100BN = (2.0, 5.0)
ACM_PARQUET = DATA_CURVES / "acm_term_premium.parquet"


def _near(df: pd.DataFrame, dt: str) -> pd.Series:
    i = (df["DATE"] - pd.Timestamp(dt)).abs().idxmin()
    return df.loc[i]


def acm_decomposition() -> dict[str, float]:
    """Change in the ACM 10-year fitted yield, term premium, and risk-neutral
    (expected-rate) component over the program window, in basis points."""
    df = pd.read_parquet(ACM_PARQUET)
    df["DATE"] = pd.to_datetime(df["DATE"])
    r0, r1 = _near(df, SAMPLE_START), _near(df, WINDOW_END)
    return {
        "acm_start": r0["DATE"].date().isoformat(),
        "acm_end": r1["DATE"].date().isoformat(),
        "d_yield_bp": 100.0 * (r1["ACMY10"] - r0["ACMY10"]),
        "d_term_premium_bp": 100.0 * (r1["ACMTP10"] - r0["ACMTP10"]),
        "d_expected_rate_bp": 100.0 * (r1["ACMRNY10"] - r0["ACMRNY10"]),
    }


def inversion(ls_removed_10y_equiv_bn: float, d_term_premium_bp: float) -> dict[str, float]:
    """Trivial-effect and QE-scale inversion.

    Given the buyback's ten-year-equivalent removal and the observed term-premium
    move, report the term-premium compression the buyback plausibly achieved and
    the removal that would have been needed to offset the whole term-premium rise.
    """
    lo, hi = ELASTICITY_BP_PER_100BN
    comp_lo = ls_removed_10y_equiv_bn / 100.0 * lo
    comp_hi = ls_removed_10y_equiv_bn / 100.0 * hi
    needed_lo = d_term_premium_bp / hi * 100.0   # smaller need with the higher elasticity
    needed_hi = d_term_premium_bp / lo * 100.0
    return {
        "buyback_10y_equiv_removed_bn": ls_removed_10y_equiv_bn,
        "tp_compression_achieved_bp_low": comp_lo,
        "tp_compression_achieved_bp_high": comp_hi,
        "removal_to_offset_tp_bn_low": needed_lo,
        "removal_to_offset_tp_bn_high": needed_hi,
        "buyback_share_of_needed_pct_low": 100.0 * ls_removed_10y_equiv_bn / needed_hi,
        "buyback_share_of_needed_pct_high": 100.0 * ls_removed_10y_equiv_bn / needed_lo,
    }


def compute(panel: dict[str, pd.DataFrame]) -> dict[str, float]:
    from src.analysis.duration_accounting import compute as duration_compute
    dec = acm_decomposition()
    da = duration_compute(panel)
    inv = inversion(da["ls_removed_10y_equiv_bn"], dec["d_term_premium_bp"])
    return {**dec, **inv}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from src.data.panel import build_panels
    r = compute(build_panels())
    for k, v in r.items():
        print(f"{k:36s} {v}")
