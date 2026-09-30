"""Term-premium context and the buyback inversion.

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

from src.common.config import SAMPLE_START, WINDOW_END
from src.common.paths import DATA_CURVES


# Term-premium elasticity of duration supply, in basis points of the ten-year term
# premium per $100bn of ten-year equivalents withdrawn. A calibration taken from the
# asset-purchase literature, not an estimate from this paper's data.
#
# No study in that literature reports the elasticity in this unit: the term-structure
# estimates are stated per percentage point of GDP, so converting them requires a GDP
# level, and the dollar-metric studies are stated in par, so converting them requires a
# duration. Both endpoints below are therefore derived, and the derivation is published
# with them rather than left implicit.
#
# US nominal GDP, 2026Q2, seasonally adjusted annual rate, $bn. One percentage point of
# GDP is a hundredth of this, and it is the denominator every GDP-ratio coefficient has
# to be divided by to reach a dollar unit.
GDP_BN = 32486.1
# Low end. Gagnon, Raskin, Remache and Sack (2011), Table 5: a one-percentage-point-of-GDP
# increase in longer-term debt supply raises the ten-year term premium by 6.4 basis points
# when the supply is expressed in ten-year equivalents, which is this paper's own unit.
ELASTICITY_LOW_BP_PER_PP_GDP = 6.4
# High end. Bonis, Ihrig and Wei (2017): a term-premium effect of 100 basis points against
# a ten-year-equivalent holdings stock that rose from $330bn to $2,800bn.
ELASTICITY_HIGH_TPE_BP = 100.0
ELASTICITY_HIGH_STOCK_BN = 2800.0 - 330.0
ELASTICITY_BP_PER_100BN = (
    ELASTICITY_LOW_BP_PER_PP_GDP / (GDP_BN / 100.0) * 100.0,
    ELASTICITY_HIGH_TPE_BP / (ELASTICITY_HIGH_STOCK_BN / 100.0),
)
# Upper reference point, reported in the sensitivity grid but outside the calibrated
# range. D'Amico and King (2013) report a yield shift of about 30 basis points from
# $300bn of par purchases; at the par-to-ten-year-equivalent factor implied by the
# programs in Li and Wei (2013), that is of the order of 17 basis points per $100bn.
# It is not used for the calibration because it measures a shift in the level of the
# whole curve, in par dollars, over a crisis window, rather than a term premium per
# unit of duration withdrawn in a calm one. The grid reports it so the reading at that
# elasticity is visible rather than excluded silently.
ELASTICITY_UPPER_REFERENCE = 17.0
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
        # Endpoint levels in percent. Published so the manuscript can show the
        # differencing that produces each headline move rather than assert it.
        "acm_yield_start_pct": float(r0["ACMY10"]),
        "acm_yield_end_pct": float(r1["ACMY10"]),
        "acm_term_premium_start_pct": float(r0["ACMTP10"]),
        "acm_term_premium_end_pct": float(r1["ACMTP10"]),
        "acm_expected_rate_start_pct": float(r0["ACMRNY10"]),
        "acm_expected_rate_end_pct": float(r1["ACMRNY10"]),
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
        "elasticity_bp_per_100bn_low": lo,
        "elasticity_bp_per_100bn_high": hi,
        "buyback_10y_equiv_removed_bn": ls_removed_10y_equiv_bn,
        "tp_compression_achieved_bp_low": comp_lo,
        "tp_compression_achieved_bp_high": comp_hi,
        "removal_to_offset_tp_bn_low": needed_lo,
        "removal_to_offset_tp_bn_high": needed_hi,
        "buyback_share_of_needed_pct_low": 100.0 * ls_removed_10y_equiv_bn / needed_hi,
        "buyback_share_of_needed_pct_high": 100.0 * ls_removed_10y_equiv_bn / needed_lo,
    }


# The grid spans the whole converted literature, from below the calibrated low end to
# the upper reference point the calibration excludes, so the reading at every elasticity
# any cited study implies can be read off the table.
ELASTICITY_GRID = (1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 11.0, ELASTICITY_UPPER_REFERENCE)


def sensitivity_grid(removed_10y_equiv_bn: dict[str, float],
                     d_term_premium_bp: float) -> pd.DataFrame:
    """Compression and required withdrawal across the elasticity grid and both arms.

    The calibrated range is the middle of this grid; the rows on either side of it
    show how far the elasticity would have to move for the reading to change. Each
    arm is priced on its own withdrawal, so the table also answers whether counting
    every buyback operation rather than the liquidity support arm alone alters the
    conclusion."""
    rows = []
    for theta in ELASTICITY_GRID:
        row = {"elasticity_bp_per_100bn": theta,
               "offset_needed_tn": d_term_premium_bp / theta * 100.0 / 1000.0}
        for arm, removed in removed_10y_equiv_bn.items():
            row[f"compression_bp_{arm}"] = removed / 100.0 * theta
            row[f"share_of_needed_pct_{arm}"] = 100.0 * removed / (d_term_premium_bp / theta * 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def compute(panel: dict[str, pd.DataFrame]) -> dict[str, float]:
    from src.analysis.duration_accounting import compute as duration_compute
    dec = acm_decomposition()
    da = duration_compute(panel)
    inv = inversion(da["ls_removed_10y_equiv_bn"], dec["d_term_premium_bp"])
    inv_all = inversion(da["all_removed_10y_equiv_bn"], dec["d_term_premium_bp"])
    return {**dec, **inv,
            "tp_compression_all_ops_bp_low": inv_all["tp_compression_achieved_bp_low"],
            "tp_compression_all_ops_bp_high": inv_all["tp_compression_achieved_bp_high"]}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from src.data.panel import build_panels
    r = compute(build_panels())
    for k, v in r.items():
        print(f"{k:36s} {v}")
