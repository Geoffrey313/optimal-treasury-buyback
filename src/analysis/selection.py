"""The Treasury's reaction function (US-03): which off-the-run securities does the
buyback program repurchase?

This is the paper's one identified empirical result. It is a positive, descriptive
question (what the Treasury does), not a causal one (what effect the buyback has)
or a normative one (what size is optimal), and it is exactly the selection that
defeats the causal identification. It is estimated with a linear probability model
of the purchase indicator on security characteristics, with operation fixed
effects (so the comparison is within an operation, across eligible securities) and
standard errors clustered by security. A linear model is used rather than a logit
because the operation fixed effects cause separation in a logit.

Honesty: all predictors are reported, significant or not. The robust, significant
drivers are maturity and coupon (the program buys the shorter, lower-coupon
off-the-run securities). The direct liquidity measures (the on/off-the-run spread
and the bid-ask spread) are not robustly signed across specifications, so the
result is stated as a maturity/coupon reaction function, not as "the Treasury buys
the least liquid securities".
"""
from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from src.common.config import SAMPLE_START
from src.common import schema

PREDICTORS = ("ytm", "coupon", "ofr_spread_z", "bid_ask_z")


def build_selection_panel(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per eligible security per Liquidity Support operation (since the
    sample start), with the purchase indicator and the security characteristics."""
    from src.engine.absorption import normalize_cusip
    ops, opsec, secday = panel["ops"], panel["opsec"], panel["secday"]
    ls = set(ops[(ops["operation_type"] == "Liquidity Support")
                 & (pd.to_datetime(ops["operation_date"]) >= SAMPLE_START)]["operation_date"])
    d = opsec[opsec["operation_date"].isin(ls)].copy()
    d["bought"] = d["bought"].astype(float)
    d["ytm"] = (pd.to_datetime(d["maturity_date"])
                - pd.to_datetime(d["operation_date"])).dt.days / 365.25
    d["coupon"] = pd.to_numeric(d["coupon_rate"], errors="coerce")
    d["base"] = normalize_cusip(d["cusip"])
    sd = secday.copy()
    sd["base"] = normalize_cusip(sd["cusip"])
    liq = sd.groupby("base")[["ofr_spread", "bid_ask"]].mean()
    d = d.merge(liq, on="base", how="left")
    for c in ("ofr_spread", "bid_ask"):  # standardize the liquidity measures
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    d["op"] = pd.to_datetime(d["operation_date"]).astype(str)
    return d.dropna(subset=["bought", *PREDICTORS])


def estimate_reaction_function(panel: dict[str, pd.DataFrame]) -> dict[str, schema.Estimate]:
    """Linear probability model of the purchase indicator on maturity, coupon, and
    the standardized liquidity measures, with operation fixed effects and standard
    errors clustered by security. Returns one Estimate per predictor."""
    d = build_selection_panel(panel)
    fit = smf.ols("bought ~ ytm + coupon + ofr_spread_z + bid_ask_z + C(op)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["base"]})
    out: dict[str, schema.Estimate] = {}
    for v in PREDICTORS:
        out[v] = schema.Estimate(
            value=float(fit.params[v]), se=float(fit.bse[v]),
            tstat=float(fit.tvalues[v]), n=int(fit.nobs),
            note=f"LPM, operation FE, SE clustered by CUSIP; R2={fit.rsquared:.3f}")
    # Reserved non-predictor key carrying the fit R-squared for the results table.
    out["r2"] = schema.Estimate(value=float(fit.rsquared), se=0.0, tstat=0.0,
                                n=int(fit.nobs), note="within-sample R2")
    return out


def robustness_within_maturity(panel: dict[str, pd.DataFrame]) -> dict[str, schema.Estimate]:
    """Same reaction function, but with maturity-bucket fixed effects added to the
    operation fixed effects, so the liquidity coefficients are identified within a
    maturity band. This is the specification that tests whether the liquidity signs
    are robust: maturity and the off/on-the-run status are collinear (older, shorter
    off-the-runs trade at wider spreads), so absorbing the maturity band is the
    honest check on whether liquidity carries independent information."""
    d = build_selection_panel(panel).copy()
    d["mbucket"] = pd.cut(d["ytm"], bins=[0, 2, 5, 10, 30, 100],
                          labels=["0-2", "2-5", "5-10", "10-30", "30+"])
    d = d.dropna(subset=["mbucket"])
    fit = smf.ols("bought ~ coupon + ofr_spread_z + bid_ask_z + C(op) + C(mbucket)",
                  data=d).fit(cov_type="cluster", cov_kwds={"groups": d["base"]})
    out: dict[str, schema.Estimate] = {}
    for v in ("coupon", "ofr_spread_z", "bid_ask_z"):
        out[v] = schema.Estimate(
            value=float(fit.params[v]), se=float(fit.bse[v]),
            tstat=float(fit.tvalues[v]), n=int(fit.nobs),
            note=f"LPM, operation + maturity-bucket FE, SE clustered by CUSIP; R2={fit.rsquared:.3f}")
    out["r2"] = schema.Estimate(value=float(fit.rsquared), se=0.0, tstat=0.0,
                                n=int(fit.nobs), note="within-sample R2")
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from src.data.panel import build_panels
    p = build_panels()
    print("Baseline (operation FE):")
    for v, e in estimate_reaction_function(p).items():
        print(f"  {v:14s} coef={e.value:+.4f} t={e.tstat:+.2f} n={e.n}")
    print("Within maturity bucket (operation + maturity-bucket FE):")
    for v, e in robustness_within_maturity(p).items():
        print(f"  {v:14s} coef={e.value:+.4f} t={e.tstat:+.2f} n={e.n}")
