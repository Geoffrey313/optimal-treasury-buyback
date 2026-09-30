"""Strengthened identification of the buyback price effect (US-04).

The direct estimates in :mod:`src.analysis.event_study` are reported as bounds
because the Treasury selects which securities to repurchase, and the reaction
function (:mod:`src.analysis.selection`) shows the selection is on maturity and
coupon. This module turns that diagnosis into an identification strategy: it
compares each repurchased security only with the eligible-but-not-repurchased
securities in the same maturity-by-coupon cell, so treated and control securities
are comparable on the documented selection margins, and it re-tests the
parallel-trends assumption on the matched sample.

The maturity and coupon cell grid is fixed a priori (standard buckets), not chosen
to make the pre-trend hold. The decision is honest: if the matched pre-trend test
no longer rejects, the design identifies a local effect conditional on the cell;
if it still rejects, the honest bound stands and is reinforced by the sensitivity
analysis. Either way the identified effect is conditional on maturity and coupon,
never unconditional: selection on unobservables (repo specialness, dealer
inventory) is not removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed a-priori cell grid (never tuned to the outcome).
MATURITY_BINS = [0, 2, 5, 10, 30, 100]
MATURITY_LABELS = ["0-2", "2-5", "5-10", "10-30", "30+"]
COUPON_BINS = [-0.01, 2, 4, 6, 100]
COUPON_LABELS = ["<2", "2-4", "4-6", "6+"]
BASE_LEN = 8
BALANCE_VARS = ("ytm", "coupon", "outstanding")


def _base(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.slice(0, BASE_LEN)


def build_security_table(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per eligible security: treatment status, its maturity-by-coupon
    cell (measured at the security's first eligible operation), and the average
    amount outstanding used for the balance check."""
    opsec, secday = panel["opsec"], panel["secday"]
    d = opsec.copy()
    d["base"] = _base(d["cusip"])
    d["ytm"] = (pd.to_datetime(d["maturity_date"])
                - pd.to_datetime(d["operation_date"])).dt.days / 365.25
    d["coupon"] = pd.to_numeric(d["coupon_rate"], errors="coerce")
    d = d.sort_values("operation_date")
    g = d.groupby("base")
    tab = pd.DataFrame({
        "treated": g["bought"].any(),
        "first_elig": g["operation_date"].min(),
        "ytm": g["ytm"].first(),        # at the first eligible operation
        "coupon": g["coupon"].first(),
    }).reset_index()
    tab["mbucket"] = pd.cut(tab["ytm"], bins=MATURITY_BINS, labels=MATURITY_LABELS)
    tab["cbucket"] = pd.cut(tab["coupon"], bins=COUPON_BINS, labels=COUPON_LABELS)
    tab["cell"] = tab["mbucket"].astype(str) + " x " + tab["cbucket"].astype(str)
    sd = secday.copy()
    sd["base"] = _base(sd["cusip"])
    outstanding = sd.groupby("base")["amount_outstanding"].mean().rename("outstanding")
    tab = tab.merge(outstanding, on="base", how="left")
    return tab.dropna(subset=["mbucket", "cbucket"])


def matched_set(tab: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep securities in cells that hold at least one treated and at least one
    eligible-not-bought control (common support); return them and the cells."""
    by_cell = tab.groupby("cell")["treated"].agg(n_treated="sum", n_total="count")
    by_cell["n_control"] = by_cell["n_total"] - by_cell["n_treated"]
    supported = by_cell[(by_cell["n_treated"] >= 1) & (by_cell["n_control"] >= 1)].index
    return tab[tab["cell"].isin(supported)].copy(), sorted(supported)


def covariate_balance(tab_all: pd.DataFrame, tab_matched: pd.DataFrame) -> pd.DataFrame:
    """Standardized differences between treated and control, before matching (all
    eligible securities) and after matching (within supported cells)."""
    rows = []
    for stage, t in [("before", tab_all), ("after", tab_matched)]:
        tr, co = t[t["treated"]], t[~t["treated"]]
        for v in BALANCE_VARS:
            a, b = tr[v].dropna(), co[v].dropna()
            pooled_sd = np.sqrt((a.var() + b.var()) / 2) if len(a) > 1 and len(b) > 1 else np.nan
            std_diff = (a.mean() - b.mean()) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan
            rows.append({"stage": stage, "variable": v,
                         "treated_mean": a.mean(), "control_mean": b.mean(),
                         "std_diff": std_diff, "n_treated": len(a), "n_control": len(b)})
    return pd.DataFrame(rows)


def matched_event_study(panel: dict[str, pd.DataFrame], tab_matched: pd.DataFrame,
                        K: int = 8, alpha: float = 0.05):
    """Run the staggered event study on the matched securities only, so the
    never-treated controls are the eligible-not-bought securities in the treated
    cells, and run the joint pre-trend test on it."""
    from src.analysis.event_study import estimate_event_study, parallel_trends_test
    keep = set(tab_matched["base"])
    sd = panel["secday"].copy()
    sd = sd[_base(sd["cusip"]).isin(keep)]
    op = panel["opsec"].copy()
    op = op[_base(op["cusip"]).isin(keep)]
    theta = estimate_event_study(sd, op, outcome="ofr_spread", K=K, alpha=alpha)
    return theta, parallel_trends_test(theta, alpha=alpha)


def matched_bound(theta: pd.DataFrame, alpha: float = 0.05) -> dict:
    """The bound itself: the aggregated post-repurchase effect on the on/off-the-run
    spread, in basis points, with its confidence interval, so the reader sees which
    effects the matched design excludes. Post periods (k>=0) are averaged with weights
    proportional to their treated counts; the variance uses the event-time covariance
    when available (Sun-Abraham path), otherwise a conservative independent
    approximation. This is a bound, not an identified effect: the matched pre-trend is
    rejected (see :func:`matched_event_study`)."""
    from scipy import stats
    scale = 1e4  # yield fraction -> basis points
    post = theta[(theta["k"] >= 0)].dropna(subset=["theta", "se"]).copy()
    idx = theta.attrs.get("theta_index")
    vcov = theta.attrs.get("theta_vcov")
    w = post["n"].to_numpy(dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(post), 1.0 / len(post))
    att = float(w @ post["theta"].to_numpy())
    if vcov is not None and idx is not None:
        pos = [idx.index(int(k)) for k in post["k"]]
        V = np.asarray(vcov)[np.ix_(pos, pos)]
        var = float(w @ V @ w)
    else:  # conservative: ignore positive cross-covariances
        var = float(np.sum((w * post["se"].to_numpy()) ** 2))
    z = float(stats.norm.ppf(1 - alpha / 2))
    se = np.sqrt(var) if var > 0 else np.nan
    return {
        "att_bp": att * scale,
        "se_bp": se * scale,
        "ci_low_bp": (att - z * se) * scale,
        "ci_high_bp": (att + z * se) * scale,
        "n_post_periods": int(len(post)),
    }


def _delta_spread(panel: dict[str, pd.DataFrame], matched: pd.DataFrame,
                  event_dates: pd.Series, window_days: int = 30) -> pd.DataFrame:
    """Security-level change in the on/off-the-run spread from a pre to a post window
    around each security's event date (its own repurchase cohort for treated units;
    a per-cell pseudo-event date for controls). Returns one row per security with the
    change and the pre-period liquidity level."""
    sd = panel["secday"].copy()
    sd["base"] = _base(sd["cusip"])
    sd["date"] = pd.to_datetime(sd["date"])
    sd = sd[sd["base"].isin(set(matched["base"]))]
    rows = []
    for base, g in sd.groupby("base"):
        e = event_dates.get(base)
        if pd.isna(e):
            continue
        e = pd.Timestamp(e)
        g = g.sort_values("date")
        pre = g[(g["date"] >= e - pd.Timedelta(days=window_days)) & (g["date"] < e)]
        post = g[(g["date"] > e) & (g["date"] <= e + pd.Timedelta(days=window_days))]
        if pre.empty or post.empty:
            continue
        pre_s = pd.to_numeric(pre["ofr_spread"], errors="coerce").mean()
        post_s = pd.to_numeric(post["ofr_spread"], errors="coerce").mean()
        ba = pd.to_numeric(pre["bid_ask"], errors="coerce").mean()
        if pd.isna(pre_s) or pd.isna(post_s):
            continue
        rows.append({"base": base, "d_spread": float(post_s - pre_s),
                     "bid_ask_pre": float(ba) if pd.notna(ba) else float("nan")})
    return pd.DataFrame(rows)


def _cohort_dates(panel: dict[str, pd.DataFrame]) -> pd.Series:
    """First repurchase (cohort) date per treated base CUSIP."""
    op = panel["opsec"].copy()
    op["base"] = _base(op["cusip"])
    bought = op[op["bought"].astype(bool)]
    return pd.to_datetime(bought.groupby("base")["operation_date"].min())


def _event_dates(matched: pd.DataFrame, cohorts: pd.Series, per_cell: bool) -> pd.Series:
    """Event date per security: the treated unit's own cohort; for a control, the
    median treated cohort of its cell (per_cell=True) or the global median treated
    cohort (per_cell=False, used only for the dating-insensitivity check)."""
    m = matched.copy()
    m["cohort"] = m["base"].map(cohorts)
    global_med = pd.to_datetime(m.loc[m["treated"], "cohort"]).median()
    if per_cell:
        cell_med = (m[m["treated"]].groupby("cell")["cohort"]
                    .apply(lambda s: pd.to_datetime(s).median()))
        control_date = m["cell"].map(cell_med).fillna(global_med)
    else:
        control_date = pd.Series(global_med, index=m.index)
    dates = m["cohort"].where(m["treated"], control_date)
    return pd.Series(dates.values, index=m["base"].values)


def _oster_delta(beta_s, r_s, beta_l, r_l, r_max, target=0.0):
    """Oster (2019) delta for coefficient stability, linear approximation:
    delta = (beta_l - target)(R_l - R_s) / [(beta_s - beta_l)(R_max - R_l)].
    It is the proportionality of selection on unobservables to selection on
    observables that would move the controlled coefficient to `target`."""
    denom = (beta_s - beta_l) * (r_max - r_l)
    if denom == 0:
        return float("nan")
    return (beta_l - target) * (r_l - r_s) / denom


def oster_fragility(panel: dict[str, pd.DataFrame], matched: pd.DataFrame,
                    cohorts: pd.Series) -> dict:
    """Oster coefficient-stability delta as a bound-fragility measure (NOT an
    identification test, and separate from the parallel-trends rejection that is the
    reason for the bound). A short regression of the security-level spread change on
    the repurchase indicator and a long regression that adds the selection
    observables (maturity, coupon, liquidity, size) are compared; delta is the
    selection on unobservables, relative to observables, that would drive the
    controlled coefficient to zero. R_max is set to Oster's default min(1, 1.3 R_long).
    Reported for the per-cell control dating and re-computed under global dating as an
    insensitivity check."""
    import statsmodels.formula.api as smf
    out = {}
    covars = matched.set_index("base")[["treated", "ytm", "coupon", "outstanding"]]
    for tag, per_cell in (("percell", True), ("global", False)):
        ev = _event_dates(matched, cohorts, per_cell=per_cell)
        d = _delta_spread(panel, matched, ev)
        d = d.merge(covars, left_on="base", right_index=True, how="inner").dropna(
            subset=["d_spread", "treated", "ytm", "coupon", "bid_ask_pre", "outstanding"])
        d["bought"] = d["treated"].astype(float)
        short = smf.ols("d_spread ~ bought", data=d).fit()
        long = smf.ols("d_spread ~ bought + ytm + coupon + bid_ask_pre + outstanding",
                       data=d).fit()
        b_s, b_l = float(short.params["bought"]), float(long.params["bought"])
        r_s, r_l = float(short.rsquared), float(long.rsquared)
        r_max = min(1.0, 1.3 * r_l)
        delta = _oster_delta(b_s, r_s, b_l, r_l, r_max)
        out[tag] = {"beta_short": b_s, "beta_long": b_l, "r_short": r_s, "r_long": r_l,
                    "r_max": r_max, "delta": delta, "n": int(d.shape[0]),
                    "sign_flips": bool(b_s * b_l < 0)}
    # Insensitivity check: Oster delta is reportable only if it is stable across the
    # control-dating choice. On this null the controlled coefficient barely moves, so
    # the (beta_short - beta_long) denominator is near-degenerate and delta is unstable
    # (its sign flips with the dating). When that happens Oster is uninformative here
    # and is not carried into the manuscript; the bound is the confidence interval and
    # the reason for the bound is the rejected parallel trend, not omitted-variable bias.
    d1, d2 = out["percell"]["delta"], out["global"]["delta"]
    out["stable"] = bool(np.isfinite(d1) and np.isfinite(d2) and (d1 * d2 > 0))
    out["reportable"] = out["stable"]
    return out


def compute(panel: dict[str, pd.DataFrame], K: int = 8, alpha: float = 0.05) -> dict:
    """S1 + S2: build the matched sample, the covariate balance, and the matched
    event study with its pre-trend decision."""
    tab = build_security_table(panel)
    matched, supported = matched_set(tab)
    balance = covariate_balance(tab, matched)
    theta, pretrend = matched_event_study(panel, matched, K=K, alpha=alpha)
    bound = matched_bound(theta, alpha=alpha)
    oster = oster_fragility(panel, matched, _cohort_dates(panel))
    n_treated = int(tab["treated"].sum())
    n_treated_matched = int(matched["treated"].sum())
    return {
        "n_securities": int(len(tab)),
        "n_treated": n_treated,
        "n_treated_matched": n_treated_matched,
        "n_treated_dropped": n_treated - n_treated_matched,
        "n_control_matched": int((~matched["treated"]).sum()),
        "n_supported_cells": len(supported),
        "supported_cells": supported,
        "balance": balance,
        "theta": theta,
        "pretrend": pretrend,
        "bound": bound,
        "oster": oster,
        "identified": (not pretrend["reject"]),  # gate: pre-trend no longer rejected
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from src.data.panel import build_panels
    r = compute(build_panels())
    print(f"securities={r['n_securities']} treated={r['n_treated']} "
          f"treated_matched={r['n_treated_matched']} dropped={r['n_treated_dropped']} "
          f"controls_matched={r['n_control_matched']} cells={r['n_supported_cells']}")
    print("\nCovariate balance (standardized differences):")
    print(r["balance"].to_string(index=False))
    pt = r["pretrend"]
    print(f"\nMatched pre-trend joint test: stat={pt['stat']:.2f} df={pt['df']} "
          f"p={pt['p_value']:.3f} method={pt['method']} -> "
          f"{'REJECTED (keep bound)' if pt['reject'] else 'NOT rejected (local effect identified)'}")
    b = r["bound"]
    print(f"\nBound (post-repurchase effect on the spread): {b['att_bp']:+.4f} bp, "
          f"95% CI [{b['ci_low_bp']:+.4f}, {b['ci_high_bp']:+.4f}] bp "
          f"over {b['n_post_periods']} post periods")
    print("\nOster bound-fragility (cross-sectional coefficient stability, separate from "
          "the pre-trend):")
    for tag in ("percell", "global"):
        o = r["oster"][tag]
        print(f"  {tag:8s} beta_short={o['beta_short']:+.3e} beta_long={o['beta_long']:+.3e} "
              f"R_long={o['r_long']:.3f} R_max={o['r_max']:.3f} delta={o['delta']:+.3f} "
              f"sign_flips={o['sign_flips']} n={o['n']}")
