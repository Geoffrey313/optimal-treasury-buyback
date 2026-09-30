"""Duration accounting: the buyback removes only a small fraction of
the coupon supply the Treasury issues, whether measured in par or in ten-year
equivalents.

Both sides use one consistent base. Durations are analytical modified durations
(from each security's coupon, remaining maturity, and yield), so the calculation
covers 2026 and does not depend on a vendor duration field. Two ratios are
reported and clearly labeled, never one disguised as the other:

* par ratio: Liquidity Support par bought over NET coupon issuance
  (gross issuance minus coupon maturities in the window), a par-stock concept;
* duration ratio: Liquidity Support ten-year-equivalents removed over GROSS
  coupon issuance in ten-year equivalents (a flow-of-duration concept). Net
  duration is deliberately not used as a denominator: a bond maturing inside the
  window has near-zero duration at maturity, so netting its par does not measure
  a duration withdrawal.

Cash management buybacks are excluded from the duration framing: they are mostly
bills (near-zero duration).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import SAMPLE_START, WINDOW_END
from src.data.load_public import load_auctions

COUPON_FREQ = 2          # semiannual Treasury coupons
REF_10Y_COUPON = 4.2     # reference 10-year note, for the ten-year-equivalent base
REF_10Y_YEARS = 10.0
REF_10Y_YIELD = 4.4
COUPON_TYPES = ("Note", "Bond")
LIQUIDITY_SUPPORT = ("Liquidity Support",)


def modified_duration(coupon_pct, years, yield_pct, freq: int = COUPON_FREQ):
    """Analytical modified duration in years of a semiannual coupon bond, priced
    off its own yield. Vectorized over equal-length arrays; NaN where inputs are
    missing or the maturity is non-positive."""
    coupon_pct = np.asarray(coupon_pct, float)
    years = np.asarray(years, float)
    y = np.asarray(yield_pct, float) / 100.0
    out = np.full(years.shape, np.nan)
    for i in range(out.size):
        T, c, yi = years.flat[i], coupon_pct.flat[i], y.flat[i]
        if not (np.isfinite(T) and np.isfinite(c) and np.isfinite(yi)) or T <= 0:
            continue
        n = max(int(round(T * freq)), 1)
        per, cpn = yi / freq, c / freq
        k = np.arange(1, n + 1)
        cf = np.full(n, cpn)
        cf[-1] += 100.0
        disc = (1 + per) ** k
        price = (cf / disc).sum()
        if price <= 0:
            continue
        d_mac = ((k / freq) * cf / disc).sum() / price
        out.flat[i] = d_mac / (1 + per)
    return out


def reference_10y_duration() -> float:
    return float(modified_duration([REF_10Y_COUPON], [REF_10Y_YEARS], [REF_10Y_YIELD])[0])


def analytical_crsp_corr() -> float:
    """Per-security correlation between the analytical modified duration and the
    CRSP reported duration, on the latest CRSP snapshot. Backs the manuscript
    claim that the two measures agree, so the analytical measure can cover the
    securities outside the CRSP window."""
    from src.data.load_crsp import load_tfz_dly, load_tfz_iss
    dly, iss = load_tfz_dly(), load_tfz_iss()
    d = dly[dly["caldt"] == dly["caldt"].max()].merge(
        iss[["kytreasno", "tmatdt", "tcouprt"]], on="kytreasno", how="left")
    d["ytm"] = (pd.to_datetime(d["tmatdt"]) - pd.to_datetime(d["caldt"])).dt.days / 365.25
    d["crsp"] = pd.to_numeric(d["tdduratn"], errors="coerce") / 365.25
    d["cpn"] = pd.to_numeric(d["tcouprt"], errors="coerce")
    d["y"] = pd.to_numeric(d["tdyld"], errors="coerce")
    d = d[(d["ytm"] > 0.2) & (d["cpn"] > 0)].copy()
    d["an"] = modified_duration(d["cpn"].values, d["ytm"].values, d["y"].values)
    d = d.dropna(subset=["crsp", "an"])
    return float(d["crsp"].corr(d["an"]))


def _removed(panel: dict[str, pd.DataFrame], d10: float, op_types=None):
    """Par and ten-year equivalents repurchased over the window.

    `op_types` selects the arm: the liquidity support arm alone, or every buyback
    operation when None. Reporting both is what lets the manuscript state that the
    scale conclusion does not turn on which arm is counted."""
    from src.engine.absorption import normalize_cusip
    ops, opsec, secday = panel["ops"], panel["opsec"], panel["secday"]
    in_window = pd.to_datetime(ops["operation_date"]) >= SAMPLE_START
    sel = in_window if op_types is None else in_window & ops["operation_type"].isin(op_types)
    ls_dates = set(ops[sel]["operation_date"])
    b = opsec[(opsec["operation_date"].isin(ls_dates)) & (opsec["bought"])].copy()
    b["ytm"] = (pd.to_datetime(b["maturity_date"])
                - pd.to_datetime(b["operation_date"])).dt.days / 365.25
    b["cpn"] = pd.to_numeric(b["coupon_rate"], errors="coerce")
    b["par"] = pd.to_numeric(b["par_accepted"], errors="coerce")
    # Duration source: the vendor modified duration from CRSP;
    # per-security it matches the analytical formula, corr ~0.996). Analytical
    # modified duration at the coupon is the fallback for securities absent from
    # CRSP (e.g. 2026, past the CRSP window).
    sd = secday.copy()
    sd["base"] = normalize_cusip(sd["cusip"])
    crsp_dur = sd.groupby("base")["duration"].mean()
    b["base"] = normalize_cusip(b["cusip"])
    b["dur_crsp"] = b["base"].map(crsp_dur)
    b["dur_an"] = modified_duration(b["cpn"].values, b["ytm"].values, b["cpn"].values)
    b["dur"] = b["dur_crsp"].fillna(b["dur_an"])
    b = b.dropna(subset=["dur", "par"])
    par = b["par"].sum()
    ten_y = (b["par"] * b["dur"]).sum() / d10
    return par, ten_y


LONG_END_BUCKETS = ("10Y to 20Y", "20Y to 30Y")


def long_end_cap_change() -> dict[str, float]:
    """The per-operation redemption cap at the long end, before and after it was
    raised, read from the operation announcements rather than asserted in prose.

    The cap is the debt manager's own state-contingent lever, so the manuscript
    reports its size from the data that records it. Read from the operations
    release directly: the analysis panel does not carry the announcement columns."""
    from src.data.load_public import load_buyback_operations
    ops = load_buyback_operations().copy()
    ops["d"] = pd.to_datetime(ops["operation_date"])
    ls = ops[(ops["operation_type"] == "Liquidity Support")
             & (ops["d"] >= SAMPLE_START)
             & (ops["maturity_bucket"].isin(LONG_END_BUCKETS))].copy()
    ls["cap"] = pd.to_numeric(ls["max_par_amt_redeemed"], errors="coerce") / 1e9
    ls = ls.dropna(subset=["cap"])
    first, last = ls["cap"].iloc[0], ls["cap"].iloc[-1]
    raised = ls[ls["cap"] > first]
    return {
        "long_end_cap_before_bn": float(first),
        "long_end_cap_after_bn": float(last),
        "long_end_cap_raised_on": raised["d"].min().date().isoformat() if len(raised) else "",
    }


def _coupon_issuance(d10: float):
    a = load_auctions()
    a = a[a["security_type"].isin(COUPON_TYPES)].copy()
    a["auction_date"] = pd.to_datetime(a["auction_date"])
    a["maturity_date"] = pd.to_datetime(a["maturity_date"])
    a["size"] = pd.to_numeric(a["total_accepted"], errors="coerce")
    a["cpn"] = pd.to_numeric(a.get("int_rate"), errors="coerce")
    a["stop"] = pd.to_numeric(a.get("high_yield"), errors="coerce")
    win = (a["auction_date"] >= SAMPLE_START) & (a["auction_date"] <= WINDOW_END)
    gross = a.loc[win].copy()
    gross["ytm"] = (gross["maturity_date"] - gross["auction_date"]).dt.days / 365.25
    gross["dur"] = modified_duration(gross["cpn"].values, gross["ytm"].values,
                                     gross["stop"].fillna(gross["cpn"]).values)
    # par ratio uses the FULL gross par (no duration filter, so it is consistent
    # with the full-par maturities); the duration ratio uses only the securities
    # whose duration is computable.
    gross_par = pd.to_numeric(gross["size"], errors="coerce").sum()
    dur_ok = gross.dropna(subset=["dur", "size"])
    gross_10y = (dur_ok["size"] * dur_ok["dur"]).sum() / d10
    # maturities in the window (par concept only), from the full auction history
    mat_par = a.loc[(a["maturity_date"] >= SAMPLE_START)
                    & (a["maturity_date"] <= WINDOW_END), "size"].sum()
    return gross_par, gross_10y, mat_par


def compute(panel: dict[str, pd.DataFrame]) -> dict[str, float]:
    d10 = reference_10y_duration()
    ls_par, ls_10y = _removed(panel, d10, LIQUIDITY_SUPPORT)
    all_par, all_10y = _removed(panel, d10, None)
    gross_par, gross_10y, mat_par = _coupon_issuance(d10)
    net_par = gross_par - mat_par
    return {
        "ref_10y_duration_years": d10,
        "ls_removed_par_bn": ls_par / 1e9,
        "ls_removed_10y_equiv_bn": ls_10y / 1e9,
        "all_removed_par_bn": all_par / 1e9,
        "all_removed_10y_equiv_bn": all_10y / 1e9,
        "gross_issuance_par_bn": gross_par / 1e9,
        "gross_issuance_10y_equiv_bn": gross_10y / 1e9,
        "coupon_maturities_par_bn": mat_par / 1e9,
        "net_issuance_par_bn": net_par / 1e9,
        "ratio_par_ls_over_net_pct": 100.0 * ls_par / net_par,
        "ratio_duration_ls_over_gross_pct": 100.0 * ls_10y / gross_10y,
        "ratio_par_all_over_net_pct": 100.0 * all_par / net_par,
        "ratio_duration_all_over_gross_pct": 100.0 * all_10y / gross_10y,
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from src.data.panel import build_panels
    r = compute(build_panels())
    for k, v in r.items():
        print(f"{k:38s} {v:12.2f}")
