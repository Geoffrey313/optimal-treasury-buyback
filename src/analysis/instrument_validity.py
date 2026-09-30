"""CA-2.2b — validity gate for the maturity-month instrument z_i.

The instrument is usable only if tax-month maturity does not predict
systematically different security characteristics. This module implements the
two checks that decide the gate here: the covariate balance between z_i = 1 and
z_i = 0, and the first-stage strength of z_i on the accepted amount. If either
fails, US-02 CA-2.2b requires abandoning z_i, so the 2SLS(z_i) estimate of the
absorption cost is not reported.

TODO(CA-2.2b, later): add the exclusion pre-trend test (that z_i does not predict
pre-operation price or liquidity dynamics after controls). It is not implemented
yet because the covariate balance already fails decisively, which by itself
fails the gate; the pre-trend test would only matter if the balance passed.

This module only diagnoses; it does not estimate lambda_B.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# Weak-instrument rule of thumb: first-stage F below this is weak.
WEAK_IV_F_THRESHOLD: float = 10.0
# Balance is deemed to fail if any covariate differs at this two-sided level.
BALANCE_ALPHA: float = 0.05


@dataclass
class BalanceRow:
    covariate: str
    mean_z1: float
    mean_z0: float
    diff: float
    tstat: float
    pvalue: float


@dataclass
class InstrumentVerdict:
    passed: bool
    reason: str
    balance: list[BalanceRow] = field(default_factory=list)
    first_stage_f: float = float("nan")


def _two_sample_t(x1: pd.Series, x0: pd.Series) -> tuple[float, float]:
    x1 = pd.to_numeric(x1, errors="coerce").dropna()
    x0 = pd.to_numeric(x0, errors="coerce").dropna()
    t, p = stats.ttest_ind(x1, x0, equal_var=False)
    return float(t), float(p)


def covariate_balance(opsec_core: pd.DataFrame) -> list[BalanceRow]:
    """Balance of security characteristics between z_i = 1 and z_i = 0.

    Covariates come straight from the operation x security panel: remaining
    years to maturity (maturity_date minus operation_date) and coupon rate.
    """
    df = opsec_core.copy()
    df["z_i"] = df["z_i"].astype("float64")
    df["years_to_maturity"] = (
        pd.to_datetime(df["maturity_date"]) - pd.to_datetime(df["operation_date"])
    ).dt.days / 365.25
    rows: list[BalanceRow] = []
    for cov in ("years_to_maturity", "coupon_rate"):
        z1, z0 = df.loc[df["z_i"] == 1, cov], df.loc[df["z_i"] == 0, cov]
        t, p = _two_sample_t(z1, z0)
        rows.append(BalanceRow(cov, float(z1.mean()), float(z0.mean()),
                               float(z1.mean() - z0.mean()), t, p))
    return rows


def first_stage_f(opsec_core: pd.DataFrame) -> float:
    """First-stage F of z_i on the accepted amount, with coupon/maturity controls
    and operation fixed effects (F = t^2 for the single instrument)."""
    import statsmodels.formula.api as smf

    df = opsec_core.copy()
    df["z_i"] = df["z_i"].astype("float64")
    df["par_accepted"] = pd.to_numeric(df["par_accepted"], errors="coerce")
    df["coupon_rate"] = pd.to_numeric(df["coupon_rate"], errors="coerce")
    df["ytm"] = (pd.to_datetime(df["maturity_date"])
                 - pd.to_datetime(df["operation_date"])).dt.days / 365.25
    df["op"] = pd.to_datetime(df["operation_date"]).astype(str)
    df = df.dropna(subset=["par_accepted", "z_i", "coupon_rate", "ytm"])
    fit = smf.ols("par_accepted ~ z_i + coupon_rate + ytm + C(op)", data=df).fit()
    return float(fit.tvalues["z_i"] ** 2)


def evaluate_instrument(opsec_core: pd.DataFrame) -> InstrumentVerdict:
    """Run CA-2.2b and return a pass/fail verdict."""
    balance = covariate_balance(opsec_core)
    f = first_stage_f(opsec_core)
    failed_cov = [b.covariate for b in balance if b.pvalue < BALANCE_ALPHA]
    weak = f < WEAK_IV_F_THRESHOLD
    if failed_cov or weak:
        bits = []
        if failed_cov:
            bits.append("covariate imbalance on " + ", ".join(failed_cov))
        if weak:
            bits.append(f"weak first stage (F={f:.2f} < {WEAK_IV_F_THRESHOLD:.0f})")
        return InstrumentVerdict(False, "CA-2.2b FAIL: " + "; ".join(bits),
                                 balance=balance, first_stage_f=f)
    return InstrumentVerdict(True, "CA-2.2b PASS", balance=balance, first_stage_f=f)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from src.common.config import SAMPLE_START
    from src.data.panel import build_panels

    P = build_panels()
    ops = P["ops"]
    dates = set(ops[(ops["operation_type"] == "Liquidity Support")
                    & (ops["operation_date"] >= SAMPLE_START)]["operation_date"])
    opsec_core = P["opsec"][P["opsec"]["operation_date"].isin(dates)]

    v = evaluate_instrument(opsec_core)
    print(v.reason)
    print(f"first-stage F(z_i) = {v.first_stage_f:.2f}")
    for b in v.balance:
        print(f"  {b.covariate:20s} z1={b.mean_z1:8.3f} z0={b.mean_z0:8.3f} "
              f"diff={b.diff:8.3f} t={b.tstat:7.2f} p={b.pvalue:.1e}")
