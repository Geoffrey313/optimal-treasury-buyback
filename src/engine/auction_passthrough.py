"""Auction channel pass-through :math:`\\rho`.

Tests whether the secondary-market liquidity gain from a buyback reaches the
primary market. Implements Eq.~(auctionpass) of the model,

    Y_{tau,a} = eta + rho * BB_{s(tau),a} + W_{tau,a}' psi + omega_{tau,a},

for each auction outcome ``Y`` in {tail, bid-to-cover, indirect share,
concession}, where ``BB_{s,t}`` is the intensity of buybacks in sector ``s`` over
the weeks before the auction (built from ``OPS``), and ``W`` are controls (issue
size, with two-way sector and month fixed effects). A negative ``rho`` for the
tail and a positive ``rho`` for the bid-to-cover ratio indicate that recent
buybacks make later auctions cheaper.

Because a buyback funded in the same sector both compresses the on/off-the-run
spread and adds supply, ``rho`` (compression channel) and ``lambda_I`` (supply
channel) are estimated on the same auctions so the two channels are separated;
the issue size enters as a control here to hold the supply channel fixed while
``rho`` picks up the compression channel.

Standard errors are clustered by sector; because there are only a handful of
sectors, a wild cluster bootstrap p-value and confidence interval are reported
alongside each estimate and its classic clustered t. Each outcome returns an
:class:`src.common.schema.Estimate`.

Inputs are the ``AUCTION`` and ``OPS`` panels of :mod:`src.common.schema`.
"""
from __future__ import annotations

from typing import Final

import pandas as pd
import statsmodels.formula.api as smf

from src.common import schema
from src.common.config import MONTHS_PER_YEAR
from src.engine.absorption import (
    bootstrap_note,
    coerce_categorical,
    coerce_numeric,
    has_estimable_design,
    month_period,
    wild_cluster_bootstrap,
)

#: Length of the pre-auction window over which buyback intensity is accumulated.
INTENSITY_WINDOW_DAYS: Final[int] = 28
_INTENSITY: Final[str] = "bb_intensity"
_ISSUE_CONTROL: Final[str] = "issue_size"
_CLUSTER_KEY: Final[str] = "sector"
#: Year-month time-effect column (per-auction-date dummies are singular).
_PERIOD: Final[str] = "period"

#: Auction outcomes regressed on buyback intensity (columns of AUCTION).
OUTCOMES: Final[tuple[str, ...]] = (
    "tail",
    "bid_to_cover",
    "indirect_share",
    "concession",
)


#: Upper bound (years) standing in for an open-ended ">30Y" maturity bucket.
OPEN_ENDED_MATURITY_YEARS: Final[float] = 100.0
_RANGE_TOKEN: Final[str] = r"(\d+(?:\.\d+)?)\s*(mo|y)?"


def _parse_maturity_range(label: str) -> tuple[float, float]:
    """Parse a maturity-bucket label into a (low, high) range in years.

    Handles both label conventions used across the panels, e.g. ``"10Y to 20Y"``
    and ``"10-20Y"`` (both -> (10, 20)), ``"1Mo to 2Y"`` -> (1/12, 2), and the
    open-ended ``">30Y"`` -> (30, :data:`OPEN_ENDED_MATURITY_YEARS`). Labels with
    no parseable number (e.g. ``"null"``) return ``(nan, nan)``.
    """
    import math
    import re

    text = str(label).strip().lower()
    matches = re.findall(_RANGE_TOKEN, text)
    if not matches:
        return (math.nan, math.nan)
    values = []
    for number, unit in matches:
        years = float(number) / MONTHS_PER_YEAR if unit == "mo" else float(number)
        values.append(years)
    low = min(values)
    high = max(values) if len(values) > 1 else OPEN_ENDED_MATURITY_YEARS
    if ">" in text and len(values) == 1:
        low, high = values[0], OPEN_ENDED_MATURITY_YEARS
    elif "<" in text and len(values) == 1:
        low, high = 0.0, values[0]
    return (low, high)


def buyback_intensity(auction: pd.DataFrame, ops: pd.DataFrame) -> pd.Series:
    """Compute the pre-auction buyback intensity ``BB_{s,t}`` for each auction.

    For each auction, sums the accepted par (``q_real``) of operations executed
    within :data:`INTENSITY_WINDOW_DAYS` before the auction date whose maturity
    bucket overlaps the auction's sector. The two panels label maturity ranges
    differently (``OPS.maturity_bucket`` vs ``AUCTION.sector``), so both are
    parsed to year ranges and matched by overlap. Auctions with no recent
    overlapping operation get zero.
    """
    au = auction[["auction_date", _CLUSTER_KEY]].reset_index()
    au[["_lo", "_hi"]] = au[_CLUSTER_KEY].apply(
        lambda s: pd.Series(_parse_maturity_range(s))
    )

    op = ops[["operation_date", "maturity_bucket", "q_real"]].copy()
    op[["_op_lo", "_op_hi"]] = op["maturity_bucket"].apply(
        lambda s: pd.Series(_parse_maturity_range(s))
    )
    op = op.dropna(subset=["_op_lo", "_op_hi", "q_real"])

    window = pd.to_timedelta(INTENSITY_WINDOW_DAYS, unit="D")
    merged = au.merge(op, how="cross")
    in_window = (merged["operation_date"] < merged["auction_date"]) & (
        merged["operation_date"] >= merged["auction_date"] - window
    )
    overlap = (merged["_lo"] < merged["_op_hi"]) & (merged["_op_lo"] < merged["_hi"])
    contributes = in_window & overlap
    merged["_contrib"] = merged["q_real"].where(contributes, other=0.0)
    intensity = merged.groupby("index")["_contrib"].sum()
    return intensity.reindex(auction.index).fillna(0.0)


def build_passthrough_panel(
    auction: pd.DataFrame, ops: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the auction-level pass-through frame with ``BB_{s,t}`` and
    controls."""
    au = auction.copy()
    au[_INTENSITY] = buyback_intensity(auction, ops)
    au[_PERIOD] = month_period(au["auction_date"])
    keep = list(OUTCOMES) + [_INTENSITY, _ISSUE_CONTROL, _CLUSTER_KEY, _PERIOD]
    frame = au[[c for c in keep if c in au.columns]]
    numeric = [c for c in (*OUTCOMES, _INTENSITY, _ISSUE_CONTROL) if c in frame.columns]
    frame = coerce_numeric(frame, numeric)
    frame = coerce_categorical(frame, [_CLUSTER_KEY, _PERIOD])
    return frame


def _passthrough_formula(outcome: str) -> str:
    """OLS formula for Eq.~(auctionpass) for one outcome, with two-way sector and
    month fixed effects and the issue-size control that separates the compression
    channel from the supply channel."""
    return (
        f"{outcome} ~ {_INTENSITY} + {_ISSUE_CONTROL} "
        f"+ C({_CLUSTER_KEY}) + C({_PERIOD})"
    )


def estimate_rho(
    auction: pd.DataFrame, ops: pd.DataFrame
) -> dict[str, schema.Estimate]:
    """Estimate the pass-through ``rho`` for each auction outcome.

    Returns a dict keyed by outcome (``tail``, ``bid_to_cover``,
    ``indirect_share``, ``concession``); each value is the coefficient on buyback
    intensity under two-way sector and month fixed effects with the issue-size
    control. Alongside the classic sector-clustered t, a wild cluster bootstrap
    p-value and confidence interval are reported in the note, since the panel has
    few sectors. An outcome with no usable observation returns a clearly-noted
    NaN.

    The buyback-intensity source is pinned to the Liquidity Support program (the
    arm whose secondary-market liquidity effect the pass-through tests); cash
    management and small-value operations follow the tax calendar and are
    excluded, so the estimate does not depend on the caller's ``ops`` selection.
    """
    if "operation_type" in ops.columns:
        ops = ops[ops["operation_type"] == "Liquidity Support"]
    panel = build_passthrough_panel(auction, ops)
    results: dict[str, schema.Estimate] = {}
    for outcome in OUTCOMES:
        if outcome not in panel.columns:
            continue
        sub = panel.dropna(subset=[outcome, _INTENSITY, _ISSUE_CONTROL])
        if not has_estimable_design(sub, _INTENSITY):
            results[outcome] = schema.Estimate(
                value=float("nan"),
                se=float("nan"),
                n=0,
                note=f"rho on {outcome}: empty design or no buyback-intensity variation",
            )
            continue
        fit = smf.ols(_passthrough_formula(outcome), data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub[_CLUSTER_KEY]}
        )
        wcb = wild_cluster_bootstrap(fit, _INTENSITY, sub[_CLUSTER_KEY])
        results[outcome] = schema.Estimate(
            value=float(fit.params[_INTENSITY]),
            se=float(fit.bse[_INTENSITY]),
            tstat=float(fit.tvalues[_INTENSITY]),
            n=int(fit.nobs),
            note=f"rho on {outcome}; sector+month FE; classic sector-clustered t; "
            + bootstrap_note(wcb),
        )
    return results
