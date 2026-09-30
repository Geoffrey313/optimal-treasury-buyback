"""Issuance impact :math:`\\lambda_I`.

Estimates the supply elasticity of yields from the yield change around each
auction. Implements Eq.~(issimpact) of the model,

    Delta y_{tau,a} = c + lambda_I * Issue_{tau,a} + W_{tau,a}' psi + v_{tau,a},

where ``Delta y`` is the change in the Treasury par yield at the auctioned tenor
around auction ``a`` and ``Issue`` is the amount sold (``issue_size`` in
``AUCTION``, now the emitted amount). The slope ``lambda_I`` is the supply
elasticity that the preferred-habitat model predicts to be positive.

Delta y is measured on the daily Treasury par-yield curve (tenor level), not on
CRSP CUSIP-level yields: each auction's ``security_term`` is mapped to the
nearest par-yield tenor, and Delta y is the change across a declared window
around the auction date. The par-yield curve covers 2024-2026, including the
2026 out-of-sample period that CRSP does not.

The specification carries two-way fixed effects, sector and month. The sample is
split by security type: notes and bonds together are the main specification,
because the preferred-habitat mechanism targets coupon securities, and bills are
a separate specification reported as robustness. Because there are only a handful
of sectors, inference rests on a wild cluster bootstrap p-value and confidence
interval reported alongside the point estimate and the classic sector-clustered
t. The estimate is returned as :class:`src.common.schema.Estimate` in raw units
(yield change per dollar sold); the entry point rescales it to a
per-billion-dollar figure for reading.

Inputs are the ``AUCTION`` panel of :mod:`src.common.schema` and the daily
Treasury par-yield curve shipped in ``data/curves/``.
"""
from __future__ import annotations

import re
from typing import Final, Mapping, Sequence

import pandas as pd
import statsmodels.formula.api as smf

from src.common import schema
from src.common.config import DAYS_PER_WEEK, DAYS_PER_YEAR, MONTHS_PER_YEAR
from src.data.load_public import load_par_yields
from src.engine.absorption import (
    bootstrap_note,
    coerce_categorical,
    coerce_numeric,
    fit_clustered,
    has_estimable_design,
    month_period,
    wild_cluster_bootstrap,
)

# ----- Regressor / control names -----
_ISSUE: Final[str] = "issue_size"
_CLUSTER_KEY: Final[str] = "sector"
_CLASS: Final[str] = "security_type"
#: Year-month time-effect column.
_PERIOD: Final[str] = "period"
_DELTA_Y: Final[str] = "delta_y"
_TENOR_COL: Final[str] = "tenor_col"

#: Default security classes for the main specification (preferred-habitat targets
#: coupon securities).
DEFAULT_CLASSES: Final[tuple[str, ...]] = ("Note", "Bond")

#: Treasury par-yield curve tenors and their maturity in years. Single source of
#: truth for mapping an auction term to a curve column.
PAR_YIELD_TENORS: Final[Mapping[str, float]] = {
    "1 Mo": 1.0 / MONTHS_PER_YEAR,
    "2 Mo": 2.0 / MONTHS_PER_YEAR,
    "3 Mo": 3.0 / MONTHS_PER_YEAR,
    "4 Mo": 4.0 / MONTHS_PER_YEAR,
    "6 Mo": 6.0 / MONTHS_PER_YEAR,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}

#: Declared event windows. Each maps to (pre_allows_exact, post_allows_exact):
#: whether the pre / post as-of match may land on the auction day itself.
#: "sym" = [-1, +1] (strictly before and strictly after),
#: "post" = [0, +1] (auction day to the day after),
#: "pre" = [-1, 0] (day before to the auction day).
WINDOWS: Final[Mapping[str, tuple[bool, bool]]] = {
    "sym": (False, False),
    "post": (True, False),
    "pre": (False, True),
}
DEFAULT_WINDOW: Final[str] = "sym"

_DATE_COL: Final[str] = "Date"
_PAR_YIELD_COL: Final[str] = "par_yield"
_TERM_TOKEN: Final[str] = r"(\d+(?:\.\d+)?)\s*-?\s*(year|yr|month|mo|week|wk|day)"


# --------------------------------------------------------------------------
# Term -> tenor mapping
# --------------------------------------------------------------------------
def term_to_years(term: str) -> float:
    """Parse a security term (e.g. ``"7-Year 3-Month"``, ``"52-Week"``) to years.

    Sums the year, month, week and day tokens found in the label. Returns NaN
    when no token parses.
    """
    text = str(term).strip().lower()
    total = 0.0
    found = False
    for number, unit in re.findall(_TERM_TOKEN, text):
        value = float(number)
        found = True
        if unit in ("year", "yr"):
            total += value
        elif unit in ("month", "mo"):
            total += value / MONTHS_PER_YEAR
        elif unit in ("week", "wk"):
            total += value * DAYS_PER_WEEK / DAYS_PER_YEAR
        else:  # day
            total += value / DAYS_PER_YEAR
    return total if found else float("nan")


def nearest_tenor(term: str) -> str:
    """Map a security term to the nearest par-yield tenor column by maturity."""
    years = term_to_years(term)
    if pd.isna(years):
        return ""
    return min(PAR_YIELD_TENORS, key=lambda col: abs(PAR_YIELD_TENORS[col] - years))


# --------------------------------------------------------------------------
# Delta y on the par-yield curve
# --------------------------------------------------------------------------
def _long_par_yields(par_yields: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide par-yield curve to long (Date, tenor_col, par_yield)."""
    value_cols = [c for c in PAR_YIELD_TENORS if c in par_yields.columns]
    long = par_yields.melt(
        id_vars=[_DATE_COL],
        value_vars=value_cols,
        var_name=_TENOR_COL,
        value_name=_PAR_YIELD_COL,
    )
    long = coerce_numeric(long, [_PAR_YIELD_COL])
    long[_DATE_COL] = pd.to_datetime(long[_DATE_COL])
    return long.dropna(subset=[_PAR_YIELD_COL]).sort_values(_DATE_COL)


def _asof_par_yield(
    auctions: pd.DataFrame,
    long: pd.DataFrame,
    *,
    direction: str,
    allow_exact: bool,
) -> pd.Series:
    """Par yield at the nearest curve date on one side of the auction date.

    ``direction`` is ``"backward"`` (pre) or ``"forward"`` (post); ``allow_exact``
    decides whether a match on the auction day itself counts.
    """
    left = auctions[["auction_date", _TENOR_COL]].sort_values("auction_date")
    merged = pd.merge_asof(
        left,
        long,
        by=_TENOR_COL,
        left_on="auction_date",
        right_on=_DATE_COL,
        direction=direction,
        allow_exact_matches=allow_exact,
    )
    return merged.set_index(left.index)[_PAR_YIELD_COL]


def build_issuance_panel(
    auction: pd.DataFrame,
    par_yields: pd.DataFrame | None = None,
    *,
    window: str = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Assemble the auction-level regression frame with par-yield ``delta_y``.

    Maps each auction to its nearest par-yield tenor, measures ``delta_y`` across
    the declared ``window`` around the auction date, and keeps the issue size,
    the sector, the month, and the security type for the class split.
    """
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {tuple(WINDOWS)}, got {window!r}")
    if par_yields is None:
        par_yields = load_par_yields()

    au = auction.copy()
    au[_TENOR_COL] = au["security_term"].map(nearest_tenor)
    au = au[au[_TENOR_COL] != ""]

    long = _long_par_yields(par_yields)
    pre_exact, post_exact = WINDOWS[window]
    pre = _asof_par_yield(au, long, direction="backward", allow_exact=pre_exact)
    post = _asof_par_yield(au, long, direction="forward", allow_exact=post_exact)
    au[_DELTA_Y] = post - pre

    au[_PERIOD] = month_period(au["auction_date"])

    keep = [_DELTA_Y, _ISSUE, _CLUSTER_KEY, _PERIOD, _CLASS]
    frame = au[keep]
    frame = coerce_numeric(frame, [_DELTA_Y, _ISSUE])
    frame = coerce_categorical(frame, [_CLUSTER_KEY, _PERIOD, _CLASS])
    return frame.dropna(subset=[_DELTA_Y, _ISSUE])


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------
def _issuance_formula() -> str:
    """OLS formula for Eq.~(issimpact): issue size with sector and month FE."""
    return f"{_DELTA_Y} ~ {_ISSUE} + C({_CLUSTER_KEY}) + C({_PERIOD})"


def estimate_lambda_I(
    auction: pd.DataFrame,
    secday: pd.DataFrame | None = None,
    *,
    classes: Sequence[str] = DEFAULT_CLASSES,
    window: str = DEFAULT_WINDOW,
    par_yields: pd.DataFrame | None = None,
) -> schema.Estimate:
    """Estimate the issuance slope ``lambda_I`` (Eq. issimpact).

    Regresses the par-yield change around each auction on the issue size with
    two-way sector and month fixed effects. ``classes`` selects the security
    types: the default ``("Note", "Bond")`` is the main coupon specification;
    pass ``("Bill",)`` for the bill robustness spec. ``window`` selects the
    declared event window. ``secday`` is accepted for interface compatibility and
    is unused (Delta y comes from the par-yield curve).

    The point estimate carries the classic sector-clustered t and, because there
    are few sectors, a wild cluster bootstrap p-value and confidence interval in
    the note. Raw units: yield change per dollar sold.
    """
    panel = build_issuance_panel(auction, par_yields, window=window)
    panel = panel[panel[_CLASS].isin(tuple(classes))]
    label = "+".join(classes)
    if not has_estimable_design(panel, _ISSUE):
        return schema.Estimate(
            value=float("nan"),
            se=float("nan"),
            n=0,
            note=f"lambda_I [{label}]: empty or rank-deficient design",
        )

    model = smf.ols(_issuance_formula(), data=panel)
    fit, n_clusters = fit_clustered(model, panel[_CLUSTER_KEY])
    wcb = wild_cluster_bootstrap(fit, _ISSUE, panel[_CLUSTER_KEY])
    inference = (
        "classic sector-clustered t; "
        if n_clusters >= 2
        else "single sector -> HC1 robust t; "
    )
    note = (
        f"lambda_I OLS [{label}]; par-yield delta_y window={window}; "
        f"sector+month FE; " + inference + bootstrap_note(wcb)
    )
    return schema.Estimate(
        value=float(fit.params[_ISSUE]),
        se=float(fit.bse[_ISSUE]),
        tstat=float(fit.tvalues[_ISSUE]),
        n=int(fit.nobs),
        note=note,
    )
