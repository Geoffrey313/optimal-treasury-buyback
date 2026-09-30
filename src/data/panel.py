"""Panel assembly for the Treasury-buyback study.

This module turns the loaded source tables (public Treasury operations, auctions,
par yields, and licensed CRSP daily/issue files) into the four analysis panels
defined by the shared data contract in :mod:`src.common.schema`:

    * OPS      - one row per buyback operation
    * OPSEC    - one row per eligible security per operation
    * SECDAY   - one row per security per trading day (secondary market)
    * AUCTION  - one row per auction

It reads only through the loader interfaces in :mod:`src.data.load_public` and
:mod:`src.data.load_crsp`; it never touches the filesystem or the network
directly (the loaders own paths, from :mod:`src.common.paths`). All constants
come from :mod:`src.common.config`; the only local constants are the
winsorization thresholds and the maturity-sector buckets, both documented below
and both kept out of the shared config only because they are private to panel
construction.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.common import config, schema
from src.data.load_crsp import load_tfz_dly, load_tfz_iss
from src.data.load_public import (
    load_auctions,
    load_buyback_operations,
    load_buyback_security_details,
    load_par_yields,
)

# --------------------------------------------------------------------------
# Local constants (private to panel construction; not shared config)
# --------------------------------------------------------------------------
# Winsorization thresholds for price-change and liquidity columns: clip each
# column to its 1st and 99th within-sample percentiles so that a handful of
# stale-quote or thin-market observations do not drive the estimates.
WINSOR: tuple[float, float] = (0.01, 0.99)

# Maturity sectors, keyed by the on-the-run tenor structure of US Treasuries.
# Each entry is (label, upper bound in years); a security is placed in the first
# bucket whose upper bound its remaining years-to-maturity does not exceed.
# Ordered from short to long; the final bucket is open-ended.
SECTOR_BUCKETS: tuple[tuple[str, float], ...] = (
    ("0-2Y", 2.0),
    ("2-3Y", 3.0),
    ("3-5Y", 5.0),
    ("5-7Y", 7.0),
    ("7-10Y", 10.0),
    ("10-20Y", 20.0),
    ("20-30Y", 30.0),
    (">30Y", math.inf),
)

# Secondary-market columns winsorized in SECDAY: the derived liquidity measures.
_SECDAY_WINSOR_COLS: tuple[str, ...] = ("bid_ask", "ofr_spread")


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------
def _col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> pd.Series:
    """Return the first present column among ``candidates`` as a Series.

    The loaders are developed in parallel and may expose a source field under
    one of a few equivalent names; resolving by candidate list keeps this module
    robust to that without hardcoding a single spelling. A required field that is
    absent raises ``KeyError``; an optional one falls back to an all-NaN column.
    """
    for name in candidates:
        if name in df.columns:
            return df[name]
    if required:
        raise KeyError(
            f"none of {list(candidates)} present in columns {list(df.columns)}"
        )
    return pd.Series(np.nan, index=df.index)


def _dates(series: pd.Series) -> pd.Series:
    """Coerce a column to timezone-naive datetimes (unparseable -> NaT)."""
    return pd.to_datetime(series, errors="coerce")


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise ratio that returns NaN wherever the denominator is 0/NaN."""
    denom = pd.to_numeric(denominator, errors="coerce")
    num = pd.to_numeric(numerator, errors="coerce")
    return num.divide(denom.where(denom != 0))


def _years_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    """Signed years from ``earlier`` to ``later``, in mean Gregorian years."""
    return (_dates(later) - _dates(earlier)).dt.days / config.DAYS_PER_YEAR


def _sector_of(years: float) -> str:
    """Map a years-to-maturity value to its sector label."""
    if pd.isna(years):
        return ""
    for label, upper in SECTOR_BUCKETS:
        if years <= upper:
            return label
    return SECTOR_BUCKETS[-1][0]


def _assign_sector(years_to_maturity: pd.Series) -> pd.Series:
    """Vectorized :func:`_sector_of` over a years-to-maturity Series."""
    return years_to_maturity.map(_sector_of)


def _select(data: dict[str, pd.Series], contract: dict[str, str]) -> pd.DataFrame:
    """Assemble a DataFrame with exactly the contract columns, in order."""
    return pd.DataFrame({name: data[name] for name in contract})


# --------------------------------------------------------------------------
# Winsorization (module-level helper)
# --------------------------------------------------------------------------
def winsorize(
    df: pd.DataFrame,
    cols: Iterable[str],
    lo: float = WINSOR[0],
    hi: float = WINSOR[1],
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``cols`` clipped to their [lo, hi] quantiles.

    Only columns that are present and numeric are clipped; the rest are left
    untouched so the helper is safe to call with a superset of column names.
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        lower, upper = values.quantile([lo, hi])
        out[col] = values.clip(lower=lower, upper=upper)
    return out


# --------------------------------------------------------------------------
# OPS: one row per buyback operation
# --------------------------------------------------------------------------
def build_ops_panel() -> pd.DataFrame:
    """Assemble the operation-level panel per :data:`schema.OPS`."""
    src = load_buyback_operations()

    offered_total = pd.to_numeric(
        _col(src, ("total_par_amt_offered", "offered_total")), errors="coerce"
    )
    q_cap = pd.to_numeric(
        _col(src, ("max_par_amt_redeemed", "q_cap")), errors="coerce"
    )
    q_real = pd.to_numeric(
        _col(src, ("total_par_amt_accepted", "q_real")), errors="coerce"
    )

    data = {
        "operation_date": _dates(_col(src, ("operation_date",))),
        "operation_type": _col(src, ("operation_type", "buyback_type")).astype("string"),
        "maturity_bucket": _col(src, ("maturity_bucket",), required=False).astype("string"),
        "settlement_date": _dates(_col(src, ("settlement_date",), required=False)),
        "q_cap": q_cap,
        "q_real": q_real,
        "offered_total": offered_total,
        "n_eligible": pd.to_numeric(
            _col(src, ("nbr_issues_eligible", "n_eligible", "total_securities_offered")),
            errors="coerce",
        ).astype("Int64"),
        "n_accepted": pd.to_numeric(
            _col(src, ("nbr_issues_accepted", "n_accepted", "total_securities_accepted")),
            errors="coerce",
        ).astype("Int64"),
        "offer_to_cap": _ratio(offered_total, q_cap),
        "offer_to_accept": _ratio(offered_total, q_real),
    }
    return _select(data, schema.OPS)


# --------------------------------------------------------------------------
# OPSEC: one row per eligible security per operation
# --------------------------------------------------------------------------
def build_opsec_panel() -> pd.DataFrame:
    """Assemble the operation x security panel per :data:`schema.OPSEC`."""
    src = load_buyback_security_details()

    maturity_date = _dates(_col(src, ("maturity_date", "maturity_dt")))
    maturity_month = maturity_date.dt.month.astype("Int64")
    par_accepted = pd.to_numeric(
        _col(src, ("par_amt_accepted", "par_accepted")), errors="coerce"
    )

    data = {
        "operation_date": _dates(_col(src, ("operation_date",))),
        "cusip": _col(src, ("cusip_nbr", "cusip")).astype("string"),
        "coupon_rate": pd.to_numeric(
            _col(src, ("coupon_rate_pct", "coupon_rate", "int_rate")), errors="coerce"
        ),
        "maturity_date": maturity_date,
        "maturity_month": maturity_month,
        "par_accepted": par_accepted,
        "wavg_price": pd.to_numeric(
            _col(src, ("weighted_avg_accepted_price", "wavg_price"), required=False),
            errors="coerce",
        ),
        # A row exists only for a security that was eligible in the operation.
        "eligible": pd.Series(True, index=src.index),
        "bought": par_accepted > 0,
        # Z_i: security matures in a heavy tax-inflow month.
        "z_i": maturity_month.isin(config.TAX_INFLOW_MONTHS).fillna(False),
    }
    return _select(data, schema.OPSEC)


# --------------------------------------------------------------------------
# SECDAY: one row per security per trading day (secondary market)
# --------------------------------------------------------------------------
def _attach_issue_attributes(dly: pd.DataFrame, iss: pd.DataFrame) -> pd.DataFrame:
    """Merge daily quotes with the issue crosswalk on ``kytreasno``."""
    key = "kytreasno"
    iss_cols = {
        key: _col(iss, (key,)),
        "tcusip": _col(iss, ("tcusip",)),
        "tmatdt": _dates(_col(iss, ("tmatdt",))),
        "tcouprt": pd.to_numeric(_col(iss, ("tcouprt",), required=False), errors="coerce"),
        "issue_date": _dates(_col(iss, ("tdatdt", "tfcaldt", "issue_date"))),
    }
    iss_small = pd.DataFrame(iss_cols).drop_duplicates(subset=[key])
    return dly.merge(iss_small, on=key, how="left")


def _mark_on_the_run(df: pd.DataFrame) -> pd.DataFrame:
    """Flag the on-the-run security per (date, sector) and set its benchmark.

    The on-the-run security is the most recently issued one (max issue date)
    among those already issued on that date; its yield is the sector benchmark
    from which the off-the-run spread of every security is measured.
    """
    out = df.copy()
    out["on_the_run"] = False
    bench = pd.Series(np.nan, index=out.index)

    for (day, sector), grp in out.groupby(["date", "sector"], sort=False):
        if not sector:
            continue
        issued = grp[grp["issue_date"].notna() & (grp["issue_date"] <= day)]
        if issued.empty:
            continue
        otr_idx = issued["issue_date"].idxmax()
        otr_cusip = out.at[otr_idx, "cusip"]
        otr_yield = out.at[otr_idx, "yield"]
        sel = grp.index
        bench.loc[sel] = otr_yield
        out.loc[sel[grp["cusip"] == otr_cusip], "on_the_run"] = True

    out["ofr_spread"] = pd.to_numeric(out["yield"], errors="coerce") - bench
    return out


def build_secday_panel() -> pd.DataFrame:
    """Assemble the security x day secondary-market panel per :data:`schema.SECDAY`."""
    dly = load_tfz_dly()
    iss = load_tfz_iss()
    merged = _attach_issue_attributes(dly, iss)

    date = _dates(_col(merged, ("caldt", "date", "mcaldt")))
    bid = pd.to_numeric(_col(merged, ("tdbid", "bid")), errors="coerce")
    ask = pd.to_numeric(_col(merged, ("tdask", "ask")), errors="coerce")

    frame = pd.DataFrame(
        {
            "cusip": _col(merged, ("tcusip", "cusip")).astype("string"),
            "date": date,
            "bid": bid,
            "ask": ask,
            "price": pd.to_numeric(_col(merged, ("tdnomprc", "price")), errors="coerce"),
            "yield": pd.to_numeric(_col(merged, ("tdyld", "yield")), errors="coerce"),
            # CRSP tdduratn is quoted in DAYS; convert to YEARS at the source so
            # every consumer reads duration in years (see schema.SECDAY).
            "duration": pd.to_numeric(
                _col(merged, ("tdduratn", "duration"), required=False), errors="coerce"
            ).where(lambda s: s > 0) / config.DAYS_PER_YEAR,
            "amount_outstanding": pd.to_numeric(
                _col(merged, ("tdpubout", "amount_outstanding"), required=False),
                errors="coerce",
            ),
            "bid_ask": ask - bid,
            "issue_date": merged["issue_date"],
            "maturity_date": merged["tmatdt"],
        }
    )
    frame["sector"] = _assign_sector(_years_between(frame["maturity_date"], frame["date"]))
    frame = _mark_on_the_run(frame)

    # Winsorize the derived liquidity (price-change scale) columns.
    frame = winsorize(frame, _SECDAY_WINSOR_COLS)

    return _select({name: frame[name] for name in schema.SECDAY}, schema.SECDAY)


# --------------------------------------------------------------------------
# AUCTION: one row per auction
# --------------------------------------------------------------------------
def _tenor_label_years(label: str) -> float:
    """Years represented by a par-yield column label, e.g. '1 Mo' -> 1/12,
    '2 Yr' -> 2, '1.5 Month' -> 0.125."""
    n = float(re.findall(r"[\d.]+", label)[0])
    is_months = "Mo" in label or "Month" in label
    return n / config.MONTHS_PER_YEAR if is_months else n


def _term_to_years(term: pd.Series) -> pd.Series:
    """Convert an auction security term ('7-Year', '52-Week', '13-Week', '4-Day',
    '2-Month') to years."""
    def one(t: object) -> float:
        if not isinstance(t, str):
            return float("nan")
        found = re.findall(r"[\d.]+", t)
        if not found:
            return float("nan")
        n, low = float(found[0]), t.lower()
        if "week" in low:
            return n * config.DAYS_PER_WEEK / config.DAYS_PER_YEAR
        if "day" in low:
            return n / config.DAYS_PER_YEAR
        if "month" in low:
            return n / config.MONTHS_PER_YEAR
        if "year" in low:
            return n
        return float("nan")
    return term.map(one)


def _auction_concession(
    auction_date: pd.Series, security_term: pd.Series, stop_out_yield: pd.Series
) -> pd.Series:
    """Concession proxy = stop-out yield minus the par-yield of the nearest tenor
    on the last business day BEFORE the auction (a tenor-level secondary-yield
    benchmark). Positive means the auction cleared above the pre-auction curve
    (the Treasury paid up). The true when-issued tail needs WI quotes not in the
    public data, so ``tail`` stays NaN; this concession is the measurable analogue.
    """
    py = load_par_yields().copy()
    py["Date"] = pd.to_datetime(py["Date"])
    tenor_cols = [c for c in py.columns if c not in ("source_year", "Date")]
    col_years = np.array([_tenor_label_years(c) for c in tenor_cols])

    years = _term_to_years(security_term)
    def nearest(y: float) -> object:
        return None if pd.isna(y) else tenor_cols[int(np.abs(col_years - y).argmin())]

    au = pd.DataFrame({
        "_idx": range(len(auction_date)),
        "auction_date": pd.to_datetime(auction_date.values),
        "tcol": years.map(nearest).values,
        "stop": pd.to_numeric(stop_out_yield, errors="coerce").values,
    })
    out = pd.Series(np.nan, index=auction_date.index, dtype="float64")
    for tcol, grp in au.dropna(subset=["tcol"]).groupby("tcol"):
        series = (py[["Date", tcol]].dropna().rename(columns={tcol: "py"})
                  .sort_values("Date"))
        merged = pd.merge_asof(
            grp.sort_values("auction_date"), series,
            left_on="auction_date", right_on="Date",
            direction="backward", allow_exact_matches=False,
        )
        vals = (merged["stop"] - merged["py"]).to_numpy()
        out.iloc[merged["_idx"].to_numpy()] = vals
    return out


def build_auction_panel() -> pd.DataFrame:
    """Assemble the auction panel per :data:`schema.AUCTION`."""
    src = load_auctions()

    auction_date = _dates(_col(src, ("auction_date",)))
    maturity_date = _dates(_col(src, ("maturity_date", "maturity_dt")))
    total_accepted = pd.to_numeric(
        _col(src, ("total_accepted",)), errors="coerce"
    )

    data = {
        "auction_date": auction_date,
        "cusip": _col(src, ("cusip",)).astype("string"),
        "security_term": _col(src, ("security_term", "term")).astype("string"),
        "security_type": _col(src, ("security_type",)).astype("string"),
        "maturity_date": maturity_date,
        "sector": _assign_sector(_years_between(maturity_date, auction_date)),
        # issue_size is the emitted amount (amount sold), NOT a ratio; the
        # offer/accept ratio is kept separately for allocation tightness.
        "issue_size": total_accepted,
        "offer_accept_ratio": _ratio(_col(src, ("offering_amt",)), total_accepted),
        "stop_out_yield": pd.to_numeric(_col(src, ("high_yield",)), errors="coerce"),
        "bid_to_cover": pd.to_numeric(
            _col(src, ("bid_to_cover_ratio", "bid_to_cover")), errors="coerce"
        ),
        "indirect_share": _ratio(_col(src, ("indirect_bidder_accepted",)), total_accepted),
        "direct_share": _ratio(_col(src, ("direct_bidder_accepted",)), total_accepted),
        "primary_share": _ratio(
            _col(src, ("primary_dealer_accepted", "primary_bidder_accepted")),
            total_accepted,
        ),
        "reopening": _reopening_flag(_col(src, ("reopening",))),
        # tail = stop-out minus when-issued yield: WI quotes are not in the public
        # data, so tail stays NaN. concession is the measurable tenor-level analogue
        # (stop-out minus the par-yield curve the business day before the auction).
        "tail": pd.Series(np.nan, index=src.index),
        "concession": _auction_concession(
            auction_date,
            _col(src, ("security_term", "term")),
            pd.to_numeric(_col(src, ("high_yield",)), errors="coerce"),
        ),
    }
    return _select(data, schema.AUCTION)


def _reopening_flag(series: pd.Series) -> pd.Series:
    """Coerce a reopening indicator ('Yes'/'No', bool, 0/1) to boolean."""
    if series.dtype == bool:
        return series
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"yes", "true", "y", "1"})


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_panels() -> dict[str, pd.DataFrame]:
    """Build and return all four analysis panels keyed by short name."""
    return {
        "ops": build_ops_panel(),
        "opsec": build_opsec_panel(),
        "secday": build_secday_panel(),
        "auction": build_auction_panel(),
    }


if __name__ == "__main__":
    # Shape check on the assembled panels; a missing input raises the loader's
    # own message naming the file to regenerate.
    for name, panel in build_panels().items():
        print(f"{name:>8}: {panel.shape}")
