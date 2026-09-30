"""Shared data contract for the whole pipeline.

This module is the single source of truth for the column names of the assembled
panels and for the shape of the estimation outputs. Every acquisition, panel,
engine, and analysis module imports from here so that the parts fit together.
Do not rename a field without updating every consumer.
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Operation-level panel (one row per buyback operation)
# --------------------------------------------------------------------------
OPS = {
    "operation_date": "date",       # operation date
    "operation_type": "str",        # Liquidity Support | Cash Management | Small Value
    "maturity_bucket": "str",       # e.g. "20Y to 30Y"
    "settlement_date": "date",
    "q_cap": "float",               # max_par_amt_redeemed  (Q^cap, the instrument)
    "q_real": "float",              # total_par_amt_accepted (Q^real, an outcome)
    "offered_total": "float",       # total_par_amt_offered
    "n_eligible": "int",
    "n_accepted": "int",
    "offer_to_cap": "float",        # offered_total / q_cap
    "offer_to_accept": "float",     # offered_total / q_real
}

# --------------------------------------------------------------------------
# Operation x security panel (one row per eligible security per operation)
# --------------------------------------------------------------------------
OPSEC = {
    "operation_date": "date",
    "cusip": "str",
    "coupon_rate": "float",
    "maturity_date": "date",
    "maturity_month": "int",        # 1..12
    "par_accepted": "float",        # Buy_{i,w}; 0 if eligible but not bought
    "wavg_price": "float",          # weighted average accepted price (nullable)
    "eligible": "bool",             # Elig_{i,w} (always True: row exists = eligible)
    "bought": "bool",               # par_accepted > 0
    "z_i": "bool",                  # maturity_month in TAX_INFLOW_MONTHS
}

# --------------------------------------------------------------------------
# Security x day panel (secondary market, from CRSP + derived liquidity)
# --------------------------------------------------------------------------
SECDAY = {
    "cusip": "str",
    "date": "date",
    "bid": "float",                 # tdbid
    "ask": "float",                 # tdask
    "price": "float",               # tdnomprc
    "yield": "float",               # tdyld
    "duration": "float",            # modified duration in YEARS (tdduratn is days; converted in panel)
    "amount_outstanding": "float",  # tdpubout
    "sector": "str",                # maturity sector bucket
    "on_the_run": "bool",           # most recent issue of its sector
    "bid_ask": "float",             # ask - bid
    "ofr_spread": "float",          # yield - on-the-run benchmark yield (same maturity)
}

# --------------------------------------------------------------------------
# Auction panel (one row per auction)
# --------------------------------------------------------------------------
AUCTION = {
    "auction_date": "date",
    "cusip": "str",
    "security_term": "str",
    "security_type": "str",         # Bill | Note | Bond
    "maturity_date": "date",
    "sector": "str",
    "issue_size": "float",          # amount sold at auction = total_accepted (the emitted amount)
    "offer_accept_ratio": "float",  # offering_amt / total_accepted (allocation tightness, ~1)
    "stop_out_yield": "float",      # high_yield
    "bid_to_cover": "float",
    "indirect_share": "float",      # indirect_bidder_accepted / total_accepted
    "direct_share": "float",
    "primary_share": "float",
    "reopening": "bool",
    "tail": "float",                # derived: stop_out - when-issued/secondary yield
    "concession": "float",          # derived: auction yield - secondary yield before
}


# --------------------------------------------------------------------------
# Estimation outputs
# --------------------------------------------------------------------------
@dataclass
class Estimate:
    """A single estimated coefficient with inference."""
    value: float
    se: float
    tstat: float = float("nan")
    n: int = 0
    note: str = ""


