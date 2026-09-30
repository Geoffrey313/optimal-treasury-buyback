# CRSP Treasury layer (licensed, not redistributed)

CRSP US Treasury data is licensed through WRDS and cannot be published. This directory
holds nothing but the present file in the repository.

A WRDS subscriber regenerates the two parquet files below from their WRDS seat, with
credentials read from `.env.local`:

- Library: `crsp_a_treasuries`
- Tables: `tfz_dly` (per-CUSIP daily bid/ask, price, yield, duration, outstanding),
  saved as `tfz_dly.parquet`, and `tfz_iss` (CUSIP crosswalk: `tcusip` <-> `kytreasno`,
  maturity, coupon), saved as `tfz_iss.parquet`.
- Coverage: through 2025-12-31. For 2026 (out-of-sample stress only), per-CUSIP prices
  come from the TreasuryDirect end-of-day price service.

Non-subscribers: the CRSP-derived fields are available on request from the contact in
the top-level README.
