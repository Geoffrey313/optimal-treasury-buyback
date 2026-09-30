# CRSP Treasury layer (licensed, not redistributed)

CRSP US Treasury data is licensed through WRDS and cannot be published. This directory
is intentionally empty in the repository.

A WRDS subscriber regenerates it by running the acquisition step (kept outside the
published repo) with credentials in `.env.local`:

- Library: `crsp_a_treasuries`
- Tables: `tfz_dly` (per-CUSIP daily bid/ask, price, yield, duration, outstanding) and
  `tfz_iss` (CUSIP crosswalk: `tcusip` <-> `kytreasno`, maturity, coupon).
- Coverage: through 2025-12-31. For 2026 (out-of-sample stress only), per-CUSIP prices
  come from the TreasuryDirect end-of-day price service.

Non-subscribers: the CRSP-derived fields are available on request from the contact in
the top-level README.
