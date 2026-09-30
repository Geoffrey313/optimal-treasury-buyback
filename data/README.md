# Reproduction data

This directory holds the **transformed input data** needed to run `reproduce.py`
end to end. Raw and licensed data are never stored here (see the licensing note).

## Layout

```
data/
├── README.md                     this file
├── buybacks/                     [public, shipped] Treasury buyback operations & per-CUSIP details
│   ├── operations.parquet        one row per operation (date, type, maturity_bucket,
│   │                             max_par_amt_redeemed = Q^cap, total offered/accepted, counts)
│   └── security_details.parquet  one row per operation x CUSIP (accepted par, wtd-avg price, eligible flag)
├── auctions/                     [public, shipped] Treasury auction results
│   └── auctions.parquet          stop-out yield, bid-to-cover, bidder shares, sizes, CUSIP, term
├── curves/                       [public, shipped] daily Treasury par yield curve (tenor level), incl. 2026
│   └── par_yields.parquet
│   └── acm_term_premium.parquet   ACM 10Y term premium / risk-neutral yield (NY Fed, public)
└── crsp/                         [licensed, NOT shipped] CRSP Treasury daily per-CUSIP prices/yields
    └── README.md                 how a WRDS subscriber regenerates this layer
```

## Sources and licensing

| Layer | Source | Redistributable? |
|-------|--------|------------------|
| `buybacks/` | U.S. Treasury Fiscal Data — Treasury Securities Buybacks (public API) | Yes — shipped |
| `auctions/` | U.S. Treasury Fiscal Data — Auctions Query + TreasuryDirect TA_WS (public) | Yes — shipped |
| `curves/` | U.S. Treasury Daily Par Yield Curve (public CSV feed) | Yes — shipped |
| `curves/acm_term_premium.parquet` | ACM term premia (Federal Reserve Bank of New York, public) | Yes — shipped |
| `crsp/` | CRSP US Treasury Database via WRDS (`crsp_a_treasuries.tfz_dly`, `tfz_iss`) | **No — licensed** |

**CRSP layer is not redistributed.** CRSP data is licensed through WRDS and cannot be
published. A researcher with a WRDS academic seat regenerates it by running the
acquisition step (documented in `crsp/README.md`) with credentials in `.env.local`.
All CRSP-derived fields are therefore *available on request* to non-subscribers, or
reproduced locally by any WRDS subscriber. The public layers above are sufficient to
reproduce every result that does not depend on per-CUSIP secondary prices.

## Coverage note (2026 gap)

CRSP daily coverage ends 2025-12-31. The core estimation window (2024–2025) is fully
covered by CRSP. Tenor-level 2026 data (par yields) is public and shipped. Per-CUSIP
2026 prices, used only in the out-of-sample 2026 stress test, are obtained from the
TreasuryDirect end-of-day price service and documented at acquisition time.

## Contact

Withheld for double-blind review; route requests through the editor.
