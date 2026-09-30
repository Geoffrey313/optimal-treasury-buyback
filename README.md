# Buybacks Against the Term Premium: Scale, Selection, and the Level of Long Rates

**Authors:** withheld for double-blind review.

## Abstract

In 2024 the U.S. Treasury restarted a regular buyback program, yet long-term yields kept
rising, raising the question whether repurchases can lower the level of long rates. This
paper shows that a program of the observed size cannot. Using the fitted decomposition of
Adrian, Crump and Moench (2013), the rise in the ten-year yield over the program window is
a term-premium phenomenon: the ten-year term premium rose by about eighty basis points
while the expected-rate component fell as policy eased. Buybacks act on the term premium,
through the supply channel of preferred-habitat models, but they are small. A duration
accounting shows the liquidity-support program removed about six percent of net coupon par
and about three percent of gross coupon duration. Passing the program's ten-year-equivalent
removal through term-premium elasticities calibrated from the quantitative-easing
literature implies a compression of the term premium of only about five to thirteen basis
points; offsetting the eighty-basis-point rise would have required one and a half to four
trillion dollars of ten-year equivalents, a quantitative-easing-scale operation. The
Treasury's reaction function is systematic, repurchasing shorter-maturity, lower-coupon
off-the-run securities and so withdrawing little of the duration that carries the term
premium.

## Research questions

1. What moved the level of long rates over the program window, the expected-rate component
   or the term premium?
2. How large is the buyback relative to the coupon duration the Treasury supplies, in a
   common ten-year-equivalent unit?
3. How much term-premium compression does a removal of that size buy, and what size would
   be required to move the level?
4. Which securities does the Treasury choose to repurchase?

## Hypotheses

- **H1.** The rise in the ten-year yield is a term-premium phenomenon: in the decomposition
  of Adrian, Crump and Moench (2013), the term-premium component accounts for the level
  move while the expected-rate component does not.
- **H2.** Measured in a common ten-year-equivalent unit, the liquidity-support program
  removes only a small fraction of the net coupon duration issued over the same window,
  because purchases are funded by bill issuance and leave gross debt roughly unchanged.
- **H3.** Passing the program's removal through supply elasticities calibrated from the
  quantitative-easing literature yields a compression of only a few basis points;
  offsetting the observed term-premium rise would require a removal of
  quantitative-easing scale.
- **H4.** The program's reaction function is systematic: within each operation it
  repurchases the shorter-maturity, lower-coupon off-the-run securities, so the purchases
  reach only a limited part of the duration that carries the term premium.

## Main results

- The ten-year yield over the window rose about 19 bp: term premium +80 bp, expected-rate
  component -61 bp. The level move was a term-premium move.
- The liquidity-support program removed about $227 bn par / $252 bn ten-year equivalents:
  5.7% of net coupon issuance (par) and 3.3% of gross coupon issuance (ten-year
  equivalents), stated on their separate denominators.
- Calibrated inversion: the program compresses the term premium by about 5 to 13 bp;
  offsetting the +80 bp rise would need about $1.6 to $4.0 trillion of ten-year
  equivalents (quantitative-easing scale).
- The Treasury's reaction function is the identified object: within each operation it
  systematically repurchases the shorter-maturity, lower-coupon off-the-run securities
  (maturity and coupon significant).
- Because that selection is systematic, a naive treatment effect is not identified; the
  direct treatment-effect estimates and their diagnostics are provided as supplementary
  robustness in `results/`.

## Repository structure

```
src/            reproduction package (common, data, engine, analysis, figures)
data/           transformed input data (public layers shipped; CRSP layer on request)
reproduce.py    single deterministic entry point (writes results/ and the manuscript numbers)
requirements.txt pinned dependencies
.env.example    expected credential keys, empty values
```

## Reproduction

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env.local` and fill WRDS credentials (needed only to regenerate
   the licensed CRSP layer; the public layers ship with the repo).
4. `python reproduce.py` runs the pipeline: it reads the transformed data, runs the estimation and the
   accounting, and writes the result tables (results/) and the manuscript's headline
   numbers. Deterministic: outputs carry a stable digest across re-runs.

## Data availability

Public layers (Treasury buybacks, auctions, par yield curve, ACM term premium) ship in
`data/`. The CRSP U.S. Treasury layer is licensed through Wharton Research Data Services
and is not redistributed; a subscriber regenerates it, and it is available on request
otherwise. See `data/README.md`.

## Contact

Author identities and contact details are withheld for double-blind review. Requests for
the licensed CRSP layer, or any question about reproducing the results, should be routed
through the editor.
