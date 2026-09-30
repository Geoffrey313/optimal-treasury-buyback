#!/usr/bin/env python3
"""Deterministic end-to-end reproduction entry point.

Chains: read transformed data -> engine (cost/benefit estimates) -> analysis
(auction pass-through) -> a results table. Writes results/estimates.csv (a
gitignored, reproducible output; never committed) and prints it.

Run:
    python reproduce.py            # both languages (figures step is a later task)
    python reproduce.py --lang en

Determinism: a fixed seed is set; the wild bootstrap draws from it; published
numbers carry no timestamp. It also writes the manuscript's headline numbers as LaTeX macros and the
descriptive-statistics table, so every cited number comes from the code.
"""
from __future__ import annotations

import argparse
import csv
import random
import re

import numpy as np
import pandas as pd

from src.common.config import RANDOM_SEED, SAMPLE_START
from src.common.paths import REPO_ROOT, RESULTS_DIR
from src.data.panel import build_panels
from src.engine.absorption import estimate_lambda_B, normalize_cusip
from src.engine.issuance import estimate_lambda_I
from src.engine.auction_passthrough import estimate_rho


# Representative buyback sizes for reading the mapped costs in dollars: the 2026
# long-end per-operation cap and the pre-2026 aggregate quarterly cap.
REPRESENTATIVE_Q = {"per-operation $6bn": 6e9, "quarterly $30bn": 30e9}


def set_determinism(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_panel() -> dict[str, pd.DataFrame]:
    """S0 - assemble the reproducible panels."""
    return build_panels()


def core_liquidity_support(panel: dict[str, pd.DataFrame]):
    """Return (opsec_core, ls_ops): the Liquidity Support operations since the
    sample start and the operation x security rows they cover."""
    ops = panel["ops"]
    ls_ops = ops[(ops["operation_type"] == "Liquidity Support")
                 & (ops["operation_date"] >= SAMPLE_START)]
    dates = set(ls_ops["operation_date"])
    opsec_core = panel["opsec"][panel["opsec"]["operation_date"].isin(dates)]
    return opsec_core, ls_ops


def run_engine(panel: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Absorption cost and issuance impact on the core sample."""
    opsec_core, _ = core_liquidity_support(panel)
    secday, auction = panel["secday"], panel["auction"]
    # lambda_B rests on OLS with operation fixed effects. The 2SLS(z_i) variant is
    # NOT reported: the maturity-month instrument z_i fails the CA-2.2b validity
    # gate (weak first stage and covariate imbalance) -- see run diagnostics.
    return {
        "lambda_B (OLS, op FE)": estimate_lambda_B(opsec_core, secday, method="ols"),
        "lambda_I (notes+bonds)": estimate_lambda_I(auction, classes=("Note", "Bond")),
        "lambda_I (bills)": estimate_lambda_I(auction, classes=("Bill",)),
    }


def run_analysis(panel: dict[str, pd.DataFrame], estimates: dict[str, object]) -> dict[str, object]:
    """Auction pass-through (rho); intensity is pinned to Liquidity Support inside."""
    results = dict(estimates)
    for outcome, est in estimate_rho(panel["auction"], panel["ops"]).items():
        results[f"rho ({outcome})"] = est
    return results


# Raw slopes are per dollar; this rescales to per $bn for readability. It is a
# units rescaling only, NOT the structural value-curvature mapping (workplan
# P7.1 / CA-2.6a), which is still to be done before any economic reading.
PER_BN = 1e9




def run_selection(panel: dict[str, pd.DataFrame]) -> None:
    """The Treasury's reaction function (US-03): the paper's one identified result.
    Writes results/selection.csv."""
    from src.analysis.selection import estimate_reaction_function
    r = estimate_reaction_function(panel)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "selection.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["predictor", "coef", "se", "tstat", "n"])
        for v, e in r.items():
            w.writerow([v, f"{e.value:.6f}", f"{e.se:.6f}", f"{e.tstat:.4f}", e.n])
    print("\nTreasury reaction function (selection) -> selection.csv")
    print(f"  maturity t={r['ytm'].tstat:+.2f}, coupon t={r['coupon'].tstat:+.2f} "
          f"(robust); liquidity measures spec-sensitive; n={r['ytm'].n}")


def run_event_study(panel: dict[str, pd.DataFrame]) -> None:
    """Backs the manuscript's bound that the difference-in-differences parallel
    trends do not hold: the staggered event study on the on/off-the-run spread,
    with the joint pre-trend test. Writes results/event_study.csv."""
    from src.analysis.event_study import estimate_event_study, parallel_trends_test
    opsec_core, _ = core_liquidity_support(panel)
    theta = estimate_event_study(panel["secday"], opsec_core, outcome="ofr_spread", K=8)
    pt = parallel_trends_test(theta)
    RESULTS_DIR.mkdir(exist_ok=True)
    theta.to_csv(RESULTS_DIR / "event_study.csv", index=False)
    with open(RESULTS_DIR / "event_study_pretrend.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in pt.items():
            w.writerow([k, v])
    print("\nEvent study (bounds) -> event_study.csv")
    print(f"  parallel-trends joint test: stat={pt.get('stat'):.1f}, "
          f"p={pt.get('p_value'):.3f} -> pre-trends "
          f"{'rejected' if pt.get('reject') else 'not rejected'}")


def run_matched_identification(panel: dict[str, pd.DataFrame]) -> None:
    """US-04: strengthen identification with a maturity-by-coupon matched event
    study (eligible-not-bought controls in the same cell) and re-test parallel
    trends. Writes results/matched_balance.csv, matched_event_study.csv and
    matched_pretrend.csv, and prints the decision gate."""
    from src.analysis.matched_identification import compute as matched_compute
    r = matched_compute(panel, K=8)
    RESULTS_DIR.mkdir(exist_ok=True)
    r["balance"].to_csv(RESULTS_DIR / "matched_balance.csv", index=False)
    r["theta"].to_csv(RESULTS_DIR / "matched_event_study.csv", index=False)
    pt, b = r["pretrend"], r["bound"]
    with open(RESULTS_DIR / "matched_pretrend.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k in ("n_treated", "n_treated_matched", "n_treated_dropped",
                  "n_control_matched", "n_supported_cells", "identified"):
            w.writerow([k, r[k]])
        for k, v in pt.items():
            w.writerow([f"pretrend_{k}", v])
    with open(RESULTS_DIR / "matched_bound.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in b.items():
            w.writerow([k, v])
    o = r["oster"]
    with open(RESULTS_DIR / "matched_oster.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dating", "beta_short", "beta_long", "r_long", "r_max", "delta", "n"])
        for tag in ("percell", "global"):
            x = o[tag]
            w.writerow([tag, x["beta_short"], x["beta_long"], x["r_long"],
                        x["r_max"], x["delta"], x["n"]])
        w.writerow(["reportable", o["reportable"], "", "", "", "", ""])
    print("\nMatched identification (US-04) -> matched_balance.csv, matched_event_study.csv, matched_bound.csv, matched_oster.csv")
    print(f"  {r['n_treated_matched']}/{r['n_treated']} treated matched to "
          f"{r['n_control_matched']} controls in {r['n_supported_cells']} maturity-by-coupon cells")
    print(f"  matched parallel-trends: stat={pt['stat']:.1f}, p={pt['p_value']:.3f} -> "
          f"{'NOT rejected: local effect identified' if r['identified'] else 'REJECTED: bound stands'}")
    print(f"  bound: post effect {b['att_bp']:+.4f} bp, 95% CI "
          f"[{b['ci_low_bp']:+.4f}, {b['ci_high_bp']:+.4f}] bp (effects outside this are excluded)")
    print(f"  Oster fragility: delta={o['percell']['delta']:+.2f} (per-cell) vs "
          f"{o['global']['delta']:+.2f} (global) -> "
          f"{'reportable' if o['reportable'] else 'UNSTABLE (dating flips sign): dropped, keep CI + pre-trend'}")


def run_duration_accounting(panel: dict[str, pd.DataFrame]) -> None:
    """US-03 S1: buyback duration removal vs coupon issuance, par AND ten-year
    equivalents, labeled. Writes results/duration_accounting.csv."""
    from src.analysis.duration_accounting import compute
    r = compute(panel)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "duration_accounting.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in r.items():
            w.writerow([k, f"{v:.4f}"])
    print("\nDuration accounting (US-03 S1) -> duration_accounting.csv")
    print(f"  HEADLINE: PAR ratio LS/net = {r['ratio_par_ls_over_net_pct']:.1f}% ; "
          f"DURATION ratio LS/gross = {r['ratio_duration_ls_over_gross_pct']:.1f}%")


def run_term_premium(panel: dict[str, pd.DataFrame]) -> None:
    """US-03 S4: ACM term-premium decomposition and the buyback inversion.
    Writes results/term_premium.csv."""
    from src.analysis.term_premium import compute
    r = compute(panel)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "term_premium.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in r.items():
            w.writerow([k, v])
    print("\nTerm premium & inversion (US-03 S4) -> term_premium.csv")
    print(f"  ACM 10Y: term premium {r['d_term_premium_bp']:+.0f}bp, "
          f"expected rate {r['d_expected_rate_bp']:+.0f}bp; "
          f"offset needs ${r['removal_to_offset_tp_bn_low']/1000:.1f}-"
          f"{r['removal_to_offset_tp_bn_high']/1000:.1f}tn (QE scale)")


def emit_manuscript_macros(panel: dict[str, pd.DataFrame]) -> None:
    """Single source for the manuscript's headline numbers. Writes a LaTeX macro
    file into each manuscript figures directory, so every cited number comes from
    the code and cannot drift when the pipeline is re-run (research-repo-rules)."""
    from src.analysis.duration_accounting import compute as da_compute, analytical_crsp_corr
    from src.analysis.term_premium import compute as tp_compute
    from src.analysis.selection import estimate_reaction_function, robustness_within_maturity
    da, tp = da_compute(panel), tp_compute(panel)
    sel = estimate_reaction_function(panel)
    selw = robustness_within_maturity(panel)
    macros = {
        "durCorr": f"{analytical_crsp_corr():.3f}",
        "selN": f"{sel['ytm'].n}",
        "selMaturityT": f"{sel['ytm'].tstat:.1f}",
        "selCouponT": f"{sel['coupon'].tstat:.1f}",
        # Marginal effects in percentage points: in a linear probability model the
        # coefficient is the change in the purchase probability per unit of regressor.
        "selMaturityCoefPct": f"{abs(sel['ytm'].value) * 100:.1f}",
        "selCouponCoefPct": f"{abs(sel['coupon'].value) * 100:.1f}",
        "selCouponTwithin": f"{selw['coupon'].tstat:.1f}",
        "selOfrTwithin": f"{selw['ofr_spread_z'].tstat:.1f}",
        "selBidaskTwithin": f"{selw['bid_ask_z'].tstat:.1f}",
        "lsRemovedParBn": f"{da['ls_removed_par_bn']:.0f}",
        "lsRemovedTenYBn": f"{da['ls_removed_10y_equiv_bn']:.0f}",
        "grossIssuanceParTn": f"{da['gross_issuance_par_bn']/1000:.1f}",
        "grossIssuanceTenYTn": f"{da['gross_issuance_10y_equiv_bn']/1000:.1f}",
        "netIssuanceParTn": f"{da['net_issuance_par_bn']/1000:.1f}",
        "ratioParPct": f"{da['ratio_par_ls_over_net_pct']:.1f}",
        "ratioDurationPct": f"{da['ratio_duration_ls_over_gross_pct']:.1f}",
        "tpRiseBp": f"{tp['d_term_premium_bp']:.0f}",
        "expectedRateBp": f"{tp['d_expected_rate_bp']:.0f}",
        "expectedRateAbsBp": f"{abs(tp['d_expected_rate_bp']):.0f}",
        "yieldMoveBp": f"{tp['d_yield_bp']:.0f}",
        "tpCompressionLowBp": f"{tp['tp_compression_achieved_bp_low']:.0f}",
        "tpCompressionHighBp": f"{tp['tp_compression_achieved_bp_high']:.0f}",
        "offsetLowTn": f"{tp['removal_to_offset_tp_bn_low']/1000:.1f}",
        "offsetHighTn": f"{tp['removal_to_offset_tp_bn_high']/1000:.1f}",
    }
    header = "% Generated by reproduce.py -- do not edit; numbers come from the code.\n"
    for lang in ("en", "fr"):
        # French uses a comma decimal marker; render the numeric values accordingly.
        body = header + "".join(
            f"\\newcommand{{\\fig{k}}}{{{v.replace('.', ',') if lang == 'fr' else v}}}\n"
            for k, v in macros.items())
        d = REPO_ROOT / "manuscript" / lang / "ssrn" / "figures"
        d.mkdir(parents=True, exist_ok=True)
        (d / "headline.tex").write_text(body)
    print("wrote manuscript headline macros (single source) to en/fr figures/headline.tex")


_DESC_LABELS = {
    "en": {"cap": "Variable", "n": "Obs", "hdr": ("Mean", "Std", "25\\%", "Median", "75\\%"),
           "rows": ["Accepted par per security (\\$bn)", "Modified duration (years)",
                    "Years to maturity", "Auction size (\\$bn)", "Auction stop-out yield (\\%)"],
           "cap": "Descriptive statistics"},
    "fr": {"n": "Obs", "hdr": ("Moyenne", "\\'Ecart-type", "25\\%", "M\\'ediane", "75\\%"),
           "rows": ["Par accept\\'e par titre (\\$mrd)", "Duration modifi\\'ee (ann\\'ees)",
                    "Ann\\'ees jusqu'\\`a \\'ech\\'eance", "Taille d'adjudication (\\$mrd)",
                    "Rendement d'adjudication (\\%)"],
           "cap": "Statistiques descriptives"},
}


def _descriptive_rows(panel: dict[str, pd.DataFrame]):
    from src.analysis.duration_accounting import modified_duration
    opsec_core, _ = core_liquidity_support(panel)
    b = opsec_core[opsec_core["bought"].astype(bool)].copy()
    b["ytm"] = (pd.to_datetime(b["maturity_date"]) - pd.to_datetime(b["operation_date"])).dt.days / 365.25
    b["cpn"] = pd.to_numeric(b["coupon_rate"], errors="coerce")
    b["dur"] = modified_duration(b["cpn"].values, b["ytm"].values, b["cpn"].values)
    a = panel["auction"]
    a = a[a["security_type"].isin(["Note", "Bond"])].copy()
    a["adate"] = pd.to_datetime(a["auction_date"])
    a = a[a["adate"] >= SAMPLE_START]
    series = [
        pd.to_numeric(b["par_accepted"], errors="coerce") / 1e9,
        b["dur"], b["ytm"],
        pd.to_numeric(a["issue_size"], errors="coerce") / 1e9,
        pd.to_numeric(a["stop_out_yield"], errors="coerce"),
    ]
    out = []
    for s in series:
        s = s.dropna()
        out.append([int(s.size)] + [s.mean(), s.std(),
                    s.quantile(.25), s.median(), s.quantile(.75)])
    return out


def emit_descriptive_table(panel: dict[str, pd.DataFrame]) -> None:
    """Appendix descriptive statistics (P11), generated from the code."""
    rows = _descriptive_rows(panel)
    note = {
        "en": ("Statistics over the liquidity support securities repurchased and the note "
               "and bond auctions since May 2024. Buyback and auction quantities are "
               "reported unwinsorized; the liquidity measures used in the direct estimates "
               "are winsorized at the first and ninety-ninth percentiles."),
        "fr": ("Statistiques sur les titres rachet\\'es au titre du soutien \\`a la liquidit\\'e "
               "et les adjudications de notes et bonds depuis mai 2024. Les quantit\\'es de "
               "rachat et d'adjudication sont rapport\\'ees non winsoris\\'ees ; les mesures de "
               "liquidit\\'e utilis\\'ees dans les estimations directes sont winsoris\\'ees aux "
               "premier et quatre-vingt-dix-neuvi\\`eme centiles."),
    }
    for lang in ("en", "fr"):
        L = _DESC_LABELS[lang]
        comma = (lang == "fr")

        def fmt(v):
            s = f"{v:.2f}"
            return s.replace(".", ",") if comma else s
        lines = [f"\\begin{{longtblr}}[caption={{{L['cap']}}}, label={{table:descriptive}}]{{",
                 "  width=\\linewidth, colspec={X[3,l]" + "X[1,r]" * 6 + "}, row{1}={font=\\bfseries}}",
                 "\\toprule",
                 "Variable & " + L["n"] + " & " + " & ".join(L["hdr"]) + " \\\\", "\\midrule"]
        for label, r in zip(L["rows"], rows):
            vals = " & ".join(f"{r[0]}" if i == 0 else fmt(v) for i, v in enumerate(r))
            lines.append(f"{label} & {vals} \\\\")
        lines += ["\\bottomrule", "\\end{longtblr}", f"\\tablenotes{{\\footnotesize {note[lang]}\\\\}}"]
        d = REPO_ROOT / "manuscript" / lang / "ssrn" / "figures"
        d.mkdir(parents=True, exist_ok=True)
        # Six result columns fits portrait, below the landscape threshold.
        (d / "descriptive.tex").write_text("{\n\\small\n" + "\n".join(lines) + "\n}\n")
    print("wrote descriptive-statistics table (P11) to en/fr figures/descriptive.tex")


# --------------------------------------------------------------------------
# Variable definitions (F-008) and correlation matrix (F-010), code-generated
# --------------------------------------------------------------------------
#: One row per variable: (name, definition, calculation, source), per language.
#: The definitions live here (single source), so the appendix table never drifts.
_VARDEF = {
    "en": {
        "cap": "Variable definitions and construction",
        "hdr": ("Variable", "Definition", "Construction", "Source"),
        "rows": [
            ("Accepted par", "Par amount repurchased for a security in an operation",
             "Accepted par per CUSIP-operation", "Treasury buyback results"),
            ("Modified duration", "Interest-rate sensitivity of a security, in years",
             "Closed form from coupon, maturity and yield; CRSP cross-check on the affected basket",
             "Computed; CRSP U.S. Treasury"),
            ("Years to maturity", "Remaining time to maturity at the operation date",
             "Maturity date minus operation date, in years", "Treasury buyback results"),
            ("Coupon rate", "Annual coupon rate of the security", "As issued",
             "Treasury buyback results"),
            ("On/off-the-run spread", "Yield gap of a security to its on-the-run benchmark",
             "Security yield minus the maturity-matched on-the-run yield", "CRSP U.S. Treasury"),
            ("Bid-ask spread", "Quoted bid-ask spread of a security", "Ask price minus bid price",
             "CRSP U.S. Treasury"),
            ("Auction size", "Amount sold at a note or bond auction",
             "Total accepted, competitive plus noncompetitive", "fiscaldata auctions"),
            ("Stop-out yield", "Highest accepted yield at an auction", "Auction high yield",
             "fiscaldata auctions"),
            ("Ten-year term premium", "Term-premium component of the ten-year yield",
             "Fitted ACM ten-year term premium", "Adrian, Crump and Moench (NY Fed)"),
            ("Net coupon issuance", "Gross coupon issuance net of coupon maturities over the window",
             "Sum of issue sizes minus maturing par", "fiscaldata auctions"),
        ],
        "note": ("Liquidity measures used in the direct estimates are winsorized at the first and "
                 "ninety-ninth percentiles; buyback and auction quantities are unwinsorized."),
        "corr_cap": "Correlation matrix of repurchased-security characteristics",
        "corr_vars": ("Accepted par", "Modified duration", "Years to maturity", "Coupon rate"),
        "corr_note": ("Pairwise Pearson correlations over the liquidity-support securities "
                      "repurchased since May 2024."),
    },
    "fr": {
        "cap": "D\\'efinition et construction des variables",
        "hdr": ("Variable", "D\\'efinition", "Construction", "Source"),
        "rows": [
            ("Par accept\\'e", "Montant de par rachet\\'e pour un titre dans une op\\'eration",
             "Par accept\\'e par CUSIP-op\\'eration", "R\\'esultats de rachat du Tr\\'esor"),
            ("Duration modifi\\'ee", "Sensibilit\\'e du titre au taux, en ann\\'ees",
             "Forme ferm\\'ee \\`a partir du coupon, de la maturit\\'e et du rendement ; recoupement CRSP sur le panier concern\\'e",
             "Calcul\\'ee ; CRSP U.S. Treasury"),
            ("Ann\\'ees jusqu'\\`a \\'ech\\'eance", "Dur\\'ee restante \\`a la date d'op\\'eration",
             "Date d'\\'ech\\'eance moins date d'op\\'eration, en ann\\'ees", "R\\'esultats de rachat du Tr\\'esor"),
            ("Taux de coupon", "Taux de coupon annuel du titre", "\\`A l'\\'emission",
             "R\\'esultats de rachat du Tr\\'esor"),
            ("\\'Ecart march\\'e r\\'ecent/hors", "\\'Ecart de rendement au titre de r\\'ef\\'erence",
             "Rendement du titre moins le rendement du march\\'e r\\'ecent de m\\^eme maturit\\'e", "CRSP U.S. Treasury"),
            ("\\'Ecart bid-ask", "\\'Ecart bid-ask cot\\'e du titre", "Prix ask moins prix bid",
             "CRSP U.S. Treasury"),
            ("Taille d'adjudication", "Montant vendu \\`a une adjudication de note ou bond",
             "Total accept\\'e, comp\\'etitif plus non comp\\'etitif", "Adjudications fiscaldata"),
            ("Rendement d'adjudication", "Rendement le plus \\'elev\\'e accept\\'e", "Rendement haut d'adjudication",
             "Adjudications fiscaldata"),
            ("Prime de terme \\`a dix ans", "Composante prime de terme du rendement \\`a dix ans",
             "Prime de terme ACM \\`a dix ans", "Adrian, Crump et Moench (NY Fed)"),
            ("\\'Emission nette de coupons", "\\'Emission brute nette des \\'ech\\'eances sur la fen\\^etre",
             "Somme des tailles d'\\'emission moins le par arrivant \\`a \\'ech\\'eance", "Adjudications fiscaldata"),
        ],
        "note": ("Les mesures de liquidit\\'e utilis\\'ees dans les estimations directes sont "
                 "winsoris\\'ees aux premier et quatre-vingt-dix-neuvi\\`eme centiles ; les quantit\\'es "
                 "de rachat et d'adjudication sont non winsoris\\'ees."),
        "corr_cap": "Matrice de corr\\'elation des caract\\'eristiques des titres rachet\\'es",
        "corr_vars": ("Par accept\\'e", "Duration modifi\\'ee", "Ann\\'ees \\`a \\'ech\\'eance", "Taux de coupon"),
        "corr_note": ("Corr\\'elations de Pearson deux \\`a deux sur les titres rachet\\'es au titre du "
                      "soutien \\`a la liquidit\\'e depuis mai 2024."),
    },
}


def _correlation_matrix(panel: dict[str, pd.DataFrame]):
    """Pairwise Pearson correlations of repurchased-security characteristics.

    Security-level only (par, duration, years to maturity, coupon); auction-level
    variables live on a different unit of observation and are not mixed in.
    """
    from src.analysis.duration_accounting import modified_duration
    opsec_core, _ = core_liquidity_support(panel)
    b = opsec_core[opsec_core["bought"].astype(bool)].copy()
    b["ytm"] = (pd.to_datetime(b["maturity_date"]) - pd.to_datetime(b["operation_date"])).dt.days / 365.25
    b["cpn"] = pd.to_numeric(b["coupon_rate"], errors="coerce")
    b["par"] = pd.to_numeric(b["par_accepted"], errors="coerce")
    b["dur"] = modified_duration(b["cpn"].values, b["ytm"].values, b["cpn"].values)
    m = b[["par", "dur", "ytm", "cpn"]].corr(method="pearson")
    return m.to_numpy()


def emit_variable_tables(panel: dict[str, pd.DataFrame]) -> None:
    """Variable-definitions table (F-008) and correlation matrix (F-010), from the code."""
    corr = _correlation_matrix(panel)
    for lang in ("en", "fr"):
        V = _VARDEF[lang]
        comma = (lang == "fr")

        def fmt(v):
            s = f"{v:.2f}"
            return s.replace(".", ",") if comma else s

        # Variable definitions (four-column longtblr).
        vdef = [f"\\begin{{longtblr}}[caption={{{V['cap']}}}, label={{table:vardef}}]{{",
                "  width=\\linewidth, colspec={X[2,l]X[4,l]X[4,l]X[3,l]}, row{1}={font=\\bfseries}}",
                "\\toprule", "%s & %s & %s & %s \\\\" % V["hdr"], "\\midrule"]
        for name, defn, calc, src in V["rows"]:
            vdef.append(f"{name} & {defn} & {calc} & {src} \\\\")
        vdef += ["\\bottomrule", "\\end{longtblr}", f"\\tablenotes{{\\footnotesize {V['note']}\\\\}}"]

        # Correlation matrix (lower-triangular, symmetric labels).
        cvars = V["corr_vars"]
        head = " & " + " & ".join(cvars)
        clines = [f"\\begin{{longtblr}}[caption={{{V['corr_cap']}}}, label={{table:corr}}]{{",
                  "  width=\\linewidth, colspec={X[2,l]" + "X[1,r]" * len(cvars) + "}, row{1}={font=\\bfseries}}",
                  "\\toprule", head + " \\\\", "\\midrule"]
        for i, rlab in enumerate(cvars):
            cells = [fmt(corr[i][j]) if j <= i else "" for j in range(len(cvars))]
            clines.append(f"{rlab} & " + " & ".join(cells) + " \\\\")
        clines += ["\\bottomrule", "\\end{longtblr}", f"\\tablenotes{{\\footnotesize {V['corr_note']}\\\\}}"]

        d = REPO_ROOT / "manuscript" / lang / "ssrn" / "figures"
        d.mkdir(parents=True, exist_ok=True)
        (d / "variables.tex").write_text("{\n\\small\n" + "\n".join(vdef) + "\n}\n")
        (d / "correlation.tex").write_text("{\n\\small\n" + "\n".join(clines) + "\n}\n")
    print("wrote variable-definitions (F-008) and correlation-matrix (F-010) tables to en/fr figures/")


_SEL_LABELS = {
    "en": {"cap": "The Treasury's reaction function", "col1": "Predictor",
           "hdr": ("(1) Operation FE", "(2) + maturity band"),
           "rows": {"ytm": "Years to maturity", "coupon": "Coupon rate",
                    "ofr_spread_z": "On/off-the-run spread (standardized)",
                    "bid_ask_z": "Bid-ask spread (standardized)"},
           "absorbed": "absorbed",
           "depvar_prefix": "Dependent variable:", "depvar": "$\\mathbf{1}$(repurchased)",
           "feop": "Operation FE", "fembucket": "Maturity-band FE",
           "cluster": "Cluster", "clval": "CUSIP", "rsq": "R-squared",
           "yes": "Yes", "no": "No", "nobs": "Observations",
           "note": ("Linear probability model of the purchase indicator on security "
                    "characteristics across the eligible off-the-run securities within each "
                    "operation, standard errors clustered by security, coefficients with the "
                    "$t$-statistic in parentheses. Column (2) adds maturity-band fixed effects, "
                    "so the remaining coefficients are identified within a maturity band and "
                    "maturity itself is absorbed. Maturity and coupon are the robust significant "
                    "drivers; the liquidity measures are not robust to absorbing the maturity band, "
                    "with which they are collinear, one weakening to the ten-percent margin and the "
                    "other losing significance and changing sign. ***, **, * denote significance at "
                    "1\\%, 5\\%, 10\\% (two-sided).")},
    "fr": {"cap": "La fonction de r\\'eaction du Tr\\'esor", "col1": "Pr\\'edicteur",
           "hdr": ("(1) EF op\\'eration", "(2) + bande de maturit\\'e"),
           "rows": {"ytm": "Ann\\'ees jusqu'\\`a \\'ech\\'eance", "coupon": "Taux de coupon",
                    "ofr_spread_z": "\\'Ecart march\\'e r\\'ecent/hors (standardis\\'e)",
                    "bid_ask_z": "\\'Ecart bid-ask (standardis\\'e)"},
           "absorbed": "absorb\\'ee",
           "depvar_prefix": "Variable expliqu\\'ee :", "depvar": "$\\mathbf{1}$(rachet\\'e)",
           "feop": "EF d'op\\'eration", "fembucket": "EF de bande de maturit\\'e",
           "cluster": "Regroupement", "clval": "CUSIP", "rsq": "R-deux",
           "yes": "Oui", "no": "Non", "nobs": "Observations",
           "note": ("Mod\\`ele de probabilit\\'e lin\\'eaire de l'indicateur d'achat sur les "
                    "caract\\'eristiques du titre parmi les titres hors march\\'e r\\'ecent "
                    "\\'eligibles au sein de chaque op\\'eration, \\'ecarts-types group\\'es par "
                    "titre, coefficients avec la statistique $t$ entre parenth\\`eses. La colonne "
                    "(2) ajoute des effets fixes de bande de maturit\\'e, de sorte que les "
                    "coefficients restants sont identifi\\'es au sein d'une bande et la maturit\\'e "
                    "elle-m\\^eme est absorb\\'ee. La maturit\\'e et le coupon sont les moteurs "
                    "significatifs robustes ; les mesures de liquidit\\'e ne r\\'esistent pas \\`a "
                    "l'absorption de la bande de maturit\\'e, \\`a laquelle elles sont "
                    "colin\\'eaires, l'une s'affaiblissant au seuil de dix pour cent et l'autre "
                    "perdant sa significativit\\'e et changeant de signe. ***, **, * : "
                    "significativit\\'e \\`a 1\\%, 5\\%, 10\\% (bilat\\'eral).")},
}


def run_figures(panel: dict[str, pd.DataFrame], lang: str) -> None:
    """Render every manuscript figure into the language figure directories. Reads
    the shipped data and the analysis modules, so no plotted number is hand-set."""
    from src.figures.plots import render_all
    langs = ("en", "fr") if lang == "both" else (lang,)
    for language in langs:
        names = render_all(panel, language)
    print(f"wrote {len(names)} pgfplots figures per language ({', '.join(names)}) to figures/*.tex")


def emit_selection_table(panel: dict[str, pd.DataFrame]) -> None:
    """Reaction-function table (US-03) in the house coefficient-table style: two
    columns (operation fixed effects; operation and maturity-band fixed effects),
    the coefficient with significance stars over the clustered t-statistic in
    parentheses, and a fixed-effects / observations / R-squared footer block."""
    from src.analysis.selection import estimate_reaction_function, robustness_within_maturity
    r = estimate_reaction_function(panel)
    rw = robustness_within_maturity(panel)

    def stars(t):
        a = abs(t)
        return "***" if a >= 2.576 else "**" if a >= 1.96 else "*" if a >= 1.645 else ""
    order = ["ytm", "coupon", "ofr_spread_z", "bid_ask_z"]
    for lang in ("en", "fr"):
        L = _SEL_LABELS[lang]
        comma = (lang == "fr")

        def dec(x, p=3):
            s = f"{x:.{p}f}"
            return s.replace(".", ",") if comma else s

        def coef(e):
            return "" if e is None else f"{dec(e.value)}{stars(e.tstat)}"

        def tval(e):
            return "" if e is None else f"({dec(e.tstat, 2)})"
        col2 = {"ytm": None, "coupon": rw["coupon"],
                "ofr_spread_z": rw["ofr_spread_z"], "bid_ask_z": rw["bid_ask_z"]}
        n = f"{r['ytm'].n:,}" if not comma else f"{r['ytm'].n:,}".replace(",", "\\,")
        reg = []
        for k in order:
            reg.append(f"    {L['rows'][k]} & {coef(r[k])} & {coef(col2[k])} \\\\")
            reg.append(f"     & {tval(r[k])} & {tval(col2[k])} \\\\")
        lines = [
            "{",
            "\\begin{table}[ht]",
            "    \\centering",
            f"    \\caption{{{L['cap']}}}\\label{{table:selection}}",
            f"    \\tablenotes{{\\footnotesize {L['note']}\\\\}}",
            "    \\footnotesize",
            "    \\begin{tblr}{width=0.72\\linewidth, colspec={l*{2}{X[1,c]}}}",
            "    \\toprule",
            "     & (1) & (2) \\\\",
            f"    {L['depvar_prefix']} & \\SetCell[c=2]{{c}}{{{L['depvar']}}} & \\\\",
            "    \\midrule",
            *reg,
            "     & & \\\\",
            f"    {L['feop']} & {L['yes']} & {L['yes']} \\\\",
            f"    {L['fembucket']} & {L['no']} & {L['yes']} \\\\",
            f"    {L['cluster']} & {L['clval']} & {L['clval']} \\\\",
            "     & & \\\\",
            f"    {L['nobs']} & {n} & {n} \\\\",
            # R-squared at two decimals, the house convention for this paper.
            f"    {L['rsq']} & {dec(r['r2'].value, 2)} & {dec(rw['r2'].value, 2)} \\\\",
            "    \\bottomrule",
            "    \\end{tblr}",
            "\\end{table}",
            "}",
        ]
        d = REPO_ROOT / "manuscript" / lang / "ssrn" / "figures"
        d.mkdir(parents=True, exist_ok=True)
        (d / "selection.tex").write_text("\n".join(lines) + "\n")
    print("wrote reaction-function table (selection) to en/fr figures/selection.tex")


# Numbers the abstract and the introduction spell out in words, as the manuscript form
# rules require, together with the computed macro each one stands for and the band inside
# which the word remains the correct rounding. Prose cannot carry a macro, so this is what
# keeps it from drifting silently away from the code when the data are refreshed.
PROSE_BANDS = {
    "tpRiseBp":            (75.0, 85.0,  "eighty"),
    "expectedRateAbsBp":   (55.0, 64.9,  "sixty"),
    "ratioParPct":         (5.5,  6.49,  "six"),
    "ratioDurationPct":    (2.5,  3.49,  "three"),
    "tpCompressionLowBp":  (4.5,  5.49,  "five"),
    "tpCompressionHighBp": (12.5, 13.49, "thirteen"),
    "offsetLowTn":         (1.4,  1.74,  "one and a half"),
    "offsetHighTn":        (3.5,  4.49,  "four"),
}


def check_prose_numbers() -> None:
    """Guard the spelled-out numbers in the prose against the computed values.

    The manuscript writes numbers in words in the running text, so those occurrences are
    not macro-linked and would drift in silence if a recomputation moved them. This reads
    the generated macros and fails the run when a value leaves the band in which its word
    is still the right rounding, naming the word to update."""
    macro_file = REPO_ROOT / "manuscript" / "en" / "ssrn" / "figures" / "headline.tex"
    if not macro_file.exists():
        return
    values = {m: float(v) for m, v in re.findall(
        r"\\newcommand\{\\fig(\w+)\}\{(-?[\d.]+)\}", macro_file.read_text())}
    drift = []
    for macro, (lo, hi, word) in PROSE_BANDS.items():
        if macro not in values:
            continue
        v = values[macro]
        if not (lo <= abs(v) <= hi):
            drift.append(f"  {macro} = {v} left the band [{lo}, {hi}] for the word "
                         f"\"{word}\"; update the prose in the abstract and introduction")
    if drift:
        print("\nPROSE/CODE DRIFT: a spelled-out number no longer matches the computed value")
        print("\n".join(drift))
        raise SystemExit(1)
    print(f"prose/code consistency: {len(PROSE_BANDS)} spelled-out numbers match the macros")


def emit_outputs(results: dict[str, object], language: str) -> None:
    """Write the estimates table (CSV) and print it. Also writes the descriptive-statistics table."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "estimates.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["parameter", "value_raw_per_dollar", "value_per_bn",
                    "se", "tstat", "n", "note"])
        for name, e in results.items():
            w.writerow([name, f"{e.value:.6e}", f"{e.value * PER_BN:.6f}",
                        f"{e.se:.6e}", f"{e.tstat:.4f}", e.n, e.note])
    print(f"[{language}] wrote {path}")
    print("Raw slopes are per dollar; 'per $bn' is a readability rescaling, NOT")
    print("the structural value-curvature mapping (CA-2.6a, still to do).\n")
    print(f"{'parameter':26s} {'per $bn':>12s} {'t':>8s} {'n':>6s}")
    for name, e in results.items():
        print(f"{name:26s} {e.value * PER_BN:12.4f} {e.tstat:8.2f} {e.n:6d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=["en", "fr", "both"], default="both")
    args = parser.parse_args()

    set_determinism()
    panel = build_panel()

    # CA-2.2b gate on the maturity-month instrument (must pass before z_i is used).
    from src.analysis.instrument_validity import evaluate_instrument
    opsec_core, _ = core_liquidity_support(panel)
    verdict = evaluate_instrument(opsec_core)
    print(f"CA-2.2b instrument gate: {verdict.reason}")
    print(f"  first-stage F(z_i) = {verdict.first_stage_f:.2f}")
    for b in verdict.balance:
        print(f"  balance {b.covariate:18s} diff={b.diff:8.3f} t={b.tstat:7.2f}")
    print()
    # Persist the blocking gate's verdict into the frozen trace (a referee asks for it).
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "instrument_validity.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "value", "detail"])
        w.writerow(["verdict", "PASS" if verdict.passed else "FAIL", verdict.reason])
        w.writerow(["first_stage_F", f"{verdict.first_stage_f:.4f}", "z_i on par_accepted, +controls +op FE"])
        for b in verdict.balance:
            w.writerow([f"balance:{b.covariate}", f"{b.tstat:.4f}",
                        f"z1={b.mean_z1:.4f} z0={b.mean_z0:.4f} diff={b.diff:.4f} p={b.pvalue:.3e}"])

    estimates = run_engine(panel)
    results = run_analysis(panel, estimates)

    # The estimates table is language-independent; emit once (figures per language
    # are written per language).
    emit_outputs(results, args.lang if args.lang != "both" else "en")
    run_selection(panel)
    run_duration_accounting(panel)
    run_term_premium(panel)

    # Supplementary robustness. These are not reported in the paper, which rests on the
    # decomposition, the accounting, the inversion and the reaction function. They are
    # reproduced here because the README offers the direct treatment-effect estimates and
    # their diagnostics in results/, and that offer has to be honoured by this entry point.
    run_event_study(panel)
    run_matched_identification(panel)
    emit_manuscript_macros(panel)
    emit_descriptive_table(panel)
    emit_variable_tables(panel)
    emit_selection_table(panel)
    run_figures(panel, args.lang)
    check_prose_numbers()


if __name__ == "__main__":
    main()
