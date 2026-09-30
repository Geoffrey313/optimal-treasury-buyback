"""Staggered event study of buyback treatment on secondary-market outcomes (P2).

A security is *treated* on the first buyback operation at which it is bought
(``OPSEC.bought``). Because operations are staggered across securities and a
security's own later operations make it a "already treated" unit, a two-way
fixed-effects event study is biased by the forbidden negative weighting of
Goodman-Bacon / de Chaisemartin. This module therefore uses a staggered-robust
event-study estimator:

* If the :mod:`differences` package (Callaway and Sant'Anna, 2021) imports, the
  group-time ATT(g, t) are estimated with never-treated securities as controls
  and aggregated to event time.
* Otherwise a clean Sun and Abraham (2021) interaction-weighted estimator is
  built by hand: cohort x relative-time interactions with security and day
  fixed effects, the saturated coefficients then averaged with weights equal to
  each cohort's share of the treated securities observed at that lead/lag. This
  removes the contamination that plain TWFE event-study dummies suffer under
  heterogeneous, staggered treatment.

Standard errors are clustered by security in both paths. The reference period
is the last pre-treatment lead ``k = -1`` (omitted). Leads and lags are binned
at the endpoints ``+/- K`` so that observations far from an operation do not
contaminate the window.

The returned frame carries, in ``.attrs``, the coefficient covariance of the
event-time thetas (``theta_vcov``) so that :func:`parallel_trends_test` can run
an exact joint Wald test of the pre-period coefficients.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from src.engine.absorption import normalize_cusip

from src.common import schema

# Column names of the returned event-study frame (kept here, not repeated).
K_COL = "k"
THETA_COL = "theta"
SE_COL = "se"
CI_LOW_COL = "ci_low"
CI_HIGH_COL = "ci_high"
N_COL = "n"
ESTIMATOR_COL = "estimator"

REFERENCE_K = -1          # omitted (normalizing) relative period
DEFAULT_ALPHA = 0.05      # two-sided confidence level for the reported CIs


# --------------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------------

def _cohorts(opsec: pd.DataFrame) -> pd.DataFrame:
    """First operation date at which each security is bought (its cohort).

    Keyed on the normalized base CUSIP so cohorts align with SECDAY.
    """
    bought = opsec.loc[opsec["bought"].astype(bool), ["cusip", "operation_date"]].copy()
    bought["cusip"] = normalize_cusip(bought["cusip"])
    first = bought.groupby("cusip", as_index=False)["operation_date"].min()
    return first.rename(columns={"operation_date": "cohort_date"})


def build_event_panel(
    secday: pd.DataFrame,
    opsec: pd.DataFrame,
    outcome: str,
    K: int,
) -> pd.DataFrame:
    """Assemble the security x day estimation panel with cohort and event time.

    Returns a frame with columns ``cusip, period, cohort_period, rel, y`` where
    ``rel`` is the endpoint-binned relative period and never-treated securities
    carry ``cohort_period = NaN`` and ``rel = NaN``.
    """
    if outcome not in secday.columns:
        raise KeyError(f"outcome {outcome!r} not in SECDAY schema {list(schema.SECDAY)}")
    if K < 1:
        raise ValueError(f"K must be a positive integer, got {K}")

    panel = secday[["cusip", "date", outcome]].copy()
    panel["cusip"] = normalize_cusip(panel["cusip"])
    panel = panel.rename(columns={outcome: "y"}).dropna(subset=["y"])

    # A single date -> period map shared by both operation and market dates.
    all_dates = pd.concat([panel["date"], opsec["operation_date"]], ignore_index=True)
    order = np.sort(pd.unique(all_dates))
    lookup = pd.Series(np.arange(len(order)), index=order)
    panel["period"] = panel["date"].map(lookup)

    cohorts = _cohorts(opsec)
    cohorts["cohort_period"] = cohorts["cohort_date"].map(lookup)
    panel = panel.merge(cohorts[["cusip", "cohort_period"]], on="cusip", how="left")

    rel = panel["period"] - panel["cohort_period"]
    # Endpoint binning keeps far-out leads/lags in the window's boundary terms.
    rel = rel.clip(lower=-K, upper=K)
    panel["rel"] = rel
    return panel


# --------------------------------------------------------------------------
# Sun-Abraham interaction-weighted estimator (self-contained fallback)
# --------------------------------------------------------------------------
def _sun_abraham(panel: pd.DataFrame, K: int, alpha: float) -> pd.DataFrame:
    """Interaction-weighted event study with security and day fixed effects.

    Saturated cohort x relative-time interactions are fit with two-way fixed
    effects and clustered (by security) inference via ``linearmodels.PanelOLS``;
    the group-specific coefficients are then averaged at each lead/lag with
    weights equal to each cohort's share of treated securities.
    """
    from linearmodels.panel import PanelOLS

    treated = panel.dropna(subset=["cohort_period"]).copy()
    treated["cohort_period"] = treated["cohort_period"].astype(int)

    rels = [k for k in range(-K, K + 1) if k != REFERENCE_K]
    cohort_ids = np.sort(treated["cohort_period"].unique())

    # Cohort sizes = number of distinct securities in each cohort.
    cohort_size = (
        treated.groupby("cohort_period")["cusip"].nunique().reindex(cohort_ids).fillna(0)
    )

    # Build cohort x relative-time interaction dummies on the full panel in one
    # shot (avoids fragmenting the frame with thousands of single inserts); a
    # never-treated (or out-of-window) row has all interactions equal to zero.
    design = panel.set_index(["cusip", "period"])
    y = design["y"]

    treated_mask = design["cohort_period"].notna() & design["rel"].isin(rels)
    if treated_mask.any():
        keys = (
            "g" + design.loc[treated_mask, "cohort_period"].astype("Int64").astype(str)
            + "_k" + design.loc[treated_mask, "rel"].astype("Int64").astype(str)
        )
        dummies = pd.get_dummies(keys, dtype=float)
        X = dummies.reindex(index=design.index, columns=dummies.columns, fill_value=0.0)
        # Keep only interaction columns that actually switch on.
        X = X.loc[:, X.sum(axis=0) > 0]
    else:
        X = design[[]]

    if X.shape[1] == 0:
        # No treated securities matched the market panel (e.g. no CUSIP overlap)
        # so there are no event dummies to identify; return a clean NaN frame.
        return _empty_event_frame(K, note="no identified event-time interactions")

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True,
                   check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    params = res.params
    vcov = res.cov
    kept = list(params.index)

    # Interaction-weighted aggregation to event time with delta-method SEs.
    z = _z(alpha)
    rows = []
    for k in range(-K, K + 1):
        if k == REFERENCE_K:
            rows.append((k, 0.0, 0.0, 0.0, 0.0, 0))
            continue
        weight = np.zeros(len(kept))
        present = []
        for g in cohort_ids:
            col = f"g{g}_k{k}"
            if col in kept:
                present.append((g, col))
        if not present:
            rows.append((k, np.nan, np.nan, np.nan, np.nan, 0))
            continue
        denom = sum(cohort_size[g] for g, _ in present)
        for g, col in present:
            weight[kept.index(col)] = cohort_size[g] / denom if denom > 0 else 0.0
        theta = float(weight @ params.values)
        var = float(weight @ vcov.values @ weight)
        se = float(np.sqrt(var)) if var > 0 else float("nan")
        n = int(sum(cohort_size[g] for g, _ in present))
        rows.append((k, theta, se, theta - z * se, theta + z * se, n))

    out = pd.DataFrame(
        rows, columns=[K_COL, THETA_COL, SE_COL, CI_LOW_COL, CI_HIGH_COL, N_COL]
    )
    out[ESTIMATOR_COL] = "sun_abraham_iw"

    # Stash the theta covariance (leads/lags only) for the joint pre-trend test.
    _attach_theta_vcov(out, kept, params, vcov, cohort_ids, cohort_size, K)
    return out


def _attach_theta_vcov(out, kept, params, vcov, cohort_ids, cohort_size, K):
    """Compute Cov(theta_k, theta_j) via the stacked weight matrix and store it."""
    ks = [k for k in range(-K, K + 1) if k != REFERENCE_K]
    W = np.zeros((len(ks), len(kept)))
    for i, k in enumerate(ks):
        present = [(g, f"g{g}_k{k}") for g in cohort_ids if f"g{g}_k{k}" in kept]
        denom = sum(cohort_size[g] for g, _ in present)
        for g, col in present:
            if denom > 0:
                W[i, kept.index(col)] = cohort_size[g] / denom
    theta_cov = W @ vcov.values @ W.T
    out.attrs["theta_vcov"] = theta_cov
    out.attrs["theta_index"] = ks


# --------------------------------------------------------------------------
# Callaway - Sant'Anna path (preferred, via the `differences` library)
# --------------------------------------------------------------------------
def _cs_column_label(col) -> str:
    """Last non-empty label of a `differences` (possibly MultiIndex) column."""
    if isinstance(col, tuple):
        parts = [str(p) for p in col if str(p) != ""]
        return parts[-1].lower() if parts else ""
    return str(col).lower()


def _callaway_santanna(panel: pd.DataFrame, K: int, alpha: float) -> pd.DataFrame:
    """Group-time ATT via Callaway and Sant'Anna, aggregated to event time.

    Never-treated securities are the comparison group; inference is clustered at
    the security (entity) level, which is the estimator's default, so no extra
    ``cluster_var`` is passed (passing the entity index name is rejected by the
    library). The library reports every relative period in the sample, so the
    result is filtered to the ``[-K, K]`` window and the reference period
    ``k = -1`` is set to zero to match this module's contract.
    """
    from differences import ATTgt

    data = panel.dropna(subset=["y"]).copy()
    # `differences` cohort convention: 0 marks never-treated units.
    data["cohort_period"] = data["cohort_period"].fillna(0).astype(int)
    data = data.set_index(["cusip", "period"])

    att = ATTgt(data=data[["cohort_period"]].assign(y=data["y"]),
                cohort_column="cohort_period")
    att.fit(formula="y", control_group="never_treated", alpha=alpha,
            progress_bar=False)
    agg = att.aggregate("event", alpha=alpha).reset_index()

    # Flatten the library's MultiIndex columns onto this module's schema.
    label_to_col = {}
    for c in agg.columns:
        label = _cs_column_label(c)
        if label == "relative_period":
            label_to_col[K_COL] = c
        elif label == "att":
            label_to_col[THETA_COL] = c
        elif label == "std_error":
            label_to_col[SE_COL] = c
        elif label == "lower":
            label_to_col[CI_LOW_COL] = c
        elif label == "upper":
            label_to_col[CI_HIGH_COL] = c
    if K_COL not in label_to_col or THETA_COL not in label_to_col:
        raise KeyError("unexpected `differences` aggregate layout")

    frame = pd.DataFrame({name: agg[col] for name, col in label_to_col.items()})
    frame[K_COL] = frame[K_COL].astype(int)
    frame = frame[(frame[K_COL] >= -K) & (frame[K_COL] <= K)].copy()

    # Ensure a complete, ordered [-K, K] grid with k = -1 as the zero reference.
    grid = pd.DataFrame({K_COL: [k for k in range(-K, K + 1)]})
    frame = grid.merge(frame, on=K_COL, how="left")
    ref = frame[K_COL] == REFERENCE_K
    frame.loc[ref, [THETA_COL, SE_COL, CI_LOW_COL, CI_HIGH_COL]] = 0.0

    # Treated-observation counts per event time (raw, un-binned relative period).
    treated = panel.dropna(subset=["cohort_period"]).copy()
    treated["rel_raw"] = (treated["period"] - treated["cohort_period"]).astype(int)
    counts = treated.groupby("rel_raw")["cusip"].nunique()
    frame[N_COL] = frame[K_COL].map(counts).fillna(0).astype(int)
    frame.loc[ref, N_COL] = 0

    frame[ESTIMATOR_COL] = "callaway_santanna"
    frame = frame.sort_values(K_COL).reset_index(drop=True)
    # CS analytic aggregate does not expose the full cross-k covariance, so the
    # joint pre-trend test uses the independent-chi2 fallback for this path.
    frame.attrs["theta_index"] = [k for k in range(-K, K + 1) if k != REFERENCE_K]
    return frame


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def estimate_event_study(
    secday: pd.DataFrame,
    opsec: pd.DataFrame,
    outcome: str = "ofr_spread",
    K: int = 10,
    alpha: float = DEFAULT_ALPHA,
    prefer_library: bool = True,
) -> pd.DataFrame:
    """Estimate a staggered-robust event study of buybacks on ``outcome``.

    Parameters
    ----------
    secday, opsec:
        Panels following :mod:`src.common.schema` (``SECDAY`` and ``OPSEC``).
    outcome:
        SECDAY column to explain, e.g. ``"ofr_spread"`` or ``"bid_ask"``.
    K:
        Half-width of the event window; coefficients are returned for
        ``k in [-K, K]`` with ``k = -1`` omitted as the reference period.
    alpha:
        Two-sided level for the reported confidence intervals.
    prefer_library:
        If True and :mod:`differences` imports, use Callaway - Sant'Anna;
        otherwise fall back to the built-in Sun - Abraham estimator.

    Returns
    -------
    DataFrame with columns ``k, theta, se, ci_low, ci_high, n, estimator`` and,
    in ``.attrs``, ``theta_vcov`` / ``theta_index`` for the joint pre-trend test.
    """
    panel = build_event_panel(secday, opsec, outcome=outcome, K=K)

    if prefer_library:
        try:
            return _callaway_santanna(panel, K=K, alpha=alpha)
        except Exception:
            # Any incompatibility -> deterministic self-contained estimator.
            pass
    return _sun_abraham(panel, K=K, alpha=alpha)


# --------------------------------------------------------------------------
# Parallel-trends (pre-period) joint test
# --------------------------------------------------------------------------
def parallel_trends_test(theta_df: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> dict:
    """Joint test that every pre-treatment coefficient ``k < -1`` equals zero.

    Uses the exact event-time covariance stored in ``theta_df.attrs`` when it is
    available (Wald statistic ``theta' V^{-1} theta ~ chi2_q``); otherwise falls
    back to a conservative test that treats the pre-period thetas as independent
    (sum of squared t-statistics), and flags this in the returned ``method``.
    """
    from scipy import stats

    pre = theta_df[(theta_df[K_COL] < REFERENCE_K)].copy()
    pre = pre.dropna(subset=[THETA_COL, SE_COL])
    q = len(pre)
    if q == 0:
        return {"stat": float("nan"), "df": 0, "p_value": float("nan"),
                "reject": False, "method": "no_pre_periods", "alpha": alpha}

    theta_vcov = theta_df.attrs.get("theta_vcov")
    theta_index = theta_df.attrs.get("theta_index")

    if theta_vcov is not None and theta_index is not None:
        idx = [theta_index.index(k) for k in pre[K_COL].tolist() if k in theta_index]
        V = np.asarray(theta_vcov)[np.ix_(idx, idx)]
        theta = pre[THETA_COL].to_numpy()[: len(idx)]
        try:
            stat = float(theta @ np.linalg.pinv(V) @ theta)
            method = "wald_joint"
        except np.linalg.LinAlgError:
            stat = float(np.sum((pre[THETA_COL] / pre[SE_COL]) ** 2))
            method = "independent_fallback"
    else:
        stat = float(np.sum((pre[THETA_COL] / pre[SE_COL]) ** 2))
        method = "independent_fallback"

    p_value = float(stats.chi2.sf(stat, df=q))
    return {"stat": stat, "df": q, "p_value": p_value,
            "reject": bool(p_value < alpha), "method": method, "alpha": alpha}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _empty_event_frame(K: int, note: str) -> pd.DataFrame:
    """A NaN event-study frame (k in [-K, K], k=-1 = 0) for an unidentified design."""
    rows = []
    for k in range(-K, K + 1):
        if k == REFERENCE_K:
            rows.append((k, 0.0, 0.0, 0.0, 0.0, 0))
        else:
            rows.append((k, np.nan, np.nan, np.nan, np.nan, 0))
    out = pd.DataFrame(
        rows, columns=[K_COL, THETA_COL, SE_COL, CI_LOW_COL, CI_HIGH_COL, N_COL]
    )
    out[ESTIMATOR_COL] = "unidentified"
    out.attrs["note"] = note
    out.attrs["theta_index"] = [k for k in range(-K, K + 1) if k != REFERENCE_K]
    out.attrs["theta_vcov"] = np.full((2 * K, 2 * K), np.nan)
    return out


def _z(alpha: float) -> float:
    """Two-sided normal critical value for confidence level ``1 - alpha``."""
    from scipy import stats

    return float(stats.norm.ppf(1.0 - alpha / 2.0))
