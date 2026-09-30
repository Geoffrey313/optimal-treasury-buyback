"""Absorption cost :math:`\\lambda_B`.

Estimates the marginal cost of absorbing repurchased securities from the
end-of-day price change around each buyback operation. Implements
Eq.~(buyimpact) of the model,

    Delta P_{i,w} = a + lambda_B * Buy_{i,w} + X_{i,w}' phi + u_{i,w},

where ``Delta P`` is the end-of-day price change of security ``i`` around
operation window ``w`` (taken from ``SECDAY`` around the operation date),
``Buy`` is ``par_accepted`` (``OPSEC``), and the controls ``X`` are the coupon,
the age (time to maturity), and the maturity sector. The slope ``lambda_B`` is
read as a marginal cost of absorption: it bundles the price impact of a large
purchase, the selection of the cheapest offers, and the liquidity value handed
to sellers.

Inputs are the ``OPSEC`` and ``SECDAY`` panels of
:mod:`src.common.schema`. The estimator is ordinary least squares of the price
change on the accepted amount, with an operation fixed effect and standard
errors clustered by security (CUSIP), returned as a
:class:`src.common.schema.Estimate` in raw units (price change per dollar of
par); the entry point rescales it to a per-billion-dollar figure for reading.

The module also holds the estimation helpers shared with
:mod:`src.engine.issuance` and :mod:`src.engine.auction_passthrough`: dtype
coercion, the base-CUSIP join key, year-month time labels, and the wild cluster
bootstrap used when a design has few clusters.
"""
from __future__ import annotations

from typing import Final, Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.common import schema
from src.common.config import DAYS_PER_YEAR, RANDOM_SEED

# ----- Event-window definition (named, not magic) -----
#: Days before the operation used for the pre-operation reference price.
PRE_OFFSET_DAYS: Final[int] = 1
#: Days after the operation used for the post-operation price.
POST_OFFSET_DAYS: Final[int] = 1

#: Name of the endogenous regressor and of the instrument.
_ENDOG: Final[str] = "par_accepted"
_INSTRUMENT: Final[str] = "z_i"
#: Cluster key for standard errors.
_CLUSTER_KEY: Final[str] = "cusip"
#: CRSP quotes the 8-character base CUSIP; buyback/auction feeds carry the full
#: 9-character CUSIP (base + check digit). Joins normalize to the shared base.
CUSIP_BASE_LENGTH: Final[int] = 8
#: Internal normalized-CUSIP join key.
_CUSIP_KEY: Final[str] = "_cusip_base"
#: Operation fixed-effect (delta_w) categorical column.
_OP_FE: Final[str] = "op_id"


# --------------------------------------------------------------------------
# Shared helpers (imported by the other engine estimation modules)
# --------------------------------------------------------------------------
def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy of ``df`` with the given columns cast to plain float64.

    The assembled panels use pandas *nullable* dtypes (``Float64``, ``Int64``,
    ``boolean``), which numpy, statsmodels, and linearmodels reject when building
    a design matrix. Casting to numpy ``float64`` turns booleans into 0/1 and
    maps ``pd.NA`` to ``np.nan`` so the estimators accept the columns.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype("float64")
    return out


def coerce_categorical(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with the given grouping columns as plain ``str`` (object).

    Nullable ``string`` and ``category`` dtypes can trip patsy's formula parser;
    plain object strings are safe for ``C(...)`` terms and for cluster groups.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype(object).astype(str)
    return out


def normalize_cusip(series: pd.Series) -> pd.Series:
    """Return the shared base CUSIP so buyback/auction feeds join to CRSP.

    CRSP quotes the 8-character base CUSIP; the fiscal-data feeds carry the full
    9-character CUSIP (base plus a check digit). Truncating both to
    :data:`CUSIP_BASE_LENGTH` aligns the keys.
    """
    return series.astype(str).str.strip().str.slice(0, CUSIP_BASE_LENGTH)


def month_period(dates: pd.Series) -> pd.Series:
    """Return a year-month label ('2024-05') for coarse time effects.

    Per-date fixed effects are collinear when a panel has roughly one row per
    date (auctions, operations), which yields a singular, over-specified design.
    Year-month effects keep a time control without exhausting the degrees of
    freedom.
    """
    return pd.to_datetime(dates).dt.to_period("M").astype(str)


def _empty_estimate(note: str) -> schema.Estimate:
    """A NaN estimate for a genuinely empty or rank-deficient design."""
    return schema.Estimate(value=float("nan"), se=float("nan"), n=0, note=note)


def has_estimable_design(
    panel: pd.DataFrame, term: str, min_extra_rows: int = 1
) -> bool:
    """True if ``panel`` can support a regression that identifies ``term``.

    Requires at least one row, positive variation in ``term``, and more rows than
    a conservative column count so the design is not over-specified.
    """
    if panel.empty or term not in panel.columns:
        return False
    if panel[term].nunique(dropna=True) < 2:
        return False
    return len(panel) > min_extra_rows


# --------------------------------------------------------------------------
# Inference with few clusters: wild cluster bootstrap (Cameron-Gelbach-Miller)
# --------------------------------------------------------------------------
#: Bootstrap replications for the wild cluster bootstrap.
WILD_BOOTSTRAP_REPS: Final[int] = 999
#: Two-sided confidence level for the bootstrap interval.
WILD_BOOTSTRAP_ALPHA: Final[float] = 0.05


def wild_cluster_bootstrap(
    fit,
    target_name: str,
    groups: pd.Series,
    *,
    n_reps: int = WILD_BOOTSTRAP_REPS,
    alpha: float = WILD_BOOTSTRAP_ALPHA,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Wild cluster bootstrap for one coefficient with few clusters.

    Follows Cameron, Gelbach and Miller (2008) with Rademacher weights. The
    p-value uses the restricted (null-imposed) bootstrap on the cluster-robust
    t-statistic, which is the reliable variant when the number of clusters is
    small. The confidence interval is the percentile interval of the unrestricted
    (point-estimate-imposed) bootstrap coefficient distribution, which stays
    robust when the design is near-degenerate (many fixed effects, few clusters)
    and the studentized ratio would be unstable. ``fit`` is a fitted statsmodels
    OLS, ``groups`` the cluster labels aligned to the fitted rows.

    Returns a dict with ``t0`` (observed cluster-robust t), ``p_value``,
    ``ci_low``, ``ci_high`` and ``reps``.
    """
    rng = np.random.default_rng(seed)
    exog = np.asarray(fit.model.exog, dtype=float)
    endog = np.asarray(fit.model.endog, dtype=float)
    names = list(fit.model.exog_names)
    codes, uniques = pd.factorize(pd.Series(groups).to_numpy())
    n_clusters = len(uniques)
    if target_name not in names or n_clusters < 2:
        return {
            "t0": float("nan"),
            "p_value": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "reps": 0,
        }
    j = names.index(target_name)
    n_obs = exog.shape[0]

    # Use the SVD-based pseudoinverse of X (not of X'X): forming X'X squares the
    # condition number and destroys precision for ill-conditioned fixed-effect
    # designs (many dummies), which would bias the recomputed coefficient.
    x_pinv = np.linalg.pinv(exog)  # k x n, beta(y) = x_pinv @ y
    beta = x_pinv @ endog
    resid = endog - exog @ beta
    rank = int(np.linalg.matrix_rank(exog))

    # Only the target coefficient is needed: beta_j(y) = row_j . y, and its
    # clustered variance is corr * sum_g (sum_{i in g} res_i h_i)^2, with
    # h = X (X'X)^-1 e_j. This is O(N) per replication.
    row_j = x_pinv[j, :]  # n-vector
    a_j = x_pinv @ row_j  # (X'X)^-1 e_j, numerically stable via SVD pinv
    h = exog @ a_j
    correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - rank))

    def _se_target(res: np.ndarray) -> float:
        cluster_sums = np.bincount(codes, weights=res * h, minlength=n_clusters)
        var_j = correction * float(np.sum(cluster_sums**2))
        return np.sqrt(var_j) if var_j > 0 else 0.0

    se0 = _se_target(resid)
    t0 = beta[j] / se0 if se0 > 0 else float("nan")

    # Restricted (null-imposed) design: drop the target column.
    keep = [c for c in range(exog.shape[1]) if c != j]
    exog_r = exog[:, keep]
    fitted_r = exog_r @ (np.linalg.pinv(exog_r) @ endog)
    resid_r = endog - fitted_r
    fitted_full = exog @ beta  # unrestricted (CI) DGP imposes the point estimate

    t_null = np.empty(n_reps)
    beta_ci = np.empty(n_reps)
    for b in range(n_reps):
        w = rng.choice(np.array([-1.0, 1.0]), size=n_clusters)[codes]
        # p-value DGP: impose the null, resample restricted residuals.
        y_null = fitted_r + w * resid_r
        res_null = y_null - exog @ (x_pinv @ y_null)
        se_null = _se_target(res_null)
        t_null[b] = (row_j @ y_null) / se_null if se_null > 0 else 0.0
        # CI DGP: impose the point estimate, resample full residuals; beta*_j = row_j . y*.
        beta_ci[b] = float(row_j @ (fitted_full + w * resid))

    p_value = (1.0 + np.sum(np.abs(t_null) >= abs(t0))) / (n_reps + 1.0)
    ci_low = float(np.quantile(beta_ci, alpha / 2.0))
    ci_high = float(np.quantile(beta_ci, 1.0 - alpha / 2.0))
    return {
        "t0": float(t0),
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "reps": int(n_reps),
    }


def fit_clustered(model, groups: pd.Series):
    """Fit an OLS with cluster-robust SE, or HC1 when a single cluster remains.

    Cluster-robust inference needs at least two clusters; with one (e.g. bills,
    which all sit in the short-end sector) statsmodels divides by zero, so the
    fit falls back to heteroskedasticity-robust HC1. Returns the fit and the
    number of clusters used.
    """
    n_clusters = int(pd.Series(groups).nunique())
    if n_clusters >= 2:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        fit = model.fit(cov_type="HC1")
    return fit, n_clusters


def bootstrap_note(wcb: dict[str, float]) -> str:
    """Format a wild-cluster-bootstrap result for an :class:`schema.Estimate`
    note."""
    if not wcb.get("reps"):
        return "wild cluster bootstrap not available (fewer than 2 clusters)"
    return (
        f"wild cluster bootstrap (Rademacher, {wcb['reps']} reps): "
        f"p={wcb['p_value']:.3f}, "
        f"{int((1 - WILD_BOOTSTRAP_ALPHA) * 100)}% CI "
        f"[{wcb['ci_low']:.3e}, {wcb['ci_high']:.3e}]"
    )


# --------------------------------------------------------------------------
# Panel construction: Delta P and controls around each operation
# --------------------------------------------------------------------------
def _nearest_price(
    operations: pd.DataFrame,
    secday: pd.DataFrame,
    *,
    offset_days: int,
    direction: str,
    value_col: str,
) -> pd.Series:
    """Return, per operation row, the price observed ``offset_days`` before or
    after the operation date.

    ``direction`` is ``"backward"`` (last observation on/around the pre date) or
    ``"forward"`` (first observation on/around the post date). The join is a
    per-CUSIP as-of merge, so it tolerates non-trading days.
    """
    sign = -1 if direction == "backward" else +1
    left = operations[[_CLUSTER_KEY, "operation_date"]].copy()
    left[_CUSIP_KEY] = normalize_cusip(left[_CLUSTER_KEY])
    left["target_date"] = left["operation_date"] + pd.to_timedelta(
        sign * offset_days, unit="D"
    )
    left = left.sort_values("target_date")
    right = secday[[_CLUSTER_KEY, "date", value_col]].copy()
    right[_CUSIP_KEY] = normalize_cusip(right[_CLUSTER_KEY])
    right = (
        right[[_CUSIP_KEY, "date", value_col]]
        .dropna(subset=[value_col])
        .sort_values("date")
    )
    merged = pd.merge_asof(
        left,
        right,
        by=_CUSIP_KEY,
        left_on="target_date",
        right_on="date",
        direction=direction,
    )
    return merged.set_index(left.index)[value_col]


def build_absorption_panel(
    opsec: pd.DataFrame, secday: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the operation-by-security regression frame.

    Adds the end-of-day price change ``delta_p`` around the operation, the age
    (time to maturity, in years) and the maturity ``sector`` (from ``SECDAY``),
    keeping the coupon, the accepted amount, and the instrument.
    """
    ops = opsec.copy()

    pre = _nearest_price(
        ops, secday, offset_days=PRE_OFFSET_DAYS, direction="backward", value_col="price"
    )
    post = _nearest_price(
        ops, secday, offset_days=POST_OFFSET_DAYS, direction="forward", value_col="price"
    )
    ops["delta_p"] = post - pre

    ops["age"] = (
        pd.to_datetime(ops["maturity_date"]) - pd.to_datetime(ops["operation_date"])
    ).dt.days / DAYS_PER_YEAR

    ops[_CUSIP_KEY] = normalize_cusip(ops[_CLUSTER_KEY])
    sector = secday[[_CLUSTER_KEY, "sector"]].copy()
    sector[_CUSIP_KEY] = normalize_cusip(sector[_CLUSTER_KEY])
    sector = (
        sector[[_CUSIP_KEY, "sector"]]
        .dropna()
        .drop_duplicates(subset=[_CUSIP_KEY])
    )
    ops = ops.merge(sector, on=_CUSIP_KEY, how="left")

    keep = [
        "delta_p",
        _ENDOG,
        "coupon_rate",
        "age",
        "sector",
        _INSTRUMENT,
        _CLUSTER_KEY,
        "operation_date",
    ]
    frame = ops[keep].copy()
    # Operation fixed effect delta_w: a categorical label per operation window.
    frame[_OP_FE] = frame["operation_date"].astype(str)
    # Nullable panel dtypes -> plain numpy floats / object strings for the design.
    frame = coerce_numeric(frame, ["delta_p", _ENDOG, "coupon_rate", "age", _INSTRUMENT])
    frame = coerce_categorical(frame, ["sector", _CLUSTER_KEY, _OP_FE])
    return frame.dropna(subset=["delta_p", _ENDOG, "coupon_rate", "age"])


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------
def _estimate_from_ols(fit, key: str, note: str) -> schema.Estimate:
    """Pack a statsmodels fit into an :class:`schema.Estimate` for one term."""
    return schema.Estimate(
        value=float(fit.params[key]),
        se=float(fit.bse[key]),
        tstat=float(fit.tvalues[key]),
        n=int(fit.nobs),
        note=note,
    )


def _absorption_formula() -> str:
    """OLS formula for Eq.~(buyimpact) with the operation fixed effect delta_w.

    The operation fixed effect ``C(op_id)`` absorbs any operation-wide shock
    (market conditions on the operation day, program-wide demand), so
    ``lambda_B`` is identified from the cross-section of securities within an
    operation. It is identified because each operation covers many securities.
    """
    return f"delta_p ~ {_ENDOG} + coupon_rate + age + C(sector) + C({_OP_FE})"


def estimate_lambda_B(
    opsec: pd.DataFrame, secday: pd.DataFrame, *, method: str = "ols"
) -> schema.Estimate:
    """Estimate the absorption slope ``lambda_B`` (Eq. buyimpact).

    The estimator is ordinary least squares with an operation fixed effect and
    standard errors clustered by CUSIP; ``method`` accepts only ``"ols"``. The
    returned estimate is the marginal cost of absorption in raw units (price
    change per dollar of par accepted).
    """
    if method != "ols":
        raise ValueError(f"method must be 'ols', got {method!r}")

    panel = build_absorption_panel(opsec, secday)
    if not has_estimable_design(panel, _ENDOG):
        return _empty_estimate("lambda_B OLS: empty or rank-deficient design")
    fit = smf.ols(_absorption_formula(), data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel[_CLUSTER_KEY]}
    )
    return _estimate_from_ols(
        fit,
        _ENDOG,
        note="lambda_B OLS with operation FE (delta_w); SE clustered by CUSIP; "
        "raw price/$ units",
    )
