# -*- coding: utf-8 -*-
"""
Conditional-logit with page (case) & product fixed effects + position controls.
Inputs: path to results/df_choice.csv and the list of badges selected in the UI.
Outputs: DataFrame with columns: badge, beta, p, sign; also writes results/table_badges.csv.
sign: '+' if p<.05 & beta>0; '−' if p<.05 & beta<0; '0' otherwise.

Robustness:
- If no badges are selected, missing columns, or no complete screens → returns an empty,
  correctly-shaped DataFrame without raising.
- If statsmodels/patsy/scipy are available, uses MLE; otherwise returns an empty table.
"""
from __future__ import annotations
import pathlib
from typing import List
import pandas as pd

try:
    import numpy as np
    import statsmodels.api as sm
    import patsy as pt
    from scipy.special import expit
    from scipy.stats import norm
    _HAVE_SM = True
except Exception:
    _HAVE_SM = False

RESULTS_DIR = pathlib.Path("results")

# Map UI badge labels -> df_choice column names
_BADGE_MAP = {
    "All-in pricing": "frame",           # 1 = all-in (vs 0 = partitioned)
    "Partitioned pricing": None,         # complement of frame — exclude to avoid collinearity
    "Assurance": "assurance",
    "Scarcity tag": "scarcity",
    "Strike-through": "strike",
    "Timer": "timer",
    "social": "social_proof",
    "voucher": "voucher",
    "bundle": "bundle",
}

_EMPTY_SCHEMA = ["badge", "beta", "p", "sign"]


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_SCHEMA)


def _sign(beta: float, p: float) -> str:
    if p < 0.05 and beta > 0:
        return "+"
    if p < 0.05 and beta < 0:
        return "−"
    return "0"


def _fit_with_fallback(y: pd.DataFrame, X: pd.DataFrame):
    """
    Primary: statsmodels Logit MLE.
    Fallback: ridge-IRLS if MLE fails numerically.
    """
    # Try canonical MLE first
    try:
        fit = sm.Logit(y.values.ravel(), X).fit(disp=0, maxiter=2000, method="lbfgs")
        params = pd.Series(fit.params, index=X.columns)
        pvals = pd.Series(fit.pvalues, index=X.columns)
        if params.isna().any() or pvals.isna().any():
            raise RuntimeError("NaN in MLE estimates")
        return params, pvals
    except Exception:
        # Ridge-IRLS fallback
        alpha = 1e-2
        I = np.eye(X.shape[1])
        beta = np.zeros(X.shape[1])
        # tiny jitter to break perfect separation symmetry
        rng = np.random.default_rng(0)
        beta += rng.normal(scale=1e-6, size=X.shape[1])

        for _ in range(200):
            xb = np.clip(X.values @ beta, -35, 35)
            p = expit(xb)
            W = p * (1 - p)
            if np.max(W) < 1e-12:
                break
            z = xb + (y.values.ravel() - p) / np.maximum(W, 1e-12)
            Xw = X.values * W[:, None]
            H = X.T.values @ Xw + 2.0 * alpha * I
            g = X.T.values @ (W * z)
            beta_new = np.linalg.pinv(H) @ g
            if np.linalg.norm(beta_new - beta, ord=np.inf) < 1e-7:
                beta = beta_new
                break
            beta = beta_new

        xb = np.clip(X.values @ beta, -35, 35)
        p = expit(xb)
        W = p * (1 - p)
        Xw = X.values * W[:, None]
        H = X.T.values @ Xw + 2.0 * alpha * I
        cov = np.linalg.pinv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        z = beta / np.where(se == 0, np.nan, se)
        pvals = 2 * (1 - norm.cdf(np.abs(z)))

        params = pd.Series(beta, index=X.columns)
        pvals = pd.Series(pvals, index=X.columns)
        return params, pvals


def run_logit(df_choice_path: pathlib.Path, selected_badges: List[str]) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: badge, beta, p, sign.
    Writes results/table_badges.csv if non-empty.
    """
    # If toolchain not present, exit gracefully
    if not _HAVE_SM:
        return _empty_table()

    df_choice_path = pathlib.Path(df_choice_path)
    if not df_choice_path.exists():
        return _empty_table()

    df = pd.read_csv(df_choice_path)
    if df.empty:
        return _empty_table()

    # Ensure required columns exist
    required = {"case_id", "title", "chosen", "row_top", "col1", "col2", "col3"}
    if not required.issubset(set(df.columns)):
        return _empty_table()

    # Keep only complete 8-alternative screens
    sizes = df.groupby("case_id").size()
    keep = sizes[sizes == 8].index
    df = df[df["case_id"].isin(keep)].copy()
    if df.empty:
        return _empty_table()

    # Translate selected badges to df columns, drop unknowns/None
    cols = []
    for b in (selected_badges or []):
        k = _BADGE_MAP.get(b)
        if k and k in df.columns:
            cols.append(k)

    # Nothing to estimate → empty table
    if not cols:
        return _empty_table()

    # Build design: position controls + selected badges + case FE + product (title) FE
    base_cols = ["row_top", "col1", "col2", "col3"]
    rhs_terms = base_cols + cols + ["C(case_id)", "C(title)"]

    # Ensure there is variance in each column referenced
    # (patsy will also drop zero-variance columns, but we add a guard)
    for c in base_cols:
        if c not in df.columns:
            return _empty_table()

    # Build matrices
    try:
        formula = "chosen ~ -1 + " + " + ".join(rhs_terms)
        y, X = pt.dmatrices(formula, df, return_type="dataframe")

        # Drop zero-variance columns to avoid singularities
        if X.shape[1] == 0:
            return _empty_table()
        X = X.loc[:, X.std(axis=0) > 0]
        if X.shape[1] == 0:
            return _empty_table()

        # Fit
        params, pvals = _fit_with_fallback(y, X)

        # Collect per-badge stats (skip any badge whose column got dropped)
        rows = []
        for b in selected_badges:
            k = _BADGE_MAP.get(b)
            if not k:
                continue
            if k not in params.index:
                # Column may have been dropped (no variance / collinear) — skip
                continue
            beta = float(params[k])
            p = float(pvals[k]) if k in pvals.index else float("nan")
            rows.append({"badge": b, "beta": beta, "p": p, "sign": _sign(beta, p)})

        # If nothing ended up estimable, return empty schema
        if not rows:
            return _empty_table()

        out = pd.DataFrame(rows, columns=_EMPTY_SCHEMA).sort_values("badge").reset_index(drop=True)

        # Persist
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(RESULTS_DIR / "table_badges.csv", index=False)
        return out

    except Exception:
        # Any modelling failure should not crash the pipeline
        return _empty_table()
