# -*- coding: utf-8 -*-
"""
Conditional-logit with case (screen) & product fixed effects + position controls.
Inputs: path to results/df_choice.csv and the list of badges selected in the UI.
Outputs: DataFrame with columns: badge, beta, p, sign; also writes results/table_badges.csv.
sign: '+' if p<.05 & beta>0; '−' if p<.05 & beta<0; '0' otherwise.

Robustness:
- If no badges are selected, missing columns, or no complete screens → returns an empty,
  correctly shaped DataFrame without raising.
- Uses statsmodels Logit with patsy design matrices; falls back to ridge-IRLS if needed.
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

# ---------- UI label → df_choice column ----------
# Includes the new single-toggle label. Legacy labels map to the same 'frame' column.
_BADGE_TO_COL = {
    "All-in v. partitioned pricing": "frame",   # 1 = all-in, 0 = partitioned
    "All-in pricing":                "frame",
    "Partitioned pricing":           "frame",   # same column; coding reversed conceptually if used alone
    "Assurance":       "assurance",
    "Scarcity tag":    "scarcity",
    "Strike-through":  "strike",
    "Timer":           "timer",
    "social":          "social_proof",
    "voucher":         "voucher",
    "bundle":          "bundle",
}

# Pretty labels for reporting
_PRETTY = {
    "All-in v. partitioned pricing": "Pricing frame (β for all-in vs partitioned)",
    "All-in pricing":                "Pricing frame (β for all-in vs partitioned)",
    "Partitioned pricing":           "Pricing frame (β for all-in vs partitioned)",
    "Assurance":       "Assurance",
    "Scarcity tag":    "Scarcity tag",
    "Strike-through":  "Strike-through",
    "Timer":           "Timer",
    "social":          "Social proof",
    "voucher":         "Voucher",
    "bundle":          "Bundle",
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
    """Primary: statsmodels Logit; Fallback: ridge-IRLS."""
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
    The runner will write results/table_badges.csv if non-empty.
    """
    if not _HAVE_SM:
        return _empty_table()

    df_choice_path = pathlib.Path(df_choice_path)
    if not df_choice_path.exists():
        return _empty_table()

    df = pd.read_csv(df_choice_path)
    if df.empty:
        return _empty_table()

    # Need complete 8-alternative choice sets and position controls
    required = {"case_id", "title", "chosen", "row_top", "col1", "col2", "col3"}
    if not required.issubset(df.columns):
        return _empty_table()

    sizes = df.groupby("case_id").size()
    keep_cases = sizes[sizes == 8].index
    df = df[df["case_id"].isin(keep_cases)].copy()
    if df.empty:
        return _empty_table()

    selected_badges = list(selected_badges or [])

    # Map UI labels → present columns, keeping each column once
    mapped_cols: list[str] = []
    col_to_ui: dict[str, str] = {}
    for ui_lab in selected_badges:
        col = _BADGE_TO_COL.get(ui_lab)
        if not col:
            continue
        if col in df.columns and col not in mapped_cols:
            # Only include if there is variation (drop constants defensively)
            if df[col].nunique(dropna=False) > 1:
                mapped_cols.append(col)
                col_to_ui[col] = ui_lab

    # covariates
    covars = []
    if "ln_price" in df.columns and df["ln_price"].notna().any():
        covars.append("ln_price")

    # base position controls
    base_cols = ["row_top", "col1", "col2", "col3"]
    if not set(base_cols).issubset(df.columns):
        return _empty_table()

    # nothing to estimate
    if not mapped_cols and not covars:
        return _empty_table()

    # Fixed effects for case and product title (conditional logit emulation)
    rhs_terms = base_cols + mapped_cols + covars + ["C(case_id)", "C(title)"]
    formula = "chosen ~ -1 + " + " + ".join(rhs_terms)

    try:
        y, X = pt.dmatrices(formula, df, return_type="dataframe")

        # Drop zero-variance cols (guard against singularities)
        keep = [c for c in X.columns if X[c].std(ddof=0) > 0]
        X = X[keep]
        if X.shape[1] == 0:
            return _empty_table()

        params, pvals = _fit_with_fallback(y, X)

        # Report only the lever columns we asked to estimate (mapped_cols)
        out_rows = []
        for col in mapped_cols:
            if col in X.columns and col in params.index:
                beta = float(params[col]); pval = float(pvals[col])
                ui_lab = col_to_ui.get(col, col)
                label = _PRETTY.get(ui_lab, ui_lab)
                out_rows.append({"badge": label, "beta": beta, "p": pval, "sign": _sign(beta, pval)})

        return pd.DataFrame(out_rows, columns=_EMPTY_SCHEMA) if out_rows else _empty_table()

    except Exception:
        return _empty_table()
