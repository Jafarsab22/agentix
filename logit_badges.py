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

# Map UI badge labels -> df_choice column names (ground-truth)
# Includes backward-compat labels and the new single-toggle pricing comparison.
_BADGE_TO_COL = {
    # Pricing frame (new single toggle)
    "All-in v. partitioned pricing": "frame",   # 1 = all-in (vs 0 = partitioned)
    # Back-compat UI labels (both map to same binary; do NOT include both at once)
    "All-in pricing": "frame",
    "Partitioned pricing": "frame",  # complement of frame; same column but reversed coding if used alone
    # Non-frame badges
    "Assurance": "assurance",
    "Scarcity tag": "scarcity",
    "Strike-through": "strike",
    "Timer": "timer",
    "social": "social_proof",
    "voucher": "voucher",
    "bundle": "bundle",
}

# Human-readable labels for output table/CSV
_UI_LABEL_TO_REPORT = {
    "All-in v. partitioned pricing": "Pricing frame (β for all-in vs partitioned)",
    "All-in pricing":                "Pricing frame (β for all-in vs partitioned)",
    "Partitioned pricing":           "Pricing frame (β for all-in vs partitioned)",
    "Assurance":                     "Assurance",
    "Scarcity tag":                  "Scarcity tag",
    "Strike-through":                "Strike-through",
    "Timer":                         "Timer",
    "social":                        "Social proof",
    "voucher":                       "Voucher",
    "bundle":                        "Bundle",
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
    Writes results/table_badges.csv if non-empty (runner handles file I/O).
    """
    if not _HAVE_SM:
        return _empty_table()

    df_choice_path = pathlib.Path(df_choice_path)
    if not df_choice_path.exists():
        return _empty_table()

    df = pd.read_csv(df_choice_path)
    if df.empty:
        return _empty_table()

    # Require complete 8-alternative screens and the standard controls
    required = {"case_id", "title", "chosen", "row_top", "col1", "col2", "col3"}
    if not required.issubset(df.columns):
        return _empty_table()

    sizes = df.groupby("case_id").size()
    keep_cases = sizes[sizes == 8].index
    df = df[df["case_id"].isin(keep_cases)].copy()
    if df.empty:
        return _empty_table()

    # ---------- build variable list ----------
    selected_badges = list(selected_badges or [])

    # Map UI labels → dataframe columns; deduplicate and keep only present columns
    mapped_cols = []
    col_to_ui = {}  # for reporting back under the requested label (esp. frame)
    for ui_lab in selected_badges:
        col = _BADGE_TO_COL.get(ui_lab)
        if not col:
            continue
        if col in df.columns and col not in mapped_cols:
            mapped_cols.append(col)
            col_to_ui[col] = ui_lab

    # If user accidentally included both legacy "All-in pricing" and "Partitioned pricing",
    # keep only one appearance (both map to 'frame' anyway).
    # With the new single toggle, only "All-in v. partitioned pricing" will be present.

    # Optional covariates (price)
    covars = []
    if "ln_price" in df.columns and df["ln_price"].notna().any():
        covars.append("ln_price")

    # Nothing to estimate
    if not mapped_cols and not covars:
        return _empty_table()

    # Base position controls
    base_cols = ["row_top", "col1", "col2", "col3"]
    if not set(base_cols).issubset(df.columns):
        return _empty_table()

    # Fixed effects: case (screen) + product title
    rhs_terms = base_cols + mapped_cols + covars + ["C(case_id)", "C(title)"]
    formula = "chosen ~ -1 + " + " + ".join(rhs_terms)

    try:
        y, X = pt.dmatrices(formula, df, return_type="dataframe")

        # Drop zero-variance columns (defensive)
        keep = [c for c in X.columns if X[c].std(ddof=0) > 0]
        X = X[keep]
        if X.shape[1] == 0:
            return _empty_table()

        params, pvals = _fit_with_fallback(y, X)

        # Prepare output rows ONLY for the levers (mapped_cols) that survived in X
        out_rows = []
        for col in mapped_cols:
            if col in X.columns and col in params.index:
                beta = float(params[col])
                pval = float(pvals[col])
                ui_lab = col_to_ui.get(col, col)

                # Use reporting label (pretty name) for pricing frame row
                label = _UI_LABEL_TO_REPORT.get(ui_lab, ui_lab)
                out_rows.append({
                    "badge": label,
                    "beta": beta,
                    "p": pval,
                    "sign": _sign(beta, pval),
                })

        return pd.DataFrame(out_rows, columns=_EMPTY_SCHEMA) if out_rows else _empty_table()

    except Exception:
        return _empty_table()
