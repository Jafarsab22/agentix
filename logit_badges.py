# -*- coding: utf-8 -*-
"""
Conditional-logit with page (case) & product FEs + position controls.
Inputs: df_choice.csv path and the list of badges that were selected in the UI.
Outputs: DataFrame with columns: badge, beta, p, sign; writes results/table_badges.csv.
sign: '+' if p<.05 & beta>0; '−' if p<.05 & beta<0; '0' otherwise.
"""
from __future__ import annotations
import pathlib
from typing import List, Tuple
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

_BADGE_MAP = {
    "All-in pricing": "frame",           # 1=all-in (vs 0=partitioned)
    "Partitioned pricing": None,         # complement of frame — excluded to avoid perfect collinearity
    "Assurance": "assurance",
    "Scarcity tag": "scarcity",
    "Strike-through": "strike",
    "Timer": "timer",
    "social": "social_proof",
    "voucher": "voucher",
    "bundle": "bundle",
}

def _sign(beta: float, p: float) -> str:
    if p < 0.05 and beta > 0:  return "+"
    if p < 0.05 and beta < 0:  return "−"
    return "0"

def _fit_with_fallback(y, X):
    try:
        fit = sm.Logit(y.values.ravel(), X).fit(disp=0, maxiter=2000, method="lbfgs")
        params = pd.Series(fit.params, index=X.columns)
        pvals  = pd.Series(fit.pvalues, index=X.columns)
        if params.isna().any() or pvals.isna().any():
            raise RuntimeError("MLE NaN")
        return params, pvals
    except Exception:
        # ridge-IRLS
        alpha = 1e-2
        I = np.eye(X.shape[1])
        beta = np.zeros(X.shape[1])
        rng = np.random.default_rng(0)
        beta += rng.normal(scale=1e-6, size=X.shape[1])
        for _ in range(200):
            xb = np.clip(X.values @ beta, -35, 35)
            p  = expit(xb)
            W  = p * (1 - p)
            if np.max(W) < 1e-12: break
            z  = xb + (y.values.ravel() - p) / np.maximum(W, 1e-12)
            Xw = X.values * W[:, None]
            H  = X.T.values @ Xw + 2.0 * alpha * I
            g  = X.T.values @ (W * z)
            beta_new = np.linalg.pinv(H) @ g
            if np.linalg.norm(beta_new - beta, ord=np.inf) < 1e-7:
                beta = beta_new; break
            beta = beta_new
        xb = np.clip(X.values @ beta, -35, 35)
        p  = expit(xb)
        W  = p * (1 - p)
        Xw = X.values * W[:, None]
        H  = X.T.values @ Xw + 2.0 * alpha * I
        cov = np.linalg.pinv(H)
        se  = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        z   = beta / np.where(se == 0, np.nan, se)
        pvals = 2 * (1 - norm.cdf(np.abs(z)))
        params = pd.Series(beta, index=X.columns)
        pvals  = pd.Series(pvals, index=X.columns)
        return params, pvals

def run_logit(df_choice_path: pathlib.Path, selected_badges: List[str]) -> pd.DataFrame:
    if not _HAVE_SM: return pd.DataFrame()
    df_choice_path = pathlib.Path(df_choice_path)
    if not df_choice_path.exists(): return pd.DataFrame()

    df = pd.read_csv(df_choice_path)
    if df.empty: return pd.DataFrame()

    # keep only complete 8-alternative screens
    sizes = df.groupby("case_id").size()
    keep = sizes[sizes == 8].index
    df = df[df["case_id"].isin(keep)].copy()
    if df.empty: return pd.DataFrame()

    # build RHS: position controls + selected badges + page FE + product FE
    base_cols = ["row_top","col1","col2","col3"]
    cols = []
    for b in selected_badges:
        k = _BADGE_MAP.get(b)
        if k and k in df.columns:
            cols.append(k)
    if not cols:  # nothing to estimate
        return pd.DataFrame()

    rhs = base_cols + cols + ["C(case_id)", "C(title)"]
    formula = "chosen ~ -1 + " + " + ".join(rhs)

    y, X = pt.dmatrices(formula, df, return_type="dataframe")
    # drop zero-variance columns (rare but safe)
    X = X.loc[:, X.std(axis=0) > 0]

    params, pvals = _fit_with_fallback(y, X)

    rows = []
    for b in selected_badges:
        k = _BADGE_MAP.get(b)
        if not k or k not in params.index:
            continue
        beta = float(params[k]); p = float(pvals[k])
        rows.append({"badge": b, "beta": beta, "p": p, "sign": _sign(beta, p)})

    out = pd.DataFrame(rows).sort_values("badge").reset_index(drop=True)
    if not out.empty:
        (RESULTS_DIR / "table_badges.csv").write_text(out.to_csv(index=False), encoding="utf-8")
    return out
