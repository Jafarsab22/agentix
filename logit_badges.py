# logit_badges.py — clean, Colab-parity implementation
# Conditional-logit with page & product fixed effects, position controls,
# no intercept, and a robust ridge-IRLS fallback. Dark badges are separate
# indicators (not a single categorical). Entry point run_logit(...) returns
# a compact table with columns: [badge, beta, p, sign] where sign is "+/-/0".

from __future__ import annotations

from typing import Union, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import patsy as pt
from scipy.special import expit
from scipy.stats import norm

# ---------------------- configuration ----------------------
BADGE_VARS = [
    "assurance", "scarcity", "strike", "timer", "social_proof", "voucher", "bundle"
]
POSITION_VARS = ["row_top", "col1", "col2", "col3"]

LABELS = {
    "row_top": "Row 1",
    "col1": "Column 1",
    "col2": "Column 2",
    "col3": "Column 3",
    "frame": "All-in framing",
    "assurance": "Purchase assurance",
    "scarcity": "Scarcity tag",
    "strike": "Strike-through tag",
    "timer": "Countdown timer",
    "social_proof": "Social proof",
    "voucher": "Voucher",
    "bundle": "Bundle",
}

# Colab parity: for small N we fit ridge directly (avoid MLE separation)
RIDGE_CUTOFF_CASES = 50  # use ridge when number of complete screens < 50

# ---------------------- small utilities ----------------------
def _ensure_case_and_prod(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "case_id" not in df.columns:
        if "set_id" in df.columns:
            df["case_id"] = df["set_id"].astype(str)
        else:
            df["case_id"] = "S0001"
    if "prod_id" not in df.columns:
        if "title" in df.columns:
            df["prod_id"] = df["title"].astype(str)
        else:
            df["prod_id"] = df.groupby("case_id").cumcount().astype(str)
    return df

def _add_position_controls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "row" in df.columns:
        df["row_top"] = (df["row"].astype(int) == 0).astype(int)
    else:
        df["row_top"] = 0
    if "col" in df.columns:
        df["col"] = df["col"].astype(int)
        df["col1"] = (df["col"] == 0).astype(int)
        df["col2"] = (df["col"] == 1).astype(int)
        df["col3"] = (df["col"] == 2).astype(int)
    else:
        df["col1"] = 0
        df["col2"] = 0
        df["col3"] = 0
    return df

def _drop_nonvarying(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["frame"] + BADGE_VARS:
        if c in df.columns and df[c].nunique(dropna=False) <= 1:
            df = df.drop(columns=[c])
    return df

# ---------------------- ridge IRLS (fallback) ----------------------
def _ridge_logit_irls(y: np.ndarray, X: np.ndarray, alpha: float = 1e-2, max_iter: int = 200, tol: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
    n, k = X.shape
    beta = np.zeros(k)
    rng = np.random.default_rng(0)
    beta += rng.normal(scale=1e-6, size=k)
    I = np.eye(k)
    for _ in range(max_iter):
        xb = np.clip(X @ beta, -35, 35)
        p = expit(xb)
        W = p * (1 - p)
        if float(np.max(W)) < 1e-12:
            break
        z = xb + (y - p) / np.maximum(W, 1e-12)
        Xw = X * W[:, None]
        H = X.T @ Xw + 2.0 * alpha * I
        g = X.T @ (W * z)
        beta_new = np.linalg.pinv(H) @ g
        if float(np.linalg.norm(beta_new - beta, ord=np.inf)) < tol:
            beta = beta_new
            break
        beta = beta_new
    xb = np.clip(X @ beta, -35, 35)
    p = expit(xb)
    W = p * (1 - p)
    Xw = X * W[:, None]
    H = X.T @ Xw + 2.0 * np.eye(k)
    cov = np.linalg.pinv(H)
    return beta, cov

# ---------------------- data-aware formula ----------------------
def _fmla_from_df(df: pd.DataFrame, fe_case: bool = True, fe_prod: bool = True) -> str:
    rhs = []
    for v in POSITION_VARS:
        if v in df.columns:
            rhs.append(v)
    for v in ["frame"] + BADGE_VARS:
        if v in df.columns:
            rhs.append(v)
    if fe_case and "case_cat" in df.columns:
        rhs.append("C(case_cat)")
    if fe_prod and "prod_cat" in df.columns:
        rhs.append("C(prod_cat)")
    if not rhs:
        return "chosen ~ -1 + 0"  # benign constant-less formula
    return "chosen ~ -1 + " + " + ".join(rhs)

# ---------------------- estimators ----------------------
def _fit_mle(df: pd.DataFrame, fe_case: bool = True, fe_prod: bool = True):
    fmla = _fmla_from_df(df, fe_case, fe_prod)
    return smf.logit(fmla, data=df).fit(disp=0, maxiter=2000, method="lbfgs")

def _fit_ridge(df: pd.DataFrame, fe_case: bool = True, fe_prod: bool = True, alpha: float = 1e-2):
    fmla = _fmla_from_df(df, fe_case, fe_prod)
    y, X = pt.dmatrices(fmla, df, return_type="dataframe")
    beta, cov = _ridge_logit_irls(y.values.ravel(), X.values, alpha=alpha)
    params = pd.Series(beta, index=X.columns)
    bse = pd.Series(np.sqrt(np.diag(cov)), index=X.columns)
    z = params / bse.replace(0, np.nan)
    pvals = pd.Series(2 * (1 - norm.cdf(np.abs(z))), index=X.columns)
    class Wrap: ...
    w = Wrap(); w.params = params; w.bse = bse; w.pvalues = pvals
    return w

def _estimate(df: pd.DataFrame):
    ok = df.groupby("case_id").size()
    gg = df[df["case_id"].isin(ok[ok == 8].index)].copy()
    if gg.empty:
        raise RuntimeError("No complete 8-alternative screens available for estimation.")
    n_cases = int(gg["case_id"].nunique())

    # Colab parity: use ridge directly for small N to avoid MLE separation
    if n_cases < RIDGE_CUTOFF_CASES:
        fit = _fit_ridge(gg, fe_case=True, fe_prod=True, alpha=1e-2)
        mode = "Ridge-IRLS, page+product FE"
        return fit, mode

    try:
        fit = _fit_mle(gg, fe_case=True, fe_prod=True)
        se_ok = pd.Series(np.asarray(getattr(fit, "bse")).ravel(), index=list(fit.params.index))
        if se_ok.isna().any():
            raise RuntimeError("MLE SE NaN")
        mode = "MLE, page+product FE"
    except Exception:
        try:
            fit = _fit_ridge(gg, fe_case=True, fe_prod=True, alpha=1e-2)
            mode = "Ridge-IRLS, page+product FE"
        except Exception:
            fit = _fit_ridge(gg, fe_case=True, fe_prod=False, alpha=1e-2)
            mode = "Ridge-IRLS, page FE only"
    return fit, mode

# ---------------------- output shaping ----------------------
def _tidy_table(fit) -> pd.DataFrame:
    idx = list(getattr(fit.params, "index", []))
    cs = pd.Series(np.asarray(fit.params).ravel(), index=idx)
    ps = pd.Series(np.asarray(getattr(fit, "pvalues")).ravel(), index=idx)

    keys = []
    if "frame" in cs.index:
        keys.append("frame")
    for k in BADGE_VARS:
        if k in cs.index:
            keys.append(k)

    rows = []
    for k in keys:
        beta = float(cs.get(k, np.nan))
        p = float(ps.get(k, np.nan))
        # User’s convention: sign is purely based on coefficient sign
        sign = "+" if beta > 0 else ("-" if beta < 0 else "0")
        rows.append({"badge": LABELS.get(k, k), "beta": beta, "p": p, "sign": sign})
    return pd.DataFrame(rows, columns=["badge", "beta", "p", "sign"])

# ---------------------- public entry point ----------------------
def run_logit(path_or_df: Union[str, Path, pd.DataFrame], selected_badges: List[str] | None = None) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)

    df = _ensure_case_and_prod(df)
    df = _add_position_controls(df)

    for c in ["chosen", "frame"] + BADGE_VARS:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    df["case_cat"] = df["case_id"].astype("category")
    df["prod_cat"] = df["prod_id"].astype("category")

    df = _drop_nonvarying(df)

    if not any(c in df.columns for c in ["frame"] + BADGE_VARS):
        return pd.DataFrame(columns=["badge", "beta", "p", "sign"])

    fit, _ = _estimate(df)
    table = _tidy_table(fit)
    return table
