
# logit_badges.py — Agentix v1.7 compatible
# Conditional-logit with page & product fixed effects, position controls,
# and a robust ridge-IRLS fallback. Mirrors the Colab specification while
# treating dark badges as separate indicators (not a single categorical).
# Entry point: run_logit(path_or_df, selected_badges=None) → DataFrame
# Returns compact table with columns: [badge, beta, p, sign].

from __future__ import annotations

import json
from typing import Union, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import patsy as pt
from scipy.special import expit
from scipy.stats import norm


# ---------------------- helpers ----------------------
BADGE_VARS = [
    "assurance", "scarcity", "strike", "timer", "social_proof", "voucher", "bundle"
]
POSITION_VARS = ["row_top", "col1", "col2", "col3"]


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


# ---------- fast ridge-IRLS (penalised Newton) ----------
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


# ---------------- estimation core (mirrors Colab) ----------------
def _fmla(fe_case: bool = True, fe_prod: bool = True) -> str:
    rhs = POSITION_VARS + [
        "frame",
        # dark badges and other non-frame badges as separate indicators
        "assurance", "scarcity", "strike", "timer", "social_proof", "voucher", "bundle",
    ]
    if fe_case:
        rhs.append("C(case_cat)")
    if fe_prod:
        rhs.append("C(prod_cat)")
    return "chosen ~ -1 + " + " + ".join(rhs)


def _fit_mle(df: pd.DataFrame, fe_case: bool = True, fe_prod: bool = True):
    return smf.logit(_fmla(fe_case, fe_prod), data=df).fit(disp=0, maxiter=2000, method="lbfgs")


def _fit_ridge(df: pd.DataFrame, fe_case: bool = True, fe_prod: bool = True, alpha: float = 1e-2):
    y, X = pt.dmatrices(_fmla(fe_case, fe_prod), df, return_type="dataframe")
    beta, cov = _ridge_logit_irls(y.values.ravel(), X.values, alpha=alpha)
    params = pd.Series(beta, index=X.columns)
    bse = pd.Series(np.sqrt(np.diag(cov)), index=X.columns)
    z = params / bse.replace(0, np.nan)
    pvals = pd.Series(2 * (1 - norm.cdf(np.abs(z))), index=X.columns)

    class Wrap:
        pass

    w = Wrap()
    w.params = params
    w.bse = bse
    w.pvalues = pvals
    return w


def _estimate_slice(gslice: pd.DataFrame, n_cases: int):
    ok = gslice.groupby("case_id").size()
    gg = gslice[gslice["case_id"].isin(ok[ok == 8].index)].copy()
    if gg.empty:
        raise RuntimeError("No complete 8-alternative screens available for estimation.")

    try:
        fit = _fit_mle(gg, fe_case=True, fe_prod=True)
        se_focus = pd.Series(np.asarray(getattr(fit, "bse")).ravel(), index=list(fit.params.index))
        if se_focus.isna().any():
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


def _tidy_effects(fit) -> pd.DataFrame:
    idx = list(getattr(fit.params, "index", []))
    cs = pd.Series(np.asarray(fit.params).ravel(), index=idx)
    bs = pd.Series(np.asarray(getattr(fit, "bse")).ravel(), index=idx)
    ps = pd.Series(np.asarray(getattr(fit, "pvalues")).ravel(), index=idx)

    labels = [
        ("Position effects", "Row 1", "row_top"),
        ("Position effects", "Column 1", "col1"),
        ("Position effects", "Column 2", "col2"),
        ("Position effects", "Column 3", "col3"),
        ("Badge/Lever effects", "All-in framing", "frame"),
        ("Badge/Lever effects", "Purchase assurance", "assurance"),
        ("Badge/Lever effects", "Scarcity tag", "scarcity"),
        ("Badge/Lever effects", "Strike-through tag", "strike"),
        ("Badge/Lever effects", "Countdown timer", "timer"),
        ("Badge/Lever effects", "Social proof", "social_proof"),
        ("Badge/Lever effects", "Voucher", "voucher"),
        ("Badge/Lever effects", "Bundle", "bundle"),
    ]

    rows = []
    for _, lab, key in labels:
        beta = float(cs.get(key, np.nan))
        se = float(bs.get(key, np.nan))
        p = float(ps.get(key, np.nan))
        sign = "↑" if (p < 0.05 and beta > 0) else ("↓" if (p < 0.05 and beta < 0) else "")
        rows.append({"badge": lab, "beta": beta, "p": p, "sign": sign})
    return pd.DataFrame(rows)


# ---------------- public entry point ----------------
def run_logit(path_or_df: Union[str, Path, pd.DataFrame], selected_badges: List[str] | None = None) -> pd.DataFrame:
    """
    Mirrors the Colab conditional-logit specification while keeping dark badges as separate indicators.
    Behaviour:
      • No intercept; includes page (case) and product fixed effects, and position controls.
      • Retains frame as a separate 0/1 regressor.
      • Treats each non-frame badge as its own indicator column (scarcity, strike, timer, etc.).
      • Keeps only complete 8-alternative screens.
      • Falls back to ridge-IRLS when MLE is unstable.
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)

    df = _ensure_case_and_prod(df)
    df = _add_position_controls(df)

    # cast known binary columns to 0/1
    for c in ["chosen", "frame"] + BADGE_VARS:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # categorical encodings for FE
    df["case_cat"] = df["case_id"].astype("category")
    df["prod_cat"] = df["prod_id"].astype("category")

    # drop non-varying levers within the dataset
    for b in ["frame"] + BADGE_VARS:
        if b in df.columns and df[b].nunique(dropna=False) <= 1:
            df.drop(columns=[b], inplace=True)

    n_cases = int(df["case_id"].nunique())
    fit, mode = _estimate_slice(df, n_cases)
    table = _tidy_effects(fit)
    return table


# Optional: simple CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to results/df_choice.csv")
    args = parser.parse_args()
    t = run_logit(args.csv_path)
    print(t.to_csv(index=False))
