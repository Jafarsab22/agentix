# -*- coding: utf-8 -*-
"""
logit_badges — v1.10 (2025-10-30)

Purpose
    Estimate within-screen (conditional) logit effects for an 8-alternative choice
    design with page fixed effects, handling small-N runs robustly.

Key assumptions
    • Data source: results/df_choice.csv as produced by the runner (8 rows per screen).
    • Baseline utilities absorb screen-level factors via page FEs (case_id dummies).
    • Position controls included: row_top, col1, col2, col3 (baseline = bottom row, 4th col).
    • Price control: ln_price included if present and varying.
    • Badges estimated separately: frame (1 = ALL-IN), assurance, scarcity, strike, timer,
      social_proof, voucher, bundle. Any badge with no variation is dropped.
    • Small-N stability: below MIN_CASES (default 10) we always use ridge-IRLS with page FEs;
      for larger N we still default to ridge-IRLS (fast & stable). Product FEs are added
      only when N >= MIN_CASES to avoid overparameterisation in tiny runs.

Returns
    pandas.DataFrame with columns:
      ['badge','beta','se','p','q_bh','odds_ratio','ci_low','ci_high',
       'ame_pp','evid_score','price_eq','sign']

Usage
    from logit_badges import run_logit
    tbl = run_logit("results/df_choice.csv", badge_filter=["frame","assurance","scarcity"])
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# ----------------- knobs -----------------
MIN_CASES = 10          # don’t report effects when fewer than 10 complete screens
RIDGE_ALPHA_SMALL = 5e-2
RIDGE_ALPHA_LARGE = 1e-2
MAX_ITER = 300
TOL = 1e-7

ALLOWED = ["frame","assurance","scarcity","strike","timer","social_proof","voucher","bundle"]
POS_COLS = ["row_top","col1","col2","col3"]


# ----------------- core: ridge-IRLS with FE -----------------
def _ridge_logit_irls(y: np.ndarray, X: np.ndarray, alpha: float, max_iter: int = MAX_ITER, tol: float = TOL):
    """
    Penalised Newton–Raphson / IRLS for logistic regression:
      min_beta  -loglik(beta) + alpha * ||beta||^2

    Returns (beta, cov) where cov is (approx) inverse penalised Fisher info.
    """
    n, k = X.shape
    beta = np.zeros(k, dtype=np.float64)
    # tiny jitter to avoid flat start in separable corners
    rng = np.random.default_rng(0)
    beta += rng.normal(scale=1e-6, size=k)

    I = np.eye(k, dtype=np.float64)

    for _ in range(max_iter):
        xb = np.clip(X @ beta, -35.0, 35.0)           # numeric guard
        p = expit(xb)
        W = p * (1.0 - p)                             # weights
        if np.max(W) < 1e-12:
            break
        z = xb + (y - p) / np.maximum(W, 1e-12)      # working response
        Xw = X * W[:, None]
        H = X.T @ Xw + 2.0 * alpha * I               # penalised Hessian
        g = X.T @ (W * z)
        beta_new = np.linalg.pinv(H) @ g
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    xb = np.clip(X @ beta, -35.0, 35.0)
    p = expit(xb)
    W = p * (1.0 - p)
    Xw = X * W[:, None]
    H = X.T @ Xw + 2.0 * alpha * I
    cov = np.linalg.pinv(H)
    return beta, cov, p


# ----------------- helpers -----------------
def _complete_screens(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("case_id").size()
    keep = counts[counts == 8].index
    return df[df["case_id"].isin(keep)].copy()

def _is_var(col: pd.Series) -> bool:
    try:
        return (pd.to_numeric(col, errors="coerce").fillna(0.0).astype(float).nunique(dropna=False) > 1)
    except Exception:
        return False

def _build_design(df: pd.DataFrame, badge_filter: list[str] | None, n_cases: int):
    # base controls
    cols = [c for c in POS_COLS if c in df.columns]
    if "ln_price" in df.columns and _is_var(df["ln_price"]):
        cols.append("ln_price")

    # badge set
    if badge_filter:
        targets = [b for b in badge_filter if b in ALLOWED]
    else:
        targets = list(ALLOWED)

    dropped = []
    for b in targets:
        if b in df.columns and _is_var(df[b]):
            cols.append(b)
        else:
            dropped.append(b)

    if dropped:
        print(f"[logit] dropped constant columns: {dropped}", flush=True)

    # main matrix
    X_main = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float64)

    # page FE (always)
    d_case = pd.get_dummies(df["case_id"], drop_first=True, dtype=np.float64)

    # product FE only when enough screens (avoid overfit for tiny N)
    d_prod = None
    if n_cases >= MIN_CASES and "title" in df.columns:
        d_prod = pd.get_dummies(df["title"], drop_first=True, dtype=np.float64)

    if d_prod is not None:
        X = pd.concat([X_main, d_case, d_prod], axis=1)
        fe_cols = d_case.shape[1] + d_prod.shape[1]
    else:
        X = pd.concat([X_main, d_case], axis=1)
        fe_cols = d_case.shape[1]

    X = X.astype(np.float64)
    y = pd.to_numeric(df["chosen"], errors="coerce").fillna(0.0).astype(np.float64).values

    return X, y, cols, fe_cols


def _tidy(beta, cov, p_hat, cols, label_map, b_price: float | None):
    """
    Build the public effects table for the requested levers only (no position, no ln_price).
    """
    idx = list(cols)
    beta_s = pd.Series(beta[:len(idx)], index=idx, dtype=float)
    se_s = pd.Series(np.sqrt(np.diag(cov)[:len(idx)]), index=idx, dtype=float)

    # subset to the lever columns we want to report
    report_keys = [k for k in idx if (k in ALLOWED or k == "frame") and k != "ln_price" and k not in POS_COLS]
    if not report_keys:
        return pd.DataFrame(columns=["badge","beta","se","p","q_bh","odds_ratio","ci_low","ci_high","ame_pp","evid_score","price_eq","sign"])

    z = beta_s[report_keys] / se_s[report_keys].replace(0.0, np.nan)
    pvals = 2.0 * (1.0 - norm.cdf(np.abs(z.values)))
    # BH-FDR across reported badges
    _, q_bh, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    # AME: average over p*(1-p)*beta (percentage points)
    wbar = float(np.mean(p_hat * (1.0 - p_hat))) if p_hat.size else 0.0

    out = []
    for i, k in enumerate(report_keys):
        b = float(beta_s[k])
        se = float(se_s[k])
        p = float(pvals[i])
        q = float(q_bh[i])
        orx = math.exp(b)
        ci_l = math.exp(b - 1.96 * se)
        ci_h = math.exp(b + 1.96 * se)
        ame_pp = 100.0 * wbar * b
        # crude evidence score in [0,1]
        evid = max(0.0, 1.0 - p)

        if b_price is not None and abs(b_price) > 1e-9:
            price_eq = abs(b / b_price)
        else:
            price_eq = float("nan")

        sign = "↑" if (p < 0.05 and b > 0) else ("↓" if (p < 0.05 and b < 0) else "0")
        out.append({
            "badge": label_map.get(k, k),
            "beta": b, "se": se, "p": p, "q_bh": q,
            "odds_ratio": orx, "ci_low": ci_l, "ci_high": ci_h,
            "ame_pp": ame_pp, "evid_score": evid, "price_eq": price_eq,
            "sign": sign
        })
    return pd.DataFrame(out)


# ----------------- public API -----------------
def run_logit(path_csv: str, badge_filter: list[str] | None = None):
    """
    Read df_choice.csv, fit ridge-FE logit, and return a tidy badge-effects table.
    badge_filter: optional subset like ['frame','assurance','scarcity', ...].
    """
    df = pd.read_csv(path_csv)
    df = _complete_screens(df)

    n_cases = df["case_id"].nunique()
    print(f"[logit] fit_mode = ridge_default; screens={n_cases}; rows={len(df)}", flush=True)

    # show quick variability diagnostics (as in your runner debug)
    for k in ["frame","assurance","scarcity","strike","timer","social_proof","voucher","bundle"]:
        if k in df.columns:
            try:
                print(f"DEBUG {k}_unique=", int(pd.to_numeric(df[k], errors="coerce").fillna(-1).nunique(dropna=False)))
            except Exception:
                pass

    X, y, cols, fe_cols = _build_design(df, badge_filter, n_cases)

    # separate ln_price & keep for price_eq
    b_price_idx = None
    if "ln_price" in cols:
        b_price_idx = cols.index("ln_price")

    alpha = RIDGE_ALPHA_LARGE if n_cases >= MIN_CASES else RIDGE_ALPHA_SMALL

    # fit
    beta, cov, p_hat = _ridge_logit_irls(y, X.values, alpha=alpha)

    # report design size
    main_cols = len(cols)
    print(f"[logit] design: main={main_cols}; FE={fe_cols}; total_cols={X.shape[1]}", flush=True)

    b_price = float(beta[b_price_idx]) if b_price_idx is not None else None

    # pretty labels
    label_map = {
        "frame": "All-in framing",
        "assurance": "Assurance",
        "scarcity": "Scarcity tag",
        "strike": "Strike-through",
        "timer": "Timer",
        "social_proof": "Social proof",
        "voucher": "Voucher",
        "bundle": "Bundle",
    }

    table = _tidy(beta, cov, p_hat, cols, label_map, b_price)

    # deterministic column order expected by the UI
    pref_cols = ["badge","beta","se","p","q_bh","odds_ratio","ci_low","ci_high","ame_pp","evid_score","price_eq","sign"]
    table = table.reindex(columns=pref_cols)

    return table
