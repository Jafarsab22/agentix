# -*- coding: utf-8 -*-
"""
logit_badges.py — v1.9 (2025-10-30)

Purpose
    Conditional-logit with page (screen) and product fixed effects, robust to
    small-N via ridge-IRLS. Reports BOTH position effects (Row 1, Col 1–3) and
    lever effects (frame, assurance, scarcity, strike, timer, social_proof,
    voucher, bundle). Also computes business metrics: odds ratios, BH-q, AME
    (percentage points), and price-equivalent λ for each binary regressor.

Key assumptions
    • Baselines: bottom row and rightmost column (col4) as positional bases.
    • Price slope: prefer ln_price if available/variable; fall back to price.
    • Badge filtering: if badge_filter is provided, we include ONLY those non-
      position levers among the seven non-frame badges plus frame if present.
      Position terms are always included; FE are always included for estimation
      but not reported.
    • Heat-map: 2×4 grid in log-odds units from the estimated row/column terms.

API
    run_logit(path_csv, badge_filter=None, model_label=None) -> pandas.DataFrame
    Writes PNG heat-map into results/position_heatmap_{model}.png (overwrites).
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
import patsy as pt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# For the heat-map
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- Helpers ----------------------

_POS_LABELS = {
    "row_top": "Row 1",
    "col1": "Column 1",
    "col2": "Column 2",
    "col3": "Column 3",
}

_BADGE_LABELS = {
    "frame": "All-in framing",
    "assurance": "Assurance",
    "scarcity": "Scarcity tag",
    "strike": "Strike-through",
    "timer": "Countdown timer",
    "social_proof": "Social proof",
    "voucher": "Voucher",
    "bundle": "Bundle",
}

_ALLOWED_BADGES = list(_BADGE_LABELS.keys())

def _bh_q(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR control (monotone)."""
    m = len(pvals)
    if m == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=float)
    ranks[order] = np.arange(1, m + 1)
    q = pvals * m / ranks
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q_sorted
    return np.clip(out, 0.0, 1.0)

def _stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "0"

def _fmt(x, nd=3):
    try:
        return float(x)
    except Exception:
        return float("nan")

def _select_price_column(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    """Prefer ln_price; else price; else return ('', empty)."""
    if "ln_price" in df.columns and df["ln_price"].notna().var() > 0:
        return "ln_price", df["ln_price"]
    if "price" in df.columns and df["price"].notna().var() > 0:
        return "price", df["price"]
    return "", pd.Series(dtype=float)

def _design_columns(df: pd.DataFrame, badge_filter: Optional[Iterable[str]]) -> Tuple[List[str], List[str], List[str]]:
    # Position terms always included
    pos_cols = ["row_top", "col1", "col2", "col3"]

    # Filter badges if requested; normalise and intersect with available columns
    want = set([str(x).strip().lower() for x in (badge_filter or []) if x])
    if not want:
        badge_cols = [c for c in _ALLOWED_BADGES if c in df.columns]
    else:
        badge_cols = [c for c in _ALLOWED_BADGES if c in want and c in df.columns]

    # Price
    price_key, _ = _select_price_column(df)
    price_cols = [price_key] if price_key else []

    return pos_cols, badge_cols, price_cols

def _drop_constant_cols(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    dropped = []
    keep = []
    for c in X.columns:
        v = X[c]
        if v.dtype.kind not in "fiu":
            try:
                v = pd.to_numeric(v, errors="coerce")
            except Exception:
                pass
        if v.notna().var() <= 0:
            dropped.append(c)
        else:
            keep.append(c)
    return X[keep], dropped

def _fit_mle(formula: str, data: pd.DataFrame):
    return smf.logit(formula, data=data).fit(disp=0, maxiter=2000, method="lbfgs")

def _ridge_logit_irls(y: np.ndarray, X: np.ndarray, alpha: float = 1e-2, max_iter: int = 300, tol: float = 1e-7):
    """
    Penalised IRLS: min_beta -loglik + alpha * ||beta||^2
    Returns (beta, cov, p_hat).
    """
    n, k = X.shape
    beta = np.zeros(k, dtype=float)
    rng = np.random.default_rng(0)
    beta += rng.normal(scale=1e-6, size=k)

    I = np.eye(k, dtype=float)

    for _ in range(max_iter):
        xb = np.clip(X @ beta, -35.0, 35.0)
        p = expit(xb)
        W = p * (1.0 - p)  # n
        if float(np.max(W)) < 1e-12:
            break
        z = xb + (y - p) / np.maximum(W, 1e-12)

        Xw = X * W[:, None]
        H = X.T @ Xw + 2.0 * alpha * I
        g = X.T @ (W * z)

        beta_new = np.linalg.pinv(H) @ g
        if np.linalg.norm(beta_new - beta, ord=np.inf) < tol:
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

def _fit_fast_ridge(formula: str, data: pd.DataFrame, alpha: float = 1e-2):
    y, X = pt.dmatrices(formula, data, return_type="dataframe")
    # Ensure numeric float arrays (avoid expit dtype error)
    yv = np.asarray(y.values, dtype=float).ravel()
    Xv = np.asarray(X.values, dtype=float)
    beta, cov, p = _ridge_logit_irls(yv, Xv, alpha=alpha)
    params = pd.Series(beta, index=X.columns)
    bse = pd.Series(np.sqrt(np.diag(cov)), index=X.columns)
    z = params / bse.replace(0, np.nan)
    pvals = pd.Series(2.0 * (1.0 - norm.cdf(np.abs(z))), index=X.columns)

    class Wrap:
        pass
    w = Wrap()
    w.params = params
    w.bse = bse
    w.pvalues = pvals
    w._X = Xv
    w._p = p
    w._columns = X.columns
    return w

def _heatmap_path_for(model_label: Optional[str]) -> Path:
    tag = (str(model_label or "model").strip().lower()
           .replace(" ", "_").replace("/", "_").replace("-", "_"))
    return RESULTS_DIR / f"position_heatmap_{tag}.png"

# ---------------------- Main API ----------------------

def run_logit(choice_csv_path: str, badge_filter: Optional[Iterable[str]] = None, model_label: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(choice_csv_path)
    # Keep only complete 8-alternative screens
    counts = df.groupby("case_id").size()
    df = df[df["case_id"].isin(counts[counts == 8].index)].copy()

    if df.empty:
        return pd.DataFrame(columns=["badge","beta","se","p","q_bh","odds_ratio","ci_low","ci_high","ame_pp","evid_score","price_eq","sign"])

    # Titles are constant per slot identity; use as product FE
    # Guarantee required columns are ints
    for c in ("row_top","col1","col2","col3","frame","assurance","scarcity","strike","timer","social_proof","voucher","bundle","chosen"):
        if c in df.columns:
            df[c] = df[c].astype(int)

    pos_cols, badge_cols, price_cols = _design_columns(df, badge_filter)

    # Build formula with FE: case FE (screen) and product FE (title)
    rhs_terms = pos_cols + badge_cols + price_cols + ["C(case_id)", "C(title)"]
    # Drop columns with zero variance BEFORE estimation (except FE which are symbolic)
    to_check = pos_cols + badge_cols + price_cols
    zero_vars = [c for c in to_check if c in df.columns and df[c].var() <= 0]
    if zero_vars:
        # print(f"[logit] dropped constant columns: {zero_vars}")
        rhs_terms = [t for t in rhs_terms if t not in zero_vars]

    # Price term presence
    price_key, price_series = _select_price_column(df)

    formula = "chosen ~ -1 + " + " + ".join(rhs_terms)

    # Fit ladder: MLE if reasonably sized; else ridge FE
    n_cases = df["case_id"].nunique()
    # threshold consistent with your Colab approach
    try_mle = n_cases >= 30

    fit = None
    mode = ""
    try:
        if try_mle:
            fit = _fit_mle(formula, df)
            mode = "MLE (page+product FE)"
        else:
            raise RuntimeError("force_ridge_smallN")
    except Exception:
        try:
            fit = _fit_fast_ridge(formula, df, alpha=1e-2)
            mode = "Ridge-IRLS (page+product FE)"
        except Exception:
            # final fallback: drop product FE if extremely saturated
            formula2 = "chosen ~ -1 + " + " + ".join([t for t in rhs_terms if not t.startswith("C(title)")])
            fit = _fit_fast_ridge(formula2, df, alpha=3e-2)
            mode = "Ridge-IRLS (page FE)"

    # Extract named params
    if hasattr(fit, "params"):
        params = pd.Series(np.asarray(fit.params).ravel(), index=list(getattr(fit, "params").index))
        bse = pd.Series(np.asarray(getattr(fit, "bse")).ravel(), index=list(getattr(fit, "bse").index))
        pvals = pd.Series(np.asarray(getattr(fit, "pvalues")).ravel(), index=list(getattr(fit, "pvalues").index))
    else:
        # defensive
        return pd.DataFrame(columns=["badge","beta","se","p","q_bh","odds_ratio","ci_low","ci_high","ame_pp","evid_score","price_eq","sign"])

    # Helper to get a scalar by key (0 if missing)
    def g(key: str) -> float:
        return float(params.get(key, 0.0))

    # Collect rows for position and selected badges
    rows = []

    for k in pos_cols:
        lab = _POS_LABELS.get(k, k)
        rows.append((lab, params.get(k, np.nan), bse.get(k, np.nan), pvals.get(k, np.nan)))

    for k in badge_cols:
        lab = _BADGE_LABELS.get(k, k)
        rows.append((lab, params.get(k, np.nan), bse.get(k, np.nan), pvals.get(k, np.nan)))

    # Convert to DataFrame and compute metrics
    out = pd.DataFrame(rows, columns=["badge","beta","se","p"]).copy()
    out["beta"] = out["beta"].apply(_fmt)
    out["se"] = out["se"].apply(_fmt)
    out["p"] = out["p"].apply(_fmt)

    # Odds ratios and Wald CIs in OR-space
    out["odds_ratio"] = np.exp(out["beta"])
    zcrit = 1.959963984540054
    out["ci_low"] = np.exp(out["beta"] - zcrit * out["se"])
    out["ci_high"] = np.exp(out["beta"] + zcrit * out["se"])

    # BH q-values across the reported rows
    out["q_bh"] = _bh_q(out["p"].values)

    # Approx AME (percentage points): mean(W)*beta * 100, where W = p(1-p)
    try:
        if hasattr(fit, "_p"):
            Wbar = float(np.mean(getattr(fit, "_p") * (1.0 - getattr(fit, "_p"))))
        else:
            # crude fallback if MLE: use sigmoid at 0
            Wbar = 0.25
    except Exception:
        Wbar = 0.25
    out["ame_pp"] = out["beta"] * Wbar * 100.0

    # Evidence score (1 - p) as a very compact synthesis for UI sorting
    out["evid_score"] = 1.0 - out["p"].clip(0.0, 1.0)

    # Price equivalent λ = exp(-b/β_price) - 1 (if slope identified)
    if price_key:
        b_price = float(params.get(price_key, np.nan))
        if (not math.isnan(b_price)) and abs(b_price) > 1e-8:
            out["price_eq"] = np.exp(-out["beta"] / b_price) - 1.0
        else:
            out["price_eq"] = np.nan
    else:
        out["price_eq"] = np.nan

    # Sign arrow for the UI
    def _sign_row(r):
        if r["p"] < 0.05 and r["odds_ratio"] > 1.0:
            return "↑"
        if r["p"] < 0.05 and r["odds_ratio"] < 1.0:
            return "↓"
        return "0"
    out["sign"] = out.apply(_sign_row, axis=1)

    # -------------------- Heat-map (position leverage) --------------------
    # Grid of additive utility adjustments by cell (log-odds units)
    b_row = g("row_top")
    b_c1, b_c2, b_c3 = g("col1"), g("col2"), g("col3")

    grid = np.array([
        [b_row + b_c1, b_row + b_c2, b_row + b_c3, b_row + 0.0],  # top row
        [0.0 + b_c1,   0.0 + b_c2,   0.0 + b_c3,   0.0 + 0.0],    # bottom row
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(3.6, 2.2), dpi=160)
    im = ax.imshow(grid, aspect="auto")
    ax.set_xticks([0,1,2,3], labels=["C1","C2","C3","C4"])
    ax.set_yticks([0,1], labels=["Row 1","Row 2"])
    for (i, j), val in np.ndenumerate(grid):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_title("Position leverage (log-odds)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = _heatmap_path_for(model_label)
    fig.tight_layout()
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)

    # Attach estimator note as attributes for debugging (not used by caller)
    out.attrs["estimator"] = mode
    out.attrs["heatmap_path"] = str(out_path)

    # Order rows: position first, then badges (stable UI)
    def _order_key(lbl: str) -> Tuple[int, str]:
        if lbl in ("Row 1","Column 1","Column 2","Column 3"):
            return (0, lbl)
        return (1, lbl)
    out = out.sort_values(by=["badge"], key=lambda s: s.map(lambda x: _order_key(str(x))))
    out = out.reset_index(drop=True)
    return out
