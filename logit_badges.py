# -*- coding: utf-8 -*-
"""
logit_badges — v1.13 (2025-10-30)

Purpose
    Estimate within-screen (conditional) logit effects with page FEs, robust on small-N,
    and generate an 8-cell position heat-map (selection rates) for quick diagnostics.

Key assumptions
    • Input: results/df_choice.csv (8 rows per screen).
    • FEs: page (case_id) always; product FEs added only when N_cases ≥ 50.
    • Controls: row_top, col1, col2, col3 (baseline = bottom row, 4th column).
    • Optional: ln_price if present and varying.
    • Independent badge columns: frame, assurance, scarcity, strike, timer,
      social_proof, voucher, bundle (dropped if no variation).
    • Small-N: ridge-penalised IRLS with SPD guards; numeric-safe SEs.

Returns
    run_logit(...) -> pandas.DataFrame with columns:
      ['badge','beta','se','p','q_bh','odds_ratio','ci_low','ci_high',
       'ame_pp','evid_score','price_eq','sign']

Extras
    save_position_heatmap(csv_path, out_png_path, title=None) -> out_png_path
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# for heat-map
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- public labels -----------------
BADGE_LABELS = {
    "frame": "All-in framing",
    "assurance": "Assurance",
    "scarcity": "Scarcity tag",
    "strike": "Strike-through",
    "timer": "Timer",
    "social_proof": "Social proof",
    "voucher": "Voucher",
    "bundle": "Bundle",
}
POSITION_LABELS = {
    "row_top": "Row 1",
    "col1": "Column 1",
    "col2": "Column 2",
    "col3": "Column 3",
}

# ----------------- knobs -----------------
MIN_CASES = 10
PROD_FE_MIN_CASES = 50
RIDGE_ALPHA_SMALL = 5e-2
RIDGE_ALPHA_LARGE = 1e-2
MAX_ITER = 300
TOL = 1e-7

ALLOWED = ["frame","assurance","scarcity","strike","timer","social_proof","voucher","bundle"]
POS_COLS = ["row_top","col1","col2","col3"]

# ----------------- core: ridge-IRLS with FE -----------------
def _ridge_logit_irls(y: np.ndarray, X: np.ndarray, alpha: float, max_iter: int = MAX_ITER, tol: float = TOL):
    n, k = X.shape
    beta = np.zeros(k, dtype=np.float64)
    rng = np.random.default_rng(0)
    beta += rng.normal(scale=1e-6, size=k)

    I = np.eye(k, dtype=np.float64)

    for _ in range(max_iter):
        xb = np.clip(X @ beta, -35.0, 35.0)
        p  = expit(xb)
        W  = p * (1.0 - p)
        if np.max(W) < 1e-12:
            break
        z  = xb + (y - p) / np.maximum(W, 1e-12)
        Xw = X * W[:, None]
        H  = X.T @ Xw + 2.0 * alpha * I
        H += 1e-12 * I  # SPD guard

        g  = X.T @ (W * z)
        try:
            beta_new = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.pinv(H) @ g
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    xb = np.clip(X @ beta, -35.0, 35.0)
    p  = expit(xb)
    W  = p * (1.0 - p)
    Xw = X * W[:, None]
    H  = X.T @ Xw + 2.0 * alpha * I
    H += 1e-12 * I
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
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
    cols = [c for c in POS_COLS if c in df.columns]
    if "ln_price" in df.columns and _is_var(df["ln_price"]):
        cols.append("ln_price")

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

    X_main = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float64)
    d_case = pd.get_dummies(df["case_id"], drop_first=True, dtype=np.float64)

    d_prod = None
    if (n_cases >= PROD_FE_MIN_CASES) and ("title" in df.columns):
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

def _tidy(beta, cov, p_hat, cols, label_map, b_price: float | None, include_positions: bool = True):
    idx = list(cols)
    beta_s = pd.Series(beta[:len(idx)], index=idx, dtype=float)

    cov_diag = np.diag(cov).astype(np.float64)
    cov_diag[~np.isfinite(cov_diag)] = 0.0
    cov_diag = np.maximum(cov_diag, 0.0)
    se_s = pd.Series(np.sqrt(cov_diag[:len(idx)]), index=idx, dtype=float)

    keys_in_data = set(idx)
    ordered_keys = []
    if include_positions:
        ordered_keys.extend([k for k in ["row_top","col1","col2","col3"] if k in keys_in_data])
    ordered_keys.extend([k for k in ALLOWED if k in keys_in_data])

    if not ordered_keys:
        return pd.DataFrame(columns=["badge","beta","se","p","q_bh","odds_ratio","ci_low","ci_high","ame_pp","evid_score","price_eq","sign"])

    se_vec = se_s[ordered_keys].replace(0.0, np.nan)
    z = beta_s[ordered_keys] / se_vec
    pvals = np.array([1.0 if (not np.isfinite(zz)) else 2.0*(1.0 - norm.cdf(abs(zz))) for zz in z], dtype=float)

    try:
        _, q_bh, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    except Exception:
        q_bh = np.full_like(pvals, np.nan, dtype=float)

    wbar = float(np.mean(p_hat * (1.0 - p_hat))) if p_hat.size else 0.0

    out = []
    for i, k in enumerate(ordered_keys):
        b  = float(beta_s[k])
        se = float(se_s[k])
        p  = float(pvals[i])
        q  = float(q_bh[i]) if np.isfinite(q_bh[i]) else float("nan")
        orx  = math.exp(b)
        ci_l = math.exp(b - 1.96*se) if np.isfinite(se) and se > 0 else float("nan")
        ci_h = math.exp(b + 1.96*se) if np.isfinite(se) and se > 0 else float("nan")
        ame_pp = 100.0 * wbar * b
        evid = max(0.0, 1.0 - p) if np.isfinite(p) else 0.0
        price_eq = (abs(b / b_price) if (b_price is not None and abs(b_price) > 1e-12) else float("nan"))
        sign = "↑" if (p < 0.05 and b > 0) else ("↓" if (p < 0.05 and b < 0) else "0")
        label = POSITION_LABELS.get(k, BADGE_LABELS.get(k, k))
        out.append({
            "badge": label,
            "beta": b, "se": se, "p": p, "q_bh": q,
            "odds_ratio": orx, "ci_low": ci_l, "ci_high": ci_h,
            "ame_pp": ame_pp, "evid_score": evid, "price_eq": price_eq,
            "sign": sign
        })
    return pd.DataFrame(out)

# ----------------- public API -----------------
def run_logit(path_csv: str, badge_filter: list[str] | None = None):
    df = pd.read_csv(path_csv)
    df = _complete_screens(df)

    n_cases = df["case_id"].nunique()
    print(f"[logit] fit_mode = ridge_default; screens={n_cases}; rows={len(df)}", flush=True)

    for k in ["frame","assurance","scarcity","strike","timer","social_proof","voucher","bundle"]:
        if k in df.columns:
            try:
                print(f"DEBUG {k}_unique=", int(pd.to_numeric(df[k], errors="coerce").fillna(-1).nunique(dropna=False)))
            except Exception:
                pass

    X, y, cols, fe_cols = _build_design(df, badge_filter, n_cases)

    b_price_idx = cols.index("ln_price") if "ln_price" in cols else None
    alpha = RIDGE_ALPHA_LARGE if n_cases >= MIN_CASES else RIDGE_ALPHA_SMALL

    beta, cov, p_hat = _ridge_logit_irls(y, X.values, alpha=alpha)
    main_cols = len(cols)
    print(f"[logit] design: main={main_cols}; FE={fe_cols}; total_cols={X.shape[1]}", flush=True)

    b_price = float(beta[b_price_idx]) if b_price_idx is not None else None

    table = _tidy(beta, cov, p_hat, cols, {**POSITION_LABELS, **BADGE_LABELS}, b_price, include_positions=True)

    pref_cols = ["badge","beta","se","p","q_bh","odds_ratio","ci_low","ci_high","ame_pp","evid_score","price_eq","sign"]
    table = table.reindex(columns=pref_cols)
    return table

# ----------------- heat-map generator -----------------
def save_position_heatmap(path_csv: str, out_png_path: str, title: str | None = None) -> str:
    """
    Create a 2×4 heat-map of selection rates by (row, col), saving to PNG.
    Row labels: Row 1 (top), Row 2 (bottom). Columns: 1..4.
    """
    df = pd.read_csv(path_csv)

    # keep only complete 8-alt screens
    counts = df.groupby("case_id").size()
    keep = counts[counts == 8].index
    df = df[df["case_id"].isin(keep)].copy()

    # numeric guards
    df["row"] = pd.to_numeric(df["row"], errors="coerce").fillna(-1).astype(int)
    df["col"] = pd.to_numeric(df["col"], errors="coerce").fillna(-1).astype(int)
    df["chosen"] = pd.to_numeric(df["chosen"], errors="coerce").fillna(0.0).astype(float)

    # mean chosen by cell; fill missing with 0
    mat = np.zeros((2, 4), dtype=float)
    g = df.groupby(["row","col"])["chosen"].mean()
    for (r, c), v in g.items():
        if 0 <= int(r) <= 1 and 0 <= int(c) <= 3:
            mat[int(r), int(c)] = float(v)

    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=144)
    im = ax.imshow(mat, vmin=0.0, vmax=1.0)
    for r in range(2):
        for c in range(4):
            ax.text(c, r, f"{100.0*mat[r,c]:.1f}%", ha="center", va="center", fontsize=9)

    ax.set_xticks(range(4)); ax.set_xticklabels(["Col 1","Col 2","Col 3","Col 4"])
    ax.set_yticks([0,1]);    ax.set_yticklabels(["Row 1","Row 2"])
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    if title:
        ax.set_title(title, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Selection rate", rotation=270, labelpad=12)

    fig.tight_layout()
    fig.savefig(out_png_path)
    plt.close(fig)
    return out_png_path
