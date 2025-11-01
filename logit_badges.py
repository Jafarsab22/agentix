# -*- coding: utf-8 -*-
"""
logit_badges — v1.15 (2025-10-31)

Purpose
    Estimate within-screen (conditional) logit effects for an 8-alternative choice
    design with page AND product fixed effects, handling small-N runs robustly.

Key assumptions
    • Input: results/df_choice.csv produced by the runner (8 rows per screen).
    • Screen FE via case_id dummies; product FE via title dummies (always included if present).
    • Position controls: row_top, col1, col2, col3 (baseline = bottom row, Column 4).
    • Attribute: ln_price computed from price if present and > 0 (guarded).
    • Badges estimated independently: frame, assurance, scarcity, strike, timer,
      social_proof, voucher, bundle. Any non-varying regressor is dropped.
    • Small-N stability: ridge-IRLS with numeric guards; no MLE branch (penalised estimator is default).

Returns
    pandas.DataFrame with columns:
      ['section','badge','beta','se','p','q_bh','odds_ratio','ci_low','ci_high',
       'ame_pp','evid_score','price_eq','sign']
"""

from __future__ import annotations

import math
import time
import pathlib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# ----------------- knobs -----------------
MIN_CASES = 10
RIDGE_ALPHA_SMALL = 5e-2
RIDGE_ALPHA_LARGE = 1e-2
MAX_ITER = 300
TOL = 1e-7

BADGE_KEYS = ["frame","assurance","scarcity","strike","timer","social_proof","voucher","bundle"]
POS_COLS = ["row_top","col1","col2","col3"]

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
POS_LABELS = {
    "row_top": "Row 1",
    "col1": "Column 1",
    "col2": "Column 2",
    "col3": "Column 3",
}
ATTR_LABELS = {"ln_price": "ln(price)"}

# ----------------- core: ridge-IRLS with FE -----------------
def _ridge_logit_irls(y: np.ndarray, X: np.ndarray, alpha: float, max_iter: int = MAX_ITER, tol: float = TOL):
    n, k = X.shape
    beta = np.zeros(k, dtype=np.float64)
    rng = np.random.default_rng(0)
    beta += rng.normal(scale=1e-6, size=k)
    I = np.eye(k, dtype=np.float64)

    for _ in range(max_iter):
        xb = np.clip(X @ beta, -35.0, 35.0)
        p = expit(xb)
        W = p * (1.0 - p)
        if np.max(W) < 1e-12:
            break
        z = xb + (y - p) / np.maximum(W, 1e-12)
        Xw = X * W[:, None]
        H = X.T @ Xw + 2.0 * alpha * I
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

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in POS_COLS + BADGE_KEYS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    if "price" in df.columns:
        pr = pd.to_numeric(df["price"], errors="coerce")
        pr = pr.where(pr > 0)
        if pr.notna().any():
            df["ln_price"] = np.log(pr).fillna(0.0)

    if "title" in df.columns:
        df["title"] = df["title"].astype(str).fillna("")
    return df

def _build_design(df: pd.DataFrame, badge_filter: list[str] | None, n_cases: int):
    cols = [c for c in POS_COLS if c in df.columns]
    if "ln_price" in df.columns and _is_var(df["ln_price"]):
        cols.append("ln_price")

    targets = BADGE_KEYS if not badge_filter else [b for b in badge_filter if b in BADGE_KEYS]
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
    if "title" in df.columns:
        d_prod = pd.get_dummies(df["title"], drop_first=True, dtype=np.float64)
        X = pd.concat([X_main, d_case, d_prod], axis=1)
        fe_cols = d_case.shape[1] + d_prod.shape[1]
        print(f"[logit] product FEs included: {d_prod.shape[1]} dummies", flush=True)
    else:
        X = pd.concat([X_main, d_case], axis=1)
        fe_cols = d_case.shape[1]
        print("[logit] product FEs skipped (no 'title' column)", flush=True)

    X = X.astype(np.float64)
    y = pd.to_numeric(df["chosen"], errors="coerce").fillna(0.0).astype(np.float64).values
    return X, y, cols, fe_cols

def _tidy(beta, cov, p_hat, cols, b_price: float | None):
    idx = list(cols)
    beta_s = pd.Series(beta[:len(idx)], index=idx, dtype=float)
    se_s = pd.Series(np.sqrt(np.diag(cov)[:len(idx)]), index=idx, dtype=float)

    keys_pos = [k for k in ["row_top","col1","col2","col3"] if k in idx]
    keys_attr = [k for k in ["ln_price"] if k in idx]
    keys_badge = [k for k in BADGE_KEYS if k in idx]

    def make_rows(keys, section, lab_map):
        if not keys:
            return []
        z = beta_s[keys] / se_s[keys].replace(0.0, np.nan)
        pvals = 2.0 * (1.0 - norm.cdf(np.abs(z.values)))
        _, q_bh, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh") if len(keys) > 1 else ([None], [pvals[0]], None, None)

        wbar = float(np.mean(p_hat * (1.0 - p_hat))) if p_hat.size else 0.0
        out = []
        for i, k in enumerate(keys):
            b = float(beta_s[k]); se = float(se_s[k])
            p = float(pvals[i]); q = float(q_bh[i])
            orx = math.exp(b)
            ci_l = math.exp(b - 1.96 * se)
            ci_h = math.exp(b + 1.96 * se)
            ame_pp = 100.0 * wbar * b
            evid = max(0.0, 1.0 - p)
            if (b_price is not None) and (abs(b_price) > 1e-12) and (k != "ln_price"):
                price_eq = abs(b / b_price)
            else:
                price_eq = float("nan")
            sign = "+" if (p < 0.05 and b > 0) else ("-" if (p < 0.05 and b < 0) else "0")
            out.append({
                "section": section,
                "badge": lab_map.get(k, k),
                "beta": b, "se": se, "p": p, "q_bh": q,
                "odds_ratio": orx, "ci_low": ci_l, "ci_high": ci_h,
                "ame_pp": ame_pp, "evid_score": evid, "price_eq": price_eq,
                "sign": sign
            })
        return out

    rows = []
    rows += make_rows(keys_pos, "Position effects", POS_LABELS)
    rows += make_rows(keys_badge, "Badge/lever effects", BADGE_LABELS)
    rows += make_rows(keys_attr, "Attribute effects", ATTR_LABELS)

    return pd.DataFrame(rows)

# ----------------- public API -----------------
def run_logit(path_csv: str, badge_filter: list[str] | None = None):
    df = pd.read_csv(path_csv)
    df = _prepare_df(df)
    df = _complete_screens(df)

    n_cases = df["case_id"].nunique()
    print(f"[logit] fit_mode = ridge_default; screens={n_cases}; rows={len(df)}", flush=True)

    for k in BADGE_KEYS:
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

    table = _tidy(beta, cov, p_hat, cols, b_price)

    pref_cols = ["section","badge","beta","se","p","q_bh","odds_ratio","ci_low","ci_high","ame_pp","evid_score","price_eq","sign"]
    table = table.reindex(columns=pref_cols)

    order = {
        "Position effects": ["Row 1","Column 1","Column 2","Column 3"],
        "Badge/lever effects": [BADGE_LABELS[k] for k in BADGE_KEYS if k in cols],
        "Attribute effects": ["ln(price)"],
    }
    sorter = []
    for sec in ["Position effects","Badge/lever effects","Attribute effects"]:
        for lab in order[sec]:
            sorter.append((sec, lab))
    idx_map = {t:i for i,t in enumerate(sorter)}
    table["__ord"] = table[["section","badge"]].apply(tuple, axis=1).map(idx_map).fillna(1e9)
    table = table.sort_values(["__ord"]).drop(columns=["__ord"]).reset_index(drop=True)

    return table

# ----------------- heatmap back-end config -----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- empirical heat-map (darker = higher selection) -----------------
def save_position_heatmap_empirical(path_csv: str,
                                    out_png_path: str,
                                    title: str | None = None) -> str:
    """
    Render a 2×4 heat-map of EMPIRICAL selection rates by (row, col) using df_choice.csv.
    We average the binary 'chosen' over all rows in each (row, col) cell.

    Conventions:
      • Rows: Row 1 = top (index 0), Row 2 = bottom (index 1).
      • Columns: 1..4 correspond to indices 0..3.
      • Darker shades indicate a higher observed selection rate.
    """
    df = pd.read_csv(path_csv)
    df = _complete_screens(df).copy()

    df["row"] = pd.to_numeric(df.get("row", 0), errors="coerce").fillna(-1).astype(int)
    df["col"] = pd.to_numeric(df.get("col", 0), errors="coerce").fillna(-1).astype(int)
    df["chosen"] = pd.to_numeric(df.get("chosen", 0), errors="coerce").fillna(0).astype(int)

    mat = np.zeros((2, 4), dtype=float)
    g = df.groupby(["row", "col"])["chosen"].mean()
    for (r, c), v in g.items():
        r = int(r); c = int(c)
        if 0 <= r <= 1 and 0 <= c <= 3:
            mat[r, c] = float(v)

    vmax_auto = float(mat.max())
    vhi = vmax_auto if vmax_auto > 0 else 1.0
    vlo = 0.0

    fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=144)
    # Use non-reversed 'Greys' but invert the normalization explicitly so higher -> darker
    im = ax.imshow(mat, cmap="Greys", vmin=vhi, vmax=vlo)

    for r in range(2):
        for c in range(4):
            val = float(mat[r, c])
            thresh = 0.5 * vhi
            txt_color = "white" if val >= thresh else "black"
            ax.text(c, r, f"{100.0 * val:.1f}%", ha="center", va="center", fontsize=9, color=txt_color)

    ax.set_xticks(range(4)); ax.set_xticklabels(["Col 1", "Col 2", "Col 3", "Col 4"])
    ax.set_yticks([0, 1]);    ax.set_yticklabels(["Row 1", "Row 2"])
    ax.set_xlabel("Column");  ax.set_ylabel("Row")
    ax.set_title(title or "Empirical webpage heatmap of AI shopping agents", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Empirical selection rate", rotation=270, labelpad=12)

    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    pathlib.Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)
    return out_png_path

# ----------------- probability-based heat-map (darker = higher selection) -----------------
def save_position_heatmap(path_csv: str,
                          out_png_path: str,
                          title: str | None = None) -> str:
    """
    Render a 2×4 heat-map of MODEL-IMPLIED selection probabilities by (row, col).
    Steps: rebuild design X with the same helpers used in run_logit(), fit ridge logit,
    compute p̂ for every alternative, then average p̂ within each grid cell.
    Darker shades indicate higher predicted selection.

    Rows: Row 1 (top, index 0), Row 2 (bottom, index 1).
    Columns: 1..4 (indices 0..3).
    """
    df = pd.read_csv(path_csv)
    df = _prepare_df(df)
    df = _complete_screens(df)

    n_cases = int(df["case_id"].nunique())
    X, y, cols, fe_cols = _build_design(df, badge_filter=None, n_cases=n_cases)
    alpha = RIDGE_ALPHA_LARGE if n_cases >= MIN_CASES else RIDGE_ALPHA_SMALL
    beta, cov, p_hat = _ridge_logit_irls(y, X.values, alpha=alpha)

    df = df.copy()
    df["p_hat"] = p_hat
    mat = np.zeros((2, 4), dtype=float)
    g = df.groupby(["row", "col"])["p_hat"].mean()
    for (r, c), v in g.items():
        r = int(r); c = int(c)
        if 0 <= r <= 1 and 0 <= c <= 3:
            mat[r, c] = float(v)

    fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=144)
    # Explicit inverted normalization so higher probability -> darker shade
    im = ax.imshow(mat, cmap="Greys", vmin=1.0, vmax=0.0)

    for r in range(2):
        for c in range(4):
            ax.text(c, r, f"{100.0 * mat[r, c]:.1f}%", ha="center", va="center", fontsize=9)

    ax.set_xticks(range(4)); ax.set_xticklabels(["Col 1", "Col 2", "Col 3", "Col 4"])
    ax.set_yticks([0, 1]);    ax.set_yticklabels(["Row 1", "Row 2"])
    ax.set_xlabel("Column");  ax.set_ylabel("Row")
    ax.set_title(title or "Probability webpage heatmap of AI shopping agents", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Model-implied selection probability", rotation=270, labelpad=12)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    pathlib.Path(out_png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)
    return out_png_path

# --------- convenience wrapper so logit_badges owns heatmap creation ---------
def generate_heatmaps(path_csv: str,
                      out_dir: str = "results",
                      title_prefix: str | None = None,
                      file_tag: str | None = None) -> dict:
    """
    Create both heatmaps and return their file paths in a dict.
    Keeps generation inside logit_badges while letting callers delegate.

    Returns
    -------
    {
      "position_heatmap_empirical": "<path>",
      "position_heatmap_prob": "<path>",
      "position_heatmap": "<path>",           # alias (probability)
      "position_heatmap_png": "<path>"        # alias (probability)
    }
    """
    od = pathlib.Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)

    tag = file_tag or str(int(time.time()))
    emp_path  = od / f"heatmap_empirical_{tag}.png"
    prob_path = od / f"heatmap_probability_{tag}.png"

    emp_title  = f"{title_prefix} — empirical"    if title_prefix else None
    prob_title = f"{title_prefix} — probability"  if title_prefix else None

    emp_p = save_position_heatmap_empirical(path_csv, str(emp_path), title=emp_title)
    prob_p = save_position_heatmap(path_csv, str(prob_path), title=prob_title)

    return {
        "position_heatmap_empirical": emp_p,
        "position_heatmap_prob": prob_p,
        "position_heatmap": prob_p,
        "position_heatmap_png": prob_p,
    }
