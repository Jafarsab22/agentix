# -*- coding: utf-8 -*-
"""
logit_badges — v1.13 (2025-10-30)

Purpose
    Estimate within-screen (conditional) logit effects for an 8-alternative choice
    design with page fixed effects, handling small-N runs robustly.

Key assumptions
    • Input: results/df_choice.csv produced by the runner (8 rows per screen).
    • Screen FE via case_id dummies; optional product FE when N is adequate.
    • Position controls: row_top, col1, col2, col3 (baseline = bottom row, Column 4).
    • Attribute: ln_price computed from price if present and > 0 (guarded).
    • Badges estimated independently: frame, assurance, scarcity, strike, timer,
      social_proof, voucher, bundle. Any non-varying regressor is dropped.
    • Small-N stability: below MIN_CASES (default 10) we use ridge-IRLS with page FE;
      for larger N we still default to ridge-IRLS (fast & stable). Product FE added
      only when N >= MIN_CASES.

Returns
    pandas.DataFrame with columns:
      ['section','badge','beta','se','p','q_bh','odds_ratio','ci_low','ci_high',
       'ame_pp','evid_score','price_eq','sign']
Glossary:
    β (beta): Estimated log-odds coefficient for the regressor (relative to the baseline cell, with screen/product FEs).
    SE: Standard error of β (from the penalised Fisher information when ridge-IRLS is used).
    p: Two-sided p-value for H₀: β = 0.
    q_bh: Benjamini–Hochberg FDR–adjusted p-value across the reported effects.
    Odds ratio: exp(β). For a binary lever, the multiplicative change in choice odds when it turns on (0→1); for ln(price), per one log-unit increase (≈ ×2.718 in price).
    CI low / CI high: 95% confidence interval bounds for the odds ratio.
    AME (pp): Average marginal effect on choice probability, in percentage points, averaged over observed screens.
    Evidence: 1 − p, a 0–1 summary score for visual ranking (not a substitute for p/q_bh).
    Price-eq λ: |β / β_price|, the log-price change that would have the same utility impact as the lever; to express as an equivalent % price change use exp(λ) − 1.
    Effect: Sign flag—“+” if p < 0.05 and β > 0; “−” if p < 0.05 and β < 0; “0” otherwise.
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
import matplotlib
matplotlib.use("Agg")  # safe for headless servers
import matplotlib.pyplot as plt

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
    # ensure numeric dummies exist for positions if provided as bools/strings
    for c in POS_COLS + BADGE_KEYS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    # ln(price) if feasible
    if "price" in df.columns:
        pr = pd.to_numeric(df["price"], errors="coerce")
        pr = pr.where(pr > 0)
        if pr.notna().any():
            df["ln_price"] = np.log(pr).fillna(0.0)
    return df

def _build_design(df: pd.DataFrame, badge_filter: list[str] | None, n_cases: int):
    cols = [c for c in POS_COLS if c in df.columns]
    # attribute
    if "ln_price" in df.columns and _is_var(df["ln_price"]):
        cols.append("ln_price")

    # badges
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
            # price-equivalent only for binary levers; treat ln_price as NaN
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

    table = _tidy(beta, cov, p_hat, cols, b_price)

    pref_cols = ["section","badge","beta","se","p","q_bh","odds_ratio","ci_low","ci_high","ame_pp","evid_score","price_eq","sign"]
    table = table.reindex(columns=pref_cols)

    # deterministic within-section order
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
    #im = ax.imshow(mat, vmin=0.0, vmax=1.0)
    im = ax.imshow(mat, cmap="viridis_r", vmin=0.0, vmax=1.0)

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



