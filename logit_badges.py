# -*- coding: utf-8 -*-
"""
logit_badges — v1.17 (2025-11-01)

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
    • Small-N stability: ridge-IRLS with numeric guards (penalised estimator is default).
    • AME stability: for binary regressors, AME computed as average Δp of setting x_j=1 vs 0
      with all other covariates fixed; for ln_price (continuous), AME uses mean[p(1−p)]·β.

New in v1.17
    • Cluster-robust standard errors at the screen/case level are now the default
      (configurable via CLUSTER_BY_CASE). Implemented via a sandwich estimator that
      uses unpenalised score sums per case and the penalised information matrix for
      the “bread.”
    • MAX_ITER increased to 500 to match the reference paper’s optimisation cap.

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
from scipy.special import expit, logit
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# ----------------- knobs -----------------
MIN_CASES = 10
RIDGE_ALPHA_SMALL = 5e-2
RIDGE_ALPHA_LARGE = 1e-2
MAX_ITER = 500  # NEW v1.17: raised to align with reference
TOL = 1e-7
EPS = 1e-9  # prob clamp
CLUSTER_BY_CASE = True  # NEW v1.17: report case-level cluster-robust SEs by default

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

# ----------------- NEW v1.17: case-clustered robust covariance -----------------

def _cluster_cov_logit_ridge(X: np.ndarray, y: np.ndarray, beta_hat: np.ndarray, case_ids: np.ndarray, alpha: float) -> np.ndarray:
    """
    Case-level clustered (sandwich) covariance for ridge-penalised logistic regression.

    Bread: (X'WX + 2αI)^{-1}
    Meat:  Σ_c s_c s_c',  where s_i = x_i (y_i − p_i) and s_c = Σ_{i∈c} s_i

    Notes
    -----
    • Uses *unpenalised* scores in the meat (standard practice) and the penalised
      expected information in the bread to reflect the estimator actually used.
    • Returns a full k×k covariance, aligned with the full design (main + FE).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta_hat = np.asarray(beta_hat, float)

    eta = np.clip(X @ beta_hat, -35.0, 35.0)
    p = expit(eta)
    r = y - p
    U = X * r[:, None]  # per-observation score vectors (unpenalised)

    # Sum scores within cases
    g = pd.Series(case_ids).astype("category").cat.codes.to_numpy()
    G = int(g.max()) + 1
    S = np.zeros((G, X.shape[1]), dtype=float)
    for j in range(G):
        S[j] = U[g == j].sum(axis=0)

    meat = S.T @ S
    w = p * (1.0 - p)
    Xw = X * w[:, None]
    H_unpen = X.T @ Xw
    H_pen = H_unpen + 2.0 * alpha * np.eye(X.shape[1])
    bread = np.linalg.inv(H_pen)

    V = bread @ meat @ bread
    return V

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
    return X, y, cols, fe_cols, X_main.values


def _average_marginal_effects(cols: list[str],
                              beta_main: np.ndarray,
                              p_hat: np.ndarray,
                              X_main: np.ndarray) -> dict[str, float]:
    """
    Stable AME:
      • For binary regressors (0/1): average Δp of flipping x_j to 1 vs 0:
            p1_i = σ(η_i + β_j*(1 - x_ij))
            p0_i = σ(η_i - β_j*x_ij)
            AME = 100 * mean(p1_i - p0_i)
      • For ln_price (continuous): 100 * mean[p_i(1-p_i)] * β_j
    """
    p = np.clip(p_hat, EPS, 1.0 - EPS)
    eta = logit(p)

    ame = {}
    name_to_idx = {c: i for i, c in enumerate(cols)}
    for c in cols:
        j = name_to_idx[c]
        b = float(beta_main[j])
        x = X_main[:, j].astype(float)

        if c == "ln_price":
            ame[c] = 100.0 * float(np.mean(p * (1.0 - p))) * b
        else:
            # treat as binary regressor
            p1 = expit(eta + b * (1.0 - x))
            p0 = expit(eta - b * x)
            ame[c] = 100.0 * float(np.mean(p1 - p0))

        # clamp to feasible bounds
        ame[c] = float(np.clip(ame[c], -100.0, 100.0))
    return ame


def _tidy(beta, cov, p_hat, cols, b_price: float | None, ame_map: dict[str, float]):
    idx = list(cols)
    beta_s = pd.Series(beta[:len(idx)], index=idx, dtype=float)
    se_s = pd.Series(np.sqrt(np.diag(cov)[:len(idx)]), index=idx, dtype=float)

    keys_pos = [k for k in ["row_top","col1","col2","col3"] if k in idx]
    keys_attr = [k for k in ["ln_price"] if k in idx]
    keys_badge = [k for k in BADGE_KEYS if k in idx]

    def make_rows(keys, section, lab_map, allow_price_eq: bool):
        if not keys:
            return []
        z = beta_s[keys] / se_s[keys].replace(0.0, np.nan)
        pvals = 2.0 * (1.0 - norm.cdf(np.abs(z.values)))
        _, q_bh, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh") if len(keys) > 1 else ([None], [pvals[0]], None, None)

        out = []
        for i, k in enumerate(keys):
            b = float(beta_s[k]); se = float(se_s[k])
            p = float(pvals[i]); q = float(q_bh[i])
            orx = math.exp(b)
            ci_l = math.exp(b - 1.96 * se)
            ci_h = math.exp(b + 1.96 * se)
            ame_pp = float(ame_map.get(k, 0.0))
            evid = max(0.0, 1.0 - p)

            if allow_price_eq and (b_price is not None) and (abs(b_price) > 1e-12):
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
    # No price-equivalent λ for position controls
    rows += make_rows(keys_pos, "Position effects", POS_LABELS, allow_price_eq=False)
    # Price-equivalent λ only for badges/levers (not for ln(price) or positions)
    rows += make_rows(keys_badge, "Badge/lever effects", BADGE_LABELS, allow_price_eq=True)
    rows += make_rows(keys_attr, "Attribute effects", ATTR_LABELS, allow_price_eq=False)

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

    X, y, cols, fe_cols, X_main = _build_design(df, badge_filter, n_cases)

    b_price_idx = cols.index("ln_price") if "ln_price" in cols else None
    alpha = RIDGE_ALPHA_LARGE if n_cases >= MIN_CASES else RIDGE_ALPHA_SMALL

    beta, cov_naive, p_hat = _ridge_logit_irls(y, X.values, alpha=alpha)

    # NEW v1.17: replace naive covariance with case-clustered robust covariance when enabled
    cov = cov_naive
    if CLUSTER_BY_CASE:
        cov = _cluster_cov_logit_ridge(X.values, y, beta, df["case_id"].to_numpy(), alpha)

    main_cols = len(cols)
    print(f"[logit] design: main={main_cols}; FE={fe_cols}; total_cols={X.shape[1]}", flush=True)

    b_price = float(beta[b_price_idx]) if b_price_idx is not None else None

    # Stable AME map
    ame_map = _average_marginal_effects(cols, beta[:main_cols], p_hat, X_main)

    table = _tidy(beta, cov, p_hat, cols, b_price, ame_map)

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
    """
    df = pd.read_csv(path_csv)
    df = _complete_screens(df).copy()

    df["row"] = pd.to_numeric(df.get("row", 0), errors="coerce").fillna(-1).astype(int)
    df["col"] = pd.to_numeric(df.get("col", 0), errors="coerce").fillna(-1).astype(int)
    df["chosen"] = pd.to_numeric(df.get("chosen", 0), errors="coerce").fillna(0).astype(int)

    mat = np.zeros((2, 4), dtype=float)
    g = df.groupby(["row", "col"])['chosen'].mean()
    for (r, c), v in g.items():
        r = int(r); c = int(c)
        if 0 <= r <= 1 and 0 <= c <= 3:
            mat[r, c] = float(v)

    vmax_auto = float(mat.max())
    vhi = vmax_auto if vmax_auto > 0 else 1.0

    fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=144)
    im = ax.imshow(mat, cmap="Greys", vmin=vhi, vmax=0.0)

    for r in range(2):
        for c in range(4):
            val = float(mat[r, c])
            txt_color = "white" if val >= 0.5 * vhi else "black"
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
    """
    df = pd.read_csv(path_csv)
    df = _prepare_df(df)
    df = _complete_screens(df)

    n_cases = int(df["case_id"].nunique())
    X, y, cols, fe_cols, _X_main = _build_design(df, badge_filter=None, n_cases=n_cases)
    alpha = RIDGE_ALPHA_LARGE if n_cases >= MIN_CASES else RIDGE_ALPHA_SMALL
    beta, cov_naive, p_hat = _ridge_logit_irls(y, X.values, alpha=alpha)

    df = df.copy()
    df["p_hat"] = p_hat
    mat = np.zeros((2, 4), dtype=float)
    g = df.groupby(["row", "col"])['p_hat'].mean()
    for (r, c), v in g.items():
        r = int(r); c = int(c)
        if 0 <= r <= 1 and 0 <= c <= 3:
            mat[r, c] = float(v)

    fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=144)
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
