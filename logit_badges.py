# -*- coding: utf-8 -*-
"""
Conditional-logit with screen fixed effects + small-N ridge fallback
Version: v1.9 (2025-10-30)

Behaviour
    • Requires ≥ 10 complete screens (MIN_CASES). Below that returns an empty table.
    • Uses ridge-IRLS (α=1e-2) with screen fixed effects (one-hot on case_id, no intercept).
    • Estimates independent effects for: frame, assurance, scarcity, strike, timer,
      social_proof, voucher, bundle (no grouping).
    • Includes position dummies (row_top, col1–col3) and ln_price when available.
    • Returns a tidy table for UI consumption.

API
    run_logit(csv_path, badge_filter=None) -> pandas.DataFrame
        badge_filter: optional iterable restricting the reported badges to a subset of
        {"frame","assurance","scarcity","strike","timer","social_proof","voucher","bundle"}.
        If None, all available badges in the CSV are considered.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

MIN_CASES   = 10
RIDGE_ALPHA = 1e-2

# Canonical badge columns present in df_choice.csv
BADGE_CANON = [
    "frame", "assurance", "scarcity", "strike", "timer",
    "social_proof", "voucher", "bundle"
]
HUMAN_LABEL = {
    "frame": "All-in framing",
    "assurance": "Assurance",
    "scarcity": "Scarcity tag",
    "strike": "Strike-through",
    "timer": "Timer",
    "social_proof": "Social proof",
    "voucher": "Voucher",
    "bundle": "Bundle",
}

def _read_choice(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # Legacy support: mirror dark_* columns if the canonical ones are missing
    if "scarcity" not in df.columns and "dark_scarcity" in df.columns:
        df["scarcity"] = df["dark_scarcity"].astype(int)
    if "strike" not in df.columns and "dark_strike" in df.columns:
        df["strike"] = df["dark_strike"].astype(int)
    if "timer" not in df.columns and "dark_timer" in df.columns:
        df["timer"] = df["dark_timer"].astype(int)
    # Coerce common ints
    for c in ["row","col","row_top","col1","col2","col3","chosen"]:
        if c in df.columns:
            df[c] = df[c].astype(int, errors="ignore")
    for c in BADGE_CANON:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    return df

def _complete_screens_only(df: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in df.columns:
        raise ValueError("df_choice.csv is missing column 'case_id'.")
    sizes = df.groupby("case_id").size()
    keep = sizes[sizes == 8].index
    out = df[df["case_id"].isin(keep)].copy()
    return out

def _one_hot_case_id(df: pd.DataFrame) -> pd.DataFrame:
    # Screen FE (drop_first to avoid intercept/collinearity)
    return pd.get_dummies(df["case_id"], prefix="fe", drop_first=True)

def _ridge_logit_irls(y: np.ndarray, X: np.ndarray, alpha: float = RIDGE_ALPHA,
                      max_iter: int = 200, tol: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
    n, k = X.shape
    beta = np.zeros(k, dtype=float)
    I = np.eye(k, dtype=float)
    rng = np.random.default_rng(0)
    beta += rng.normal(scale=1e-6, size=k)
    for _ in range(max_iter):
        xb = np.clip(X @ beta, -35, 35)
        p  = expit(xb)
        W  = p * (1 - p)
        if float(np.max(W)) < 1e-12:
            break
        z  = xb + (y - p) / np.maximum(W, 1e-12)
        Xw = X * W[:, None]
        H  = X.T @ Xw + 2.0 * alpha * I
        g  = X.T @ (W * z)
        try:
            beta_new = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.pinv(H) @ g
        if np.linalg.norm(beta_new - beta, ord=np.inf) < tol:
            beta = beta_new
            break
        beta = beta_new
    # SEs from penalised Fisher info
    xb = np.clip(X @ beta, -35, 35)
    p  = expit(xb)
    W  = p * (1 - p)
    Xw = X * W[:, None]
    H  = X.T @ Xw + 2.0 * alpha * np.eye(X.shape[1], dtype=float)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return beta, se

def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(m); ranks[order] = np.arange(1, m+1)
    q = pvals * m / ranks
    q_sorted = np.minimum.accumulate(np.flip(np.sort(q)))
    q_mon = np.empty_like(q); q_mon[order] = np.flip(q_sorted)
    return np.clip(q_mon, 0, 1)

def _drop_constant_cols(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    dropped = []
    keep_cols = []
    for c in X.columns:
        v = X[c].values
        if np.allclose(v, v[0]):
            dropped.append(c)
        else:
            keep_cols.append(c)
    return X[keep_cols].copy(), dropped

def _ame_dummy(X: pd.DataFrame, beta: np.ndarray, colname: str) -> float:
    if colname not in X.columns:
        return float("nan")
    j = list(X.columns).index(colname)
    xb = X.values @ beta
    xj = X[colname].values
    bj = beta[j]
    p1 = expit(np.clip(xb + bj * (1 - xj), -35, 35))
    p0 = expit(np.clip(xb - bj * xj, -35, 35))
    return float(np.mean(p1 - p0))

def run_logit(csv_path: str, badge_filter: Iterable[str] | None = None) -> pd.DataFrame:
    df = _read_choice(csv_path)
    df = _complete_screens_only(df)
    n_cases = df["case_id"].nunique()
    print(f"[logit] screens={n_cases}; rows={len(df)}", flush=True)

    if n_cases < MIN_CASES:
        print(f"[logit] insufficient screens (<{MIN_CASES}); returning empty.", flush=True)
        return pd.DataFrame(columns=[
            "badge","beta","se","p","q_bh","odds_ratio","ci_low","ci_high","ame_pp","evid_score","price_eq","sign"
        ])

    # RHS: position dummies
    rhs_cols: List[str] = []
    for c in ["row_top","col1","col2","col3"]:
        if c in df.columns:
            rhs_cols.append(c)

    # Price block (optional)
    include_price = ("ln_price" in df.columns) and df["ln_price"].notnull().any()
    if include_price:
        rhs_cols.append("ln_price")

    # Badges to consider/report
    available = [c for c in BADGE_CANON if c in df.columns]
    if badge_filter is None:
        chosen_badges = available
    else:
        chosen_badges = [b for b in badge_filter if b in available]
    rhs_cols += chosen_badges

    # Main design + FE
    X_main = df[rhs_cols].astype(float).reset_index(drop=True)
    FE = _one_hot_case_id(df)
    X = pd.concat([X_main, FE], axis=1)

    # Drop constant columns (no variation) before fit
    X, dropped = _drop_constant_cols(X)
    if dropped:
        print(f"[logit] dropped constant columns: {dropped}", flush=True)

    y = df["chosen"].astype(int).values

    print(f"[logit] design: main={len(rhs_cols)}; FE={FE.shape[1]}; total_cols={X.shape[1]}", flush=True)

    beta, se = _ridge_logit_irls(y, X.values)

    # Collect coefficients and SEs
    idx = list(X.columns)
    coefs = dict(zip(idx, beta))
    ses   = dict(zip(idx, se))

    # Helper accessors
    def _get_b(c): return float(coefs[c]) if c in coefs else float("nan")
    def _get_se(c): return float(ses[c])   if c in ses   else float("nan")

    # p-values (normal approx)
    pvals = {}
    for c in idx:
        if _get_se(c) == 0 or math.isnan(_get_se(c)):
            pvals[c] = 1.0
        else:
            z = _get_b(c) / _get_se(c)
            pvals[c] = float(2 * (1 - norm.cdf(abs(z))))

    # Compute AMEs on the fitted X for badge dummies only
    ame_map = {}
    for b in chosen_badges:
        ame_map[b] = _ame_dummy(X, beta, b) if b in X.columns else float("nan")

    # Price-equivalent effect: |beta_badge| / |beta_lnprice| (if present and nonzero)
    b_price = abs(_get_b("ln_price")) if include_price and "ln_price" in X.columns else float("nan")

    rows = []
    for b in chosen_badges:
        lab = HUMAN_LABEL.get(b, b)
        bb = _get_b(b)
        ss = _get_se(b)
        pp = pvals.get(b, 1.0)
        orx = float(math.exp(bb)) if not math.isnan(bb) else float("nan")
        ci_l = float(math.exp(bb - 1.96 * ss)) if not (math.isnan(bb) or math.isnan(ss)) else float("nan")
        ci_h = float(math.exp(bb + 1.96 * ss)) if not (math.isnan(bb) or math.isnan(ss)) else float("nan")
        ame = float(ame_map.get(b, float("nan")))
        # convert to percentage points
        ame_pp = ame * 100.0 if not math.isnan(ame) else float("nan")
        price_eq = (abs(bb) / b_price) if (b_price and not math.isnan(b_price) and b_price > 0) else float("nan")
        sign = "↑" if (pp < 0.05 and bb > 0) else ("↓" if (pp < 0.05 and bb < 0) else "0")
        rows.append({
            "badge": lab,
            "beta": bb,
            "se": ss,
            "p": pp,
            # q-values filled later across reported badges
            "q_bh": float("nan"),
            "odds_ratio": orx,
            "ci_low": ci_l,
            "ci_high": ci_h,
            "ame_pp": ame_pp,
            "evid_score": 0.0 if pp <= 0 else float(max(0.0, -math.log10(pp))),
            "price_eq": price_eq,
            "sign": sign
        })

    out = pd.DataFrame(rows)

    # BH q-values on the p's we actually report
    if not out.empty:
        q = _bh_qvalues(out["p"].values.astype(float))
        out["q_bh"] = q

    # Sort by badge name for stable display
    out = out.sort_values("badge").reset_index(drop=True)

    # Debug uniques (helps catch “only one level” mistakes during small-N tests)
    for c in ["frame","assurance","scarcity","strike","timer","social_proof","voucher","bundle"]:
        if c in df.columns:
            try:
                u = int(df[c].nunique(dropna=False))
            except Exception:
                u = "NA"
            print(f"[logit.debug] {c}_unique={u}", flush=True)

    return out
