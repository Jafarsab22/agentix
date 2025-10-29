#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================
# CONDITIONAL (FIXED-EFFECTS) LOGIT FOR AI-AGENT READINESS SCORING
# ================================================================
# Purpose
#   Estimate within-screen effects of e-commerce levers (badges, position, price,
#   rating, reviews) on AI agents’ choices, producing a defensible table for
#   readiness scoring and site advice.
#
# Identification & assumptions
#   • One choice per screen (choice set). Outcome chosen ∈ {0,1}.
#   • Absorb screen fixed effects (controls unobserved context) and product fixed
#     effects (controls latent quality). This approximates conditional logit under
#     one-choice-per-set and delivers the same β for within-screen covariates.
#   • Covariates vary within screens (position dummies, ln_price, rating, ln_reviews,
#     badges). SEs are cluster-robust at the screen level.
#
# Variables expected (column names; additional columns are ignored safely)
#   Required:  screen_id, product, chosen, price
#   Optional:  row, col, rating, reviews, model, badge_* (any number of badge_… columns)
#     - row is 0 for top row, 1 for bottom row (paper style).
#     - col is 0..3 for left→right (paper style). Baseline is bottom row and col4 (index 3).
#
# Reported statistics (for each coefficient)
#   beta        : log-odds coefficient. >0 raises the odds of being chosen.
#   se          : cluster-robust standard error (cluster = screen_id).
#   p           : two-sided p-value for H0: beta = 0.
#   q_bh        : Benjamini–Hochberg FDR-adjusted p-value across all reported terms.
#   odds_ratio  : exp(beta); multiplicative effect on odds.
#   ci_low/high : 95% CI bounds for odds_ratio.
#   ame_pp      : average marginal effect on probability scale (percentage points).
#                 Approximated as 100 * mean_i[p_i(1−p_i)] * beta.
#   evid_score  : |beta| / se, a simple signal-to-noise index.
#
# ================================================================
# Schema (from your choice file, fixed):
#   case_id, run_id, set_id, model, category, title, row, col,
#   row_top, col1, col2, col3,
#   frame, assurance, scarcity, strike, timer, social_proof, voucher, bundle,
#   chosen, price, ln_price
#
# We map only: case_id -> screen_id, title -> product. No other aliasing.
#
# Outputs:
#   • CSV (CLI): logit_readiness_results.csv
#   • App API: run_logit(...) returns list[dict] rows for the “Badge Effects” table
#
# Statistics reported per parameter:
#   beta        : log-odds coefficient
#   se          : cluster-robust standard error (cluster = screen_id)
#   p           : two-sided p-value
#   q_bh        : Benjamini–Hochberg FDR-adjusted p-value
#   odds_ratio  : exp(beta)
#   ci_low/high : 95% CI bounds on odds_ratio
#   ame_pp      : average marginal effect (percentage points)
#   evid_score  : |beta| / se
#
# Position effects: uses your row_top, col1–col3 if present; derives from row/col if needed.
# Price: uses ln_price if present; otherwise computes ln(price).
# Badges: frame, assurance, scarcity, strike, timer, social_proof, voucher, bundle
# ================================================================

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


BADGE_VARS: List[str] = [
    "frame", "assurance", "scarcity", "strike",
    "timer", "social_proof", "voucher", "bundle"
]

# -----------------------
# Robust I/O loader
# -----------------------

def _load_df(df_or_path: Union[pd.DataFrame, str, Dict[str, Any], bytes, bytearray, Any]) -> pd.DataFrame:
    """Accept DataFrame, string/Path, dict payloads with choice_path, file-like, or bytes."""
    import pathlib, io

    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()

    if isinstance(df_or_path, (str, pathlib.Path)):
        p = pathlib.Path(str(df_or_path))
        if not p.exists():
            raise FileNotFoundError(f"Choice file not found: {p}")
        suf = p.suffix.lower()
        if suf in {".csv", ".txt"}:
            return pd.read_csv(p)
        if suf == ".parquet":
            return pd.read_parquet(p)
        if suf == ".json":
            return pd.read_json(p)
        return pd.read_csv(p)

    if isinstance(df_or_path, dict):
        # direct keys
        for k in ("choice_path", "path", "file", "csv"):
            v = df_or_path.get(k)
            if isinstance(v, (str, pathlib.Path)):
                return _load_df(v)
        # nested
        for k in ("paths", "files", "data", "payload"):
            sub = df_or_path.get(k)
            if isinstance(sub, dict):
                for kk in ("choice", "choices", "path", "file", "csv"):
                    vv = sub.get(kk)
                    if isinstance(vv, (str, pathlib.Path)):
                        return _load_df(vv)
        # last resort: interpret as column->list mapping
        try:
            return pd.DataFrame(df_or_path)
        except Exception:
            pass

    if hasattr(df_or_path, "read"):
        try:
            return pd.read_csv(df_or_path)
        except Exception:
            try:
                df_or_path.seek(0)
            except Exception:
                pass
            return pd.read_json(df_or_path)

    if isinstance(df_or_path, (bytes, bytearray)):
        bio = io.BytesIO(df_or_path)
        try:
            return pd.read_csv(bio)
        except Exception:
            bio.seek(0)
            return pd.read_json(bio)

    raise TypeError("run_logit could not find a choice file: pass a DataFrame, a file path, or a payload dict containing it.")

# -----------------------
# Schema + feature prep
# -----------------------

def _rename_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map your confirmed headers to internal names."""
    return df.rename(columns={"case_id": "screen_id", "title": "product"})

def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = ["screen_id", "product", "chosen", "price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

def _make_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Use existing row_top, col1–col3 if present; else derive from row/col (baseline: bottom row, 4th col)."""
    df = df.copy()
    if "row_top" not in df.columns and "row" in df.columns:
        df["row_top"] = (pd.to_numeric(df["row"], errors="coerce").fillna(1).astype(int) == 0).astype(int)
    if not all(c in df.columns for c in ["col1", "col2", "col3"]) and "col" in df.columns:
        ci = pd.to_numeric(df["col"], errors="coerce").fillna(3).astype(int)
        if "col1" not in df.columns: df["col1"] = (ci == 0).astype(int)
        if "col2" not in df.columns: df["col2"] = (ci == 1).astype(int)
        if "col3" not in df.columns: df["col3"] = (ci == 2).astype(int)
    return df

def _make_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ln_price exists; keep file-supplied ln_price if already present."""
    df = df.copy()
    price_num = pd.to_numeric(df["price"], errors="coerce")
    if (price_num <= 0).any():
        raise ValueError("Found non-positive prices; ln(price) undefined.")
    if "ln_price" not in df.columns:
        df["ln_price"] = np.log(price_num)
    return df

def _collect_badge_columns(df: pd.DataFrame) -> List[str]:
    """Return badge columns that exist and vary (at least 2 distinct values)."""
    return [c for c in BADGE_VARS if c in df.columns and df[c].nunique(dropna=True) > 1]

# -----------------------
# Matrix sanitisation
# -----------------------

def _drop_constant_cols(X: pd.DataFrame) -> pd.DataFrame:
    nun = X.nunique(dropna=False)
    keep = nun[nun > 1].index.tolist()
    return X[keep].copy()

def _coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.apply(pd.to_numeric, errors="coerce")
    Xn = Xn.fillna(0.0)
    return Xn.astype("float64")

# -----------------------
# Design builders
# -----------------------

def _build_design(df: pd.DataFrame, include_product_fe: bool = True) -> Tuple[pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Build y, X with absorbed fixed effects for screen_id and optionally product.
    Returns: y, X (float64), clusters (screen_id), list_of_predictor_names_in_order
    """
    df = df.copy()

    # levers: position first, then ln_price
    lever_cols: List[str] = []
    for c in ["row_top", "col1", "col2", "col3"]:
        if c in df.columns:
            lever_cols.append(c)
    if "ln_price" in df.columns:
        lever_cols.append("ln_price")

    # optional attributes if present (kept minimal)
    for opt in ["rating", "ln_reviews"]:
        if opt in df.columns:
            lever_cols.append(opt)

    badge_cols = _collect_badge_columns(df)

    # model dummies (no interactions)
    model_cols: List[str] = []
    if "model" in df.columns:
        dummies = pd.get_dummies(df["model"], prefix="model", drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        model_cols = list(dummies.columns)

    clusters = df["screen_id"].astype(str)

    # absorb FE via dummies (no intercept)
    dmy_cols = ["screen_id"]
    if include_product_fe:
        dmy_cols.append("product")
    df_fe = pd.get_dummies(df, columns=dmy_cols, drop_first=True)

    x_vars = lever_cols + badge_cols + model_cols
    fe_cols = []
    if include_product_fe:
        fe_cols = [c for c in df_fe.columns if c.startswith("screen_id_") or c.startswith("product_")]
    else:
        fe_cols = [c for c in df_fe.columns if c.startswith("screen_id_")]

    x_vars_existing = [c for c in x_vars if c in df_fe.columns]
    X = df_fe[x_vars_existing + fe_cols]

    # sanitise X
    X = _drop_constant_cols(X)
    X = _coerce_numeric(X)

    y = pd.to_numeric(df_fe["chosen"], errors="coerce").fillna(0).astype(int)

    return y, X, clusters, x_vars_existing

# -----------------------
# Estimation
# -----------------------

def _fit_glm_logit(y, X, clusters):
    """GLM(Logit) with cluster-robust SEs; returns result object."""
    model = sm.GLM(y, X, family=sm.families.Binomial())
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        res = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
    return res

def _fit_ridge_logit(y, X, alpha=1.0):
    """
    Ridge-penalised logit fallback (no clusters).
    Returns params and approximate covariance from observed Hessian at the solution.
    """
    logit = sm.Logit(y, X)
    res = logit.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=500, disp=False)
    params = res.params.copy()

    # Approximate covariance via (X' W X)^(-1) with W = p*(1-p) at fitted p
    p = 1.0 / (1.0 + np.exp(-np.asarray(X @ params)))
    W = p * (1.0 - p)
    # add small ridge on the Hessian inversion for numerical stability
    Xw = X.values * W[:, None]
    XtWX = X.T.values @ Xw
    ridge_eps = 1e-6
    try:
        cov = np.linalg.inv(XtWX + ridge_eps * np.eye(XtWX.shape[0]))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(XtWX + ridge_eps * np.eye(XtWX.shape[0]))
    se = pd.Series(np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=np.inf)), index=X.columns)
    return params, se

def _tidy_from_params(params, se, X):
    """Build tidy table given params and se; compute p, q_bh, OR, CI, AME, evid."""
    params = pd.Series(params, index=X.columns) if not isinstance(params, pd.Series) else params
    se = pd.Series(se, index=X.columns) if not isinstance(se, pd.Series) else se

    z = params / se.replace(0.0, np.nan)
    pvals = 2.0 * (1.0 - sm.stats.norm.cdf(np.abs(z)))
    # FDR
    _, q_bh, _, _ = multipletests(pvals.values, method="fdr_bh")
    q_bh = pd.Series(q_bh, index=params.index)

    # safe odds ratios with clipping to avoid overflow
    CLIP = 40.0  # exp(±40) ~ 2.35e17, beyond this we call it Inf/0 for display
    pruned = params.clip(lower=-CLIP, upper=CLIP)
    orx = np.exp(pruned)
    ci_low = np.exp(pruned - 1.96 * se)
    ci_high = np.exp(pruned + 1.96 * se)

    # AME (approx) using average p*(1-p) at midpoint (0.25) if not available here
    # We recompute weights using the canonical logit weight ~ mean 0.25 as neutral fallback
    weight_mean = 0.25
    ame_pp = 100.0 * weight_mean * params

    evid = (params.abs() / se.replace(0.0, np.nan))

    out = pd.DataFrame({
        "beta": params,
        "se": se,
        "p": pvals,
        "q_bh": q_bh,
        "odds_ratio": orx,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ame_pp": ame_pp,
        "evid_score": evid
    })
    return out

def fit_logit_and_tidy(y: pd.Series, X: pd.DataFrame, clusters: pd.Series, allow_product_fe: bool = True) -> pd.DataFrame:
    """
    Try cluster-robust GLM with screen+product FE. If separation/near-saturation:
      1) retry without product FE
      2) ridge-penalised logit with approximate SEs (no clusters)
    """
    # Attempt 1: full FE
    try:
        res = _fit_glm_logit(y, X, clusters)
        params = res.params.copy()
        bse = res.bse.copy()
        pvals = res.pvalues.copy()
        _, q_bh, _, _ = multipletests(pvals.values, method="fdr_bh")
        q_bh = pd.Series(q_bh, index=pvals.index)

        # OR with clipping
        CLIP = 40.0
        pruned = params.clip(lower=-CLIP, upper=CLIP)
        odds_ratio = np.exp(pruned)
        ci_low = np.exp(pruned - 1.96 * bse)
        ci_high = np.exp(pruned + 1.96 * bse)

        p_hat = res.predict(X)
        weight_mean = float(np.mean(p_hat * (1.0 - p_hat)))
        ame_pp = 100.0 * weight_mean * params

        evid_score = (params.abs() / bse.replace(0.0, np.nan))

        out = pd.DataFrame({
            "beta": params,
            "se": bse,
            "p": pvals,
            "q_bh": q_bh,
            "odds_ratio": odds_ratio,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ame_pp": ame_pp,
            "evid_score": evid_score
        })
        return out
    except Exception as e1:
        # fall through to reduced FE or ridge
        pass

    # Attempt 2: drop product FE if present and try again
    if allow_product_fe:
        try:
            # rebuild without product FE
            # Note: we assume the caller can rebuild; here we just signal by raising
            raise RuntimeError("RETRY_NO_PRODUCT_FE")
        except RuntimeError:
            pass

    # Attempt 3: ridge-penalised fallback
    params, se = _fit_ridge_logit(y, X, alpha=1.0)
    out = _tidy_from_params(params, se, X)
    return out

# -----------------------
# Results shaping for app
# -----------------------

def _results_to_rows(results_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert tidy results (index = parameter) into the app's list-of-dicts for badges only."""
    rows: List[Dict[str, Any]] = []
    idx = list(results_df.index.astype(str))

    b_price = None
    if "ln_price" in idx and "beta" in results_df.columns and pd.notna(results_df.loc["ln_price", "beta"]):
        try:
            b_price = float(results_df.loc["ln_price", "beta"])
        except Exception:
            b_price = None

    keep = [p for p in BADGE_VARS if p in results_df.index]
    for p in keep:
        r = results_df.loc[p]
        beta = float(r.get("beta", np.nan)) if pd.notna(r.get("beta", np.nan)) else np.nan
        se = float(r.get("se", np.nan)) if "se" in r else np.nan
        pval = float(r.get("p", np.nan)) if "p" in r else np.nan
        q_bh = float(r.get("q_bh", np.nan)) if "q_bh" in r else np.nan
        orx = float(r.get("odds_ratio", np.nan)) if "odds_ratio" in r else np.nan
        ci_l = float(r.get("ci_low", np.nan)) if "ci_low" in r else np.nan
        ci_h = float(r.get("ci_high", np.nan)) if "ci_high" in r else np.nan
        ame = float(r.get("ame_pp", np.nan)) if "ame_pp" in r else np.nan
        evid = float(r.get("evid_score", np.nan)) if "evid_score" in r else np.nan

        price_eq = np.nan
        if b_price is not None and np.isfinite(b_price) and b_price != 0.0 and np.isfinite(beta):
            try:
                price_eq = float(np.exp(-beta / b_price))
            except Exception:
                price_eq = np.nan

        sign = "0"
        if np.isfinite(pval) and pval < 0.05 and np.isfinite(beta):
            sign = "+" if beta > 0 else ("-" if beta < 0 else "0")

        rows.append({
            "badge": p,
            "beta": beta,
            "se": se,
            "p": pval,
            "q_bh": q_bh,
            "odds_ratio": orx,
            "ci_low": ci_l,
            "ci_high": ci_h,
            "ame_pp": ame,
            "evid_score": evid,
            "price_eq": price_eq,
            "sign": sign
        })

    rows.sort(key=lambda r: str(r.get("badge", "")))
    return rows

# -----------------------
# Public API for the app
# -----------------------

def run_logit(df_or_path: Union[pd.DataFrame, str, Dict[str, Any], bytes, bytearray, Any],
              selected_badges: List[str] | None = None,
              min_cases: int = 2,
              use_price: bool = True) -> List[Dict[str, Any]]:
    """App-facing API. Returns list[dict] for the 'Badge Effects' table."""
    # Load + schema
    df = _load_df(df_or_path)
    df = _rename_core_columns(df)
    _ensure_required_columns(df)

    # Basic cleaning and typing
    df = df.dropna(subset=["chosen", "screen_id", "product", "price"]).copy()
    df["chosen"] = pd.to_numeric(df["chosen"], errors="coerce").fillna(0).astype(int)

    # Derive position/log vars if needed
    df = _make_position_dummies(df)
    df = _make_logs(df)

    # Guardrail for tiny pilots
    n_screens = df["screen_id"].nunique()
    if n_screens < min_cases:
        return []

    # Build design with product FE first; if near-saturated, we’ll drop them below
    y, X_full, clusters, _xvars = _build_design(df, include_product_fe=True)

    # Heuristic near-saturation check
    near_sat = (X_full.shape[1] >= max(df.shape[0] - 2, 1))
    if near_sat:
        y, X, clusters, _ = _build_design(df, include_product_fe=False)
    else:
        X = X_full

    # Fit with robust fallbacks
    try:
        results = fit_logit_and_tidy(y, X, clusters, allow_product_fe=not near_sat)
    except RuntimeError as e:
        # Retry without product FE if requested via signal
        if str(e).startswith("RETRY_NO_PRODUCT_FE"):
            y, X, clusters, _ = _build_design(df, include_product_fe=False)
            try:
                results = fit_logit_and_tidy(y, X, clusters, allow_product_fe=False)
            except Exception:
                # Final ridge fallback
                params, se = _fit_ridge_logit(y, X, alpha=1.0)
                results = _tidy_from_params(params, se, X)
        else:
            # Final ridge fallback
            params, se = _fit_ridge_logit(y, X, alpha=1.0)
            results = _tidy_from_params(params, se, X)
    except Exception:
        # Final ridge fallback
        params, se = _fit_ridge_logit(y, X, alpha=1.0)
        results = _tidy_from_params(params, se, X)

    rows = _results_to_rows(results)

    # Optional filter by a subset of badges
    if selected_badges:
        want = set(s.strip().lower() for s in selected_badges)
        rows = [r for r in rows if str(r["badge"]).strip().lower() in want]

    return rows

# -----------------------
# CLI entrypoint
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Conditional (FE) logit for AI-agent readiness.")
    parser.add_argument("--input", type=str, default="ai_agent_choices.csv", help="Input CSV with choices.")
    parser.add_argument("--output", type=str, default="logit_readiness_results.csv", help="Output CSV for tidy results.")
    parser.add_argument("--min_screens", type=int, default=2, help="Minimum number of screens required to proceed.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = _rename_core_columns(df)
    _ensure_required_columns(df)

    df = df.dropna(subset=["chosen", "screen_id", "product", "price"]).copy()
    df["chosen"] = pd.to_numeric(df["chosen"], errors="coerce").fillna(0).astype(int)

    df = _make_position_dummies(df)
    df = _make_logs(df)

    n_screens = df["screen_id"].nunique()
    if n_screens < args.min_screens:
        raise ValueError(f"Insufficient screens for inference: {n_screens} < {args.min_screens}")

    # Primary build
    y, X_full, clusters, _ = _build_design(df, include_product_fe=True)
    near_sat = (X_full.shape[1] >= max(df.shape[0] - 2, 1))
    if near_sat:
        y, X, clusters, _ = _build_design(df, include_product_fe=False)
    else:
        X = X_full

    try:
        results = fit_logit_and_tidy(y, X, clusters, allow_product_fe=not near_sat)
    except Exception:
        params, se = _fit_ridge_logit(y, X, alpha=1.0)
        results = _tidy_from_params(params, se, X)

    results.round(6).to_csv(args.output, index=True)

    # Console summary for common levers
    show_keys = []
    for v in ["row_top", "col1", "col2", "col3", "ln_price"] + BADGE_VARS:
        if v in results.index:
            show_keys.append(v)

    print("Model fitted and results written to", args.output)
    if show_keys:
        cols = ["beta", "se", "p", "q_bh", "odds_ratio", "ci_low", "ci_high", "ame_pp", "evid_score"]
        safe_cols = [c for c in cols if c in results.columns]
        print("=== Key Lever Effects (position, price, badges) ===")
        print(results.loc[show_keys, safe_cols].to_string())

if __name__ == "__main__":
    main()







