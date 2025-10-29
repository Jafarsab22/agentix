#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================
# CONDITIONAL (FIXED-EFFECTS) LOGIT FOR AI-AGENT READINESS SCORING
# ================================================================
# Purpose
#   Estimate within-screen effects of e-commerce levers (badges, position, price)
#   on AI agents’ choices, returning a tidy table for the UI.
#
# Identification & assumptions
#   • One choice per screen (choice set). chosen ∈ {0,1}.
#   • Absorb screen fixed effects (and optionally product fixed effects).
#   • Covariates vary within screens (position dummies, ln_price, badges).
#   • SEs: cluster-robust by screen when clusters are sufficient; otherwise HC1.
#
# Input schema (confirmed)
#   case_id, run_id, set_id, model, category, title, row, col,
#   row_top, col1, col2, col3,
#   frame, assurance, scarcity, strike, timer, social_proof, voucher, bundle,
#   chosen, price, ln_price
#
# Column mapping used internally
#   case_id → screen_id
#   title   → product
#
# Outputs
#   • CLI: writes logit_readiness_results.csv
#   • API: run_logit(...) → list[dict] rows for the “Badge Effects” UI table
#
# Estimation strategy (robust to tiny pilots and separation)
#   1) Build X with screen FE (+ product FE unless near-saturated).
#   2) Fit GLM(Logit).
#      - If number of screens (clusters) ≥ 30 → cluster-robust covariance.
#      - Else → HC1 sandwich covariance (non-cluster) to avoid NaNs.
#   3) If GLM fails or is numerically unstable, fall back to ridge-penalised logit
#      and approximate SEs from the Hessian.
# ================================================================

from __future__ import annotations

import argparse
import io
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


# Badge variables in your data (keys must match column names)
BADGE_VARS: List[str] = [
    "frame", "assurance", "scarcity", "strike",
    "timer", "social_proof", "voucher", "bundle"
]

# Human-readable labels for UI
BADGE_LABELS: Dict[str, str] = {
    "frame": "All-in v. partitioned pricing",
    "assurance": "Assurance",
    "scarcity": "Scarcity tag",
    "strike": "Strike-through",
    "timer": "Timer",
    "social_proof": "social",
    "voucher": "voucher",
    "bundle": "bundle",
}

# -----------------------
# Robust I/O loader
# -----------------------

def _load_df(df_or_path: Union[pd.DataFrame, str, Path, Dict[str, Any], bytes, bytearray, Any]) -> pd.DataFrame:
    """
    Accept:
      - pandas.DataFrame
      - str / pathlib.Path to CSV/TXT/Parquet/JSON
      - dict payloads with a path under common keys (choice_path, path, file, csv),
        including nested (payload['paths']['choice'])
      - file-like objects (with .read)
      - raw CSV/JSON bytes
    """
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()

    if isinstance(df_or_path, (str, Path)):
        p = Path(str(df_or_path))
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
        for k in ("choice_path", "path", "file", "csv"):
            v = df_or_path.get(k)
            if isinstance(v, (str, Path)):
                return _load_df(v)
        for k in ("paths", "files", "data", "payload"):
            sub = df_or_path.get(k)
            if isinstance(sub, dict):
                for kk in ("choice", "choices", "path", "file", "csv"):
                    vv = sub.get(kk)
                    if isinstance(vv, (str, Path)):
                        return _load_df(vv)
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
    """Map confirmed headers to internal names."""
    return df.rename(columns={"case_id": "screen_id", "title": "product"})

def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = ["screen_id", "product", "chosen", "price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

def _make_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Use row_top, col1–col3 if present; else derive from row/col (baseline: bottom row, 4th col)."""
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

def _harmonize_badges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse mutually exclusive frame variants into a single contrast:
    frame = 1 (all-in) vs 0 (partitioned). Leaves original columns intact.
    """
    df = df.copy()
    has_allin = "frame_allin" in df.columns
    has_part  = "frame_partitioned" in df.columns
    if "frame" not in df.columns and (has_allin or has_part):
        if has_allin and has_part:
            fa = pd.to_numeric(df["frame_allin"], errors="coerce").fillna(0).astype(int)
            fp = pd.to_numeric(df["frame_partitioned"], errors="coerce").fillna(0).astype(int)
            df["frame"] = ((fa == 1) & (fp == 0)).astype(int)
        elif has_allin:
            df["frame"] = pd.to_numeric(df["frame_allin"], errors="coerce").fillna(0).astype(int)
        else:
            df["frame"] = 0
    return df

def _collect_badge_columns(df: pd.DataFrame) -> List[str]:
    """Return badge columns that exist and vary (≥2 distinct values)."""
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

    lever_cols: List[str] = []
    for c in ["row_top", "col1", "col2", "col3"]:
        if c in df.columns:
            lever_cols.append(c)
    if "ln_price" in df.columns:
        lever_cols.append("ln_price")

    badge_cols = _collect_badge_columns(df)

    model_cols: List[str] = []
    if "model" in df.columns:
        dummies = pd.get_dummies(df["model"], prefix="model", drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        model_cols = list(dummies.columns)

    clusters = df["screen_id"].astype(str)

    dmy_cols = ["screen_id"]
    if include_product_fe:
        dmy_cols.append("product")
    df_fe = pd.get_dummies(df, columns=dmy_cols, drop_first=True)

    x_vars = lever_cols + badge_cols + model_cols
    fe_cols = [c for c in df_fe.columns if c.startswith("screen_id_")]
    if include_product_fe:
        fe_cols += [c for c in df_fe.columns if c.startswith("product_")]

    x_vars_existing = [c for c in x_vars if c in df_fe.columns]
    X = df_fe[x_vars_existing + fe_cols]

    X = _drop_constant_cols(X)
    X = _coerce_numeric(X)

    y = pd.to_numeric(df_fe["chosen"], errors="coerce").fillna(0).astype(int)

    return y, X, clusters, x_vars_existing


# -----------------------
# Estimation
# -----------------------

def _fit_glm_logit(y, X, cov_type: str, cov_kwds: dict | None):
    """GLM(Logit) with configurable covariance; returns result object."""
    model = sm.GLM(y, X, family=sm.families.Binomial())
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = model.fit(cov_type=cov_type, cov_kwds=(cov_kwds or {}))
    return res

def _tidy_from_params(params, se, X):
    """Build tidy table given params and se; compute p, q_bh, OR, CI, AME, evid."""
    params = pd.Series(params, index=X.columns) if not isinstance(params, pd.Series) else params
    se     = pd.Series(se,     index=X.columns) if not isinstance(se,     pd.Series) else se

    # z and two-sided p
    z = params / se.replace(0.0, np.nan)
    pvals_arr = 2.0 * (1.0 - (0.5 * (1.0 + (2.0 / math.sqrt(math.pi)) *
                                     np.vectorize(math.erf)((np.abs(z)) / math.sqrt(2.0)))))
    pvals = pd.Series(np.asarray(pvals_arr, dtype=float), index=params.index)

    # Benjamini–Hochberg on numpy array
    _, q_bh_arr, _, _ = multipletests(np.asarray(pvals, dtype=float), method="fdr_bh")
    q_bh = pd.Series(q_bh_arr, index=params.index)

    # Odds ratios with clipping to avoid overflow
    CLIP = 40.0  # exp(±40) ≈ 2.35e17
    pruned = params.clip(lower=-CLIP, upper=CLIP)
    orx = np.exp(pruned)

    # CI on OR scale with clipping (use `se` here!)
    ci_arg_lo = np.clip(pruned - 1.96 * se, -CLIP, CLIP)
    ci_arg_hi = np.clip(pruned + 1.96 * se, -CLIP, CLIP)
    ci_low  = np.exp(ci_arg_lo)
    ci_high = np.exp(ci_arg_hi)

    # AME (neutral weight), evidence
    ame_pp = 100.0 * 0.25 * params
    evid   = (params.abs() / se.replace(0.0, np.nan))

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

def _fit_ridge_logit(y, X, alpha=1.0):
    """
    Ridge-penalised logit fallback (no clusters).
    Returns params and approximate SEs from observed Hessian at the solution.
    Robust to separation and tiny samples.
    """
    logit = sm.Logit(y, X)
    alpha_use = float(alpha)
    if X.shape[0] <= 120:
        alpha_use = max(alpha_use, 1.0)
    res = logit.fit_regularized(alpha=alpha_use, L1_wt=0.0, maxiter=1000, disp=False)

    params = pd.Series(res.params, index=X.columns)

    # Stabilised weights
    lin = np.asarray(X @ params)
    lin = np.clip(lin, -40.0, 40.0)
    p = 1.0 / (1.0 + np.exp(-lin))
    W = np.maximum(p * (1.0 - p), 1e-6)

    Xv = X.values
    XtWX = Xv.T @ (Xv * W[:, None])
    ridge_eps = 1e-4
    try:
        cov = np.linalg.inv(XtWX + ridge_eps * np.eye(XtWX.shape[0]))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(XtWX + ridge_eps * np.eye(XtWX.shape[0]))
    se = pd.Series(np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=np.inf)), index=X.columns)
    return params, se


def fit_logit_and_tidy(y: pd.Series, X: pd.DataFrame, clusters: pd.Series,
                       allow_product_fe: bool = True) -> pd.DataFrame:
    """
    Try GLM(Logit) with cluster-robust SEs. If degenerate (few clusters,
    non-finite params/SEs, or rank deficiency), fall back to ridge and
    return finite statistics. The returned DataFrame has attrs['fit_mode'].
    """
    n_clusters = pd.Series(clusters).nunique()
    MIN_CLUSTERS = 8
    if n_clusters < MIN_CLUSTERS or np.linalg.matrix_rank(X) < min(X.shape):
        params, se = _fit_ridge_logit(y, X, alpha=1.0)
        out = _tidy_from_params(params, se, X)
        out.attrs["fit_mode"] = "ridge_precheck"
        return out

    try:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})

        params = res.params.copy()
        bse = res.bse.copy()

        if (not np.all(np.isfinite(bse))) or (not np.all(np.isfinite(params))):
            raise RuntimeError("Degenerate GLM fit (non-finite params/SEs).")

        pvals = res.pvalues.copy()
        _, q_bh, _, _ = multipletests(pvals.values, method="fdr_bh")
        q_bh = pd.Series(q_bh, index=pvals.index)

        CLIP = 40.0
        pruned = params.clip(lower=-CLIP, upper=CLIP)
        odds_ratio = np.exp(pruned)
        ci_low = np.exp(pruned - 1.96 * bse)
        ci_high = np.exp(pruned + 1.96 * bse)

        p_hat = res.predict(X)
        weight_mean = float(np.mean(np.clip(p_hat * (1.0 - p_hat), 1e-6, 0.25)))
        ame_pp = 100.0 * weight_mean * params

        evid_score = (params.abs() / bse.replace(0.0, np.nan))

        out = pd.DataFrame({
            "beta": params, "se": bse, "p": pvals, "q_bh": q_bh,
            "odds_ratio": odds_ratio, "ci_low": ci_low, "ci_high": ci_high,
            "ame_pp": ame_pp, "evid_score": evid_score
        })
        out.attrs["fit_mode"] = "glm_cluster"
        return out

    except Exception:
        params, se = _fit_ridge_logit(y, X, alpha=1.0)
        out = _tidy_from_params(params, se, X)
        out.attrs["fit_mode"] = "ridge_fallback"
        return out


# -----------------------
# Results shaping for app
# -----------------------

def _results_to_rows(results_df: pd.DataFrame) -> List[Dict[str, Any]]:
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
            "badge": BADGE_LABELS.get(p, p),
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

def run_logit(df_or_path: Union[pd.DataFrame, str, Path, Dict[str, Any], bytes, bytearray, Any],
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

    # Derive position/log vars; harmonise frame variants
    df = _make_position_dummies(df)
    df = _make_logs(df)
    df = _harmonize_badges(df)

    # Guardrail for tiny pilots
    n_screens = df["screen_id"].nunique()
    if n_screens < min_cases:
        return []

    # Build design (start with product FE; if near-saturated, drop them)
    y, X_full, clusters, _ = _build_design(df, include_product_fe=True)
    near_sat = (X_full.shape[1] >= max(df.shape[0] - 2, 1))
    if near_sat:
        y, X, clusters, _ = _build_design(df, include_product_fe=False)
    else:
        X = X_full

    # Fit
    try:
        results = fit_logit_and_tidy(y, X, clusters, allow_product_fe=not near_sat)
    except Exception:
        params, se = _fit_ridge_logit(y, X, alpha=1.0)
        results = _tidy_from_params(params, se, X)
        results.attrs["fit_mode"] = "ridge_fallback_uncaught"

    # Optional debug: which path did we take?
    fit_mode = getattr(results, "attrs", {}).get("fit_mode")
    if fit_mode:
        print(f"[logit] fit_mode = {fit_mode}")

    # Shape rows for the app table
    rows = _results_to_rows(results)

    # Optional filter to only the user-selected badge labels
    if selected_badges:
        want = {s.strip().lower() for s in selected_badges}
        rows = [r for r in rows if r.get("badge") and r["badge"].strip().lower() in want]

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
    df = _harmonize_badges(df)

    n_screens = df["screen_id"].nunique()
    if n_screens < args.min_screens:
        raise ValueError(f"Insufficient screens for inference: {n_screens} < {args.min_screens}")

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





