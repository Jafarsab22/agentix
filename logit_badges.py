#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================
# CONDITIONAL (FIXED-EFFECTS) LOGIT FOR AI-AGENT READINESS SCORING
# ================================================================

from __future__ import annotations

import argparse
import io
import math
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
    "social_proof": "Social proof",
    "voucher": "Voucher",
    "bundle": "Bundle",
}

# -----------------------
# Robust I/O loader
# -----------------------

def _load_df(df_or_path: Union[pd.DataFrame, str, Path, Dict[str, Any], bytes, bytearray, Any]) -> pd.DataFrame:
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
    return df.rename(columns={"case_id": "screen_id", "title": "product"})

def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = ["screen_id", "product", "chosen", "price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

def _filter_complete_screens(df: pd.DataFrame, alts_per_screen: int) -> pd.DataFrame:
    counts = df.groupby("screen_id")["product"].transform("count")
    return df.loc[counts == alts_per_screen].copy()

def _make_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "row_top" not in df.columns and "row" in df.columns:
        row_i = pd.to_numeric(df["row"], errors="coerce")
        df["row_top"] = (row_i == 0).astype(int).fillna(0)
    if (not all(c in df.columns for c in ["col1", "col2", "col3"])) and "col" in df.columns:
        ci = pd.to_numeric(df["col"], errors="coerce")
        if "col1" not in df.columns:
            df["col1"] = (ci == 0).astype(int).fillna(0)
        if "col2" not in df.columns:
            df["col2"] = (ci == 1).astype(int).fillna(0)
        if "col3" not in df.columns:
            df["col3"] = (ci == 2).astype(int).fillna(0)
    for c in ["row_top", "col1", "col2", "col3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def _make_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price_num = pd.to_numeric(df["price"], errors="coerce")
    if (price_num <= 0).any():
        raise ValueError("Found non-positive prices; ln(price) undefined.")
    if "ln_price" not in df.columns:
        df["ln_price"] = np.log(price_num)
    return df

def _harmonize_badges(df: pd.DataFrame) -> pd.DataFrame:
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
    for c in BADGE_VARS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def _collect_badge_columns(df: pd.DataFrame) -> List[str]:
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

def _build_design(df: pd.DataFrame, include_product_fe: bool = True, use_price: bool = True) -> Tuple[pd.Series, pd.DataFrame, pd.Series, List[str]]:
    df = df.copy()

    lever_cols: List[str] = []
    for c in ["row_top", "col1", "col2", "col3"]:
        if c in df.columns:
            lever_cols.append(c)
    if use_price and "ln_price" in df.columns:
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

def _fit_glm_logit(y: pd.Series, X: pd.DataFrame, cov_type: str, cov_kwds: dict | None) -> sm.GLM:
    model = sm.GLM(y, X, family=sm.families.Binomial())
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = model.fit(cov_type=cov_type, cov_kwds=(cov_kwds or {}))
    return res

def _tidy_from_params(params, se, X):
    params = pd.Series(params, index=X.columns) if not isinstance(params, pd.Series) else params
    se     = pd.Series(se,     index=X.columns) if not isinstance(se,     pd.Series) else se

    z = params / se.replace(0.0, np.nan)
    z_vals = np.asarray(z, dtype=float)
    pvals_arr = np.array([math.erfc(abs(zi) / math.sqrt(2.0)) if np.isfinite(zi) else np.nan for zi in z_vals], dtype=float)
    pvals = pd.Series(pvals_arr, index=params.index)

    if np.isfinite(pvals_arr).any():
        _, q_bh_arr, _, _ = multipletests(np.nan_to_num(pvals_arr, nan=1.0), method="fdr_bh")
    else:
        q_bh_arr = np.full_like(pvals_arr, fill_value=np.nan, dtype=float)
    q_bh = pd.Series(q_bh_arr, index=params.index)

    CLIP = 40.0
    pruned = params.clip(lower=-CLIP, upper=CLIP)
    orx = np.exp(pruned)

    ci_arg_lo = np.clip(pruned - 1.96 * se, -CLIP, CLIP)
    ci_arg_hi = np.clip(pruned + 1.96 * se, -CLIP, CLIP)
    ci_low  = np.exp(ci_arg_lo)
    ci_high = np.exp(ci_arg_hi)

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

def _fit_ridge_logit(y: pd.Series, X: pd.DataFrame, alpha: float = 0.05) -> Tuple[pd.Series, pd.Series]:
    """
    Ridge-penalised logit fallback (no clusters).
    Gentle L2 penalty by default; robust to separation and small samples.
    SEs are derived from (X'WX + αI)^(-1).
    """
    logit = sm.Logit(y, X)
    alpha_use = float(alpha)

    res = logit.fit_regularized(alpha=alpha_use, L1_wt=0.0, maxiter=1000, disp=False)
    params = pd.Series(res.params, index=X.columns)

    lin = np.asarray(X @ params)
    lin = np.clip(lin, -40.0, 40.0)
    p = 1.0 / (1.0 + np.exp(-lin))
    W = np.maximum(p * (1.0 - p), 1e-6)

    Xv = X.values
    XtWX = Xv.T @ (Xv * W[:, None])
    # Penalised Fisher information inverse
    A = XtWX + alpha_use * np.eye(XtWX.shape[0])
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(A)
    se = pd.Series(np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=np.inf)), index=X.columns)
    return params, se

def fit_glm_tidy_with_cov_strategy(y: pd.Series, X: pd.DataFrame, clusters: pd.Series, cluster_min: int = 30) -> Tuple[pd.DataFrame, str]:
    n_clusters = pd.Series(clusters).nunique()
    cov_type = "cluster" if n_clusters >= cluster_min else "HC1"
    cov_kwds = {"groups": clusters} if cov_type == "cluster" else None

    res = _fit_glm_logit(y, X, cov_type=cov_type, cov_kwds=cov_kwds)
    params = res.params.copy()
    bse = res.bse.copy()

    if (not np.all(np.isfinite(params))) or (not np.all(np.isfinite(bse))):
        raise RuntimeError("Degenerate GLM fit (non-finite params/SEs).")

    pvals = res.pvalues.copy()
    if np.isfinite(pvals.values).any():
        _, q_bh_vals, _, _ = multipletests(np.nan_to_num(pvals.values, nan=1.0), method="fdr_bh")
        q_bh = pd.Series(q_bh_vals, index=pvals.index)
    else:
        q_bh = pd.Series(np.full_like(pvals.values, np.nan, dtype=float), index=pvals.index)

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
    mode = "glm_cluster" if cov_type == "cluster" else "glm_hc1"
    return out, mode

# -----------------------
# Results shaping for app
# -----------------------

def _results_to_rows(results_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = list(results_df.index.astype(str))

    b_price = None
    if ("ln_price" in idx) and ("beta" in results_df.columns) and pd.notna(results_df.loc["ln_price", "beta"]):
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

def run_logit(
    df_or_path: Union[pd.DataFrame, str, Path, Dict[str, Any], bytes, bytearray, Any],
    selected_badges: List[str] | None = None,
    min_cases: int = 10,
    use_price: bool = True,
    alts_per_screen: int = 8,
    ridge_threshold: int = 30,
    cluster_min: int = 30
) -> List[Dict[str, Any]]:
    df = _load_df(df_or_path)
    df = _rename_core_columns(df)
    _ensure_required_columns(df)

    df = df.dropna(subset=["chosen", "screen_id", "product", "price"]).copy()
    df["chosen"] = pd.to_numeric(df["chosen"], errors="coerce").fillna(0).astype(int)

    df = _make_position_dummies(df)
    df = _make_logs(df)
    df = _harmonize_badges(df)

    df = _filter_complete_screens(df, alts_per_screen=alts_per_screen)

    n_screens = df["screen_id"].nunique()
    if n_screens < min_cases:
        return []

    if use_price and "ln_price" in df.columns:
        within_var = df.groupby("screen_id")["ln_price"].nunique()
        if (within_var <= 1).all():
            use_price = False

    y, X_full, clusters, _ = _build_design(df, include_product_fe=True, use_price=use_price)
    near_sat = (X_full.shape[1] >= max(df.shape[0] - 2, 1))
    if near_sat:
        y, X, clusters, _ = _build_design(df, include_product_fe=False, use_price=use_price)
    else:
        X = X_full

    X = _drop_constant_cols(X)

    try:
        if n_screens < ridge_threshold:
            params, se = _fit_ridge_logit(y, X, alpha=0.05)
            results = _tidy_from_params(params, se, X)
            results.attrs["fit_mode"] = "ridge_default"
        else:
            results, mode = fit_glm_tidy_with_cov_strategy(y, X, clusters, cluster_min=cluster_min)
            results.attrs["fit_mode"] = mode
    except Exception:
        params, se = _fit_ridge_logit(y, X, alpha=0.05)
        results = _tidy_from_params(params, se, X)
        results.attrs["fit_mode"] = "ridge_fallback_uncaught"

    fit_mode = getattr(results, "attrs", {}).get("fit_mode")
    if fit_mode:
        print(f"[logit] fit_mode = {fit_mode}; screens={n_screens}; rows={df.shape[0]}; cols={X.shape[1]}")

    rows = _results_to_rows(results)

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
    parser.add_argument("--min_screens", type=int, default=10, help="Minimum COMPLETE screens required to proceed.")
    parser.add_argument("--alts_per_screen", type=int, default=8, help="Alternatives per complete screen (default 8).")
    parser.add_argument("--ridge_threshold", type=int, default=30, help="Use ridge when screens < this threshold.")
    parser.add_argument("--cluster_min", type=int, default=30, help="Min clusters for cluster-robust SEs.")
    parser.add_argument("--no_price", action="store_true", help="If set, drop ln_price even if present.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = _rename_core_columns(df)
    _ensure_required_columns(df)

    df = df.dropna(subset=["chosen", "screen_id", "product", "price"]).copy()
    df["chosen"] = pd.to_numeric(df["chosen"], errors="coerce").fillna(0).astype(int)

    df = _make_position_dummies(df)
    df = _make_logs(df)
    df = _harmonize_badges(df)

    df = _filter_complete_screens(df, alts_per_screen=int(args.alts_per_screen))

    n_screens = df["screen_id"].nunique()
    if n_screens < int(args.min_screens):
        raise ValueError(f"Insufficient COMPLETE screens for inference: {n_screens} < {int(args.min_screens)}")

    use_price = not bool(args.no_price)
    if use_price and "ln_price" in df.columns:
        within_var = df.groupby("screen_id")["ln_price"].nunique()
        if (within_var <= 1).all():
            use_price = False

    y, X_full, clusters, _ = _build_design(df, include_product_fe=True, use_price=use_price)
    near_sat = (X_full.shape[1] >= max(df.shape[0] - 2, 1))
    if near_sat:
        y, X, clusters, _ = _build_design(df, include_product_fe=False, use_price=use_price)
    else:
        X = X_full

    X = _drop_constant_cols(X)

    if n_screens < int(args.ridge_threshold):
        params, se = _fit_ridge_logit(y, X, alpha=0.05)
        results = _tidy_from_params(params, se, X)
        results.attrs["fit_mode"] = "ridge_default"
    else:
        try:
            results, mode = fit_glm_tidy_with_cov_strategy(y, X, clusters, cluster_min=int(args.cluster_min))
            results.attrs["fit_mode"] = mode
        except Exception:
            params, se = _fit_ridge_logit(y, X, alpha=0.05)
            results = _tidy_from_params(params, se, X)
            results.attrs["fit_mode"] = "ridge_fallback"

    results.round(6).to_csv(args.output, index=True)

    show_keys: List[str] = []
    for v in ["row_top", "col1", "col2", "col3", "ln_price"] + BADGE_VARS:
        if v in results.index:
            show_keys.append(v)

    print("Model fitted and results written to", args.output)
    print(f"[logit] fit_mode = {getattr(results, 'attrs', {}).get('fit_mode')}; screens={n_screens}; rows={df.shape[0]}; cols={X.shape[1]}")
    if show_keys:
        cols = ["beta", "se", "p", "q_bh", "odds_ratio", "ci_low", "ci_high", "ame_pp", "evid_score"]
        safe_cols = [c for c in cols if c in results.columns]
        print("=== Key Lever Effects (position, price, badges) ===")
        print(results.loc[show_keys, safe_cols].to_string())

if __name__ == "__main__":
    main()
