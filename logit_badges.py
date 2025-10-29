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
# Outputs
#   • CSV: logit_readiness_results.csv (index=parameter name; columns as above)
#
# Notes
#   • Uses absorbed dummies for FE (screen_id_*, product_*). For very large data,
#     consider true conditional likelihood (“clogit”) or high-dimensional FE solvers.
#   • Interactions: badge × model and position × model to capture heterogeneity.
#   • Price-equivalent trade-offs are computed downstream in app.py.
# ================================================================

import argparse
import sys
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = ["screen_id", "product", "chosen", "price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def _make_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Create paper-style position dummies: row_top; col1, col2, col3 (baseline: bottom row, col4)."""
    if "row" in df.columns:
        df["row_top"] = (df["row"].astype(int) == 0).astype(int)
    if "col" in df.columns:
        ci = df["col"].astype(int)
        df["col1"] = (ci == 0).astype(int)
        df["col2"] = (ci == 1).astype(int)
        df["col3"] = (ci == 2).astype(int)
    return df

def _make_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Create ln_price and ln_reviews where applicable; guard against non-positive price."""
    if (df["price"] <= 0).any():
        raise ValueError("Found non-positive prices; ln(price) undefined. Clean or filter the data.")
    df["ln_price"] = np.log(df["price"].astype(float))
    if "reviews" in df.columns:
        df["ln_reviews"] = np.log1p(df["reviews"].astype(float))
    return df

def _collect_badge_columns(df: pd.DataFrame) -> list:
    """Any column starting with badge_ is treated as a lever."""
    return [c for c in df.columns if c.startswith("badge_")]

def _collect_model_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Make model dummies if a 'model' column exists; return updated df."""
    if "model" in df.columns:
        df = pd.get_dummies(df, columns=["model"], drop_first=True)
    return df

def _add_interactions(df: pd.DataFrame, badge_cols: list, model_cols: list, pos_cols: list) -> list:
    """Create badge×model and position×model interactions; return list of interaction column names."""
    inter_cols = []
    for b in badge_cols:
        for m in model_cols:
            name = f"{b}_x_{m}"
            df[name] = df[b] * df[m]
            inter_cols.append(name)
    for pvar in pos_cols:
        if pvar in df.columns:
            for m in model_cols:
                name = f"{pvar}_x_{m}"
                df[name] = df[pvar] * df[m]
                inter_cols.append(name)
    return inter_cols

def _build_design(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.Series, list]:
    """
    Build y, X with absorbed FEs, retain original screen_id for clustering.
    Returns: y, X, clusters, list_of_predictor_names_in_order
    """
    # Base levers (include position first for paper-style presentation)
    lever_cols = []
    for c in ["row_top", "col1", "col2", "col3"]:
        if c in df.columns:
            lever_cols.append(c)
    if "ln_price" in df.columns:
        lever_cols.append("ln_price")
    if "rating" in df.columns:
        lever_cols.append("rating")
    if "ln_reviews" in df.columns:
        lever_cols.append("ln_reviews")

    badge_cols = _collect_badge_columns(df)
    df = _collect_model_dummies(df)
    model_cols = [c for c in df.columns if c.startswith("model_")]
    inter_cols = _add_interactions(df, badge_cols, model_cols, ["row_top", "col1", "col2", "col3"])

    # Keep original screen_id for clustering
    clusters = df["screen_id"].astype(str)

    # Absorb fixed effects via dummies (no intercept)
    df_fe = pd.get_dummies(df, columns=["screen_id", "product"], drop_first=True)

    # Assemble X variable list
    x_vars = lever_cols + badge_cols + model_cols + inter_cols
    fe_cols = [c for c in df_fe.columns if c.startswith("screen_id_") or c.startswith("product_")]

    # Make sure all requested columns exist before selection
    x_vars_existing = [c for c in x_vars if c in df_fe.columns]
    X = df_fe[x_vars_existing + fe_cols]
    y = df_fe["chosen"].astype(int)

    return y, X, clusters, x_vars_existing

def _check_one_choice_per_screen(df: pd.DataFrame) -> None:
    counts = df.groupby("screen_id")["chosen"].sum()
    ok = (counts == 1).all()
    if not ok:
        bad = counts[(counts != 1)]
        print(f"Warning: {bad.shape[0]} screens violate one-choice-per-screen. Proceed with caution.", file=sys.stderr)

def fit_logit_and_tidy(y: pd.Series, X: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """
    Fit Binomial GLM (logit), cluster-robust SEs by screen, compute p, q_bh,
    odds-ratios and CIs, AMEs, and evidence score. Return tidy DataFrame indexed by parameter name.
    """
    # Fit model without intercept because FE dummies already include baselines.
    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})

    params = res.params.copy()
    bse = res.bse.copy()
    pvals = res.pvalues.copy()

    # Benjamini–Hochberg FDR adjustment over all parameters
    _, q_bh, _, _ = multipletests(pvals.values, method="fdr_bh")
    q_bh = pd.Series(q_bh, index=pvals.index)

    # Odds ratios and 95% CI on OR scale
    odds_ratio = np.exp(params)
    ci_low = np.exp(params - 1.96 * bse)
    ci_high = np.exp(params + 1.96 * bse)

    # Average marginal effects (approx.) — use global mean of p*(1−p)
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

def main():
    parser = argparse.ArgumentParser(description="Conditional (FE) logit for AI-agent readiness.")
    parser.add_argument("--input", type=str, default="ai_agent_choices.csv", help="Input CSV with choices.")
    parser.add_argument("--output", type=str, default="logit_readiness_results.csv", help="Output CSV for tidy results.")
    parser.add_argument("--min_screens", type=int, default=10, help="Minimum number of screens required to proceed.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    _ensure_required_columns(df)

    # Basic cleaning
    df = df.dropna(subset=["chosen", "screen_id", "product", "price"]).copy()
    _check_one_choice_per_screen(df)

    # Create logs and paper-style position dummies
    df = _make_logs(df)
    df = _make_position_dummies(df)

    # Build design with absorbed FE and cluster groups
    y, X, clusters, x_vars = _build_design(df)

    # Sample size checks
    n_screens = df["screen_id"].nunique()
    n_rows = df.shape[0]
    if n_screens < args.min_screens:
        raise ValueError(f"Insufficient screens for inference: {n_screens} < {args.min_screens}")
    if X.shape[1] >= n_rows:
        print("Warning: predictors (including FE dummies) nearly exhaust sample size; consider using a true clogit or reducing FE dimensionality.", file=sys.stderr)

    # Fit and tidy
    results = fit_logit_and_tidy(y, X, clusters)

    # Persist
    results.round(6).to_csv(args.output, index=True)

    # Console summary for common levers, if present
    show_keys = []
    for v in ["row_top", "col1", "col2", "col3", "ln_price", "rating", "ln_reviews"]:
        if v in results.index:
            show_keys.append(v)
    badge_keys = [ix for ix in results.index if ix.startswith("badge_")]
    show_keys.extend(badge_keys)

    print("Model fitted and results written to logit_readiness_results.csv")
    if show_keys:
        cols = ["beta", "se", "p", "q_bh", "odds_ratio", "ci_low", "ci_high", "ame_pp", "evid_score"]
        safe_cols = [c for c in cols if c in results.columns]
        print("=== Key Lever Effects (position, price, badges) ===")
        print(results.loc[show_keys, safe_cols].to_string())

if __name__ == "__main__":
    main()
