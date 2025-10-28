# -*- coding: utf-8 -*-
"""
logit_badges_clean.py — Conditional logit with fixed effects, robust small‑N handling,
continuous evidence metrics, and a backward‑compatible table for the runner.

Notes & assumptions (please read):
1) Model family and link: Binomial GLM with logit link fitted by IRLS. This is
   equivalent to a logistic regression. We approximate a conditional-logit setup
   by absorbing choice-set (case) and product fixed effects through categorical
   dummies (no intercept), which yields the same estimates for the coefficients
   of interest when each case contains exactly one picked alternative (y ∈ {0,1}).
2) Fixed effects: We include C(case_id) and C(prod_id). This captures arbitrary
   case- and product-level heterogeneity and aligns with the storefront’s design
   (8 alternatives per case). Position controls are included as row/column dummies
   when available (row ∈ {1,2}, col ∈ {1,2,3,4}). No global intercept is used.
3) Penalisation at small N: If the number of complete cases (n_cases) is below
   ridge_cutoff_cases (default 50), we fit a ridge‑penalised GLM to stabilise
   estimates. Penalisation does not target fixed effects differently from other
   regressors; it is an ℓ2 penalty on all coefficients. In the MLE regime
   (n_cases ≥ ridge_cutoff_cases), the penalty is set to 0.
4) Inference: p-values come from (a) robust (HC1) covariance when unpenalised,
   or (b) the penalised Fisher information when ridge>0. For multiple badges we
   also compute Benjamini–Hochberg FDR-adjusted q-values across the lever terms.
   We additionally report odds ratios and 95% CIs, and a continuous evidence score
   s_evid = -log10(q) (using p when q is not computable).
5) Average marginal effects (AME): For a binary indicator x, the derivative of
   the logit probability w.r.t. x at observation i is p_i*(1-p_i)*β_x. We report
   AME_x = mean_i p_i*(1-p_i)*β_x, as percentage points (×100). This is standard
   for small, per‑unit changes and aligns with practice in site‑level reporting.
6) Inputs expected: Either a pandas DataFrame with the choice data or a path to
   a CSV file. Required columns: [choice, case_id, prod_id] and at least one of
   the lever columns in LEVER_VARS. Optional: row, col, price, ln_price. The
   function will coerce types and create missing IDs if not present.
7) Dropping non‑varying levers: Any lever that is constant in the analysis slice
   is removed prior to estimation. This is deliberate and avoids singular design
   matrices. We log which levers were dropped in the returned metadata.
8) Backward compatibility: The primary entry point run_logit(...) returns a
   DataFrame with columns [badge, beta, p, sign] as expected by the runner, plus
   richer columns (se, q_bh, odds_ratio, ci_low, ci_high, ame_pp, evid_score).
   The runner can safely ignore the extras. Column names match the existing
   conventions ("All-in framing", "Purchase assurance", etc.).
9) Missing data policy: Choice rows with missing in any active regressor are
   dropped listwise. "Complete cases" are defined at the case_id level: a case
   contributes only if it has exactly 8 alternatives and one choice==1.
10) Reproducibility: The estimator itself is deterministic given the input.

Author: Agentix (2025-10-28)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import patsy as pt
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LEVER_VARS = [
    ("frame", "All-in framing"),
    ("assurance", "Purchase assurance"),
    ("scarcity", "Scarcity tag"),
    ("strike", "Strike-through tag"),
    ("timer", "Countdown timer"),
    ("social_proof", "Social proof"),
    ("voucher", "Voucher"),
    ("bundle", "Bundle")
]

@dataclass
class LogitResult:
    table: pd.DataFrame
    meta: Dict[str, Any]


def run_logit(
    df_or_path: Union[str, Path, pd.DataFrame],
    selected_badges: List[str] | None = None,
    min_cases: int = 10,
    ridge_cutoff_cases: int = 50,
    ridge_alpha: float = 1.0,
    use_price: bool = False,
) -> pd.DataFrame:
    """Fit the conditional logit and return a tidy effects table.

    Backward compatibility: the second positional argument may be a list of
    selected badge names coming from the runner UI. This function accepts that
    as ``selected_badges``. All other parameters are keyword-friendly.
    """
    # Load and normalise data
    df = _load_df(df_or_path)
    df = _coerce_schema(df)
    df = _filter_complete_cases(df)

    n_cases = df["case_id"].nunique()

    # Map UI names → canonical labels used in LEVER_VARS
    ui_to_label = {
        "All-in v. partitioned pricing": "All-in framing",
        "All-in framing": "All-in framing",
        "Assurance": "Purchase assurance",
        "Purchase assurance": "Purchase assurance",
        "Scarcity": "Scarcity tag",
        "Scarcity tag": "Scarcity tag",
        "Strike-through": "Strike-through tag",
        "Strike-through tag": "Strike-through tag",
        "Timer": "Countdown timer",
        "Countdown timer": "Countdown timer",
        "Social proof": "Social proof",
        "Voucher": "Voucher",
        "Bundle": "Bundle",
    }

    # Identify varying levers
    present: List[Tuple[str, str]] = []
    for var, label in LEVER_VARS:
        if var in df.columns and df[var].nunique() > 1:
            present.append((var, label))

    # Optional filtering by selected_badges from UI
    if selected_badges:
        want_labels = {ui_to_label.get(s, s) for s in selected_badges}
        present = [(v, lbl) for (v, lbl) in present if lbl in want_labels]

    if n_cases < min_cases or len(present) == 0:
        return _empty_table()

    # Build design formula: no intercept, FE for case & product, optional price, position controls
    x_terms: List[str] = []
    x_terms.extend(var for var, _ in present)
    if use_price:
        if "ln_price" in df.columns:
            x_terms.append("ln_price")
        elif "price" in df.columns:
            x_terms.append("price")
    # Position controls
    for c in ["row", "col"]:
        if c in df.columns:
            x_terms.append(f"C({c})")
    # Fixed effects
    x_terms.append("C(case_id)")
    x_terms.append("C(prod_id)")

    formula = "choice ~ -1 + " + " + ".join(x_terms)

    # Build design matrices
    y, X = pt.dmatrices(formula, df, return_type="dataframe")

    # Ridge or MLE
    ridge = float(ridge_alpha) if n_cases < ridge_cutoff_cases else 0.0

    if ridge > 0.0:
        # Penalised GLM (L2) via regularized Logit
        model = sm.Logit(y, X)
        res = model.fit_regularized(method="l1", alpha=ridge, L1_wt=0.0, disp=False, maxiter=1000)
        params = res.params
        # Approximate covariance by penalised Fisher info (X'WX + 2*alpha*I)^-1
        p_pred = res.predict()
        W = np.asarray(p_pred * (1.0 - p_pred))
        XtW = (X.T * W)
        H = XtW @ X + 2.0 * ridge * np.eye(X.shape[1])
        cov = np.linalg.pinv(H)
        se = pd.Series(np.sqrt(np.diag(cov)), index=X.columns)
    else:
        model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.logit()))
        res = model.fit(cov_type="HC1", maxiter=1000)
        params = res.params
        se = res.bse

    # Compute tidy stats for levers only (exclude FE and position controls)
    lever_index = [col for col in X.columns if col in [v for v, _ in present]]
    if len(lever_index) == 0:
        return _empty_table()

    out = pd.DataFrame({
        "badge_var": lever_index,
        "beta": params[lever_index].astype(float),
        "se": se[lever_index].astype(float),
    })
    out["z"] = out["beta"] / out["se"]

    # Two-sided p-values from normal approximation
    from scipy.stats import norm
    out["p"] = 2.0 * (1.0 - norm.cdf(np.abs(out["z"])) )

    # Benjamini–Hochberg q-values
    out = out.sort_values("p").reset_index(drop=True)
    m = out.shape[0]
    ranks = np.arange(1, m + 1)
    out["q_bh"] = np.minimum.accumulate((out["p"].values * m / ranks)[::-1])[::-1]

    # Evidence score and significance flags
    eps = 1e-16
    out["evid_score"] = -np.log10(np.maximum(out["q_bh"].fillna(out["p"]), eps))
    out["sig_5pct"] = out["p"] < 0.05

    # Odds ratios and CIs
    out["odds_ratio"] = np.exp(out["beta"]) 
    out["ci_low"] = np.exp(out["beta"] - 1.96 * out["se"]) 
    out["ci_high"] = np.exp(out["beta"] + 1.96 * out["se"]) 

    # AME (percentage points)
    if ridge > 0.0:
        p_hat = res.predict()
    else:
        p_hat = res.predict(res.model.exog)
    V = np.asarray(p_hat * (1.0 - p_hat))
    ame = {}
    for v, _label in present:
        if v in X.columns:
            beta_v = params[v]
            ame[v] = float(100.0 * np.mean(V * beta_v))
    out["ame_pp"] = out["badge_var"].map(ame)

    # Human-friendly labels and signs
    label_map = {var: label for var, label in LEVER_VARS}
    out["badge"] = out["badge_var"].map(label_map)
    out["dir"] = out["beta"].apply(lambda b: "+" if b > 0 else ("-" if b < 0 else "0"))
    out["sign"] = np.where(out["p"] >= 0.05, "0", np.where(out["beta"] > 0, "+", np.where(out["beta"] < 0, "-", "0")))

    # Backward-compatible subset and enriched columns
    ordered_cols = [
        "badge", "beta", "p", "sign",
        "dir", "se", "q_bh", "odds_ratio", "ci_low", "ci_high", "ame_pp", "evid_score",
    ]
    out = out[ordered_cols]

    # Stable sort by badge label for predictable tables
    out = out.sort_values("badge").reset_index(drop=True)

    return out

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "badge", "beta", "p", "sign",
        "se", "q_bh", "odds_ratio", "ci_low", "ci_high", "ame_pp", "evid_score"
    ])


def _load_df(df_or_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()
    path = Path(df_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Choice file not found: {path}")
    return pd.read_csv(path)


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Schema normalisation so runner/logit never disagree on column names.
    Accept any of {"choice","chosen","picked","y"} as the outcome and map to "choice".
    Create missing IDs; coerce lever dtypes; handle optional controls.
    """
    df = df.copy()

    # --- Outcome: unify to 'choice' ---
    if "choice" not in df.columns:
        for cand in ("chosen", "picked", "y"):
            if cand in df.columns:
                df["choice"] = df[cand]
                break
    if "choice" not in df.columns:
        raise KeyError("No outcome column found. Expected one of: 'choice','chosen','picked','y'.")
    df["choice"] = pd.to_numeric(df["choice"], errors="coerce").fillna(0).astype(int)

    # --- IDs ---
    if "case_id" not in df.columns:
        if "set_id" in df.columns:
            df["case_id"] = df["set_id"].astype(str)
        else:
            df["case_id"] = "S0001"
    if "prod_id" not in df.columns:
        if "title" in df.columns:
            df["prod_id"] = df["title"].astype(str)
        else:
            # ensure uniqueness within case if nothing else is available
            df["prod_id"] = df.groupby("case_id").cumcount().astype(str)

    # --- Levers to int ---
    for var, _label in LEVER_VARS:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors="coerce").fillna(0).astype(int)

    # --- Optional controls ---
    for c in ["row", "col"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["price", "ln_price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _filter_complete_cases(df: pd.DataFrame) -> pd.DataFrame:
    # keep only cases with 8 alternatives and exactly one choice==1
    g = df.groupby("case_id")
    mask = (g["choice"].transform("size") == 8) & (g["choice"].transform("sum") == 1)
    return df.loc[mask].copy()

# ---------------------------------------------------------------------------
# Runner-side helper (optional): write full effects table to CSV with metadata
# ---------------------------------------------------------------------------

def write_badge_effects_csv(df_badges: pd.DataFrame, badges_effects_path: Union[str, Path], job_meta: Dict[str, Any], include_legacy: bool = True, legacy_path: Union[str, Path] | None = None) -> None:
    """Persist the full effects table plus job metadata.

    Parameters
    ----------
    df_badges : DataFrame
        Output of run_logit(...).
    badges_effects_path : str | Path
        Destination CSV for the rich effects table.
    job_meta : dict
        Keys you want to prepend as columns (e.g., job_id, timestamp, product, brand, price, currency, n_iterations).
    include_legacy : bool
        If True, also write a legacy 4-column CSV to legacy_path (or alongside if provided).
    legacy_path : str | Path | None
        Destination for the legacy file (badge,beta,p,sign). Required if include_legacy is True.
    """
    badges_effects_path = Path(badges_effects_path)
    if include_legacy and legacy_path is None:
        raise ValueError("legacy_path must be provided when include_legacy=True")

    # Preferred column order; keep whatever is present
    pref = [
        "badge", "beta", "p", "sign", "dir",
        "se", "q_bh", "odds_ratio", "ci_low", "ci_high", "ame_pp", "evid_score"
    ]
    cols = [c for c in pref if c in df_badges.columns]
    df_out = df_badges[cols].copy()

    # Prepend metadata columns in a stable order
    meta_keys = list(job_meta.keys())
    for k in meta_keys[::-1]:
        df_out.insert(0, k, job_meta[k])

    badges_effects_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(badges_effects_path, index=False)

    if include_legacy:
        legacy_cols = [c for c in ["badge", "beta", "p", "sign"] if c in df_badges.columns]
        df_legacy = df_badges[legacy_cols].copy()
        Path(legacy_path).parent.mkdir(parents=True, exist_ok=True)
        df_legacy.to_csv(legacy_path, index=False)
