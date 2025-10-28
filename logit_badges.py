# logit_badges_v2.py
# Robust badge-effects estimator for Agentix v1.7 data
# - K−1 reference coding for non-frame badges (deterministic baseline)
# - Uses ln_price_z (z-scored log price) as the single price regressor
# - Includes screen fixed effects via C(case_id) only (omit product/title FE)
# - Optional L2-regularised logistic as a stability check (sklearn optional)
# - Single-line prints only

import sys
import json
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pathlib as _pl

# sklearn is optional
try:
    from sklearn.linear_model import LogisticRegression
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

BADGE_COLS_CANON = [
    "assurance", "scarcity", "strike", "timer", "social_proof", "voucher", "bundle"
]


def _pick_present_badges(df: pd.DataFrame) -> List[str]:
    return [c for c in BADGE_COLS_CANON if c in df.columns]


def _choose_reference_badge(present_badges: List[str]) -> str:
    if not present_badges:
        return ""
    return sorted(present_badges)[0]


def _build_design(
    df: pd.DataFrame,
    use_case_fe: bool = True,
    ref_badge: str = ""
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if "chosen" not in df.columns:
        raise ValueError("Dataframe must contain 'chosen' column (0/1).")
    if "ln_price" not in df.columns:
        raise ValueError("Dataframe must contain 'ln_price' column.")

    X_parts = []
    df = df.copy()

    # Price regressor (z-scored)
    df["ln_price_z"] = (df["ln_price"] - df["ln_price"].mean()) / df["ln_price"].std(ddof=0)
    X_parts.append(df[["ln_price_z"]])

    # Frame (if available)
    if "frame" in df.columns:
        X_parts.append(df[["frame"]].astype(float))

    # Badges with K−1 coding
    badges = _pick_present_badges(df)
    if badges:
        base = ref_badge or _choose_reference_badge(badges)
        keep = [b for b in badges if b != base]
        if keep:
            X_parts.append(df[keep].astype(float))
    else:
        base = ""
        keep = []

    # Screen fixed effects via case_id if available
    if use_case_fe:
        if "case_id" not in df.columns:
            raise ValueError("use_case_fe=True requires 'case_id' in dataframe.")
        d_case = pd.get_dummies(df["case_id"], prefix="scr", drop_first=True)
        X_parts.append(d_case.astype(float))

    X = pd.concat(X_parts, axis=1)

    # Clean: replace inf/NaN, drop non-numeric, drop constants
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        X = X.drop(columns=non_num)
    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        X = X.drop(columns=const_cols)

    y = df["chosen"].astype(int)
    feat_list = ["ln_price_z"] + (["frame"] if "frame" in df.columns else []) + [c for c in keep]
    return X, y, feat_list


def fit_logit(X: pd.DataFrame, y: pd.Series):
    X_sm = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X_sm)
    res = model.fit(disp=False, maxiter=500)
    return res


def fit_logit_l2(X: pd.DataFrame, y: pd.Series):
    if not _HAVE_SK:
        return None
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, fit_intercept=True)
    clf.fit(X, y)
    return clf


def run_estimation(df: pd.DataFrame, use_case_fe: bool = True, ref_badge: str = "") -> dict:
    present = _pick_present_badges(df)
    base = ref_badge or _choose_reference_badge(present)
    X, y, feat_order = _build_design(df, use_case_fe=use_case_fe, ref_badge=base)
    res = fit_logit(X, y)
    clf = fit_logit_l2(X, y)

    out = {
        "n": int(len(y)),
        "k": int(X.shape[1]),
        "ref_badge": base,
        "features": feat_order,
        "converged": bool(res.mle_retvals.get("converged", False)),
        "llf": float(res.llf),
        "llnull": float(res.llnull),
        "pseudo_r2": float(1 - res.llf / res.llnull) if res.llnull != 0 else float("nan"),
        "params": res.params.to_dict(),
        "bse": res.bse.to_dict(),
        "pvalues": res.pvalues.to_dict(),
    }

    if clf is not None:
        coefs = np.r_[clf.intercept_, clf.coef_.ravel()]
        out["sklearn_intercept_coef"] = float(coefs[0])
        out["sklearn_first10"] = [(c, float(v)) for c, v in list(zip(["intercept"] + X.columns.tolist(), coefs))[:10]]
        out["sklearn_n_iter"] = int(clf.n_iter_[0]) if hasattr(clf, "n_iter_") else -1
    else:
        out["sklearn_note"] = "sklearn not installed; skipping L2 check."
    return out


def _format_effect_rows(result: dict) -> pd.DataFrame:
    # Build a compact table for UI consumption: badge | beta | p | sign
    params = result.get("params", {})
    pvals = result.get("pvalues", {})

    def _sign(beta: float, p: float) -> str:
        if p < 0.05 and beta > 0:
            return "↑"
        if p < 0.05 and beta < 0:
            return "↓"
        return ""

    pretty_names = {
        "frame": "Pricing frame (all-in)",
        "assurance": "Assurance",
        "scarcity": "Scarcity",
        "strike": "Strike-through",
        "timer": "Timer",
        "social_proof": "Social proof",
        "voucher": "Voucher",
        "bundle": "Bundle",
    }

    rows = []
    # Preserve a sensible order: frame first (if present), then canonical badges in BADGE_COLS_CANON order
    keys = []
    if "frame" in params:
        keys.append("frame")
    keys.extend([c for c in BADGE_COLS_CANON if c in params])

    for k in keys:
        beta = float(params[k])
        p = float(pvals.get(k, np.nan))
        rows.append({
            "badge": pretty_names.get(k, k),
            "beta": beta,
            "p": p,
            "sign": _sign(beta, p)
        })

    return pd.DataFrame(rows)


# ---------- NEW: wrapper expected by agent_runner.run_job_sync ----------

# --- add this to logit_badges.py ---

def run_logit(path_or_df: Union[str, _pl.Path, pd.DataFrame], selected_badges=None) -> pd.DataFrame:
    """
    Adapter for agent_runner.run_job_sync.
    Returns a compact effects table DataFrame with columns [badge, beta, p, sign].
    Includes 'frame' only if it varies; includes only non-frame badges that exist and vary.
    Applies screen fixed effects via 'case_id'.
    """
    # Load
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)

    # Ensure 'case_id' for fixed effects
    if "case_id" not in df.columns:
        if "set_id" in df.columns:
            df["case_id"] = df["set_id"].astype(str)
        else:
            df["case_id"] = "S0001"

    # Keep only known columns if present
    keep_cols = [c for c in [
        "case_id", "chosen", "ln_price", "frame",
        "assurance", "scarcity", "strike", "timer", "social_proof", "voucher", "bundle"
    ] if c in df.columns]
    dfm = df[keep_cols].copy()

    # Drop badges with no variance
    for b in ["assurance", "scarcity", "strike", "timer", "social_proof", "voucher", "bundle"]:
        if b in dfm.columns and dfm[b].nunique(dropna=False) <= 1:
            dfm.drop(columns=[b], inplace=True)

    # Drop frame if no variance
    if "frame" in dfm.columns and dfm["frame"].nunique(dropna=False) <= 1:
        dfm.drop(columns=["frame"], inplace=True)

    # If only price + FE remain, result table may be empty (and that is correct)
    result = run_estimation(dfm, use_case_fe=True, ref_badge="")
    table = _format_effect_rows(result)
    return table



def pretty_print(result: dict):
    print(f"n={result['n']} k={result['k']} ref_badge={result['ref_badge']} converged={result['converged']} pseudo_r2={result['pseudo_r2']:.4f}")
    if "ln_price_z" in result["params"]:
        print(f"ln_price_z beta={result['params']['ln_price_z']:.4f} se={result['bse']['ln_price_z']:.4f} p={result['pvalues']['ln_price_z']:.4g}")
    if "frame" in result["params"]:
        print(f"frame beta={result['params']['frame']:.4f} se={result['bse']['frame']:.4f} p={result['pvalues']['frame']:.4g}")
    for c in BADGE_COLS_CANON:
        if c in result["params"]:
            print(f"{c} beta={result['params'][c]:.4f} se={result['bse'][c]:.4f} p={result['pvalues'][c]:.4g}")
    if "sklearn_intercept_coef" in result:
        print(f"sklearn_intercept={result['sklearn_intercept_coef']:.4f} sklearn_n_iter={result['sklearn_n_iter']}")
    if "sklearn_note" in result:
        print(result["sklearn_note"])


def main(argv=None):
    argv = argv or sys.argv[1:]
    path = argv[0] if len(argv) > 0 else "results/df_choice.csv"
    use_case_fe = True if (len(argv) < 2 or argv[1].strip().lower() != "nofe") else False
    ref = argv[2].strip().lower() if len(argv) > 2 else ""
    df = pd.read_csv(path)
    if "case_id" not in df.columns:
        if "set_id" in df.columns:
            df["case_id"] = df["set_id"].astype(str)
        else:
            df["case_id"] = "S0001"
    result = run_estimation(df, use_case_fe=use_case_fe, ref_badge=ref)
    pretty_print(result)
    if len(argv) > 3:
        out_path = argv[3]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2))
        print(f"saved_json={out_path}")


if __name__ == "__main__":
    main()

