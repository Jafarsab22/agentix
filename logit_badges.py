# logit_badges_v2.py
# Robust badge-effects estimator for Agentix v1.7 data
# - K−1 reference coding for non-frame badges (deterministic baseline)
# - Uses ln_price_z (z-scored log price) as the single price regressor
# - Includes screen fixed effects via C(case_id) only (omit product/title FE)
# - Offers L2-regularised logistic as a stability check
# - Single-line prints only

import sys
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Tuple
from sklearn.linear_model import LogisticRegression

BADGE_COLS_CANON = [
    "assurance","scarcity","strike","timer","social_proof","voucher","bundle"
]


def _pick_present_badges(df: pd.DataFrame) -> List[str]:
    return [c for c in BADGE_COLS_CANON if c in df.columns]


def _choose_reference_badge(present_badges: List[str]) -> str:
    # Deterministic ref: alphabetically first among present badges
    if not present_badges:
        return ""
    return sorted(present_badges)[0]


def _build_design(df: pd.DataFrame, use_case_fe: bool = True, ref_badge: str = "") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if "chosen" not in df.columns:
        raise ValueError("Dataframe must contain 'chosen' column (0/1).")
    if "ln_price" not in df.columns:
        raise ValueError("Dataframe must contain 'ln_price' column.")
    X_parts = []
    # Price regressor (z-scored)
    df = df.copy()
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
        if len(keep) == 0:
            # Only one badge observed; include none to avoid collinearity
            keep = []
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
    features = X.columns.tolist()
    return X, y, ["ln_price_z","frame"] + [c for c in keep]


def fit_logit(X: pd.DataFrame, y: pd.Series):
    X_sm = sm.add_constant(X, has_constant='add')
    model = sm.Logit(y, X_sm)
    res = model.fit(disp=False, maxiter=500)
    return res


def fit_logit_l2(X: pd.DataFrame, y: pd.Series):
    clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000, fit_intercept=True)
    clf.fit(X, y)
    return clf


def run_estimation(df: pd.DataFrame, use_case_fe: bool = True, ref_badge: str = "") -> dict:
    present = _pick_present_badges(df)
    base = ref_badge or _choose_reference_badge(present)
    X, y, feat_order = _build_design(df, use_case_fe=use_case_fe, ref_badge=base)
    res = fit_logit(X, y)
    try:
        clf = fit_logit_l2(X, y)
    except Exception:
        clf = None
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
    return out


def pretty_print(result: dict):
    print(f"n={result['n']} k={result['k']} ref_badge={result['ref_badge']} converged={result['converged']} pseudo_r2={result['pseudo_r2']:.4f}")
    if 'ln_price_z' in result['params']:
        print(f"ln_price_z beta={result['params']['ln_price_z']:.4f} se={result['bse']['ln_price_z']:.4f} p={result['pvalues']['ln_price_z']:.4g}")
    if 'frame' in result['params']:
        print(f"frame beta={result['params']['frame']:.4f} se={result['bse']['frame']:.4f} p={result['pvalues']['frame']:.4g}")
    for c in BADGE_COLS_CANON:
        if c in result['params']:
            print(f"{c} beta={result['params'][c]:.4f} se={result['bse'][c]:.4f} p={result['pvalues'][c]:.4g}")
    if 'sklearn_intercept_coef' in result:
        print(f"sklearn_intercept={result['sklearn_intercept_coef']:.4f} sklearn_n_iter={result['sklearn_n_iter']}")


def main(argv=None):
    argv = argv or sys.argv[1:]
    path = argv[0] if len(argv) > 0 else "results/df_choice.csv"
    use_case_fe = True if (len(argv) < 2 or argv[1].strip().lower() != "nofe") else False
    ref = argv[2].strip().lower() if len(argv) > 2 else ""
    df = pd.read_csv(path)
    # Derive case_id if not present; fallback to set_id
    if "case_id" not in df.columns:
        if "set_id" in df.columns:
            df["case_id"] = df["set_id"].astype(str)
        else:
            df["case_id"] = "S0001"
    result = run_estimation(df, use_case_fe=use_case_fe, ref_badge=ref)
    pretty_print(result)
    # Optional JSON dump if user passes a 4th argument
    if len(argv) > 3:
        out_path = argv[3]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=2))
        print(f"saved_json={out_path}")


if __name__ == "__main__":
    main()
