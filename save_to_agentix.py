# save_to_agentix.py  —  Agentix DB + artifact persistence helper
# v1.10 (2025-11-02)
#   • Always upload artifacts (df_choice.csv, badges_effects.csv, heatmap PNG)
#     to Hostinger even when a run does not meet DB persistence criteria.
#   • DB persistence policy unchanged (requires any significant badge OR
#     n_iterations ≥ 250).
#   • New: robust df_choice.csv discovery — if the path is not present in
#     results/artifacts, we search common locations for the latest
#     df_choice*.csv and upload it.

import base64
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from glob import glob

import requests


class AgentixSaverError(Exception):
    pass


# ---------- small utils ----------

def _to_sql_ts(ts: str | None) -> str:
    if not ts or not isinstance(ts, str):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    t = ts.strip()
    try:
        if "T" in t:
            t = t.replace("Z", "").replace("z", "").replace("T", " ")
        if "." in t:
            t = t.split(".", 1)[0]
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm_text(s: str) -> str:
    return (s or "").strip().lower()


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _make_norm_key(
    product: str,
    brand_type: str,
    model_name: str,
    badges: List[str],
    price_value,
    price_currency: str,
    est_model: str,
) -> str:
    badges_sorted = ",".join(sorted([_norm_text(b) for b in (badges or []) if _norm_text(b)]))
    price_bucket = f"{float(price_value):.2f}"
    basis = "|".join(
        [
            _norm_text(product),
            _norm_text(brand_type),
            _norm_text(model_name),
            badges_sorted,
            _norm_text(price_currency),
            price_bucket,
            _norm_text(est_model),
        ]
    )
    return _sha256_hex(basis)


def _extract_all_effects(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in results.get("logit_table_rows") or []:
        badge = r.get("badge", "")
        if not str(badge).strip():
            continue
        beta = r.get("beta", None)
        p_raw = r.get("p", r.get("p_value"))
        try:
            p_value = float(p_raw) if p_raw is not None else None
        except Exception:
            p_value = None
        row = {
            "badge": badge,
            "beta": beta,
            "p_value": p_value,
            "sign": r.get("sign", "0"),
        }
        for k in ("se", "ci_low", "ci_high"):
            if k in r:
                try:
                    row[k] = float(r[k])
                except Exception:
                    pass
        out.append(row)
    return out


def _has_any_significant(effects: List[Dict[str, Any]], alpha: float) -> bool:
    for r in effects:
        p = r.get("p_value")
        try:
            if p is not None and float(p) < alpha:
                return True
        except Exception:
            pass
    return False


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"ok": False, "status": resp.status_code, "text": resp.text}
    if resp.status_code >= 400:
        raise AgentixSaverError(f"HTTP {resp.status_code} from {url}: {data}")
    return data


# ---------- artifact upload helpers ----------

def _file_to_b64(path: str | Path) -> tuple[str, str] | None:
    try:
        p = Path(path)
        if not p.exists() or p.is_dir():
            return None
        raw = p.read_bytes()
        return p.name, base64.b64encode(raw).decode("utf-8")
    except Exception:
        return None


def _collect_artifact_paths(results: Dict[str, Any], payload: Dict[str, Any]) -> List[str]:
    a = results.get("artifacts", {})
    cands = [
        results.get("df_choice_path"),
        a.get("df_choice_path"),
        results.get("choice_csv_path"),
        a.get("choice_csv_path"),
        a.get("effects_csv"),
        a.get("position_heatmap_empirical"),
    ]

    # If df_choice path is missing, search common locations for the latest df_choice*.csv
    have_df = any(p and str(p).lower().endswith('.csv') and 'df_choice' in str(p).lower() for p in cands)
    if not have_df:
        patterns = ['df_choice*.csv', 'results/df_choice*.csv', '**/df_choice*.csv']
        latest = None
        latest_mtime = -1
        for pat in patterns:
            for fp in glob(pat, recursive=True):
                try:
                    mt = Path(fp).stat().st_mtime
                    if mt > latest_mtime:
                        latest, latest_mtime = fp, mt
                except Exception:
                    pass
        if latest:
            cands.insert(0, latest)

    return [str(p) for p in cands if p]


def _upload_artifacts(base_url: str, run_id: str, files: List[str]) -> Dict[str, Any]:
    packed = []
    for f in files:
        pair = _file_to_b64(f)
        if pair:
            fn, b64 = pair
            packed.append({"filename": fn, "data_base64": b64})
    if not packed:
        return {"ok": False, "reason": "no_files"}
    url = f"{base_url.rstrip('/')}/sendAgentixFiles.php"
    return _post_json(url, {"run_id": run_id, "files": packed})


# ---------- main entry ----------

def persist_results_if_qualify(
    results: Dict[str, Any],
    payload: Dict[str, Any],
    base_url: str = "https://aireadyworkforce.pro/Agentix",
    app_version: str = "Agentix v1.6",
    est_model: str = "clogit_lnprice",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Persist the run + effects to Hostinger and (always) upload artifacts to
    /Agentix/Results using sendAgentixFiles.php so they are downloadable at:
        https://aireadyworkforce.pro/Agentix/Results/<run_id>_...
    """

    run_id = results.get("job_id") or payload.get("job_id") or f"run-{uuid.uuid4()}"
    ts_utc = _to_sql_ts(results.get("ts") or payload.get("ts"))

    product = payload.get("product") or ""
    brand_type = payload.get("brand") or payload.get("brand_type") or ""
    model_name = payload.get("model") or payload.get("model_name") or ""
    try:
        price_value = float(payload.get("price") or payload.get("price_value") or 0)
    except Exception:
        price_value = 0.0
    price_currency = payload.get("currency") or payload.get("price_currency") or ""
    n_iterations = int(payload.get("n_iterations") or 0)
    badges = payload.get("badges") or []

    all_effects = _extract_all_effects(results)
    has_sig = _has_any_significant(all_effects, alpha=alpha)

    # Always attempt to upload artifacts (even if we don't persist to DB)
    upload_info = None
    try:
        artifact_paths = _collect_artifact_paths(results, payload)
        if artifact_paths:
            upload_info = _upload_artifacts(base_url, run_id, artifact_paths)
    except Exception as e:
        upload_info = {"ok": False, "error": str(e)}

    # If run doesn't meet policy, return early but include upload_info
    if not (has_sig or n_iterations >= 250):
        return {
            "stored": False,
            "reason": "does_not_meet_persistence_criteria",
            "has_significant": has_sig,
            "n_iterations": n_iterations,
            "effects_count": len(all_effects),
            "run_id": run_id,
            "files_response": upload_info,
        }

    # Defaults for NOT NULL columns on agentix_runs (kept for backward compat)
    price_anchor_bucket = payload.get("price_anchor_bucket") or f"{price_value:.2f}"
    frame_scheme = payload.get("frame_scheme") or ("all_in" if any(_norm_text(b) == "frame" for b in badges) else "standard")
    badges_sorted = ",".join(sorted([_norm_text(b) for b in (badges or []) if _norm_text(b)]))
    badge_set_hash = payload.get("badge_set_hash") or _sha256_hex(badges_sorted)
    seed_in = payload.get("random_seed") or payload.get("seed")
    try:
        random_seed = int(seed_in) if seed_in is not None else int(int(_sha256_hex(run_id), 16) % (2**63 - 1))
    except Exception:
        random_seed = int(int(_sha256_hex(run_id), 16) % (2**63 - 1))
    spec_str = json.dumps({
        "est_model": est_model,
        "cluster": results.get("vcov_type") or "case_cluster",
        "ridge_alpha": payload.get("ridge_alpha"),
        "max_iter": payload.get("max_iter", 500),
        "features": results.get("features"),
    }, sort_keys=True, default=str)
    est_spec_hash = payload.get("est_spec_hash") or _sha256_hex(spec_str)
    runner_version = payload.get("runner_version") or app_version

    n_cards = results.get("n_cards") or payload.get("n_cards")
    n_users = results.get("n_users") or payload.get("n_users")
    balance_pvalue = results.get("balance_pvalue")

    norm_key = _make_norm_key(
        product, brand_type, model_name, badges, price_value, price_currency, est_model
    )

    run_doc = {
        "run_id": run_id,
        "ts_utc": ts_utc,
        "product": product,
        "brand_type": brand_type,
        "model_name": model_name,
        "price_value": float(price_value),
        "price_currency": price_currency,
        "price_anchor_bucket": price_anchor_bucket,
        "n_iterations": n_iterations,
        "n_cards": n_cards,
        "n_users": n_users,
        "badges_csv": ",".join(sorted([str(b).strip() for b in badges if str(b).strip()])),
        "frame_scheme": frame_scheme,
        "badge_set_hash": badge_set_hash,
        "random_seed": random_seed,
        "has_significant": 1 if has_sig else 0,
        "n_sig_badges": sum(1 for r in all_effects if (r.get("p_value") is not None and float(r["p_value"]) < alpha)),
        "est_model": est_model,
        "est_spec_hash": est_spec_hash,
        "app_version": app_version,
        "runner_version": runner_version,
        "norm_key": norm_key,
        "qc_pass": results.get("qc_pass", 1),
        "balance_pvalue": balance_pvalue,
        "superseded_by": None,
        "approved_for_reuse": 1,
    }

    runs_url = f"{base_url.rstrip('/')}/sendAgentixRuns.php"
    effects_url = f"{base_url.rstrip('/')}/sendAgentixEffects.php"

    run_res = _post_json(runs_url, run_doc)
    if not run_res.get("ok", False):
        raise AgentixSaverError(f"sendAgentixRuns failed: {run_res}")

    if run_res.get("stored") is False:
        return {
            "stored": False,
            "reason": run_res.get("reason", "unknown"),
            "has_significant": has_sig,
            "n_iterations": n_iterations,
            "effects_count": len(all_effects),
            "run_id": run_id,
            "run_response": run_res,
            "files_response": upload_info,
        }

    eff_res = {"ok": True, "rows_upserted": 0}
    if all_effects:
        effects_payload = {"run_id": run_id, "effects": all_effects}
        eff_res = _post_json(effects_url, effects_payload)
        if not eff_res.get("ok", False):
            raise AgentixSaverError(f"sendAgentixEffects failed: {eff_res}")

    return {
        "stored": True,
        "run_id": run_id,
        "run_response": run_res,
        "effects_response": eff_res,
        "files_response": upload_info,
        "has_significant": has_sig,
        "n_iterations": n_iterations,
        "effects_count": len(all_effects),
    }
