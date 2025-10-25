# save_to_agentix.py  —  Agentix DB persistence helper
# (2025-10-24) Patch: stable run_id, full effects payload, robust ts normalization.

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

import requests


class AgentixSaverError(Exception):
    pass


# ---------- small utils ----------

def _to_sql_ts(ts: str | None) -> str:
    """
    Normalize various incoming timestamps to MySQL DATETIME 'YYYY-MM-DD HH:MM:SS'.
    Accepts ISO with T/Z or SQL-like. Falls back to current UTC if parsing fails.
    """
    if not ts or not isinstance(ts, str):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    t = ts.strip()
    # common shapes: "2025-10-24T17:36:56.553569Z", "2025-10-24 17:36:56"
    try:
        if "T" in t:
            # drop trailing Z if present, replace T with space
            t = t.replace("Z", "").replace("z", "").replace("T", " ")
        # trim microseconds if present
        if "." in t:
            t = t.split(".", 1)[0]
        # validate by parsing
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _norm_text(s: str) -> str:
    return (s or "").strip().lower()


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
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _extract_all_effects(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return all badge rows (badge, beta, p_value, sign) from results['logit_table_rows'].
    No alpha filtering here; persistence policy is applied outside.
    """
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
        sign = r.get("sign", "0")
        out.append({"badge": badge, "beta": beta, "p_value": p_value, "sign": sign})
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


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"ok": False, "status": resp.status_code, "text": resp.text}
    if resp.status_code >= 400:
        raise AgentixSaverError(f"HTTP {resp.status_code} from {url}: {data}")
    return data


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
    Persistence policy:
      • Persist when (any badge significant at alpha) OR (n_iterations >= 250).
      • When persisting, send ALL effects rows for archival completeness.
    Run identity:
      • run_id := results['job_id'] (preferred)  else payload['job_id'] else random UUID.
      • ts_utc  := normalized results['ts'] (preferred) else payload['ts'] else now().
    """

    # ---- identify the run consistently with what the UI shows ----
    run_id = results.get("job_id") or payload.get("job_id") or f"run-{uuid.uuid4()}"
    ts_utc = _to_sql_ts(results.get("ts") or payload.get("ts"))

    # ---- inputs (for agentix_runs) ----
    product = payload.get("product") or ""
    brand_type = payload.get("brand") or ""
    model_name = payload.get("model") or ""
    try:
        price_value = float(payload.get("price") or 0)
    except Exception:
        price_value = 0.0
    price_currency = payload.get("currency") or ""
    n_iterations = int(payload.get("n_iterations") or 0)
    badges = payload.get("badges") or []

    # ---- effects & policy ----
    all_effects = _extract_all_effects(results)
    has_sig = _has_any_significant(all_effects, alpha=alpha)

    if not (has_sig or n_iterations >= 250):
        return {
            "stored": False,
            "reason": "does_not_meet_persistence_criteria",
            "has_significant": has_sig,
            "n_iterations": n_iterations,
            "effects_count": len(all_effects),
        }

    # ---- norm key (canonization) ----
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
        "n_iterations": n_iterations,
        "badges_csv": ",".join(sorted([str(b).strip() for b in badges if str(b).strip()])),
        "has_significant": 1 if has_sig else 0,
        "n_sig_badges": sum(1 for r in all_effects if (r.get("p_value") is not None and float(r["p_value"]) < alpha)),
        "est_model": est_model,
        "app_version": app_version,
        "norm_key": norm_key,
        "superseded_by": None,
        "approved_for_reuse": 1,
    }

    runs_url = f"{base_url.rstrip('/')}/sendAgentixRuns.php"
    effects_url = f"{base_url.rstrip('/')}/sendAgentixEffects.php"

    # ---- 1) persist the run (server enforces canonization/supersede) ----
    run_res = _post_json(runs_url, run_doc)
    if not run_res.get("ok", False):
        raise AgentixSaverError(f"sendAgentixRuns failed: {run_res}")

    # If server returns stored:false due to policy, stop here (nothing else to send)
    if run_res.get("stored") is False:
        return {
            "stored": False,
            "reason": run_res.get("reason", "unknown"),
            "has_significant": has_sig,
            "n_iterations": n_iterations,
            "effects_count": len(all_effects),
            "run_id": run_id,
        }

    # ---- 2) persist the effects (send ALL rows for the archived run) ----
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
        "has_significant": has_sig,
        "n_iterations": n_iterations,
        "effects_count": len(all_effects),
    }
