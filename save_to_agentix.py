import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

import requests


class AgentixSaverError(Exception):
    pass


def _now_sql_utc() -> str:
    """UTC timestamp formatted for MySQL DATETIME (no T/Z)."""
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


def _extract_significant_effects(results: Dict[str, Any], alpha: float = 0.05) -> List[Dict[str, Any]]:
    rows = results.get("logit_table_rows") or []
    sig_rows: List[Dict[str, Any]] = []
    for r in rows:
        p_raw = r.get("p", r.get("p_value"))
        try:
            p = float(p_raw) if p_raw is not None else None
        except Exception:
            p = None
        if p is not None and p < alpha:
            sig_rows.append(
                {
                    "badge": r.get("badge", ""),
                    "beta": r.get("beta", None),
                    "p_value": p,
                    "sign": r.get("sign", "0"),
                }
            )
    return sig_rows


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"ok": False, "status": resp.status_code, "text": resp.text}
    if resp.status_code >= 400:
        raise AgentixSaverError(f"HTTP {resp.status_code} from {url}: {data}")
    return data


def persist_results_if_qualify(
    results: Dict[str, Any],
    payload: Dict[str, Any],
    base_url: str = "https://aireadyworkforce.pro/Agentix",
    app_version: str = "Agentix v1.6",
    est_model: str = "clogit_lnprice",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    # SQL-safe UTC timestamp; prefer payload['ts'] iff it is already SQL-style
    ts = payload.get("ts")
    if isinstance(ts, str) and "T" not in ts:
        ts_utc = ts
    else:
        ts_utc = _now_sql_utc()

    base = (payload.get("job_id") or "run")[:40]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    run_id = f"{base}-n{int(payload.get('n_iterations') or 0)}-{stamp}"

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

    sig_effects = _extract_significant_effects(results, alpha=alpha)
    has_sig = len(sig_effects) > 0
    n_sig = len(sig_effects)

    # Store only if: any significant badge OR (no significant badge AND n_iterations >= 250)
    if not (has_sig or (not has_sig and n_iterations >= 250)):
        return {
            "stored": False,
            "reason": "does_not_meet_persistence_criteria",
            "has_significant": has_sig,
            "n_sig_badges": n_sig,
            "n_iterations": n_iterations,
        }

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
        "n_sig_badges": n_sig,
        "est_model": est_model,
        "app_version": app_version,
        "norm_key": norm_key,
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
            "n_sig_badges": n_sig,
            "n_iterations": n_iterations,
        }

    eff_res = {"ok": True, "rows_upserted": 0}
    if has_sig:
        effects_payload = {"run_id": run_id, "effects": sig_effects}
        eff_res = _post_json(effects_url, effects_payload)
        if not eff_res.get("ok", False):
            raise AgentixSaverError(f"sendAgentixEffects failed: {eff_res}")

    return {
        "stored": True,
        "run_response": run_res,
        "effects_response": eff_res,
        "has_significant": has_sig,
        "n_sig_badges": n_sig,
        "n_iterations": n_iterations,
    }
