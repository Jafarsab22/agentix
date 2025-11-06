# save_to_agentix.py  —  Agentix DB + artifact persistence helper

import base64
import hashlib
import json
import os
import uuid
import math
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


def _extract_all_effects(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rows = results.get("logit_table_rows") or []
    for r in rows:
        if not isinstance(r, dict):
            continue
        badge = r.get("badge", "")
        if not str(badge).strip():
            continue
        eff: Dict[str, Any] = {
            "badge": badge,
            "beta": r.get("beta"),
            "se": r.get("se") or r.get("std_err") or r.get("stderr"),
            "p": r.get("p", r.get("p_value")),
            "q_bh": r.get("q_bh") or r.get("q") or r.get("qvalue") or r.get("q_value"),
            "odds_ratio": r.get("odds_ratio", r.get("or")),
            "ci_low": r.get("ci_low") or r.get("ci_l") or r.get("ci_lo") or r.get("cil"),
            "ci_high": r.get("ci_high") or r.get("ci_h") or r.get("ci_hi") or r.get("cih"),
            "ame_pp": r.get("ame_pp") or r.get("ame") or r.get("marginal_pp"),
            "evid_score": r.get("evid_score") or r.get("evidence") or r.get("evid"),
            "price_eq": r.get("price_eq") or r.get("lambda") or r.get("price_equiv") or r.get("price_equivalent"),
            "sign": r.get("sign", "0"),
        }
        out.append(eff)
    return out


def _has_any_significant(effects: List[Dict[str, Any]], alpha: float) -> bool:
    for r in effects:
        p = r.get("p")
        try:
            if p is not None and float(p) < alpha:
                return True
        except Exception:
            pass
    return False


def _sanitize_effect_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in row.items():
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                out[k] = None
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    print(f"[Agentix][POST] {url}")
    print(f"[Agentix][payload] {json.dumps(payload, ensure_ascii=False)[:800]}")
    resp = requests.post(url, json=payload, timeout=timeout)
    raw_text = resp.text
    print(f"[Agentix][resp {resp.status_code}] {raw_text[:800]}", flush=True)
    try:
        data = resp.json()
    except Exception:
        data = {"ok": False, "status": resp.status_code, "text": raw_text}
    if not data.get("ok", True):
        print(f"[Agentix][app-error] {data}", flush=True)
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


def _resolve(p: str | Path | None) -> str | None:
    if not p:
        return None
    try:
        rp = str(Path(p).resolve())
        return rp if Path(rp).exists() else None
    except Exception:
        return None


def _collect_artifact_paths(results: Dict[str, Any], payload: Dict[str, Any]) -> List[str]:
    a = (results.get("artifacts") or {}) if isinstance(results, dict) else {}

    keys_new = [
        "df_choice",
        "effects_csv",
        "badges_effects",
        "table_badges",
        "position_heatmap_empirical",
        "position_heatmap_png",
    ]
    keys_old = [
        "df_choice_path",
        "choice_csv_path",
    ]

    cands: List[str | None] = []
    for k in keys_new:
        cands.append(a.get(k))
    for k in keys_old:
        cands.append(results.get(k))
        cands.append(a.get(k))
    for k in ("effects_csv", "df_choice_path", "choice_csv_path"):
        cands.append(results.get(k))

    have_df = any(p and "df_choice" in str(p).lower() for p in cands if p)
    if not have_df:
        for pat in ("results/df_choice*.csv", "df_choice*.csv", "**/df_choice*.csv"):
            for fp in glob(pat, recursive=True):
                cands.append(fp)

    exts_ok = {".csv", ".png"}
    out: List[str] = []
    seen = set()
    for p in cands:
        rp = _resolve(p)
        if not rp:
            continue
        ext = Path(rp).suffix.lower()
        if ext not in exts_ok:
            continue
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _upload_artifacts(base_url: str, job_id: str, files: List[str]) -> Dict[str, Any]:
    packed = []
    for f in files:
        pair = _file_to_b64(f)
        if pair:
            fn, b64 = pair
            packed.append({"filename": fn, "data_base64": b64})
    if not packed:
        return {"ok": False, "reason": "no_files", "files_tried": files}

    url = f"{base_url.rstrip('/')}/sendAgentixFiles.php"
    if str(os.getenv("AGENTIX_DEBUG_FILES", "")).strip().lower() in ("1", "true", "yes"):
        url += "?debug=1"

    res = _post_json(url, {"run_id": job_id, "job_id": job_id, "files": packed})
    res["files_tried"] = files
    return res


# ---------- main entry ----------

def persist_results_if_qualify(
    results: Dict[str, Any],
    payload: Dict[str, Any],
    base_url: str = "https://aireadyworkforce.pro/Agentix",
    app_version: str = "Agentix v1.6",
    est_model: str = "clogit_lnprice",
    alpha: float = 0.05,
) -> Dict[str, Any]:

    job_id = results.get("job_id") or payload.get("job_id") or f"job-{uuid.uuid4()}"
    ts_utc = _to_sql_ts(results.get("ts") or payload.get("ts"))

    product = payload.get("product") or ""
    brand = payload.get("brand") or payload.get("brand_type") or ""
    model = payload.get("model") or payload.get("model_name") or ""

    try:
        price_value = float(payload.get("price") or payload.get("price_value") or 0)
    except Exception:
        price_value = 0.0
    currency = payload.get("currency") or payload.get("price_currency") or ""
    n_iterations = int(payload.get("n_iterations") or 0)
    badges = payload.get("badges") or []

    # extract → sanitize
    all_effects_raw = _extract_all_effects(results)
    all_effects = [_sanitize_effect_row(r) for r in all_effects_raw]
    has_sig = _has_any_significant(all_effects, alpha=alpha)

    # this is your qualifying flag
    qualifies = bool(has_sig or n_iterations >= 2)

    # upload artifacts
    try:
        artifact_paths = _collect_artifact_paths(results, payload)
        upload_info = _upload_artifacts(base_url, job_id, artifact_paths)
    except Exception as e:
        upload_info = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # run document
    run_doc = {
        "job_id": job_id,
        "ts_utc": ts_utc,
        "product": product,
        "brand": brand,
        "model": model,
        "price_value": float(price_value),
        "currency": currency,
        "n_iterations": n_iterations,
        "badges_csv": ",".join(sorted([str(b).strip() for b in badges if str(b).strip()])),
        "frame_scheme": payload.get("frame_scheme") or "standard",
        "has_significant": 1 if has_sig else 0,
        "n_sig_badges": sum(1 for r in all_effects if (r.get("p") is not None and float(r["p"]) < alpha)),
        "app_version": app_version,
        "runner_version": payload.get("runner_version") or app_version,
        "est_model": est_model,
    }

    runs_url = f"{base_url.rstrip('/')}/sendAgentixRuns.php"
    effects_url = f"{base_url.rstrip('/')}/sendAgentixEffects.php"

    # 1) insert/upsert run
    run_res = _post_json(runs_url, run_doc)
    print(f"[Agentix] run_res = {json.dumps(run_res, ensure_ascii=False)}", flush=True)
    if not run_res.get("ok", False):
        return {
            "stored": False,
            "job_id": job_id,
            "run_response": run_res,
            "files_response": upload_info,
            "reason": "run_insert_failed",
        }

    # 2) always send effects (sanitized) so we see errors in logs
    eff_res = {"ok": True, "rows_upserted": 0}
    if all_effects:
        effects_payload = {"job_id": job_id, "effects": all_effects}
        eff_res = _post_json(effects_url, effects_payload)
        print(f"[Agentix] effects_res = {json.dumps(eff_res, ensure_ascii=False)}", flush=True)

    return {
        "stored": True,
        "job_id": job_id,
        "run_response": run_res,
        "effects_response": eff_res,
        "files_response": upload_info,
        "has_significant": has_sig,
        "qualified_by_rules": qualifies,
        "n_iterations": n_iterations,
        "effects_count": len(all_effects),
    }
