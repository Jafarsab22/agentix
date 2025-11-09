# app.py
import json, uuid, os, time, pathlib, csv, requests
from datetime import datetime
import gradio as gr
from agent_runner import run_job_sync
from agent_runner import submit_job_async as runner_submit_job_async
from agent_runner import poll_job as runner_poll_job
from agent_runner import fetch_job as runner_fetch_job
import traceback, logging
from urllib.parse import quote

# NEW: delegate live A/B to the dedicated module
from ABTesting import submit_live_ab, poll_live_ab, fetch_live_ab, cancel_live_ab

logging.basicConfig(level=logging.INFO)

RESULTS_DIR = pathlib.Path("results"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = pathlib.Path("jobs"); JOBS_DIR.mkdir(parents=True, exist_ok=True)
EFFECTS_DIR = RESULTS_DIR / "effects"; EFFECTS_DIR.mkdir(parents=True, exist_ok=True)

# URL of your PHP endpoint that prints JSON from agentix_cross_parameters
CROSS_PARAMS_URL = os.getenv(
    "AGENTIX_CROSS_PARAMS_URL",
    "https://aireadyworkforce.pro/Agentix/getCrossParameters.php",
)

# --- replacement parser (single source of truth) ---
def load_params_from_php(url: str = CROSS_PARAMS_URL):
    """
    Returns dict keyed by cue/badge with numeric values.
    Accepts either:
      A) {"ok": true, "model": "...", "params": { "<badge>": {"beta":..., "M":..., "C":..., "R":..., "s":..., "price_weight":...}, ...}}
      B) [ {"badge":"...", "beta":..., "m_val":..., "c_val":..., "r_val":..., "price_weight": ...}, ... ]
    """
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    data = r.json()

    # A: object with "params"
    if isinstance(data, dict) and isinstance(data.get("params"), dict):
        out = {}
        for badge, vals in data["params"].items():
            if not badge or not isinstance(vals, dict):
                continue
            out[str(badge)] = {
                "beta": float(vals.get("beta", 0.0)),
                "M": float(vals.get("M", 0.0)),
                "C": float(vals.get("C", 0.0)),
                "R": float(vals.get("R", 0.0)),
            }
            if vals.get("price_weight") is not None:
                out[str(badge)]["price_weight"] = float(vals["price_weight"])
        return out

    # B: legacy list rows
    if isinstance(data, list):
        out = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            badge = row.get("badge")
            if not badge:
                continue
            out[str(badge)] = {
                "beta": float(row.get("beta") or 0.0),
                "M": float(row.get("m_val") or 0.0),
                "C": float(row.get("c_val") or 0.0),
                "R": float(row.get("r_val") or 0.0),
            }
            if row.get("price_weight") is not None:
                out[str(badge)]["price_weight"] = float(row["price_weight"])
        return out

    raise ValueError("Unexpected JSON shape from cross-parameters endpoint")

# load the parameters (do this once)
try:
    SCORE_PARAMS = load_params_from_php()
    logging.info("Loaded %d cue parameters from PHP.", len(SCORE_PARAMS or {}))
except Exception as e:
    logging.exception("Could not load cross parameters from PHP")
    SCORE_PARAMS = None

# Optional storefront helpers
try:
    from storefront import build_storefront_from_payload, save_storefront
except Exception:
    build_storefront_from_payload = None
    save_storefront = None

# -------- UI choices --------
MODEL_CHOICES = ["GPT-4.1-mini"]  # extend later if needed
BADGE_CHOICES = [
    "All-in v. partitioned pricing",
    "Assurance",
    "Scarcity tag",
    "Strike-through",
    "Timer",
    "social",
    "voucher",
    "bundle",
]
CURRENCY_CHOICES = ["£", "$", "EUR"]

# Subset of badges estimated separately in the logit (used to compute B)
SEPARATE_BADGES = {
    "All-in v. partitioned pricing",
    "Assurance",
    "Strike-through",   # “Strike”
    "Timer",
}

# Admin secret (set in platform env)
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")

# ---------- Renderer selection ----------
RENDER_URL_TPL = os.environ.get("RENDER_URL_TPL", "")  # empty → inline HTML in agent_runner

# ---------- helpers ----------

def _catch_and_report(fn):
    """Wrap a Gradio handler, show a readable error, and log to results/ + console."""
    def _inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            err_path = RESULTS_DIR / f"ui_error_{int(time.time())}.log"
            try:
                err_path.write_text(tb, encoding="utf-8")
            except Exception:
                pass
            print(tb, flush=True)
            logging.exception("Gradio handler failed")
            msg = f"❌ {type(e).__name__}: {e}\n\n```\n{tb}\n```"
            return (msg, "{}") if fn.__name__ in ("run_now", "fetch_job_ui") else msg
    return _inner

def _ceil_to_8(n: int) -> int:
    return int(((int(n) + 7) // 8) * 8)

def _auto_iterations_from_badges(badges: list[str]) -> int:
    sel = set(badges or [])
    b = 0
    if "All-in v. partitioned pricing" in sel:
        b += 1
    for lab in ("Assurance", "Strike-through", "Timer"):
        if lab in sel:
            b += 1
    base = max(100, 30 * b)
    return _ceil_to_8(base)

def _validate_inputs(product_name, price, currency, n_iterations):
    if not product_name or not product_name.strip():
        return "Please enter a product name."
    try:
        price_val = float(price)
        if price_val < 0:
            return "Please enter a valid non-negative price."
    except Exception:
        return "Please enter a valid non-negative price."
    if currency not in CURRENCY_CHOICES:
        return "Please choose a currency."
    try:
        n_iter_val = int(n_iterations) if n_iterations is not None else 50
        if n_iter_val <= 0:
            return "Iterations must be positive."
    except Exception:
        return "Please enter a valid integer for iterations."
    return ""

def _build_payload(*, job_id, product, brand, model, badges, price, currency, n_iterations, fresh=True):
    csv_badges = ",".join([b.strip() for b in (badges or []) if str(b).strip()])
    tpl = (RENDER_URL_TPL or "").strip()
    if not tpl:
        render_url = ""
    else:
        render_url = (tpl
            .replace("{csv}", csv_badges)
            .replace("{price}", str(float(price) if price is not None else 0.0))
            .replace("{currency}", str(currency or "")))
    return {
        "job_id": job_id,
        "ts": datetime.utcnow().isoformat() + "Z",
        "product": (product or "").strip(),
        "brand": (brand or "").strip(),
        "model": model,
        "badges": badges or [],
        "price": float(price) if price is not None else 0.0,
        "currency": currency,
        "n_iterations": int(n_iterations) if n_iterations else 50,
        "fresh": bool(fresh),
        "catalog_seed": 777,
        "render_url": render_url,
    }

# --- helper to export effects with run metadata (used for local download only) ---
def _export_badge_effects(rows_sorted: list[dict], payload: dict, job_id: str):
    if not rows_sorted:
        return None, None
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    base = f"{ts}_badge_effects"
    csv_path = EFFECTS_DIR / f"{base}.csv"
    html_path = EFFECTS_DIR / f"{base}.html"
    fieldnames = [
        "job_id", "timestamp", "product", "brand", "model",
        "price", "currency", "n_iterations",
        "badge", "beta", "se", "p", "q_bh",
        "odds_ratio", "ci_low", "ci_high", "ame_pp",
        "evid_score", "price_eq", "sign"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_sorted:
            w.writerow({
                "job_id": job_id,
                "timestamp": payload.get("ts", ""),
                "product": payload.get("product", ""),
                "brand": payload.get("brand", ""),
                "model": payload.get("model", ""),
                "price": payload.get("price", ""),
                "currency": payload.get("currency", ""),
                "n_iterations": payload.get("n_iterations", ""),
                "badge": r.get("badge", ""),
                "beta": r.get("beta", ""),
                "se": r.get("se", ""),
                "p": r.get("p", r.get("p_value", "")),
                "q_bh": r.get("q_bh", ""),
                "odds_ratio": r.get("odds_ratio", ""),
                "ci_low": r.get("ci_low", ""),
                "ci_high": r.get("ci_high", ""),
                "ame_pp": r.get("ame_pp", ""),
                "evid_score": r.get("evid_score", ""),
                "price_eq": r.get("price_eq", ""),
                "sign": r.get("sign", "0"),
            })
    def _fmt(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "—"
    meta_rows = [
        ("Product", payload.get("product", "")),
        ("Brand / Type", payload.get("brand", "")),
        ("Model", payload.get("model", "")),
        ("Price", f"{payload.get('price','')} {payload.get('currency','')}".strip()),
        ("Iterations", str(payload.get("n_iterations", ""))),
        ("Job ID", job_id),
        ("Timestamp", payload.get("ts", "")),
    ]
    parts = [
        "<html><head><meta charset='utf-8'><title>Estimates of the Conditional Logit Regression</title>",
        "<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;padding:16px}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:8px}"
        "th{text-align:left;background:#f6f6f6}"
        "td.num{text-align:right}</style>",
        "</head><body>",
        "<h2>Estimates of the Conditional Logit Regression</h2>",
        "<h3>Run metadata</h3><table>",
    ]
    for k, v in meta_rows:
        parts.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
    parts.append("</table>")
    parts.append("<h3 style='margin-top:18px'>Effects table</h3>")
    parts.append("<table>")
    parts.append(
        "<tr>"
        "<th>Badge</th><th>β</th><th>SE</th><th>p</th><th>q_bh</th>"
        "<th>Odds ratio</th><th>CI low</th><th>CI high</th>"
        "<th>AME (pp)</th><th>Evidence</th><th>Price-eq λ</th>"
        "<th>Effect</th>"
        "</tr>"
    )
    for r in rows_sorted:
        parts.append(
            "<tr>"
            f"<td>{r.get('badge','')}</td>"
            f"<td class='num'>{_fmt(r.get('beta'))}</td>"
            f"<td class='num'>{_fmt(r.get('se'))}</td>"
            f"<td class='num'>{_fmt(r.get('p', r.get('p_value')))}</td>"
            f"<td class='num'>{_fmt(r.get('q_bh'))}</td>"
            f"<td class='num'>{_fmt(r.get('odds_ratio'))}</td>"
            f"<td class='num'>{_fmt(r.get('ci_low'))}</td>"
            f"<td class='num'>{_fmt(r.get('ci_high'))}</td>"
            f"<td class='num'>{_fmt(r.get('ame_pp'))}</td>"
            f"<td class='num'>{_fmt(r.get('evid_score'))}</td>"
            f"<td class='num'>{_fmt(r.get('price_eq'))}</td>"
            f"<td style='text-align:center'>{r.get('sign','0')}</td>"
            "</tr>"
        )
    parts.append("</table></body></html>")
    html_path.write_text("".join(parts), encoding="utf-8")
    return str(csv_path), str(html_path)

# ---------- Shared formatter for sync + async paths ----------
def _format_results_from_dict(results: dict) -> tuple[str, dict]:
    rows = results.get("logit_table_rows") or []
    inputs = results.get("inputs") or {}
    model_name = str(results.get("model_requested") or inputs.get("model") or "model")
    product_name = str(inputs.get("product") or "")
    artifacts = results.setdefault("artifacts", {}) or {}

    def _fmt(x, nd=3):
        try:
            v = float(x)
            if v != v:
                return "—"
            return f"{v:.{nd}f}"
        except Exception:
            return "—"

    def _effect_symbol(s):
        s = (s or "0").strip()
        if s in ("↑", "+"):
            return "+"
        if s in ("↓", "-"):
            return "-"
        return "0"

    sec_map = {"Position effects": [], "Badge/lever effects": [], "Attribute effects": []}
    for r in rows:
        sec = r.get("section")
        b = str(r.get("badge", ""))
        if not sec:
            if b in ("Row 1", "Column 1", "Column 2", "Column 3"):
                sec = "Position effects"
            elif b == "ln(price)":
                sec = "Attribute effects"
            else:
                sec = "Badge/lever effects"
        sec_map.setdefault(sec, [])
        sec_map[sec].append(r)

    order_pos = {"Row 1": 0, "Column 1": 1, "Column 2": 2, "Column 3": 3}
    order_attr = {"ln(price)": 0}

    def _render_section(title, rlist):
        if not rlist:
            return ""
        if title == "Position effects":
            rlist = sorted(rlist, key=lambda r: order_pos.get(str(r.get("badge", "")), 99))
        elif title == "Attribute effects":
            rlist = sorted(rlist, key=lambda r: order_attr.get(str(r.get("badge", "")), 99))
        lines = [
            f"\n\n{title}\n",
            "| Badge | β | SE | p | q_bh | Odds ratio | CI low | CI high | AME (pp) | Evidence | Price-eq λ | Effect |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for r in rlist:
            lines.append(
                f"| {r.get('badge','')} | "
                f"{_fmt(r.get('beta'))} | "
                f"{_fmt(r.get('se'))} | "
                f"{_fmt(r.get('p', r.get('p_value')), nd=4)} | "
                f"{_fmt(r.get('q_bh'), nd=4)} | "
                f"{_fmt(r.get('odds_ratio'))} | "
                f"{_fmt(r.get('ci_low'))} | "
                f"{_fmt(r.get('ci_high'))} | "
                f"{_fmt(r.get('ame_pp'))} | "
                f"{_fmt(r.get('evid_score'))} | "
                f"{_fmt(r.get('price_eq'))} | "
                f"{_effect_symbol(r.get('sign'))} |"
            )
        return "\n".join(lines)

    title_block = f"*Model = {model_name}*\n\n*Product = {product_name or '(enter product name)'}*\n"
    if rows:
        header = "### Estimates of the Conditional Logit Regression\n\n" + title_block
        msg_parts = [header]
        msg_parts.append(_render_section("Position effects", sec_map.get("Position effects", [])))
        msg_parts.append(_render_section("Badge/lever effects", sec_map.get("Badge/lever effects", [])))
        msg_parts.append(_render_section("Attribute effects", sec_map.get("Attribute effects", [])))

        glossary_md = (
            "\n\n*Glossary*\n\n"
            "| Column | Meaning |\n"
            "|---|---|\n"
            "| β | Log-odds coefficient for the lever (positive increases choice odds). |\n"
            "| SE | Standard error of β. |\n"
            "| p | Two-sided p-value for H₀: β = 0. |\n"
            "| q_bh | Benjamini–Hochberg FDR-adjusted p across the displayed rows. |\n"
            "| Odds ratio | exp(β); multiplicative change in odds. |\n"
            "| CI low / CI high | 95% confidence interval bounds for the odds ratio. |\n"
            "| AME (pp) | Average marginal effect in percentage points. |\n"
            "| Evidence | 1 − p (compact signal strength in [0,1]). |\n"
            "| Price-eq λ | Effect scaled by |β_price|; blank for ln(price). |\n"
            "| Effect | Sign of β at p < .05: +, −, else 0. |\n"
        )
        msg_parts.append(glossary_md)

        def _embed_png(path, alt_text, caption):
            try:
                with open(path, "rb") as _f:
                    import base64 as _b64
                    _b = _b64.b64encode(_f.read()).decode("utf-8")
                return (
                    f'\n\n<img alt="{alt_text}" src="data:image/png;base64,{_b}" '
                    f'style="max-width:560px;border:1px solid #ddd;border-radius:6px;margin-top:10px" />\n'
                    f"\n{caption}\n"
                )
            except Exception:
                return ""

        emp_path = artifacts.get("position_heatmap_empirical", "")
        prob_path = artifacts.get("position_heatmap_prob", "") or artifacts.get("position_heatmap") or artifacts.get("position_heatmap_png") or ""
        if emp_path:
            msg_parts.append(_embed_png(
                emp_path,
                "Empirical selection heatmap (darker = higher observed selection rate)",
                "*Empirical selection heatmap (darker = higher observed selection rate)*"
            ))
        if prob_path:
            msg_parts.append(_embed_png(
                prob_path,
                "Model-implied probability heatmap (darker = higher predicted selection probability)",
                "*Model-implied probability heatmap (darker = higher predicted selection probability)*"
            ))

        return "\n".join([p for p in msg_parts if p]), results

    return "No badge effects computed.", results

@_catch_and_report
def run_now(product_name: str, brand_name: str, model_name: str, badges: list[str], price, currency: str, n_iterations):
    err = _validate_inputs(product_name, price, currency, n_iterations)
    if err:
        return err, "{}"

    import uuid
    job_id = f"job-preview-{uuid.uuid4().hex[:8]}"
    payload = _build_payload(
        job_id=job_id,
        product=product_name,
        brand=brand_name,
        model=model_name,
        badges=badges,
        price=price,
        currency=currency,
        n_iterations=n_iterations,
        fresh=True,
    )

    import importlib, logit_badges
    logit_badges = importlib.reload(logit_badges)

    results = run_job_sync(payload)

    msg, _results_obj = _format_results_from_dict(results)

    rows = results.get("logit_table_rows") or []
    csv_path, html_path = _export_badge_effects(rows, payload, job_id)
    artifacts = results.setdefault("artifacts", {})
    if csv_path:
        artifacts["effects_csv"] = csv_path
    if html_path:
        artifacts["effects_html"] = html_path

    try:
        import threading, base64 as _b64, requests, pathlib as _pl
        def _persist_async(_results, _payload):
            try:
                run_id_local = _results.get("job_id") or _payload.get("job_id") or job_id
                arts = _results.get("artifacts", {}) or {}
                send_list = []
                for key in ("effects_csv", "df_choice", "position_heatmap_empirical"):
                    p = arts.get(key)
                    if not p:
                        continue
                    try:
                        with open(p, "rb") as fh:
                            send_list.append({
                                "filename": _pl.Path(p).name,
                                "data_base64": _b64.b64encode(fh.read()).decode("utf-8")
                            })
                    except Exception:
                        pass
                if send_list:
                    try:
                        url = "https://aireadyworkforce.pro/Agentix/sendAgentixFiles.php"
                        _ = requests.post(url, json={"run_id": run_id_local, "files": send_list}, timeout=45)
                    except Exception:
                        pass

                try:
                    from save_to_agentix import persist_results_if_qualify
                    _ = persist_results_if_qualify(
                        _results,
                        _payload,
                        base_url="https://aireadyworkforce.pro/Agentix",
                        app_version="app-1",
                        est_model="logit-1",
                        alpha=0.05,
                    )
                except Exception:
                    pass
            except Exception:
                pass

        threading.Thread(target=_persist_async, args=(results, payload), name=f"persist-{job_id}", daemon=True).start()
        results.setdefault("artifacts", {})["agentix_persist_started"] = True
    except Exception as _bg_e:
        results.setdefault("artifacts", {})["agentix_persist_error_init"] = str(_bg_e)

    try:
        from save_to_agentix import persist_results_if_qualify
        persist_info = persist_results_if_qualify(
            results,
            payload,
            base_url="https://aireadyworkforce.pro/Agentix",
            app_version="app-1",
            est_model="logit-1",
            alpha=0.05,
        )
        results.setdefault("artifacts", {})["agentix_persist"] = persist_info
    except Exception as e:
        results.setdefault("artifacts", {})["agentix_persist_error"] = str(e)

    import json as _json
    return msg, _json.dumps(results, ensure_ascii=False, indent=2)

@_catch_and_report
def search_database(product_name: str):
    product = (product_name or "").strip()
    if not product:
        return "<p>Enter a product name to search.</p>", "{}"

    base_url = "https://aireadyworkforce.pro/Agentix/searchAgentix.php"

    try:
        r = requests.get(base_url, params={"product": product, "limit": 50}, timeout=12)
        ct = (r.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            data = r.json()
        else:
            data = json.loads(r.text)
    except Exception as e:
        msg = f"Could not reach the database API ({e})."
        return f"<p>{msg}</p>", "{}"

    if not data.get("ok"):
        return f"<p>{data.get('message','No relevant results found.')}</p>", json.dumps(data, ensure_ascii=False, indent=2)

    runs = data.get("runs") or []
    effects = data.get("effects") or []
    if not runs:
        return "<p>No relevant results found.</p>", json.dumps(data, ensure_ascii=False, indent=2)

    run = runs[0]
    rid = run.get("run_id")
    rows = [e for e in effects if e.get("run_id") == rid]
    if not rows:
        return "<p>No relevant results found.</p>", json.dumps(data, ensure_ascii=False, indent=2)

    product_out = run.get("product", product)
    brand_out   = run.get("brand_type", "")
    model_out   = run.get("model_name", "")
    price_out   = run.get("price_value", "")
    curr_out    = run.get("price_currency", "")
    n_iter_out  = run.get("n_iterations", "")

    header = (
        "<table style='border-collapse:collapse;width:100%'>"
        "<thead><tr>"
        "<th style='text-align:left'>product</th>"
        "<th style='text-align:left'>brand</th>"
        "<th style='text-align:left'>model</th>"
        "<th style='text-align:right'>price</th>"
        "<th style='text-align:center'>currency</th>"
        "<th style='text-align:right'>n_iterations</th>"
        "<th style='text-align:left'>badge</th>"
        "<th style='text-align:right'>beta</th>"
        "<th style='text-align:right'>p</th>"
        "<th style='text-align:center'>sign</th>"
        "</tr></thead><tbody>"
    )
    body = []
    for e in rows:
        body.append(
            "<tr>"
            f"<td>{product_out}</td>"
            f"<td>{brand_out}</td>"
            f"<td>{model_out}</td>"
            f"<td style='text-align:right'>{price_out}</td>"
            f"<td style='text-align:center'>{curr_out}</td>"
            f"<td style='text-align:right'>{n_iter_out}</td>"
            f"<td>{e.get('badge','')}</td>"
            f"<td style='text-align:right'>{e.get('beta','')}</td>"
            f"<td style='text-align:right'>{e.get('p_value', e.get('p',''))}</td>"
            f"<td style='text-align:center'>{e.get('sign','0')}</td>"
            "</tr>"
        )
    table_html = header + "".join(body) + "</tbody></table>"
    dl_url = f"{base_url}?product={quote(product)}&limit=50&format=csv"
    html = f"{table_html}<p style='margin-top:8px'><a href='{dl_url}'>⬇️ Download CSV</a></p>"

    return html, json.dumps(data, ensure_ascii=False, indent=2)

# ---------- Async UI wrappers delegating to agent_runner ----------
@_catch_and_report
@_catch_and_report
def submit_job_ui(product_name: str, brand_name: str, model_name: str, badges: list[str], price, currency: str, n_iterations):
    err = _validate_inputs(product_name, price, currency, n_iterations)
    if err:
        return "", f"❌ {err}"
    import uuid as _uuid
    payload = _build_payload(
        job_id=f"job-{_uuid.uuid4().hex[:8]}",
        product=product_name, brand=brand_name, model=model_name,
        badges=badges, price=price, currency=currency,
        n_iterations=n_iterations, fresh=True,
    )
    try:
        r = runner_submit_job_async(payload)
        if r.get("ok"):
            job_id = r.get("job_id", "")
            status = r.get("status", "running")
            total = r.get("n_iterations", payload.get("n_iterations", 0))
            message = f"✅ Submitted. Job {job_id} is {status}. 0/{total} iterations."
            return job_id, message

        return "", f"❌ Submit failed: {r}"
    except Exception as e:
        return "", f"❌ Submit error: {type(e).__name__}: {e}"

@_catch_and_report
def poll_job_ui(job_id: str):
    job_id = (job_id or "").strip()
    if not job_id:
        return "Enter a Job ID first."
    try:
        r = runner_poll_job(job_id)
        if not r.get("ok"):
            return f"⚠️ {r.get('error','unknown error')}"
        status = r.get("status", "unknown")
        done = r.get("iterations_done")
        total = r.get("n_iterations")
        if done is not None and total is not None:
            return f"Job {job_id}: {status} — {done}/{total} iterations"
        return f"Job {job_id}: {status}"
    except Exception as e:
        return f"❌ Poll error: {type(e).__name__}: {e}"

@_catch_and_report
@_catch_and_report
def fetch_job_ui(job_id: str):
    job_id = (job_id or "").strip()
    if not job_id:
        return "Enter a Job ID first.", "{}", None, None, None
    try:
        r = runner_fetch_job(job_id)
        if not r.get("ok"):
            st = r.get("status") or r.get("error","not_ready")
            return f"Job {job_id}: {st}", "{}", None, None, None
        raw = r.get("results_json") or "{}"
        try:
            results = json.loads(raw)
        except Exception:
            results = {}
        msg, _res = _format_results_from_dict(results)

        arts = results.get("artifacts") or {}
        badges_src = arts.get("effects_csv") or arts.get("badges_effects")
        df_choice_src = arts.get("df_choice")
        log_compare_src = arts.get("log_compare")

        def _copy_with_job(src_path: str | None, job_id: str) -> str | None:
            if not src_path:
                return None
            p = pathlib.Path(src_path)
            if not p.exists():
                return None
            new_name = f"{p.stem}_{job_id}{p.suffix}"
            dst = p.parent / new_name
            try:
                import shutil
                shutil.copy(p, dst)
                return str(dst)
            except Exception:
                return str(p)

        badges_out = _copy_with_job(badges_src, job_id)
        df_choice_out = _copy_with_job(df_choice_src, job_id)
        log_compare_out = _copy_with_job(log_compare_src, job_id)

        return (
            msg,
            json.dumps(results, ensure_ascii=False, indent=2),
            badges_out,
            df_choice_out,
            log_compare_out,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return f"❌ Fetch error: {type(e).__name__}: {e}", "{}", None, None, None

# --- Admin helpers ---
@_catch_and_report
def _list_storefront_jobs(admin_key: str):
    if not ADMIN_KEY or admin_key != ADMIN_KEY:
        return gr.update(choices=[], value=None), gr.update(value="Invalid or missing admin key.")
    sf_dir = pathlib.Path("storefront")
    if not sf_dir.exists():
        return gr.update(choices=[], value=None), "No storefront directory yet."
    jobs = sorted([p.stem for p in sf_dir.glob("*.html")])
    if not jobs:
        return gr.update(choices=[], value=None), "No storefront HTML files found."
    return gr.update(choices=jobs, value=jobs[-1]), f"Found {len(jobs)} storefront(s). Select one to preview."

@_catch_and_report
def _preview_storefront(admin_key: str, job_id: str):
    if not ADMIN_KEY or admin_key != ADMIN_KEY:
        return gr.update(value=""), "Invalid or missing admin key."
    if not job_id:
        return gr.update(value=""), "Select a job first."
    html_path = pathlib.Path("storefront") / f"{job_id}.html"
    if not html_path.exists():
        return gr.update(value=""), f"Not found: {html_path}"
    html = html_path.read_text(encoding="utf-8")
    return gr.update(value=html), f"Rendered {html_path}"

def _list_stats_files(admin_key: str):
    if not ADMIN_KEY or admin_key != ADMIN_KEY:
        return ("Invalid or missing admin key.", None, None, None, None)

    res = pathlib.Path("results")
    if not res.exists():
        return ("No results/ directory yet.", None, None, None, None)

    agg_choice = res / "df_choice.csv"
    agg_long   = res / "df_long.csv"
    agg_log    = res / "log_compare.jsonl"
    badges_effects_path = res / "badges_effects.csv"

    msgs = []
    if agg_choice.exists():        msgs.append(f"• Found {agg_choice}")
    if agg_long.exists():          msgs.append(f"• Found {agg_long}")
    if agg_log.exists():           msgs.append(f"• Found {agg_log}")
    if badges_effects_path.exists():
        msgs.append(f"• Found {badges_effects_path}")
    if not msgs:
        msgs = ["No aggregate files yet. Run a simulation first."]

    return (
        "\n".join(msgs),
        str(agg_choice) if agg_choice.exists() else None,
        str(agg_long)   if agg_long.exists()   else None,
        str(agg_log)    if agg_log.exists()    else None,
        str(badges_effects_path) if badges_effects_path.exists() else None,
    )

@_catch_and_report
def preview_example(product_name: str, brand_name: str, model_name: str, badges: list[str], price, currency: str):
    import uuid  # ensure in-scope
    payload = _build_payload(
        job_id=f"preview-{uuid.uuid4().hex[:8]}",
        product=product_name,
        brand=brand_name,
        model=model_name,
        badges=badges,
        price=price,
        currency=currency,
        n_iterations=1,
        fresh=False,
    )
    from agent_runner import preview_one
    res = preview_one(payload)
    img_html = (
        f'<div style="margin-top:8px">'
        f'<div style="font-weight:600;margin-bottom:6px">Example screen ({res.get("set_id","S0001")})</div>'
        f'<img alt="Agentix example screen" src="{res.get("image_b64","")}" '
        f'style="max-width:100%;border:1px solid #ddd;border-radius:8px" />'
        f"</div>"
    )
    return img_html

@_catch_and_report
def _preview_badges_effects(admin_key: str):
    if not ADMIN_KEY or admin_key != ADMIN_KEY:
        return "<p>Invalid or missing admin key.</p>"

    import pandas as pd
    path = pathlib.Path("results") / "badges_effects.csv"
    if not path.exists():
        return "<p>No badges_effects.csv yet. Run a simulation first.</p>"

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f"<p>Could not read {path}: {e}</p>"

    preferred = [
        "badge", "beta", "se", "p", "q_bh",
        "odds_ratio", "ci_low", "ci_high",
        "ame_pp", "evid_score", "price_eq", "sign"
    ]
    view_cols = [c for c in df.columns]
    if preferred and all(c in df.columns for c in preferred):
        view_cols = preferred

    html = df[view_cols].to_html(index=False, border=1, justify="center")
    return html

# === Scoring helpers (auto-detect only; manual tools removed) =================
import base64, time as _time, uuid as _uuid2, os as _os, json as _json2, requests as _req2

# import scorers (do NOT import load_params_from_php from score_image to avoid shadowing)
try:
    from score_image import score_grid_2x4
except Exception:
    score_grid_2x4 = None

try:
    from score_image import score_single_card
except Exception:
    score_single_card = None

try:
    from agent_runner import detect_levers as _agent_detect_levers  # optional
except Exception:
    _agent_detect_levers = None

UPLOADS_DIR = pathlib.Path("uploads"); UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# --- Detection vocabulary & prompt (drop-in replacement) ----------------------

_CUE_EXCLUDE = {"ln(price)", "Row 1", "Column 1", "Column 2", "Column 3"}

# Canonical cue names we want the model to use (must match your params where possible)
_EXPECTED_CUES = [
    "All-in framing",
    "Assurance",
    "Scarcity tag",
    "Strike-through",
    "Timer",
    "social",
    "voucher",
    "bundle",
    
]

# Build the allowed set from PHP params ∪ expected defaults, then remove exclusions
if SCORE_PARAMS and isinstance(SCORE_PARAMS, dict):
    _param_cues = [k for k in SCORE_PARAMS.keys() if k not in _CUE_EXCLUDE]
else:
    _param_cues = []
_allowed_cues = []
_seen = set()
for name in (_param_cues + _EXPECTED_CUES):
    if name not in _CUE_EXCLUDE and name not in _seen:
        _allowed_cues.append(name)
        _seen.add(name)
CUE_CHOICES_SCORER = _allowed_cues

# Synonym/variant normalisation to the canonical keys above
_NORMALISE = {
    # all-in
    "all-in framing": "All-in framing",
    "all in framing": "All-in framing",
    "all-in v. partitioned pricing": "All-in framing",
    "all-in price": "All-in framing",
    "price includes tax": "All-in framing",
    "inc vat": "All-in framing",
    "including vat": "All-in framing",
    "including shipping": "All-in framing",

    # assurance
    "assurance": "Assurance",
    "returns": "Assurance",
    "free returns": "Assurance",
    "warranty": "Assurance",
    "guarantee": "Assurance",
    "money-back": "Assurance",

    # scarcity
    "scarcity": "Scarcity tag",
    "scarcity tag": "Scarcity tag",
    "low stock": "Scarcity tag",
    "only x left": "Scarcity tag",
    "limited stock": "Scarcity tag",
    "selling fast": "Scarcity tag",

    # strike-through
    "strike-through": "Strike-through",
    "strikethrough": "Strike-through",
    "sale price": "Strike-through",
    "was now": "Strike-through",
    "was £": "Strike-through",
    "discounted from": "Strike-through",

    # timer
    "timer": "Timer",
    "countdown": "Timer",
    "ends in": "Timer",
    "limited time": "Timer",
    "deal ends": "Timer",
    "hours left": "Timer",

    # social proof
    "social": "social",
    "social proof": "social",
    "x bought": "social",
    "x sold": "social",
    "people viewing": "social",
    "bestseller": "social",

    # voucher / coupon
    "voucher": "voucher",
    "coupon": "voucher",
    "promo code": "voucher",
    "use code": "voucher",
    "apply voucher": "voucher",
    "clip coupon": "voucher",

    # bundle
    "bundle": "bundle",
    "bundle & save": "bundle",
    "2 for": "bundle",
    "buy 1 get 1": "bundle",
    "multi-buy": "bundle",

   
}
def _norm_label(x: str) -> str:
    k = (x or "").strip().lower()
    return _NORMALISE.get(k, (x or "").strip())

def _fmt_num(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

def _file_to_data_url(path_like: str) -> str:
    p = pathlib.Path(path_like)
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    ext = (p.suffix or "").lower()
    mime = "image/png"
    if ext in {".jpg", ".jpeg"}: mime = "image/jpeg"
    elif ext == ".webp": mime = "image/webp"
    return f"data:{mime};base64,{b64}"

def _build_detection_prompt() -> str:
    vocab = ", ".join(CUE_CHOICES_SCORER)
    return (
        "You are an e-commerce UI analyst. Detect ONLY these cues (use exactly these labels): "
        f"{vocab}.\n"
        "Definitions and evidence requirements:\n"
        "All-in framing = the shown price explicitly includes taxes/shipping/fees (e.g., “£399 inc. VAT”, “price includes tax/shipping”). "
        "“Price excludes VAT” is NOT all-in. Do not infer from generic ‘Deal’ or delivery text.\n"
        "Assurance = explicit returns/warranty/guarantee statements (e.g., “30-day returns”, “2-year warranty”, “money-back guarantee”). "
        "FREE / fast delivery, Prime, and dispatch dates are NOT assurance.\n"
        "Scarcity tag = explicit low stock or limited availability (e.g., “Only 3 left”, “Low stock”, “Selling fast”, “Limited stock”). "
        "“In stock” or delivery dates are NOT scarcity.\n"
        "Strike-through = a price visibly crossed-out OR a textual previous-price marker (evidence must include one of: a crossed-out number; "
        "‘was £’, ‘RRP’, ‘List price’, ‘Previous price’, ‘Save £’ next to the price). ‘Deal’, ‘Prime’, or coloured badges alone are NOT strike-through.\n"
        "Timer = a countdown or deadline (e.g., “Ends in 02:14:10”, “Sale ends today”, “X hours left”).\n"
        "social = social proof (stars, 1–5 ★, review counts, “bought”, “viewing now”, “Bestseller”).\n"
        "voucher = coupon/promo (e.g., “Use code SAVE10”, “Apply voucher”, “Clip coupon”).\n"
        "bundle = multi-item offer (e.g., “2 for £50”, “Buy 1 get 1 50% off”, “Bundle & save”). “Pack of 10” alone is NOT a bundle price deal.\n"
        "ratings = star graphics or numeric ratings like “4.3/5”, optionally with review counts.\n"
        "Rules: Return a STRICT JSON object using only the allowed labels. If a cue lacks the evidence above, omit it. Prefer precision over recall. "
        "Zoom into fine print and read small text; do not guess.\n"
        "Output formats: Single image → {\"cues\":[<labels>]}; Grid 2×4 (row-major 8 cells) → {\"grid\":[[<labels>],..., [<labels>]]}.\n"
    )

def _fallback_openai_detect(image_b64: str, mode: str = "single"):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}

    sys = _build_detection_prompt()
    user_text = "Mode: single" if mode == "single" else "Mode: grid_2x4"

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_b64}}
            ]},
        ],
        "max_tokens": 600,
        "temperature": 0
    }

    r = requests.post(url, headers=headers, json=data, timeout=(15, 240))
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:500]}")
    txt = r.json()["choices"][0]["message"]["content"]

    # Best-effort JSON recovery
    try:
        obj = json.loads(txt)
    except Exception:
        i, j = txt.find("{"), txt.rfind("}")
        obj = json.loads(txt[i:j+1]) if i >= 0 and j > i else {}

    # Post-process to ensure only allowed canonical labels are returned
    if mode == "single":
        raw = [x for x in (obj.get("cues") or []) if isinstance(x, str)]
        cues = []
        for x in raw:
            canon = _norm_label(x)
            if canon in CUE_CHOICES_SCORER and canon not in cues:
                cues.append(canon)
        return cues

    # grid
    out = []
    for cell in (obj.get("grid") or [])[:8]:
        clean = set()
        if isinstance(cell, list):
            for x in cell:
                if not isinstance(x, str):
                    continue
                canon = _norm_label(x)
                if canon in CUE_CHOICES_SCORER:
                    clean.add(canon)
        out.append(clean)
    while len(out) < 8:
        out.append(set())
    return out

def _detect_with_agent_or_fallback(image_b64: str, mode: str):
    if _agent_detect_levers is not None:
        res = _agent_detect_levers(image_b64, mode)
        if mode == "single":
            cues = [x for x in (res.get("cues") or []) if isinstance(x, str)]
            return [_norm_label(x) for x in cues]
        grid = res.get("grid") or []
        out = []
        for cell in grid[:8]:
            if isinstance(cell, (list, set, tuple)):
                out.append(set(_norm_label(x) for x in cell))
            else:
                out.append(set())
        while len(out) < 8: out.append(set())
        return out
    return _fallback_openai_detect(image_b64, mode)

@_catch_and_report
def _auto_single_from_image(filepath: str) -> tuple:
    if not filepath:
        return gr.update(value=None), "No image.", "",""
    if score_single_card is None:
        return gr.update(value=None), "Scoring utility not available.", "",""
    data_url = _file_to_data_url(filepath)
    cues = _detect_with_agent_or_fallback(data_url, "single")
    cues = [c for c in cues if c in CUE_CHOICES_SCORER]
    res = score_single_card(set(cues))
    badges_md = "#### Identified badges\n" + (", ".join(cues) if cues else "—")
    score_md = (
        "#### Single card score\n\n| Metric | Value |\n|---|---:|\n"
        f"| raw | {_fmt_num(res.get('raw'))} |\n"
        f"| price_weight | {_fmt_num(res.get('price_weight'))} |\n"
        f"| final | {_fmt_num(res.get('final'))} |\n"
        f"| ∑ s_i | {_fmt_num(res.get('sum_s'))} |\n"
        f"| ∑ w_i | {_fmt_num(res.get('sum_w'))} |\n"
    )
    return gr.update(value=filepath), badges_md, score_md, "✅ Detected and scored."

@_catch_and_report
def _auto_grid_from_image(filepath: str) -> tuple:
    if not filepath:
        return gr.update(value=None), "No image.", "", ""
    if SCORE_PARAMS is None or not isinstance(SCORE_PARAMS, dict):
        return gr.update(value=None), "Scoring utility not available (no params).", "", ""
    if score_grid_2x4 is None:
        return gr.update(value=None), "Scoring utility not available.", "", ""

    data_url = _file_to_data_url(filepath)
    detected_grid = _detect_with_agent_or_fallback(data_url, "grid_2x4")

    norm_cells = []
    for cell in (detected_grid[:8] + [set()] * 8)[:8]:
        if not isinstance(cell, (list, set, tuple)):
            cell = []
        clean = set(c for c in cell if c in CUE_CHOICES_SCORER)
        norm_cells.append(clean)

    cards = [{"cues": cell, "price": None} for cell in norm_cells]

    # Try scorer with params; if its signature doesn't accept params, call without.
    try:
        res = score_grid_2x4(cards, SCORE_PARAMS)
    except TypeError:
        res = score_grid_2x4(cards)

    b_lines = [
        "#### Identified badges per card (row-major)",
        "",
        "| Card | Row | Col | Badges |",
        "|---:|---:|---:|---|",
    ]
    for i, cell in enumerate(norm_cells, 1):
        r = 1 if i <= 4 else 2
        c = ((i - 1) % 4) + 1
        b_lines.append(
            f"| {i} | {r} | {c} | {', '.join(sorted(cell)) if cell else '—'} |"
        )
    badges_md = "\n".join(b_lines)

    s_lines = [
        "#### Grid 2×4 scores",
        "",
        "| Card | Row | Col | option A (β·x) | option B (scores) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for i, card_res in enumerate(res.get("cards", []), 1):
        s_lines.append(
            f"| {i} | {card_res.get('row')} | {card_res.get('col')} | "
            f"{_fmt_num(card_res.get('option_a'))} | {_fmt_num(card_res.get('option_b'))} |"
        )

    s_lines += [
        "",
        "##### Aggregates",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| mean_option_a | {_fmt_num(res.get('mean_option_a'))} |",
        f"| mean_option_b | {_fmt_num(res.get('mean_option_b'))} |",
        f"| best_option_a | {_fmt_num(res.get('best_option_a'))} |",
        f"| best_option_b | {_fmt_num(res.get('best_option_b'))} |",
    ]
    score_md = "\n".join(s_lines)

    return gr.update(value=filepath), badges_md, score_md, "✅ Detected and scored."

@_catch_and_report
def _handle_image_upload(file) -> tuple:
    if file is None:
        return gr.update(value=None), "No file selected."
    try:
        import shutil
        src = pathlib.Path(file.name)
        ext = src.suffix if src.suffix else ".png"
        new_name = f"img_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
        dest = UPLOADS_DIR / new_name
        shutil.copy(src, dest)
        return gr.update(value=str(dest)), f"✅ Saved to {dest}"
    except Exception as e:
        return gr.update(value=None), f"❌ Upload failed: {type(e).__name__}: {e}"

# ---------- UI ----------
with gr.Blocks(title="Agentix - AI Agent Buying Behavior") as demo:
    gr.Markdown(
        "# Agentix\n"
        "Agentix helps you understand how marketing badges—such as scarcity indicators and strike-through pricing—affect AI agents’ buying behaviour on your e-commerce site.\n\n"
        "Before you simulate, search our database for prior significant results by filling in the product name.\n\n"
        "If nothing is found, run a new simulation to estimate badge effects.\n\n"
    )
    with gr.Row():
        product = gr.Textbox(label="Product name", placeholder="e.g., smart phone, washing machine", scale=2)
        brand = gr.Textbox(label="Brand (optional)", placeholder="e.g., Apple, Samsung", scale=1)
        model = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="AI Agent", scale=1)
    with gr.Row():
        price = gr.Number(label="Price", value=0.0, precision=2)
        currency = gr.Dropdown(choices=CURRENCY_CHOICES, value=CURRENCY_CHOICES[0], label="Currency")

    with gr.Row():
        auto_iter = gr.Checkbox(label="Automatic calculations of iterations", value=True, scale=1)
        manual_iter = gr.Checkbox(label="Manual calculations of iterations", value=False, scale=1)

    default_auto_iters = _auto_iterations_from_badges([])
    n_iterations = gr.Number(label="Iterations", value=default_auto_iters, precision=0, interactive=False)

    badges = gr.CheckboxGroup(choices=BADGE_CHOICES, label="Select badges (multi-select)")

    search_btn = gr.Button("Search our database", variant="secondary")

    with gr.Row():
        submit_async_btn = gr.Button("Submit long run (async)", variant="primary")
        job_id_box = gr.Textbox(label="Job ID", placeholder="Will appear after submit", interactive=False)
    with gr.Row():
        poll_btn = gr.Button("Poll status", variant="secondary")
        fetch_btn = gr.Button("Fetch results", variant="secondary")
    async_status = gr.Markdown()

    run_btn = gr.Button("Run simulation now", variant="secondary", visible=False)

    preview_btn = gr.Button("Preview one example screen", variant="secondary")
    preview_view = gr.HTML(label="Preview")

    results_md = gr.Markdown()
    results_json = gr.Code(label="Results JSON", language="json")
    badges_file = gr.File(label="badges_effects.csv")
    df_choice_file = gr.File(label="df_choice.csv")
    log_compare_file = gr.File(label="log_compare.jsonl")

    search_btn.click(
        fn=search_database,
        inputs=[product],
        outputs=[results_md, results_json],
    )

    submit_async_btn.click(
        fn=submit_job_ui,
        inputs=[product, brand, model, badges, price, currency, n_iterations],
        outputs=[job_id_box, async_status],
    )
    poll_btn.click(
        fn=poll_job_ui,
        inputs=[job_id_box],
        outputs=[async_status],
    )
    fetch_btn.click(
        fn=fetch_job_ui,
        inputs=[job_id_box],
        outputs=[results_md, results_json, badges_file, df_choice_file, log_compare_file],
    )

    preview_btn.click(
        fn=preview_example,
        inputs=[product, brand, model, badges, price, currency],
        outputs=[preview_view],
    )
    run_btn.click(
        fn=run_now,
        inputs=[product, brand, model, badges, price, currency, n_iterations],
        outputs=[results_md, results_json],
    )

    @_catch_and_report
    def _on_badges_change(badges_sel: list[str], auto_checked: bool, manual_checked: bool):
        if auto_checked and not manual_checked:
            n = _auto_iterations_from_badges(badges_sel)
            return gr.update(value=n, interactive=False)
        return gr.update(interactive=True)

    @_catch_and_report
    def _toggle_auto(auto_checked: bool, badges_sel: list[str]):
        if auto_checked:
            n = _auto_iterations_from_badges(badges_sel)
            return gr.update(value=False), gr.update(value=n, interactive=False)
        return gr.update(value=True), gr.update(value=100, interactive=True)

    @_catch_and_report
    def _toggle_manual(manual_checked: bool, badges_sel: list[str]):
        if manual_checked:
            return gr.update(value=False), gr.update(value=100, interactive=True)
        n = _auto_iterations_from_badges(badges_sel)
        return gr.update(value=True), gr.update(value=n, interactive=False)

    auto_iter.change(
        fn=_toggle_auto,
        inputs=[auto_iter, badges],
        outputs=[manual_iter, n_iterations],
    )
    manual_iter.change(
        fn=_toggle_manual,
        inputs=[manual_iter, badges],
        outputs=[auto_iter, n_iterations],
    )
    badges.change(
        fn=_on_badges_change,
        inputs=[badges, auto_iter, manual_iter],
        outputs=[n_iterations],
    )

    with gr.Accordion("Admin preview (storefront)", open=False):
        admin_key_in = gr.Textbox(label="Admin key", type="password", placeholder="Enter ADMIN_KEY", scale=1)
        refresh = gr.Button("List storefronts")
        job_picker = gr.Dropdown(label="Job ID", choices=[], interactive=True, scale=2)
        preview_btn2 = gr.Button("Preview storefront")
        admin_status = gr.Markdown()
        html_view = gr.HTML(label="Storefront preview")
        refresh.click(_list_storefront_jobs, inputs=[admin_key_in], outputs=[job_picker, admin_status])
        preview_btn2.click(_preview_storefront, inputs=[admin_key_in, job_picker], outputs=[html_view, admin_status])

    gr.Markdown("### Stats files (aggregates)")
    stats_refresh = gr.Button("Refresh stats")
    stats_status = gr.Markdown()
    stats_choice = gr.File(label="df_choice.csv")
    stats_long = gr.File(label="df_long.csv")
    stats_log = gr.File(label="log_compare.jsonl")
    stats_badges = gr.File(label="badges_effects.csv")

    stats_badges_preview_btn = gr.Button("Preview badges_effects table")
    stats_badges_preview_html = gr.HTML(label="badges_effects preview")

    stats_badges_preview_btn.click(
        _preview_badges_effects,
        inputs=[admin_key_in],
        outputs=[stats_badges_preview_html],
    )

    stats_refresh.click(
        _list_stats_files,
        inputs=[admin_key_in],
        outputs=[stats_status, stats_choice, stats_long, stats_log, stats_badges],
    )

    gr.Markdown("## Scoring from image (auto-detect)")
    with gr.Tabs():
        with gr.Tab("Auto-detect — Single card"):
            auto_single_img = gr.Image(label="Upload single product image", type="filepath")
            auto_single_go = gr.Button("Detect badges and score", variant="primary")
            auto_single_preview = gr.Image(label="Preview", interactive=False)
            auto_single_badges = gr.Markdown()
            auto_single_score = gr.Markdown()
            auto_single_status = gr.Markdown()
            auto_single_go.click(
                fn=_auto_single_from_image,
                inputs=[auto_single_img],
                outputs=[auto_single_preview, auto_single_badges, auto_single_score, auto_single_status],
            )

        with gr.Tab("Auto-detect — Grid 2×4"):
            auto_grid_img = gr.Image(label="Upload 2×4 grid image", type="filepath")
            auto_grid_go = gr.Button("Detect badges and score grid", variant="primary")
            auto_grid_preview = gr.Image(label="Preview", interactive=False)
            auto_grid_badges = gr.Markdown()
            auto_grid_score = gr.Markdown()
            auto_grid_status = gr.Markdown()
            auto_grid_go.click(
                fn=_auto_grid_from_image,
                inputs=[auto_grid_img],
                outputs=[auto_grid_preview, auto_grid_badges, auto_grid_score, auto_grid_status],
            )

        with gr.Accordion("A/B logs (download)", open=False):
            ab_list_btn = gr.Button("Refresh list")
            ab_logs_dd  = gr.Dropdown(label="Select a log file", choices=[], interactive=True)
            ab_file_out = gr.File(label="Download selected file")

            def _ab_list_ui():
                try:
                    from ABTesting import list_ab_logs
                    files = list_ab_logs()
                    return gr.update(choices=files, value=(files[-1] if files else None))
                except Exception as e:
                    return gr.update(choices=[], value=None, label=f"Error: {e}")

            def _ab_download_ui(path):
                try:
                    from ABTesting import get_ab_log
                    return get_ab_log(path)
                except Exception:
                    return None

            ab_list_btn.click(_ab_list_ui, inputs=[], outputs=[ab_logs_dd])
            ab_logs_dd.change(_ab_download_ui, inputs=[ab_logs_dd], outputs=[ab_file_out])

        with gr.Tab("Live A/B — GPT choices (async)"):
            lab = gr.Markdown("Upload your two variants and choose the number of trials. Each trial builds a fresh 2×4 grid (4×A, 4×B) with randomised positions and calls the agent once.")
            abA = gr.Image(label="Image A (e.g., strike-through)", type="filepath")
            abB = gr.Image(label="Image B (e.g., scarcity)", type="filepath")
            abN = gr.Number(label="Trials", value=300, precision=0)
            abCategory = gr.Textbox(label="Category (optional)", value="smartphone")
            abModel = gr.Textbox(label="Model name (optional)", placeholder="e.g., gpt-4.1-mini")
            abSubmit = gr.Button("Submit live A/B job", variant="primary")
            abJob = gr.Textbox(label="Job ID", interactive=False)
            abStatus = gr.Markdown()
            abPoll = gr.Button("Poll")
            abFetch = gr.Button("Fetch results")
            abMD = gr.Markdown()
            abJSON = gr.Code(label="Result JSON", language="json")
            abStop = gr.Button("Stop")

            abSubmit.click(submit_live_ab, inputs=[abA, abB, abN, abCategory, abModel], outputs=[abJob, abStatus])
            abPoll.click(poll_live_ab, inputs=[abJob], outputs=[abStatus])
            abFetch.click(fetch_live_ab, inputs=[abJob], outputs=[abMD, abJSON])
            abStop.click(cancel_live_ab, inputs=[abJob], outputs=[abStatus])

    gr.Markdown("### Upload image (save only)")
    upload_preview = gr.Image(label="Uploaded image preview", interactive=False)
    upload_btn = gr.UploadButton("Upload image", file_types=["image"], file_count="single")
    upload_status = gr.Markdown()
    upload_btn.upload(
        fn=_handle_image_upload,
        inputs=[upload_btn],
        outputs=[upload_preview, upload_status],
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)








