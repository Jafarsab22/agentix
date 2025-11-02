# app.py 
import json, uuid, os, time, pathlib, csv, requests
from datetime import datetime
import gradio as gr
from agent_runner import run_job_sync
import traceback, logging
from html import escape
from urllib.parse import quote   

logging.basicConfig(level=logging.INFO)

# ---------- Async job queue (breaks the 900s ceiling) ----------
import threading, uuid, json

_JOBS = {}
_JOBS_LOCK = threading.Lock()

def _bg_run_job(job_id: str, args_tuple: tuple):
    try:
        msg, results_json = run_now(*args_tuple)  # your existing long function
        with _JOBS_LOCK:
            _JOBS[job_id] = {"status": "done", "msg": msg, "results_json": results_json}
    except Exception as e:
        with _JOBS_LOCK:
            _JOBS[job_id] = {"status": "error", "error": f"{type(e).__name__}: {e}"}

def submit_job_async(product_name, brand_name, model_name, badges, price, currency, n_iterations):
    jid = f"job-{uuid.uuid4().hex[:8]}"
    args_tuple = (product_name, brand_name, model_name, badges, price, currency, n_iterations)
    with _JOBS_LOCK:
        _JOBS[jid] = {"status": "queued"}
    threading.Thread(target=_bg_run_job, args=(jid, args_tuple), name=f"runner-{jid}", daemon=True).start()
    return {"ok": True, "job_id": jid}

def poll_job(job_id: str):
    with _JOBS_LOCK:
        info = _JOBS.get(job_id)
    if not info:
        return {"ok": False, "status": "unknown"}
    return {"ok": True, "status": info["status"]}

def fetch_job(job_id: str):
    with _JOBS_LOCK:
        info = _JOBS.get(job_id)
    if not info:
        return {"ok": False, "status": "unknown"}
    if info["status"] == "done":
        return {"ok": True, "status": "done", "msg": info["msg"], "results_json": info["results_json"]}
    if info["status"] == "error":
        return {"ok": False, "status": "error", "error": info["error"]}
    return {"ok": True, "status": info["status"]}


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
            # Handlers return (markdown, json); pad if needed
            return (msg, "{}") if fn.__name__ in ("run_now",) else msg
    return _inner


RESULTS_DIR = pathlib.Path("results"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = pathlib.Path("jobs"); JOBS_DIR.mkdir(parents=True, exist_ok=True)
EFFECTS_DIR = RESULTS_DIR / "effects"; EFFECTS_DIR.mkdir(parents=True, exist_ok=True)

# Optional storefront helpers
try:
    from storefront import build_storefront_from_payload, save_storefront
except Exception:
    build_storefront_from_payload = None
    save_storefront = None

# -------- UI choices --------
MODEL_CHOICES = ["OpenAI GPT-4.1-mini"]  # extend later if needed
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

def _ceil_to_8(n: int) -> int:
    return int(((int(n) + 7) // 8) * 8)

def _auto_iterations_from_badges(badges: list[str]) -> int:
    sel = set(badges or [])
    b = 0
    if "All-in v. partitioned pricing" in sel:
        b += 1
    # Non-frame levers you currently estimate separately in the logit
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
    base = f"{ts}_{job_id}_badge_effects"
    csv_path = EFFECTS_DIR / f"{base}.csv"
    html_path = EFFECTS_DIR / f"{base}.html"
    # Extended fields preserved if present in rows
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

    # Ensure latest estimator code is used
    import importlib, logit_badges
    logit_badges = importlib.reload(logit_badges)

    results = run_job_sync(payload)

    # ---- Render grouped tables (Position, Badge/lever, Attribute) ----
    rows = results.get("logit_table_rows") or []

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

    # Partition rows into sections (robust to older payloads without 'section')
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

    # Order within each section
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
                f"{_fmt(r.get('p'), nd=4)} | "
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

    # Title block (model + product)
    title_block = (
        f"*Model = {model_name}*\n\n"
        f"*Product = {product_name or '(enter product name)'}*\n"
    )

    if rows:
        header = "### Estimates of the Conditional Logit Regression\n\n" + title_block
        msg_parts = [header]
        msg_parts.append(_render_section("Position effects", sec_map.get("Position effects", [])))
        msg_parts.append(_render_section("Badge/lever effects", sec_map.get("Badge/lever effects", [])))
        msg_parts.append(_render_section("Attribute effects", sec_map.get("Attribute effects", [])))

        # ---- Glossary (placed immediately after Attribute effects) ----
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

        # Local export (CSV/HTML) – unchanged
        csv_path, html_path = _export_badge_effects(rows, payload, job_id)
        artifacts = results.setdefault("artifacts", {})
        if csv_path:
            artifacts["effects_csv"] = csv_path
        if html_path:
            artifacts["effects_html"] = html_path

        # ---- Inline heat-maps: show BOTH empirical and probability maps ----
        def _embed_png(path, alt_text, caption):
            try:
                with open(path, "rb") as _f:
                    import base64 as _b64
                    _b = _b64.b64encode(_f.read()).decode("utf-8")
                return (
                    f'\n\n<img alt="{alt_text}" '
                    f'src="data:image/png;base64,{_b}" '
                    f'style="max-width:560px;border:1px solid #ddd;border-radius:6px;margin-top:10px" />\n'
                    f"\n{caption}\n"
                )
            except Exception:
                return ""

        emp_path = artifacts.get("position_heatmap_empirical", "")
        prob_path = artifacts.get("position_heatmap_prob", "") or artifacts.get("position_heatmap") or artifacts.get("position_heatmap_png") or ""

        if emp_path:
            msg_parts.append(
                _embed_png(
                    emp_path,
                    "Empirical selection heatmap (darker = higher observed selection rate)",
                    "*Empirical selection heatmap (darker = higher observed selection rate)*"
                )
            )
        if prob_path:
            msg_parts.append(
                _embed_png(
                    prob_path,
                    "Model-implied probability heatmap (darker = higher predicted selection probability)",
                    "*Model-implied probability heatmap (darker = higher predicted selection probability)*"
                )
            )

        msg = "\n".join([p for p in msg_parts if p])
    else:
        msg = "No badge effects computed."

    # ---------------- [NEW] Fire-and-forget persistence of artifacts + DB ----------------
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
    # ---------------- [END NEW] ----------------------------------------------------------

    # Original best-effort DB call (safe if it runs twice server-side).
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

    import json
    return msg, json.dumps(results, ensure_ascii=False, indent=2)





@_catch_and_report
def search_database(product_name: str):
    """Query Agentix DB and return (HTML table, pretty_json)."""
    product = (product_name or "").strip()
    if not product:
        return "<p>Enter a product name to search.</p>", "{}"

    base_url = "https://aireadyworkforce.pro/Agentix/searchAgentix.php"

    # --- Call the browser-proven GET path ---
    try:
        r = requests.get(base_url, params={"product": product, "limit": 50}, timeout=12)
        ct = (r.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            data = r.json()
        else:
            # last-chance parse in case the server mislabeled
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

    # Pick the most recent run and its effects
    run = runs[0]
    rid = run.get("run_id")
    rows = [e for e in effects if e.get("run_id") == rid]

    if not rows:
        return "<p>No relevant results found.</p>", json.dumps(data, ensure_ascii=False, indent=2)

    # Shared run-level fields
    product_out = run.get("product", product)
    brand_out   = run.get("brand_type", "")
    model_out   = run.get("model_name", "")
    price_out   = run.get("price_value", "")
    curr_out    = run.get("price_currency", "")
    n_iter_out  = run.get("n_iterations", "")

    # Build HTML table (renders in Gradio Markdown/HTML)
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

    # CSV download (Option A): served by the PHP endpoint
    dl_url = f"{base_url}?product={quote(product)}&limit=50&format=csv"
    html = f"{table_html}<p style='margin-top:8px'><a href='{dl_url}'>⬇️ Download CSV</a></p>"

    return html, json.dumps(data, ensure_ascii=False, indent=2)

# --- Admin helpers --- (continued from Part 1)

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
    badges_effects_path = res / "badges_effects.csv"  # single source of truth

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
    """Render results/badges_effects.csv as an HTML table for quick inspection."""
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
    view_cols = [c for c in preferred if c in df.columns]
    if not view_cols:
        view_cols = list(df.columns)

    html = df[view_cols].to_html(index=False, border=1, justify="center")
    return html



# ---------- UI logic for Automatic / Manual iterations ----------

@_catch_and_report
def _on_badges_change(badges: list[str], auto_checked: bool, manual_checked: bool):
    if auto_checked and not manual_checked:
        n = _auto_iterations_from_badges(badges)
        # Disabled (non-interactive) shows greyed text, signalling system-set value
        return gr.update(value=n, interactive=False)
    # Manual mode: keep field editable; do not override user value
    return gr.update(interactive=True)

@_catch_and_report
def _toggle_auto(auto_checked: bool, badges: list[str]):
    if auto_checked:
        # Enforce exclusivity: turning Auto on turns Manual off and disables input
        n = _auto_iterations_from_badges(badges)
        return gr.update(value=False), gr.update(value=n, interactive=False)
    # If Auto is unticked, fall back to Manual enabled
    return gr.update(value=True), gr.update(value=100, interactive=True)

@_catch_and_report
def _toggle_manual(manual_checked: bool, badges: list[str]):
    if manual_checked:
        # Enforce exclusivity: turning Manual on turns Auto off and enables input (default 100)
        return gr.update(value=False), gr.update(value=100, interactive=True)
    # If Manual is unticked, return to Auto
    n = _auto_iterations_from_badges(badges)
    return gr.update(value=True), gr.update(value=n, interactive=False)


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

    # Iterations controls: two tick boxes above the iterations field
    with gr.Row():
        auto_iter = gr.Checkbox(label="Automatic calculations of iterations", value=True, scale=1)
        manual_iter = gr.Checkbox(label="Manual calculations of iterations", value=False, scale=1)

    # Default auto value based on zero selected badges: ceil_to_8(max(100, 0)) = 104
    default_auto_iters = _auto_iterations_from_badges([])
    n_iterations = gr.Number(label="Iterations", value=default_auto_iters, precision=0, interactive=False)

    badges = gr.CheckboxGroup(choices=BADGE_CHOICES, label="Select badges (multi-select)")

    # Search-first workflow
    search_btn = gr.Button("Search our database", variant="secondary")
    run_btn = gr.Button("Run simulation now", variant="primary")
    preview_btn = gr.Button("Preview one example screen", variant="secondary")
    preview_view = gr.HTML(label="Preview")

    # Main results area: table markdown + JSON for debugging
    results_md = gr.Markdown()
    results_json = gr.Code(label="Results JSON", language="json")

    # Wire search to show results exactly where the table normally appears
    search_btn.click(
        fn=search_database,
        inputs=[product],
        outputs=[results_md, results_json],
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

    # --- Interactions for Automatic / Manual iterations and badges changes ---
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
    stats_badges = gr.File(label="badges_effects.csv")  # shows latest effects export
    
    # Preview of badges_effects.csv inside the app
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)

















