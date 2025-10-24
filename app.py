# app.py — Part 1/2
import json, uuid, os, time, pathlib, csv, requests
from datetime import datetime
import gradio as gr
from agent_runner import run_job_sync
import traceback, logging
from html import escape

logging.basicConfig(level=logging.INFO)


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
    "All-in pricing",
    "Partitioned pricing",
    "Assurance",
    "Scarcity tag",
    "Strike-through",
    "Timer",
    "social",
    "voucher",
    "bundle",
]
CURRENCY_CHOICES = ["£", "$", "EUR"]

# Admin secret (set in platform env)
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")

# ---------- Renderer selection ----------
RENDER_URL_TPL = os.environ.get("RENDER_URL_TPL", "")  # empty → inline HTML in agent_runner

# ---------- helpers ----------

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
    fieldnames = [
        "job_id", "timestamp", "product", "brand", "model",
        "price", "currency", "n_iterations", "badge", "beta", "p", "sign"
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
                "p": r.get("p", ""),
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
        "<html><head><meta charset='utf-8'><title>Badge effects</title>",
        "<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;padding:16px}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:8px}"
        "th{text-align:left;background:#f6f6f6}"
        "td.num{text-align:right}</style>",
        "</head><body>",
        "<h2>Badge effects</h2>",
        "<h3>Run metadata</h3><table>",
    ]
    for k, v in meta_rows:
        parts.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
    parts.append("</table>")
    parts.append("<h3 style='margin-top:18px'>Effects table</h3>")
    parts.append("<table>")
    parts.append("<tr><th>Badge</th><th>β (effect size)</th><th>p (&lt;0.05 is significant)</th>"
                 "<th>Effect (0=no effect; +=positive effect; -=negative effect)</th></tr>")
    for r in rows_sorted:
        parts.append(
            "<tr>"
            f"<td>{r.get('badge','')}</td>"
            f"<td class='num'>{_fmt(r.get('beta'))}</td>"
            f"<td class='num'>{_fmt(r.get('p'))}</td>"
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

    results = run_job_sync(payload)

    # Build the badge-effects table from results["logit_table_rows"]
    rows = results.get("logit_table_rows") or []

    if rows:
        rows_sorted = sorted(rows, key=lambda r: str(r.get("badge", "")))
        header = "### Badge Effects\n\n"
        table = [
            "| Badge | β (effect size) | p (<0.05 is significant) | Effect (0=no effect; +=positive effect; -=negative effect) |",
            "|---|---:|---:|:---:|",
        ]
        def _fmt(x, nd=3):
            try:
                return f"{float(x):.{nd}f}"
            except Exception:
                return "—"
        for r in rows_sorted:
            table.append(
                f"| {r.get('badge','')} | {_fmt(r.get('beta'))} | {_fmt(r.get('p'))} | {r.get('sign','0')} |"
            )
        # local export (optional download via Stats)
        csv_path, html_path = _export_badge_effects(rows_sorted, payload, job_id)
        artifacts = results.setdefault("artifacts", {})
        if csv_path: artifacts["effects_csv"] = csv_path
        if html_path: artifacts["effects_html"] = html_path
        msg = header + "\n".join(table)
    else:
        msg = "No badge effects computed."

    # Persist to Hostinger DB via PHP endpoints (best-effort)
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

    return msg, json.dumps(results, ensure_ascii=False, indent=2)


@_catch_and_report
def search_database(product_name: str):
    """Query Agentix DB for badge effects. Returns (markdown_table, pretty_json)."""
    product = (product_name or "").strip()
    if not product:
        return "Enter a product name to search.", "{}"

    url = "https://aireadyworkforce.pro/Agentix/searchAgentix.php"

    def _parse_payload(text, ct):
        try:
            return json.loads(text)
        except Exception:
            return {"ok": False, "message": f"Non-JSON response (CT={ct}): {text[:160]}"}

    # --- 1) Primary: POST JSON ---
    try:
        resp = requests.post(url, json={"product": product}, timeout=12)
        ct = (resp.headers.get("content-type") or "").lower()
        data = _parse_payload(resp.text, ct) if "application/json" in ct else {"ok": False, "message": f"CT={ct} body={resp.text[:160]}"}
    except Exception as e:
        data = {"ok": False, "message": f"POST failed: {e}"}

    # --- 2) Fallbacks if needed ---
    need_fallback = (not data.get("ok")) or (len(data.get("runs", [])) == 0 and len(data.get("rows", [])) == 0)
    if need_fallback:
        try:
            # Fallback GET (works in your browser)
            resp2 = requests.get(url, params={"product": product, "limit": 50}, timeout=12)
            ct2 = (resp2.headers.get("content-type") or "").lower()
            data2 = _parse_payload(resp2.text, ct2) if "application/json" in ct2 else {"ok": False, "message": f"CT={ct2} body={resp2.text[:160]}"}
            # Prefer the successful payload
            if data2.get("ok") and (len(data2.get("runs", [])) or len(data2.get("rows", []))):
                data = data2
        except Exception as e:
            # keep original data; we’ll surface the better message below
            if not data.get("ok"):
                data = {"ok": False, "message": f"{data.get('message','')} | GET failed: {e}"}

    # --- 3) Normalize shapes (rows vs runs+effects) ---
    rows = []
    if "rows" in data and data["rows"]:
        # Some versions return a flat list of rows
        rows = data["rows"]
    else:
        runs = data.get("runs", [])
        effects = data.get("effects", [])
        if runs:
            top_run_id = runs[0].get("run_id")
            rows = [r for r in effects if r.get("run_id") == top_run_id]

    if not data.get("ok"):
        # server-level error
        return data.get("message", "No relevant results found."), json.dumps(data, ensure_ascii=False, indent=2)

    if not rows:
        # ok==True but no matching rows
        return "No relevant results found.", json.dumps(data, ensure_ascii=False, indent=2)

    # --- 4) Render Markdown table for the UI ---
    lines = [
        "### Badge Effects (from database)\n",
        "| Badge | β (effect size) | p (<0.05 significant) | Effect |",
        "|---|---:|---:|:---:|",
    ]
    for r in rows:
        badge = r.get("badge", "")
        beta  = r.get("beta", "")
        pval  = r.get("p_value", r.get("p", ""))
        sign  = r.get("sign", "0")
        lines.append(f"| {badge} | {beta} | {pval} | {sign} |")

    return "\n".join(lines), json.dumps(data, ensure_ascii=False, indent=2)

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

    # Consolidated badge effects file (latest export copied to results/badges_effects.csv)
    badges_effects_path = res / "badges_effects.csv"
    eff_dir = res / "effects"
    if eff_dir.exists():
        csvs = sorted(eff_dir.glob("*.csv"))
        if csvs:
            import shutil
            try:
                shutil.copyfile(csvs[-1], badges_effects_path)
            except Exception:
                pass

    msg = []
    if agg_choice.exists(): msg.append(f"• Found {agg_choice}")
    if agg_long.exists():   msg.append(f"• Found {agg_long}")
    if agg_log.exists():    msg.append(f"• Found {agg_log}")
    if badges_effects_path.exists(): msg.append(f"• Found {badges_effects_path}")
    if not msg:
        msg = ["No aggregate files yet. Run a simulation first."]

    return (
        "\n".join(msg),
        str(agg_choice) if agg_choice.exists() else None,
        str(agg_long)   if agg_long.exists()   else None,
        str(agg_log)    if agg_log.exists()    else None,
        str(badges_effects_path) if badges_effects_path.exists() else None,
    )


@_catch_and_report
def preview_example(product_name: str, brand_name: str, model_name: str, badges: list[str], price, currency: str):
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


# ---------- UI ----------
with gr.Blocks(title="Agentix - AI Agent Buying Behavior") as demo:
    gr.Markdown(
        "# Agentix\n"
        "Search our database for prior significant results before you simulate. "
        "If nothing is found, run a new simulation to estimate badge effects.\n\n"
    )
    with gr.Row():
        product = gr.Textbox(label="Product name", placeholder="e.g., smart phone, washing machine", scale=2)
        brand = gr.Textbox(label="Brand (optional)", placeholder="e.g., Apple, Samsung", scale=1)
        model = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="AI Agent", scale=1)
    with gr.Row():
        price = gr.Number(label="Price", value=0.0, precision=2)
        currency = gr.Dropdown(choices=CURRENCY_CHOICES, value=CURRENCY_CHOICES[0], label="Currency")
        n_iterations = gr.Number(label="Iterations", value=50, precision=0)
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

    stats_refresh.click(
        _list_stats_files,
        inputs=[admin_key_in],
        outputs=[stats_status, stats_choice, stats_long, stats_log, stats_badges],
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)


