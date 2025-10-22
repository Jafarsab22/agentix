# app.py
import json, uuid, os, time, pathlib, csv
from datetime import datetime
import gradio as gr
from agent_runner import run_job_sync
import traceback, logging
from datetime import datetime
logging.basicConfig(level=logging.INFO)

def _catch_and_report(fn):
    """Wrap a Gradio handler, show a readable error, and log to results/ + console."""
    def _inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()

            # Write to results/ so you can download from the Admin section later
            err_path = RESULTS_DIR / f"ui_error_{int(time.time())}.log"
            try:
                err_path.write_text(tb, encoding="utf-8")
            except Exception:
                pass

            # Print to container logs
            print(tb, flush=True)
            logging.exception("Gradio handler failed")

            msg = f"❌ {type(e).__name__}: {e}\n\n```\n{tb}\n```"
            # Handlers return (markdown, json); pad if needed
            return (msg, "{}") if fn.__name__ in ("run_now", "queue_job") else msg
    return _inner

RESULTS_DIR = pathlib.Path("results"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = pathlib.Path("jobs"); JOBS_DIR.mkdir(parents=True, exist_ok=True)

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
# IMPORTANT: leave empty to use inline HTML (no remote renderer needed).
# If you have a hosted renderer, set this ENV to a reachable HTTPS URL template.
# You may include {category},{seed},{catalog_seed},{set_id},{csv},{price},{currency}
RENDER_URL_TPL = os.environ.get(
    "RENDER_URL_TPL",
    ""  # empty → inline renderer in agent_runner.py
)

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
    """Create a payload the runner expects. If no remote renderer is configured, pass an empty render_url to trigger inline HTML."""
    csv = ",".join([b.strip() for b in (badges or []) if str(b).strip()])

    tpl = (RENDER_URL_TPL or "").strip()

    # If tpl is empty → inline renderer in agent_runner.py
    if not tpl:
        render_url = ""
    else:
        # Allow templates that reference price/currency too
        render_url = (tpl
            .replace("{csv}", csv)
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
        "render_url": render_url,  # "" → inline HTML path in agent_runner._episode
    }

# ---------- Queue (UI only; does not run the simulation) ----------
@_catch_and_report
def queue_job(product_name: str, brand_name: str, model_name: str, badges: list[str], price, currency: str, n_iterations):
    err = _validate_inputs(product_name, price, currency, n_iterations)
    if err:
        return err, "{}"

    job_id = f"job-{uuid.uuid4().hex[:10]}"
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

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    (JOBS_DIR / f"{job_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # (Optional) storefront snapshot
    try:
        if build_storefront_from_payload and save_storefront:
            html, meta = build_storefront_from_payload(payload)
            html_path, meta_path = save_storefront(job_id, html, meta)
            payload["storefront_paths"] = {"html": html_path, "json": meta_path}
    except Exception as e:
        payload["storefront_error"] = str(e)

    msg = [
        "### ✅ Simulation request created",
        f"**Job ID:** {job_id}",
        f"**Product:** {payload['product']}",
        f"**Brand:** {payload['brand'] or '—'}",
        f"**Model:** {payload['model']}",
        f"**Price:** {payload['price']} {payload['currency']}",
        f"**Iterations:** {payload['n_iterations']}",
        f"**Selected badges:** {', '.join(payload['badges']) if payload['badges'] else 'None'}",
        "",
        "_Runner will read jobs/*.json and write aggregates under results/._",
        "_Rendering mode_: " + ("remote URL" if payload["render_url"] else "inline HTML"),
    ]
    return "\n".join(msg), json.dumps(payload, ensure_ascii=False, indent=2)

# create an export folder once
EFFECTS_DIR = pathlib.Path("results") / "effects"
EFFECTS_DIR.mkdir(parents=True, exist_ok=True)
# ---------- Run now (calls runner immediately) ----------
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

    # Identify render mode for the status line
    mode = "inline HTML" if not (payload.get("render_url") or "").strip() else "external URL"

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

        # export CSV + HTML with run metadata so we know which data belongs to what
        csv_path, html_path = _export_badge_effects(rows_sorted, payload, job_id)

        # stash paths into artifacts for the admin view
        artifacts = results.setdefault("artifacts", {})
        if csv_path:
            artifacts["effects_csv"] = csv_path
        if html_path:
            artifacts["effects_html"] = html_path

        meta = (
            f"\n\nProduct: {product_name}"
            f"\nType/Brand: {brand_name}"
            f"\nModel: {model_name}"
            f"\nPrice: {price} {currency}"
            f"\nIterations: {n_iterations}"
        )

        saved = ""
        if html_path or csv_path:
            saved = "\n\nSaved badge-effects to:" + (f"\n• HTML: {html_path}" if html_path else "") + (f"\n• CSV: {csv_path}" if csv_path else "")

        msg = f"Rendered via: {mode}{meta}\n\n" + header + "\n".join(table) + saved

    else:
        note = "No badge effects computed."
        # If the CSV exists, mention it explicitly (useful when effects were filtered out)
        art = results.get("artifacts", {}) or {}
        csv_path = art.get("table_badges") or ""
        if csv_path:
            note += f" See {csv_path} for details."
        msg = f"Rendered via: {mode}\n\n{note}"

    return msg, json.dumps(results, ensure_ascii=False, indent=2)

# --- helper to export effects with run metadata (append after your admin helpers) ---
def _export_badge_effects(rows_sorted: list[dict], payload: dict, job_id: str):
    """
    Write CSV and HTML files for the badge-effects table, including:
    product, brand/type, model, price, currency, n_iterations, job_id, timestamp.
    Returns (csv_path, html_path) as strings (or (None, None) if nothing written).
    """
    if not rows_sorted:
        return None, None

    ts = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    base = f"{ts}_{job_id}_badge_effects"
    csv_path = EFFECTS_DIR / f"{base}.csv"
    html_path = EFFECTS_DIR / f"{base}.html"

    # CSV with metadata per row
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

    # Simple HTML (table like the screenshot + a metadata block)
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

# ---------- Admin helpers ----------
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

# --- Admin helpers for the new exports (add below your existing admin helpers) ---
def _list_effect_tables(admin_key: str):
    if not ADMIN_KEY or admin_key != ADMIN_KEY:
        return ("Invalid or missing admin key.", None, None)
    if not EFFECTS_DIR.exists():
        return ("No effects/ directory yet.", None, None)
    csv_files = sorted(EFFECTS_DIR.glob("*.csv"))
    html_files = sorted(EFFECTS_DIR.glob("*.html"))
    latest_csv = str(csv_files[-1]) if csv_files else None
    latest_html = str(html_files[-1]) if html_files else None
    if not latest_csv and not latest_html:
        return ("No badge-effects exports yet. Run a simulation first.", None, None)
    msg = "Latest badge-effects exports:\n" + ("\n• " + latest_csv if latest_csv else "") + ("\n• " + latest_html if latest_html else "")
    return (msg, latest_csv, latest_html)

def _preview_effect_file(admin_key: str, path: str):
    if not ADMIN_KEY or admin_key != ADMIN_KEY:
        return gr.update(value=""), "Invalid or missing admin key."
    if not path:
        return gr.update(value=""), "Select an effects file first."
    p = pathlib.Path(path)
    if not p.exists():
        return gr.update(value=""), f"Not found: {p}"
    if p.suffix.lower() == ".html":
        return gr.update(value=p.read_text(encoding="utf-8")), f"Rendered {p}"
    return gr.update(value=""), f"Selected {p} (download via file path)."

def _list_stats_files(admin_key: str):
    if not ADMIN_KEY or admin_key != ADMIN_KEY:
        return ("Invalid or missing admin key.", None, None, None, None, None)

    res = pathlib.Path("results")
    if not res.exists():
        return ("No results/ directory yet.", None, None, None, None, None)

    # existing aggregate files
    agg_choice = res / "df_choice.csv"
    agg_long = res / "df_long.csv"
    agg_log = res / "log_compare.jsonl"

    # new: latest badge-effects exports (CSV + HTML)
    eff_dir = res / "effects"
    latest_eff_csv = None
    latest_eff_html = None
    if eff_dir.exists():
        csvs = sorted(eff_dir.glob("*.csv"))
        htmls = sorted(eff_dir.glob("*.html"))
        latest_eff_csv = str(csvs[-1]) if csvs else None
        latest_eff_html = str(htmls[-1]) if htmls else None

    msg = []
    if agg_choice.exists(): msg.append(f"• Found {agg_choice}")
    if agg_long.exists(): msg.append(f"• Found {agg_long}")
    if agg_log.exists(): msg.append(f"• Found {agg_log}")
    if latest_eff_csv:   msg.append(f"• Found {latest_eff_csv}")
    if latest_eff_html:  msg.append(f"• Found {latest_eff_html}")
    if not msg:
        msg = ["No aggregate files yet. Run a simulation first."]

    return (
        "\n".join(msg),
        str(agg_choice) if agg_choice.exists() else None,
        str(agg_long) if agg_long.exists() else None,
        str(agg_log) if agg_log.exists() else None,
        latest_eff_csv,
        latest_eff_html,
    )


@_catch_and_report
def preview_example(product_name: str, brand_name: str, model_name: str, badges: list[str], price, currency: str):
    # Build a minimal payload (no need for n_iterations; we force 1 inside preview_one)
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
    # Render inline via data URL; no files saved
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
        "Simulate how an AI agent is expected to react to selected e-commerce badges for a product.\n\n"
        "Enter the product name, category, price, and number of iterations (default: 50). The current AI agent is set to OpenAI GPT-4.1-mini.\n\n"
        "When ready, click **Run simulation now** to generate the results. A table showing the estimated effects of each badge will appear below.\n\n"
        "_Examples:_ **frame** (All-in vs. Partitioned pricing), **assurance** (e.g., Free returns), **scarcity tag** (e.g., Only 3 left in stock), **strike-through** (e.g., £120 → £89.99), **timer** (e.g., Deal ends in 2 hours), **social** (e.g., 2k bought this last month), **voucher** (e.g., 10% off code SAVE10), **bundle** (e.g., Buy 2 save 10%)."

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

    run_btn = gr.Button("Run simulation now", variant="primary")
    queue_btn = gr.Button("Queue simulation job", variant="secondary")
    preview_btn = gr.Button("Preview one example screen", variant="secondary")
    preview_view = gr.HTML(label="Preview")

    results_md = gr.Markdown()
    results_json = gr.Code(label="Results JSON (debug)", language="json")
    out_md = gr.Markdown()
    out_json = gr.Code(label="Queued job payload (for debugging)", language="json")

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
    queue_btn.click(
        fn=queue_job,
        inputs=[product, brand, model, badges, price, currency, n_iterations],
        outputs=[out_md, out_json],
    )

    with gr.Accordion("Admin preview (storefront)", open=False):
        admin_key_in = gr.Textbox(label="Admin key", type="password", placeholder="Enter ADMIN_KEY", scale=1)
        refresh = gr.Button("List storefronts")
        job_picker = gr.Dropdown(label="Job ID", choices=[], interactive=True, scale=2)
        preview_btn = gr.Button("Preview storefront")
        admin_status = gr.Markdown()
        html_view = gr.HTML(label="Storefront preview")
        refresh.click(_list_storefront_jobs, inputs=[admin_key_in], outputs=[job_picker, admin_status])
        preview_btn.click(_preview_storefront, inputs=[admin_key_in, job_picker], outputs=[html_view, admin_status])

    gr.Markdown("### Stats files (aggregates)")
    stats_refresh = gr.Button("Refresh stats")
    stats_status = gr.Markdown()
    stats_choice = gr.File(label="df_choice.csv")
    stats_long = gr.File(label="df_long.csv")
    stats_log = gr.File(label="log_compare.jsonl")
    stats_refresh.click(
        _list_stats_files,
        inputs=[admin_key_in],
        outputs=[stats_status, stats_choice, stats_long, stats_log],
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)










