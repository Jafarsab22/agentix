# app.py
import json, uuid, os, time, pathlib
from datetime import datetime
import gradio as gr
from agent_runner import run_job_sync
import traceback, logging

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

# ---------- Run now (calls runner immediately) ----------
@_catch_and_report
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
    rows = results.get("logit_table_rows") or []
    mode = "remote URL" if payload.get("render_url") else "inline HTML"

    def _fmt(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "—"

    if rows:
        # stable ordering by badge label
        rows_sorted = sorted(rows, key=lambda r: str(r.get("badge", "")))
        #header = f"Rendered via: {mode}\n\n"
        header = "Badges Effects"
        table = [
            "| Badge | β (effect size) | p (<0.05 is significant) | Effect (0=no effect; +=positive effect; -=negative effect) |",
            "|---|---:|---:|:---:|",
        ]
        for r in rows_sorted:
            table.append(
                f"| {r.get('badge','')} | {_fmt(r.get('beta'))} | {_fmt(r.get('p'))} | {r.get('sign','0')} |"
            )
        msg = header + "\n".join(table)
    else:
        note = "No badge effects computed."
        # If the CSV exists, mention it explicitly (useful when effects were filtered out)
        art = results.get("artifacts", {}) or {}
        csv_path = art.get("table_badges") or ""
        if csv_path:
            note += f" See {csv_path} for details."
        msg = f"Rendered via: {mode}\n\n{note}"

    return msg, json.dumps(results, ensure_ascii=False, indent=2)


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

def _list_stats_files(admin_key: str):
    if not ADMIN_KEY or admin_key != ADMIN_KEY:
        return ("Invalid or missing admin key.", None, None, None)
    res = pathlib.Path("results")
    if not res.exists():
        return ("No results/ directory yet.", None, None, None)
    agg_choice = res / "df_choice.csv"
    agg_long = res / "df_long.csv"
    agg_log = res / "log_compare.jsonl"
    msg = []
    if agg_choice.exists(): msg.append(f"• Found {agg_choice}")
    if agg_long.exists(): msg.append(f"• Found {agg_long}")
    if agg_log.exists(): msg.append(f"• Found {agg_log}")
    if not msg: msg = ["No aggregate files yet. Run a simulation first."]
    return ("\n".join(msg),
            str(agg_choice) if agg_choice.exists() else None,
            str(agg_long) if agg_long.exists() else None,
            str(agg_log) if agg_log.exists() else None)

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

    results_md = gr.Markdown()
    results_json = gr.Code(label="Results JSON (debug)", language="json")
    out_md = gr.Markdown()
    out_json = gr.Code(label="Queued job payload (for debugging)", language="json")

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






