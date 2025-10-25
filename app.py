# app.py 
import json, uuid, os, time, pathlib, csv, requests
from datetime import datetime
import gradio as gr
from agent_runner import run_job_sync
import traceback, logging
from html import escape
from urllib.parse import quote   

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

    stats_refresh.click(
        _list_stats_files,
        inputs=[admin_key_in],
        outputs=[stats_status, stats_choice, stats_long, stats_log, stats_badges],
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)




