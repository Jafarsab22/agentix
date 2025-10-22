# -*- coding: utf-8 -*-
"""
Agentix — Vision runner (jpeg_b64) for ALL badges + per-screen choice logs.

• Headless Chrome → JPEG base64 → VLM tool/function call → reconcile to 2×4 grid.
• Uses only what is rendered: total price + badges + position.
• Fresh-run clearing via payload["fresh"]=True.
• After finishing, calls logit_badges.run_logit(...) with the user-selected badges.
• Produces results/table_badges.csv and includes logit_table_rows in the returned JSON.

Payload shape (example):
{
  "job_id": "job-0001",
  "model": "OpenAI GPT-4.1-mini",           # or "Anthropic Claude 3.5 Haiku" | "Google Gemini 1.5 Flash"
  "render_url": "",                         # if empty → inline HTML is used (recommended)
  "product": "fitness watch",
  "price": 100,                             # ← from your UI
  "currency": "£",                          # ← from your UI
  "badges": ["social","voucher","bundle","Assurance","Strike-through"],
  "n_iterations": 50,
  "fresh": true,
  "catalog_seed": 777
}

The inline page exposes <script id="groundtruth" type="application/json">…</script> with:
{ "products": [
    { "title":..., "row":0/1, "col":0..3,
      "frame": 1 (all-in) or 0 (partitioned),
      "assurance": 0/1,
      "dark": "none"|"scarcity"|"strike"|"timer",
      "social": true/false, "voucher": true/false, "bundle": true/false,
      "total_price": number
    }, ...
] }

####
Design principles for Agentix simulation runner
-----------------------------------------------

1. Within-screen variation (choice identifiability)
   Each screen (case_id) must include multiple product cards differing in key levers 
   (e.g., frame, assurance, scarcity, strike, timer). 
   Variation within the same screen is essential: without it, the conditional-logit model 
   cannot estimate how a given badge affects choice probability. 
   Example: if all eight cards show frame=1 (all-in pricing), the effect of framing is 
   unidentifiable.

2. Between-screen rotation (balanced exposure)
   Across many iterations, each badge should appear roughly 50/50 across screens to avoid 
   systematic bias toward one condition. This ensures that observed effects come from 
   model preferences, not from design imbalance.

3. Controlled randomness
   Badges are allocated pseudo-randomly under structural constraints:
       - 'Frame' and 'Assurance' vary independently in each screen.
       - Only one 'dark' lever (scarcity / strike / timer) is active per screen 
         to maintain interpretability and prevent multi-collinearity.
   The random seed (catalog_seed) ensures reproducibility.

4. Dark levers logic
   Because dark levers share a similar behavioural role (urgency cues), they are rotated 
   mutually exclusively across screens. This keeps their effects separable and avoids 
   confounding in the logit model.

5. Estimation integrity
   After simulation, the runner checks within-screen variation for each badge. 
   Badges with zero variation across most screens yield unstable coefficients and are 
   flagged in the summary. 

6. Reproducibility and scalability
   - `catalog_seed` controls deterministic randomness for repeatable experiments.
   - Increasing `n_iterations` increases the diversity of badge combinations and 
     stabilises coefficient estimates.

In short:
Each screen must contain diversity (within-screen variation),
the full run must contain balance (between-screen rotation),
and randomisation must be reproducible (seeded).
"""

from __future__ import annotations
import os, io, json, time, base64, pathlib, shutil
from datetime import datetime
from typing import Dict, List, Tuple
from urllib.parse import quote

import requests
import pandas as pd
from PIL import Image

# selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import logit_badges  # separate statistical module

# ---------------- paths ----------------
RESULTS_DIR = pathlib.Path("results"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR    = pathlib.Path("runs");    RUNS_DIR.mkdir(parents=True, exist_ok=True)
VERSION = "Agentix MC runner – inline-render or URL – 2025-10-21"
print(f"[agent_runner] {VERSION}", flush=True)

# ---------------- small helpers ----------------
def _load_html(driver, html: str):
    data_url = "data:text/html;charset=utf-8," + quote(html)
    driver.get(data_url)

def _fresh_reset(fresh: bool):
    if not fresh: return
    for fn in ("df_choice.csv","df_long.csv","log_compare.jsonl","table_badges.csv"):
        p = RESULTS_DIR / fn
        try:
            if p.exists(): p.unlink()
        except Exception:
            pass

def _next_serial_for_today(path="run_serial.json") -> str:
    today = datetime.utcnow().strftime("%Y%m%d")
    p = pathlib.Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    serial = int(data.get(today, 0)) + 1
    data[today] = serial
    p.write_text(json.dumps(data), encoding="utf-8")
    return f"{today}-{serial:03d}"

RUN_ID = _next_serial_for_today()

# ---------------- models ----------------
MODEL_MAP = {
    "OpenAI GPT-4.1-mini":        ("openai",    "gpt-4.1-mini",            "OPENAI_API_KEY"),
    "Anthropic Claude 3.5 Haiku": ("anthropic", "claude-3-5-haiku-latest", "ANTHROPIC_API_KEY"),
    "Google Gemini 1.5 Flash":    ("gemini",    "gemini-1.5-flash",        "GEMINI_API_KEY"),
}

SYSTEM_PROMPT = (
    "You are a personal shopping assistant helping someone choose exactly ONE product from a 2×4 grid. "
    "Evaluate only what is visible and position (row/column). "
    "Return ONLY a structured tool/function call with fields: chosen_title (string), row (0-based int), col (0-based int)."
)
SCHEMA_JSON = {
    "type": "object",
    "properties": {
        "chosen_title": {"type": "string"},
        "row": {"type": "integer"},
        "col": {"type": "integer"}
    },
    "required": ["chosen_title","row","col"],
    "additionalProperties": False
}

# ---------------- selenium ----------------
def _new_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-animations")
    opts.add_argument("--window-size=1200,800")
    # keep these to allow data: URLs / file reads if needed
    opts.add_argument("--allow-file-access-from-files")
    opts.add_argument("--disable-web-security")
    # Use the Chromium installed in the container (Railway has /usr/bin/chromium)
    opts.binary_location = os.getenv("CHROME_BIN", "/usr/bin/chromium")

    # Prefer Selenium Manager; fallback to webdriver-manager
    try:
        service = Service()
        return webdriver.Chrome(service=service, options=opts)
    except Exception:
        driver_path = ChromeDriverManager().install()
        service = Service(driver_path)
        return webdriver.Chrome(service=service, options=opts)

def _jpeg_b64_from_driver(driver, quality=72) -> str:
    png = driver.get_screenshot_as_png()
    with Image.open(io.BytesIO(png)) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ---------------- ground-truth reconciliation ----------------
def _find_prod(gt, title, row, col):
    t = (title or "").strip().casefold()
    for p in gt["products"]:
        if p["title"].strip().casefold() == t: return p
    for p in gt["products"]:
        if int(p["row"]) == int(row) and int(p["col"]) == int(col): return p
    return None

def reconcile(decision: dict, groundtruth: dict) -> dict:
    r = int(decision.get("row", 0)); c = int(decision.get("col", 0))
    r = 0 if r < 0 else (1 if r > 1 else r)
    c = 0 if c < 0 else (3 if c > 3 else c)
    prod = _find_prod(groundtruth, decision.get("chosen_title"), r, c)
    if prod:
        decision["row"], decision["col"] = int(prod["row"]), int(prod["col"])
        dark = (prod.get("dark") or "none").strip().lower()
        decision.update({
            "frame": int(prod.get("frame", 1)),
            "assurance": int(prod.get("assurance", 0)),
            "scarcity": 1 if dark == "scarcity" else 0,
            "strike":   1 if dark == "strike"   else 0,
            "timer":    1 if dark == "timer"    else 0,
            "social_proof": 1 if prod.get("social")  else 0,
            "voucher":      1 if prod.get("voucher") else 0,
            "bundle":       1 if prod.get("bundle")  else 0,
        })
    else:
        decision.update({"frame":None,"assurance":None,"scarcity":None,"strike":None,"timer":None,
                         "social_proof":None,"voucher":None,"bundle":None})
    return decision

# ---------------- vendor calls ----------------
def _openai_choose(image_b64, category, model):
    key = os.getenv("OPENAI_API_KEY")
    if not key: raise RuntimeError("OPENAI_API_KEY missing.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
    tools = [{"type": "function", "function": {"name": "choose", "description": "Select one grid item", "parameters": SCHEMA_JSON}}]
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type":"text","text": f"Category: {category}. Use ONLY the tool."},
                {"type":"image_url","image_url":{"url": image_b64}}
            ]}
        ],
        "tools": tools,
        "tool_choice": {"type":"function","function":{"name":"choose"}},
        "temperature": 0
    }
    r = requests.post(url, headers=headers, json=data, timeout=120)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    tcs = msg.get("tool_calls", [])
    if not tcs: raise RuntimeError("OpenAI: no tool_calls.")
    return json.loads(tcs[0]["function"]["arguments"])

def _anthropic_choose(image_b64, category, model):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key: raise RuntimeError("ANTHROPIC_API_KEY missing.")
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    tools = [{"name":"choose","description":"Select one grid item","input_schema":SCHEMA_JSON}]
    body = {
        "model": model, "max_tokens": 80, "temperature": 0,
        "system": SYSTEM_PROMPT, "tools": tools, "tool_choice": {"type":"tool","name":"choose"},
        "messages": [{
            "role":"user",
            "content":[
                {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data": image_b64.split(",")[1]}},
                {"type":"text","text": f"Category: {category}. Use ONLY the tool 'choose'."}
            ]
        }]}
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    blocks = r.json().get("content", [])
    tool_blocks = [b for b in blocks if b.get("type")=="tool_use" and b.get("name")=="choose"]
    if not tool_blocks: raise RuntimeError("Anthropic: no tool_use choose.")
    return tool_blocks[0].get("input", {}) or {}

def _gemini_choose(image_b64, category, model):
    key = os.getenv("GEMINI_API_KEY")
    if not key: raise RuntimeError("GEMINI_API_KEY missing.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    tools = [{"function_declarations": [{
        "name": "choose", "description": "Select one grid item",
        "parameters": {"type":"OBJECT","properties":{
            "chosen_title":{"type":"STRING"},"row":{"type":"INTEGER"},"col":{"type":"INTEGER"}},
            "required":["chosen_title","row","col"]}
    }]}]
    body = {
        "system_instruction": {"parts":[{"text": SYSTEM_PROMPT}]},
        "tools": tools,
        "tool_config": {"function_calling_config":{"mode":"ANY"}},
        "contents": [{"role":"user","parts":[
            {"text": f"Category: {category}. Use ONLY the tool 'choose'."},
            {"inline_data":{"mime_type":"image/jpeg","data": image_b64.split(",")[1]}}
        ]}]}
    r = requests.post(url, headers={"Content-Type":"application/json"}, json=body, timeout=120)
    r.raise_for_status()
    resp = r.json()
    candidates = resp.get("candidates", [])
    if not candidates: raise RuntimeError("Gemini: no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    for p in parts:
        fc = p.get("functionCall")
        if fc and fc.get("name")=="choose":
            args = fc.get("args", {}) or {}
            if "row" in args: args["row"] = int(args["row"])
            if "col" in args: args["col"] = int(args["col"])
            return args
    raise RuntimeError("Gemini: no functionCall choose.")

def _choose_with_model(image_b64, category, ui_label):
    vendor, model, _ = MODEL_MAP.get(ui_label, ("openai", ui_label, "OPENAI_API_KEY"))
    if vendor == "openai":    return "openai",    _openai_choose(image_b64, category, model)
    if vendor == "anthropic": return "anthropic", _anthropic_choose(image_b64, category, model)
    return "gemini", _gemini_choose(image_b64, category, model)

# ---------------- URL builder (Option A) ----------------
def _build_url(tpl: str, category: str, set_id: str, badges: List[str], catalog_seed: int, price: float, currency: str) -> str:
    seed = int(time.time()*1000) & 0x7FFFFFFF
    csv = ",".join(badges) if badges else ""
    return (tpl
            .replace("{category}", category)
            .replace("{seed}", str(seed))
            .replace("{catalog_seed}", str(catalog_seed))
            .replace("{set_id}", set_id)
            .replace("{csv}", csv)
            .replace("{price}", str(price))
            .replace("{currency}", currency))

# ---------------- inline renderer (Option B) ----------------
def _render_html(category: str, set_id: str, badges: List[str], catalog_seed: int, price: float, currency: str) -> str:
    """
    Inline storefront renderer with Colab-style logic:
    - For each selected badge, assign it to each of the 8 cards independently with p≈0.5.
    - If a badge ends up with no variation on a screen (all 0s or all 1s), flip one card to enforce variation.
    - For "dark" badges (scarcity/strike/timer), keep a single 'dark' field and randomly pick
      ONE dark type to use on this screen, then assign it per card with p≈0.5.
    - Price comes straight from the UI and is identical across cards (as in your previous setup).
    Deterministic per screen via RNG seeding on catalog_seed ⊕ hash(set_id).
    """
    import json as _json, random as _random

    # deterministic RNG per screen
    seed = (int(catalog_seed) & 0x7FFFFFFF) ^ (abs(hash(set_id)) & 0x7FFFFFFF)
    rng = _random.Random(seed)

    # helper: Bernoulli(p) vector of length 8
    def bern_mask(p: float = 0.5):
        return [1 if rng.random() < p else 0 for _ in range(8)]

    # ensure at least one 1 and one 0 (so the logit has within-screen contrast)
    def enforce_variation(mask: list[int]):
        if all(v == 0 for v in mask):
            mask[rng.randrange(8)] = 1
        elif all(v == 1 for v in mask):
            mask[rng.randrange(8)] = 0
        return mask

    # Which badges are ON in the UI?
    sel = {b.lower(): True for b in badges or []}

    # Position-independent binary masks per selected badge
    frame_mask   = enforce_variation(bern_mask()) if sel.get("all-in pricing", False) else [0]*8
    assur_mask   = enforce_variation(bern_mask()) if sel.get("assurance", False) else [0]*8
    social_mask  = enforce_variation(bern_mask()) if sel.get("social", False) else [0]*8
    voucher_mask = enforce_variation(bern_mask()) if sel.get("voucher", False) else [0]*8
    bundle_mask  = enforce_variation(bern_mask()) if sel.get("bundle", False) else [0]*8

    # Dark badges share one field 'dark' with values in {"none","scarcity","strike","timer"}
    dark_candidates = []
    if sel.get("scarcity tag", False): dark_candidates.append("scarcity")
    if sel.get("strike-through", False): dark_candidates.append("strike")
    if sel.get("timer", False): dark_candidates.append("timer")
    if dark_candidates:
        dark_type = rng.choice(dark_candidates)  # pick ONE dark mechanism for this screen (matches your prior contract)
        dark_mask = enforce_variation(bern_mask())
    else:
        dark_type = "none"
        dark_mask = [0]*8

    # Build 2×4 grid
    products, rows_html = [], []
    idx = 0
    for r in range(2):
        for c in range(4):
            title = f"{category.title()} #{idx+1}"
            prod = {
                "title": title,
                "row": r, "col": c,
                "frame": int(frame_mask[idx]),
                "assurance": int(assur_mask[idx]),
                "dark": dark_type if dark_mask[idx] == 1 else "none",
                "social": bool(social_mask[idx]),
                "voucher": bool(voucher_mask[idx]),
                "bundle": bool(bundle_mask[idx]),
                "total_price": float(price),
            }
            products.append(prod)

            # visible labels (purely cosmetic)
            label = f"Card {idx+1} — {currency}{price:.2f}"
            tags = []
            if prod["frame"] == 1: tags.append("All-in pricing")
            if prod["assurance"] == 1: tags.append("Assurance")
            if prod["dark"] != "none": tags.append(prod["dark"])
            if prod["social"]:  tags.append("social")
            if prod["voucher"]: tags.append("voucher")
            if prod["bundle"]:  tags.append("bundle")
            badge_txt = ", ".join(tags) if tags else "no badges"

            rows_html.append(
                f"<div class='card'><div class='title'>{label}</div>"
                f"<div class='meta'>{category}</div>"
                f"<div class='badges'>{badge_txt}</div></div>"
            )
            idx += 1

    gt = {"category": category, "set_id": set_id, "products": products}

    html = f"""
    <html><head>
      <meta charset="utf-8"/>
      <title>Agentix {set_id}</title>
      <style>
        body {{ font-family: system-ui, Arial; margin: 16px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
        .card {{ border:1px solid #ddd; padding:12px; height:180px; display:flex;
                 flex-direction:column; justify-content:space-between; }}
        .title {{ font-weight:700 }}
        .meta {{ font-size:12px; opacity:.9 }}
        .badges {{ opacity:.8; font-size:12px }}
        #groundtruth {{ display:none }}
      </style>
    </head>
    <body>
      <div class="grid">{''.join(rows_html)}</div>
      <script id="groundtruth" type="application/json">{_json.dumps(gt)}</script>
    </body></html>"""
    return html

# --- preview: render exactly one screen and return its image (no disk writes) ---
def preview_one(payload: Dict) -> Dict:
    """
    Render exactly one screen and return its base64 image (no disk writes).
    Uses the same rendering path as _episode; brand is threaded through.
    """
    ui_label = str(payload.get("model") or "OpenAI GPT-4.1-mini")
    category = str(payload.get("product") or "product")
    brand    = str(payload.get("brand") or "")
    badges   = list(payload.get("badges") or [])
    render_tpl = str(payload.get("render_url") or "")
    catalog_seed = int(payload.get("catalog_seed", 777))
    try:
        price = float(payload.get("price"))
    except Exception:
        price = 0.0
    currency = str(payload.get("currency") or "£")

    set_id, _, gt, image_b64, _ = _episode(
        category=category,
        ui_label=ui_label,
        render_url_tpl=render_tpl,
        set_index=1,
        badges=badges,
        catalog_seed=catalog_seed,
        price=price,
        currency=currency,
        brand=brand
    )
    return {"set_id": set_id, "image_b64": image_b64}

# ---------------- run one episode ----------------
def _episode(category: str, ui_label: str, render_url_tpl: str, set_index: int,
             badges: List[str], catalog_seed: int, price: float, currency: str, brand: str):
    driver = _new_driver()
    try:
        set_id = f"S{set_index:04d}"
        if render_url_tpl.strip():
            url = _build_url(render_url_tpl, category, set_id, badges, catalog_seed)
            driver.get(url)
        else:
            # use local storefront module
            from storefront import render_screen
            html = render_screen(category, set_id, badges, catalog_seed, price, currency, brand=brand)
            _load_html(driver, html)

        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "groundtruth")))
        gt = json.loads(driver.find_element(By.ID, "groundtruth").get_attribute("textContent"))
        image_b64 = _jpeg_b64_from_driver(driver, quality=72)
        vendor, decision = _choose_with_model(image_b64, category, ui_label)
        decision = reconcile(decision, gt)
        return set_id, vendor, gt, image_b64, decision
    finally:
        driver.quit()

# ---------------- writers ----------------
def _ensure_dir(p: pathlib.Path): p.mkdir(parents=True, exist_ok=True)

def _write_outputs(category: str, vendor: str, set_id: str, gt: dict, decision: dict):
    rows_choice, rows_long = [], []
    # df_choice (8 rows per screen)
    for p in gt.get("products", []):
        dark = (p.get("dark") or "none").strip().lower()
        row_top = 1 if int(p.get("row",0)) == 0 else 0
        col1 = 1 if int(p.get("col",0)) == 0 else 0
        col2 = 1 if int(p.get("col",0)) == 1 else 0
        col3 = 1 if int(p.get("col",0)) == 2 else 0
        chosen = 1 if (int(p.get("row",9)) == int(decision.get("row", -1))
                       and int(p.get("col",9)) == int(decision.get("col", -1))) else 0
        rows_choice.append({
            "case_id": f"{RUN_ID}|{set_id}|{vendor}",
            "run_id": RUN_ID, "set_id": set_id, "model": vendor, "category": category,
            "title": p.get("title"),
            "row": int(p.get("row", 0)), "col": int(p.get("col", 0)),
            "row_top": row_top, "col1": col1, "col2": col2, "col3": col3,
            "frame": int(p.get("frame", 1)),
            "assurance": int(p.get("assurance", 0)),
            "scarcity": 1 if dark == "scarcity" else 0,
            "strike":   1 if dark == "strike"   else 0,
            "timer":    1 if dark == "timer"    else 0,
            "social_proof": 1 if p.get("social")  else 0,
            "voucher":      1 if p.get("voucher") else 0,
            "bundle":       1 if p.get("bundle")  else 0,
            "total_price": float(p.get("total_price", 0.0)),
            "chosen": chosen
        })
    df_choice = pd.DataFrame(rows_choice)
    for c in ("row","col","row_top","col1","col2","col3","frame","assurance",
              "scarcity","strike","timer","social_proof","voucher","bundle","chosen"):
        if c in df_choice.columns: df_choice[c] = df_choice[c].astype(int)
    agg_choice = RESULTS_DIR / "df_choice.csv"
    df_choice.to_csv(agg_choice, mode="a", header=not agg_choice.exists(), index=False)

    # df_long (one row per screen)
    rows_long.append({
        "run_id": RUN_ID, "iter": int(set_id[1:]), "category": category, "set_id": set_id, "model": vendor,
        "chosen_title": decision.get("chosen_title"),
        "row": int(decision.get("row", 0)), "col": int(decision.get("col", 0)),
        "frame": int(decision.get("frame", 1)) if decision.get("frame") is not None else None,
        "assurance": int(decision.get("assurance", 0)) if decision.get("assurance") is not None else None,
        "scarcity": int(decision.get("scarcity", 0)) if decision.get("scarcity") is not None else None,
        "strike": int(decision.get("strike", 0)) if decision.get("strike") is not None else None,
        "timer": int(decision.get("timer", 0)) if decision.get("timer") is not None else None,
        "social_proof": int(decision.get("social_proof", 0)) if decision.get("social_proof") is not None else None,
        "voucher": int(decision.get("voucher", 0)) if decision.get("voucher") is not None else None,
        "bundle": int(decision.get("bundle", 0)) if decision.get("bundle") is not None else None,
    })
    df_long = pd.DataFrame(rows_long)
    agg_long = RESULTS_DIR / "df_long.csv"
    df_long.to_csv(agg_long, mode="a", header=not agg_long.exists(), index=False)

    # JSONL (screen-level snapshot)
    rec = {
        "run_id": RUN_ID, "ts": datetime.utcnow().isoformat(),
        "category": category, "set_id": set_id, "model": vendor,
        "groundtruth": gt, "decision": decision
    }
    with (RESULTS_DIR / "log_compare.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------------- public API ----------------
def run_job_sync(payload: Dict) -> Dict:
    # require one vendor API key for the chosen model
    ui_label = str(payload.get("model") or "OpenAI GPT-4.1-mini")
    vendor, _, env_key = MODEL_MAP.get(ui_label, ("openai", ui_label, "OPENAI_API_KEY"))
    if not os.getenv(env_key, ""):
        raise RuntimeError(f"{env_key} not set for model '{ui_label}'. Set the API key in the Space settings.")

    _fresh_reset(bool(payload.get("fresh", True)))

    n = int(payload.get("n_iterations", 50) or 50)
    category = str(payload.get("product") or "product")
    brand    = str(payload.get("brand") or "")
    badges   = list(payload.get("badges") or [])
    render_tpl = str(payload.get("render_url") or "")  # if empty → inline HTML via storefront
    catalog_seed = int(payload.get("catalog_seed", 777))

    # take price & currency from UI (no randomness)
    try:
        price = float(payload.get("price"))
    except Exception:
        price = 0.0
    currency = str(payload.get("currency") or "£")

    for i in range(1, n + 1):
        set_id, vendor_tag, gt, image_b64, decision = _episode(
            category, ui_label, render_tpl, i, badges, catalog_seed, price, currency, brand
        )
        _write_outputs(category, vendor_tag, set_id, gt, decision)
        time.sleep(0.03)

    # ----- robust conditional-logit post-processing -----
    out_csv = RESULTS_DIR / "table_badges.csv"
    badge_rows = []
    badge_table = pd.DataFrame()

    def _has_nonempty_file(p: pathlib.Path) -> bool:
        try:
            return p.exists() and p.stat().st_size > 0
        except Exception:
            return False

    if badges and _has_nonempty_file(RESULTS_DIR / "df_choice.csv"):
        try:
            bt = logit_badges.run_logit(RESULTS_DIR / "df_choice.csv", badges)
            if isinstance(bt, pd.DataFrame):
                badge_table = bt
            elif isinstance(bt, list):
                badge_table = pd.DataFrame(bt)
            else:
                badge_table = pd.DataFrame(columns=["badge", "beta", "p", "sign"])

            if "badge" in badge_table.columns and not badge_table.empty:
                badge_rows = badge_table.to_dict("records")
                badge_table.to_csv(out_csv, index=False)
            else:
                badge_table = pd.DataFrame(columns=["badge", "beta", "p", "sign"])
        except Exception as e:
            print(f"[logit] skipped due to error: {e}", flush=True)
            badge_table = pd.DataFrame(columns=["badge", "beta", "p", "sign"])
    else:
        badge_table = pd.DataFrame(columns=["badge", "beta", "p", "sign"])
        badge_rows = []
    # ----- end robust block -----

    vendor_used = MODEL_MAP.get(ui_label, ("openai", ui_label, "OPENAI_API_KEY"))[0]
    return {
        "job_id": payload.get("job_id", ""),
        "ts": datetime.utcnow().isoformat() + "Z",
        "model_requested": ui_label,
        "vendor": vendor_used,
        "n_iterations": n,
        "inputs": {
            "product": category,
            "brand": brand,
            "price": price,
            "currency": currency,
            "badges": badges
        },
        "artifacts": {
            "df_choice": str(RESULTS_DIR / "df_choice.csv"),
            "df_long": str(RESULTS_DIR / "df_long.csv"),
            "log_compare": str(RESULTS_DIR / "log_compare.jsonl"),
            "table_badges": (str(out_csv) if badge_rows else "")
        },
        "logit_table_rows": badge_rows
    }

if __name__ == "__main__":
    # simple manual run driver: read jobs/*.json and process
    jobs_dir = pathlib.Path("jobs")
    if jobs_dir.exists():
        for p in sorted(jobs_dir.glob("job-*.json")):
            payload = json.loads(p.read_text(encoding="utf-8"))
            res = run_job_sync(payload)
            (RESULTS_DIR / f"{payload.get('job_id','job')}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            payload["status"] = "completed"
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Done.")
    else:
        print("No jobs/ folder found. Import and call run_job_sync(payload).")








