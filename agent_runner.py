# -*- coding: utf-8 -*-
"""
Agentix — Vision runner (jpeg_b64) for per-card, two-stage design (pricing frame + one non-frame badge).
Version: v1.7 (2025-10-25)

Key changes in v1.7
- Aligns with the two-stage experimental design:
  (1) Pricing frame (all-in vs partitioned) is assigned independently 50/50 per screen with 4/4 blocking
      when the user enables the pricing comparison; otherwise it is fixed (defaulting to all-in).
  (2) Exactly one non-frame badge is drawn uniformly per card from the user-selected set.
- Ground-truth ingestion remains tolerant to both legacy and v1.7 storefront schemas.
- The logit downstream should now treat 'frame' as a first-class effect and report it in the table/CSV.

Payload shape (example):
{
  "job_id": "job-0001",
  "model": "OpenAI GPT-4.1-mini",
  "render_url": "",                         # if empty → inline HTML is used (recommended)
  "product": "fitness watch",
  "brand": "",
  "price": 100,
  "currency": "£",
  "badges": ["All-in v. partitioned pricing","Assurance","Strike-through","Timer"],
  "n_iterations": 50,
  "fresh": true,
  "catalog_seed": 777
}

The inline page exposes <div id="groundtruth">…</div> (textContent = JSON) with either:
  • {"products": [ ... ]}   or   • [ ... ]   (runner handles both).

Each product row contains:
  title, row, col,
  frame (1=all-in, 0=partitioned),
  assurance, scarcity, strike, timer, social_proof, voucher, bundle,
  price and ln_price.

Design principles (v1.7)
- Within-screen variation: each 2×4 screen mixes price levels (Latin-square placement) and levers.
- Pricing frame: mutually exclusive; if randomised, exactly 4 all-in and 4 partitioned per screen.
- One non-frame badge per card drawn uniformly from selected set (if any).
- Reproducibility: all randomness is keyed by catalog_seed, brand, category, and set_id.
"""

from __future__ import annotations
import os, io, json, time, base64, pathlib, shutil, math
from datetime import datetime
from typing import Dict, List, Tuple
from urllib.parse import quote
import requests
import pandas as pd
from PIL import Image
import numpy as np
from hashlib import blake2b

# selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, WebDriverException
from urllib3.exceptions import ReadTimeoutError

import logit_badges  # statistical module (must include 'frame' in EFFECT_VARS)

# ---------------- paths ----------------
RESULTS_DIR = pathlib.Path("results"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR    = pathlib.Path("runs");    RUNS_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR    = pathlib.Path("pages");   HTML_DIR.mkdir(parents=True, exist_ok=True)

VERSION = "Agentix MC runner – inline-render or URL – 2025-10-25 (v1.7)"
print(f"[agent_runner] {VERSION}", flush=True)

# ---------------- small helpers ----------------
def _load_html(driver, html: str):
    data_url = "data:text/html;charset=utf-8," + quote(html)
    driver.get(data_url)

def _fresh_reset(fresh: bool):
    if not fresh: 
        return
    for fn in ("df_choice.csv","df_long.csv","log_compare.jsonl","table_badges.csv"):
        p = RESULTS_DIR / fn
        try:
            if p.exists():
                p.unlink()
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

#enforce badges variations; the price frame between all-in v. partitioned is enforced in storefront.py file
def _rng_from_job(job_id: str) -> np.random.Generator:
    h = blake2b(job_id.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(h, "little")
    return np.random.default_rng(seed)

def enforce_within_screen_variation(
    df: pd.DataFrame,
    screen_col: str = "case_id",
    binary_badges: list[str] = None,
    frame_pair: tuple[str, str] = ("frame_allin", "frame_partitioned"),
    job_id: str = "job"
) -> pd.DataFrame:
    """
    Ensure each screen has both levels (0/1) for every binary badge listed.
    (The storefront already handles the 'frame' 4/4 blocking; this function
    won’t touch 'frame' unless you add frame_* columns in the future.)
    """
    if binary_badges is None:
        binary_badges = ["assurance", "scarcity", "strike", "timer", "social_proof", "voucher", "bundle"]

    rng = _rng_from_job(job_id)
    df = df.copy()

    # coerce selected cols to {0,1}
    for col in binary_badges:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # operate per screen
    for sid, g in df.groupby(screen_col, sort=False):
        idx = g.index.to_list()
        n = len(idx)
        if n < 2:
            continue  # cannot vary with <2 alts

        # 1) binary badges: force ~50/50 within each screen if column exists
        for b in binary_badges:
            if b not in g.columns:
                continue
            vals = g[b].to_numpy().astype(int).copy()
            n1 = int(vals.sum())
            n  = len(vals)

            # if no variation OR highly imbalanced (<=1 or >= n-1), rebalance to floor(n/2) ones
            if (n1 == 0) or (n1 == n) or (n1 <= 1) or (n1 >= n - 1):
                k = n // 2  # for 8 alts -> 4 ones, 4 zeros
                choose = rng.choice(n, size=k, replace=False)
                vals[:] = 0
                vals[choose] = 1
                df.loc[idx, b] = vals


    return df


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
    opts.add_argument("--allow-file-access-from-files")
    opts.add_argument("--disable-web-security")
    try:
        opts.page_load_strategy = "none"
    except Exception:
        pass

    opts.binary_location = os.getenv("CHROME_BIN", "/usr/bin/chromium")

    try:
        service = Service()
        driver = webdriver.Chrome(service=service, options=opts)
    except Exception:
        driver_path = ChromeDriverManager().install()
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=opts)

    try:
        driver.set_page_load_timeout(45)
        driver.set_script_timeout(45)
        driver.implicitly_wait(2)
        if hasattr(driver, "command_executor") and hasattr(driver.command_executor, "_client_config"):
            driver.command_executor._client_config.timeout = 5000
        if hasattr(driver, "command_executor") and hasattr(driver.command_executor, "set_timeout"):
            try:
                driver.command_executor.set_timeout(5000)
            except Exception:
                pass
    except Exception:
        pass

    return driver


def _jpeg_b64_from_driver(driver, quality=72) -> str:
    png = driver.get_screenshot_as_png()
    with Image.open(io.BytesIO(png)) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ---------------- ground-truth reconciliation ----------------
def _products_from_gt(gt) -> List[dict]:
    """
    Accept either {"products":[...]} or plain list [...] from the storefront.
    """
    if isinstance(gt, dict) and "products" in gt:
        return list(gt.get("products") or [])
    if isinstance(gt, list):
        return gt
    return []

def _find_prod(gt, title, row, col):
    products = _products_from_gt(gt)
    t = (title or "").strip().casefold()
    for p in products:
        if str(p.get("title", "")).strip().casefold() == t:
            return p
    for p in products:
        if int(p.get("row", -9)) == int(row) and int(p.get("col", -9)) == int(col):
            return p
    return None

def reconcile(decision: dict, groundtruth: dict) -> dict:
    """
    Align the model's decision to the ground truth schema; propagate lever flags and price fields.
    Works with either:
      • legacy 'dark' string + 'social' bool  OR
      • v1.7 separate scarcity/strike/timer ints + 'social_proof' int.
    """
    r = int(decision.get("row", 0)); c = int(decision.get("col", 0))
    r = 0 if r < 0 else (1 if r > 1 else r)
    c = 0 if c < 0 else (3 if c > 3 else c)
    prod = _find_prod(groundtruth, decision.get("chosen_title"), r, c)

    if prod:
        decision["row"], decision["col"] = int(prod.get("row", r)), int(prod.get("col", c))
        decision["frame"] = int(prod.get("frame", 1)) if prod.get("frame") is not None else None
        decision["assurance"] = int(prod.get("assurance", 0)) if prod.get("assurance") is not None else None

        dark_str = (str(prod.get("dark", "")).strip().lower()) if "dark" in prod else None
        if dark_str is not None:
            decision["scarcity"] = 1 if dark_str == "scarcity" else 0
            decision["strike"]   = 1 if dark_str == "strike"   else 0
            decision["timer"]    = 1 if dark_str == "timer"    else 0
        else:
            decision["scarcity"] = int(prod.get("scarcity", 0))
            decision["strike"]   = int(prod.get("strike", 0))
            decision["timer"]    = int(prod.get("timer", 0))

        decision["social_proof"] = int(prod.get("social_proof", 1 if prod.get("social") else 0)) if ("social_proof" in prod or "social" in prod) else 0
        decision["voucher"]      = int(prod.get("voucher", 0)) if ("voucher" in prod) else 0
        decision["bundle"]       = int(prod.get("bundle", 0)) if ("bundle" in prod) else 0

        price_val = prod.get("price", prod.get("total_price", None))
        if price_val is not None:
            decision["price"] = float(price_val)
            decision["ln_price"] = float(prod.get("ln_price", math.log(max(float(price_val), 1e-8))))
        else:
            decision["price"] = None
            decision["ln_price"] = None
    else:
        decision.update({"frame":None,"assurance":None,"scarcity":None,"strike":None,"timer":None,
                         "social_proof":None,"voucher":None,"bundle":None,"price":None,"ln_price":None})
    return decision

# ---------------- vendor calls ----------------
def _openai_choose(image_b64, category, model):
    key = os.getenv("OPENAI_API_KEY")
    if not key: 
        raise RuntimeError("OPENAI_API_KEY missing.")
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
    if not tcs: 
        raise RuntimeError("OpenAI: no tool_calls.")
    return json.loads(tcs[0]["function"]["arguments"])

def _anthropic_choose(image_b64, category, model):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key: 
        raise RuntimeError("ANTHROPIC_API_KEY missing.")
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
    if not tool_blocks: 
        raise RuntimeError("Anthropic: no tool_use choose.")
    return tool_blocks[0].get("input", {}) or {}

def _gemini_choose(image_b64, category, model):
    key = os.getenv("GEMINI_API_KEY")
    if not key: 
        raise RuntimeError("GEMINI_API_KEY missing.")
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
    if not candidates: 
        raise RuntimeError("Gemini: no candidates.")
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
def _render_html(category: str, set_id: str, badges: List[str], catalog_seed: int, price_anchor: float, currency: str, brand: str = "") -> str:
    """
    Inline storefront renderer (v1.7): delegates to storefront.render_screen(...),
    which implements the two-stage design (frame 4/4 blocked when randomised, plus one
    uniformly drawn non-frame badge per card) and Latin-square price placement.
    """
    from storefront import render_screen
    html = render_screen(
        category=category,
        set_id=set_id,
        badges=badges,
        catalog_seed=catalog_seed,
        price_anchor=float(price_anchor),
        currency=currency,
        brand=brand,
    )
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

    driver = _new_driver()
    try:
        set_id, _, gt, image_b64, _ = _episode(
            driver=driver,
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
    finally:
        try: driver.quit()
        except Exception: pass

# ---------------- run one episode ----------------
def _episode(
    driver,
    category: str,
    ui_label: str,
    render_url_tpl: str,
    set_index: int,
    badges: List[str],
    catalog_seed: int,
    price: float,
    currency: str,
    brand: str,
):
    set_id = f"S{set_index:04d}"

    for attempt in range(2):
        try:
            used_inline = False
            if render_url_tpl.strip():
                try:
                    url = _build_url(render_url_tpl, category, set_id, badges, catalog_seed, price, currency)
                    driver.get(url)
                    WebDriverWait(driver, 7, poll_frequency=0.5).until(
                        EC.presence_of_element_located((By.ID, "groundtruth"))
                    )
                except (TimeoutException, WebDriverException, ReadTimeoutError):
                    used_inline = True
            else:
                used_inline = True

            if used_inline:
                html = _render_html(category, set_id, badges, catalog_seed, price, currency, brand=brand)
                _load_html(driver, html)

            locator = (By.ID, "groundtruth")
            try:
                WebDriverWait(driver, 45, poll_frequency=0.5).until(
                    EC.presence_of_element_located(locator)
                )
            except (TimeoutException, ReadTimeoutError):
                driver.refresh()
                WebDriverWait(driver, 25, poll_frequency=0.5).until(
                    EC.presence_of_element_located(locator)
                )

            gt_text = driver.find_element(By.ID, "groundtruth").get_attribute("textContent")
            gt = json.loads(gt_text)

            image_b64 = _jpeg_b64_from_driver(driver, quality=72)
            model_label, decision = _choose_with_model(image_b64, category, ui_label)
            decision = reconcile(decision, gt)
            return set_id, model_label, gt, image_b64, decision

        except (TimeoutException, ReadTimeoutError):
            if attempt == 0:
                continue
            raise

# ---------------- writers ----------------
def _ensure_dir(p: pathlib.Path): 
    p.mkdir(parents=True, exist_ok=True)

def _write_outputs(category: str, vendor: str, set_id: str, gt: dict, decision: dict, payload: dict):
    rows_choice, rows_long = [], []
    products = _products_from_gt(gt)

    # df_choice (8 rows per screen)
    for p in products:
        dark = (p.get("dark") or "none").strip().lower() if "dark" in p else None
        row_top = 1 if int(p.get("row",0)) == 0 else 0
        col1 = 1 if int(p.get("col",0)) == 0 else 0
        col2 = 1 if int(p.get("col",0)) == 1 else 0
        col3 = 1 if int(p.get("col",0)) == 2 else 0
        chosen = 1 if (int(p.get("row",9)) == int(decision.get("row", -1))
                       and int(p.get("col",9)) == int(decision.get("col", -1))) else 0

        scarcity = 1 if dark == "scarcity" else int(p.get("scarcity", 0)) if dark is None else 0
        strike   = 1 if dark == "strike"   else int(p.get("strike", 0))   if dark is None else 0
        timer    = 1 if dark == "timer"    else int(p.get("timer", 0))    if dark is None else 0

        price_val = p.get("price", p.get("total_price", None))
        ln_price  = p.get("ln_price", math.log(max(float(price_val), 1e-8))) if price_val is not None else None

        rec = {
            "case_id": f"{RUN_ID}|{set_id}|{vendor}",
            "run_id": RUN_ID, "set_id": set_id, "model": vendor, "category": category,
            "title": p.get("title"),
            "row": int(p.get("row", 0)), "col": int(p.get("col", 0)),
            "row_top": row_top, "col1": col1, "col2": col2, "col3": col3,
            "frame": int(p.get("frame", 1)),
            "assurance": int(p.get("assurance", 0)),
            "scarcity": int(scarcity),
            "strike":   int(strike),
            "timer":    int(timer),
            "social_proof": int(p.get("social_proof", 1 if p.get("social") else 0)) if ("social_proof" in p or "social" in p) else 0,
            "voucher": int(p.get("voucher", 0)),
            "bundle":  int(p.get("bundle", 0)),
            "chosen": chosen
        }
        if price_val is not None:
            rec["price"] = float(price_val)
            rec["ln_price"] = float(ln_price)
        rows_choice.append(rec)

    df_choice = pd.DataFrame(rows_choice)
    for c in ("row","col","row_top","col1","col2","col3","frame","assurance",
              "scarcity","strike","timer","social_proof","voucher","bundle","chosen"):
        if c in df_choice.columns: 
            df_choice[c] = df_choice[c].astype(int)
    # Map UI labels -> internal column names
    _ui_to_key = {
        "All-in v. partitioned pricing": "frame",   # frame handled separately by storefront
        "Assurance": "assurance",
        "Scarcity tag": "scarcity",
        "Strike-through": "strike",
        "Timer": "timer",
        "social": "social_proof",
        "voucher": "voucher",
        "bundle": "bundle",
    }
    
    _selected = payload.get("badges") or []
    _selected_keys = [_ui_to_key[b] for b in _selected if b in _ui_to_key and _ui_to_key[b] != "frame"]
    
    df_choice = enforce_within_screen_variation(
        df_choice,
        screen_col="case_id",
        binary_badges=_selected_keys,   # << only what the user picked
        job_id=RUN_ID
    )

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
        "price": float(decision.get("price")) if decision.get("price") is not None else None,
        "ln_price": float(decision.get("ln_price")) if decision.get("ln_price") is not None else None,
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
    ui_label = str(payload.get("model") or "OpenAI GPT-4.1-mini")
    vendor, _, env_key = MODEL_MAP.get(ui_label, ("openai", ui_label, "OPENAI_API_KEY"))
    if not os.getenv(env_key, ""):
        raise RuntimeError(f"{env_key} not set for model '{ui_label}'. Set the API key in the Space settings.")

    _fresh_reset(bool(payload.get("fresh", True)))

    n = int(payload.get("n_iterations", 100) or 100)
    category = str(payload.get("product") or "product")
    brand    = str(payload.get("brand") or "")
    badges   = [str(b) for b in (payload.get("badges") or [])]  # pass-through; storefront handles frame logic
    render_tpl = str(payload.get("render_url") or "")
    catalog_seed = int(payload.get("catalog_seed", 777))

    try:
        price = float(payload.get("price"))
    except Exception:
        price = 0.0
    currency = str(payload.get("currency") or "£")

    # Optional: quick visibility to ensure the pricing toggle is present
    print(f"[runner] badges from UI: {badges}", flush=True)

    driver = _new_driver()
    try:
        for i in range(1, n + 1):
            set_id, _model_label, gt, image_b64, decision = _episode(
                driver=driver,
                category=category,
                ui_label=ui_label,
                render_url_tpl=render_tpl,
                set_index=i,
                badges=badges,                 # <- unchanged; storefront will map the toggle to both frames
                catalog_seed=catalog_seed,
                price=price,
                currency=currency,
                brand=brand,
            )
            _write_outputs(category, ui_label, set_id, gt, decision, payload)
            time.sleep(0.03)
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    # ----- conditional-logit post-processing (single file: badges_effects.csv) -----
    from pathlib import Path
    from datetime import datetime
    from uuid import uuid4
    import pandas as pd
    import logit_badges
    
    # derive ids (local, defined)
    ts = datetime.utcnow().isoformat() + "Z"
    job_id = payload.get("job_id") or f"run-{uuid4().hex[:8]}"
    
    # paths
    effects_path = RESULTS_DIR / "badges_effects.csv"   # the ONE file we will write/use
    effects_path.parent.mkdir(parents=True, exist_ok=True)
    
    badge_rows = []
    badge_table = pd.DataFrame()
    artifacts = {}
    
    # sanity/debug
    choice_path = RESULTS_DIR / "df_choice.csv"
    print("DEBUG choice_path_exists=", choice_path.exists())
    if choice_path.exists():
        _df_dbg = pd.read_csv(choice_path)
        print("DEBUG rows=", len(_df_dbg))
        print("DEBUG cases=", _df_dbg["case_id"].nunique() if "case_id" in _df_dbg.columns else "NA")
        for _c in ["frame","assurance","scarcity","strike","timer","social_proof","voucher","bundle"]:
            if _c in _df_dbg.columns:
                try:
                    print(f"DEBUG {_c}_unique=", int(_df_dbg[_c].nunique(dropna=False)))
                except Exception:
                    print(f"DEBUG {_c}_unique= NA")
    
    print("DEBUG logit_module_path=", getattr(logit_badges, "__file__", "NA"))
    
    def _has_nonempty_file(p: Path) -> bool:
        try:
            return p.exists() and p.stat().st_size > 0
        except Exception:
            return False
    
    if _has_nonempty_file(choice_path):
        try:
            # run the model (accepts the UI badge list as 2nd positional arg)
            badge_table = logit_badges.run_logit(choice_path, badges)
            if not isinstance(badge_table, pd.DataFrame):
                badge_table = pd.DataFrame(badge_table)
    
            print("DEBUG badge_table_shape=", tuple(badge_table.shape))
            print("DEBUG badge_table_cols=", list(badge_table.columns))
    
            if "badge" in badge_table.columns and not badge_table.empty:
                # preferred rich columns; keep only those that exist
                pref_cols = [
                    "badge","beta","p","sign","dir",
                    "se","q_bh","odds_ratio","ci_low","ci_high","ame_pp","evid_score"
                ]
                cols = [c for c in pref_cols if c in badge_table.columns]
                df_rich = badge_table[cols].copy()
    
                # prepend metadata columns (stable order)
                job_meta = {
                    "job_id": job_id,
                    "timestamp": ts,
                    "product": category,
                    "brand": brand,
                    "model": ui_label,      # or another model string you track
                    "price": price,
                    "currency": currency,
                    "n_iteration": n,       # keeping your existing naming
                }
                for k in list(job_meta.keys())[::-1]:
                    df_rich.insert(0, k, job_meta[k])
    
                # write the ONE file (UTF-8 BOM so Excel shows £)
                df_rich.to_csv(effects_path, index=False, encoding="utf-8-sig")
    
                # rows for JSON response
                badge_rows = badge_table.to_dict("records")
    
                # all artifact keys point to the same single file for compatibility
                artifacts["badges_effects"] = str(effects_path)
                artifacts["effects_csv"] = str(effects_path)
                artifacts["table_badges"] = str(effects_path)
            else:
                print("DEBUG empty_or_missing_badge_table")
        except Exception as e:
            print("[logit] skipped due to error:", repr(e), flush=True)
    else:
        print("DEBUG choice file missing or empty")
    # ----- end post-processing -----
    
    vendor_used = MODEL_MAP.get(ui_label, ("openai", ui_label, "OPENAI_API_KEY"))[0]
    return {
        "job_id": payload.get("job_id", job_id),
        "ts": ts,
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
            # single place to read from:
            "badges_effects": artifacts.get("badges_effects", ""),
            # compatibility keys mapped to the same file:
            "effects_csv": artifacts.get("effects_csv", ""),
            "table_badges": artifacts.get("table_badges", "")
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










