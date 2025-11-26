# -*- coding: utf-8 -*-
"""
Agentix — Vision Runner (jpeg_b64) for two-stage design
Version: v1.11 (2025-11-02) — patched

Key assumptions and alignment
    1) One choice per screen; each screen has exactly 8 alternatives (2×4).
    2) Frame is assigned independently of non-frame badges. If the UI selects the
       frame comparison, we block 4 ALL-IN and 4 PARTITIONED per screen; otherwise
       frame is fixed. This matches storefront.render_screen(...).
    3) Exactly one non-frame visual badge per card, drawn from the enabled set
       with a balanced per-screen allocation that includes a true “none” cell.
    4) Prices follow eight log-symmetric levels around the anchor and are placed
       with an 8×8 Latin square to remove price–position confounds.
    5) Ground truth is a plain list of 8 dicts under the hidden #groundtruth div
       in the HTML (no wrapper object); keys match the logit module:
       case_id/set_id, title, row/col, row_top, col1–col3, frame, assurance,
       scarcity, strike, timer, social_proof, voucher, bundle, price, ln_price.
    6) Post-processing uses logit_badges.run_logit(...) which absorbs screen FEs
       and includes product FEs by default (as set in that module).

Patch summary
    • Fixes “chosen stuck at (0,0)” by preventing silent defaulting to (0,0).
      - More generous tokens + defensive parsing in ALL vendor calls (azure, Anthropic, Gemini).
      - reconcile(...) no longer coerces missing/invalid row/col to 0; if unresolved, sets row=col=-1.
    • Removes runner-side heatmap generation; heatmaps are owned by logit_badges only.
NEW
    • Submit-and-poll async wrapper added at bottom: submit_job_async(), poll_job(), fetch_job().
      This avoids long-held HTTP streams; run_job_sync(...) is unchanged in interface.
    • Honour payload["job_id"] as RUN_ID for coherent file tagging across async jobs.
    • Add a process-wide semaphore so only one long run owns Selenium at a time on small instances.
    • Persist lightweight progress to jobs/{job_id}.json for resilience across restarts.
"""

from __future__ import annotations

import os
import io
import json
import time
import base64
import pathlib
import math
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Iterable

import requests
import pandas as pd
from PIL import Image

# selenium / webdriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from urllib3.exceptions import ReadTimeoutError

import logit_badges  # exposes run_logit

# ---------------- paths / version ----------------
RESULTS_DIR = pathlib.Path("results"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR    = pathlib.Path("runs");    RUNS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR    = pathlib.Path("jobs");    JOBS_DIR.mkdir(parents=True, exist_ok=True)

VERSION = "Agentix MC runner – inline-render or URL – 2025-10-31 (v1.10, patched)"
print(f"[agent_runner] {VERSION}", flush=True)

# ---------- strict output schema (prevents header drift) ----------
CHOICE_COLS = [
    "case_id","run_id","set_id","model","category","title",
    "row","col","row_top","col1","col2","col3",
    "frame","assurance","scarcity","strike","timer","social_proof","voucher","bundle",
    "chosen","case_valid","price","ln_price"
]

# ---------------- concurrency: single-run semaphore ----------------
_SIM_SEMAPHORE = threading.Semaphore(1)

# ---------------- small helpers ----------------
def _load_html(driver, html: str):
    from urllib.parse import quote
    data_url = "data:text/html;charset=utf-8," + quote(html)
    driver.get(data_url)

def _fresh_reset(fresh: bool):
    if not fresh:
        return
    for fn in ("df_choice.csv", "df_long.csv", "log_compare.jsonl", "table_badges.csv", "badges_effects.csv"):
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

def _write_progress(job_id: str, status: str, done: int | None = None, total: int | None = None,
                    last_set: str | None = None, artifacts: Dict[str, str] | None = None, error: str | None = None) -> None:
    try:
        rec = {
            "job_id": job_id,
            "status": status,
            "done": done,
            "total": total,
            "last_set": last_set,
            "ts": datetime.utcnow().isoformat() + "Z",
            "artifacts": artifacts or {},
            "error": error
        }
        (JOBS_DIR / f"{job_id}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# ---------------- models / schema ----------------
MODEL_MAP = {
    # Both of these use Azure OpenAI and AZURE_OPENAI_API_KEY
    "GPT-4.1-mini":        ("azure",   "gpt-4.1-mini",             "AZURE_OPENAI_API_KEY"),
    "GPT-5-chat":                  ("azure",   "gpt-5-chat",               "AZURE_OPENAI_API_KEY"),

    # Still supported if you want them
    "Claude 3.5 Haiku": ("anthropic", "claude-3-5-haiku-latest", "ANTHROPIC_API_KEY"),
    "Gemini 2.5 Flash Lite":    ("gemini",    "gemini-2.5-flash-lite",        "GEMINI_API_KEY"),
    "Gemini 3 Pro": ("gemini",    "gemini-3-pro-preview",        "GEMINI_API_KEY"),
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
    "required": ["chosen_title", "row", "col"],
    "additionalProperties": False
}

# ---- Badge filter (non-frame only, for post-estimation table) ----
def _normalize_badge_filter(badges: Iterable[str]) -> list[str]:
    mapping = {
        "all-in v. partitioned pricing": "frame",
        "all-in pricing": "frame",
        "partitioned pricing": "frame",
        "assurance": "assurance",
        "scarcity tag": "scarcity",
        "scarcity": "scarcity",
        "strike-through": "strike",
        "strike": "strike",
        "timer": "timer",
        "social proof": "social_proof",
        "social": "social_proof",
        "voucher": "voucher",
        "bundle": "bundle",
    }
    allowed = {"frame", "assurance", "scarcity", "strike", "timer", "social_proof", "voucher", "bundle"}
    out: list[str] = []
    for b in (badges or []):
        k = mapping.get(str(b).strip().lower())
        if k in allowed and k not in out:
            out.append(k)
    return out

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

    bin_hint = os.getenv("CHROME_BIN")
    if bin_hint:
        opts.binary_location = bin_hint

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
        data = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")

# ---------------- ground-truth reconciliation ----------------
def _products_from_gt(gt) -> List[dict]:
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
    Defensive reconciliation:
      • Never silently coerce missing/invalid row/col to 0.
      • Prefer title match; otherwise use coordinates only if both present and in-bounds.
      • If unresolved, set row=col=-1 so no card is falsely marked as chosen.
    """
    r_raw = decision.get("row", None)
    c_raw = decision.get("col", None)

    r = None
    c = None
    try:
        if r_raw is not None:
            r = int(r_raw)
            if r < 0 or r > 1:
                r = None
    except Exception:
        r = None
    try:
        if c_raw is not None:
            c = int(c_raw)
            if c < 0 or c > 3:
                c = None
    except Exception:
        c = None

    title = (decision.get("chosen_title") or "").strip()
    prod = None
    if title:
        prod = _find_prod(groundtruth, title, r if r is not None else -9, c if c is not None else -9)
    if prod is None and (r is not None and c is not None):
        prod = _find_prod(groundtruth, "", r, c)

    if prod:
        decision["row"] = int(prod.get("row", r if r is not None else -1))
        decision["col"] = int(prod.get("col", c if c is not None else -1))
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
        decision["row"] = -1
        decision["col"] = -1
        decision.update({"frame": None, "assurance": None, "scarcity": None, "strike": None, "timer": None,
                         "social_proof": None, "voucher": None, "bundle": None, "price": None, "ln_price": None})
    return decision

# ---------- Azure OpenAI: robust POST with retries ----------
def _post_with_retries_azure(
    url,
    headers,
    payload,
    timeout=(90, 1800),
    max_attempts=20,
    backoff_base=0.75,
    backoff_cap=12.0,
):
    import random
    sess = requests.Session()
    for attempt in range(1, max_attempts + 1):
        try:
            r = sess.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (409, 425, 429, 500, 502, 503, 504, 529) or (500 <= r.status_code < 600):
                sleep_s = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
                sleep_s *= random.uniform(0.6, 1.4)
                time.sleep(sleep_s)
                continue
            return r
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError):
            sleep_s = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
            sleep_s *= random.uniform(0.6, 1.4)
            time.sleep(sleep_s)
            if attempt == max_attempts:
                raise
    raise RuntimeError("Azure retry exhausted without success.")


# ---------- Azure OpenAI ----------
def call_azure(image_b64, category, deployment_name=None):
    """
    Call an Azure OpenAI chat deployment with image + tool call.
    Expects these environment variables:
      AZURE_OPENAI_ENDPOINT    e.g. "https://<your-resource>.openai.azure.com"
      AZURE_OPENAI_API_KEY     the key from this Azure resource
      AZURE_OPENAI_API_VERSION (optional, e.g. '2024-02-15-preview')
      AZURE_OPENAI_DEPLOYMENT  (optional default deployment name)
    """
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set.")

    #endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    endpoint = "https://info-mia2xmp7-eastus2.cognitiveservices.azure.com"
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set.")

    api_version = "2025-01-01-preview"
    deployment = deployment_name  

    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    #url = "https://info-mia2xmp7-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview"
    headers = {"api-key": key, "content-type": "application/json"}

    tools = [{
        "type": "function",
        "function": {
            "name": "choose",
            "description": "Select one product from the 2×4 grid.",
            "parameters": SCHEMA_JSON,
        },
    }]

    data = {
        "model": deployment,
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Category: {category}. Use ONLY the 'choose' tool.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_b64,
                        },
                    },
                ],
            },
        ],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "choose"}},
        "max_tokens": 64,
        "temperature": 0,
    }

    print(f"[azure] url={url}", flush=True)
    print(f"[azure] deployment={deployment} api_version={api_version}", flush=True)

    try:
        # note: use the same keyword names as the helper definition
        r = _post_with_retries_azure(
            url,
            headers,
            data,
            timeout=(12, 240),
            max_attempts=6,
            backoff_base=0.9,
            backoff_cap=12.0,
        )
    except Exception as e:
        print(f"[azure] HTTP exception before status: {type(e).__name__}: {e}", flush=True)
        raise RuntimeError(f"Azure HTTP failure: {type(e).__name__}: {e}")

    print(f"[azure] status={r.status_code}", flush=True)

    if r.status_code >= 400:
        err_code = None
        err_msg = None
        try:
            err_json = r.json()
            err_obj = err_json.get("error") or err_json
            err_code = err_obj.get("code")
            err_msg = err_obj.get("message")
        except Exception:
            err_json = None

        if err_json is not None and err_msg:
            raise RuntimeError(
                f"Azure API error {r.status_code} [{deployment}] {err_code}: {err_msg}"
            )
        body = r.text[:800] if hasattr(r, "text") else "<no body>"
        raise RuntimeError(
            f"Azure API error {r.status_code} [{deployment}]: {body}"
        )

    try:
        resp = r.json()
    except Exception as e:
        raise RuntimeError(f"Azure returned non-JSON response: {type(e).__name__}: {e}")

    try:
        msg = resp["choices"][0]["message"]
    except Exception as e:
        raise RuntimeError(
            f"Azure JSON shape unexpected, missing choices[0].message: {type(e).__name__}: {e}. "
            f"Raw JSON: {json.dumps(resp, ensure_ascii=False)[:800]}"
        )

    tcs = msg.get("tool_calls", [])
    if not tcs:
        raise RuntimeError(
            f"Azure returned no tool_calls. Raw message: {json.dumps(msg, ensure_ascii=False)[:800]}"
        )

    raw = tcs[0]["function"].get("arguments")

    if isinstance(raw, str):
        try:
            args = json.loads(raw)
        except Exception:
            args = {}
    elif isinstance(raw, dict):
        args = dict(raw)
    else:
        args = {}

    title = args.get("chosen_title")
    args["chosen_title"] = title if isinstance(title, str) else ""
    try:
        args["row"] = int(args.get("row"))
    except Exception:
        args["row"] = None
    try:
        args["col"] = int(args.get("col"))
    except Exception:
        args["col"] = None

    return args


    
# ---------- Anthropic ----------
def _post_with_retries(url, headers, payload, timeout=1200, attempts=5, backoff=0.9):
    import random
    for i in range(attempts):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (409, 425, 429, 500, 502, 503, 504, 529) or (500 <= r.status_code < 600):
                time.sleep((backoff ** i) * (1.0 + random.random()))
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException:
            if i == attempts - 1:
                raise
            time.sleep((backoff ** i) * (1.0 + random.random()))

def call_anthropic(image_b64: str, category: str, model_name: str):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY missing.")
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    tools = [{"name": "choose", "description": "Select one grid item", "input_schema": SCHEMA_JSON}]
    body = {
        "model": model_name, "max_tokens": 128, "temperature": 0,
        "system": SYSTEM_PROMPT, "tools": tools, "tool_choice": {"type": "tool", "name": "choose"},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64.split(",")[1]}},
                {"type": "text", "text": f"Category: {category}. Use ONLY the tool 'choose'."}
            ]
        }]}
    r = _post_with_retries(url, headers, body, timeout=240)
    blocks = r.json().get("content", [])
    tool_blocks = [b for b in blocks if b.get("type") == "tool_use" and b.get("name") == "choose"]
    if not tool_blocks:
        raise RuntimeError("Anthropic: no tool_use choose.")
    args = tool_blocks[0].get("input", {}) or {}
    title = args.get("chosen_title")
    args["chosen_title"] = title if isinstance(title, str) else ""
    try:
        args["row"] = int(args.get("row"))
    except Exception:
        args["row"] = None
    try:
        args["col"] = int(args.get("col"))
    except Exception:
        args["col"] = None
    return args

# ---------- Gemini ----------
def call_gemini(image_b64: str, category: str, model_name: str):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing.")

    # SAFE DEBUG: never print full key
    print(
        f"[GEMINI] model={model_name} key_present={bool(key)} "
        f"key_len={len(key)} key_prefix={key[:6]}***",
        flush=True,
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"

    tools = [{
        "function_declarations": [{
            "name": "choose",
            "description": "Select one grid item",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "chosen_title": {"type": "STRING"},
                    "row": {"type": "INTEGER"},
                    "col": {"type": "INTEGER"},
                },
                "required": ["chosen_title", "row", "col"],
            },
        }]
    }]

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "tools": tools,
        "tool_config": {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["choose"],  # nudge Gemini to call the tool
            }
        },
        "contents": [{
            "role": "user",
            "parts": [
                {"text": f"Category: {category}. Use ONLY the tool 'choose'."},
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_b64.split(",")[1],
                }},
            ],
        }],
    }

    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=240,
    )
    #print(f"[GEMINI] HTTP status={r.status_code}", flush=True)
    if not r.ok:
        print(f"[GEMINI] status={r.status_code} body={r.text[:600]}", flush=True)
        r.raise_for_status()

    resp = r.json()
    # Short snippet so you can see if a tool call is present
    try:
        print(
            "[GEMINI] raw response snippet: "
            + json.dumps(resp, ensure_ascii=False)[:600],
            flush=True,
        )
    except Exception:
        pass

    candidates = resp.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini: no candidates.")

    parts = candidates[0].get("content", {}).get("parts", []) or []
    args = {}

    # Scan all parts for a functionCall named "choose"
    for p in parts:
        fc = p.get("functionCall")
        if fc and fc.get("name") == "choose":
            raw = fc.get("args") or {}
            # Some models return args as list of {key,value}
            if isinstance(raw, dict):
                args = dict(raw)
            elif isinstance(raw, list):
                args = {item.get("key"): item.get("value") for item in raw if isinstance(item, dict)}
            break

    if not args:
        raise RuntimeError("Gemini: no functionCall choose.")

    print(f"[GEMINI] parsed args before normalisation: {args}", flush=True)

    title = args.get("chosen_title")
    args["chosen_title"] = title if isinstance(title, str) else ""

    # Normalise to 0-based indices, in case Gemini replies 1..2 / 1..4
    try:
        r_val = int(args.get("row"))
        if r_val in (1, 2):
            r_val -= 1
        args["row"] = r_val
    except Exception:
        args["row"] = None

    try:
        c_val = int(args.get("col"))
        if c_val in (1, 2, 3, 4):
            c_val -= 1
        args["col"] = c_val
    except Exception:
        args["col"] = None

    print(f"[GEMINI] normalised args: {args}", flush=True)
    return args


def _choose_with_model(image_b64, category, ui_label):
    vendor, model_name, _ = MODEL_MAP.get(ui_label, ("azure", ui_label, "AZURE_OPENAI_API_KEY"))
    print(f"[choose_with_model] ui_label={ui_label} vendor={vendor} model_name={model_name}", flush=True)

    if vendor == "azure":
        decision = call_azure(image_b64, category, deployment_name=model_name)
    
    elif vendor == "anthropic":
        decision = call_anthropic(image_b64, category, model_name)
    elif vendor == "gemini":
        decision = call_gemini(image_b64, category, model_name)
    else:
        decision = call_azure(image_b64, category, deployment_name=model_name)

    # Log the UI label (e.g. 'gpt-5-chat') as the model in df_choice
    return ui_label, decision




# ---------------- URL builder (Option A) ----------------
def _build_url(tpl: str, category: str, set_id: str, badges: List[str], catalog_seed: int, price: float, currency: str) -> str:
    seed = int(time.time() * 1000) & 0x7FFFFFFF
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
    ui_label = str(payload.get("model") or "GPT-4.1-mini")
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
        set_id, _, gt, image_b64, decision = _episode(
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
        try:
            driver.quit()
        except Exception:
            pass

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

            try:
                model_label, decision = _choose_with_model(image_b64, category, ui_label)
            except Exception as e:
                print(
                    f"[episode] model call failed on attempt {attempt+1}: {type(e).__name__}: {e}",
                    flush=True
                )
                if attempt == 0:
                    continue
                model_label, decision = ("azure", {"row": None, "col": None, "chosen_title": ""})

            decision = reconcile(decision, gt)
            return set_id, model_label, gt, image_b64, decision

        except (TimeoutException, ReadTimeoutError) as e:
            print(f"[episode] webdriver timeout on attempt {attempt+1}: {type(e).__name__}: {e}", flush=True)
            if attempt == 0:
                continue
            raise


# ---------------- writers ----------------
def _append_choice(df_choice: pd.DataFrame, path: pathlib.Path):
    df_choice = df_choice.reindex(columns=CHOICE_COLS)
    write_header = True
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                first = f.readline().strip()
            existing = [c.strip() for c in first.split(",")] if first else []
            if existing == CHOICE_COLS:
                write_header = False
            else:
                backup = path.with_suffix(".old.csv")
                try:
                    if backup.exists():
                        backup.unlink()
                except Exception:
                    pass
                path.rename(backup)
                write_header = True
        except Exception:
            write_header = True
    df_choice.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def _write_outputs(category: str, model_label: str, set_id: str, gt: dict, decision: dict, payload: dict):
    rows_choice, rows_long = [], []
    products = _products_from_gt(gt)
    case_valid = 1 if len(products) == 8 else 0

    for p in products:
        r_p = int(p.get("row", 0)); c_p = int(p.get("col", 0))
        r_d = int(decision.get("row", -9)); c_d = int(decision.get("col", -9))
        chosen = 1 if (r_p == r_d and c_p == c_d) else 0

        row_top = 1 if r_p == 0 else 0
        col1 = 1 if c_p == 0 else 0
        col2 = 1 if c_p == 1 else 0
        col3 = 1 if c_p == 2 else 0

        dark = (p.get("dark") or "").strip().lower() if "dark" in p else None
        scarcity = 1 if dark == "scarcity" else int(p.get("scarcity", 0))
        strike   = 1 if dark == "strike"   else int(p.get("strike", 0))
        timer    = 1 if dark == "timer"    else int(p.get("timer", 0))

        price_val = p.get("price", p.get("total_price", None))
        ln_price  = (float(p.get("ln_price")) if p.get("ln_price") is not None
                     else (math.log(max(float(price_val), 1e-8)) if price_val is not None else None))

        rows_choice.append({
            "case_id": f"{RUN_ID}|{set_id}|{model_label}",
            "run_id": RUN_ID, "set_id": set_id, "model": model_label, "category": category,
            "title": p.get("title"),
            "row": r_p, "col": c_p,
            "row_top": row_top, "col1": col1, "col2": col2, "col3": col3,
            "frame": int(p.get("frame", 1)),
            "assurance": int(p.get("assurance", 0)),
            "scarcity": int(scarcity), "strike": int(strike), "timer": int(timer),
            "social_proof": int(p.get("social_proof", 1 if p.get("social") else 0)) if ("social_proof" in p or "social" in p) else 0,
            "voucher": int(p.get("voucher", 0)), "bundle": int(p.get("bundle", 0)),
            "chosen": chosen, "case_valid": case_valid,
            "price": float(price_val) if price_val is not None else None,
            "ln_price": float(ln_price) if ln_price is not None else None,
        })

    df_choice = pd.DataFrame(rows_choice)
    for c in ["row","col","row_top","col1","col2","col3","frame","assurance",
              "scarcity","strike","timer","social_proof","voucher","bundle",
              "chosen","case_valid"]:
        if c in df_choice.columns:
            df_choice[c] = pd.to_numeric(df_choice[c], errors="coerce").fillna(0).astype(int)

    df_choice = df_choice.reindex(columns=CHOICE_COLS)
    agg_choice = RESULTS_DIR / "df_choice.csv"
    df_choice.to_csv(agg_choice, mode="a", header=not agg_choice.exists(), index=False, encoding="utf-8-sig")

    rows_long.append({
        "run_id": RUN_ID, "iter": int(set_id[1:]), "category": category, "set_id": set_id, "model": model_label,
        "chosen_title": decision.get("chosen_title"),
        "row": int(decision.get("row", -1)), "col": int(decision.get("col", -1)),
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
    df_long.to_csv(RESULTS_DIR / "df_long.csv",
                   mode="a", header=not (RESULTS_DIR / "df_long.csv").exists(),
                   index=False, encoding="utf-8")

    rec = {"run_id": RUN_ID, "ts": datetime.utcnow().isoformat(),
           "category": category, "set_id": set_id, "model": model_label,
           "groundtruth": gt, "decision": decision}
    with (RESULTS_DIR / "log_compare.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")




# ---------------- public API ----------------
# ---------------- public API ----------------
def run_job_sync(payload: Dict) -> Dict:
    global RUN_ID

    _SIM_SEMAPHORE.acquire()
    try:
        ui_label = str(payload.get("model") or "GPT-5-chat")
        vendor, _, env_key = MODEL_MAP.get(ui_label, ("azure", ui_label, "AZURE_OPENAI_API_KEY"))
        print(f"[runner] ui_label={ui_label} vendor={vendor} env_key={env_key}", flush=True)

        if vendor != "azure":
            if not os.getenv(env_key, ""):
                raise RuntimeError(
                    f"{env_key} not set for model '{ui_label}'. Set the API key in the Space settings."
                )

        run_id_from_payload = str(payload.get("job_id") or payload.get("run_id") or RUN_ID)
        old_run_id = RUN_ID
        RUN_ID = run_id_from_payload

        _fresh_reset(bool(payload.get("fresh", True)))

        n = int(payload.get("n_iterations", 100) or 100)
        category = str(payload.get("product") or "product")
        brand = str(payload.get("brand") or "")
        badges = [str(b) for b in (payload.get("badges") or [])]
        render_tpl = str(payload.get("render_url") or "")
        catalog_seed = int(payload.get("catalog_seed", 777))

        try:
            price = float(payload.get("price"))
        except Exception:
            price = 0.0
        currency = str(payload.get("currency") or "£")

        print(f"[runner] badges from UI: {badges}", flush=True)

        _write_progress(RUN_ID, status="running", done=0, total=n, last_set=None, artifacts={})

        driver = _new_driver()
        last_iter = 0
        last_set_id = None
        cancelled = False
        try:
            for i in range(1, n + 1):
                set_id, model_label, gt, image_b64, decision = _episode(
                    driver=driver,
                    category=category,
                    ui_label=ui_label,
                    render_url_tpl=render_tpl,
                    set_index=i,
                    badges=badges,
                    catalog_seed=catalog_seed,
                    price=price,
                    currency=currency,
                    brand=brand,
                )
                last_iter = i
                last_set_id = set_id
                _write_outputs(category, model_label, set_id, gt, decision, payload)
                _write_progress(RUN_ID, status="running", done=i, total=n, last_set=set_id, artifacts={})

                # honour async cancellation requests (if running under submit_job_async)
                try:
                    with _JLOCK:
                        js_local = _JOBS.get(RUN_ID)
                    if js_local and getattr(js_local, "cancel_requested", False):
                        print(
                            f"[runner] cancel requested for {RUN_ID}, stopping early at iteration {i}/{n}",
                            flush=True,
                        )
                        cancelled = True
                        _write_progress(RUN_ID, "cancelled", i, n, set_id, {}, None)
                        with _JLOCK:
                            js_local.status = "cancelled"
                            js_local.end_ts = time.time()
                        break
                except NameError:
                    # async wrapper not initialised; ignore when run_job_sync is used standalone
                    pass
                except Exception:
                    # cancellation is best-effort; do not crash the run
                    pass

                time.sleep(0.03)
        finally:
            try:
                driver.quit()
            except Exception:
                pass

        ts = datetime.utcnow().isoformat() + "Z"
        job_id = RUN_ID

        effects_path = RESULTS_DIR / "badges_effects.csv"
        effects_path.parent.mkdir(parents=True, exist_ok=True)

        badge_rows: List[dict] = []
        artifacts: Dict[str, str] = {}

        choice_path = RESULTS_DIR / "df_choice.csv"
        print("DEBUG choice_path_exists=", choice_path.exists())
        if choice_path.exists():
            try:
                _df_dbg = pd.read_csv(choice_path)
                print("DEBUG rows=", len(_df_dbg))
                print(
                    "DEBUG cases=",
                    _df_dbg["case_id"].nunique() if "case_id" in _df_dbg.columns else "NA",
                )
                for _c in [
                    "frame",
                    "assurance",
                    "scarcity",
                    "strike",
                    "timer",
                    "social_proof",
                    "voucher",
                    "bundle",
                ]:
                    if _c in _df_dbg.columns:
                        try:
                            print(
                                f"DEBUG {_c}_unique=",
                                int(_df_dbg[_c].nunique(dropna=False)),
                            )
                        except Exception:
                            print(f"DEBUG {_c}_unique= NA")
            except Exception as e:
                print("DEBUG could not read df_choice.csv:", repr(e))

        print("DEBUG logit_module_path=", getattr(logit_badges, "__file__", "NA"))

        if choice_path.exists() and choice_path.stat().st_size > 0:
            try:
                badge_keys = _normalize_badge_filter(badges)
                print("DEBUG badge_filter_internal=", badge_keys)

                badge_table = logit_badges.run_logit(
                    str(choice_path), badge_filter=badge_keys if badge_keys else None
                )
                if not isinstance(badge_table, pd.DataFrame):
                    badge_table = pd.DataFrame(badge_table)

                print("DEBUG badge_table_shape=", tuple(badge_table.shape))
                print("DEBUG badge_table_cols=", list(badge_table.columns))

                if "badge" in badge_table.columns and not badge_table.empty:
                    pref_cols = [
                        "badge",
                        "beta",
                        "p",
                        "sign",
                        "se",
                        "q_bh",
                        "odds_ratio",
                        "ci_low",
                        "ci_high",
                        "ame_pp",
                        "evid_score",
                        "price_eq",
                    ]
                    cols = [c for c in pref_cols if c in badge_table.columns]
                    df_rich = badge_table[cols].copy()

                    job_meta = {
                        "job_id": job_id,
                        "timestamp": ts,
                        "product": category,
                        "brand": brand,
                        "model": ui_label,
                        "price": price,
                        "currency": currency,
                        "n_iteration": n,
                    }
                    for k in list(job_meta.keys())[::-1]:
                        df_rich.insert(0, k, job_meta[k])

                    df_rich.to_csv(effects_path, index=False, encoding="utf-8-sig")
                    badge_rows = badge_table.to_dict("records")

                    artifacts["badges_effects"] = str(effects_path)
                    artifacts["effects_csv"] = str(effects_path)
                    artifacts["table_badges"] = str(effects_path)
                else:
                    print("DEBUG empty_or_missing_badge_table")

                try:
                    hm = logit_badges.generate_heatmaps(
                        str(choice_path),
                        out_dir=str(RESULTS_DIR),
                        title_prefix=f"{category} · {ui_label}",
                        file_tag=None,
                    )
                    artifacts.update(hm)
                except Exception as e:
                    print("DEBUG generate_heatmaps skipped:", repr(e))

            except Exception as e:
                print("[logit] skipped due to error:", repr(e), flush=True)

        artifacts.setdefault("df_choice", str(RESULTS_DIR / "df_choice.csv"))
        artifacts.setdefault("df_long", str(RESULTS_DIR / "df_long.csv"))
        artifacts.setdefault("log_compare", str(RESULTS_DIR / "log_compare.jsonl"))

        results: Dict = {
            "job_id": job_id,
            "ts": ts,
            "model_requested": ui_label,
            "vendor": vendor,
            "n_iterations": n,
            "inputs": {
                "product": category,
                "brand": brand,
                "price": price,
                "currency": currency,
                "badges": badges,
            },
            "artifacts": artifacts,
            "logit_table_rows": badge_rows,
        }

        # record how many iterations actually completed and whether we exited early
        results["n_completed"] = last_iter
        results["cancelled"] = bool(cancelled)

        final_status = "cancelled" if cancelled else "done"
        final_done = last_iter
        final_last_set = (
            last_set_id if last_set_id is not None else (f"S{final_done:04d}" if final_done else None)
        )
        _write_progress(RUN_ID, final_status, final_done, n, final_last_set, artifacts, None)

        RUN_ID = old_run_id
        return results

    except Exception as e:
        try:
            _write_progress(RUN_ID, status="error", error=f"{type(e).__name__}: {e}")
        except Exception:
            pass
        raise
    finally:
        try:
            _SIM_SEMAPHORE.release()
        except Exception:
            pass


if __name__ == "__main__":
    jobs_dir = pathlib.Path("jobs")
    if jobs_dir.exists():
        for p in sorted(jobs_dir.glob("job-*.json")):
            payload = json.loads(p.read_text(encoding="utf-8"))
            try:
                res = run_job_sync(payload)
                (RESULTS_DIR / f"{payload.get('job_id','job')}.json").write_text(
                    json.dumps(res, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                payload["status"] = "completed"
                p.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                _write_progress(
                    payload.get("job_id", "job"),
                    status="done",
                    done=payload.get("n_iterations", 0),
                    total=payload.get("n_iterations", 0),
                    artifacts=res.get("artifacts", {}),
                )
            except Exception as e:
                payload["status"] = f"error: {type(e).__name__}: {e}"
                p.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                _write_progress(
                    payload.get("job_id", "job"),
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                )
        print("Done.")
    else:
        print("No jobs/ folder found. Import and call run_job_sync(payload).")


# ======================= NEW: submit-and-poll async wrapper =======================
from dataclasses import dataclass, field
import uuid

@dataclass
class _JobState:
    job_id: str
    status: str = "queued"
    start_ts: float = field(default_factory=time.time)
    end_ts: float | None = None
    results_json: str | None = None
    error: str | None = None
    cancel_requested: bool = False

_JOBS: Dict[str, _JobState] = {}
_JLOCK = threading.Lock()
_TTL_SEC = 6 * 60 * 60

def _gc_jobs(now: float | None = None) -> None:
    t = now or time.time()
    with _JLOCK:
        stale = [k for k, v in _JOBS.items() if v.end_ts and (t - v.end_ts) > _TTL_SEC]
        for k in stale:
            _JOBS.pop(k, None)

def submit_job_async(payload: Dict) -> Dict:
    jid = str(payload.get("job_id") or f"job-{uuid.uuid4().hex[:8]}")
    payload = dict(payload)
    payload["job_id"] = jid
    js = _JobState(job_id=jid, status="running")
    with _JLOCK:
        _JOBS[jid] = js

    def _worker():
        try:
            res = run_job_sync(payload)

            # persist results to Agentix (best-effort)
            try:
                from save_to_agentix import persist_results_if_qualify
                info = persist_results_if_qualify(
                    res,
                    payload,
                    base_url="https://agentyx.tech",
                    app_version="app-1",
                    est_model="logit-1",
                    alpha=0.05,
                )
                res.setdefault("artifacts", {})["agentix_persist"] = info
            except Exception as e:
                res.setdefault("artifacts", {})["agentix_persist_error"] = f"{type(e).__name__}: {e}"

            with _JLOCK:
                current_status = js.status
                js.results_json = json.dumps(res, ensure_ascii=False)
                js.end_ts = time.time()
                if current_status != "cancelled":
                    js.status = "done"
        except Exception as e:
            print(f"[worker] job {jid} failed: {type(e).__name__}: {e}", flush=True)
            with _JLOCK:
                js.status = "error"
                js.error = f"{type(e).__name__}: {e}"
                js.end_ts = time.time()
        finally:
            _gc_jobs()

    threading.Thread(target=_worker, name=f"agentix-run-{jid}", daemon=True).start()
    return {"ok": True, "job_id": jid, "status": "running"}

def poll_job(job_id: str) -> Dict:
    # get in-memory status first
    with _JLOCK:
        js = _JOBS.get(job_id)
        if not js:
            return {"ok": False, "error": "unknown_job"}
        status = js.status
        err = js.error

    # read on-disk progress written by _write_progress(...)
    progress_path = JOBS_DIR / f"{job_id}.json"
    done = None
    total = None
    if progress_path.exists():
        try:
            prog = json.loads(progress_path.read_text(encoding="utf-8"))
            done = prog.get("done")
            total = prog.get("total")
        except Exception:
            pass

    resp = {
        "ok": True,
        "job_id": job_id,
        "status": status,
        "error": err,
    }
    if done is not None:
        resp["iterations_done"] = done
    if total is not None:
        resp["n_iterations"] = total
    return resp

def fetch_job(job_id: str) -> Dict:
    with _JLOCK:
        js = _JOBS.get(job_id)
        if not js:
            return {"ok": False, "error": "unknown_job"}
        if js.status not in ("done", "cancelled"):
            return {"ok": False, "error": "not_ready", "status": js.status}
        return {
            "ok": True,
            "job_id": job_id,
            "status": js.status,
            "results_json": js.results_json or "{}",
        }

def cancel_job(job_id: str) -> Dict:
    with _JLOCK:
        js = _JOBS.get(job_id)
        if not js:
            return {"ok": False, "error": "unknown_job"}
        if js.status in ("done", "error", "cancelled"):
            return {"ok": True, "job_id": job_id, "status": js.status}
        js.cancel_requested = True
        js.status = "cancelling"
    try:
        _write_progress(job_id, "cancelling")
    except Exception:
        pass
    return {"ok": True, "job_id": job_id, "status": "cancelling"}









