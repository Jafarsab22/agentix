# -*- coding: utf-8 -*-
"""
Live A/B tester (self-contained) — uses OpenAI Vision tool-calls directly.
Creates a 2×4 grid (4×A, 4×B) with randomised positions per trial and asks the
agent to choose exactly one card. Tracks A/B pick counts asynchronously.

Exports:
    submit_live_ab(file_a, file_b, n_trials, category="", model_name="")
    poll_live_ab(job_id)
    fetch_live_ab(job_id)
    cancel_live_ab(job_id)
"""

from __future__ import annotations
import os, io, json, time, threading, random, base64, math
from typing import Dict, Tuple
from PIL import Image

# ---------------- In-memory job store ----------------
_JOBS: Dict[str, dict] = {}
_JLOCK = threading.Lock()

# ---------------- Tool schema & prompt ----------------
SCHEMA_JSON = {
    "type": "object",
    "properties": {
        "chosen_title": {"type": "string"},
        "row": {"type": "integer", "description": "0-based row index (0..1)"},
        "col": {"type": "integer", "description": "0-based col index (0..3)"}
    },
    "required": ["chosen_title", "row", "col"],
    "additionalProperties": False
}

SYSTEM_PROMPT = (
    "You are a personal shopping assistant helping someone choose exactly ONE product from a 2×4 grid. "
    "Each trial shows 8 cards in two rows and four columns. "
    "Select precisely one card and return ONLY a tool/function call named 'choose' with fields: "
    "chosen_title (string), row (0-based int), col (0-based int)."
)

# ---------------- Image helpers ----------------
def _img_to_data_url(img: Image.Image, fmt="JPEG", quality=80) -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"

def _load_and_fit(path: str, box_w: int, box_h: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((box_w, box_h), Image.LANCZOS)
    bg = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    x = (box_w - im.width) // 2
    y = (box_h - im.height) // 2
    bg.paste(im, (x, y))
    return bg

def _compose_grid_2x4(file_a: str, file_b: str, labels: list[str]) -> Tuple[str, list[list[str]]]:
    """
    labels: list of 8 'A'/'B' flags row-major. Returns (data_url, matrix[2][4]).
    """
    cell_w, cell_h = 320, 320
    pad = 8
    W = 4 * cell_w + 5 * pad
    H = 2 * cell_h + 3 * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    imgA = _load_and_fit(file_a, cell_w, cell_h)
    imgB = _load_and_fit(file_b, cell_w, cell_h)

    matrix = []
    k = 0
    for r in range(2):
        row = []
        for c in range(4):
            x = pad + c * (cell_w + pad)
            y = pad + r * (cell_h + pad)
            lab = labels[k]; k += 1
            canvas.paste(imgA if lab == "A" else imgB, (x, y))
            row.append(lab)
        matrix.append(row)

    # Optional thin grid lines to make cells obvious to the model
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        # vertical lines
        for c in range(5):
            x = pad + c * (cell_w + pad) - 4 if c > 0 else pad - 4
            draw.line([(x, pad), (x, H - pad)], fill=(220, 220, 220), width=1)
        # horizontal lines
        for r in range(3):
            y = pad + r * (cell_h + pad) - 4 if r > 0 else pad - 4
            draw.line([(pad, y), (W - pad, y)], fill=(220, 220, 220), width=1)
    except Exception:
        pass

    return _img_to_data_url(canvas, fmt="JPEG", quality=80), matrix

# ---------------- OpenAI vision call (self-contained) ----------------
def _post_with_retries_openai(url, headers, payload,
                              timeout=(12, 240),
                              max_attempts=6,
                              backoff_base=0.75,
                              backoff_cap=8.0):
    import requests, random, time as _t
    s = requests.Session()
    for attempt in range(1, max_attempts + 1):
        try:
            r = s.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (409, 425, 429, 500, 502, 503, 504, 529) or (500 <= r.status_code < 600):
                sleep_s = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
                sleep_s *= random.uniform(0.7, 1.4)
                _t.sleep(sleep_s); continue
            return r
        except Exception:
            if attempt == max_attempts:
                raise
            sleep_s = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
            sleep_s *= random.uniform(0.7, 1.4)
            _t.sleep(sleep_s)
    raise RuntimeError("OpenAI retry exhausted without success.")

def _openai_choose(image_b64: str, category: str = "", model_name: str | None = None) -> dict:
    """
    Calls OpenAI with a forced function call 'choose' and returns parsed args.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model = model_name or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}

    tools = [{
        "type": "function",
        "function": {"name": "choose", "description": "Select one product from the 2×4 grid.", "parameters": SCHEMA_JSON}
    }]

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Category: {category}. Use ONLY the 'choose' tool. No prose."},
                {"type": "image_url", "image_url": {"url": image_b64}}
            ]}
        ],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "choose"}},
        "max_tokens": 64,
        "temperature": 0
    }

    r = _post_with_retries_openai(url, headers, data)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:500]}")

    msg = r.json()["choices"][0]["message"]
    tcs = msg.get("tool_calls", [])
    if not tcs:
        # Rarely models answer in text; try to salvage JSON
        raw_text = (msg.get("content") or "").strip()
        try:
            obj = json.loads(raw_text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    raw = tcs[0]["function"].get("arguments")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}

# ---------------- Robust coordinate extraction ----------------
def _rc_from_args(args: dict) -> Tuple[int | None, int | None]:
    """
    Accept:
      • row/col (0- or 1-indexed)
      • single index via keys: index/slot/position/choice/chosen_index/selected_index
      • nested args under 'arguments' or 'args'
    Returns zero-based (row, col) or (None, None).
    """
    if not isinstance(args, dict):
        return None, None

    # direct row/col
    try:
        if "row" in args and "col" in args:
            r = int(args["row"]); c = int(args["col"])
            r = r - 1 if 1 <= r <= 2 else r
            c = c - 1 if 1 <= c <= 4 else c
            if (0 <= r <= 1) and (0 <= c <= 3):
                return r, c
    except Exception:
        pass

    # single index
    for k in ("index", "slot", "position", "choice", "chosen_index", "selected_index"):
        if k in args:
            try:
                v = int(args[k])
            except Exception:
                continue
            v = v - 1 if 1 <= v <= 8 else v
            if 0 <= v <= 7:
                return v // 4, v % 4

    # nested
    for k in ("arguments", "args"):
        if k in args:
            try:
                nested = json.loads(args[k]) if isinstance(args[k], str) else dict(args[k])
            except Exception:
                nested = {}
            r, c = _rc_from_args(nested)
            if r is not None and c is not None:
                return r, c

    return None, None

# ---------------- One trial ----------------
def _trial_once(file_a: str, file_b: str, category: str | None, model_name: str | None):
    labels = ["A"] * 4 + ["B"] * 4
    random.shuffle(labels)
    data_url, layout = _compose_grid_2x4(file_a, file_b, labels)

    args = _openai_choose(data_url, category or "", model_name=model_name)
    r0, c0 = _rc_from_args(args)
    if r0 is None or c0 is None:
        return None, layout
    return layout[r0][c0], layout   # "A" or "B" (or None)

# ---------------- Background worker ----------------
def _bg_run_live_ab(job_id: str, file_a: str, file_b: str, n_trials: int, category: str | None, model_name: str | None):
    try:
        a = b = invalid = 0
        cancelled = False

        for t in range(1, n_trials + 1):
            with _JLOCK:
                if _JOBS.get(job_id, {}).get("cancel", False):
                    cancelled = True
            if cancelled:
                break

            chosen, _layout = _trial_once(file_a, file_b, category, model_name)
            if chosen == "A":
                a += 1
            elif chosen == "B":
                b += 1
            else:
                invalid += 1

            if (t % 10 == 0) or (t == n_trials):
                with _JLOCK:
                    _JOBS[job_id] = {
                        "status": "running", "progress": t, "total": n_trials,
                        "a": a, "b": b, "invalid": invalid, "cancel": _JOBS.get(job_id, {}).get("cancel", False)
                    }

        completed = max(1, a + b + invalid)
        valid = max(1, a + b)
        ra, rb = a / valid, b / valid
        p_pool = (a + b) / (2 * max(1, n_trials))
        se = math.sqrt(max(p_pool * (1 - p_pool) * (2 / max(1, n_trials)), 0.0))
        z = (ra - rb) / se if se > 0 else 0.0

        def ci(p, n):
            s = math.sqrt(max(p * (1 - p) / max(1, n), 0.0))
            return (p - 1.96 * s, p + 1.96 * s)

        res = {
            "a": a, "b": b, "invalid": invalid, "n_trials": n_trials, "completed": completed,
            "rate_a": ra, "rate_b": rb, "ci_a": ci(ra, valid), "ci_b": ci(rb, valid),
            "z_two_prop": z,
            "note": ("Cancelled early; " if cancelled else "") +
                    "one OpenAI call per trial; 4×A/4×B with randomised positions each trial."
        }
        with _JLOCK:
            _JOBS[job_id] = {"status": "cancelled" if cancelled else "done", "result": res}
    except Exception as e:
        with _JLOCK:
            _JOBS[job_id] = {"status": "error", "error": f"{type(e).__name__}: {e}"}

# ---------------- Public API ----------------
def submit_live_ab(file_a, file_b, n_trials, category: str = "", model_name: str = ""):
    if not file_a or not file_b:
        return "", "❌ Please upload both images."
    try:
        n = int(n_trials)
        if n <= 0:
            return "", "❌ Trials must be positive."
    except Exception:
        return "", "❌ Invalid trials."

    job_id = f"ab-{int(time.time())}-{random.randint(1000,9999)}"
    with _JLOCK:
        _JOBS[job_id] = {"status": "queued", "progress": 0, "total": n, "a": 0, "b": 0, "invalid": 0, "cancel": False}
    th = threading.Thread(
        target=_bg_run_live_ab,
        args=(job_id, file_a, file_b, n, category or "", model_name or None),
        name=f"live-ab-{job_id}",
        daemon=True
    )
    th.start()
    return job_id, f"✅ Submitted live A/B job {job_id} with {n} trials."

def poll_live_ab(job_id: str):
    job_id = (job_id or "").strip()
    with _JLOCK:
        info = _JOBS.get(job_id)
    if not info:
        return "Unknown job."
    if info.get("status") == "running":
        p = info.get("progress", 0); tot = info.get("total", 0)
        a = info.get("a", 0); b = info.get("b", 0); inv = info.get("invalid", 0)
        return f"Job {job_id}: running — {p}/{tot} trials; A={a}, B={b}, invalid={inv}"
    return f"Job {job_id}: {info.get('status','unknown')}"

def fetch_live_ab(job_id: str):
    job_id = (job_id or "").strip()
    with _JLOCK:
        info = _JOBS.get(job_id)
    if not info:
        return "Unknown job.", ""
    if info.get("status") != "done" and info.get("status") != "cancelled":
        return f"Job {job_id}: {info.get('status','not_ready')}", ""
    r = info.get("result") or {}
    md = (
        "### Live A/B results (agent choices)\n\n"
        "| Variant | Picks | Rate | 95% CI |\n"
        "|---|---:|---:|---|\n"
        f"| A | {r.get('a',0)} | {r.get('rate_a',0):.3f} | "
        f"[{r.get('ci_a',(0,0))[0]:.3f}, {r.get('ci_a',(0,0))[1]:.3f}] |\n"
        f"| B | {r.get('b',0)} | {r.get('rate_b',0):.3f} | "
        f"[{r.get('ci_b',(0,0))[0]:.3f}, {r.get('ci_b',(0,0))[1]:.3f}] |\n"
        f"\n*z* = {r.get('z_two_prop',0):.2f} (two-proportion, pooled)\n\n"
        f"*{r.get('note','')}*"
    )
    return md, json.dumps(r, ensure_ascii=False, indent=2)

def cancel_live_ab(job_id: str):
    job_id = (job_id or "").strip()
    with _JLOCK:
        info = _JOBS.get(job_id)
        if not info:
            return "Unknown job ID."
        info["cancel"] = True
    return f"🛑 Stop requested for {job_id}. It will halt after the current trial."
