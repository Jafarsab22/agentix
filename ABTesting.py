# -*- coding: utf-8 -*-
"""
Live A/B tester — instrumented for debugging.
Builds a 2×4 grid (4×A, 4×B) per trial, calls OpenAI with a forced tool call,
extracts a single selection, and tracks A/B picks.

Adds:
  • Robust parsing of many tool-call formats.
  • Per-trial diagnostics (written to results/ab_debug_<job_id>.jsonl).
  • Aggregate diagnostic counters returned in the JSON you already fetch in the UI.
"""

from __future__ import annotations
import os, io, json, time, threading, random, base64, math, pathlib
from typing import Dict, Tuple, Any
from PIL import Image

# ---------------- In-memory job store ----------------
_JOBS: Dict[str, dict] = {}
_JLOCK = threading.Lock()

RESULTS_DIR = pathlib.Path("results"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Tool schema & prompt ----------------
SCHEMA_JSON = {
  "type": "object",
  "properties": {
    "chosen_title": {"type": "string"},   # optional
    "row": {"type": "integer"},
    "col": {"type": "integer"}
  },
  "required": ["row", "col"],
  "additionalProperties": False
}


SYSTEM_PROMPT = (
  "You are a shopping assistant choosing exactly ONE product from a 2×4 grid. "
  "Return ONLY a function/tool call with fields row and col (0-based). "
  "If a title field exists, set chosen_title to ''. No prose."
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

    # light grid lines (helps some models localise cells)
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        for c in range(1, 4):
            x = pad + c * (cell_w + pad) - (pad // 2)
            draw.line([(x, pad), (x, H - pad)], fill=(220, 220, 220), width=1)
        y = pad + cell_h + pad // 2
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

def _openai_choose(image_b64: str, category: str = "", model_name: str | None = None) -> tuple[dict, dict]:
    import requests
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
        "max_tokens": 512,
        "temperature": 0
    }

    r = _post_with_retries_openai(url, headers, data)
    diag = {"model": model, "status_code": r.status_code, "tool_calls": False, "raw_args": None, "parse_error": None}
    if r.status_code >= 400:
        diag["error_text"] = r.text[:600]
        return {}, diag

    js = r.json()
    msg = js["choices"][0]["message"]
    tcs = msg.get("tool_calls", [])
    # Preferred: tool call
    if tcs:
        diag["tool_calls"] = True
        raw = tcs[0]["function"].get("arguments")
        diag["raw_args"] = raw if isinstance(raw, str) else (json.dumps(raw) if isinstance(raw, dict) else str(raw))
        if isinstance(raw, str):
            try:
                return json.loads(raw), diag
            except Exception as e:
                diag["parse_error"] = f"tool_args_json_parse: {type(e).__name__}"
                return {}, diag
        if isinstance(raw, dict):
            return raw, diag
        diag["parse_error"] = "tool_args_unexpected_type"
        return {}, diag

    # Fallback: model put JSON in content instead of tool
    raw_text = (msg.get("content") or "").strip()
    diag["tool_calls"] = False
    diag["raw_args"] = raw_text[:600]
    if raw_text:
        # try to extract the first {...} block
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict):
                return obj, diag
        except Exception:
            i, j = raw_text.find("{"), raw_text.rfind("}")
            if i >= 0 and j > i:
                try:
                    obj = json.loads(raw_text[i:j+1])
                    if isinstance(obj, dict):
                        return obj, diag
                except Exception as e:
                    diag["parse_error"] = f"content_json_slice_parse: {type(e).__name__}"
                    return {}, diag
    return {}, diag


def _rc_from_args(args: dict) -> tuple[int | None, int | None]:
    """
    Robustly extract zero-based (row, col) from a wide variety of shapes:
      - row/col (0- or 1-indexed), possibly as strings or floats
      - single index under: index/slot/position/choice/chosen_index/selected_index
      - 'coordinates': {'row':..., 'column':...} or {'r':..., 'c':...}
      - alternate names: 'column', 'col_index', 'row_index'
      - string patterns: 'r2c3', 'row 2, col 3', 'top-left', 'bottom right', etc.
      - nested JSON under 'arguments' or 'args'
    """
    def _to_int(x):
        try:
            if isinstance(x, (int,)):
                return x
            if isinstance(x, float):
                return int(x)
            if isinstance(x, str):
                s = x.strip().lower()
                # handle numerals inside strings e.g. "2", " 3 "
                return int(float(s))
        except Exception:
            return None
        return None

    if not isinstance(args, dict):
        return None, None

    # direct row/col under several names
    row_keys = ("row", "r", "row_idx", "row_index")
    col_keys = ("col", "c", "col_idx", "col_index", "column", "column_index")
    r = c = None
    for rk in row_keys:
        if rk in args:
            r = _to_int(args[rk])
            break
    for ck in col_keys:
        if ck in args:
            c = _to_int(args[ck])
            break
    if r is not None and c is not None:
        # normalise 1-based to 0-based if it looks like 1..2 or 1..4
        if 1 <= r <= 2: r -= 1
        if 1 <= c <= 4: c -= 1
        if 0 <= r <= 1 and 0 <= c <= 3:
            return r, c

    # single index
    for k in ("index", "slot", "position", "choice", "chosen_index", "selected_index"):
        if k in args:
            v = _to_int(args[k])
            if v is None:
                continue
            if 1 <= v <= 8: v -= 1
            if 0 <= v <= 7:
                return v // 4, v % 4

    # nested coordinates blocks
    for key in ("coordinates", "coord", "pos"):
        if key in args and isinstance(args[key], dict):
            r2, c2 = _rc_from_args(args[key])
            if r2 is not None and c2 is not None:
                return r2, c2

    # nested tool-args in strings or dicts
    for vk in ("arguments", "args"):
        if vk in args and isinstance(args[vk], (str, dict)):
            try:
                nested = json.loads(args[vk]) if isinstance(args[vk], str) else dict(args[vk])
            except Exception:
                nested = {}
            r2, c2 = _rc_from_args(nested)
            if r2 is not None and c2 is not None:
                return r2, c2

    # string patterns
    for k, v in list(args.items()):
        if isinstance(v, str):
            s = v.strip().lower()
            # r2c3 pattern
            if s.startswith("r") and "c" in s:
                try:
                    parts = s.replace("row", "r").replace("col", "c").replace(" ", "")
                    rpart = parts.split("c")[0][1:]
                    cpart = parts.split("c")[1]
                    rr, cc = _to_int(rpart), _to_int(cpart)
                    if rr is not None and cc is not None:
                        if 1 <= rr <= 2: rr -= 1
                        if 1 <= cc <= 4: cc -= 1
                        if 0 <= rr <= 1 and 0 <= cc <= 3:
                            return rr, cc
                except Exception:
                    pass
            # 'row 2, col 3'
            if "row" in s and "col" in s:
                try:
                    import re
                    m = re.search(r"row\s*([0-9]+).*(col|column)\s*([0-9]+)", s)
                    if m:
                        rr, cc = _to_int(m.group(1)), _to_int(m.group(3))
                        if 1 <= rr <= 2: rr -= 1
                        if 1 <= cc <= 4: cc -= 1
                        if 0 <= rr <= 1 and 0 <= cc <= 3:
                            return rr, cc
                except Exception:
                    pass
            # 'top-left', 'bottom right'
            if any(w in s for w in ("top", "bottom", "left", "right")):
                rr = 0 if "top" in s else (1 if "bottom" in s else None)
                cc = 0 if "left" in s else (1 if "center" in s else (3 if "far right" in s else (2 if "right" in s else None)))
                if rr is not None and cc is not None:
                    return rr, cc

    return None, None

# ---------------- One trial ----------------
def _trial_once(file_a: str, file_b: str, category: str | None, model_name: str | None, dbg_file) -> tuple[str | None, list[list[str]], dict]:
    labels = ["A"] * 4 + ["B"] * 4
    random.shuffle(labels)
    data_url, layout = _compose_grid_2x4(file_a, file_b, labels)

    args, diag = _openai_choose(data_url, category or "", model_name=model_name)
    r0, c0 = _rc_from_args(args)
    chosen = None if (r0 is None or c0 is None) else layout[r0][c0]

    # write one compact line per trial to JSONL
    try:
        dbg_file.write(json.dumps({
            "t": int(time.time()),
            "tool_calls": diag.get("tool_calls", False),
            "parse_error": diag.get("parse_error"),
            "args": args,
            "r0": r0, "c0": c0,
            "chosen": chosen
        }, ensure_ascii=False) + "\n")
        dbg_file.flush()
    except Exception:
        pass

    return chosen, layout, diag

# ---------------- Background worker ----------------
def _bg_run_live_ab(job_id: str, file_a: str, file_b: str, n_trials: int, category: str | None, model_name: str | None):
    try:
        a = b = invalid = 0
        diag_counts = {"api_calls": 0, "tool_missing": 0, "parse_error": 0, "null_coords": 0, "oob_coords": 0}
        examples: list[dict[str, Any]] = []  # keep up to 5 diagnostic examples

        dbg_path = RESULTS_DIR / f"ab_debug_{job_id}.jsonl"
        with open(dbg_path, "a", encoding="utf-8") as dbg:
            for t in range(1, n_trials + 1):
                with _JLOCK:
                    if _JOBS.get(job_id, {}).get("cancel", False):
                        break

                chosen, layout, diag = _trial_once(file_a, file_b, category, model_name, dbg)
                diag_counts["api_calls"] += 1
                if not diag.get("tool_calls", False):
                    diag_counts["tool_missing"] += 1
                if diag.get("parse_error"):
                    diag_counts["parse_error"] += 1

                # track A/B or invalid
                if chosen == "A":
                    a += 1
                elif chosen == "B":
                    b += 1
                else:
                    diag_counts["null_coords"] += 1
                    invalid += 1

                # retain a few recent examples
                if len(examples) < 5:
                    examples.append({
                        "raw_args": diag.get("raw_args"),
                        "tool_calls": diag.get("tool_calls", False),
                        "parse_error": diag.get("parse_error"),
                        "mapped_choice": chosen
                    })

                if (t % 10 == 0) or (t == n_trials):
                    with _JLOCK:
                        _JOBS[job_id] = {
                            "status": "running", "progress": t, "total": n_trials,
                            "a": a, "b": b, "invalid": invalid, "cancel": _JOBS.get(job_id, {}).get("cancel", False)
                        }

        # summarise
        valid = max(1, a + b)
        ra, rb = a / valid, b / valid
        p_pool = (a + b) / (2 * max(1, n_trials))
        se = math.sqrt(max(p_pool * (1 - p_pool) * (2 / max(1, n_trials)), 0.0))
        z = (ra - rb) / se if se > 0 else 0.0

        def ci(p, n):
            s = math.sqrt(max(p * (1 - p) / max(1, n), 0.0))
            return (p - 1.96 * s, p + 1.96 * s)

        res = {
            "a": a, "b": b, "invalid": invalid, "n_trials": n_trials,
            "rate_a": ra, "rate_b": rb, "ci_a": ci(ra, valid), "ci_b": ci(rb, valid),
            "z_two_prop": z,
            "note": "one OpenAI call per trial; 4×A/4×B with randomised positions each trial.",
            "diagnostics": {
                "job_debug_file": str(dbg_path),
                "counts": diag_counts,
                "examples": examples
            }
        }
        with _JLOCK:
            _JOBS[job_id] = {"status": "done", "result": res}
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
    if info.get("status") != "done":
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
    # Return summary JSON including diagnostics; your UI already shows this JSON.
    return md, json.dumps(r, ensure_ascii=False, indent=2)

def cancel_live_ab(job_id: str):
    job_id = (job_id or "").strip()
    with _JLOCK:
        info = _JOBS.get(job_id)
        if not info:
            return "Unknown job ID."
        info["cancel"] = True
    return f"🛑 Stop requested for {job_id}. It will halt after the current trial."
