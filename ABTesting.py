# ABTesting.py
# Live A/B testing core (logic + async job handling).
# Builds a simple 2×4 grid with random allocation of images A and B (4 each),
# calls the OpenAI vision agent once per trial, robustly parses its selection,
# and tracks pick rates. UI/formatting is done in app.py.

from __future__ import annotations

import io, os, time, json, base64, threading, random, math
from typing import Dict, Tuple, List, Optional
from PIL import Image

# In-process job store (simple and resilient for small deployments)
_JOBS: Dict[str, dict] = {}
_LOCK = threading.Lock()

# ------------ Utilities to build the 2×4 image grid -------------------------

def _img_to_data_url(img: Image.Image, fmt="JPEG", quality=85) -> str:
    bio = io.BytesIO()
    img.save(bio, fmt, quality=quality)
    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"

def _load_and_fit(path: str, box_w: int, box_h: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((box_w, box_h), Image.LANCZOS)
    # letterbox to exact size
    bg = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    x = (box_w - im.width) // 2
    y = (box_h - im.height) // 2
    bg.paste(im, (x, y))
    return bg

def _compose_grid_2x4(file_a: str, file_b: str, labels: List[str]) -> Tuple[str, List[List[str]]]:
    """
    labels: 8-length list with values "A"/"B" in row-major order.
    Returns: (data_url, 2x4 matrix of labels).
    """
    cell_w, cell_h = 320, 320
    pad = 8
    W = 4 * cell_w + 5 * pad
    H = 2 * cell_h + 3 * pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))

    imgA = _load_and_fit(file_a, cell_w, cell_h)
    imgB = _load_and_fit(file_b, cell_w, cell_h)

    matrix: List[List[str]] = []
    k = 0
    for r in range(2):
        row_labels: List[str] = []
        for c in range(4):
            x = pad + c * (cell_w + pad)
            y = pad + r * (cell_h + pad)
            lab = labels[k]; k += 1
            canvas.paste(imgA if lab == "A" else imgB, (x, y))
            row_labels.append(lab)
        matrix.append(row_labels)

    return _img_to_data_url(canvas, fmt="JPEG", quality=80), matrix

# -------- Robust extraction of (row, col) from model tool/JSON output --------

def _rc_from_args(args: dict) -> Tuple[Optional[int], Optional[int]]:
    """
    Accept common shapes returned by different tool-call formats:
      - {'row': 1|2, 'col': 1|2|3|4}   (1-indexed)
      - {'row': 0|1, 'col': 0|1|2|3}   (0-indexed)
      - {'index'|'slot'|'position'|'choice'|'chosen_index'|'selected_index': 1..8 or 0..7}
      - possibly nested under 'arguments' (str/dict) or 'args'
    Returns zero-based (row, col) or (None, None) if unparseable.
    """
    if not isinstance(args, dict):
        return None, None

    # direct row/col
    def _to_zero_based(v, hi):
        try:
            v = int(v)
        except Exception:
            return None
        if 1 <= v <= hi:
            return v - 1
        if 0 <= v <= (hi - 1):
            return v
        return None

    if "row" in args or "r" in args or "row_idx" in args or "row_index" in args:
        for rk in ("row", "r", "row_idx", "row_index"):
            if rk in args:
                r0 = _to_zero_based(args.get(rk), 2)
                if r0 is None:
                    break
                for ck in ("col", "c", "col_idx", "col_index"):
                    if ck in args:
                        c0 = _to_zero_based(args.get(ck), 4)
                        if c0 is not None:
                            return r0, c0

    # single index
    for ik in ("index", "slot", "position", "choice", "chosen_index", "selected_index"):
        if ik in args:
            try:
                v = int(args.get(ik))
            except Exception:
                v = None
            if v is None:
                continue
            if 1 <= v <= 8:
                v -= 1
            if 0 <= v <= 7:
                return v // 4, v % 4

    # nested payloads
    for vk in ("arguments", "args"):
        if vk in args and isinstance(args[vk], (str, dict)):
            try:
                nested = json.loads(args[vk]) if isinstance(args[vk], str) else dict(args[vk])
            except Exception:
                nested = {}
            r0, c0 = _rc_from_args(nested)
            if r0 is not None and c0 is not None:
                return r0, c0

    return None, None

# ---------------- One trial (build grid → call model → parse pick) ----------------

def _trial_once(file_a: str, file_b: str, category: str | None, model_name: str | None) -> Tuple[Optional[str], List[List[str]]]:
    """
    Build a fresh 2×4 grid with 4×A and 4×B at random positions.
    Call OpenAI vision model via agent_runner.call_openai.
    Return ("A"/"B"/None, matrix_labels).
    """
    labels = ["A"] * 4 + ["B"] * 4
    random.shuffle(labels)
    data_url, mat = _compose_grid_2x4(file_a, file_b, labels)

    from agent_runner import call_openai
    args = call_openai(data_url, category or "", model_name=model_name)

    r0, c0 = _rc_from_args(args)
    if r0 is None or c0 is None:
        return None, mat
    return mat[r0][c0], mat

# ---------------- Background worker and public async API ---------------------

def _bg_run(job_id: str, file_a: str, file_b: str, n_trials: int, category: str, model_name: str):
    try:
        a = b = invalid = 0
        cancelled = False
        for t in range(1, n_trials + 1):
            # allow cancellation
            with _LOCK:
                if _JOBS.get(job_id, {}).get("cancel"):
                    cancelled = True
            if cancelled:
                break

            chosen, layout = _trial_once(file_a, file_b, category, model_name)
            if chosen == "A":
                a += 1
            elif chosen == "B":
                b += 1
            else:
                invalid += 1

            if (t % 10 == 0) or (t == n_trials):
                with _LOCK:
                    _JOBS[job_id] = {
                        "status": "running", "progress": t, "total": n_trials,
                        "a": a, "b": b, "invalid": invalid, "cancel": _JOBS.get(job_id, {}).get("cancel", False)
                    }

        completed = max(1, a + b + invalid)  # trials attempted (at least 1 for denominators)
        valid = max(1, a + b)

        ra = a / valid
        rb = b / valid

        # Two-proportion z (pooled) using valid picks only
        p_pool = (a + b) / (2 * max(1, valid))
        se = math.sqrt(p_pool * (1 - p_pool) * (2 / max(1, valid)))
        z = (ra - rb) / se if se > 0 else 0.0

        def ci(p, n):
            s = math.sqrt(max(p * (1 - p) / max(1, n), 0.0))
            return (p - 1.96 * s, p + 1.96 * s)

        res = {
            "a": a, "b": b, "invalid": invalid, "n_trials": completed,
            "rate_a": ra, "rate_b": rb,
            "ci_a": ci(ra, valid), "ci_b": ci(rb, valid),
            "z_two_prop": z,
            "detector_note": f"{'Cancelled early; ' if cancelled else ''}one API call per trial; positions randomised each trial."
        }
        with _LOCK:
            _JOBS[job_id] = {"status": ("cancelled" if cancelled else "done"), "result": res}
    except Exception as e:
        with _LOCK:
            _JOBS[job_id] = {"status": "error", "error": f"{type(e).__name__}: {e}"}

def submit_live_ab(file_a: str, file_b: str, n_trials: int, category: str = "", model_name: str = "") -> Dict:
    if not file_a or not file_b:
        return {"ok": False, "error": "missing_files"}
    try:
        n = int(n_trials)
        if n <= 0:
            return {"ok": False, "error": "bad_trials"}
    except Exception:
        return {"ok": False, "error": "bad_trials"}

    job_id = f"ab-{uuid4_short()}"
    with _LOCK:
        _JOBS[job_id] = {"status": "queued", "progress": 0, "total": n, "a": 0, "b": 0, "invalid": 0, "cancel": False}
    th = threading.Thread(
        target=_bg_run,
        args=(job_id, file_a, file_b, n, category or "", model_name or ""),
        name=f"live-ab-{job_id}",
        daemon=True
    )
    th.start()
    return {"ok": True, "job_id": job_id}

def poll_live_ab(job_id: str) -> Dict:
    job_id = (job_id or "").strip()
    with _LOCK:
        info = _JOBS.get(job_id)
    if not info:
        return {"ok": True, "status": "unknown"}
    if info.get("status") == "running":
        return {
            "ok": True, "status": "running",
            "progress": info.get("progress", 0), "total": info.get("total", 0),
            "a": info.get("a", 0), "b": info.get("b", 0), "invalid": info.get("invalid", 0)
        }
    if info.get("status") == "error":
        return {"ok": True, "status": "error", "error": info.get("error","")}
    return {"ok": True, "status": info.get("status", "unknown")}

def fetch_live_ab(job_id: str) -> Dict:
    job_id = (job_id or "").strip()
    with _LOCK:
        info = _JOBS.get(job_id)
    if not info:
        return {"ok": False, "error": "unknown_job"}
    if info.get("status") != "done" and info.get("status") != "cancelled":
        return {"ok": False, "error": "not_ready", "status": info.get("status")}
    return {"ok": True, "status": info.get("status"), "result": info.get("result", {})}

def cancel_live_ab(job_id: str) -> Dict:
    job_id = (job_id or "").strip()
    with _LOCK:
        if job_id not in _JOBS:
            return {"ok": False, "error": "unknown_job"}
        _JOBS[job_id]["cancel"] = True
    return {"ok": True}

# ----------------- small helpers -----------------

def uuid4_short() -> str:
    import uuid
    return uuid.uuid4().hex[:8]
