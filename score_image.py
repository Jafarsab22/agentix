"""
score_image.py
---------------
Pure detection + scoring helpers.

Detect cues on e-commerce cards (single or 2×4 grid) and score them.
The app should import these functions and pass PHP-loaded parameters.

Public API:
- build_allowed_cues(params, expected, exclude) -> list[str]
- detect_single_from_image(filepath, allowed_labels) -> list[str]
- detect_grid_from_image(filepath, allowed_labels) -> list[set[str]]   # 8 cells row-major
- score_single_card(cues_set, params) -> dict with 'raw','final','sum_s','sum_w','price_weight'
- score_grid_2x4(cards, params) -> dict with per-card + aggregates

Scoring modes

Option A — Linear utility (β·x)
- What it is: Standard conditional-logit style utility. For each present cue j,
  add β_j * x_j to the card’s utility (x_j is usually 0/1, or ln(price) if present).
- Card score: U = Σ_j β_j x_j. The card with the largest U is predicted to be chosen.
- Why use it: Most faithful to the pooled regression; tends to be the most predictive
  in our validations because it mirrors how the model was estimated.
- Price term: ln(price) enters exactly like any other regressor via its β_price.

Option B — Evidence-weighted readiness (M·C·R)
- What it is: Converts each cue into a bounded, direction-aware contribution,
  then normalises by the cue evidence weight to make scores comparable across cards.
- Cue contribution: s_i = sign(β_i) * M_i * C_i * R_i
  where
    M_i = magnitude proxy (from |AME_pp| via tanh(|AME|/k)),
    C_i = confidence (exp(−γ · [ln(CI_high) − ln(CI_low)])),
    R_i = reliability (e.g., evid_score × (1 − q_bh), clamped to [0,1]).
- Card score (readiness): readiness_raw = (Σ_i s_i) / (Σ_i C_i * R_i), clipped to [−1, +1].
  If the card has a price cue, multiply by price_weight (e.g., min(1, |β_price|/β_ref)).
- Why use it: More conservative and comparable across cards when cue quality varies;
  it down-weights noisy or weakly evidenced cues by design, so it’s usually less
  predictive than Option A but more robust as a quality-weighted “readiness” index.

Notes
- Position effects (Row/Column) are added automatically for 2×4 grids; they are ignored
  for single-card scoring unless explicitly supplied.
- Price handling can be contextual:
  • Option A: ln(price) is a standard regressor (β_price · ln(price)).
  • Option B: ln(price) contributes via its own s_i term, and a global price_weight
    can be applied to the card/page score if desired.
"""


from __future__ import annotations
import io, base64, json, math, os, time
from typing import Dict, Any, Iterable, Sequence, Set, Optional, List, Tuple

import numpy as np
from PIL import Image, ImageFilter
import requests

# ---------------------- vocab helpers ----------------------

def build_allowed_cues(params: Dict[str, Dict[str, float]],
                       expected: Iterable[str],
                       exclude: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for name in list(params.keys()) + list(expected):
        if name in exclude:
            continue
        if name not in seen:
            out.append(name); seen.add(name)
    return out

_NORMALISE = {
    # all-in
    "all-in framing": "All-in framing",
    "all in framing": "All-in framing",
    "all-in v. partitioned pricing": "All-in framing",
    "all-in price": "All-in framing",
    "price includes tax": "All-in framing",
    "inc vat": "All-in framing",
    "including vat": "All-in framing",
    "including shipping": "All-in framing",

    # assurance (exclude delivery)
    "assurance": "Assurance",
    "returns": "Assurance",
    "free returns": "Assurance",
    "warranty": "Assurance",
    "guarantee": "Assurance",
    "money-back": "Assurance",

    # scarcity
    "scarcity": "Scarcity tag",
    "scarcity tag": "Scarcity tag",
    "low stock": "Scarcity tag",
    "only x left": "Scarcity tag",
    "limited stock": "Scarcity tag",
    "selling fast": "Scarcity tag",

    # strike-through
    "strike-through": "Strike-through",
    "strikethrough": "Strike-through",
    "sale price": "Strike-through",
    "was now": "Strike-through",
    "was £": "Strike-through",
    "discounted from": "Strike-through",
    "rrp": "Strike-through",
    "list price": "Strike-through",
    "previous price": "Strike-through",

    # timer
    "timer": "Timer",
    "countdown": "Timer",
    "ends in": "Timer",
    "limited time": "Timer",
    "deal ends": "Timer",
    "hours left": "Timer",

    # social proof / ratings
    "social": "social",
    "social proof": "social",
    "x bought": "social",
    "x sold": "social",
    "people viewing": "social",
    "bestseller": "social",
    "ratings": "ratings",
    "stars": "ratings",
    "4.3/5": "ratings",

    # voucher / coupon
    "voucher": "voucher",
    "coupon": "voucher",
    "promo code": "voucher",
    "use code": "voucher",
    "apply voucher": "voucher",
    "clip coupon": "voucher",

    # bundle
    "bundle": "bundle",
    "bundle & save": "bundle",
    "2 for": "bundle",
    "buy 1 get 1": "bundle",
    "multi-buy": "bundle",
}

def _norm_label(x: str) -> str:
    k = (x or "").strip().lower()
    return _NORMALISE.get(k, (x or "").strip())

# ---------------------- prompt & API ----------------------

def _build_detection_prompt(allowed: list[str]) -> str:
    vocab = ", ".join(allowed)
    return (
        "You are an e-commerce UI analyst. Detect these cues: "
        f"{vocab}.\n"
        "Definitions and evidence requirements:\n"
        "All-in framing = the shown price explicitly includes taxes/shipping/fees e.g., ‘£399 inc. VAT’, ‘price includes tax/shipping’. "
        "‘Price excludes VAT’ is NOT all-in. Do not infer from generic ‘Deal’ or delivery text.\n"
        "Assurance = explicit returns/warranty/guarantee statements e.g., ‘30-day returns’, ‘2-year warranty’, ‘money-back guarantee’. "
        "FREE delivery, Prime, and dispatch dates are NOT assurance.\n"
        "Scarcity tag = explicit low stock or limited availability e.g., ‘Only 3 left’, ‘Low stock’, ‘Selling fast’, ‘Limited stock’. "
        "‘In stock’ or delivery dates are NOT scarcity.\n"
        "Strike-through = a price visibly crossed-out OR a textual previous-price marker e.g., crossed-out digits, ‘was’, ‘RRP’, ‘List price’, ‘Previous price’.\n"
        "Timer = a countdown or deadline e.g., ‘Ends in 02:14:10’, ‘limited time’, ‘Sale ends today’, ‘X hours left’.\n"
        "social = social proof e.g., ‘2k bought in last month’, ‘viewing now’, ‘Bestseller’.\n"
        "voucher = coupon/promo e.g., ‘Use code SAVE10’, ‘Apply voucher’, ‘£15 off’, ‘Clip coupon’.\n"
        "bundle = multi-item offer e.g., ‘2 for £50’, ‘Buy 1 get 1 50% off’, ‘Bundle & save’. ‘Pack of 10’ alone is NOT a bundle price deal.\n"
        "ratings = stars or numeric ratings like ‘4.3/5’.\n"
        "Rules: Return STRICT JSON using only the allowed labels; omit any cue without the evidence above. Focus ONLY on the provided crop (one product card).\n"
        "Format: {\"cues\":[...]}.\n"
    )

def _img_to_data_url(img: Image.Image) -> str:
    # Upscale small crops for better tiny-text detection (RRP, “Only 3 left”).
    w, h = img.size
    target_min = 720  # aim for ~720px on the shorter side
    scale = max(1.0, target_min / max(1, min(w, h)))
    img2 = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img2.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

# ---- bounded-latency OpenAI call (no retries unless configured) --------------

def _openai_call(image_data_url: str, prompt: str) -> dict:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {}
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"

    connect_timeout = float(os.getenv("OPENAI_CONNECT_TIMEOUT", "6"))
    read_timeout = float(os.getenv("OPENAI_READ_TIMEOUT", "30"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
    backoff_base = float(os.getenv("OPENAI_BACKOFF_BASE", "1.2"))

    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]},
        ],
        "max_tokens": 800,
        "temperature": 0
    }

    attempts = 1 + max(0, max_retries)
    for i in range(attempts):
        try:
            r = requests.post(url, headers=headers, json=payload,
                              timeout=(connect_timeout, read_timeout))
            if r.status_code >= 400:
                if 500 <= r.status_code < 600 and i < attempts - 1:
                    time.sleep(backoff_base ** (i + 1))
                    continue
                return {}
            txt = r.json()["choices"][0]["message"]["content"]
            try:
                return json.loads(txt)
            except Exception:
                a, b = txt.find("{"), txt.rfind("}")
                return json.loads(txt[a:b+1]) if a >= 0 and b > a else {}
        except requests.exceptions.RequestException:
            if i < attempts - 1:
                time.sleep(backoff_base ** (i + 1))
                continue
            return {}
    return {}

# ---------------------- detection APIs ----------------------
# --- add at top of score_image.py imports if not present ---
from PIL import Image, ImageFilter
import os

# ... keep your existing helpers (_norm_label, _img_to_data_url, _build_detection_prompt, _openai_call) ...


def _detect_from_pil(img: Image.Image, allowed_labels: list[str]) -> list[str]:
    """
    Run the same single-card detection pipeline on a PIL image.
    Includes optional unsharp mask and min-size upscaling to help tiny text.
    Returns canonical labels (deduped) filtered to 'allowed_labels'.
    """
    # optional sharpening (can be disabled via env)
    use_sharpen = os.getenv("AGENTIX_SHARPEN", "1") not in ("0", "false", "False")
    if use_sharpen:
        try:
            img = img.filter(ImageFilter.UnsharpMask(radius=1.6, percent=140))
        except Exception:
            pass

    # upscale when the smallest side is below a threshold (helps “was £…”, tiny text)
    w, h = img.size
    min_target = int(os.getenv("AGENTIX_MIN_SIDE", "820"))  # default ~820px
    if min(w, h) < min_target:
        scale = min_target / float(min(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    data_url = _img_to_data_url(img)
    prompt = _build_detection_prompt(allowed_labels)
    obj = _openai_call(data_url, prompt)  # returns {} on failure/timeout

    raw = [x for x in (obj.get("cues") or []) if isinstance(x, str)]
    out: list[str] = []
    for x in raw:
        canon = _norm_label(x)
        if canon in allowed_labels and canon not in out:
            out.append(canon)
    return out

def detect_single_from_image(filepath: str, allowed_labels: list[str]) -> list[str]:
    im = Image.open(filepath).convert("RGB")
    data_url = _img_to_data_url(im)
    prompt = _build_detection_prompt(allowed_labels)
    obj = _openai_call(data_url, prompt)
    raw = [x for x in (obj.get("cues") or []) if isinstance(x, str)]
    out: list[str] = []
    for x in raw:
        canon = _norm_label(x)
        if canon in allowed_labels and canon not in out:
            out.append(canon)
    return out

# ---- gutter-based 2×4 cropping (prevents neighbour bleed) --------------------

def _to_gray_np(im: Image.Image) -> np.ndarray:
    g = im.convert("L")
    arr = np.asarray(g, dtype=np.float32) / 255.0
    return arr

def _smooth1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    k = max(1, k | 1)  # odd
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same")

def _find_gutter_positions(arr: np.ndarray, n_cuts: int, expected_fracs: list[float]) -> list[int]:
    # "Ink" = darker→higher. Gutters are low-ink troughs.
    ink_cols = np.mean(1.0 - arr, axis=0)
    ink_rows = np.mean(1.0 - arr, axis=1)
    # Choose projection depending on whether we search vertical or horizontal later.
    # This function will be called with either arr or arr.T so we always read along axis=0.
    proj = np.mean(1.0 - arr, axis=0)
    # Smooth to suppress product photos/ratings noise.
    smooth = _smooth1d(proj, max(9, arr.shape[0] // 80))
    # Normalise
    smin, smax = float(smooth.min()), float(smooth.max())
    if smax - smin < 1e-6:
        smooth = np.zeros_like(smooth)
    else:
        smooth = (smooth - smin) / (smax - smin)

    N = len(smooth)
    positions: list[int] = []
    window = max(6, N // 100)

    for frac in expected_fracs:
        guess = int(round(frac * N))
        lo = max(0, guess - 3 * window)
        hi = min(N, guess + 3 * window)
        seg = smooth[lo:hi]
        if len(seg) == 0:
            positions.append(guess)
            continue
        j = int(np.argmin(seg))
        pos = lo + j
        # Deconflict with previously chosen cuts
        for _ in range(3):
            if all(abs(pos - q) >= window for q in positions):
                break
            # move to next local minimum in the segment
            seg2 = seg.copy()
            seg2[max(0, j - window):min(len(seg2), j + window + 1)] = 1.0
            j = int(np.argmin(seg2))
            pos = lo + j
        positions.append(pos)

    positions.sort()
    return positions[:n_cuts]

def _compute_grid_boxes(im: Image.Image) -> list[tuple[int, int, int, int]]:
    """
    Detect approximate 2×4 grid boxes in an Amazon-like screenshot by analysing
    whitespace gutters between cards. Returns a list of 8 (x0, y0, x1, y1)
    tuples in row-major order. Produces debug overlay for visual validation.
    """
    import numpy as np, cv2, os
    os.makedirs("results", exist_ok=True)

    # Convert to grayscale
    arr = np.array(im.convert("L"))
    h, w = arr.shape

    # Gaussian blur → softens edges
    blur = cv2.GaussianBlur(arr, (3, 3), 0)

    # Normalize and invert (white → low value)
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    inv = 255 - norm

    # Horizontal & vertical projections (mean brightness)
    v_proj = np.mean(norm, axis=0)
    h_proj = np.mean(norm, axis=1)

    # Detect vertical gaps (bright → white columns)
    v_thresh = np.percentile(v_proj, 95)
    h_thresh = np.percentile(h_proj, 95)

    v_gap_idx = np.where(v_proj > v_thresh)[0]
    h_gap_idx = np.where(h_proj > h_thresh)[0]

    def _group_gaps(idxs, min_run=10):
        groups, cur = [], []
        for i in idxs:
            if not cur or i - cur[-1] <= 2:
                cur.append(i)
            else:
                if len(cur) >= min_run:
                    groups.append((cur[0], cur[-1]))
                cur = [i]
        if len(cur) >= min_run:
            groups.append((cur[0], cur[-1]))
        return groups

    v_gaps = _group_gaps(v_gap_idx)
    h_gaps = _group_gaps(h_gap_idx)

    # Convert gaps → segment edges (include image borders)
    x_edges = [0] + [int(np.mean(g)) for g in v_gaps] + [w]
    y_edges = [0] + [int(np.mean(g)) for g in h_gaps] + [h]

    # Expect ~5 x-edges and ~3 y-edges (2×4 grid)
    if len(x_edges) < 5:
        step = w // 4
        x_edges = [i * step for i in range(5)]
    if len(y_edges) < 3:
        step = h // 2
        y_edges = [i * step for i in range(3)]

    boxes = []
    for r in range(2):
        for c in range(4):
            x0, x1 = x_edges[c], x_edges[c + 1]
            y0, y1 = y_edges[r], y_edges[r + 1]
            pad = 3  # small margin to avoid overlap
            boxes.append((
                max(0, x0 + pad),
                max(0, y0 + pad),
                min(w, x1 - pad),
                min(h, y1 - pad)
            ))

    # Create debug overlay
    dbg = np.array(im.copy())
    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 3)
        cv2.putText(
            dbg, str(i),
            (x0 + 10, y0 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (255, 0, 0), 3, cv2.LINE_AA
        )

    debug_path = os.path.join("results", "grid_debug.png")
    cv2.imwrite(debug_path, cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] grid overlay saved → {debug_path}")

    return boxes

def detect_single_from_image(filepath: str, allowed_labels: list[str]) -> list[str]:
    """
    Public API (unchanged signature): detect cues on a single image file path.
    Internally delegates to the PIL-based helper for consistency with grid.
    """
    im = Image.open(filepath).convert("RGB")
    return _detect_from_pil(im, allowed_labels)


def detect_grid_from_image(filepath: str, allowed_labels: list[str]) -> list[set]:
    """
    Detect 8 cards in a 2×4 grid by locating gutters (via _compute_grid_boxes),
    then applying the *same* single-card detector to each crop.
    Also saves the 8 crops to results/crops for debugging/preview in the UI.
    Returns list[set] of canonical labels, length 8, row-major.
    """
    import time
    import pathlib
    import glob

    # Hard deadline so we never hang the UI
    deadline = time.time() + float(os.getenv("OPENAI_GRID_DEADLINE_SEC", "70"))

    # Open image and compute 8 cell boxes
    im = Image.open(filepath).convert("RGB")
    boxes = _compute_grid_boxes(im)  # must return 8 (x0,y0,x1,y1) boxes

    # Prepare debug crops dir
    crops_dir = pathlib.Path("results") / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous run’s crops so the UI shows only current ones
    try:
        for p in crops_dir.glob("crop_*.png"):
            p.unlink(missing_ok=True)  # Python 3.8+: wrap in try/except if needed
    except Exception:
        pass

    out: list[set] = []
    for idx, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        if time.time() >= deadline:
            out.append(set())
            continue

        # Crop
        crop = im.crop((x0, y0, x1, y1))

        # Save debug crop (crop_{i}_r{row}_c{col}.png) for display in app.py
        try:
            r = 1 if idx <= 4 else 2
            c = ((idx - 1) % 4) + 1
            fname = crops_dir / f"crop_{idx}_r{r}_c{c}.png"
            crop.save(fname, format="PNG")
        except Exception:
            # non-fatal – keep going
            pass

        # Run the *same* single-card detector the single-image UI uses
        try:
            labels = _detect_from_pil(crop, allowed_labels)
        except Exception:
            labels = []

        # Canonicalise + filter to allowed labels
        clean = set()
        for x in labels:
            canon = _norm_label(x)
            if canon in allowed_labels:
                clean.add(canon)
        out.append(clean)

    # Pad to 8 in case of early deadline
    while len(out) < 8:
        out.append(set())

    return out

# ---------------------- scoring ----------------------

def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)

def _ln_or_none(price: Optional[float]) -> Optional[float]:
    if price is None:
        return None
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    return math.log(price)

def score_single_card(card_cues: Iterable[str],
                      params: Dict[str, Dict[str, float]],
                      *,
                      price: Optional[float] = None,
                      extra_cues: Iterable[str] = ()) -> Dict[str, float]:
    """Readiness-like summary for a single card, plus a simple 'raw'."""
    cues = set(card_cues) | set(extra_cues)
    ln_price = _ln_or_none(price)
    if ln_price is not None:
        cues.add("ln(price)")

    sum_s = 0.0
    sum_w = 0.0
    raw = 0.0

    for cue in cues:
        p = params.get(cue)
        if p is None:
            continue
        beta = float(p.get("beta") or 0.0)
        C = float(p.get("C") or 0.0)
        R = float(p.get("R") or 0.0)
        M = float(p.get("M") or 0.0)
        w_i = C * R
        s_i = float(p.get("s")) if p.get("s") is not None else _sign(beta) * M * C * R
        sum_s += s_i
        sum_w += w_i
        raw += beta

    if sum_w <= 1e-12:
        readiness = 0.0
    else:
        readiness = max(-1.0, min(1.0, sum_s / sum_w))

    if ln_price is not None:
        pp = params.get("ln(price)")
        if pp and pp.get("price_weight") is not None:
            pw = max(0.0, min(1.0, float(pp["price_weight"])))
            readiness *= pw

    return {
        "raw": raw,
        "final": readiness,
        "sum_s": sum_s,
        "sum_w": sum_w,
        "price_weight": (params.get("ln(price)", {}).get("price_weight", None) or 0.0)
    }

def score_card_option_a(card_cues: Iterable[str],
                        params: Dict[str, Dict[str, float]],
                        *, price: Optional[float] = None,
                        extra_cues: Iterable[str] = ()) -> float:
    U = 0.0
    for cue in card_cues:
        p = params.get(cue)
        if p and p.get("beta") is not None:
            U += float(p["beta"])
    for cue in extra_cues:
        p = params.get(cue)
        if p and p.get("beta") is not None:
            U += float(p["beta"])
    ln_price = _ln_or_none(price)
    if ln_price is not None:
        p = params.get("ln(price)")
        if p and p.get("beta") is not None:
            U += float(p["beta"]) * ln_price
    return U

def score_card_option_b(card_cues: Iterable[str],
                        params: Dict[str, Dict[str, float]],
                        *, price: Optional[float] = None,
                        extra_cues: Iterable[str] = ()) -> float:
    sum_s = 0.0
    sum_w = 0.0
    all_cues: Set[str] = set(card_cues) | set(extra_cues)
    ln_price = _ln_or_none(price)
    if ln_price is not None:
        all_cues.add("ln(price)")
    for cue in all_cues:
        p = params.get(cue)
        if not p:
            continue
        C = float(p.get("C") or 0.0)
        R = float(p.get("R") or 0.0)
        w_i = C * R
        if p.get("s") is not None:
            s_i = float(p["s"])
        else:
            beta = float(p.get("beta") or 0.0)
            M = float(p.get("M") or 0.0)
            s_i = _sign(beta) * M * C * R
        sum_s += s_i
        sum_w += w_i
    if sum_w <= 1e-12:
        readiness = 0.0
    else:
        readiness = max(-1.0, min(1.0, sum_s / sum_w))
    if ln_price is not None:
        p_price = params.get("ln(price)")
        if p_price and p_price.get("price_weight") is not None:
            pw = max(0.0, min(1.0, float(p_price["price_weight"])))
            readiness *= pw
    return readiness

def score_grid_2x4(cards: Sequence[dict],
                   params: Dict[str, Dict[str, float]]) -> dict:
    if len(cards) != 8:
        raise ValueError("score_grid_2x4 expects exactly 8 cards (2×4)")

    out_cards: List[dict] = []
    sum_a = 0.0
    sum_b = 0.0
    best_a = float("-inf")
    best_b = float("-inf")

    for idx, card in enumerate(cards):
        r = 1 if idx < 4 else 2
        c = (idx % 4) + 1

        extra = [f"Row {r}"]
        if c == 1:
            extra.append("Column 1")
        elif c == 2:
            extra.append("Column 2")
        elif c == 3:
            extra.append("Column 3")
        # col 4: no extra cue

        cues = set(card.get("cues") or [])
        price = card.get("price")

        s_a = score_card_option_a(cues, params, price=price, extra_cues=extra)
        s_b = score_card_option_b(cues, params, price=price, extra_cues=extra)

        out_cards.append({
            "row": r,
            "col": c,
            "cues": sorted(cues),
            "price": price,
            "option_a": s_a,
            "option_b": s_b,
        })

        sum_a += s_a
        sum_b += s_b
        best_a = max(best_a, s_a)
        best_b = max(best_b, s_b)

    return {
        "cards": out_cards,
        "mean_option_a": sum_a / 8.0,
        "mean_option_b": sum_b / 8.0,
        "best_option_a": best_a,
        "best_option_b": best_b,
    }
