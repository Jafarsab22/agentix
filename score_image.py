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
"""

from __future__ import annotations
import io, base64, json, math, os, time
from typing import Dict, Any, Iterable, Sequence, Set, Optional, List
from PIL import Image
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
        "You are an e-commerce UI analyst. Detect ONLY these cues (use exactly these labels): "
        f"{vocab}.\n"
        "Definitions and evidence requirements:\n"
        "All-in framing = the shown price explicitly includes taxes/shipping/fees (e.g., “£399 inc. VAT”, “price includes tax/shipping”). "
        "“Price excludes VAT” is NOT all-in. Do not infer from generic ‘Deal’ or delivery text.\n"
        "Assurance = explicit returns/warranty/guarantee statements (e.g., “30-day returns”, “2-year warranty”, “money-back guarantee”). "
        "FREE / fast delivery, Prime, and dispatch dates are NOT assurance.\n"
        "Scarcity tag = explicit low stock or limited availability (e.g., “Only 3 left”, “Low stock”, “Selling fast”, “Limited stock”). "
        "“In stock” or delivery dates are NOT scarcity.\n"
        "Strike-through = a price visibly crossed-out OR a textual previous-price marker (evidence must include one of: a crossed-out number; "
        "‘was £’, ‘RRP’, ‘List price’, ‘Previous price’, ‘Save £’ next to the price). ‘Deal’, ‘Prime’, or coloured badges alone are NOT strike-through.\n"
        "Timer = a countdown or deadline (e.g., “Ends in 02:14:10”, “Sale ends today”, “X hours left”).\n"
        "social = social proof (stars, 1–5 ★, review counts, “bought”, “viewing now”, “Bestseller”).\n"
        "voucher = coupon/promo (e.g., “Use code SAVE10”, “Apply voucher”, “Clip coupon”).\n"
        "bundle = multi-item offer (e.g., “2 for £50”, “Buy 1 get 1 50% off”, “Bundle & save”). “Pack of 10” alone is NOT a bundle price deal.\n"
        "ratings = star graphics or numeric ratings like “4.3/5”.\n"
        "Rules: Return STRICT JSON using only the allowed labels; omit any cue without the evidence above. Zoom into fine print.\n"
        "Formats: Single → {\"cues\":[...]}; Grid 2×4 → {\"grid\":[[...],...]} (8 arrays row-major).\n"
    )

def _img_to_data_url(img: Image.Image) -> str:
    # upscale small crops for better OCR on tiny “RRP/Only 1 left”
    w, h = img.size
    scale = max(1.0, 512 / max(1, min(w, h)))
    img2 = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img2.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

def _openai_call(image_data_url: str, prompt: str) -> dict:
    """
    Robust call with retries and configurable timeouts.
    Env vars:
      OPENAI_API_KEY (required)
      OPENAI_MODEL (default: gpt-4.1-mini)
      OPENAI_BASE_URL (default: https://api.openai.com/v1)
      OPENAI_CONNECT_TIMEOUT (default: 30)  # seconds
      OPENAI_MAX_RETRIES (default: 3)
      OPENAI_BACKOFF_BASE (default: 1.5)
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"

    connect_timeout = float(os.getenv("OPENAI_CONNECT_TIMEOUT", "30"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    backoff_base = float(os.getenv("OPENAI_BACKOFF_BASE", "1.5"))

    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]},
        ],
        "max_tokens": 600,
        "temperature": 0
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload,
                              timeout=(connect_timeout, 240))
            if r.status_code >= 400:
                # transient 5xx → retry; otherwise break
                if 500 <= r.status_code < 600 and attempt < max_retries:
                    time.sleep(backoff_base ** attempt)
                    continue
                raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:500]}")
            txt = r.json()["choices"][0]["message"]["content"]
            try:
                return json.loads(txt)
            except Exception:
                i, j = txt.find("{"), txt.rfind("}")
                return json.loads(txt[i:j+1]) if i >= 0 and j > i else {}
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff_base ** attempt)
                continue
            # give a safe empty object so upstream can continue gracefully
            return {}
    # Fallback in the unlikely case loop ends oddly
    if last_err:
        print(f"[vision] final failure after retries: {last_err}", flush=True)
    return {}

# ---------------------- detection APIs ----------------------

def detect_single_from_image(filepath: str, allowed_labels: list[str]) -> list[str]:
    im = Image.open(filepath).convert("RGB")
    data_url = _img_to_data_url(im)
    prompt = _build_detection_prompt(allowed_labels)
    obj = _openai_call(data_url, prompt)  # returns {} on persistent failure
    raw = [x for x in (obj.get("cues") or []) if isinstance(x, str)]
    out: list[str] = []
    for x in raw:
        canon = _norm_label(x)
        if canon in allowed_labels and canon not in out:
            out.append(canon)
    return out

def detect_grid_from_image(filepath: str, allowed_labels: list[str]) -> list[Set[str]]:
    """Return 8 sets of labels, row-major, by cropping each cell and calling the detector.
       If a cell times out, it yields an empty set and processing continues."""
    im = Image.open(filepath).convert("RGB")
    W, H = im.size
    rows, cols = 2, 4
    out: list[Set[str]] = []
    prompt = _build_detection_prompt(allowed_labels)

    for r in range(rows):
        for c in range(cols):
            x0 = int(c * W/cols); x1 = int((c+1) * W/cols)
            y0 = int(r * H/rows); y1 = int((r+1) * H/rows)
            crop = im.crop((x0, y0, x1, y1))
            data_url = _img_to_data_url(crop)

            obj = _openai_call(data_url, prompt)  # safe: {} on failure
            labels = []
            if isinstance(obj.get("cues"), list):
                labels = obj["cues"]
            elif isinstance(obj.get("grid"), list) and obj["grid"]:
                labels = obj["grid"][0]

            clean: Set[str] = set()
            for x in labels:
                if not isinstance(x, str):
                    continue
                canon = _norm_label(x)
                if canon in allowed_labels:
                    clean.add(canon)
            out.append(clean)
    # Always return 8 cells
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
