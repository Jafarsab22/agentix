"""
score_image.py — cell‑aware detector + scoring
------------------------------------------------
Detect cues on e‑commerce cards (single or 2×4 grid) and score them.
Public API (unchanged):
- build_allowed_cues(params, expected, exclude) -> list[str]
- detect_single_from_image(filepath, allowed_labels) -> list[str]
- detect_grid_from_image(filepath, allowed_labels) -> list[set[str]]   # 8 cells row‑major
- score_single_card(cues_set, params, *, price=None, extra_cues=()) -> dict
- score_grid_2x4(cards, params) -> dict

Notes
- No dependency on app.py. Keep your app as is; it already imports these symbols.
- Per‑cell detection now runs on TWO focused crops in each grid cell:
  (1) the full cell; (2) a bottom text band (where strike‑through, voucher, scarcity tend to appear).
  Results are unioned with label normalisation and filtered to allowed labels.
- Robustness: each crop is upscaled for tiny text; timeouts are bounded; any failure yields an empty set for that cell.
- Optional debugging: set DEBUG_CROPS=1 to emit per‑cell PNG crops into results/crops/.
"""

from __future__ import annotations
import io, base64, json, math, os, time, pathlib
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
        "You are an e-commerce UI analyst. Detect these cues: "
        f"{vocab}.\n"
        "Definitions and evidence requirements:\n"
        "All-in framing = the shown price explicitly includes taxes/shipping/fees (e.g., ‘£399 inc. VAT’, ‘price includes tax/shipping’). "
        "‘Price excludes VAT’ is NOT all-in. Do not infer from generic ‘Deal’ or delivery text.\n"
        "Assurance = explicit returns/warranty/guarantee statements (e.g., ‘30-day returns’, ‘2-year warranty’, ‘money-back guarantee’). "
        "FREE / fast delivery, Prime, and dispatch dates are NOT assurance.\n"
        "Scarcity tag = explicit low stock or limited availability (e.g., ‘Only 3 left’, ‘Low stock’, ‘Selling fast’, ‘Limited stock’). "
        "‘In stock’ or delivery dates are NOT scarcity.\n"
        "Strike-through = a price visibly crossed-out OR a textual previous-price marker (crossed-out digits, ‘was’, ‘RRP’, ‘List price’, ‘Previous price’).\n"
        "Timer = a countdown or deadline (‘Ends in 02:14:10’, ‘limited time’, ‘Sale ends today’, ‘X hours left’).\n"
        "social = social proof (‘bought in last month’, ‘viewing now’, ‘Bestseller’).\n"
        "voucher = coupon/promo (‘Use code SAVE10’, ‘Apply voucher’, ‘Clip coupon’).\n"
        "bundle = multi-item offer (‘2 for £50’, ‘Buy 1 get 1 50% off’, ‘Bundle & save’). ‘Pack of 10’ alone is NOT a bundle price deal.\n"
        "ratings = stars or numeric ratings like ‘4.3/5’.\n"
        "Rules: Return STRICT JSON using only the allowed labels; omit any cue without the evidence above. Focus ONLY on the provided crop (one product card).\n"
        "Format: {\"cues\":[...]}.\n"
    )

# minimal, robust data-URL builder with optional upscaling

def _img_to_data_url(img: Image.Image, *, min_w=640, min_h=420) -> str:
    w, h = img.size
    scale = max(1.0, max(min_w / max(1, w), min_h / max(1, h)))
    if scale > 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# OpenAI call with bounded time + retries; returns {} on failure

def _openai_call(image_data_url: str, prompt: str) -> dict:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {}
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"

    connect_timeout = float(os.getenv("OPENAI_CONNECT_TIMEOUT", "6"))
    read_timeout = float(os.getenv("OPENAI_READ_TIMEOUT", "30"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
    backoff = float(os.getenv("OPENAI_BACKOFF_BASE", "1.5"))

    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_data_url}}]},
        ],
        "max_tokens": 500,
        "temperature": 0
    }

    attempt = 0
    while True:
        attempt += 1
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=(connect_timeout, read_timeout))
            if r.status_code >= 400:
                if 500 <= r.status_code < 600 and attempt <= max_retries:
                    time.sleep(backoff ** attempt)
                    continue
                return {}
            txt = r.json()["choices"][0]["message"]["content"]
            try:
                return json.loads(txt)
            except Exception:
                a, b = txt.find("{"), txt.rfind("}")
                return json.loads(txt[a:b+1]) if a >= 0 and b > a else {}
        except requests.exceptions.RequestException:
            if attempt <= max_retries:
                time.sleep(backoff ** attempt)
                continue
            return {}

# ---------------------- detection APIs ----------------------

def detect_single_from_image(filepath: str, allowed_labels: list[str]) -> list[str]:
    im = Image.open(filepath).convert("RGB")
    data_url = _img_to_data_url(im, min_w=800, min_h=520)
    prompt = _build_detection_prompt(allowed_labels)
    obj = _openai_call(data_url, prompt)
    raw = [x for x in (obj.get("cues") or []) if isinstance(x, str)]
    out: list[str] = []
    for x in raw:
        canon = _norm_label(x)
        if canon in allowed_labels and canon not in out:
            out.append(canon)
    return out


def _save_debug_crop(img: Image.Image, *, tag: str, r: int, c: int, k: str):
    if os.getenv("DEBUG_CROPS", "0") != "1":
        return
    d = pathlib.Path("results") / "crops"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"cell{r}{c}_{k}.png"
    try:
        img.save(p)
    except Exception:
        pass


def detect_grid_from_image(filepath: str, allowed_labels: list[str]) -> List[Set[str]]:
    """Per‑cell detection on a 2×4 grid.
    Strategy per cell: analyse (A) full cell; (B) a bottom text band (≈ last 35% height).
    Union labels from A and B after normalisation and filtering. Returns 8 sets row‑major.
    Overall deadline can be tuned via OPENAI_GRID_DEADLINE_SEC (default 70s).
    """
    deadline = time.time() + float(os.getenv("OPENAI_GRID_DEADLINE_SEC", "70"))
    im = Image.open(filepath).convert("RGB")
    W, H = im.size
    rows, cols = 2, 4
    out: List[Set[str]] = []
    prompt = _build_detection_prompt(allowed_labels)

    for rr in range(rows):
        for cc in range(cols):
            if time.time() >= deadline:
                out.append(set()); continue

            x0 = int(cc * W / cols); x1 = int((cc + 1) * W / cols)
            y0 = int(rr * H / rows); y1 = int((rr + 1) * H / rows)
            cell = im.crop((x0, y0, x1, y1))

            # small padding to include outer shadows/tooltips
            pad_x = max(6, (x1 - x0) // 60)
            pad_y = max(6, (y1 - y0) // 60)
            x0p = max(0, x0 - pad_x); y0p = max(0, y0 - pad_y)
            x1p = min(W, x1 + pad_x); y1p = min(H, y1 + pad_y)
            cell = im.crop((x0p, y0p, x1p, y1p))

            # (A) full cell
            a_url = _img_to_data_url(cell, min_w=720, min_h=480)
            _save_debug_crop(cell, tag="full", r=rr+1, c=cc+1, k="A")
            obj_a = _openai_call(a_url, prompt)
            labels_a = obj_a.get("cues") if isinstance(obj_a.get("cues"), list) else []

            # (B) bottom text band (~lower 35%) — often where prices/badges live
            cw, ch = cell.size
            band_h = int(ch * 0.35)
            text_band = cell.crop((0, ch - band_h, cw, ch))
            _save_debug_crop(text_band, tag="band", r=rr+1, c=cc+1, k="B")
            b_url = _img_to_data_url(text_band, min_w=800, min_h=360)
            obj_b = _openai_call(b_url, prompt)
            labels_b = obj_b.get("cues") if isinstance(obj_b.get("cues"), list) else []

            clean: Set[str] = set()
            for src in (labels_a or []) + (labels_b or []):
                if not isinstance(src, str):
                    continue
                canon = _norm_label(src)
                if canon in allowed_labels:
                    clean.add(canon)

            out.append(clean)

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

    readiness = 0.0 if sum_w <= 1e-12 else max(-1.0, min(1.0, sum_s / sum_w))

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
    readiness = 0.0 if sum_w <= 1e-12 else max(-1.0, min(1.0, sum_s / sum_w))
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
