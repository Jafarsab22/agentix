# -*- coding: utf-8 -*-
"""
Agentix storefront (Railway) — v1.8
Date: 2025-10-29

Purpose
    Render an 8-card e-commerce screen with orthogonalised levers (pricing frame,
    individual badges, position, and price schedule) and emit per-card ground-truth
    rows that align exactly with the logit estimator’s expected columns.

Identification & design assumptions
    • One choice per screen (exactly 8 alternatives).
    • Pricing frame is an independent factor. If both frames are enabled, assign
      exactly four ALL-IN and four PARTITIONED cards per screen (blocked 4/4).
      If only one (or neither) is enabled, the frame is fixed (default ALL-IN).
    • Non-frame visual badges are mutually exclusive on a card; draw exactly one
      per card, uniformly from the user-enabled subset (assurance, scarcity,
      strike-through, timer, social proof, voucher, bundle). If none are enabled,
      show no non-frame badge.
    • Prices follow a log-symmetric, 8-level schedule around the anchor using a
      fixed 8×8 Latin square to remove price–position confounds.
    • Ground truth includes: title, row/col, row_top, col1–col3, frame (1=ALL-IN),
      all separate badge flags, price, ln_price, case_id (=set_id), set_id,
      and category. This matches the logit API (which renames case_id→screen_id).

Estimator alignment (with logit_badges.py)
    • Position dummies: row_top, col1, col2, col3 (baseline: bottom row, 4th col).
    • Frame variable: frame (1 = ALL-IN, 0 = PARTITIONED).
    • Separate non-frame badges: assurance, scarcity, strike, timer,
      social_proof, voucher, bundle.
    • Price variables: price (level) and ln_price (log).
    • Keys present for later merges: case_id (= set_id), set_id, category, title.
    • No ‘frame’–badge collinearity: exactly one non-frame badge per card; frame
      is assigned independently of the non-frame draw.

Acronyms
    FE   = Fixed Effects
    AME  = Average Marginal Effect
    UI   = User Interface
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import json
import math
import pathlib
import random

# ------------------------------
# Constants & helpers
# ------------------------------

_LOG_DELTAS = (
    -0.3567, -0.2231, -0.1053, -0.0513,
     0.0513,  0.1053,  0.2231,  0.3567
)
_PRICE_MULTIPLIERS = tuple(math.exp(x) for x in _LOG_DELTAS)  # ~ [×0.70 … ×1.43]

@dataclass(frozen=True)
class Seeds:
    catalog_seed: int
    brand: str
    category: str

    def rng(self, *components: int | str) -> random.Random:
        """
        Deterministic, low-collision seed mixer using Python's hash() of a tuple
        of components XORed with a masked base seed. Returns a fresh Random().
        """
        base = int(self.catalog_seed) & 0x7FFFFFFF
        mix = base ^ (abs(hash((self.brand, self.category))) & 0x7FFFFFFF)
        for c in components:
            mix ^= abs(hash(c)) & 0x7FFFFFFF
        return random.Random(mix)

def _screen_index(set_id: str) -> int:
    """Extract a numeric index from set_id for Latin schedule block selection."""
    digits = "".join(ch for ch in str(set_id) if ch.isdigit())
    try:
        return int(digits) if digits else 1
    except Exception:
        return 1

# 8×8 Latin square L where L[r][c] = (c + r) % 8 to permute price levels by position
_LATIN_8 = [[(c + r) % 8 for c in range(8)] for r in range(8)]

def _price_levels_for_block(anchor: float, seeds: Seeds, block_id: int) -> List[float]:
    """
    Shuffle the 8 multiplicative price multipliers deterministically per block.
    """
    multipliers = list(_PRICE_MULTIPLIERS)
    rng = seeds.rng("price-levels", block_id)
    rng.shuffle(multipliers)
    return [round(anchor * m, 2) for m in multipliers]

def _price_for_card(anchor: float, set_id: str, card_index: int, seeds: Seeds) -> float:
    """
    Pick a price level for the card using the Latin square so that within a screen
    each relative price appears exactly once at each position.
    """
    idx = _screen_index(set_id)
    block, within = (idx - 1) // 8, (idx - 1) % 8
    price_levels = _price_levels_for_block(anchor, seeds, block)
    level_index = _LATIN_8[within][card_index]
    return price_levels[level_index]

def _format_currency(x: float, currency: str) -> str:
    return f"{currency}{x:,.2f}"

def _partition_total_into_components(total: float, seeds: Seeds, set_id: str) -> tuple[float, float, float, float]:
    """
    Decompose total into base + fees + shipping + tax with small randomised rates.
    Rounds components to two decimals; residual goes to base to sum exactly.
    """
    rng = seeds.rng("partition", set_id)
    tax_rate = rng.uniform(0.09, 0.11)
    fees_rate = rng.uniform(0.05, 0.07)
    ship_rate = rng.uniform(0.03, 0.05)

    base = total / (1 + tax_rate + fees_rate + ship_rate)
    fees = base * fees_rate
    ship = base * ship_rate
    tax = base * tax_rate

    base_r = round(base, 2)
    fees_r = round(fees, 2)
    ship_r = round(ship, 2)
    tax_r  = round(tax, 2)

    residual = round(total - (base_r + fees_r + ship_r + tax_r), 2)
    base_r = round(base_r + residual, 2)
    return base_r, fees_r, ship_r, tax_r

# ------------------------------
# Frame assignment (two-stage design)
# ------------------------------

def _frame_mode(enabled_frames: list[str]) -> str:
    """
    Map enabled frame keys to a mode:
        - "random50": both ALL-IN and PARTITIONED selected → 4/4 blocking
        - "allin":    only ALL-IN selected
        - "partitioned": only PARTITIONED selected
        - "default_allin": neither selected (legacy default)
    """
    fa = "frame_allin" in enabled_frames
    fp = "frame_partitioned" in enabled_frames
    if fa and fp:
        return "random50"
    if fa:
        return "allin"
    if fp:
        return "partitioned"
    return "default_allin"

def _assign_frame_for_card(i: int, set_id: str, seeds: Seeds, mode: str) -> tuple[int, int]:
    """
    Return (frame_allin, frame_partitioned) as 0/1, mutually exclusive.
    For "random50", use a deterministic shuffled pattern with exactly four 1s.
    """
    if mode in ("allin", "default_allin"):
        return 1, 0
    if mode == "partitioned":
        return 0, 1
    # random50 with per-screen blocking: exactly 4 ALL-IN (1) and 4 PARTITIONED (0)
    rng = seeds.rng("frame-block", set_id)
    pattern = [1] * 4 + [0] * 4
    rng.shuffle(pattern)  # deterministic given (set_id, seeds)
    v = pattern[i % 8]
    return (1, 0) if v == 1 else (0, 1)

def _balanced_badge_assignments(enabled_nonframes: list[str], seeds: Seeds, set_id: str) -> list[str]:
    """
    Deterministically allocate the 8 cards across the user-enabled non-frame badges
    PLUS a true 'none' baseline so that every screen has an identifiable reference.
    Split is as even as possible; remainder is distributed by a seeded shuffle.

    Examples
      K=1 badge  -> 4 with that badge, 4 with none
      K=2 badges -> 3,3,2 (two badges get 3 each, none gets 2)
      K=3 badges -> 2,2,2,2 (each badge 2, none 2)
      K>=4       -> floor(8/(K+1)) each, remainder spread via seeded order
    """
    # normalise and de-duplicate while preserving order
    cats = [c.strip().lower() for c in (enabled_nonframes or []) if c]
    cats = [("social" if c in ("social", "social_proof") else c) for c in cats]
    cats = list(dict.fromkeys(cats))

    # always include a real 'none' cell in the allocation
    cats_plus_none = cats + ["none"]
    K = len(cats_plus_none)
    if K <= 1:
        return ["none"] * 8

    base = 8 // K
    rem = 8 % K
    counts = {c: base for c in cats_plus_none}

    rng = seeds.rng("badge-alloc", set_id)
    order = list(cats_plus_none)
    rng.shuffle(order)
    for c in order[:rem]:
        counts[c] += 1

    assignments = []
    for c, k in counts.items():
        assignments.extend([c] * k)

    rng2 = seeds.rng("badge-assign", set_id)
    rng2.shuffle(assignments)
    return assignments

# ------------------------------
# Public API
# ------------------------------

def render_screen(
    category: str,
    set_id: str,
    badges: list[str],
    catalog_seed: int,
    price_anchor: float,
    currency: str,
    brand: str = "",
) -> str:
    """
    Render HTML for an 8-card screen and embed a hidden JSON payload with
    per-card ground truth for the estimator. All keys align with the logit API.

    Parameters
    ----------
    category : str
    set_id   : str
    badges   : list[str]   # human-readable selections from the UI
    catalog_seed : int
    price_anchor : float   # currency units (must be > 0 for ln(price))
    currency : str         # e.g., "£", "$"
    brand    : str
    """
    # Normalise the user's selected badges (lower-cased keys)
    sel = { (b or "").strip().lower(): True for b in (badges or []) }

    # Map UI labels to internal keys (accept both canonical and legacy spellings)
    enabled_frames: list[str] = []
    if sel.get("all-in v. partitioned pricing"):
        enabled_frames.extend(["frame_allin", "frame_partitioned"])
    if sel.get("all-in pricing"):        # legacy
        enabled_frames.append("frame_allin")
    if sel.get("partitioned pricing"):   # legacy
        enabled_frames.append("frame_partitioned")

    enabled_nonframes: list[str] = []
    if sel.get("assurance"):
        enabled_nonframes.append("assurance")
    if sel.get("scarcity tag") or sel.get("scarcity"):
        enabled_nonframes.append("scarcity")
    if sel.get("strike-through") or sel.get("strike"):
        enabled_nonframes.append("strike")
    if sel.get("timer"):
        enabled_nonframes.append("timer")
    if sel.get("social proof") or sel.get("social"):
        enabled_nonframes.append("social")  # internal alias; emits social_proof=1
    if sel.get("voucher"):
        enabled_nonframes.append("voucher")
    if sel.get("bundle"):
        enabled_nonframes.append("bundle")

    seeds = Seeds(catalog_seed=int(catalog_seed), brand=str(brand or ""), category=str(category or "product"))

    # Frame mode (blocked or fixed)
    frame_mode = _frame_mode(enabled_frames)

    # Prices: 8 log-symmetric levels via Latin schedule
    p0 = float(price_anchor or 0.0)
    p0 = max(p0, 0.01)  # guard ln(price)

    cards_html: list[str] = []
    gt_rows: list[dict] = []

    brand_text = (brand or "").strip()
    display_name = (f"{brand_text} {category}".strip()) or str(category)
    
    # build a deterministic per-screen assignment across selected badges + 'none'
    badge_plan = _balanced_badge_assignments(enabled_nonframes, seeds, set_id)
    assert len(badge_plan) == 8
    for i in range(8):
        # Grid coordinates: rows 0/1, cols 0..3
        r, c = (0 if i < 4 else 1), (i % 4)

        # Price for this card
        price_total = _price_for_card(p0, set_id, i, seeds)

        # Stage 1: assign pricing frame (mutually exclusive; blocked 4/4 if randomised)
        frame_allin, frame_partitioned = _assign_frame_for_card(i, set_id, seeds, frame_mode)
        is_partitioned = bool(frame_partitioned)
        
        # --- Stage 2: we now follow a deterministic balanced plan rather than a uniform random draw.
        chosen_nonframe = badge_plan[i]
            
        # Initialise badge flags (0/1)
        assurance = 0
        scarcity = 0
        strike = 0
        timer = 0
        social_proof = 0
        voucher = 0
        bundle = 0

        # Activate the selected non-frame flag
        if chosen_nonframe == "assurance":
            assurance = 1
        elif chosen_nonframe == "scarcity":
            scarcity = 1
        elif chosen_nonframe == "strike":
            strike = 1
        elif chosen_nonframe == "timer":
            timer = 1
        elif chosen_nonframe == "social":
            social_proof = 1
        elif chosen_nonframe == "voucher":
            voucher = 1
        elif chosen_nonframe == "bundle":
            bundle = 1

        # Render price depending on frame
        if is_partitioned:
            base, fees, ship, tax = _partition_total_into_components(price_total, seeds, set_id)
            price_block = f"<div class='price'>{_format_currency(base, currency)} + charges</div>"
            part_block = (
                f"<div class='pp'>+ Fees {_format_currency(fees, currency)} · "
                f"ship {_format_currency(ship, currency)} · tax {_format_currency(tax, currency)}"
                f"<br>Total {_format_currency(price_total, currency)}</div>"
            )
        else:
            price_block = f"<div class='price'>{_format_currency(price_total, currency)}</div>"
            part_block = ""

        # Visuals for badges (at most one non-frame visual shown)
        assur_block = ""
        dark_block = ""
        social_block = ""
        voucher_block = ""
        bundle_block = ""

        if assurance:
            assur_block = "<div class='badge'>Free returns · 30-day warranty</div>"

        if scarcity:
            scarcity_level = max(2, int(round(price_total % 7)) + 2)
            dark_block = f"<div class='pill warn'>Only {scarcity_level} left</div>"
        elif strike:
            strike_price = round(price_total * 1.20, 2)
            dark_block = f"<div class='pill'><s>{_format_currency(strike_price, currency)}</s></div>"
        elif timer:
            mm = 1 + ((i * 7) % 15)
            ss = 5 + ((i * 13) % 55)
            dark_block = f"<div class='pill warn'>Deal ends in {mm:02d}:{ss:02d}</div>"

        if social_proof:
            social_block = "<div class='chip'>👥 2k bought this month</div>"
        elif voucher:
            voucher_block = "<div class='chip good'>10% OFF · code SAVE10</div>"
        elif bundle:
            bundle_block = "<div class='chip info'>Buy 2, save 10%</div>"

        # Compose one card
        cards_html.append(
            f"<div class='card' style='grid-row:{r+1};grid-column:{c+1}'>"
            f"<div class='title'>{display_name}</div>"
            f"{price_block}{part_block}{assur_block}{dark_block}{social_block}{voucher_block}{bundle_block}"
            f"</div>"
        )

        # Ground-truth row (aligned with logit)
        gt_rows.append({
            "case_id": str(set_id),                 # logit renames case_id → screen_id
            "set_id": str(set_id),
            "category": str(category),
            "title": f"{display_name} #{i+1}",
            "row": r, "col": c,
            "row_top": 1 if r == 0 else 0,
            "col1": 1 if c == 0 else 0,
            "col2": 1 if c == 1 else 0,
            "col3": 1 if c == 2 else 0,
            "frame": 1 if frame_allin else 0,       # 1 = ALL-IN (matches estimator expectation)
            "assurance": assurance,
            "scarcity": scarcity,
            "strike": strike,
            "timer": timer,
            "social_proof": social_proof,
            "voucher": voucher,
            "bundle": bundle,
            "price": round(price_total, 2),
            "ln_price": math.log(max(price_total, 1e-8)),
            # Fields the runner/agent will add later:
            # "chosen": 0/1   (per agent choice)
            # "model": "..."  (optional)
            # "run_id": "..." (optional)
        })

    grid = "".join(cards_html)
    gt_json = json.dumps(gt_rows)

    # Minimal CSS; no JS required. Hidden div carries the ground truth JSON.
    html = """<!doctype html>
<html><head><meta charset='utf-8'>
<style>
body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; padding: 24px; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); grid-template-rows: repeat(2, 320px); gap: 16px; }
.card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.title { font-weight: 600; margin-bottom: 8px; }
.price { font-size: 18px; font-weight: 700; margin: 4px 0; }
.pp { color: #6b7280; font-size: 12px; }
.badge { display:inline-block; margin-top:8px; background:#eef2ff; color:#3730a3; padding:4px 8px; border-radius:9999px; font-size:12px; }
.pill { display:inline-block; margin-left:6px; background:#f3f4f6; color:#111827; padding:4px 8px; border-radius:9999px; font-size:12px; }
.pill.warn { background:#fff7ed; color:#9a3412; }
.chip { display:inline-block; margin-left:6px; background:#f1f5f9; color:#0f172a; padding:2px 8px; border-radius:9999px; font-size:12px; }
.chip.good { background:#ecfdf5; color:#065f46; }
.chip.info { background:#eff6ff; color:#1e3a8a; }
.footer { margin-top: 8px; color:#6b7280; font-size: 12px; }
</style>
</head>
<body>
<div class='grid'>
{grid}
</div>
<div id='groundtruth' style='display:none'>
{gt}
</div>
</body></html>"""

    # Use simple replacement to avoid str.format clashing with CSS braces
    return html.replace("{grid}", grid).replace("{gt}", gt_json)


# Convenience wrappers used by the runner

def build_storefront_from_payload(payload: dict) -> Tuple[str, dict]:
    """
    Build storefront HTML and a metadata JSON dict from a runner payload.
    """
    html = render_screen(
        category=str(payload.get("product") or "product"),
        set_id=str(payload.get("set_id") or "S0001"),
        badges=list(payload.get("badges") or []),
        catalog_seed=int(payload.get("catalog_seed", 777)),
        price_anchor=float(payload.get("price") or 0.0),
        currency=str(payload.get("currency") or "£"),
        brand=str(payload.get("brand") or ""),
    )
    meta = {
        "set_id": str(payload.get("set_id") or "S0001"),
        "category": str(payload.get("product") or "product"),
        "brand": str(payload.get("brand") or ""),
        "badges": list(payload.get("badges") or []),
        "catalog_seed": int(payload.get("catalog_seed", 777)),
        "price_anchor": float(payload.get("price") or 0.0),
        "currency": str(payload.get("currency") or "£"),
    }
    return html, meta


def save_storefront(job_id: str, html: str, meta: dict) -> tuple[str, str]:
    """
    Persist the rendered HTML and its metadata under runs/{job_id}.{html,json}.
    """
    runs_dir = pathlib.Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    html_path = runs_dir / f"{job_id}.html"
    meta_path = runs_dir / f"{job_id}.json"
    html_path.write_text(html, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(html_path), str(meta_path)
