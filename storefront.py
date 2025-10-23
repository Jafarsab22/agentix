# -*- coding: utf-8 -*-
"""
Agentix storefront (Railway) — v1.6
Date: 2025-10-23

Assumptions & design rationale (for methods and code review)
1) Balanced lever exposure per screen (internal validity):
   - Frame (all-in vs partitioned) and Assurance use orthogonal 4/8 masks with a 4-screen rotation.
   - The chosen dark cue (exactly one of {scarcity, strike, timer, none} per screen) uses the same balanced mask.
   - Additional independent cues (e.g., social, voucher, bundle) also use the balanced masks when enabled.

2) Seed‑randomised, blocked rotation (reproducibility + randomness):
   - Within every block of 4 consecutive screens, a seed-determined permutation of the dark types
     ensures each type appears exactly once per block while the order is randomised by seed.

3) Price variation anchored on user input (external validity + identification):
   - The user enters an anchor price P₀. We generate eight within-screen price levels
     by multiplying P₀ by log‑symmetric factors spanning approximately ±30% around P₀.
   - Assignment uses an 8×8 Latin-square schedule across 8‑screen blocks: each card position receives
     each price level exactly once per block. This keeps ln(price) orthogonal to position and to the other levers.
   - We include ln(price) in the ground truth so the conditional logit can identify price sensitivity
     and purge badge effects of any residual correlation.
   - For partitioned pricing, base+fees+shipping+taxes always sums exactly to the varied total Pᵢ to keep
     the frame a pure presentation manipulation.

4) Rounding & currency:
   - Prices are rounded deterministically to the smallest currency unit (e.g., £0.01) using the screen seed.

5) Reproducibility keys:
   - All randomisation is keyed by: catalog_seed, brand, category, and screen index (set_id),
     with clear helper functions below. Given the same inputs, the storefront is exactly reproducible.

This module is intended to be imported by agent_runner.py; the public API is render_screen(...),
plus small helpers to build/save payloads.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import json, math, pathlib, random

# ------------------------------
# Constants & basic helpers
# ------------------------------
_DARK_TYPES = ["scarcity","strike","timer","none"]

# Log-symmetric 8-level price design spanning about ±30% around the anchor.
# Levels are set by equally spaced steps in log-space for approximate symmetry.
_LOG_DELTAS = (-0.3567, -0.2231, -0.1053, -0.0513, 0.0513, 0.1053, 0.2231, 0.3567)
_PRICE_MULTIPLIERS = tuple(math.exp(x) for x in _LOG_DELTAS)  # ~ [0.70,0.80,0.90,0.95,1.05,1.11,1.25,1.43]

@dataclass(frozen=True)
class Seeds:
    catalog_seed: int
    brand: str
    category: str

    def rng(self, *components: int | str) -> random.Random:
        base = int(self.catalog_seed) & 0x7FFFFFFF
        mix = base ^ (abs(hash((self.brand, self.category))) & 0x7FFFFFFF)
        for c in components:
            mix ^= abs(hash(c)) & 0x7FFFFFFF
        return random.Random(mix)

# ------------------------------
# Balanced 4/8 masks with 4-screen rotation
# ------------------------------
_PATTERNS = [
    [1,1,1,1,0,0,0,0],
    [1,1,0,0,1,1,0,0],
    [1,0,1,0,1,0,1,0],
    [1,0,0,1,0,1,0,1],
]


def _layout_index(set_id: str) -> int:
    # 0..3 depending on screen number; you likely already have something like this
    try:
        n = int(str(set_id).strip().lstrip("S"))
    except Exception:
        n = 1
    return (n - 1) % 4

# GUARANTEED distinct offsets for the main levers
_LEVER_OFFSET = {
    "frame": 0,
    "assurance": 1,
    "dark": 2,
    "social": 3,
    "voucher": 1,   # secondary group; can reuse if not estimated together
    "bundle": 2,
}

def _balanced_mask(set_id: str) -> list[int]:
    return _PATTERNS[_layout_index(set_id)][:]

# ------------------------------
# Seed-randomised, blocked rotation for dark types (4-screen blocks)
# ------------------------------

def _dark_type_for_screen(set_id: str, seeds: Seeds) -> str:
    idx = _screen_index(set_id)
    block, within = (idx - 1) // 4, (idx - 1) % 4
    rng = seeds.rng("dark", block)
    order = list(_DARK_TYPES)
    rng.shuffle(order)
    return order[within]

# ------------------------------
# Latin-square schedule for price levels across 8-screen blocks
# ------------------------------

def _screen_index(set_id: str) -> int:
    try:
        return int("".join(ch for ch in set_id if ch.isdigit()))
    except Exception:
        return 1

# Precompute a canonical 8×8 Latin square L where L[r][c] = (c + r) % 8
_LATIN_8 = [[(c + r) % 8 for c in range(8)] for r in range(8)]


def _price_levels_for_block(anchor: float, seeds: Seeds, block_id: int) -> List[float]:
    """Return the 8 price levels for this block as a seed-shuffled mapping of the 8 multipliers."""
    multipliers = list(_PRICE_MULTIPLIERS)
    rng = seeds.rng("price-levels", block_id)
    rng.shuffle(multipliers)
    return [round(anchor * m, 2) for m in multipliers]


def _price_for_card(anchor: float, set_id: str, card_index: int, seeds: Seeds) -> float:
    """
    Latin-square assignment across 8-screen blocks:
    - Let b = floor((idx-1)/8) and t = (idx-1) % 8 be the block and the screen-within-block indices.
    - Let L be the fixed 8×8 Latin square. Card position p∈{0..7} on screen t takes the level L[t][p].
    - Map that level through a seed-shuffled list of the 8 log-symmetric multipliers for block b.
    """
    idx = _screen_index(set_id)
    block, within = (idx - 1) // 8, (idx - 1) % 8
    price_levels = _price_levels_for_block(anchor, seeds, block)
    level_index = _LATIN_8[within][card_index]
    return price_levels[level_index]

# ------------------------------
# Currency helpers
# ------------------------------

def _format_currency(x: float, currency: str) -> str:
    return f"{currency}{x:,.2f}"


def _partition_total_into_components(total: float, seeds: Seeds, set_id: str) -> tuple[float, float, float, float]:
    """
    Partition total into base + fees + shipping + tax such that the sum equals total exactly.
    Ratios are deterministic per screen, varied slightly by seed; tax is anchored around 10%.
    """
    rng = seeds.rng("partition", set_id)
    tax_rate = rng.uniform(0.09, 0.11)
    fees_rate = rng.uniform(0.05, 0.07)
    ship_rate = rng.uniform(0.03, 0.05)
    # Compute base as residual to hit exact total
    base = total / (1 + tax_rate + fees_rate + ship_rate)
    fees = base * fees_rate
    ship = base * ship_rate
    tax = base * tax_rate
    # Round to cents and fix residual on base
    base_r = round(base, 2); fees_r = round(fees, 2); ship_r = round(ship, 2); tax_r = round(tax, 2)
    residual = round(total - (base_r + fees_r + ship_r + tax_r), 2)
    base_r = round(base_r + residual, 2)
    return base_r, fees_r, ship_r, tax_r

# ------------------------------
# Ground-truth row builder for each card
# ------------------------------

def _gt_row(i: int, set_id: str, masks: dict, dark_type: str, price_total: float) -> dict:
    row = 0 if i < 4 else 1
    col = i % 4
    return {
        "title": f"Card {i+1}",
        "row": row, "col": col,
        "row_top": 1 if row == 0 else 0,
        "col1": 1 if col == 0 else 0, "col2": 1 if col == 1 else 0, "col3": 1 if col == 2 else 0,
        "frame": masks["frame"][i],
        "assurance": masks["assurance"][i],
        "scarcity": 1 if (dark_type == "scarcity" and masks["dark"][i]) else 0,
        "strike":   1 if (dark_type == "strike"   and masks["dark"][i]) else 0,
        "timer":    1 if (dark_type == "timer"    and masks["dark"][i]) else 0,
        "price": round(price_total, 2),
        "ln_price": math.log(max(price_total, 1e-8)),
    }

# ------------------------------
# Public API: render_screen + helpers
# ------------------------------

_LEVER_KEYS = ["frame","assurance","social","voucher","bundle","dark"]

def _lever_mask(set_id: str, lever_key: str) -> list[int]:
    """
    Return a balanced 4/8 mask for this lever that is orthogonal to others.
    We deterministically rotate the base layout by a lever-specific offset.
    """
    li  = _layout_index(set_id)              # 0..3 rotation by screen index
    off = _LEVER_OFFSET.get(lever_key, 0)    # guaranteed mapping
    return _PATTERNS[(li + off) % 4][:]

_LEVER_KEYS = ["frame","assurance","social","voucher","bundle","dark"]

def _lever_mask(set_id: str, lever_key: str) -> list[int]:
    """
    Return a balanced 4/8 mask for this lever that is orthogonal to others.
    We rotate the base pattern by a lever-specific offset, deterministically.
    """
    li = _layout_index(set_id)      # 0..3 rotation by screen
    off = abs(hash(lever_key)) % 4  # lever-specific offset 0..3
    pat_idx = (li + off) % 4
    return _PATTERNS[pat_idx][:]

def render_screen(
    category: str,
    set_id: str,
    badges: List[str],
    catalog_seed: int,
    price_anchor: float,
    currency: str,
    brand: str = "",
) -> str:
    seeds = Seeds(catalog_seed=catalog_seed, brand=(brand or ""), category=(category or ""))

    # Normalise selected badges
    sel = {b.lower().strip(): True for b in (badges or [])}

    # Balanced masks
    base_mask = _balanced_mask(set_id)
    masks = {
    "frame":     _lever_mask(set_id, "frame")     if sel.get("all-in pricing", False) else [0]*8,
    "assurance": _lever_mask(set_id, "assurance") if sel.get("assurance", False) else [0]*8,
    "dark":      _lever_mask(set_id, "dark"),   # applied only to whichever dark type is active
    "social":    _lever_mask(set_id, "social")    if sel.get("social", False) else [0]*8,
    "voucher":   _lever_mask(set_id, "voucher")   if sel.get("voucher", False) else [0]*8,
    "bundle":    _lever_mask(set_id, "bundle")    if sel.get("bundle", False) else [0]*8,
}
if masks["frame"] == masks["assurance"]:
    raise RuntimeError("Design error: frame and assurance masks are identical; check _LEVER_OFFSET.")


    # Choose exactly one dark mechanism per screen (blocked & seed-randomised)
    dark_candidates = []
    if sel.get("scarcity tag", False):   dark_candidates.append("scarcity")
    if sel.get("strike-through", False): dark_candidates.append("strike")
    if sel.get("timer", False):          dark_candidates.append("timer")
    dark_type = _dark_type_for_screen(set_id, seeds) if dark_candidates else "none"
    if dark_type not in dark_candidates and dark_candidates:
        # If seeded pick not available, fall back deterministically
        dark_type = sorted(dark_candidates)[0]

    # Compute price for each of the 8 cards via Latin-square assignment
    prices = [_price_for_card(price_anchor, set_id, i, seeds) for i in range(8)]

    # Precompute UI strings and ground truth rows
    brand_text = (brand or "").strip()
    display_name = f"{brand_text} {category}".strip()

    cards_html = []
    gt_rows = []

    for i in range(8):
        r, c = (0 if i < 4 else 1), (i % 4)
        f = masks["frame"][i]
        a = masks["assurance"][i]
        d = masks["dark"][i]
        price_total = prices[i]

        # Price presentation
        if f == 1:  # all-in frame
            price_block = f"<div class='price'>{_format_currency(price_total, currency)}</div>"
            part_block = ""
        else:       # partitioned frame (sum equals price_total)
            base, fees, ship, tax = _partition_total_into_components(price_total, seeds, set_id)
            price_block = f"<div class='price'>{_format_currency(base, currency)} + charges</div>"
            part_block = (
                f"<div class='pp'>+ Fees {_format_currency(fees, currency)} · ship {_format_currency(ship, currency)} · "
                f"tax {_format_currency(tax, currency)}<br>Total {_format_currency(price_total, currency)}</div>"
            )

        assur_block = "<div class='badge'>Free returns · 30-day warranty</div>" if a else ""

        if dark_type == "scarcity" and d:
            scarcity_level = max(2, int(round(price_total % 7)) + 2)
            dark_block = f"<div class='pill warn'>Only {scarcity_level} left</div>"
        elif dark_type == "strike" and d:
            strike_mult = 1.10 + (0.20 * ((i*37) % 100) / 100.0)
            strike_price = round(price_total * strike_mult, 2)
            dark_block = f"<div class='pill'><s>{_format_currency(strike_price, currency)}</s></div>"
        elif dark_type == "timer" and d:
            mm = 1 + ((i*7) % 15); ss = 5 + ((i*13) % 55)
            dark_block = f"<div class='pill warn'>Deal ends in {mm:02d}:{ss:02d}</div>"
        else:
            dark_block = ""

        social_block  = "<div class='chip'>👥 2k bought this month</div>" if masks["social"][i] else ""
        voucher_block = "<div class='chip good'>10% OFF · code SAVE10</div>" if masks["voucher"][i] else ""
        bundle_block  = "<div class='chip info'>Buy 2, save 10%</div>" if masks["bundle"][i] else ""

        cards_html.append(
            f"<div class='card' style='grid-row:{r+1};grid-column:{c+1}'>"
            f"<div class='title'>{display_name}</div>"
            f"{price_block}{part_block}"
            f"{assur_block}{dark_block}{social_block}{voucher_block}{bundle_block}"
            f"</div>"
        )

        gt_rows.append(_gt_row(i, set_id, masks, dark_type, price_total))

    grid = "".join(cards_html)

    # NOTE: not an f-string — we’ll .format(grid=..., gt=...) below
     html = """<!doctype html>
<html><head><meta charset='utf-8'>
<style>
body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; padding: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(4, 1fr); grid-template-rows: repeat(2, 320px); gap: 16px; }}
.card {{ border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
.title {{ font-weight: 600; margin-bottom: 8px; }}
.price {{ font-size: 18px; font-weight: 700; margin: 4px 0; }}
.pp {{ color: #6b7280; font-size: 12px; }}
.badge {{ display:inline-block; margin-top:8px; background:#eef2ff; color:#3730a3; padding:4px 8px; border-radius:9999px; font-size:12px; }}
.pill {{ display:inline-block; margin-left:6px; background:#f3f4f6; color:#111827; padding:4px 8px; border-radius:9999px; font-size:12px; }}
.pill.warn {{ background:#fff7ed; color:#9a3412; }}
.chip {{ display:inline-block; margin-left:6px; background:#f1f5f9; color:#0f172a; padding:2px 8px; border-radius:9999px; font-size:12px; }}
.chip.good {{ background:#ecfdf5; color:#065f46; }}
.chip.info {{ background:#eff6ff; color:#1e3a8a; }}
.footer {{ margin-top: 8px; color:#6b7280; font-size: 12px; }}
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
    gt_json = json.dumps(gt_rows)
    return html.format(grid=grid, gt=gt_json)


# Convenience wrappers used by the runner

def build_storefront_from_payload(payload: dict) -> Tuple[str, dict]:
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
    runs_dir = pathlib.Path("runs"); runs_dir.mkdir(parents=True, exist_ok=True)
    html_path = runs_dir / f"{job_id}.html"
    meta_path = runs_dir / f"{job_id}.json"
    html_path.write_text(html, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(html_path), str(meta_path)

