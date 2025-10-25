# -*- coding: utf-8 -*-
"""
Agentix storefront (Railway) — v1.7
Date: 2025-10-25
# ---------------------------------------------------------------------------
# Reviewer notes & modelling assumptions (paste near the top of storefront.py)
#
# Experimental design (two-stage, per-card):
# 1) Pricing frame is an independent factor. If both “All-in pricing” and
#    “Partitioned pricing” are selected, each card is assigned a frame with
#    50/50 probability, blocked within screen to guarantee exactly 4 all-in
#    and 4 partitioned cards. If only one frame is selected (or neither),
#    the frame is fixed (defaulting to all-in when neither is selected), and
#    no frame effect is estimable in that run.
# 2) Visual badges are assigned independently of the frame. Exactly one
#    non-frame badge (assurance, scarcity, strike-through, timer, social,
#    voucher, bundle) is drawn uniformly at random for each card from the
#    subset selected by the user. If no non-frame badges are selected, no
#    visual badge is shown.
#
# Identification and reporting:
# This orthogonal design yields clean main effects for the pricing frame and
# for each non-frame badge; pre-specified frame×badge interactions can be
# estimated if desired. The ground truth includes a legacy “frame” column
# (1 = all-in, 0 = partitioned) present on every row, plus one-hot indicators
# for the non-frame badges (at most one equals 1 on any card). The effects
# table and CSV should report the pricing-frame coefficient explicitly
# (“All-in vs Partitioned: β for all-in”), alongside the non-frame badges.
#
# Pricing schedule:
# Prices use the UI anchor as geometric centre and apply eight multiplicative
# levels symmetric in log-space, exp(Δ) where Δ ∈ {−0.3567, −0.2231, −0.1053,
# −0.0513, 0.0513, 0.1053, 0.2231, 0.3567}, yielding approximately ×0.70…×1.43
# around the anchor. Levels are placed via a fixed 8×8 Latin square so that
# within each screen every relative price appears exactly once at each card
# position, removing position–price confounds. All randomness (frame draw,
# badge draw, and partitioning rates) is deterministic and reproducible,
# seeded by (catalog_seed, brand, category, set_id, card_idx).
#
# Frame rendering and partitioning:
# The frame is presentational; it does not change the card’s total price.
# In the partitioned frame, the total is decomposed into base+fees+shipping+tax
# using rates sampled once per screen from tax≈9–11%, fees≈5–7%, shipping≈3–5%.
# Components are rounded to two decimals; any rounding residual is absorbed
# into the base so components sum exactly to the displayed total. All-in shows
# the total only. Currency formatting is symbol-first with two decimals.
#
# Other visuals:
# Visual discount/urgency/social elements do not alter the numeric total.
# Strike-through shows a notional reference price set to 1.20× the card total.
# Voucher displays “10% OFF · code SAVE10”. Bundle displays “Buy 2, save 10%”.
# Scarcity prints “Only N left” with N derived deterministically; the timer
# uses a seeded pseudo-countdown; social proof is a fixed “2k bought this month”.
#
# Implementation notes:
# Adding a new non-frame badge requires three coordinated edits: include it in
# the enabled-badge mapping, render its visual block, and emit its ground-truth
# flag. If you change the frame policy (e.g., different randomisation ratio or
# blocking scheme), update both the assignment function and the reviewer notes.
# ---------------------------------------------------------------------------

"""

# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import json, math, pathlib, random

# ------------------------------
# Constants & helpers
# ------------------------------
_LOG_DELTAS = (-0.3567, -0.2231, -0.1053, -0.0513, 0.0513, 0.1053, 0.2231, 0.3567)
_PRICE_MULTIPLIERS = tuple(math.exp(x) for x in _LOG_DELTAS)  # ~ [0.70 ... 1.43]

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

def _screen_index(set_id: str) -> int:
    try:
        return int("".join(ch for ch in set_id if ch.isdigit()))
    except Exception:
        return 1

# 8×8 Latin square L where L[r][c] = (c + r) % 8
_LATIN_8 = [[(c + r) % 8 for c in range(8)] for r in range(8)]

def _price_levels_for_block(anchor: float, seeds: Seeds, block_id: int) -> List[float]:
    multipliers = list(_PRICE_MULTIPLIERS)
    rng = seeds.rng("price-levels", block_id)
    rng.shuffle(multipliers)
    return [round(anchor * m, 2) for m in multipliers]

def _price_for_card(anchor: float, set_id: str, card_index: int, seeds: Seeds) -> float:
    idx = _screen_index(set_id)
    block, within = (idx - 1) // 8, (idx - 1) % 8
    price_levels = _price_levels_for_block(anchor, seeds, block)
    level_index = _LATIN_8[within][card_index]
    return price_levels[level_index]

def _format_currency(x: float, currency: str) -> str:
    return f"{currency}{x:,.2f}"

def _partition_total_into_components(total: float, seeds: Seeds, set_id: str) -> tuple[float, float, float, float]:
    rng = seeds.rng("partition", set_id)
    tax_rate = rng.uniform(0.09, 0.11)
    fees_rate = rng.uniform(0.05, 0.07)
    ship_rate = rng.uniform(0.03, 0.05)
    base = total / (1 + tax_rate + fees_rate + ship_rate)
    fees = base * fees_rate
    ship = base * ship_rate
    tax = base * tax_rate
    base_r = round(base, 2); fees_r = round(fees, 2); ship_r = round(ship, 2); tax_r = round(tax, 2)
    residual = round(total - (base_r + fees_r + ship_r + tax_r), 2)
    base_r = round(base_r + residual, 2)
    return base_r, fees_r, ship_r, tax_r

# ------------------------------
# Frame assignment (two-stage design)
# ------------------------------

def _frame_mode(enabled_frames: list[str]) -> str:
    # enabled_frames contains any of: "frame_allin", "frame_partitioned"
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
    # Returns (frame_allin, frame_partitioned) as 0/1, mutually exclusive.
    if mode in ("allin", "default_allin"):
        return 1, 0
    if mode == "partitioned":
        return 0, 1
    # random50 with per-screen blocking: exactly 4 all-in, 4 partitioned per screen
    rng = seeds.rng("frame-block", set_id)
    pattern = [1]*4 + [0]*4  # 1=all-in, 0=partitioned
    rng.shuffle(pattern)
    v = pattern[i % 8]
    return (1, 0) if v == 1 else (0, 1)

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
    # Normalise the user's selected badges (lower-cased keys)
    sel = { (b or "").strip().lower(): True for b in (badges or []) }
    # Map UI labels to internal keys
    enabled_frames: list[str] = []
    if sel.get("all-in v. partitioned pricing"):
        enabled_frames.extend(["frame_allin", "frame_partitioned"])

    # Legacy fallback (keep if old labels might appear)
    if sel.get("all-in pricing"):
        enabled_frames.append("frame_allin")
    if sel.get("partitioned pricing"):
        enabled_frames.append("frame_partitioned")

    enabled_nonframes: list[str] = []
    if sel.get("assurance"):            enabled_nonframes.append("assurance")
    if sel.get("scarcity tag"):         enabled_nonframes.append("scarcity")
    if sel.get("strike-through"):       enabled_nonframes.append("strike")
    if sel.get("timer"):                enabled_nonframes.append("timer")
    if sel.get("social"):               enabled_nonframes.append("social")
    if sel.get("voucher"):              enabled_nonframes.append("voucher")
    if sel.get("bundle"):               enabled_nonframes.append("bundle")
    seeds = Seeds(catalog_seed=int(catalog_seed), brand=str(brand or ""), category=str(category or "product"))

    # --- Debug guards (remove after verifying) ---
    assert all(isinstance(x, str) for x in enabled_frames), f"enabled_frames malformed: {enabled_frames}"
    # quick visibility while debugging:
    print("[storefront] enabled_frames:", enabled_frames)
    # If no frame explicitly selected, default to all-in (coherent screen)
    frame_mode = _frame_mode(enabled_frames)

    # Prices: 8 log-symmetric levels via Latin schedule
    p0 = float(price_anchor or 0.0)

    cards_html: list[str] = []
    gt_rows: list[dict] = []

    brand_text = (brand or "").strip()
    display_name = (f"{brand_text} {category}".strip()) or str(category)

    for i in range(8):
        r, c = (0 if i < 4 else 1), (i % 4)
        price_total = _price_for_card(p0, set_id, i, seeds)

        # --- Stage 1: assign pricing frame (mutually exclusive; blocked 4/4 if randomised) ---
        frame_allin, frame_partitioned = _assign_frame_for_card(i, set_id, seeds, frame_mode)
        is_partitioned = bool(frame_partitioned)

        # --- Stage 2: pick exactly one non-frame badge uniformly (if any were selected) ---
        chosen_nonframe = None
        if enabled_nonframes:
            rng = seeds.rng("nonframe", set_id, i)
            chosen_nonframe = enabled_nonframes[rng.randrange(len(enabled_nonframes))]

        # Initialise flags
        assurance = 0
        scarcity = 0
        strike = 0
        timer = 0
        social_proof = 0
        voucher = 0
        bundle = 0

        # Activate the chosen non-frame flag
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
            mm = 1 + ((i*7) % 15); ss = 5 + ((i*13) % 55)
            dark_block = f"<div class='pill warn'>Deal ends in {mm:02d}:{ss:02d}</div>"

        if social_proof:
            social_block  = "<div class='chip'>👥 2k bought this month</div>"
        elif voucher:
            voucher_block = "<div class='chip good'>10% OFF · code SAVE10</div>"
        elif bundle:
            bundle_block  = "<div class='chip info'>Buy 2, save 10%</div>"

        # Compose one card
        cards_html.append(
            f"<div class='card' style='grid-row:{r+1};grid-column:{c+1}'>"
            f"<div class='title'>{display_name}</div>"
            f"{price_block}{part_block}{assur_block}{dark_block}{social_block}{voucher_block}{bundle_block}"
            f"</div>"
        )

        # Ground-truth row
        gt_rows.append({
            "title": f"{display_name} #{i+1}",
            "row": r, "col": c,
            "row_top": 1 if r == 0 else 0,
            "col1": 1 if c == 0 else 0,
            "col2": 1 if c == 1 else 0,
            "col3": 1 if c == 2 else 0,
            "frame": 1 if frame_allin else 0,  # legacy meaning: 1 = all-in, 0 = partitioned
            "assurance": assurance,
            "scarcity":  scarcity,
            "strike":    strike,
            "timer":     timer,
            "social_proof": social_proof,
            "voucher":   voucher,
            "bundle":    bundle,
            "price": round(price_total, 2),
            "ln_price": math.log(max(price_total, 1e-8)),
        })

    grid = "".join(cards_html)
    gt_json = json.dumps(gt_rows)

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

