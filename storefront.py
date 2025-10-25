# -*- coding: utf-8 -*-
"""
Agentix storefront (Railway) — v1.7
Date: 2025-10-25
# ---------------------------------------------------------------------------
# Reviewer notes & modelling assumptions (paste near the top of storefront.py)
#
# Badge assignment is per-card and uniform over the set of user-selected badges.
# Exactly one badge is drawn for each card from that set with equal probability
# (pricing frames count as badges too). If no badge is selected we default to
# an all-in pricing frame so that the screen remains coherent. With this design,
# badges other than the pricing frame are purely presentational and do not alter
# the numeric transaction price; they only set the ground-truth flags consumed by
# the downstream logit.
#
# Pricing uses the UI anchor as the geometric center and applies eight
# multiplicative levels that are symmetric in log-space, exp(Δ) where
# Δ ∈ {−0.3567, −0.2231, −0.1053, −0.0513, 0.0513, 0.1053, 0.2231, 0.3567},
# yielding roughly ×0.70 … ×1.43 around the anchor. Levels are scheduled across
# the 8 cards via a fixed 8×8 Latin square so that in each block every relative
# price appears exactly once in each card position. This removes position–price
# confounds. All randomness (badge draw and partitioning rates) is deterministic
# and reproducible, seeded by (catalog_seed, brand, category, set_id, card_idx).
#
# Partitioned pricing is a framing only. When the partitioned frame is drawn,
# the total card price is decomposed into base+fees+shipping+tax using rates
# sampled once per screen from the ranges tax≈9–11%, fees≈5–7%, shipping≈3–5%.
# Components are rounded to two decimals and any rounding residual is absorbed
# into the base component so that base+fees+ship+tax equals the displayed total.
# All-in pricing shows only the total. Currency formatting is symbol-first with
# two decimals and no locale-specific separators beyond the thousands comma.
#
# Visual discount/urgency/social elements do not change the numeric price.
# Strike-through displays a notional reference price set to 1.20× the card’s
# total. Voucher shows “10% OFF · code SAVE10” as a label only. Bundle displays
# “Buy 2, save 10%” as a label only. Scarcity prints “Only N left” with N
# derived deterministically from the price and card index; the timer text is a
# seeded pseudo-countdown; social proof is a fixed “2k bought this month”.
# These elements exist to isolate framing effects without contaminating price.
#
# Ground-truth output includes binary indicators for each visual (assurance,
# scarcity, strike, timer, social_proof, voucher, bundle) and a legacy “frame”
# column where 1=all-in and 0=partitioned. For modelling, ln_price is the log of
# the card’s displayed total (>=1e-8 guard). Adding a new badge requires three
# coordinated edits: include it in the enabled-badge mapping, render its visual,
# and emit its ground-truth flag.
# ---------------------------------------------------------------------------
"""

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
    Storefront v1.7 — Equal per-card chance across ALL selected badges.
    Exactly one badge is rendered per card, drawn uniformly from the
    selected badge set (deterministic given seeds). Prices still follow
    the Latin-square schedule anchored on the UI price.
    """

    # Normalize the user's selected badges (lower-cased keys)
    sel = { (b or "").strip().lower(): True for b in (badges or []) }

    # Map UI labels to internal badge keys
    # We keep both frame options as badges so they compete fairly.
    enabled: list[str] = []
    if sel.get("all-in pricing"):       enabled.append("frame_allin")
    if sel.get("partitioned pricing"):  enabled.append("frame_partitioned")
    if sel.get("assurance"):            enabled.append("assurance")
    if sel.get("scarcity tag"):         enabled.append("scarcity")
    if sel.get("strike-through"):       enabled.append("strike")
    if sel.get("timer"):                enabled.append("timer")
    if sel.get("social"):               enabled.append("social")
    if sel.get("voucher"):              enabled.append("voucher")
    if sel.get("bundle"):               enabled.append("bundle")

    seeds = Seeds(catalog_seed=int(catalog_seed), brand=str(brand or ""), category=str(category or "product"))

    # If nothing is selected, we still show all-in pricing as a harmless default
    if not enabled:
        enabled = ["frame_allin"]

    # Deterministic per-card uniform draw over enabled badges
    def draw_badge_for_card(i: int) -> str:
        rng = seeds.rng("card-badge", set_id, i)
        return enabled[rng.randrange(len(enabled))]

    # Prices: 8 log-symmetric levels via Latin schedule
    p0 = float(price_anchor or 0.0)

    cards_html: list[str] = []
    gt_rows: list[dict] = []

    brand_text = (brand or "").strip()
    display_name = (f"{brand_text} {category}".strip()) or str(category)

    for i in range(8):
        r, c = (0 if i < 4 else 1), (i % 4)
        price_total = _price_for_card(p0, set_id, i, seeds)

        # Initialize flags
        frame_allin = 0
        frame_partitioned = 0
        assurance = 0
        scarcity = strike = timer = 0
        social_proof = voucher = bundle = 0

        chosen = draw_badge_for_card(i)

        # Render blocks based on the chosen badge
        assur_block = ""
        dark_block = ""
        social_block = voucher_block = bundle_block = ""

        # Frame (also affects how price is displayed)
        if chosen == "frame_allin":
            frame_allin = 1
        elif chosen == "frame_partitioned":
            frame_partitioned = 1
        # Assurance
        elif chosen == "assurance":
            assurance = 1
        # Dark variants
        elif chosen == "scarcity":
            scarcity = 1
        elif chosen == "strike":
            strike = 1
        elif chosen == "timer":
            timer = 1
        # Social family
        elif chosen == "social":
            social_proof = 1
        elif chosen == "voucher":
            voucher = 1
        elif chosen == "bundle":
            bundle = 1

        # Visual rendering for price (depends on frame badge if drawn)
        if frame_partitioned:
            base, fees, ship, tax = _partition_total_into_components(price_total, seeds, set_id)
            price_block = f"<div class='price'>{_format_currency(base, currency)} + charges</div>"
            part_block = (
                f"<div class='pp'>+ Fees {_format_currency(fees, currency)} · "
                f"ship {_format_currency(ship, currency)} · tax {_format_currency(tax, currency)}"
                f"<br>Total {_format_currency(price_total, currency)}</div>"
            )
        else:
            # Default and 'frame_allin' use all-in price
            frame_allin = 1 if chosen == "frame_allin" else frame_allin
            price_block = f"<div class='price'>{_format_currency(price_total, currency)}</div>"
            part_block = ""

        # Visuals for badges (only one is shown per card)
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

        cards_html.append(
            f"<div class='card' style='grid-row:{r+1};grid-column:{c+1}'>"
            f"<div class='title'>{display_name}</div>"
            f"{price_block}{part_block}{assur_block}{dark_block}{social_block}{voucher_block}{bundle_block}"
            f"</div>"
        )

        gt_rows.append({
            "title": f"{display_name} #{i+1}",
            "row": r, "col": c,
            "row_top": 1 if r == 0 else 0,
            "col1": 1 if c == 0 else 0,
            "col2": 1 if c == 1 else 0,
            "col3": 1 if c == 2 else 0,
            "frame": 1 if frame_allin else 0,          # 1 = all-in, 0 = partitioned (for backwards compat)
            "assurance": 1 if assurance else 0,
            "scarcity": 1 if scarcity else 0,
            "strike":   1 if strike else 0,
            "timer":    1 if timer else 0,
            "social_proof": 1 if social_proof else 0,
            "voucher":  1 if voucher else 0,
            "bundle":   1 if bundle else 0,
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
