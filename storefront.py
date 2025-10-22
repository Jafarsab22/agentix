# -*- coding: utf-8 -*-
"""
Storefront HTML generator for Agentix.
- Visual encoding of badges (frame, assurance, dark: scarcity/strike/timer, social, voucher, bundle)
- Deterministic 4-page rotation (balanced 4/8 masks per lever)
- Hidden #groundtruth JSON with schema expected by agent_runner

Public API:
    build_storefront_from_payload(payload) -> (html:str, meta:dict)
    render_screen(category, set_id, badges, catalog_seed, price, currency) -> html
    save_storefront(job_id, html, meta) -> (html_path:str, meta_path:str)
"""
from __future__ import annotations
import json, pathlib, random
from typing import List, Tuple

def _rotation_index(set_id: str) -> int:
    try:
        idx = int(set_id[1:])
    except Exception:
        idx = 1
    return (idx - 1) % 4

_PATTERNS = [
    [1,0,1,0, 1,0,1,0],  # L0
    [0,1,0,1, 0,1,0,1],  # L1
    [1,1,0,0, 0,0,1,1],  # L2
    [0,0,1,1, 1,1,0,0],  # L3
]

def _mask(layout: int) -> list[int]:
    return _PATTERNS[layout][:]

def _row_col(k: int) -> tuple[int,int]:
    return (0 if k < 4 else 1, k % 4)

def render_screen(category: str, set_id: str, badges: List[str], catalog_seed: int,
                  price: float, currency: str, brand: str | None = None) -> str:
    """
    Visual storefront renderer with 4-page rotation and balanced masks.
    Displays BRAND + PRODUCT as the visible card title, while keeping internal
    product titles distinct (Card 1..8) for modelling with C(title).
    """
    import json as _json, random as _random

    sel = {b.lower(): True for b in (badges or [])}
    layout = _rotation_index(set_id)
    brand_text = (brand or "").strip()
    display_name = f"{brand_text} {category}".strip()  # e.g., "Samsung TV" or "TV" if no brand

    # Balanced 4/8 masks (rotated) for independent levers
    frame_mask   = _mask(layout) if sel.get("all-in pricing", False) else [0]*8
    assur_mask   = _mask(layout) if sel.get("assurance", False) else [0]*8
    social_mask  = _mask(layout) if sel.get("social", False) else [0]*8
    voucher_mask = _mask(layout) if sel.get("voucher", False) else [0]*8
    bundle_mask  = _mask(layout) if sel.get("bundle", False) else [0]*8

    # One dark mechanism per screen (round-robin over selected)
    dark_list = []
    if sel.get("scarcity tag", False):   dark_list.append("scarcity")
    if sel.get("strike-through", False): dark_list.append("strike")
    if sel.get("timer", False):          dark_list.append("timer")
    if dark_list:
        # distribute deterministically by page index
        try:
            idx = int(set_id[1:])
        except Exception:
            idx = 1
        dark_type = dark_list[(idx - 1) % len(dark_list)]
        dark_mask = _mask(layout)
    else:
        dark_type, dark_mask = "none", [0]*8

    # Deterministic RNG for small details
    seed = (int(catalog_seed) & 0x7FFFFFFF) ^ (abs(hash(set_id)) & 0x7FFFFFFF)
    rng = _random.Random(seed)

    total = float(price)
    fees = round(total * 0.06, 2)
    shipping = round(total * 0.04, 2)
    taxes = round(total * 0.05, 2)
    base_price = round(total - (fees + shipping + taxes), 2)
    strike_price = round(total * 1.15, 2)
    scarcity_level = rng.choice([2,3,4,5])

    products, cards = [], []
    for k in range(8):
        r, c = _row_col(k)
        f = int(frame_mask[k])
        a = int(assur_mask[k])
        soc = bool(social_mask[k])
        vou = bool(voucher_mask[k])
        bun = bool(bundle_mask[k])
        dark = dark_type if dark_mask[k] == 1 else "none"

        # Internal unique title for modelling (kept as Card N)
        title = f"Card {k+1}"
        products.append({
            "title": title, "row": r, "col": c,
            "frame": f, "assurance": a, "dark": dark,
            "social": soc, "voucher": vou, "bundle": bun,
            "total_price": total,
            "base_price": base_price, "fees": fees, "shipping": shipping, "taxes": taxes,
            "strike_price": strike_price, "scarcity_level": scarcity_level,
        })

        # Visible price blocks
        if f == 1:
            price_block = f"<div class='price'>{currency}{total:,.2f} <span class='mut'>(all-in)</span></div>"
            part_block = ""
        else:
            price_block = f"<div class='price'>{currency}{base_price:,.2f} + charges</div>"
            part_block = (
                f"<div class='pp'>+ Fees {currency}{fees:,.2f} · ship {currency}{shipping:,.2f} · "
                f"tax {currency}{taxes:,.2f}<br>Total {currency}{total:,.2f}</div>"
            )

        assur_block = "<div class='badge'>Free returns · 30-day warranty</div>" if a else ""

        if dark == "scarcity":
            dark_block = f"<div class='pill warn'>Only {scarcity_level} left</div>"
        elif dark == "strike":
            dark_block = f"<div class='pill'><s>{currency}{strike_price:,.2f}</s></div>"
        elif dark == "timer":
            mm = rng.choice([1,2,3,4,5,6,7,8,9,10,15]); ss = rng.choice([5,10,20,30,40,50])
            dark_block = f"<div class='pill warn'>Deal ends in {mm:02d}:{ss:02d}</div>"
        else:
            dark_block = ""

        social_block  = "<div class='chip'>👥 2k bought this month</div>" if soc else ""
        voucher_block = "<div class='chip good'>10% OFF · code SAVE10</div>" if vou else ""
        bundle_block  = "<div class='chip info'>Buy 2, save 10%</div>" if bun else ""

        # Visible title uses brand + product; price sits below it
        cards.append(
            f"<div class='card' style='grid-row:{r+1};grid-column:{c+1}'>"
            f"<div class='title'>{display_name}</div>"
            f"{price_block}{part_block}"
            f"{assur_block}{dark_block}{social_block}{voucher_block}{bundle_block}"
            f"</div>"
        )

    gt = {"category": category, "brand": brand_text, "set_id": set_id, "products": products}

    html = f"""
    <!doctype html><html><head><meta charset="utf-8"><title>Balanced Storefront</title>
    <style>
      :root {{ --bg:#fafafa; --card:#fff; --bd:#ddd; --ink:#111; --muted:#666; }}
      html,body {{ margin:0; padding:0; background:var(--bg); color:var(--ink);
                   font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
      .wrap {{ padding: 28px; max-width: 1120px; margin: 0 auto; }}
      h2 {{ margin: 0 0 6px 0; font-weight: 800; letter-spacing: 0.2px; }}
      .sub {{ margin: 0 0 18px 0; color: var(--muted); font-size: 14px; }}
      .grid {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 16px; }}
      .card {{ background:var(--card); border:1px solid var(--bd); border-radius:14px; padding:14px; min-height:140px; }}
      .title {{ font-weight: 700; margin-bottom: 6px; }}
      .price {{ font-weight: 800; font-size: 18px; margin: 6px 0; }}
      .mut {{ color: var(--muted); font-weight:600; font-size:12px; }}
      .pp {{ color: var(--muted); font-size: 12px; line-height: 1.25; margin-top:4px; }}
      .badge {{ display:inline-block; border-radius: 999px; padding: 4px 10px; font-size:12px; margin-top:8px; border:1px solid var(--bd); background:#f3f7f4; color:#0a513b; }}
      .pill {{ display:inline-block; border-radius: 999px; padding: 4px 10px; font-size:12px; margin-top:8px; border:1px solid var(--bd); background:#f6f4ff; color:#3a2ea5; }}
      .pill.warn {{ background:#fff7f2; color:#9a4a00; }}
      .chip {{ display:inline-block; border-radius:999px; padding:3px 8px; font-size:12px; margin:6px 6px 0 0; border:1px solid var(--bd); background:#f7f7f7; }}
      .chip.good {{ background:#f2fff6; color:#105d32; }}
      .chip.info {{ background:#f0f7ff; color:#0b4f9e; }}
      #groundtruth {{ display:none }}
    </style></head><body>
      <div class="wrap">
        <h2>Category: {category}</h2>
        <div class="sub">2×4 grid; balanced masks; 4-page rotation; one dark cue per screen (rotated); promos optional.</div>
        <div class="grid" id="grid">{''.join(cards)}</div>
      </div>
      <script id="groundtruth" type="application/json">{_json.dumps(gt)}</script>
    </body></html>
    """
    return html

def build_storefront_from_payload(payload: dict) -> tuple[str, dict]:
    html = render_screen(
        category=str(payload.get("product") or "product"),
        set_id="S0001",
        badges=list(payload.get("badges") or []),
        catalog_seed=int(payload.get("catalog_seed", 777)),
        price=float(payload.get("price") or 0.0),
        currency=str(payload.get("currency") or "£"),
        brand=str(payload.get("brand") or ""),
    )
    meta = {
        "set_id": "S0001",
        "category": str(payload.get("product") or "product"),
        "brand": str(payload.get("brand") or ""),
        "badges": list(payload.get("badges") or []),
        "catalog_seed": int(payload.get("catalog_seed", 777)),
        "price": float(payload.get("price") or 0.0),
        "currency": str(payload.get("currency") or "£"),
    }
    return html, meta


def save_storefront(job_id: str, html: str, meta: dict) -> tuple[str, str]:
    import pathlib, json
    out_dir = pathlib.Path("storefront")
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"{job_id}.html"
    html_path.write_text(html, encoding="utf-8")

    meta_path = out_dir / f"{job_id}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return str(html_path), str(meta_path)
