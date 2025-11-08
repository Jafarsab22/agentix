"""
score_image.py
---------------
Pure scoring helpers.

This module is meant to be called BY your app (e.g. app.py) AFTER the app has:
1. run vision on an image,
2. figured out, per card, which badges/cues are present,
3. (optionally) extracted/parsed the price.

We support two scoring options:

OPTION A: linear utility (like your PHP validator)
    U = Σ beta_i * x_i
    - for binary cues (Assurance, Scarcity tag, …) x_i = 1
    - for ln(price), x_i = ln(actual_price)
    - we also allow extra_cues (Row 1, Column 2, …)

OPTION B: readiness (your M, C, R recipe)
    for each present cue i:
        s_i = sign(beta_i) * M_i * C_i * R_i   (or use stored s_val if in DB)
        w_i = C_i * R_i
    readiness_raw = Σ s_i / Σ w_i  (if Σ w_i > 0 else 0)
    readiness = clipped to [-1, 1]
    if ln(price) row has price_weight in DB → multiply by that

The params are NOT hardcoded here.
We load them from your PHP endpoint: getCrossParameters.php.

Your app should do something like:

    from score_image import load_params_from_php, score_grid_2x4

    PARAMS = load_params_from_php("https://.../getCrossParameters.php?model=GPT-4.1-mini")

    # later, after vision:
    res = score_grid_2x4(grid_cards, PARAMS)

Where grid_cards is a list of 8 dicts:
    [
      {"cues": {"Assurance","Scarcity tag"}, "price": 20.0},
      {"cues": set(), "price": 24.99},
      ...
    ]
in row-major order (row1: 4 cards, row2: 4 cards).
We add Row/Column cues automatically inside score_grid_2x4.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Iterable, Sequence, Set, Optional, List

import requests


# ---------------------------------------------------------------------
# 1. load parameters from PHP
# ---------------------------------------------------------------------

def load_params_from_php(url: str) -> Dict[str, Dict[str, float]]:
    """
    Call your PHP endpoint and return a dict:
        { "Badge name": {"beta":..., "M":..., "C":..., "R":..., "s":..., "price_weight":...}, ... }
    """
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"PHP returned error: {data}")
    # we just return the 'params' object
    return data["params"]


# ---------------------------------------------------------------------
# 2. common helpers
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# 3. option A: linear utility
# ---------------------------------------------------------------------

def score_card_option_a(
    card_cues: Iterable[str],
    params: Dict[str, Dict[str, float]],
    *,
    price: Optional[float] = None,
    extra_cues: Iterable[str] = (),
) -> float:
    """
    Linear utility:
        U = Σ beta_i * x_i
    - cue present → x_i = 1
    - ln(price) present → x_i = ln(price)
    """
    U = 0.0
    # binary cues
    for cue in card_cues:
        p = params.get(cue)
        if p is None:
            continue
        beta = p.get("beta")
        if beta is not None:
            U += float(beta) * 1.0

    # extra cues (positions, etc.)
    for cue in extra_cues:
        p = params.get(cue)
        if p is None:
            continue
        beta = p.get("beta")
        if beta is not None:
            U += float(beta) * 1.0

    # ln(price)
    ln_price = _ln_or_none(price)
    if ln_price is not None:
        p = params.get("ln(price)")
        if p is not None and p.get("beta") is not None:
            U += float(p["beta"]) * ln_price

    return U


# ---------------------------------------------------------------------
# 4. option B: readiness (M·C·R)
# ---------------------------------------------------------------------

def score_card_option_b(
    card_cues: Iterable[str],
    params: Dict[str, Dict[str, float]],
    *,
    price: Optional[float] = None,
    extra_cues: Iterable[str] = (),
) -> float:
    """
    Readiness method:
      s_i = sign(beta_i) * M_i * C_i * R_i   (or stored s)
      w_i = C_i * R_i
      readiness = Σ s_i / Σ w_i  (clipped)
      if ln(price) has price_weight → multiply
    """
    sum_s = 0.0
    sum_w = 0.0

    # all cues we want to consider
    all_cues: Set[str] = set(card_cues) | set(extra_cues)

    # if the card has a price, we "activate" ln(price) as a cue (like our PHP validator did)
    ln_price = _ln_or_none(price)
    if ln_price is not None:
        all_cues.add("ln(price)")

    for cue in all_cues:
        p = params.get(cue)
        if p is None:
            continue

        C = float(p.get("C") or 0.0)
        R = float(p.get("R") or 0.0)
        w_i = C * R

        # prefer stored s if present
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
        readiness = sum_s / sum_w
        readiness = max(-1.0, min(1.0, readiness))

    # optional price_weight (only if price was present)
    if ln_price is not None:
        p_price = params.get("ln(price)")
        if p_price is not None and p_price.get("price_weight") is not None:
            pw = float(p_price["price_weight"])
            pw = max(0.0, min(1.0, pw))
            readiness *= pw

    return readiness


# ---------------------------------------------------------------------
# 5. 2×4 grid scorer (this is what your app calls)
# ---------------------------------------------------------------------

def score_grid_2x4(
    cards: Sequence[dict],
    params: Dict[str, Dict[str, float]],
) -> dict:
    """
    cards: sequence of 8 dicts:
        {
          "cues": {"Assurance","Scarcity tag"},   # any strings your vision returned
          "price": 24.99                          # optional
        }
    We will automatically add position cues:
        row 1/2 and column 1/2/3 (col4 has no cue in your table)
    Returns a dict with per-card and aggregates, mirroring your app.py style.
    """
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
        # column 4 -> no extra column cue

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
