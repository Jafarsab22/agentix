"""
score_image.py — minimal scoring utility for a single product image or a 2x4 grid

Intuition recap (single run): for each lever/cue i
  s_i = sign(beta_i) * M_i * C_i * R_i
Card score S = sum(s_i over present cues) / sum(C_i*R_i over present cues)  (clipped to [-1,1])
Grid score = mean(S_j) across 8 cards and best(S_j) (optionally apply the same clamp)
Price sanity (from ln(price) beta): price_weight = min(1, abs(beta_price)/BETA_REF)
Final = raw_score * price_weight

This module deliberately avoids CV. You pass which cues are present.
You can wire this into your app.py by: (1) uploading an image, (2) collecting which cues appear
per card, (3) calling these functions.

Usage examples (programmatic):

  from score_image import score_single_card, score_grid_2x4

  # Single product with badges present
  card_cues = {"Assurance", "Timer", "Strike-through"}
  result = score_single_card(card_cues)

  # Grid 2x4: list of 8 sets of cues (row-major), we add the row/col automatically
  grid = [
      {"Assurance"}, {"All-in framing"}, set(), {"Scarcity tag"},
      {"Timer"}, set(), {"Assurance","Scarcity tag"}, set(),
  ]
  result = score_grid_2x4(grid)

Replace LEARNED_PARAMS below with values loaded from DB if desired.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import math

# ------------------------------
# Learned parameters from your phone 500-iteration run (approx)
# You can swap this dict with values read from your DB.
# For each lever: beta, M, C, R
LEARNED_PARAMS: Dict[str, Dict[str, float]] = {
    "Row 1":            {"beta": 0.824, "M": 0.74, "C": 0.78, "R": 1.00},
    "Column 1":         {"beta": 0.926, "M": 0.81, "C": 0.72, "R": 1.00},
    "Column 2":         {"beta": 0.097, "M": 0.13, "C": 0.75, "R": 0.25},
    "Column 3":         {"beta": -0.224,"M": 0.27, "C": 0.76, "R": 0.70},
    "All-in framing":   {"beta": -0.705,"M": 0.68, "C": 0.81, "R": 1.00},
    "Assurance":        {"beta": 1.656, "M": 0.98, "C": 0.70, "R": 1.00},
    "Scarcity tag":     {"beta": 0.474, "M": 0.62, "C": 0.75, "R": 0.95},
    "Strike-through":   {"beta": 0.096, "M": 0.13, "C": 0.56, "R": 0.11},
    "Timer":            {"beta": 0.711, "M": 0.74, "C": 0.55, "R": 0.99},
    "ln(price)":        {"beta": -0.665,"M": 0.68, "C": 0.92, "R": 1.00},
}

# Saturation constant (k) — shared across cues; shown for completeness in case you recompute M
K_SAT = 7.2
# Price sanity reference |beta_lnprice| expected magnitude
BETA_REF = 0.6

# ------------------------------
# Core scoring helpers

def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def cue_score_components(lever: str) -> Tuple[float, float, float, float]:
    """Return (beta, M, C, R) for lever, raising KeyError if missing."""
    d = LEARNED_PARAMS[lever]
    return d["beta"], d["M"], d["C"], d["R"]


def cue_contribution(lever: str) -> Tuple[float, float]:
    """Return (s_i, weight_i) for a single lever present on the card.
    s_i = sign(beta) * M * C * R; weight_i = C * R
    """
    beta, M, C, R = cue_score_components(lever)
    s_i = _sign(beta) * M * C * R
    w_i = C * R
    return s_i, w_i


def price_weight() -> float:
    """Compute page/card-level price sanity multiplier from ln(price) beta."""
    beta = LEARNED_PARAMS["ln(price)"]["beta"]
    return min(1.0, abs(beta) / BETA_REF) if BETA_REF > 0 else 1.0


def score_single_card(cues_present: Iterable[str]) -> Dict[str, float]:
    """Score a single product/card given the set of present cues (badges).
    Position levers are NOT applied here (standalone image). ln(price) is applied once.
    Returns dict with raw, final, and breakdown sums.
    """
    cues = set(cues_present)
    # Always include ln(price) as a model sanity cue (applies to choice utility globally)
    cues_with_price = set(cues)
    if "ln(price)" in LEARNED_PARAMS:
        cues_with_price.add("ln(price)")

    s_sum = 0.0
    w_sum = 0.0
    for lever in cues_with_price:
        if lever not in LEARNED_PARAMS:
            continue
        s_i, w_i = cue_contribution(lever)
        s_sum += s_i
        w_sum += w_i

    raw = max(-1.0, min(1.0, (s_sum / w_sum) if w_sum > 0 else 0.0))
    final = raw * price_weight()
    return {
        "raw": raw,
        "final": final,
        "sum_s": s_sum,
        "sum_w": w_sum,
        "price_weight": price_weight(),
    }


def score_grid_2x4(cards_cues: Sequence[Set[str]]) -> Dict[str, object]:
    """Score a 2x4 grid (8 cards) given a sequence of 8 cue-sets (row-major).
    Adds position levers (Row r, Column c) automatically per slot.
    Returns per-card scores and aggregate (mean, best) both raw and final.
    """
    if len(cards_cues) != 8:
        raise ValueError("cards_cues must be length 8 for a 2x4 grid")

    per_card = []
    s_price = price_weight()

    for idx, cues in enumerate(cards_cues):
        r = 1 if idx < 4 else 2
        c = (idx % 4) + 1
        position_cues = set()
        position_cues.add(f"Row {r}")
        if c == 1:
            position_cues.add("Column 1")
        elif c == 2:
            position_cues.add("Column 2")
        elif c == 3:
            position_cues.add("Column 3")
        # Column 4 has no explicit coefficient in the provided table; skip unless added.

        s_sum = 0.0
        w_sum = 0.0
        # Add badge cues
        for lever in set(cues):
            if lever not in LEARNED_PARAMS:
                continue
            s_i, w_i = cue_contribution(lever)
            s_sum += s_i
            w_sum += w_i
        # Add position cues
        for lever in position_cues:
            if lever not in LEARNED_PARAMS:
                continue
            s_i, w_i = cue_contribution(lever)
            s_sum += s_i
            w_sum += w_i
        # Optionally include ln(price) at card level as well (applies uniformly)
        if "ln(price)" in LEARNED_PARAMS:
            s_i, w_i = cue_contribution("ln(price)")
            s_sum += s_i
            w_sum += w_i

        raw = max(-1.0, min(1.0, (s_sum / w_sum) if w_sum > 0 else 0.0))
        final = raw * s_price
        per_card.append({"raw": raw, "final": final, "sum_s": s_sum, "sum_w": w_sum, "row": r, "col": c})

    raw_mean = sum(p["raw"] for p in per_card) / 8.0
    final_mean = sum(p["final"] for p in per_card) / 8.0
    raw_best = max(p["raw"] for p in per_card)
    final_best = max(p["final"] for p in per_card)

    return {
        "cards": per_card,
        "raw_mean": max(-1.0, min(1.0, raw_mean)),
        "final_mean": max(-1.0, min(1.0, final_mean)),
        "raw_best": raw_best,
        "final_best": final_best,
        "price_weight": s_price,
    }


# ------------------------------
# Optional tiny CLI for quick tests (no CV, you pass cue names)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a single product or 2x4 grid using learned parameters.")
    sub = parser.add_subparsers(dest="mode", required=True)

    one = sub.add_parser("single", help="Score a single product/card")
    one.add_argument("--cues", nargs="*", default=[], help="Badge cues present, e.g., Assurance Timer 'All-in framing'")

    grid = sub.add_parser("grid", help="Score a 2x4 grid (8 cards)")
    grid.add_argument("--card", action="append", nargs="*", default=[], help=(
        "Provide 8 times. Each --card is the list of cues for that slot (row-major). "
        "Example: --card Assurance --card 'All-in framing' Scarcity --card ... (8 of them)"
    ))

    args = parser.parse_args()

    if args.mode == "single":
        res = score_single_card(set(args.cues))
        print(res)
    else:
        if len(args.card) != 8:
            raise SystemExit("You must provide exactly 8 --card groups for the grid mode.")
        grid_sets = [set(c) for c in args.card]
        res = score_grid_2x4(grid_sets)
        print(res)
