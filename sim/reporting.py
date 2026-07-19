"""Results reporting — console time-series and structured output."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .simulation import RoundResult

if TYPE_CHECKING:
    from .compare import ComparisonResult

DISCLAIMER = (
    "NOTE: Results show model predictions under the stated behavioral assumptions "
    "and calibration, not recovered historical facts. "
    "Do not interpret outputs as empirical validation of real-world events."
)


def print_round(result: RoundResult) -> None:
    s = result.summary()
    shock_tag = f"  [SHOCK: {', '.join(s['shocks'])}]" if s["shocks"] else ""
    sign = "+" if s["price_change_pct"] >= 0 else ""
    print(f"\n{'─'*65}")
    print(
        f"  Round {s['round']:>2}"
        f"  price {s['price_before']:.4f}→{s['price_after']:.4f}"
        f"  ({sign}{s['price_change_pct']:.1f}%)"
        f"  stock {s['stock_before']:.0f}→{s['stock_after']:.0f}"
        f"{shock_tag}"
    )
    print(f"{'─'*65}")

    shortage_tag = f"  *** SHORTAGE {s['shortage']:.0f} units ***" if s["shortage"] > 0 else ""
    print(
        f"  demand={s['demand']:.0f}  consumed={s['actual_consumption']:.0f}"
        f"  arrived_supply={s['arrived_supply']:.0f}"
        f"  fill={s['fill_rate']:.0f}%{shortage_tag}"
    )

    parts = "  ".join(
        f"{a}: {info['count']} ({info['pct']:.0f}%)"
        for a, info in s["consumer_actions"].items()
    )
    print(f"  consumers → {parts}")

    if s["supplier_decisions"]:
        print("  suppliers →", end="")
        for sd in s["supplier_decisions"]:
            adj = f"{sd['adj_pct']:+.0f}%" if sd["adj_pct"] != 0 else "="
            print(f"  [{sd['type']}#{sd['id']} {sd['old_rate']:.0f}→{sd['new_rate']:.0f} {adj}]", end="")
        print()


def print_simulation_header(config: dict) -> None:
    mkt_name = list(config["markets"].keys())[0]
    mkt = config["markets"][mkt_name]
    print(f"\n{'═'*65}")
    print(f"  SIMULATION: {mkt_name.upper()}  |  {config['consumers']['size']} consumers")
    print(f"  Initial price: {mkt['initial_price']}  |  Initial stock: {mkt['initial_stock']}")
    algo = mkt.get("price_algorithm", {})
    print(
        f"  Price algo: {algo.get('type','stock_based')}"
        f"  target_days={algo.get('target_stock_days',14)}"
        f"  elasticity={algo.get('elasticity',0.3)}"
    )
    print(f"{'═'*65}")


def print_simulation_summary(results: list[RoundResult]) -> None:
    if not results:
        return

    print(f"\n{'═'*65}")
    print("  ROUND-BY-ROUND PRICE & STOCK")
    print(f"  {'Rnd':>3}  {'Price':>7}  {'Stock':>7}  {'Demand':>7}  {'Shortage':>8}  {'Pipeline':>8}")
    print(f"  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")

    for r in results:
        s = r.summary()
        print(
            f"  {s['round']:>3}  {s['price_after']:>7.4f}  {s['stock_after']:>7.0f}"
            f"  {s['demand']:>7.0f}  {s['shortage']:>8.0f}"
            f"  {'':>8}"
        )

    first_price = results[0].clearing.price_before
    last_price = results[-1].clearing.price_after
    pct = (last_price - first_price) / first_price * 100
    sign = "+" if pct >= 0 else ""
    print(f"\n  Price start→end: {first_price:.4f} → {last_price:.4f}  ({sign}{pct:.1f}%)")
    total_shortage = sum(r.clearing.shortage for r in results)
    if total_shortage > 0:
        print(f"  Total shortage across all rounds: {total_shortage:.0f} units")
    total_ms = sum(r.duration_ms for r in results)
    print(f"  Rounds: {len(results)}  |  Total time: {total_ms:.0f} ms"
          f"  |  Avg: {total_ms/len(results):.1f} ms/round")
    print(f"{'═'*65}")


def print_comparison(cmp: ComparisonResult) -> None:
    W = 78
    print(f"\n{'═'*W}")
    print(f"  COMPARISON: [{cmp.preset_a.upper()}] vs [{cmp.preset_b.upper()}]")
    print(f"{'═'*W}")
    header = (
        f"  {'Rnd':>3}  "
        f"{'Price-A':>8}  {'Price-B':>8}  {'ΔPrice':>8}  "
        f"{'Stock-A':>8}  {'Stock-B':>8}  {'ΔStock':>8}"
    )
    print(header)
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for ra, rb in zip(cmp.results_a, cmp.results_b):
        sa, sb = ra.summary(), rb.summary()
        dp = sa["price_after"] - sb["price_after"]
        ds = sa["stock_after"] - sb["stock_after"]
        print(
            f"  {sa['round']:>3}"
            f"  {sa['price_after']:>8.4f}"
            f"  {sb['price_after']:>8.4f}"
            f"  {dp:>+8.4f}"
            f"  {sa['stock_after']:>8.0f}"
            f"  {sb['stock_after']:>8.0f}"
            f"  {ds:>+8.0f}"
        )

    if cmp.results_a and cmp.results_b:
        avg_a = sum(r.clearing.price_after for r in cmp.results_a) / len(cmp.results_a)
        avg_b = sum(r.clearing.price_after for r in cmp.results_b) / len(cmp.results_b)
        short_a = sum(r.clearing.shortage for r in cmp.results_a)
        short_b = sum(r.clearing.shortage for r in cmp.results_b)
        print(f"\n  Avg price    — {cmp.preset_a}: {avg_a:.4f}  |  {cmp.preset_b}: {avg_b:.4f}")
        print(f"  Total shortage — {cmp.preset_a}: {short_a:.0f}  |  {cmp.preset_b}: {short_b:.0f}")

    print(f"\n  {DISCLAIMER}")
    print(f"{'═'*W}")


def to_json(results: list[RoundResult]) -> str:
    return json.dumps([r.summary() for r in results], indent=2)


def to_json_comparison(cmp: ComparisonResult) -> str:
    out: dict[str, Any] = {
        "comparison": {
            "preset_a": cmp.preset_a,
            "preset_b": cmp.preset_b,
        },
        "disclaimer": DISCLAIMER,
        cmp.preset_a: [r.summary() for r in cmp.results_a],
        cmp.preset_b: [r.summary() for r in cmp.results_b],
        "diff": [
            {
                "round": ra.round_number,
                "price_delta": round(ra.clearing.price_after - rb.clearing.price_after, 6),
                "stock_delta": round(ra.clearing.stock_after - rb.clearing.stock_after, 2),
                "shortage_delta": round(ra.clearing.shortage - rb.clearing.shortage, 2),
            }
            for ra, rb in zip(cmp.results_a, cmp.results_b)
        ],
    }
    return json.dumps(out, indent=2)
