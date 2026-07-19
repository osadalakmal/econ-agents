"""Results reporting — console time-series and structured output."""
from __future__ import annotations

import json
from typing import Any

from .simulation import RoundResult


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

    # Demand / supply line
    shortage_tag = f"  *** SHORTAGE {s['shortage']:.0f} units ***" if s["shortage"] > 0 else ""
    print(
        f"  demand={s['demand']:.0f}  consumed={s['actual_consumption']:.0f}"
        f"  arrived_supply={s['arrived_supply']:.0f}"
        f"  fill={s['fill_rate']:.0f}%{shortage_tag}"
    )

    # Consumer action breakdown (compact)
    parts = "  ".join(
        f"{a}: {info['count']} ({info['pct']:.0f}%)"
        for a, info in s["consumer_actions"].items()
    )
    print(f"  consumers → {parts}")

    # Supplier summary
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

    # We need market history — pull from last result's market if available
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


def to_json(results: list[RoundResult]) -> str:
    out = []
    for r in results:
        s = r.summary()
        out.append(s)
    return json.dumps(out, indent=2)
