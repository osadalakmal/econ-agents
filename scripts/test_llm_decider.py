#!/usr/bin/env python3
"""
Phase 1 acceptance test for sim/llm_decider.py.

Feeds 20 hand-picked states (spanning low/high price_change_pct, low/high
stock_ratio, low/high savings) to decide_llm and prints the action distribution.

Usage:
    # With LM Studio running at localhost:1234:
    python scripts/test_llm_decider.py

    # Custom endpoint or model:
    python scripts/test_llm_decider.py --base-url http://localhost:1234/v1 --model qwen3-4b

    # Benchmark: time 50 sequential calls to compare Qwen3-4B vs Qwen3-1.7B:
    python scripts/test_llm_decider.py --bench 50

Pass criteria:
  - Completes without crash (fallback path counts as pass if server is down)
  - Action distribution is NOT 100% one action — variety required
  - High price_change_pct + low stock_ratio states bias toward buy_more
  - High price_change_pct + high stock_ratio states bias toward buy_less
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.llm_decider import LLMDeciderConfig, decide_llm

# 20 hand-crafted states: (price, price_change_pct, stock_ratio, savings, inventory,
#                          risk_tolerance, planning_horizon, label)
TEST_STATES = [
    # --- Panic / shortage ---
    {"price": 3.50, "price_change_pct": 55.0, "stock_ratio": 0.20, "agent_savings": 200.0,
     "agent_inventory": 2.0, "risk_tolerance": 0.8, "planning_horizon_rounds": 6, "round": 5,
     "_label": "extreme shortage + price spike, bold agent"},
    {"price": 2.80, "price_change_pct": 35.0, "stock_ratio": 0.35, "agent_savings": 80.0,
     "agent_inventory": 3.0, "risk_tolerance": 0.6, "planning_horizon_rounds": 4, "round": 4,
     "_label": "severe shortage, moderate agent"},
    {"price": 1.90, "price_change_pct": 22.0, "stock_ratio": 0.55, "agent_savings": 120.0,
     "agent_inventory": 8.0, "risk_tolerance": 0.5, "planning_horizon_rounds": 3, "round": 3,
     "_label": "moderate shortage, neutral agent"},
    # --- Rising prices, ample stock ---
    {"price": 1.60, "price_change_pct": 18.0, "stock_ratio": 1.40, "agent_savings": 150.0,
     "agent_inventory": 15.0, "risk_tolerance": 0.3, "planning_horizon_rounds": 2, "round": 6,
     "_label": "price rising but stock OK, cautious agent"},
    {"price": 2.10, "price_change_pct": 30.0, "stock_ratio": 1.80, "agent_savings": 60.0,
     "agent_inventory": 12.0, "risk_tolerance": 0.2, "planning_horizon_rounds": 1, "round": 8,
     "_label": "price spiking + glut, very cautious agent"},
    # --- Glut / falling prices ---
    {"price": 0.65, "price_change_pct": -28.0, "stock_ratio": 3.20, "agent_savings": 200.0,
     "agent_inventory": 40.0, "risk_tolerance": 0.5, "planning_horizon_rounds": 3, "round": 20,
     "_label": "severe glut + price crash, high inventory"},
    {"price": 0.80, "price_change_pct": -15.0, "stock_ratio": 2.50, "agent_savings": 90.0,
     "agent_inventory": 20.0, "risk_tolerance": 0.4, "planning_horizon_rounds": 2, "round": 15,
     "_label": "glut + falling prices, moderate inventory"},
    {"price": 0.90, "price_change_pct": -8.0, "stock_ratio": 1.90, "agent_savings": 110.0,
     "agent_inventory": 14.0, "risk_tolerance": 0.6, "planning_horizon_rounds": 4, "round": 12,
     "_label": "mild surplus + falling prices, long horizon"},
    # --- Near-equilibrium ---
    {"price": 1.02, "price_change_pct": 1.5, "stock_ratio": 1.05, "agent_savings": 100.0,
     "agent_inventory": 10.0, "risk_tolerance": 0.5, "planning_horizon_rounds": 3, "round": 10,
     "_label": "near-equilibrium, neutral"},
    {"price": 0.98, "price_change_pct": -2.0, "stock_ratio": 0.95, "agent_savings": 100.0,
     "agent_inventory": 9.0, "risk_tolerance": 0.5, "planning_horizon_rounds": 3, "round": 10,
     "_label": "near-equilibrium, slight shortage"},
    # --- Low savings / budget constrained ---
    {"price": 1.50, "price_change_pct": 18.0, "stock_ratio": 0.70, "agent_savings": 15.0,
     "agent_inventory": 2.0, "risk_tolerance": 0.1, "planning_horizon_rounds": 1, "round": 5,
     "_label": "price rising, nearly broke, very cautious"},
    {"price": 2.00, "price_change_pct": 40.0, "stock_ratio": 0.45, "agent_savings": 8.0,
     "agent_inventory": 1.0, "risk_tolerance": 0.1, "planning_horizon_rounds": 1, "round": 4,
     "_label": "crisis, nearly out of savings"},
    # --- High savings / bold ---
    {"price": 1.30, "price_change_pct": 12.0, "stock_ratio": 0.80, "agent_savings": 500.0,
     "agent_inventory": 5.0, "risk_tolerance": 0.9, "planning_horizon_rounds": 8, "round": 3,
     "_label": "rising prices, rich, bold, long horizon"},
    {"price": 0.75, "price_change_pct": -22.0, "stock_ratio": 2.80, "agent_savings": 500.0,
     "agent_inventory": 30.0, "risk_tolerance": 0.9, "planning_horizon_rounds": 8, "round": 18,
     "_label": "glut + falling, rich but already stocked"},
    # --- Recovering market ---
    {"price": 1.10, "price_change_pct": -10.0, "stock_ratio": 1.10, "agent_savings": 95.0,
     "agent_inventory": 8.0, "risk_tolerance": 0.5, "planning_horizon_rounds": 4, "round": 25,
     "_label": "recovering from crisis, price falling back"},
    {"price": 1.20, "price_change_pct": -5.0, "stock_ratio": 1.30, "agent_savings": 120.0,
     "agent_inventory": 12.0, "risk_tolerance": 0.6, "planning_horizon_rounds": 5, "round": 28,
     "_label": "post-crisis, mild surplus"},
    # --- Very tight supply ---
    {"price": 4.50, "price_change_pct": 8.0, "stock_ratio": 0.12, "agent_savings": 300.0,
     "agent_inventory": 0.5, "risk_tolerance": 0.7, "planning_horizon_rounds": 6, "round": 8,
     "_label": "near-stockout, moderately bold"},
    {"price": 5.00, "price_change_pct": 3.0, "stock_ratio": 0.08, "agent_savings": 50.0,
     "agent_inventory": 0.0, "risk_tolerance": 0.3, "planning_horizon_rounds": 2, "round": 9,
     "_label": "stockout, low savings, cautious"},
    # --- Stagnant/stable ---
    {"price": 1.00, "price_change_pct": 0.0, "stock_ratio": 1.00, "agent_savings": 100.0,
     "agent_inventory": 10.0, "risk_tolerance": 0.5, "planning_horizon_rounds": 3, "round": 0,
     "_label": "pristine equilibrium, round 0"},
    {"price": 1.01, "price_change_pct": 0.5, "stock_ratio": 1.02, "agent_savings": 102.0,
     "agent_inventory": 10.5, "risk_tolerance": 0.5, "planning_horizon_rounds": 3, "round": 1,
     "_label": "trivially stable"},
]


async def run_test(cfg: LLMDeciderConfig, bench_n: int = 0) -> None:
    states = TEST_STATES if not bench_n else [TEST_STATES[0]] * bench_n
    label = "benchmark" if bench_n else "acceptance test"
    n = len(states)

    print(f"\n{'═'*65}")
    print(f"  LLM decider {label}  ({n} calls → {cfg.base_url}, model={cfg.model})")
    print(f"{'═'*65}")

    results = []
    fallbacks = 0
    t0 = time.monotonic()

    for i, state in enumerate(states):
        lbl = state.get("_label", f"state-{i}")
        obs = {k: v for k, v in state.items() if not k.startswith("_")}
        action, qf, reasoning = await decide_llm(obs, cfg)
        is_fallback = reasoning.startswith("llm-fallback") or "not installed" in reasoning
        if is_fallback:
            fallbacks += 1
        results.append((action, qf, reasoning))
        if not bench_n:
            fb_tag = " [FALLBACK]" if is_fallback else ""
            print(f"  {i+1:>2}. {lbl}")
            print(f"      → action={action:<10}  qty_factor={qf:.2f}  {fb_tag}")
            print(f"         {reasoning[:80]}")

    elapsed = time.monotonic() - t0
    counts = Counter(r[0] for r in results)

    print(f"\n{'─'*65}")
    print(f"  Action distribution ({n} calls, {elapsed:.1f}s, {elapsed/n*1000:.0f}ms/call avg):")
    for action, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / n * 30)
        print(f"    {action:<12} {count:>3}  ({count/n*100:.0f}%)  {bar}")
    if fallbacks:
        print(f"\n  ⚠  {fallbacks}/{n} calls used fallback (server unreachable or timeout)")
    else:
        print(f"\n  ✓  No fallbacks — server responded to all {n} calls")

    variety = len(counts)
    if fallbacks == n:
        print("  RESULT: all fallbacks — server is down; test inconclusive")
    elif variety < 2:
        print(f"  ✗ FAIL: only {variety} distinct action(s) — no variety in responses")
        sys.exit(1)
    else:
        print(f"  ✓ PASS: {variety} distinct actions across {n} states")
    print(f"{'═'*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM decider acceptance / benchmark test")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="qwen/qwen3-4b-thinking-2507:2")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--bench", type=int, default=0, metavar="N",
                        help="Benchmark mode: run N sequential calls to measure throughput")
    args = parser.parse_args()

    cfg = LLMDeciderConfig(
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout,
        max_retries=1,
    )
    asyncio.run(run_test(cfg, bench_n=args.bench))


if __name__ == "__main__":
    main()
