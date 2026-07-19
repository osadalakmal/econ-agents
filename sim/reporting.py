"""Results reporting — console and structured output."""
from __future__ import annotations

import json
from typing import Any

from .simulation import RoundResult


def print_round(result: RoundResult) -> None:
    s = result.summary()
    print(f"\n{'='*60}")
    print(f"  Round {s['round']} | Stimulus: {s['stimulus']}")
    print(f"{'='*60}")
    print(f"  Agents: {s['total_agents']}  |  "
          f"Aggregate demand delta: {s['aggregate_demand_delta']:+.1f}  |  "
          f"Time: {s['duration_ms']} ms")
    print()
    print(f"  {'Action':<20} {'Count':>6}  {'%':>6}")
    print(f"  {'-'*36}")
    for action, info in s["actions"].items():
        print(f"  {action:<20} {info['count']:>6}  {info['pct']:>5.1f}%")
    print()
    print("  By agent type:")
    for atype, actions in s["by_type"].items():
        total = sum(actions.values())
        parts = ", ".join(f"{a}: {c}" for a, c in sorted(actions.items()))
        print(f"    [{atype}] ({total} agents)  {parts}")

    snap = result.world_snapshot["commodities"]
    print()
    print("  World state after round:")
    for name, vals in snap.items():
        print(f"    {name}: price={vals['price']:.4f}  "
              f"supply={vals['supply']:.1f}  demand_delta={vals['demand']:.1f}")


def print_simulation_summary(results: list[RoundResult]) -> None:
    if not results:
        return
    print(f"\n{'#'*60}")
    print("  SIMULATION SUMMARY")
    print(f"{'#'*60}")
    print(f"  Rounds: {len(results)}")
    total_ms = sum(r.duration_ms for r in results)
    total_agents = len(results[0].decisions)
    print(f"  Total agent-decisions: {total_agents * len(results):,}")
    print(f"  Total time: {total_ms:.1f} ms")
    print(f"  Avg per round: {total_ms/len(results):.1f} ms")


def to_json(results: list[RoundResult]) -> str:
    out: list[dict] = []
    for r in results:
        summary = r.summary()
        summary["decisions"] = [
            {
                "agent_id": d.agent_id,
                "agent_type": d.agent_type,
                "action": d.action,
                "quantity_delta": d.quantity_delta,
                "reasoning": d.reasoning,
            }
            for d in r.decisions
        ]
        out.append(summary)
    return json.dumps(out, indent=2)
