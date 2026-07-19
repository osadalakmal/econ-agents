"""Main simulation loop — turn-based, two-phase per round."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, build_agent_pool
from .behaviors import parse_behavior_specs
from .events import Decision, Stimulus
from .world import World


@dataclass
class RoundResult:
    round_number: int
    stimulus: Stimulus
    decisions: list[Decision]
    world_snapshot: dict[str, Any]
    duration_ms: float

    def summary(self) -> dict[str, Any]:
        action_counts: dict[str, int] = {}
        type_breakdown: dict[str, dict[str, int]] = {}

        for d in self.decisions:
            action_counts[d.action] = action_counts.get(d.action, 0) + 1
            tb = type_breakdown.setdefault(d.agent_type, {})
            tb[d.action] = tb.get(d.action, 0) + 1

        total = len(self.decisions)
        return {
            "round": self.round_number,
            "stimulus": self.stimulus.name,
            "total_agents": total,
            "actions": {k: {"count": v, "pct": round(v / total * 100, 1)}
                        for k, v in sorted(action_counts.items())},
            "by_type": type_breakdown,
            "aggregate_demand_delta": round(
                sum(d.quantity_delta for d in self.decisions), 2
            ),
            "duration_ms": round(self.duration_ms, 1),
        }


class Simulation:
    """
    Orchestrates the two-phase loop:
      1. DECIDE — all agents coroutines run concurrently (asyncio.gather)
      2. SETTLE — decisions applied to world state sequentially
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._world = World(config["world"]["commodities"])
        specs = parse_behavior_specs(config["agent_types"])
        self._agents: list[Agent] = build_agent_pool(
            specs,
            population_size=config["population"]["size"],
            seed=config.get("seed", 42),
        )
        self._results: list[RoundResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, stimuli: list[Stimulus]) -> list[RoundResult]:
        """Run one round per stimulus, return all results."""
        for stimulus in stimuli:
            result = await self._run_round(stimulus)
            self._results.append(result)
        return self._results

    async def run_single(self, stimulus: Stimulus) -> RoundResult:
        result = await self._run_round(stimulus)
        self._results.append(result)
        return result

    @property
    def agents(self) -> list[Agent]:
        return self._agents

    @property
    def results(self) -> list[RoundResult]:
        return self._results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_round(self, stimulus: Stimulus) -> RoundResult:
        t0 = time.monotonic()

        # Apply stimulus to world (price/supply change)
        self._world.apply_stimulus(stimulus)
        snapshot_before = self._world.snapshot()

        # PHASE 1: DECIDE — all agents decide concurrently
        decisions: list[Decision] = await asyncio.gather(
            *[agent.decide(snapshot_before, stimulus) for agent in self._agents]
        )

        # PHASE 2: SETTLE — apply decisions to world state
        self._world.apply_decisions(decisions)
        world_snap = self._world.snapshot().snapshot()
        self._world.advance_round()

        elapsed = (time.monotonic() - t0) * 1000

        return RoundResult(
            round_number=snapshot_before.round,
            stimulus=stimulus,
            decisions=list(decisions),
            world_snapshot=world_snap,
            duration_ms=elapsed,
        )
