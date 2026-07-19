"""Agent — an asyncio coroutine that observes and decides each round."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from .behaviors import BehaviorEngine, BehaviorSpec
from .events import Decision, Stimulus
from .world import WorldState


@dataclass
class AgentState:
    """Per-agent persistent state across rounds."""
    agent_id: int
    savings: float = 100.0     # arbitrary wealth units
    inventory: float = 10.0    # units of tracked commodity on hand
    history: list[str] = field(default_factory=list)


class Agent:
    """
    Each agent runs as an asyncio coroutine during the DECIDE phase.
    Agents never share mutable state — they only read WorldState snapshots.
    """

    def __init__(
        self,
        agent_id: int,
        spec: BehaviorSpec,
        seed: int | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_type = spec.agent_type_id
        self._engine = BehaviorEngine(spec, rng=random.Random(seed))
        self.state = AgentState(agent_id=agent_id)

    async def decide(
        self,
        world: WorldState,
        stimulus: Stimulus,
    ) -> Decision:
        """
        Async so all agents can run concurrently via asyncio.gather.
        CPU-bound work stays minimal; heavy models could await a thread pool.
        """
        # Agents build their observation from the world state + stimulus
        obs = self._build_observation(world, stimulus)

        # Yield control briefly — lets other coroutines interleave
        await asyncio.sleep(0)

        action, qty_delta, reasoning = self._engine.decide(obs)

        self.state.history.append(f"R{world.round}: {action} ({reasoning})")

        return Decision(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            commodity=stimulus.commodity,
            action=action,
            quantity_delta=qty_delta,
            reasoning=reasoning,
        )

    def _build_observation(
        self, world: WorldState, stimulus: Stimulus
    ) -> dict[str, Any]:
        """Merge world state and stimulus into a flat observation dict."""
        obs: dict[str, Any] = {}
        obs.update(stimulus.observation())

        com = world.commodities.get(stimulus.commodity)
        if com:
            obs["current_price"] = com.price
            obs["current_supply"] = com.supply

        obs["agent_savings"] = self.state.savings
        obs["agent_inventory"] = self.state.inventory
        obs["round"] = world.round
        return obs


def build_agent_pool(
    specs: list[BehaviorSpec],
    population_size: int,
    seed: int = 42,
) -> list[Agent]:
    """
    Distribute agents across specs according to proportion.
    Proportions are normalised if they don't sum to 1.
    """
    rng = random.Random(seed)
    total = sum(s.proportion for s in specs)
    agents: list[Agent] = []
    agent_id = 0

    for i, spec in enumerate(specs):
        # Last spec absorbs rounding remainder
        if i < len(specs) - 1:
            count = round(population_size * spec.proportion / total)
        else:
            count = population_size - len(agents)

        for _ in range(count):
            agents.append(Agent(
                agent_id=agent_id,
                spec=spec,
                seed=rng.randint(0, 2**31),
            ))
            agent_id += 1

    return agents
