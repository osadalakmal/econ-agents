"""Consumer agents — observe market state and decide how much to buy."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from .behaviors import BehaviorEngine, BehaviorSpec
from .events import ConsumerDecision
from .world import WorldSnapshot


@dataclass
class AgentState:
    agent_id: int
    savings: float = 100.0
    inventory: float = 10.0
    history: list[str] = field(default_factory=list)


class ConsumerAgent:
    """
    Observes current market prices and stock signals; produces a demand decision.
    Runs as an asyncio coroutine so all 1000+ agents decide concurrently.
    """

    def __init__(
        self,
        agent_id: int,
        spec: BehaviorSpec,
        market_name: str,
        seed: int | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_type = spec.agent_type_id
        self._market = market_name
        self._engine = BehaviorEngine(spec, rng=random.Random(seed))
        self.state = AgentState(agent_id=agent_id)

    async def decide(self, world: WorldSnapshot) -> ConsumerDecision:
        obs = self._build_observation(world)
        await asyncio.sleep(0)  # yield — lets other coroutines interleave
        action, qty_demanded, reasoning = self._engine.decide(obs)
        self.state.history.append(f"R{world.round}: {action} qty={qty_demanded:.2f}")
        return ConsumerDecision(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            market=self._market,
            action=action,
            quantity_demanded=qty_demanded,
            reasoning=reasoning,
        )

    def _build_observation(self, world: WorldSnapshot) -> dict[str, Any]:
        mkt = world.market(self._market)
        return {
            # Market signals
            "price": mkt["price"],
            "price_change_pct": mkt["price_change_pct"],
            "stock_ratio": mkt["stock_ratio"],   # < 1 = shortage
            "stock": mkt["stock"],
            "pipeline_total": mkt["pipeline_total"],
            # Agent state
            "agent_savings": self.state.savings,
            "agent_inventory": self.state.inventory,
            "round": world.round,
        }


def build_consumer_pool(
    specs: list[BehaviorSpec],
    population_size: int,
    market_name: str,
    seed: int = 42,
) -> list[ConsumerAgent]:
    rng = random.Random(seed)
    total = sum(s.proportion for s in specs)
    agents: list[ConsumerAgent] = []

    for i, spec in enumerate(specs):
        if i < len(specs) - 1:
            count = round(population_size * spec.proportion / total)
        else:
            count = population_size - len(agents)

        for _ in range(count):
            agents.append(ConsumerAgent(
                agent_id=len(agents),
                spec=spec,
                market_name=market_name,
                seed=rng.randint(0, 2**31),
            ))

    return agents
