"""Consumer agents — observe market state and decide how much to buy."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from .behaviors import BehaviorEngine, BehaviorSpec, _action_to_quantity
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
    Runs as an asyncio coroutine so all agents decide concurrently.

    Decision modes (set once at construction):
      deterministic — first-match rule engine
      stochastic    — weighted random action table
      mixed         — deterministic first; stochastic fallback
      rational      — utility-maximising, no rules needed
      llm           — calls a local OpenAI-compatible endpoint

    Per-agent traits (risk_tolerance, planning_horizon_rounds) are sampled once
    at construction from the spec's ranges and carried across all rounds. They
    are passed into the LLM prompt so identically-observed agents with different
    traits produce different decisions without relying on persona labels.
    """

    def __init__(
        self,
        agent_id: int,
        spec: BehaviorSpec,
        market_name: str,
        seed: int | None = None,
        use_llm: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.agent_type = spec.agent_type_id
        self._market = market_name
        self._spec = spec
        self._use_llm = use_llm
        self.state = AgentState(agent_id=agent_id)

        _rng = random.Random(seed)

        # Persistent traits — sampled once, fixed for the agent's lifetime
        lo_rt, hi_rt = spec.risk_tolerance_range
        lo_ph, hi_ph = spec.planning_horizon_range
        self.traits: dict[str, Any] = {
            "risk_tolerance": _rng.uniform(lo_rt, hi_rt),
            "planning_horizon_rounds": _rng.randint(int(lo_ph), int(hi_ph)),
        }

        # Build the right engine(s) upfront
        if use_llm:
            from .llm_decider import LLMDeciderConfig
            lcfg = spec.llm_config or {}
            self._llm_config = LLMDeciderConfig(
                base_url=lcfg.get("base_url", "http://localhost:1234/v1"),
                model=lcfg.get("model", "qwen3-4b"),
                timeout_s=float(lcfg.get("timeout_s", 8.0)),
                max_retries=int(lcfg.get("max_retries", 2)),
            )
        else:
            self._llm_config = None

        if spec.mode == "rational":
            from .rational import RationalConfig, RationalEngine
            rcfg = spec.rational_config or {}
            self._rational: Any = RationalEngine(RationalConfig(
                elasticity=float(rcfg.get("elasticity", 0.4)),
                scarcity_threshold=float(rcfg.get("scarcity_threshold", 0.80)),
                base_quantity=spec.base_quantity,
            ))
        else:
            self._rational = None

        # BehaviorEngine for deterministic/stochastic/mixed (and as fallback basis
        # when llm_population_pct < 1.0 and this agent draws the non-LLM slot)
        if spec.mode in ("deterministic", "stochastic", "mixed"):
            self._engine: BehaviorEngine | None = BehaviorEngine(spec, rng=_rng)
        else:
            self._engine = None

    async def decide(self, world: WorldSnapshot) -> ConsumerDecision:
        obs = self._build_observation(world)

        if self._use_llm:
            from .llm_decider import decide_llm
            action, qty_factor, reasoning = await decide_llm(obs, self._llm_config)
            qty_demanded = _action_to_quantity(action, self._spec.base_quantity, qty_factor)

        elif self._rational is not None:
            await asyncio.sleep(0)  # yield — keep concurrency fair
            action, qty_demanded, reasoning = self._rational.decide(obs)

        elif self._engine is not None:
            await asyncio.sleep(0)
            action, qty_demanded, reasoning = self._engine.decide(obs)

        else:
            # Guard: mode=llm but use_llm=False with no engine (shouldn't happen normally)
            action, qty_demanded, reasoning = "no_change", self._spec.base_quantity, "no-engine"

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
            "stock_ratio": mkt["stock_ratio"],
            "stock": mkt["stock"],
            "pipeline_total": mkt["pipeline_total"],
            # Agent state
            "agent_savings": self.state.savings,
            "agent_inventory": self.state.inventory,
            "round": world.round,
            # Persistent traits (included for LLM prompt differentiation)
            **self.traits,
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
            # Determine whether this agent uses LLM.
            # For mode: llm, llm_config is always set (parse_behavior_specs guarantees it).
            # For other modes with an llm: block, llm_population_pct controls the split.
            use_llm = (
                spec.llm_config is not None
                and rng.random() < spec.llm_population_pct
            )
            agents.append(ConsumerAgent(
                agent_id=len(agents),
                spec=spec,
                market_name=market_name,
                seed=rng.randint(0, 2**31),
                use_llm=use_llm,
            ))

    return agents
