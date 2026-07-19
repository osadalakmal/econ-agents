"""
Supplier agents — rational producers who respond to price signals.

Decision logic:
  - Observe current price and their own production cost → compute margin
  - Evaluate ordered rules (same AST engine as consumers)
  - Each rule specifies an adjustment_factor (signed % change in rate)
  - Change is clamped by max_ramp_up / max_ramp_down
  - New production rate takes effect next round; output arrives after lag rounds
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from .behaviors import eval_condition
from .market import Market, ProductionOrder


@dataclass
class SupplierDecision:
    supplier_id: int
    supplier_type: str
    market: str
    action: str
    old_rate: float
    new_rate: float
    adjustment_pct: float
    reasoning: str


@dataclass
class SupplierState:
    supplier_id: int
    production_rate: float  # units/round (current committed rate)
    history: list[str] = field(default_factory=list)
    # Own orders not yet arrived: (arrive_round, quantity)
    pending_orders: list[tuple[int, float]] = field(default_factory=list)


class SupplierAgent:
    """
    A rational producer. Each round it:
      1. Observes market price, its own cost, stock ratio
      2. Evaluates rules → picks an adjustment
      3. Returns a SupplierDecision (simulation loop places the order)
    """

    def __init__(self, supplier_id: int, spec: "SupplierSpec") -> None:
        self.supplier_id = supplier_id
        self.supplier_type = spec.supplier_type_id
        self.market_name = spec.market
        self.production_lag = spec.production_lag
        self.max_ramp_up = spec.max_ramp_up
        self.max_ramp_down = spec.max_ramp_down
        self.state = SupplierState(
            supplier_id=supplier_id,
            production_rate=spec.initial_production_rate,
        )
        self._rules = spec.rules
        self._production_cost = spec.production_cost

    async def decide(
        self, market: Market, current_round: int
    ) -> SupplierDecision:
        await asyncio.sleep(0)

        obs = self._build_observation(market, current_round)
        action, raw_adj, reasoning = self._evaluate_rules(obs)

        # Clamp adjustment to ramp limits
        adj = max(-self.max_ramp_down, min(self.max_ramp_up, raw_adj))
        old_rate = self.state.production_rate
        new_rate = max(0.0, old_rate * (1.0 + adj))
        self.state.production_rate = new_rate

        self.state.history.append(
            f"R{current_round}: {action} (adj={adj:+.1%}) rate {old_rate:.1f}→{new_rate:.1f}"
        )

        return SupplierDecision(
            supplier_id=self.supplier_id,
            supplier_type=self.supplier_type,
            market=self.market_name,
            action=action,
            old_rate=old_rate,
            new_rate=new_rate,
            adjustment_pct=adj * 100,
            reasoning=reasoning,
        )

    def record_placed_order(self, arrive_round: int, quantity: float) -> None:
        """Called by simulation after placing this supplier's order."""
        self.state.pending_orders.append((arrive_round, quantity))

    def _expire_arrived_orders(self, current_round: int) -> None:
        self.state.pending_orders = [
            (r, q) for r, q in self.state.pending_orders if r > current_round
        ]

    def _build_observation(self, market: Market, current_round: int) -> dict[str, Any]:
        self._expire_arrived_orders(current_round)
        cost = self._production_cost
        price = market.price
        rate = self.state.production_rate
        own_pipeline = sum(q for _, q in self.state.pending_orders)
        own_pipeline_rounds = own_pipeline / max(rate, 1e-9)
        return {
            "price": price,
            "production_cost": cost,
            "margin": price / max(cost, 1e-9),
            "price_change_pct": market.price_change_pct(),
            "stock_ratio": market.stock_ratio(),
            "stock": market.stock,
            "market_pipeline_total": market.pipeline_total(),
            "own_pipeline_total": own_pipeline,
            "own_pipeline_rounds": own_pipeline_rounds,
            "current_production_rate": rate,
            "round": current_round,
        }

    def _evaluate_rules(self, obs: dict[str, Any]) -> tuple[str, float, str]:
        for rule in self._rules:
            try:
                if eval_condition(rule["condition"], obs):
                    return (
                        rule["action"],
                        rule.get("adjustment_factor", 0.0),
                        rule.get("label", rule["condition"]),
                    )
            except Exception:
                continue
        return "maintain", 0.0, "no rule matched — maintaining rate"


# ---------------------------------------------------------------------------
# Spec + pool builder
# ---------------------------------------------------------------------------

@dataclass
class SupplierSpec:
    supplier_type_id: str
    market: str
    count: int
    production_cost: float
    initial_production_rate: float
    production_lag: int
    max_ramp_up: float
    max_ramp_down: float
    rules: list[dict]


def parse_supplier_specs(supplier_cfgs: list[dict]) -> list[SupplierSpec]:
    specs = []
    for cfg in supplier_cfgs:
        specs.append(SupplierSpec(
            supplier_type_id=cfg["id"],
            market=cfg["market"],
            count=cfg["count"],
            production_cost=cfg["production_cost"],
            initial_production_rate=cfg["initial_production_rate"],
            production_lag=cfg["production_lag"],
            max_ramp_up=cfg.get("max_ramp_up", 0.20),
            max_ramp_down=cfg.get("max_ramp_down", 0.30),
            rules=cfg.get("rules", []),
        ))
    return specs


def build_supplier_pool(specs: list[SupplierSpec]) -> list[SupplierAgent]:
    agents: list[SupplierAgent] = []
    supplier_id = 0
    for spec in specs:
        for _ in range(spec.count):
            agents.append(SupplierAgent(supplier_id=supplier_id, spec=spec))
            supplier_id += 1
    return agents
