"""
Simulation — the closed-loop engine.

Six-phase round:

  1. SHOCK      — apply any scheduled exogenous shocks (floods, policy, etc.)
  2. OBSERVE    — snapshot world state (both consumers and suppliers read this)
  3. DECIDE     — consumers + suppliers decide concurrently (asyncio.gather)
  4. HARVEST    — production orders arriving this round land in stock
  5. CLEAR      — consumer demand settles against stock; price set by algorithm
  6. PLACE      — supplier orders enter production pipeline (arrive after lag)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from .agent import ConsumerAgent, build_consumer_pool
from .behaviors import parse_behavior_specs
from .events import ConsumerDecision, Shock, parse_shocks
from .market import ClearingResult, ProductionOrder
from .supplier import SupplierAgent, SupplierDecision, build_supplier_pool, parse_supplier_specs
from .world import World


@dataclass
class RoundResult:
    round_number: int
    shocks_applied: list[Shock]
    consumer_decisions: list[ConsumerDecision]
    supplier_decisions: list[SupplierDecision]
    clearing: ClearingResult
    duration_ms: float

    def summary(self) -> dict[str, Any]:
        cd = self.consumer_decisions
        total = len(cd)
        action_counts: dict[str, int] = {}
        type_breakdown: dict[str, dict[str, int]] = {}
        for d in cd:
            action_counts[d.action] = action_counts.get(d.action, 0) + 1
            tb = type_breakdown.setdefault(d.agent_type, {})
            tb[d.action] = tb.get(d.action, 0) + 1

        return {
            "round": self.round_number,
            "shocks": [s.description or s.shock_type for s in self.shocks_applied],
            "price_before": round(self.clearing.price_before, 4),
            "price_after": round(self.clearing.price_after, 4),
            "price_change_pct": round(
                (self.clearing.price_after - self.clearing.price_before)
                / max(self.clearing.price_before, 1e-9) * 100, 2
            ),
            "stock_before": round(self.clearing.stock_before, 1),
            "stock_after": round(self.clearing.stock_after, 1),
            "demand": round(self.clearing.demand, 1),
            "actual_consumption": round(self.clearing.actual_consumption, 1),
            "shortage": round(self.clearing.shortage, 1),
            "fill_rate": round(self.clearing.fill_rate * 100, 1),
            "arrived_supply": round(self.clearing.arrived_supply, 1),
            "total_agents": total,
            "consumer_actions": {
                k: {"count": v, "pct": round(v / max(total, 1) * 100, 1)}
                for k, v in sorted(action_counts.items())
            },
            "by_type": type_breakdown,
            "supplier_decisions": [
                {
                    "id": d.supplier_id,
                    "type": d.supplier_type,
                    "action": d.action,
                    "old_rate": round(d.old_rate, 1),
                    "new_rate": round(d.new_rate, 1),
                    "adj_pct": round(d.adjustment_pct, 1),
                    "reasoning": d.reasoning,
                }
                for d in self.supplier_decisions
            ],
            "duration_ms": round(self.duration_ms, 1),
        }


class Simulation:

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._market_name = list(config["markets"].keys())[0]  # primary market

        self._world = World(config["markets"])

        # Consumer agents
        consumer_cfg = config["consumers"]
        specs = parse_behavior_specs(consumer_cfg["agent_types"])
        self._consumers: list[ConsumerAgent] = build_consumer_pool(
            specs,
            population_size=consumer_cfg["size"],
            market_name=self._market_name,
            seed=config.get("seed", 42),
        )

        # Supplier agents
        supplier_specs = parse_supplier_specs(config.get("suppliers", []))
        self._suppliers: list[SupplierAgent] = build_supplier_pool(supplier_specs)

        # Scheduled shocks (indexed by round for fast lookup)
        all_shocks = parse_shocks(config.get("shocks", []))
        self._shocks_by_round: dict[int, list[Shock]] = {}
        for shock in all_shocks:
            self._shocks_by_round.setdefault(shock.round, []).append(shock)

        self._results: list[RoundResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, rounds: int) -> list[RoundResult]:
        for _ in range(rounds):
            result = await self._run_round()
            self._results.append(result)
        return self._results

    @property
    def consumers(self) -> list[ConsumerAgent]:
        return self._consumers

    @property
    def suppliers(self) -> list[SupplierAgent]:
        return self._suppliers

    @property
    def results(self) -> list[RoundResult]:
        return self._results

    # ------------------------------------------------------------------
    # Round loop
    # ------------------------------------------------------------------

    async def _run_round(self) -> RoundResult:
        t0 = time.monotonic()
        current_round = self._world.round
        market = self._world.get_market(self._market_name)

        # --- Phase 1: SHOCK ------------------------------------------------
        shocks_applied = self._apply_shocks(current_round, market)

        # --- Phase 2: OBSERVE ----------------------------------------------
        snapshot = self._world.snapshot()

        # --- Phase 3: DECIDE (concurrent) ----------------------------------
        consumer_coros = [c.decide(snapshot) for c in self._consumers]
        supplier_coros = [s.decide(market, current_round) for s in self._suppliers]

        all_decisions = await asyncio.gather(*consumer_coros, *supplier_coros)

        n_consumers = len(self._consumers)
        consumer_decisions: list[ConsumerDecision] = list(all_decisions[:n_consumers])
        supplier_decisions: list[SupplierDecision] = list(all_decisions[n_consumers:])

        # --- Phase 4: HARVEST pipeline -------------------------------------
        arrived = market.harvest_pipeline(current_round)

        # --- Phase 5: CLEAR + PRICE SET ------------------------------------
        total_demand = sum(d.quantity_demanded for d in consumer_decisions)
        clearing = market.clear(total_demand, current_round)
        clearing.arrived_supply = arrived

        # --- Phase 6: PLACE ORDERS -----------------------------------------
        for sd in supplier_decisions:
            lag = self._get_lag(sd.supplier_id)
            arrive = current_round + lag
            order = ProductionOrder(
                arrive_round=arrive,
                quantity=sd.new_rate,
                supplier_id=sd.supplier_id,
            )
            market.place_order(order)
            # Let supplier track its own committed pipeline
            supplier = next(s for s in self._suppliers if s.supplier_id == sd.supplier_id)
            supplier.record_placed_order(arrive, sd.new_rate)

        self._world.advance_round()
        elapsed = (time.monotonic() - t0) * 1000

        return RoundResult(
            round_number=current_round,
            shocks_applied=shocks_applied,
            consumer_decisions=consumer_decisions,
            supplier_decisions=supplier_decisions,
            clearing=clearing,
            duration_ms=elapsed,
        )

    def _apply_shocks(self, current_round: int, market: Any) -> list[Shock]:
        shocks = self._shocks_by_round.get(current_round, [])
        for shock in shocks:
            if shock.shock_type == "price_shock":
                market.apply_price_shock(shock.delta_pct)
            elif shock.shock_type == "stock_shock":
                market.apply_stock_shock(shock.delta_pct)
            elif shock.shock_type == "cost_shock":
                market.apply_cost_shock(shock.delta_pct)
                # Propagate cost change to all suppliers serving this market
                factor = 1 + shock.delta_pct / 100
                for s in self._suppliers:
                    if s.market_name == shock.market:
                        s._production_cost = round(s._production_cost * factor, 6)
        return shocks

    def _get_lag(self, supplier_id: int) -> int:
        for s in self._suppliers:
            if s.supplier_id == supplier_id:
                return s.production_lag
        return 1
