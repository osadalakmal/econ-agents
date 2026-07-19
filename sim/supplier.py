"""
Supplier agents — rational producers who respond to price signals.

Decision logic:
  - Observe current price and their own production cost → compute margin
  - Evaluate ordered rules (same AST engine as consumers)
  - Each rule specifies an adjustment_factor (signed % change in rate)
  - Change is clamped by max_ramp_up / max_ramp_down
  - New production rate takes effect next round; output arrives after lag rounds

Bankruptcy mechanic:
  - Suppliers track consecutive_loss_rounds when operating below production cost
  - After bankruptcy_threshold rounds below cost, the supplier goes bankrupt:
    rate forced to 0, no further orders placed
  - A bankrupt supplier restarts at restart_rate once margin >= restart_margin
  - min_production_rate prevents the multiplicative-zero trap: non-bankrupt
    suppliers cannot fall below this floor via normal rule evaluation alone
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
    bankrupt: bool = False
    consecutive_loss_rounds: int = 0


@dataclass
class SupplierState:
    supplier_id: int
    production_rate: float           # units/round (current committed rate)
    history: list[str] = field(default_factory=list)
    # Own orders not yet arrived: (arrive_round, quantity)
    pending_orders: list[tuple[int, float]] = field(default_factory=list)
    consecutive_loss_rounds: int = 0  # rounds with margin < 1.0 (below cost)
    bankrupt: bool = False            # True after bankruptcy_threshold losses


class SupplierAgent:
    """
    A rational producer. Each round it:
      1. Observes market price, its own cost, stock ratio
      2. If bankrupt: stays idle unless margin recovers past restart_margin
      3. Otherwise: evaluates rules → picks an adjustment
      4. Returns a SupplierDecision (simulation loop places the order)
    """

    def __init__(self, supplier_id: int, spec: "SupplierSpec") -> None:
        self.supplier_id = supplier_id
        self.supplier_type = spec.supplier_type_id
        self.market_name = spec.market
        self.production_lag = spec.production_lag
        self.max_ramp_up = spec.max_ramp_up
        self.max_ramp_down = spec.max_ramp_down
        self._production_cost = spec.production_cost
        self._bankruptcy_threshold = spec.bankruptcy_threshold
        self._restart_rate = spec.restart_rate
        self._restart_margin = spec.restart_margin
        self._min_production_rate = spec.min_production_rate
        self._rules = spec.rules
        self.state = SupplierState(
            supplier_id=supplier_id,
            production_rate=spec.initial_production_rate,
        )

    async def decide(
        self, market: Market, current_round: int
    ) -> SupplierDecision:
        await asyncio.sleep(0)

        obs = self._build_observation(market, current_round)
        old_rate = self.state.production_rate
        margin = obs["margin"]

        # --- Bankrupt path ---------------------------------------------------
        if self.state.bankrupt:
            if margin >= self._restart_margin:
                # Sufficient margin to justify re-entry
                self.state.bankrupt = False
                self.state.consecutive_loss_rounds = 0
                new_rate = self._restart_rate
                self.state.production_rate = new_rate
                action = "restart"
                reasoning = (
                    f"re-entering market: margin {margin:.2f} >= "
                    f"restart threshold {self._restart_margin:.2f}"
                )
            else:
                # Still bankrupt — stay idle
                new_rate = 0.0
                action = "bankrupt"
                reasoning = (
                    f"bankrupt: margin {margin:.2f} below restart "
                    f"threshold {self._restart_margin:.2f}"
                )
            adj_pct = 0.0
        else:
            # --- Normal operation path ---------------------------------------
            action, raw_adj, reasoning = self._evaluate_rules(obs)
            adj = max(-self.max_ramp_down, min(self.max_ramp_up, raw_adj))

            # Multiplicative update with non-zero floor to prevent zero-trap
            new_rate = old_rate * (1.0 + adj)
            new_rate = max(self._min_production_rate, new_rate)
            adj_pct = adj * 100

            # Track consecutive loss rounds (margin < 1.0 = below cost)
            if margin < 1.0:
                self.state.consecutive_loss_rounds += 1
                if self.state.consecutive_loss_rounds >= self._bankruptcy_threshold:
                    self.state.bankrupt = True
                    new_rate = 0.0
                    action = "bankrupt"
                    reasoning = (
                        f"capital exhausted after {self.state.consecutive_loss_rounds} "
                        f"rounds below cost (margin={margin:.2f}) — exiting market"
                    )
                    adj_pct = -100.0
            else:
                self.state.consecutive_loss_rounds = 0

        self.state.production_rate = new_rate

        self.state.history.append(
            f"R{current_round}: {action} rate {old_rate:.1f}→{new_rate:.1f} "
            f"margin={margin:.2f} loss_streak={self.state.consecutive_loss_rounds}"
        )

        return SupplierDecision(
            supplier_id=self.supplier_id,
            supplier_type=self.supplier_type,
            market=self.market_name,
            action=action,
            old_rate=old_rate,
            new_rate=new_rate,
            adjustment_pct=adj_pct,
            reasoning=reasoning,
            bankrupt=self.state.bankrupt,
            consecutive_loss_rounds=self.state.consecutive_loss_rounds,
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
            "consecutive_loss_rounds": self.state.consecutive_loss_rounds,
            "bankrupt": float(self.state.bankrupt),  # float for rule comparisons
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
    bankruptcy_threshold: int   # consecutive loss rounds before bankruptcy
    restart_rate: float         # production rate when re-entering market
    restart_margin: float       # minimum margin ratio to justify restart
    min_production_rate: float  # floor preventing multiplicative zero-trap
    rules: list[dict]


def parse_supplier_specs(supplier_cfgs: list[dict]) -> list[SupplierSpec]:
    specs = []
    for cfg in supplier_cfgs:
        initial_rate = cfg["initial_production_rate"]
        specs.append(SupplierSpec(
            supplier_type_id=cfg["id"],
            market=cfg["market"],
            count=cfg["count"],
            production_cost=cfg["production_cost"],
            initial_production_rate=initial_rate,
            production_lag=cfg["production_lag"],
            max_ramp_up=cfg.get("max_ramp_up", 0.20),
            max_ramp_down=cfg.get("max_ramp_down", 0.30),
            bankruptcy_threshold=cfg.get("bankruptcy_threshold", 6),
            restart_rate=cfg.get("restart_rate", initial_rate * 0.1),
            restart_margin=cfg.get("restart_margin", 1.40),
            min_production_rate=cfg.get("min_production_rate", initial_rate * 0.05),
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
