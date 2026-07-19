"""
Market — holds stock, a production pipeline, and a price-setting algorithm.

One Market per commodity. The simulation loop calls methods in this order
each round:
  1. harvest_pipeline(round)  → stock increases from arriving orders
  2. clear(demand)            → stock decreases; shortages recorded
  3. update_price()           → algorithm sets new price
  4. place_order(order)       → supplier orders queued for future rounds
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Production pipeline
# ---------------------------------------------------------------------------

@dataclass
class ProductionOrder:
    arrive_round: int
    quantity: float
    supplier_id: int


# ---------------------------------------------------------------------------
# Price algorithms
# ---------------------------------------------------------------------------

class PriceAlgorithm:
    def update(self, market: "Market") -> float:
        raise NotImplementedError


class StockBasedPricing(PriceAlgorithm):
    """
    Price responds to the ratio of current stock to a target buffer.

    target_stock = avg_recent_demand * target_stock_days

    stock_ratio = current_stock / target_stock
      < 1.0  → shortage  → price rises
      > 1.0  → surplus   → price falls

    Δprice = −elasticity × ln(stock_ratio)   (log keeps it symmetric)
    """

    def __init__(
        self,
        target_stock_days: float,
        elasticity: float,
        min_price: float,
        max_price: float,
        demand_window: int = 3,
    ) -> None:
        self.target_days = target_stock_days
        self.elasticity = elasticity
        self.min_price = min_price
        self.max_price = max_price
        self.demand_window = demand_window

    @property
    def target_stock_days(self) -> float:
        return self.target_days

    def update(self, market: "Market") -> float:
        recent = list(market.demand_history[-self.demand_window :])
        if not recent:
            return market.price
        avg_demand = sum(recent) / len(recent)
        if avg_demand <= 0:
            return market.price

        target_stock = avg_demand * self.target_days
        stock_ratio = market.stock / max(target_stock, 1e-9)
        delta = -self.elasticity * math.log(max(stock_ratio, 1e-6))
        new_price = market.price * (1.0 + delta)
        return max(self.min_price, min(self.max_price, round(new_price, 6)))


def build_price_algorithm(cfg: dict) -> PriceAlgorithm:
    algo_type = cfg.get("type", "stock_based")
    if algo_type == "stock_based":
        return StockBasedPricing(
            target_stock_days=cfg.get("target_stock_days", 14),
            elasticity=cfg.get("elasticity", 0.3),
            min_price=cfg.get("min_price", 0.01),
            max_price=cfg.get("max_price", 100.0),
            demand_window=cfg.get("demand_window", 3),
        )
    raise ValueError(f"Unknown price algorithm: {algo_type}")


# ---------------------------------------------------------------------------
# ClearingResult
# ---------------------------------------------------------------------------

@dataclass
class ClearingResult:
    round: int
    demand: float
    actual_consumption: float
    shortage: float
    arrived_supply: float
    stock_before: float
    stock_after: float
    price_before: float
    price_after: float

    @property
    def fill_rate(self) -> float:
        if self.demand <= 0:
            return 1.0
        return min(1.0, self.actual_consumption / self.demand)


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------

class Market:
    def __init__(self, name: str, cfg: dict) -> None:
        self.name = name
        self.price: float = cfg["initial_price"]
        self.stock: float = cfg["initial_stock"]
        self.production_cost: float = cfg.get("production_cost", 0.0)

        self._price_algo = build_price_algorithm(
            cfg.get("price_algorithm", {})
        )
        self._pipeline: deque[ProductionOrder] = deque()

        # History (indexed by round)
        self.price_history: list[float] = [self.price]
        self.stock_history: list[float] = [self.stock]
        self.demand_history: list[float] = []
        self.clearing_history: list[ClearingResult] = []

    # ------------------------------------------------------------------
    # Called by simulation loop
    # ------------------------------------------------------------------

    def harvest_pipeline(self, current_round: int) -> float:
        """Pull orders whose arrive_round <= current_round into stock."""
        arrived = 0.0
        remaining: deque[ProductionOrder] = deque()
        for order in self._pipeline:
            if order.arrive_round <= current_round:
                arrived += order.quantity
            else:
                remaining.append(order)
        self._pipeline = remaining
        self.stock += arrived
        return arrived

    def clear(self, total_demand: float, current_round: int) -> ClearingResult:
        """Settle demand against stock; record shortage."""
        stock_before = self.stock
        price_before = self.price
        actual = min(total_demand, self.stock)
        shortage = max(0.0, total_demand - self.stock)
        self.stock = max(0.0, self.stock - actual)

        self.demand_history.append(total_demand)
        self.stock_history.append(self.stock)

        # Price updated after clearing so algo sees new stock level
        new_price = self._price_algo.update(self)
        self.price = new_price
        self.price_history.append(new_price)

        result = ClearingResult(
            round=current_round,
            demand=total_demand,
            actual_consumption=actual,
            shortage=shortage,
            arrived_supply=0.0,  # caller fills this in
            stock_before=stock_before,
            stock_after=self.stock,
            price_before=price_before,
            price_after=new_price,
        )
        self.clearing_history.append(result)
        return result

    def place_order(self, order: ProductionOrder) -> None:
        self._pipeline.append(order)

    # ------------------------------------------------------------------
    # Shock application (external events)
    # ------------------------------------------------------------------

    def apply_price_shock(self, delta_pct: float) -> None:
        self.price = round(self.price * (1 + delta_pct / 100), 6)
        self.price_history[-1] = self.price  # overwrite last entry

    def apply_stock_shock(self, delta_pct: float) -> None:
        self.stock = max(0.0, self.stock * (1 + delta_pct / 100))

    def apply_cost_shock(self, delta_pct: float) -> None:
        self.production_cost = round(
            self.production_cost * (1 + delta_pct / 100), 6
        )

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def price_change_pct(self) -> float:
        if len(self.price_history) < 2:
            return 0.0
        prev = self.price_history[-2]
        if prev == 0:
            return 0.0
        return (self.price - prev) / prev * 100

    def stock_ratio(self) -> float:
        """stock / target_stock (< 1 = shortage signal)."""
        algo = self._price_algo
        if not isinstance(algo, StockBasedPricing):
            return 1.0
        recent = self.demand_history[-algo.demand_window :] if self.demand_history else []
        if not recent:
            return 1.0
        avg_demand = sum(recent) / len(recent)
        target = avg_demand * algo.target_stock_days
        return self.stock / max(target, 1e-9)

    def pipeline_total(self) -> float:
        return sum(o.quantity for o in self._pipeline)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "price": round(self.price, 4),
            "stock": round(self.stock, 2),
            "production_cost": self.production_cost,
            "stock_ratio": round(self.stock_ratio(), 3),
            "price_change_pct": round(self.price_change_pct(), 2),
            "pipeline_total": round(self.pipeline_total(), 2),
        }
