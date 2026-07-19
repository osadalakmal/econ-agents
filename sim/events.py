"""
Shocks — exogenous events applied to a market at a specific round.

Shock types:
  price_shock   — directly adjust market price by delta_pct
  stock_shock   — destroy or add stock by delta_pct
  cost_shock    — change production cost (affects supplier margins)

Consumer and supplier decisions live here too.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Shock:
    round: int
    shock_type: str        # "price_shock" | "stock_shock" | "cost_shock"
    market: str
    delta_pct: float
    description: str = ""


@dataclass
class ConsumerDecision:
    agent_id: int
    agent_type: str
    market: str
    action: str
    quantity_demanded: float   # units this agent wants to buy (>= 0)
    reasoning: str = ""


@dataclass
class SupplierDecisionRecord:
    """Thin record for reporting — mirrors SupplierDecision from supplier.py."""
    supplier_id: int
    supplier_type: str
    market: str
    action: str
    old_rate: float
    new_rate: float
    adjustment_pct: float
    reasoning: str


def parse_shocks(shock_cfgs: list[dict]) -> list[Shock]:
    return [
        Shock(
            round=s["round"],
            shock_type=s["type"],
            market=s["market"],
            delta_pct=s["delta_pct"],
            description=s.get("description", ""),
        )
        for s in shock_cfgs
    ]
