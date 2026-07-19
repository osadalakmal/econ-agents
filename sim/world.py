"""Shared world state — the single source of truth agents observe."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Commodity:
    name: str
    price: float
    supply: float   # arbitrary units
    demand: float   # aggregate demand this round


@dataclass
class WorldState:
    """Immutable snapshot passed to agents each round."""
    round: int
    commodities: dict[str, Commodity]
    history: list[dict[str, Any]] = field(default_factory=list)

    def price(self, commodity: str) -> float:
        return self.commodities[commodity].price

    def snapshot(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "commodities": {
                k: {"price": v.price, "supply": v.supply, "demand": v.demand}
                for k, v in self.commodities.items()
            },
        }


class World:
    """Mutable world — only the simulation loop writes to this."""

    def __init__(self, commodities: dict[str, dict]) -> None:
        self._commodities: dict[str, Commodity] = {
            name: Commodity(name=name, **vals)
            for name, vals in commodities.items()
        }
        self._round = 0
        self._history: list[dict[str, Any]] = []

    def apply_stimulus(self, stimulus: "Stimulus") -> None:  # noqa: F821
        com = self._commodities[stimulus.commodity]
        com.price = round(com.price * (1 + stimulus.delta_pct / 100), 4)
        com.supply = round(com.supply * (1 + stimulus.supply_delta_pct / 100), 4)

    def apply_decisions(self, decisions: list["Decision"]) -> None:  # noqa: F821
        # Aggregate demand from all agent decisions
        demand_delta: dict[str, float] = {}
        for d in decisions:
            delta = demand_delta.get(d.commodity, 0.0)
            demand_delta[d.commodity] = delta + d.quantity_delta

        for commodity, delta in demand_delta.items():
            com = self._commodities[commodity]
            com.demand = round(com.demand + delta, 4)

    def advance_round(self) -> None:
        snap = self.snapshot()
        self._history.append(snap)
        # Reset per-round demand accumulators
        for com in self._commodities.values():
            com.demand = 0.0
        self._round += 1

    def snapshot(self) -> WorldState:
        return WorldState(
            round=self._round,
            commodities=copy.deepcopy(self._commodities),
            history=list(self._history),
        )
