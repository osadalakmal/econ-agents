"""World — container for all markets; produces snapshots for agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .market import Market


@dataclass
class WorldSnapshot:
    """Immutable view of the world passed to agents each round."""
    round: int
    markets: dict[str, dict[str, Any]]

    def market(self, name: str) -> dict[str, Any]:
        return self.markets[name]


class World:
    def __init__(self, market_configs: dict[str, dict]) -> None:
        self._markets: dict[str, Market] = {
            name: Market(name=name, cfg=cfg)
            for name, cfg in market_configs.items()
        }
        self._round = 0

    @property
    def round(self) -> int:
        return self._round

    def get_market(self, name: str) -> Market:
        return self._markets[name]

    @property
    def markets(self) -> dict[str, Market]:
        return self._markets

    def snapshot(self) -> WorldSnapshot:
        return WorldSnapshot(
            round=self._round,
            markets={name: m.snapshot() for name, m in self._markets.items()},
        )

    def advance_round(self) -> None:
        self._round += 1
