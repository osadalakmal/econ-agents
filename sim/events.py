"""Economic stimuli that drive simulation rounds."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stimulus:
    """A change applied to the world at the start of a round."""
    name: str
    commodity: str
    delta_pct: float = 0.0          # price change %
    supply_delta_pct: float = 0.0   # supply change %
    metadata: dict = field(default_factory=dict)

    def observation(self) -> dict:
        """What agents can observe about this stimulus."""
        return {
            "stimulus_name": self.name,
            "commodity": self.commodity,
            "price_change_pct": self.delta_pct,
            "supply_change_pct": self.supply_delta_pct,
            **self.metadata,
        }


@dataclass
class Decision:
    """What a single agent decided to do this round."""
    agent_id: int
    agent_type: str
    commodity: str
    action: str           # e.g. "buy_more", "buy_less", "hold", "no_change"
    quantity_delta: float # positive = more demand, negative = less
    reasoning: str = ""   # for deterministic agents: which rule fired
