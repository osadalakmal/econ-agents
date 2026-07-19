"""Rational baseline agent — utility-maximising consumer under full information.

This is the counterfactual arm for comparison runs. It is deliberately simple
and transparent so its assumptions are visible rather than buried in rule lists.

Behaviour:
  - Adjusts quantity proportionally to sustained price changes (> 5% threshold),
    scaled by `elasticity`. Ignores sub-5% fluctuations (noise).
  - Responds to genuine scarcity (stock_ratio below `scarcity_threshold`) with a
    modest buy-more signal, capped to avoid hoarding.
  - Never amplifies short-term panic signals independently of actual stock state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RationalConfig:
    elasticity: float = 0.4            # quantity reduction per % price rise
    scarcity_threshold: float = 0.80   # stock_ratio below this triggers mild buy_more
    base_quantity: float = 1.0


class RationalEngine:

    def __init__(self, config: RationalConfig) -> None:
        self._cfg = config

    def decide(self, obs: dict[str, Any]) -> tuple[str, float, str]:
        """Return (action, quantity_demanded, reasoning)."""
        price_chg = obs.get("price_change_pct", 0.0)
        stock_ratio = obs.get("stock_ratio", 1.0)
        base = self._cfg.base_quantity

        # Genuine scarcity — respond only when stock is truly tight, cap the boost
        if stock_ratio < self._cfg.scarcity_threshold:
            scarcity_boost = min(
                self._cfg.scarcity_threshold / max(stock_ratio, 0.05),
                1.8,
            )
        else:
            scarcity_boost = 1.0

        # Price elasticity — ignore noise (< 5% moves), respond to sustained shifts
        if abs(price_chg) > 5.0:
            price_factor = 1.0 - self._cfg.elasticity * (price_chg / 100.0)
            price_factor = max(0.15, min(2.0, price_factor))
        else:
            price_factor = 1.0

        combined = scarcity_boost * price_factor
        qty = base * combined

        if combined > 1.05:
            return (
                "buy_more",
                qty,
                f"rational: stock_ratio={stock_ratio:.2f}, price_chg={price_chg:+.1f}%",
            )
        if combined < 0.95:
            return (
                "buy_less",
                qty,
                f"rational: price_factor={price_factor:.2f}, price_chg={price_chg:+.1f}%",
            )
        return "no_change", base, "rational: near-equilibrium"
