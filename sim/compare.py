"""Paired-run comparison — same market/suppliers/shocks, two consumer populations.

Usage:
    from sim.compare import run_comparison
    result = await run_comparison(config, "rational", "hoarder_mix")

The config must contain a top-level `presets` key mapping preset names to
`consumers` blocks (same schema as `config["consumers"]`).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .simulation import RoundResult, Simulation


@dataclass
class ComparisonResult:
    preset_a: str
    preset_b: str
    results_a: list[RoundResult]
    results_b: list[RoundResult]


async def run_comparison(
    config: dict[str, Any],
    preset_a: str,
    preset_b: str,
) -> ComparisonResult:
    """
    Run the simulation twice with identical market/supplier/shock config,
    substituting only the consumers block from the named presets.
    Both runs use the same seed so only the consumer population differs.
    """
    presets = config.get("presets", {})
    missing = [p for p in (preset_a, preset_b) if p not in presets]
    if missing:
        available = list(presets) or ["(none defined)"]
        raise ValueError(
            f"Preset(s) not found in config: {missing}. "
            f"Available presets: {available}"
        )

    rounds = config.get("rounds", 10)

    cfg_a = copy.deepcopy(config)
    cfg_a["consumers"] = copy.deepcopy(presets[preset_a])

    cfg_b = copy.deepcopy(config)
    cfg_b["consumers"] = copy.deepcopy(presets[preset_b])

    sim_a = Simulation(cfg_a)
    sim_b = Simulation(cfg_b)

    await sim_a.run(rounds)
    await sim_b.run(rounds)

    return ComparisonResult(
        preset_a=preset_a,
        preset_b=preset_b,
        results_a=sim_a.results,
        results_b=sim_b.results,
    )
