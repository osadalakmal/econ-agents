"""Entry point — load config and run simulation."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

from sim.events import Stimulus
from sim.reporting import print_round, print_simulation_summary, to_json
from sim.simulation import Simulation


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_stimuli(config: dict) -> list[Stimulus]:
    return [
        Stimulus(
            name=s["name"],
            commodity=s["commodity"],
            delta_pct=s.get("delta_pct", 0.0),
            supply_delta_pct=s.get("supply_delta_pct", 0.0),
            metadata=s.get("metadata", {}),
        )
        for s in config.get("stimuli", [])
    ]


async def run(config_path: str, output_json: str | None) -> None:
    config = load_config(config_path)
    stimuli = build_stimuli(config)

    print(f"Loading scenario: {config_path}")
    print(f"Population: {config['population']['size']} agents")
    print(f"Rounds: {len(stimuli)}")

    sim = Simulation(config)

    # Print agent type breakdown
    from collections import Counter
    type_counts = Counter(a.agent_type for a in sim.agents)
    print("\nAgent distribution:")
    for atype, count in sorted(type_counts.items()):
        print(f"  {atype:<22} {count:>4} agents  ({count/len(sim.agents)*100:.1f}%)")

    results = await sim.run(stimuli)

    for result in results:
        print_round(result)

    print_simulation_summary(results)

    if output_json:
        Path(output_json).write_text(to_json(results))
        print(f"\nDetailed results written to: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Economic agent simulation")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/rice_price.yaml",
        help="Path to scenario YAML config",
    )
    parser.add_argument(
        "--output-json",
        metavar="FILE",
        help="Write full decision log to JSON file",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args.config, args.output_json))


if __name__ == "__main__":
    main()
