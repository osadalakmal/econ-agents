"""Entry point — load config and run the closed-loop simulation."""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

import yaml

from sim.reporting import (
    print_round,
    print_simulation_header,
    print_simulation_summary,
    to_json,
)
from sim.simulation import Simulation


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run(config_path: str, output_json: str | None) -> None:
    config = load_config(config_path)
    rounds = config.get("rounds", 10)

    print_simulation_header(config)

    # Agent distribution
    consumer_cfg = config["consumers"]
    total = sum(t["proportion"] for t in consumer_cfg["agent_types"])
    print("\nConsumer distribution:")
    for t in consumer_cfg["agent_types"]:
        count = round(consumer_cfg["size"] * t["proportion"] / total)
        print(f"  {t['id']:<22} {count:>4}  ({t['mode']})")

    print("\nSupplier pool:")
    for s in config.get("suppliers", []):
        total_output = s["count"] * s["initial_production_rate"]
        print(
            f"  {s['id']:<20} ×{s['count']}  "
            f"rate={s['initial_production_rate']}/round  "
            f"lag={s['production_lag']}r  "
            f"total={total_output}/round"
        )

    if config.get("shocks"):
        print("\nScheduled shocks:")
        for sh in config["shocks"]:
            print(f"  round {sh['round']:>2}: {sh.get('description', sh['type'])}")

    sim = Simulation(config)
    results = await sim.run(rounds)

    for result in results:
        print_round(result)

    print_simulation_summary(results)

    if output_json:
        Path(output_json).write_text(to_json(results))
        print(f"\nDetailed results written to: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop economic agent simulation")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/rice_price.yaml",
        help="Path to scenario YAML config",
    )
    parser.add_argument("--output-json", metavar="FILE")
    parser.add_argument("--rounds", type=int, help="Override rounds from config")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    if args.rounds:
        config["rounds"] = args.rounds

    asyncio.run(run(args.config, args.output_json))


if __name__ == "__main__":
    main()
