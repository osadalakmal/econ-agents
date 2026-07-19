"""Entry point — load config and run the closed-loop simulation."""
from __future__ import annotations

import argparse
import asyncio
import sys
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


async def run(
    config: dict,
    output_json: str | None,
    export_gif: str | None,
    no_viz: bool,
    delay: float,
) -> None:
    rounds = config.get("rounds", 10)
    mkt_cfg = list(config["markets"].values())[0]

    print_simulation_header(config)

    consumer_cfg = config["consumers"]
    total_prop = sum(t["proportion"] for t in consumer_cfg["agent_types"])
    print("\nConsumer distribution:")
    for t in consumer_cfg["agent_types"]:
        count = round(consumer_cfg["size"] * t["proportion"] / total_prop)
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
            print(f"  round {sh['round']:>3}: {sh.get('description', sh['type'])}")

    sim = Simulation(config)

    algo = mkt_cfg.get("price_algorithm", {})
    price_min = algo.get("min_price", 0.2)
    price_max = algo.get("max_price", 4.0)

    if not no_viz:
        try:
            from sim.viz_rich import RichDisplay
            print()
            with RichDisplay(
                total_rounds=rounds,
                price_min=price_min,
                price_max=price_max,
                stock_max=mkt_cfg.get("initial_stock", 5000) * 3,
                delay=delay,
            ) as display:
                await sim.run(rounds, on_round=display.on_round)
        except ImportError:
            print("[rich not installed — running without live display]\n")
            no_viz = True

    if no_viz:
        async def plain_print(result):
            print_round(result)

        await sim.run(rounds, on_round=plain_print)

    results = sim.results
    print_simulation_summary(results)

    if output_json:
        Path(output_json).write_text(to_json(results))
        print(f"\nDetailed results written to: {output_json}")

    if export_gif:
        try:
            from sim.viz_mpl import export_gif as do_export
            do_export(results, export_gif)
        except ImportError:
            print("matplotlib or pillow not installed — skipping GIF export", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop economic agent simulation")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/rice_price.yaml",
        help="Path to scenario YAML config",
    )
    parser.add_argument("--rounds", type=int, help="Override rounds from config")
    parser.add_argument("--output-json", metavar="FILE", help="Write full decision log to JSON")
    parser.add_argument("--export-gif", metavar="FILE", help="Save animated GIF (requires matplotlib + pillow)")
    parser.add_argument("--no-viz", action="store_true", help="Disable Rich live display")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        metavar="SECS",
        help="Pause between rounds in live display (default: 0.15)",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    if args.rounds:
        config["rounds"] = args.rounds

    asyncio.run(run(config, args.output_json, args.export_gif, args.no_viz, args.delay))


if __name__ == "__main__":
    main()
