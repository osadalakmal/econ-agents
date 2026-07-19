"""Entry point — load config and run the closed-loop simulation."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from sim.reporting import (
    DISCLAIMER,
    print_comparison,
    print_round,
    print_simulation_header,
    print_simulation_summary,
    to_json,
    to_json_comparison,
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
        mode_label = t["mode"]
        llm_pct = t.get("llm_population_pct")
        if llm_pct is not None and t["mode"] != "llm":
            mode_label = f"{t['mode']} + {llm_pct*100:.0f}% llm"
        print(f"  {t['id']:<22} {count:>4}  ({mode_label})")

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

    # Print disclaimer when any LLM agents were in the run
    has_llm = any(
        t.get("mode") == "llm" or t.get("llm_population_pct", 0) > 0
        for t in config["consumers"]["agent_types"]
    )
    if has_llm:
        print(f"\n  {DISCLAIMER}")

    if output_json:
        Path(output_json).write_text(to_json(results))
        print(f"\nDetailed results written to: {output_json}")

    if export_gif:
        try:
            from sim.viz_mpl import export_gif as do_export
            do_export(results, export_gif)
        except ImportError:
            print("matplotlib or pillow not installed — skipping GIF export", file=sys.stderr)


async def run_compare(
    config: dict,
    preset_a: str,
    preset_b: str,
    output_json: str | None,
) -> None:
    from sim.compare import run_comparison

    print(f"\nRunning paired comparison: [{preset_a}] vs [{preset_b}]")
    print(f"Market/supplier/shock config held constant; only consumers differ.\n")

    cmp = await run_comparison(config, preset_a, preset_b)
    print_comparison(cmp)

    if output_json:
        Path(output_json).write_text(to_json_comparison(cmp))
        print(f"\nComparison JSON written to: {output_json}")


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
    parser.add_argument(
        "--compare",
        metavar="A:B",
        help=(
            "Run paired comparison between two consumer presets defined in the config "
            "(e.g. --compare rational:llm_driven). Requires a top-level `presets:` key."
        ),
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    if args.rounds:
        config["rounds"] = args.rounds

    if args.compare:
        parts = args.compare.split(":", 1)
        if len(parts) != 2:
            print("--compare expects format A:B (e.g. rational:llm_driven)", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_compare(config, parts[0], parts[1], args.output_json))
    else:
        asyncio.run(run(config, args.output_json, args.export_gif, args.no_viz, args.delay))


if __name__ == "__main__":
    main()
