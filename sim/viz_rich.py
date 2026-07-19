"""Rich live display — updates in-place after each round."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from .simulation import RoundResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRICE_COLOR = {True: "bold red", False: "bold green"}   # True = rising


def _hbar(value: float, total: float, width: int = 20, color: str = "cyan") -> Text:
    filled = int(round(value / max(total, 1e-9) * width))
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    return Text(bar, style=color)


def _pct_bar(pct: float, width: int = 16) -> Text:
    """pct in [0, 100]."""
    filled = int(round(pct / 100 * width))
    return Text("█" * filled + "░" * (width - filled), style="dim cyan")


# ---------------------------------------------------------------------------
# Renderable builder
# ---------------------------------------------------------------------------

def _render(
    result: RoundResult,
    total_rounds: int,
    price_min: float,
    price_max: float,
    stock_max: float,
) -> Group:
    s = result.summary()

    # ── Title row ──────────────────────────────────────────────────────────
    shock_tag = ""
    if s["shocks"]:
        shock_tag = f"  [bold yellow]⚡ {' · '.join(s['shocks'])}[/bold yellow]"
    title = (
        f"[bold]Round {s['round'] + 1}/{total_rounds}[/bold]"
        f"  ·  seed 42{shock_tag}"
    )

    # ── Market panel ───────────────────────────────────────────────────────
    price = s["price_after"]
    rising = s["price_change_pct"] >= 0
    p_color = "red" if rising else "green"
    p_arrow = "▲" if rising else "▼"
    p_span = price_max - price_min if price_max > price_min else 1.0

    mkt = Table.grid(padding=(0, 1))
    mkt.add_column(width=10)
    mkt.add_column()
    mkt.add_column(width=6, justify="right")

    mkt.add_row(
        Text("Price", style="bold"),
        _hbar(price - price_min, p_span, color=p_color),
        Text(f"{price:.3f} {p_arrow}{abs(s['price_change_pct']):.1f}%", style=f"bold {p_color}"),
    )
    mkt.add_row(
        Text("Stock", style="bold"),
        _hbar(s["stock_after"], stock_max, color="blue"),
        Text(f"{s['stock_after']:.0f}", style="blue"),
    )

    shortage = s["shortage"]
    fill_style = "red bold" if shortage > 0 else "dim"
    mkt.add_row(
        Text("Fill", style="bold"),
        _hbar(s["fill_rate"], 100, color="green" if s["fill_rate"] >= 99 else "red"),
        Text(
            f"{s['fill_rate']:.0f}%" + (f"  ⚠ -{shortage:.0f}" if shortage > 0 else ""),
            style=fill_style,
        ),
    )

    market_panel = Panel(mkt, title="[cyan]Market[/cyan]", border_style="cyan")

    # ── Consumer panel ─────────────────────────────────────────────────────
    actions = s["consumer_actions"]
    con = Table.grid(padding=(0, 1))
    con.add_column(width=10)
    con.add_column(width=16)
    con.add_column(width=5, justify="right")
    colors = {"buy_more": "red", "buy_less": "green", "hold": "yellow", "no_change": "dim"}
    for action, info in sorted(actions.items(), key=lambda x: -x[1]["count"]):
        c = colors.get(action, "white")
        con.add_row(
            Text(action, style=c),
            _pct_bar(info["pct"]),
            Text(f"{info['pct']:.0f}%", style=c),
        )

    consumer_panel = Panel(con, title="[magenta]Consumers[/magenta]", border_style="magenta")

    # ── Supplier panel ─────────────────────────────────────────────────────
    sup = Table.grid(padding=(0, 1))
    sup.add_column(width=18)
    sup.add_column(width=14, justify="right")
    sup.add_column(width=7, justify="right")

    for sd in s["supplier_decisions"]:
        adj = sd["adj_pct"]
        adj_color = "red" if adj > 0 else ("green" if adj < 0 else "dim")
        adj_str = f"{adj:+.0f}%" if adj != 0 else "="
        sup.add_row(
            Text(f"[{sd['type']}] #{sd['id']}", style="bold"),
            Text(f"{sd['old_rate']:.0f} → {sd['new_rate']:.0f}", style="white"),
            Text(adj_str, style=adj_color),
        )

    supplier_panel = Panel(sup, title="[yellow]Suppliers[/yellow]", border_style="yellow")

    # ── Demand / supply row ────────────────────────────────────────────────
    stats = Text(
        f"demand={s['demand']:.0f}  consumed={s['actual_consumption']:.0f}"
        f"  arrived={s['arrived_supply']:.0f}  {result.duration_ms:.0f}ms/round",
        style="dim",
    )

    return Group(
        Panel(
            Group(
                Text(title, justify="center"),
                Text(""),
                Columns([market_panel, consumer_panel, supplier_panel]),
                stats,
            ),
            border_style="bright_black",
        )
    )


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class RichDisplay:
    """
    Wrap the simulation loop in a Rich Live display.

    Usage:
        display = RichDisplay(total_rounds=100, price_max=4.0, stock_max=15000)
        await sim.run(rounds, on_round=display.on_round)
    """

    def __init__(
        self,
        total_rounds: int,
        price_min: float = 0.2,
        price_max: float = 4.0,
        stock_max: float = 15000.0,
        delay: float = 0.15,
    ) -> None:
        self._total = total_rounds
        self._price_min = price_min
        self._price_max = price_max
        self._stock_max = stock_max
        self._delay = delay
        self._live = Live(refresh_per_second=20, screen=False)
        self._last: RoundResult | None = None

    def __enter__(self) -> "RichDisplay":
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self._live.__exit__(*args)

    async def on_round(self, result: RoundResult) -> None:
        self._last = result
        # Update price/stock bounds dynamically
        self._price_max = max(self._price_max, result.clearing.price_after * 1.05)
        self._stock_max = max(self._stock_max, result.clearing.stock_before * 1.1)

        self._live.update(
            _render(result, self._total, self._price_min, self._price_max, self._stock_max)
        )
        if self._delay > 0:
            await asyncio.sleep(self._delay)
