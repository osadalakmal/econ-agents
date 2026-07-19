"""Matplotlib animated GIF export."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import RoundResult


def export_gif(
    results: list["RoundResult"],
    path: str,
    fps: int = 8,
    dpi: int = 120,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [r.summary() for r in results]
    n = len(rounds)

    # Pre-compute series
    prices       = [r["price_after"] for r in rounds]
    stocks       = [r["stock_after"] for r in rounds]
    demands      = [r["demand"] for r in rounds]
    shortages    = [r["shortage"] for r in rounds]
    shock_rounds = [i for i, r in enumerate(rounds) if r["shocks"]]

    # Per-type action series
    all_types    = sorted({t for r in rounds for t in r["by_type"]})
    action_names = ["buy_more", "buy_less", "hold", "no_change"]
    action_colors = ["#e74c3c", "#2ecc71", "#f39c12", "#95a5a6"]

    # Supplier total production rate per round
    sup_rates: dict[str, list[float]] = {}
    for r in rounds:
        for sd in r["supplier_decisions"]:
            key = f"{sd['type']}#{sd['id']}"
            sup_rates.setdefault(key, []).append(sd["new_rate"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor="#1a1a2e")
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    panel_bg = "#16213e"
    text_color = "#e0e0e0"
    grid_color = "#2a2a4a"

    for ax in axes.flat:
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=text_color, labelsize=8)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        ax.grid(color=grid_color, linewidth=0.5)

    ax_price, ax_stock, ax_demand, ax_supply = axes.flat

    # ── Subplot 1: Price ────────────────────────────────────────────────────
    ax_price.set_title("Market Price", fontsize=10, pad=4)
    ax_price.set_xlim(0, n - 1)
    ax_price.set_ylim(0, max(prices) * 1.15)
    ax_price.set_xlabel("Round", fontsize=8)
    price_line, = ax_price.plot([], [], color="#e74c3c", linewidth=1.8, label="price")
    price_dot,  = ax_price.plot([], [], "o", color="#e74c3c", markersize=5)
    vline_p = ax_price.axvline(0, color="#ffffff", linewidth=0.6, alpha=0.4, linestyle="--")
    for sr in shock_rounds:
        ax_price.axvline(sr, color="#f39c12", linewidth=0.8, alpha=0.5, linestyle=":")

    # ── Subplot 2: Stock ────────────────────────────────────────────────────
    ax_stock.set_title("Stock Level", fontsize=10, pad=4)
    ax_stock.set_xlim(0, n - 1)
    ax_stock.set_ylim(0, max(stocks) * 1.15)
    ax_stock.set_xlabel("Round", fontsize=8)
    stock_fill = ax_stock.fill_between([], [], color="#3498db", alpha=0.25)
    stock_line, = ax_stock.plot([], [], color="#3498db", linewidth=1.8)
    vline_s = ax_stock.axvline(0, color="#ffffff", linewidth=0.6, alpha=0.4, linestyle="--")
    for sr in shock_rounds:
        ax_stock.axvline(sr, color="#f39c12", linewidth=0.8, alpha=0.5, linestyle=":")

    # ── Subplot 3: Consumer demand ──────────────────────────────────────────
    ax_demand.set_title("Consumer Demand vs Shortage", fontsize=10, pad=4)
    ax_demand.set_xlim(0, n - 1)
    ax_demand.set_ylim(0, max(demands) * 1.2)
    ax_demand.set_xlabel("Round", fontsize=8)
    demand_line,   = ax_demand.plot([], [], color="#9b59b6", linewidth=1.8, label="demand")
    shortage_fill = [None]
    vline_d = ax_demand.axvline(0, color="#ffffff", linewidth=0.6, alpha=0.4, linestyle="--")
    for sr in shock_rounds:
        ax_demand.axvline(sr, color="#f39c12", linewidth=0.8, alpha=0.5, linestyle=":")

    # ── Subplot 4: Supplier rates ───────────────────────────────────────────
    ax_supply.set_title("Supplier Production Rates", fontsize=10, pad=4)
    ax_supply.set_xlim(0, n - 1)
    ax_supply.set_ylim(0, max(v for vals in sup_rates.values() for v in vals) * 1.2)
    ax_supply.set_xlabel("Round", fontsize=8)
    sup_palette = ["#e74c3c", "#e67e22", "#2ecc71", "#1abc9c", "#3498db", "#9b59b6", "#f1c40f"]
    sup_lines = {}
    for i, (key, _) in enumerate(sup_rates.items()):
        color = sup_palette[i % len(sup_palette)]
        line, = ax_supply.plot([], [], color=color, linewidth=1.3, alpha=0.8, label=key)
        sup_lines[key] = line
    vline_su = ax_supply.axvline(0, color="#ffffff", linewidth=0.6, alpha=0.4, linestyle="--")
    for sr in shock_rounds:
        ax_supply.axvline(sr, color="#f39c12", linewidth=0.8, alpha=0.5, linestyle=":")

    # Round counter text
    round_text = fig.text(
        0.5, 0.97, "", ha="center", va="top",
        fontsize=11, color=text_color, fontweight="bold",
    )
    shock_text = fig.text(0.5, 0.93, "", ha="center", va="top", fontsize=9, color="#f39c12")

    def update(frame: int):
        xs = list(range(frame + 1))
        ps = prices[: frame + 1]
        ss = stocks[: frame + 1]
        ds = demands[: frame + 1]
        shs = shortages[: frame + 1]

        price_line.set_data(xs, ps)
        price_dot.set_data([frame], [prices[frame]])
        vline_p.set_xdata([frame, frame])

        stock_line.set_data(xs, ss)
        for col in list(ax_stock.collections):
            col.remove()
        ax_stock.fill_between(xs, ss, color="#3498db", alpha=0.25)
        for sr in shock_rounds:
            ax_stock.axvline(sr, color="#f39c12", linewidth=0.8, alpha=0.5, linestyle=":")
        vline_s.set_xdata([frame, frame])

        demand_line.set_data(xs, ds)
        for col in list(ax_demand.collections):
            col.remove()
        ax_demand.fill_between(xs, shs, color="#e74c3c", alpha=0.3)
        for sr in shock_rounds:
            ax_demand.axvline(sr, color="#f39c12", linewidth=0.8, alpha=0.5, linestyle=":")
        vline_d.set_xdata([frame, frame])

        for key, line in sup_lines.items():
            vals = sup_rates[key][: frame + 1]
            line.set_data(list(range(len(vals))), vals)
        vline_su.set_xdata([frame, frame])

        shock_tag = rounds[frame]["shocks"]
        round_text.set_text(f"Round {frame + 1} / {n}")
        shock_text.set_text(f"⚡ {' · '.join(shock_tag)}" if shock_tag else "")

        return (price_line, price_dot, stock_line, demand_line, vline_p, vline_s, vline_d, vline_su)

    import io
    from PIL import Image as PILImage

    frames: list[PILImage.Image] = []
    print(f"Rendering {n} frames …")
    for frame in range(n):
        update(frame)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
        buf.seek(0)
        frames.append(PILImage.open(buf).copy())

    plt.close(fig)

    duration_ms = 1000 // fps
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved: {path}  ({len(frames)} frames @ {fps}fps)")
