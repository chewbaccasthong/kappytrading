"""Live monitoring dashboard - shows real-time bot status and performance.

Usage:
    python -m scripts.monitor
    python -m scripts.monitor --refresh 10
"""

import argparse
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from utils.database import Database

console = Console()


def build_dashboard(db: Database) -> Layout:
    """Build the monitoring dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )

    # Header
    layout["header"].update(
        Panel(
            Text("KAPPY TRADING BOT - LIVE MONITOR", style="bold cyan", justify="center"),
            border_style="cyan",
        )
    )

    # Performance summary
    perf = db.get_performance_summary()
    perf_table = Table(title="Performance", expand=True)
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="green")

    total_trades = perf.get("total_trades", 0)
    pnl = perf.get("total_pnl", 0)
    pnl_style = "green" if pnl >= 0 else "red"

    perf_table.add_row("Total Trades", str(total_trades))
    perf_table.add_row("Total PnL", f"[{pnl_style}]${pnl:.2f}[/{pnl_style}]")
    perf_table.add_row("Win Rate", f"{perf.get('win_rate', 0):.1%}")
    perf_table.add_row("Avg Win", f"${perf.get('avg_win', 0):.2f}")
    perf_table.add_row("Avg Loss", f"${perf.get('avg_loss', 0):.2f}")
    perf_table.add_row("Avg Edge", f"{perf.get('avg_edge', 0):.4f}")

    layout["left"].update(Panel(perf_table, title="Summary", border_style="green"))

    # Recent trades
    trades = db.get_recent_trades(15)
    trades_table = Table(title="Recent Trades", expand=True)
    trades_table.add_column("Time", style="dim")
    trades_table.add_column("Market")
    trades_table.add_column("Side")
    trades_table.add_column("Qty")
    trades_table.add_column("Price")
    trades_table.add_column("PnL")
    trades_table.add_column("Status")

    for t in trades:
        pnl_str = f"${t['pnl']:.2f}" if t["pnl"] is not None else "-"
        t_pnl = t["pnl"] or 0
        style = "green" if t_pnl > 0 else "red" if t_pnl < 0 else "dim"

        created = t["created"][:16] if t["created"] else "-"
        trades_table.add_row(
            created,
            t["ticker"][:18],
            t["side"],
            str(t["qty"]),
            f"${t['entry']:.2f}",
            f"[{style}]{pnl_str}[/{style}]",
            t["status"],
        )

    layout["right"].update(Panel(trades_table, title="Trades", border_style="yellow"))

    # Footer
    layout["footer"].update(
        Panel(
            Text(
                f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Press Ctrl+C to exit",
                style="dim",
                justify="center",
            ),
            border_style="dim",
        )
    )

    return layout


def run_monitor(refresh: int = 5):
    """Run the live monitoring dashboard."""
    db = Database()

    console.print("[bold cyan]Starting live monitor...[/bold cyan]")
    console.print(f"[dim]Refreshing every {refresh}s[/dim]\n")

    try:
        with Live(build_dashboard(db), console=console, refresh_per_second=1) as live:
            while True:
                time.sleep(refresh)
                live.update(build_dashboard(db))
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Live monitoring dashboard")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    args = parser.parse_args()
    run_monitor(refresh=args.refresh)


if __name__ == "__main__":
    main()
