"""Kappy Trading Bot - Kalshi crypto prediction market trading system.

Usage:
    python main.py              # Run in dry-run mode (no real trades)
    python main.py --live       # Run with real money (careful!)
    python main.py --once       # Run a single cycle then exit
    python main.py --status     # Show performance stats
"""

import argparse
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from config.settings import settings
from core.orchestrator import TradingOrchestrator
from utils.logger import setup_logging
from utils.database import Database

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def show_banner():
    console.print(Panel.fit(
        "[bold cyan]KAPPY TRADING BOT[/bold cyan]\n"
        "[dim]Kalshi Crypto Prediction Market Trader[/dim]\n"
        "[dim]5 Research Agents | Signal Aggregation | Risk Management[/dim]",
        border_style="cyan",
    ))


def show_status():
    """Display bot performance stats."""
    db = Database()
    perf = db.get_performance_summary()
    trades = db.get_recent_trades(20)

    # Performance summary
    table = Table(title="Performance Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Trades", str(perf.get("total_trades", 0)))
    table.add_row("Total PnL", f"${perf.get('total_pnl', 0):.2f}")
    table.add_row("Win Rate", f"{perf.get('win_rate', 0):.1%}")
    table.add_row("Avg Win", f"${perf.get('avg_win', 0):.2f}")
    table.add_row("Avg Loss", f"${perf.get('avg_loss', 0):.2f}")
    table.add_row("Avg Edge", f"{perf.get('avg_edge', 0):.4f}")

    console.print(table)

    # Recent trades
    if trades:
        trade_table = Table(title="Recent Trades")
        trade_table.add_column("ID")
        trade_table.add_column("Market")
        trade_table.add_column("Side")
        trade_table.add_column("Qty")
        trade_table.add_column("Entry")
        trade_table.add_column("PnL")
        trade_table.add_column("Edge")
        trade_table.add_column("Status")

        for t in trades[:10]:
            pnl_str = f"${t['pnl']:.2f}" if t["pnl"] is not None else "-"
            pnl_style = "green" if (t["pnl"] or 0) > 0 else "red" if (t["pnl"] or 0) < 0 else "dim"
            trade_table.add_row(
                str(t["id"]),
                t["ticker"][:20],
                t["side"],
                str(t["qty"]),
                f"${t['entry']:.2f}",
                f"[{pnl_style}]{pnl_str}[/{pnl_style}]",
                f"{t['edge']:.4f}",
                t["status"],
            )

        console.print(trade_table)


async def run_bot(live: bool = False, once: bool = False, interval: int = 300):
    """Start the trading bot."""
    show_banner()

    if live:
        console.print("[bold red]LIVE TRADING MODE[/bold red] - Real money at risk!")
        if not settings.kalshi_api_key or not settings.kalshi_private_key_path:
            console.print("[red]Error: KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH must be set in .env[/red]")
            sys.exit(1)
    else:
        console.print("[bold yellow]DRY RUN MODE[/bold yellow] - No real trades will be placed")

    console.print(f"[dim]Interval: {interval}s | Min edge: {settings.min_edge_threshold}[/dim]")
    console.print(f"[dim]Max position: {settings.max_position_size} contracts[/dim]")
    console.print(f"[dim]Max daily loss: ${settings.max_daily_loss}[/dim]")
    console.print()

    orchestrator = TradingOrchestrator(dry_run=not live)

    if once:
        await orchestrator.run_once()
    else:
        await orchestrator.run(interval_seconds=interval)


def main():
    parser = argparse.ArgumentParser(description="Kappy Trading Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (real money)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    parser.add_argument("--status", action="store_true", help="Show performance stats")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles (default: 300)")

    args = parser.parse_args()

    setup_logging()

    if args.status:
        show_status()
        return

    asyncio.run(run_bot(live=args.live, once=args.once, interval=args.interval))


if __name__ == "__main__":
    main()
