"""Backtester - replay historical data through the strategy engine.

Simulates how the trading system would have performed on past markets
by replaying historical prices through the agents and strategy engine.

Usage:
    python -m scripts.backtest --days 30
    python -m scripts.backtest --days 7 --agent technical_analysis
"""

import argparse
import asyncio
from datetime import datetime, timedelta

from rich.console import Console
from rich.table import Table

from core.models import Market, Side
from agents.technical import TechnicalAnalysisAgent
from agents.sentiment import SentimentAgent
from agents.onchain import OnChainAgent
from agents.arbitrage import ArbitrageAgent
from strategies.signal_aggregator import SignalAggregator
from strategies.risk_manager import RiskManager

console = Console()


class BacktestResult:
    """Tracks simulated trading results."""

    def __init__(self, starting_balance: float = 1000.0):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.trades: list[dict] = []
        self.wins = 0
        self.losses = 0

    def record_trade(self, market: str, side: str, qty: int, entry: float, result: bool):
        if result:
            pnl = qty * (1.0 - entry)
            self.wins += 1
        else:
            pnl = -qty * entry
            self.losses += 1

        self.balance += pnl
        self.trades.append({
            "market": market,
            "side": side,
            "qty": qty,
            "entry": entry,
            "pnl": pnl,
            "result": "WIN" if result else "LOSS",
            "balance": self.balance,
        })

    @property
    def total_pnl(self) -> float:
        return self.balance - self.starting_balance

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0

    @property
    def roi(self) -> float:
        return self.total_pnl / self.starting_balance


async def run_backtest(days: int = 30, agent_filter: str | None = None):
    """Run a backtest simulation.

    Since we can't replay actual historical Kalshi markets (they don't provide
    historical API), this creates synthetic scenarios based on historical
    crypto price data and simulates what signals the agents would have generated.
    """
    console.print(f"[bold cyan]Running backtest for {days} days...[/bold cyan]")

    # Initialize agents
    agents = []
    ta_agent = TechnicalAnalysisAgent()
    agents.append(ta_agent)

    if not agent_filter or agent_filter == "sentiment":
        agents.append(SentimentAgent())
    if not agent_filter or agent_filter == "onchain":
        agents.append(OnChainAgent())
    if not agent_filter or agent_filter == "arbitrage":
        agents.append(ArbitrageAgent())

    aggregator = SignalAggregator(min_agents=1)  # relaxed for backtest
    result = BacktestResult()

    # Generate synthetic markets based on BTC price levels
    import ccxt.async_support as ccxt
    exchange = ccxt.binance({"enableRateLimit": True})

    try:
        ohlcv = await exchange.fetch_ohlcv("BTC/USDT", "1d", limit=days)
    except Exception as e:
        console.print(f"[red]Failed to fetch historical data: {e}[/red]")
        return
    finally:
        await exchange.close()

    console.print(f"[dim]Fetched {len(ohlcv)} daily candles[/dim]")

    # For each day, create synthetic prediction markets
    for i in range(1, len(ohlcv)):
        day_close = ohlcv[i][4]
        prev_close = ohlcv[i - 1][4]
        day_date = datetime.fromtimestamp(ohlcv[i][0] / 1000)

        # Synthetic market: "Will BTC be above ${target} by end of day?"
        target = round(prev_close * 1.02, -1)  # 2% above previous close

        # What the market "would have priced" - simple model
        synthetic_yes_price = 0.40  # assume market priced it at 40%

        market = Market(
            ticker=f"BTC-ABOVE-{int(target)}-{day_date.strftime('%m%d')}",
            title=f"Will Bitcoin be above ${target:,.0f}?",
            category="Crypto",
            end_date=day_date + timedelta(hours=23),
            yes_price=synthetic_yes_price,
            no_price=1.0 - synthetic_yes_price,
            volume=1000,
            open_interest=500,
        )

        # Run technical agent (the one with real data access)
        signals = []
        for agent in agents:
            try:
                signal = await agent.analyze(market)
                if signal:
                    signals.append(signal)
            except Exception:
                continue

        if not signals:
            continue

        opportunity = aggregator.aggregate(
            market, signals, {a.name: a.weight for a in agents}
        )

        if opportunity and opportunity.edge > 0.03:
            # Simulate: did BTC actually go above the target?
            actual_result = day_close > target

            # Did we bet correctly?
            if opportunity.side == Side.YES:
                trade_won = actual_result
            else:
                trade_won = not actual_result

            result.record_trade(
                market=market.ticker,
                side=opportunity.side.value,
                qty=min(5, int(opportunity.confidence * 10)),
                entry=opportunity.entry_price,
                result=trade_won,
            )

    # Clean up agents
    for agent in agents:
        if hasattr(agent, "cleanup"):
            await agent.cleanup()

    # Display results
    console.print()
    summary = Table(title=f"Backtest Results ({days} days)")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")

    summary.add_row("Starting Balance", f"${result.starting_balance:.2f}")
    summary.add_row("Final Balance", f"${result.balance:.2f}")
    summary.add_row("Total PnL", f"${result.total_pnl:.2f}")
    summary.add_row("ROI", f"{result.roi:.1%}")
    summary.add_row("Total Trades", str(len(result.trades)))
    summary.add_row("Wins", str(result.wins))
    summary.add_row("Losses", str(result.losses))
    summary.add_row("Win Rate", f"{result.win_rate:.1%}")

    console.print(summary)

    if result.trades:
        trades_table = Table(title="Trade Log")
        trades_table.add_column("Market")
        trades_table.add_column("Side")
        trades_table.add_column("Qty")
        trades_table.add_column("Entry")
        trades_table.add_column("PnL")
        trades_table.add_column("Result")
        trades_table.add_column("Balance")

        for t in result.trades[-20:]:
            style = "green" if t["result"] == "WIN" else "red"
            trades_table.add_row(
                t["market"][:25],
                t["side"],
                str(t["qty"]),
                f"${t['entry']:.2f}",
                f"[{style}]${t['pnl']:.2f}[/{style}]",
                f"[{style}]{t['result']}[/{style}]",
                f"${t['balance']:.2f}",
            )

        console.print(trades_table)


def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategies")
    parser.add_argument("--days", type=int, default=30, help="Days of history to test")
    parser.add_argument("--agent", type=str, default=None, help="Filter to specific agent")
    args = parser.parse_args()
    asyncio.run(run_backtest(days=args.days, agent_filter=args.agent))


if __name__ == "__main__":
    main()
