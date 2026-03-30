"""Main orchestrator - coordinates agents, strategy, and execution.

This is the brain of the trading system. It:
1. Discovers crypto prediction markets on Kalshi
2. Runs all research agents in parallel on each market
3. Aggregates signals into trade opportunities
4. Applies risk management and position sizing
5. Executes trades
6. Monitors and manages positions
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import structlog

from config.settings import settings
from core.kalshi_client import KalshiClient
from core.models import (
    Market, Signal, TradeOpportunity, Order, Side, OrderStatus, PortfolioSnapshot, Position,
)
from agents.technical import TechnicalAnalysisAgent
from agents.sentiment import SentimentAgent
from agents.onchain import OnChainAgent
from agents.microstructure import MicrostructureAgent
from agents.arbitrage import ArbitrageAgent
from strategies.signal_aggregator import SignalAggregator
from strategies.risk_manager import RiskManager
from utils.database import Database
from config.settings import settings

logger = structlog.get_logger(__name__)


class TradingOrchestrator:
    """Orchestrates the entire trading pipeline."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.kalshi = KalshiClient()
        self.aggregator = SignalAggregator(min_agents=2)
        self.risk_manager = RiskManager()
        self.db = Database()

        # Initialize agents
        self.agents = []
        self._agent_weights: dict[str, float] = {}

    def _validate_config(self):
        """Check that API credentials are configured before starting."""
        key_path = Path(settings.kalshi_private_key_path)
        if not key_path.exists():
            logger.error(
                "config_error",
                message=(
                    f"Private key file not found: {key_path.absolute()}\n"
                    "  1. Run: python setup.py\n"
                    "  2. Or create kalshi_private_key.pem with your private key\n"
                    "  3. Set KALSHI_PRIVATE_KEY_PATH in .env"
                ),
            )
            sys.exit(1)

        if not settings.kalshi_api_key or settings.kalshi_api_key == "your-api-key-id-here":
            logger.error(
                "config_error",
                message=(
                    "KALSHI_API_KEY not set in .env\n"
                    "  1. Go to https://demo.kalshi.co/account/api\n"
                    "  2. Create an API key and paste the Key ID into .env"
                ),
            )
            sys.exit(1)

        logger.info(
            "config_ok",
            api_key=settings.kalshi_api_key[:8] + "...",
            key_file=str(key_path),
            base_url=settings.kalshi_base_url,
        )

    async def _init_agents(self):
        """Initialize all research agents."""
        self.agents = [
            TechnicalAnalysisAgent(weight=1.2),
            SentimentAgent(weight=0.8),
            OnChainAgent(weight=1.0),
            MicrostructureAgent(kalshi_client=self.kalshi, weight=1.1),
            ArbitrageAgent(weight=1.5),
        ]
        self._agent_weights = {a.name: a.weight for a in self.agents}
        logger.info("agents_initialized", count=len(self.agents))

    async def _get_portfolio(self) -> PortfolioSnapshot:
        """Get current portfolio state."""
        if self.dry_run:
            return PortfolioSnapshot(
                balance=1000.0,
                positions=[],
                daily_pnl=0.0,
                total_pnl=0.0,
                open_orders=[],
            )

        balance = await self.kalshi.get_balance()
        positions = await self.kalshi.get_positions()
        open_orders = await self.kalshi.get_open_orders()

        return PortfolioSnapshot(
            balance=balance,
            positions=positions,
            daily_pnl=self.risk_manager.get_daily_stats()["daily_pnl"],
            total_pnl=0.0,  # computed from DB
            open_orders=open_orders,
        )

    async def _discover_markets(self) -> list[Market]:
        """Find all open crypto prediction markets on Kalshi."""
        logger.info("discovering_markets")
        markets = await self.kalshi.search_crypto_markets()
        logger.info("markets_found", count=len(markets))
        return markets

    async def _analyze_market(self, market: Market) -> list[Signal]:
        """Run all agents on a single market in parallel."""
        tasks = [agent.analyze(market) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "agent_error",
                    agent=self.agents[i].name,
                    market=market.ticker,
                    error=str(result),
                )
            elif result is not None:
                signals.append(result)
                # Record signal in DB
                self.db.record_signal({
                    "agent_name": result.agent_name,
                    "market_ticker": result.market_ticker,
                    "side": result.side.value,
                    "confidence": result.confidence,
                    "estimated_probability": result.estimated_probability,
                    "edge": result.edge,
                    "reasoning": result.reasoning,
                })

        return signals

    async def _find_opportunities(self, markets: list[Market]) -> list[TradeOpportunity]:
        """Analyze all markets and find trade opportunities."""
        opportunities = []

        for market in markets:
            signals = await self._analyze_market(market)
            if not signals:
                continue

            logger.info(
                "signals_collected",
                market=market.ticker,
                count=len(signals),
                agents=[s.agent_name for s in signals],
            )

            opportunity = self.aggregator.aggregate(
                market, signals, self._agent_weights
            )
            if opportunity:
                opportunities.append(opportunity)
                logger.info(
                    "opportunity_found",
                    market=market.ticker,
                    side=opportunity.side.value,
                    edge=f"{opportunity.edge:.4f}",
                    confidence=f"{opportunity.confidence:.2f}",
                    ev=f"{opportunity.expected_value:.4f}",
                )

        ranked = self.aggregator.rank_opportunities(opportunities)
        logger.info("opportunities_ranked", count=len(ranked))
        return ranked

    async def _execute_trade(
        self, opportunity: TradeOpportunity, portfolio: PortfolioSnapshot
    ) -> Order | None:
        """Execute a trade for a given opportunity."""
        # Size the position
        quantity = self.risk_manager.size_position(opportunity, portfolio)
        if quantity <= 0:
            return None

        # Create order
        price_cents = int(opportunity.entry_price * 100)
        order = Order(
            market_ticker=opportunity.market.ticker,
            side=opportunity.side,
            quantity=quantity,
            price=opportunity.entry_price,
        )

        # Validate
        valid, reason = self.risk_manager.validate_order(order, portfolio)
        if not valid:
            logger.warning("order_rejected", reason=reason, market=opportunity.market.ticker)
            return None

        if self.dry_run:
            logger.info(
                "DRY_RUN_ORDER",
                market=opportunity.market.ticker,
                side=opportunity.side.value,
                qty=quantity,
                price=f"${opportunity.entry_price:.2f}",
                edge=f"{opportunity.edge:.4f}",
                ev=f"${opportunity.expected_value:.4f}",
            )
            # Record in DB even for dry runs
            self.db.record_trade({
                "market_ticker": opportunity.market.ticker,
                "side": opportunity.side.value,
                "quantity": quantity,
                "entry_price": opportunity.entry_price,
                "edge": opportunity.edge,
                "confidence": opportunity.confidence,
                "signals": [s.agent_name for s in opportunity.signals],
            })
            return order

        # Live execution
        try:
            placed_order = await self.kalshi.place_order(
                ticker=opportunity.market.ticker,
                side=opportunity.side,
                quantity=quantity,
                price=price_cents,
            )
            logger.info(
                "order_placed",
                order_id=placed_order.order_id,
                market=opportunity.market.ticker,
                side=opportunity.side.value,
                qty=quantity,
                price=price_cents,
            )

            self.db.record_trade({
                "market_ticker": opportunity.market.ticker,
                "side": opportunity.side.value,
                "quantity": quantity,
                "entry_price": opportunity.entry_price,
                "edge": opportunity.edge,
                "confidence": opportunity.confidence,
                "signals": [s.agent_name for s in opportunity.signals],
            })

            return placed_order
        except Exception as e:
            logger.error("order_failed", market=opportunity.market.ticker, error=str(e))
            return None

    async def run_cycle(self):
        """Run one complete trading cycle: discover -> analyze -> trade."""
        logger.info("cycle_start", dry_run=self.dry_run)

        portfolio = await self._get_portfolio()
        logger.info(
            "portfolio_state",
            balance=f"${portfolio.balance:.2f}",
            positions=portfolio.position_count,
            daily_pnl=f"${portfolio.daily_pnl:.2f}",
        )

        # Check if we can trade
        can_trade, reason = self.risk_manager.check_can_trade(portfolio)
        if not can_trade:
            logger.warning("trading_halted", reason=reason)
            return

        # Discover and analyze markets
        markets = await self._discover_markets()
        if not markets:
            logger.info("no_crypto_markets_found")
            return

        opportunities = await self._find_opportunities(markets)
        if not opportunities:
            logger.info("no_opportunities_found")
            return

        # Execute top opportunities
        trades_placed = 0
        for opp in opportunities:
            # Re-check risk limits after each trade
            can_trade, reason = self.risk_manager.check_can_trade(portfolio)
            if not can_trade:
                logger.info("risk_limit_reached", reason=reason)
                break

            order = await self._execute_trade(opp, portfolio)
            if order:
                trades_placed += 1

        logger.info("cycle_complete", trades_placed=trades_placed)

    async def run(self, interval_seconds: int = 300):
        """Run the trading bot continuously."""
        self._validate_config()
        logger.info("bot_starting", interval=interval_seconds, dry_run=self.dry_run)

        async with self.kalshi:
            await self._init_agents()

            while True:
                try:
                    await self.run_cycle()
                except Exception as e:
                    logger.error("cycle_error", error=str(e), exc_info=True)

                logger.info("sleeping", seconds=interval_seconds)
                await asyncio.sleep(interval_seconds)

    async def run_once(self):
        """Run a single trading cycle (useful for testing)."""
        self._validate_config()
        async with self.kalshi:
            await self._init_agents()
            await self.run_cycle()

            # Cleanup agents
            for agent in self.agents:
                if hasattr(agent, "cleanup"):
                    await agent.cleanup()

    def get_status(self) -> dict:
        """Get bot status summary."""
        perf = self.db.get_performance_summary()
        daily = self.risk_manager.get_daily_stats()
        return {
            "mode": "dry_run" if self.dry_run else "live",
            "agents": len(self.agents),
            "performance": perf,
            "daily_stats": daily,
            "recent_trades": self.db.get_recent_trades(10),
        }
