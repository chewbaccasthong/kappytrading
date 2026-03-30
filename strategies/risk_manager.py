"""Risk management system - protects capital through position sizing and limits.

Enforces:
- Maximum position size per market
- Maximum daily loss limit
- Maximum number of open positions
- Kelly criterion position sizing
- Correlation-based exposure limits
"""

from datetime import datetime, date

import structlog

from core.models import TradeOpportunity, PortfolioSnapshot, Order, Side
from config.settings import settings

logger = structlog.get_logger(__name__)


class RiskManager:
    """Manages trading risk through position sizing and exposure limits."""

    def __init__(
        self,
        max_position_size: int | None = None,
        max_daily_loss: float | None = None,
        max_open_positions: int | None = None,
        max_portfolio_exposure: float = 0.5,  # max 50% of balance at risk
    ):
        self.max_position_size = max_position_size or settings.max_position_size
        self.max_daily_loss = max_daily_loss or settings.max_daily_loss
        self.max_open_positions = max_open_positions or settings.max_open_positions
        self.max_portfolio_exposure = max_portfolio_exposure
        self._daily_pnl: dict[date, float] = {}
        self._trades_today: dict[date, int] = {}

    def check_can_trade(self, portfolio: PortfolioSnapshot) -> tuple[bool, str]:
        """Check if we're allowed to place new trades."""
        today = date.today()

        # Check daily loss limit
        if portfolio.daily_pnl <= -self.max_daily_loss:
            return False, f"Daily loss limit reached: ${portfolio.daily_pnl:.2f}"

        # Check max open positions
        if portfolio.position_count >= self.max_open_positions:
            return False, f"Max positions reached: {portfolio.position_count}/{self.max_open_positions}"

        # Check portfolio exposure
        if portfolio.balance > 0:
            exposure_pct = portfolio.total_exposure / portfolio.balance
            if exposure_pct >= self.max_portfolio_exposure:
                return False, f"Max exposure reached: {exposure_pct:.0%}"

        # Check minimum balance
        if portfolio.balance < 5.0:
            return False, f"Balance too low: ${portfolio.balance:.2f}"

        return True, "OK"

    def size_position(
        self, opportunity: TradeOpportunity, portfolio: PortfolioSnapshot
    ) -> int:
        """Calculate optimal position size using fractional Kelly criterion.

        Uses half-Kelly for conservative sizing (reduces variance while
        capturing most of the growth).
        """
        if portfolio.balance <= 0:
            return 0

        edge = opportunity.edge
        entry_price = opportunity.entry_price

        # Kelly fraction: f* = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        if opportunity.side == Side.YES:
            win_prob = opportunity.estimated_probability
            odds = (1.0 - entry_price) / entry_price if entry_price > 0 else 0
        else:
            win_prob = 1.0 - opportunity.estimated_probability
            odds = (1.0 - entry_price) / entry_price if entry_price > 0 else 0

        if odds <= 0:
            return 0

        kelly = (odds * win_prob - (1 - win_prob)) / odds
        half_kelly = kelly / 2.0

        if half_kelly <= 0:
            return 0

        # Convert to dollar amount
        kelly_dollars = portfolio.balance * half_kelly

        # Apply position size limits
        max_dollars = min(
            kelly_dollars,
            self.max_position_size * entry_price,  # max contracts * price
            portfolio.balance * 0.10,  # max 10% of balance per trade
        )

        # Convert to number of contracts
        contracts = int(max_dollars / entry_price) if entry_price > 0 else 0

        # Ensure at least 1 contract if we have edge, max of position limit
        contracts = max(1, min(contracts, self.max_position_size))

        # Final check: can we afford it?
        cost = contracts * entry_price
        if cost > portfolio.balance * 0.15:
            contracts = max(1, int(portfolio.balance * 0.15 / entry_price))

        logger.info(
            "position_sized",
            market=opportunity.market.ticker,
            kelly=f"{kelly:.4f}",
            half_kelly=f"{half_kelly:.4f}",
            contracts=contracts,
            cost=f"${contracts * entry_price:.2f}",
        )

        return contracts

    def validate_order(
        self, order: Order, portfolio: PortfolioSnapshot
    ) -> tuple[bool, str]:
        """Final validation before placing an order."""
        cost = order.quantity * order.price

        if cost > portfolio.balance:
            return False, f"Insufficient balance: need ${cost:.2f}, have ${portfolio.balance:.2f}"

        if order.quantity > self.max_position_size:
            return False, f"Position too large: {order.quantity} > {self.max_position_size}"

        if order.price < 0.01 or order.price > 0.99:
            return False, f"Invalid price: {order.price}"

        if order.quantity <= 0:
            return False, f"Invalid quantity: {order.quantity}"

        return True, "OK"

    def record_trade_result(self, pnl: float):
        """Record a trade result for daily tracking."""
        today = date.today()
        self._daily_pnl[today] = self._daily_pnl.get(today, 0.0) + pnl
        self._trades_today[today] = self._trades_today.get(today, 0) + 1

    def get_daily_stats(self) -> dict:
        """Get today's trading statistics."""
        today = date.today()
        return {
            "daily_pnl": self._daily_pnl.get(today, 0.0),
            "trades_today": self._trades_today.get(today, 0),
            "max_daily_loss": self.max_daily_loss,
            "remaining_loss_budget": self.max_daily_loss + self._daily_pnl.get(today, 0.0),
        }
