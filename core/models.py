"""Data models used throughout the trading system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    YES = "yes"
    NO = "no"


class SignalStrength(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Market:
    """A Kalshi prediction market."""
    ticker: str
    title: str
    category: str
    end_date: datetime
    yes_price: float  # 0.01 - 0.99
    no_price: float
    volume: int
    open_interest: int
    status: str = "open"

    @property
    def implied_probability(self) -> float:
        return self.yes_price

    @property
    def mid_price(self) -> float:
        return (self.yes_price + (1.0 - self.no_price)) / 2.0


@dataclass
class Signal:
    """A trading signal produced by a research agent."""
    agent_name: str
    market_ticker: str
    side: Side
    confidence: float  # 0.0 - 1.0
    estimated_probability: float  # agent's estimated true probability
    edge: float  # estimated_probability - market implied probability
    strength: SignalStrength
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class TradeOpportunity:
    """An aggregated trade opportunity from multiple signals."""
    market: Market
    side: Side
    entry_price: float
    estimated_probability: float
    edge: float
    confidence: float
    signals: list[Signal]
    recommended_size: int = 1  # number of contracts
    expected_value: float = 0.0


@dataclass
class Order:
    """An order to be placed on Kalshi."""
    market_ticker: str
    side: Side
    quantity: int
    price: float  # limit price
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Position:
    """A current position in a market."""
    market_ticker: str
    side: Side
    quantity: int
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price


@dataclass
class PortfolioSnapshot:
    """Current portfolio state."""
    balance: float
    positions: list[Position]
    daily_pnl: float
    total_pnl: float
    open_orders: list[Order]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_exposure(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def position_count(self) -> int:
        return len(self.positions)
