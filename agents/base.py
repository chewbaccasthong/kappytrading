"""Base class for all research agents."""

from abc import ABC, abstractmethod
from datetime import datetime

import structlog

from core.models import Market, Signal


class BaseAgent(ABC):
    """Abstract base for research agents that analyze markets and produce signals."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight  # relative importance in signal aggregation
        self.logger = structlog.get_logger(agent=name)
        self._last_run: datetime | None = None

    @abstractmethod
    async def analyze(self, market: Market) -> Signal | None:
        """Analyze a market and return a trading signal, or None if no edge found."""
        ...

    async def analyze_batch(self, markets: list[Market]) -> list[Signal]:
        """Analyze multiple markets and return all signals with edges."""
        signals = []
        for market in markets:
            try:
                signal = await self.analyze(market)
                if signal and abs(signal.edge) >= 0.02:
                    signals.append(signal)
            except Exception as e:
                self.logger.warning("analysis_failed", market=market.ticker, error=str(e))
        self._last_run = datetime.utcnow()
        return signals
