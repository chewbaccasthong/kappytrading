"""Market microstructure agent - analyzes Kalshi orderbook dynamics.

Looks at bid-ask spreads, order imbalances, volume patterns,
and price movements within Kalshi itself to find mispricings.
"""

from datetime import datetime

import structlog

from agents.base import BaseAgent
from core.kalshi_client import KalshiClient
from core.models import Market, Signal, Side, SignalStrength


class MicrostructureAgent(BaseAgent):
    """Analyzes Kalshi orderbook and trade data to detect mispricings
    based on market microstructure signals."""

    def __init__(self, kalshi_client: KalshiClient, weight: float = 1.1):
        super().__init__(name="microstructure", weight=weight)
        self.kalshi = kalshi_client

    async def _analyze_orderbook(self, ticker: str) -> dict:
        """Analyze orderbook depth and imbalance."""
        try:
            book = await self.kalshi.get_orderbook(ticker)
        except Exception as e:
            self.logger.warning("orderbook_fetch_failed", ticker=ticker, error=str(e))
            return {}

        yes_bids = book.get("orderbook", {}).get("yes", [])
        no_bids = book.get("orderbook", {}).get("no", [])

        # Calculate bid depth
        yes_depth = sum(level[1] for level in yes_bids) if yes_bids else 0
        no_depth = sum(level[1] for level in no_bids) if no_bids else 0
        total_depth = yes_depth + no_depth

        # Order imbalance
        imbalance = (yes_depth - no_depth) / total_depth if total_depth > 0 else 0.0

        # Spread analysis
        best_yes_bid = yes_bids[0][0] / 100.0 if yes_bids else 0.0
        best_no_bid = no_bids[0][0] / 100.0 if no_bids else 0.0
        spread = 1.0 - best_yes_bid - best_no_bid if (best_yes_bid and best_no_bid) else 1.0

        # Depth concentration - are orders clustered at one level?
        yes_concentration = 0.0
        if yes_bids and yes_depth > 0:
            yes_concentration = yes_bids[0][1] / yes_depth

        return {
            "yes_depth": yes_depth,
            "no_depth": no_depth,
            "total_depth": total_depth,
            "imbalance": imbalance,  # positive = more yes buyers
            "spread": spread,
            "best_yes_bid": best_yes_bid,
            "best_no_bid": best_no_bid,
            "yes_concentration": yes_concentration,
        }

    async def _analyze_trades(self, ticker: str) -> dict:
        """Analyze recent trade patterns."""
        try:
            trades = await self.kalshi.get_market_history(ticker, limit=50)
        except Exception as e:
            self.logger.warning("trade_history_failed", ticker=ticker, error=str(e))
            return {}

        if not trades:
            return {"trade_count": 0}

        yes_trades = [t for t in trades if t.get("taker_side") == "yes"]
        no_trades = [t for t in trades if t.get("taker_side") == "no"]

        yes_volume = sum(t.get("count", 0) for t in yes_trades)
        no_volume = sum(t.get("count", 0) for t in no_trades)
        total_volume = yes_volume + no_volume

        # Volume-weighted average price
        def vwap(trade_list):
            total_qty = sum(t.get("count", 0) for t in trade_list)
            if total_qty == 0:
                return 0.0
            return sum(t.get("yes_price", 50) * t.get("count", 0) for t in trade_list) / total_qty / 100.0

        # Price trend in recent trades
        prices = [t.get("yes_price", 50) / 100.0 for t in trades]
        price_trend = 0.0
        if len(prices) >= 5:
            recent_avg = sum(prices[:5]) / 5
            older_avg = sum(prices[-5:]) / 5
            price_trend = recent_avg - older_avg

        return {
            "trade_count": len(trades),
            "yes_volume": yes_volume,
            "no_volume": no_volume,
            "volume_imbalance": (yes_volume - no_volume) / total_volume if total_volume > 0 else 0,
            "vwap": vwap(trades),
            "price_trend": price_trend,
        }

    def _compute_micro_probability(
        self, orderbook: dict, trade_data: dict, market: Market
    ) -> float:
        """Estimate probability from microstructure signals."""
        if not orderbook:
            return market.implied_probability

        base_prob = market.implied_probability
        adjustment = 0.0

        # Orderbook imbalance signal
        # More yes buyers = price should be higher
        imbalance = orderbook.get("imbalance", 0)
        adjustment += imbalance * 0.08

        # Spread signal - wide spread means uncertainty, prices may revert
        spread = orderbook.get("spread", 0.10)
        if spread > 0.15:
            # Wide spread: price is uncertain, less conviction
            adjustment *= 0.5

        # Trade flow signal
        if trade_data:
            vol_imbalance = trade_data.get("volume_imbalance", 0)
            adjustment += vol_imbalance * 0.06

            # Price trend continuation
            price_trend = trade_data.get("price_trend", 0)
            adjustment += price_trend * 0.5

        return max(0.05, min(0.95, base_prob + adjustment))

    async def analyze(self, market: Market) -> Signal | None:
        """Analyze market microstructure for a Kalshi market."""
        orderbook = await self._analyze_orderbook(market.ticker)
        trade_data = await self._analyze_trades(market.ticker)

        if not orderbook or orderbook.get("total_depth", 0) < 10:
            # Skip illiquid markets
            return None

        estimated_prob = self._compute_micro_probability(orderbook, trade_data, market)
        market_prob = market.implied_probability
        edge = estimated_prob - market_prob

        if abs(edge) < 0.03:
            return None

        side = Side.YES if edge > 0 else Side.NO
        confidence = min(abs(edge) / 0.12, 1.0)

        # Boost confidence for liquid markets with clear signals
        if orderbook.get("total_depth", 0) > 100 and abs(orderbook.get("imbalance", 0)) > 0.3:
            confidence = min(confidence * 1.3, 1.0)

        if confidence > 0.7:
            strength = SignalStrength.STRONG_BUY
        elif confidence > 0.4:
            strength = SignalStrength.BUY
        else:
            strength = SignalStrength.NEUTRAL

        return Signal(
            agent_name=self.name,
            market_ticker=market.ticker,
            side=side,
            confidence=confidence,
            estimated_probability=estimated_prob,
            edge=edge,
            strength=strength,
            reasoning=(
                f"Microstructure: imbalance={orderbook.get('imbalance', 0):.2f}, "
                f"spread={orderbook.get('spread', 0):.3f}, "
                f"depth={orderbook.get('total_depth', 0)}, "
                f"vol_flow={trade_data.get('volume_imbalance', 0):.2f}, "
                f"est_prob={estimated_prob:.2%} vs market={market_prob:.2%}"
            ),
            metadata={"orderbook": orderbook, "trade_data": trade_data},
        )
