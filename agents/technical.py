"""Technical analysis agent - uses price momentum, moving averages, RSI, etc.

Fetches crypto price data from exchanges and computes technical indicators
to estimate the probability of price targets being hit.
"""

import re
from datetime import datetime, timedelta

import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import ta

from agents.base import BaseAgent
from core.models import Market, Signal, Side, SignalStrength


class TechnicalAnalysisAgent(BaseAgent):
    """Analyzes crypto price data with technical indicators to estimate
    whether price targets in Kalshi markets will be hit."""

    def __init__(self, weight: float = 1.2):
        super().__init__(name="technical_analysis", weight=weight)
        self._exchange = ccxt.binance({"enableRateLimit": True})

    async def _fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 200
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance."""
        try:
            ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            self.logger.warning("ohlcv_fetch_failed", symbol=symbol, error=str(e))
            return pd.DataFrame()

    def _extract_crypto_info(self, market: Market) -> tuple[str, float | None, str | None]:
        """Extract the crypto symbol, price target, and direction from market title.

        Returns: (symbol, target_price, direction) where direction is 'above'/'below'/None
        """
        title = market.title.lower()

        # Map common names to trading pairs
        symbol_map = {
            "bitcoin": "BTC/USDT", "btc": "BTC/USDT",
            "ethereum": "ETH/USDT", "eth": "ETH/USDT",
            "solana": "SOL/USDT", "sol": "SOL/USDT",
            "dogecoin": "DOGE/USDT", "doge": "DOGE/USDT",
            "xrp": "XRP/USDT", "ripple": "XRP/USDT",
            "cardano": "ADA/USDT", "ada": "ADA/USDT",
        }

        symbol = None
        for name, pair in symbol_map.items():
            if name in title:
                symbol = pair
                break

        if not symbol:
            return ("BTC/USDT", None, None)

        # Extract price target (e.g., "$50,000", "50000", "$100k")
        price_match = re.search(r"\$?([\d,]+\.?\d*)\s*k?", title)
        target_price = None
        if price_match:
            price_str = price_match.group(1).replace(",", "")
            target_price = float(price_str)
            if "k" in title[price_match.end() - 1 : price_match.end() + 1]:
                target_price *= 1000

        direction = None
        if "above" in title or "over" in title or "exceed" in title or "higher" in title:
            direction = "above"
        elif "below" in title or "under" in title or "lower" in title:
            direction = "below"

        return (symbol, target_price, direction)

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        """Compute a suite of technical indicators."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Trend indicators
        sma_20 = ta.trend.sma_indicator(close, window=20)
        sma_50 = ta.trend.sma_indicator(close, window=50)
        ema_12 = ta.trend.ema_indicator(close, window=12)
        ema_26 = ta.trend.ema_indicator(close, window=26)
        macd = ta.trend.macd_diff(close)

        # Momentum
        rsi = ta.momentum.rsi(close, window=14)
        stoch = ta.momentum.stoch(high, low, close)

        # Volatility
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        atr = ta.volatility.average_true_range(high, low, close, window=14)

        current = close.iloc[-1]

        return {
            "current_price": current,
            "sma_20": sma_20.iloc[-1],
            "sma_50": sma_50.iloc[-1],
            "ema_12": ema_12.iloc[-1],
            "ema_26": ema_26.iloc[-1],
            "macd": macd.iloc[-1],
            "rsi": rsi.iloc[-1],
            "stochastic": stoch.iloc[-1],
            "bb_upper": bb.bollinger_hband().iloc[-1],
            "bb_lower": bb.bollinger_lband().iloc[-1],
            "bb_mid": bb.bollinger_mavg().iloc[-1],
            "atr": atr.iloc[-1],
            "trend_bullish": current > sma_20.iloc[-1] > sma_50.iloc[-1],
            "trend_bearish": current < sma_20.iloc[-1] < sma_50.iloc[-1],
            "volatility_pct": (atr.iloc[-1] / current) * 100 if current > 0 else 0,
            "price_change_24h": (current - close.iloc[-24]) / close.iloc[-24] if len(close) >= 24 else 0,
        }

    def _estimate_probability(
        self, indicators: dict, target_price: float | None, direction: str | None, days_to_expiry: float
    ) -> float:
        """Estimate probability of target being hit using technical analysis."""
        current = indicators["current_price"]

        if target_price is None or direction is None:
            # No clear target - use momentum signals
            bullish_score = 0.0
            if indicators["trend_bullish"]:
                bullish_score += 0.15
            if indicators["macd"] > 0:
                bullish_score += 0.10
            if indicators["rsi"] < 70:
                bullish_score += 0.05
            if indicators["rsi"] < 30:
                bullish_score += 0.15  # oversold bounce
            return 0.50 + bullish_score

        # Calculate distance to target as percentage
        distance_pct = (target_price - current) / current

        # Base probability from distance and volatility
        daily_vol = indicators["volatility_pct"] / 100.0
        expected_move = daily_vol * np.sqrt(max(days_to_expiry, 0.1))

        if direction == "above":
            # Probability price goes above target
            if distance_pct <= 0:
                # Already above target
                base_prob = 0.70 + min(abs(distance_pct) / expected_move * 0.15, 0.20)
            else:
                z_score = distance_pct / expected_move if expected_move > 0 else 5.0
                base_prob = max(0.05, 0.50 * np.exp(-0.5 * z_score))
        else:
            # Probability price goes below target
            if distance_pct >= 0:
                # Already below target
                base_prob = 0.70 + min(abs(distance_pct) / expected_move * 0.15, 0.20)
            else:
                z_score = abs(distance_pct) / expected_move if expected_move > 0 else 5.0
                base_prob = max(0.05, 0.50 * np.exp(-0.5 * z_score))

        # Adjust for trend
        trend_adj = 0.0
        if direction == "above" and indicators["trend_bullish"]:
            trend_adj += 0.08
        elif direction == "above" and indicators["trend_bearish"]:
            trend_adj -= 0.08
        elif direction == "below" and indicators["trend_bearish"]:
            trend_adj += 0.08
        elif direction == "below" and indicators["trend_bullish"]:
            trend_adj -= 0.08

        # Adjust for momentum
        if indicators["macd"] > 0 and direction == "above":
            trend_adj += 0.04
        elif indicators["macd"] < 0 and direction == "below":
            trend_adj += 0.04

        # RSI extreme adjustments
        if indicators["rsi"] > 80:
            trend_adj -= 0.05 if direction == "above" else 0.0
            trend_adj += 0.05 if direction == "below" else 0.0
        elif indicators["rsi"] < 20:
            trend_adj += 0.05 if direction == "above" else 0.0
            trend_adj -= 0.05 if direction == "below" else 0.0

        return max(0.02, min(0.98, base_prob + trend_adj))

    async def analyze(self, market: Market) -> Signal | None:
        """Analyze a crypto prediction market using technical indicators."""
        symbol, target_price, direction = self._extract_crypto_info(market)

        df = await self._fetch_ohlcv(symbol, "1h", 200)
        if df.empty:
            return None

        indicators = self._compute_indicators(df)
        days_to_expiry = max((market.end_date - datetime.utcnow()).total_seconds() / 86400, 0.1)

        estimated_prob = self._estimate_probability(indicators, target_price, direction, days_to_expiry)
        market_prob = market.implied_probability
        edge = estimated_prob - market_prob

        if abs(edge) < 0.03:
            return None

        if edge > 0:
            side = Side.YES
            confidence = min(abs(edge) / 0.20, 1.0)
        else:
            side = Side.NO
            confidence = min(abs(edge) / 0.20, 1.0)

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
                f"TA: price={indicators['current_price']:.2f}, "
                f"RSI={indicators['rsi']:.1f}, MACD={'bullish' if indicators['macd'] > 0 else 'bearish'}, "
                f"trend={'up' if indicators['trend_bullish'] else 'down' if indicators['trend_bearish'] else 'neutral'}, "
                f"est_prob={estimated_prob:.2%} vs market={market_prob:.2%}"
            ),
            metadata={"indicators": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in indicators.items()}},
        )

    async def cleanup(self):
        """Close the exchange connection."""
        await self._exchange.close()
