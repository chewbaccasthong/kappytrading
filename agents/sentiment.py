"""Sentiment analysis agent - monitors crypto news and social media.

Aggregates sentiment from multiple sources to gauge market mood
and estimate probability shifts for prediction markets.
"""

from datetime import datetime

import httpx
import structlog
from textblob import TextBlob

from agents.base import BaseAgent
from core.models import Market, Signal, Side, SignalStrength


class SentimentAgent(BaseAgent):
    """Analyzes sentiment from crypto news and social sources to find
    mispricings in prediction markets."""

    def __init__(self, weight: float = 0.8):
        super().__init__(name="sentiment_analysis", weight=weight)
        self._http: httpx.AsyncClient | None = None

    async def _ensure_client(self):
        if not self._http:
            self._http = httpx.AsyncClient(timeout=15.0)

    async def _fetch_cryptopanic_sentiment(self, currency: str) -> dict:
        """Fetch sentiment data from CryptoPanic (free tier)."""
        await self._ensure_client()
        try:
            url = f"https://cryptopanic.com/api/free/v1/posts/?currencies={currency}&kind=news"
            resp = await self._http.get(url)
            if resp.status_code != 200:
                return {"score": 0.0, "count": 0, "headlines": []}

            data = resp.json()
            posts = data.get("results", [])

            sentiments = []
            headlines = []
            for post in posts[:20]:
                title = post.get("title", "")
                headlines.append(title)
                blob = TextBlob(title)
                sentiments.append(blob.sentiment.polarity)

            avg_score = sum(sentiments) / len(sentiments) if sentiments else 0.0
            return {"score": avg_score, "count": len(sentiments), "headlines": headlines[:5]}
        except Exception as e:
            self.logger.warning("cryptopanic_failed", error=str(e))
            return {"score": 0.0, "count": 0, "headlines": []}

    async def _fetch_coingecko_sentiment(self, coin_id: str) -> dict:
        """Fetch community sentiment from CoinGecko."""
        await self._ensure_client()
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            resp = await self._http.get(url, params={"localization": "false", "tickers": "false"})
            if resp.status_code != 200:
                return {"sentiment_up": 0.5, "sentiment_down": 0.5}

            data = resp.json()
            return {
                "sentiment_up": data.get("sentiment_votes_up_percentage", 50) / 100.0,
                "sentiment_down": data.get("sentiment_votes_down_percentage", 50) / 100.0,
                "developer_score": data.get("developer_score", 50),
                "community_score": data.get("community_score", 50),
            }
        except Exception as e:
            self.logger.warning("coingecko_sentiment_failed", error=str(e))
            return {"sentiment_up": 0.5, "sentiment_down": 0.5}

    def _map_crypto_to_ids(self, market: Market) -> tuple[str, str]:
        """Map market title to CryptoPanic currency and CoinGecko coin_id."""
        title = market.title.lower()
        mappings = {
            "bitcoin": ("BTC", "bitcoin"),
            "btc": ("BTC", "bitcoin"),
            "ethereum": ("ETH", "ethereum"),
            "eth": ("ETH", "ethereum"),
            "solana": ("SOL", "solana"),
            "sol": ("SOL", "solana"),
            "dogecoin": ("DOGE", "dogecoin"),
            "xrp": ("XRP", "ripple"),
            "cardano": ("ADA", "cardano"),
        }
        for keyword, (currency, coin_id) in mappings.items():
            if keyword in title:
                return (currency, coin_id)
        return ("BTC", "bitcoin")

    def _compute_sentiment_probability(
        self,
        news_sentiment: dict,
        community_sentiment: dict,
        market: Market,
    ) -> float:
        """Combine sentiment sources into a probability estimate."""
        # News sentiment: -1 to +1 -> map to probability adjustment
        news_score = news_sentiment.get("score", 0.0)
        news_adj = news_score * 0.15  # max ±15% adjustment from news

        # Community sentiment
        community_bullish = community_sentiment.get("sentiment_up", 0.5)
        community_adj = (community_bullish - 0.5) * 0.10  # max ±5% adjustment

        # Base from current market price
        base_prob = market.implied_probability

        # Sentiment suggests probability should be higher or lower
        adjusted = base_prob + news_adj + community_adj

        return max(0.05, min(0.95, adjusted))

    async def analyze(self, market: Market) -> Signal | None:
        """Analyze sentiment for a crypto prediction market."""
        currency, coin_id = self._map_crypto_to_ids(market)

        news_sentiment = await self._fetch_cryptopanic_sentiment(currency)
        community_sentiment = await self._fetch_coingecko_sentiment(coin_id)

        estimated_prob = self._compute_sentiment_probability(
            news_sentiment, community_sentiment, market
        )
        market_prob = market.implied_probability
        edge = estimated_prob - market_prob

        if abs(edge) < 0.03:
            return None

        side = Side.YES if edge > 0 else Side.NO
        confidence = min(abs(edge) / 0.15, 1.0)

        # Boost confidence if multiple sources agree
        news_bullish = news_sentiment.get("score", 0) > 0.1
        community_bullish = community_sentiment.get("sentiment_up", 0.5) > 0.6
        if (news_bullish and community_bullish and side == Side.YES) or \
           (not news_bullish and not community_bullish and side == Side.NO):
            confidence = min(confidence * 1.2, 1.0)

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
                f"Sentiment: news={news_sentiment['score']:.2f} ({news_sentiment['count']} articles), "
                f"community_bullish={community_sentiment.get('sentiment_up', 0.5):.0%}, "
                f"est_prob={estimated_prob:.2%} vs market={market_prob:.2%}"
            ),
            metadata={
                "news_score": news_sentiment["score"],
                "headlines": news_sentiment.get("headlines", []),
                "community_bullish": community_sentiment.get("sentiment_up", 0.5),
            },
        )

    async def cleanup(self):
        if self._http:
            await self._http.aclose()
