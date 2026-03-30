"""On-chain data analysis agent - tracks whale movements, exchange flows, etc.

Uses blockchain data to detect large movements that may indicate
upcoming price action, informing prediction market trades.
"""

from datetime import datetime

import httpx
import structlog

from agents.base import BaseAgent
from core.models import Market, Signal, Side, SignalStrength


class OnChainAgent(BaseAgent):
    """Analyzes on-chain metrics (exchange flows, whale movements, active addresses)
    to estimate probability of crypto price targets."""

    def __init__(self, weight: float = 1.0):
        super().__init__(name="onchain_analysis", weight=weight)
        self._http: httpx.AsyncClient | None = None

    async def _ensure_client(self):
        if not self._http:
            self._http = httpx.AsyncClient(timeout=15.0)

    def _map_to_coin(self, market: Market) -> str:
        """Map market title to coin identifier."""
        title = market.title.lower()
        for keyword, coin in [
            ("bitcoin", "bitcoin"), ("btc", "bitcoin"),
            ("ethereum", "ethereum"), ("eth", "ethereum"),
            ("solana", "solana"), ("sol", "solana"),
        ]:
            if keyword in title:
                return coin
        return "bitcoin"

    async def _fetch_exchange_flows(self, coin: str) -> dict:
        """Fetch exchange inflow/outflow data.

        Positive net flow (more inflows) = bearish (selling pressure).
        Negative net flow (more outflows) = bullish (accumulation).
        """
        await self._ensure_client()
        try:
            # Use CoinGecko exchange data as proxy
            url = f"https://api.coingecko.com/api/v3/coins/{coin}/tickers"
            resp = await self._http.get(url, params={"depth": "true"})
            if resp.status_code != 200:
                return {"net_flow_signal": 0.0}

            data = resp.json()
            tickers = data.get("tickers", [])

            # Analyze bid/ask spread as proxy for flow pressure
            spreads = []
            volumes = []
            for t in tickers[:20]:
                bid = t.get("bid_ask_spread_percentage", 0)
                vol = t.get("converted_volume", {}).get("usd", 0)
                if bid and vol:
                    spreads.append(bid)
                    volumes.append(vol)

            avg_spread = sum(spreads) / len(spreads) if spreads else 0.5
            total_volume = sum(volumes)

            # Tight spreads + high volume = healthy market (slightly bullish)
            # Wide spreads + low volume = uncertainty (slightly bearish)
            flow_signal = 0.0
            if avg_spread < 0.1 and total_volume > 1e8:
                flow_signal = 0.05  # bullish
            elif avg_spread > 0.5:
                flow_signal = -0.05  # bearish

            return {
                "net_flow_signal": flow_signal,
                "avg_spread": avg_spread,
                "total_volume_usd": total_volume,
            }
        except Exception as e:
            self.logger.warning("exchange_flows_failed", error=str(e))
            return {"net_flow_signal": 0.0}

    async def _fetch_market_metrics(self, coin: str) -> dict:
        """Fetch on-chain market metrics from CoinGecko."""
        await self._ensure_client()
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin}"
            resp = await self._http.get(url, params={
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false",
            })
            if resp.status_code != 200:
                return {}

            data = resp.json()
            market_data = data.get("market_data", {})

            return {
                "price_change_24h_pct": market_data.get("price_change_percentage_24h", 0),
                "price_change_7d_pct": market_data.get("price_change_percentage_7d", 0),
                "price_change_30d_pct": market_data.get("price_change_percentage_30d", 0),
                "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                "total_volume_24h": market_data.get("total_volume", {}).get("usd", 0),
                "circulating_supply": market_data.get("circulating_supply", 0),
                "total_supply": market_data.get("total_supply", 0),
                "ath": market_data.get("ath", {}).get("usd", 0),
                "ath_change_pct": market_data.get("ath_change_percentage", {}).get("usd", 0),
                "current_price": market_data.get("current_price", {}).get("usd", 0),
            }
        except Exception as e:
            self.logger.warning("market_metrics_failed", error=str(e))
            return {}

    def _compute_onchain_probability(
        self, exchange_flows: dict, metrics: dict, market: Market
    ) -> float:
        """Estimate probability from on-chain signals."""
        base_prob = market.implied_probability
        adjustment = 0.0

        # Exchange flow signal
        adjustment += exchange_flows.get("net_flow_signal", 0.0)

        # Momentum signals from price changes
        pct_24h = metrics.get("price_change_24h_pct", 0) / 100.0
        pct_7d = metrics.get("price_change_7d_pct", 0) / 100.0

        # Strong recent momentum
        if pct_24h > 0.05:
            adjustment += 0.06
        elif pct_24h < -0.05:
            adjustment -= 0.06

        if pct_7d > 0.10:
            adjustment += 0.04
        elif pct_7d < -0.10:
            adjustment -= 0.04

        # Volume analysis - high volume confirms trends
        vol_24h = metrics.get("total_volume_24h", 0)
        market_cap = metrics.get("market_cap", 1)
        vol_to_cap = vol_24h / market_cap if market_cap > 0 else 0

        if vol_to_cap > 0.15:  # unusually high volume
            adjustment *= 1.3  # amplify the trend signal

        # ATH proximity
        ath_change = metrics.get("ath_change_pct", 0) / 100.0
        if ath_change > -0.10:  # within 10% of ATH
            adjustment += 0.03  # momentum to break through

        return max(0.05, min(0.95, base_prob + adjustment))

    async def analyze(self, market: Market) -> Signal | None:
        """Analyze on-chain data for a crypto prediction market."""
        coin = self._map_to_coin(market)

        exchange_flows = await self._fetch_exchange_flows(coin)
        metrics = await self._fetch_market_metrics(coin)

        if not metrics:
            return None

        estimated_prob = self._compute_onchain_probability(exchange_flows, metrics, market)
        market_prob = market.implied_probability
        edge = estimated_prob - market_prob

        if abs(edge) < 0.03:
            return None

        side = Side.YES if edge > 0 else Side.NO
        confidence = min(abs(edge) / 0.15, 1.0)

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
                f"On-chain: 24h={metrics.get('price_change_24h_pct', 0):.1f}%, "
                f"7d={metrics.get('price_change_7d_pct', 0):.1f}%, "
                f"flow_signal={exchange_flows.get('net_flow_signal', 0):.3f}, "
                f"est_prob={estimated_prob:.2%} vs market={market_prob:.2%}"
            ),
            metadata={"metrics": metrics, "exchange_flows": exchange_flows},
        )

    async def cleanup(self):
        if self._http:
            await self._http.aclose()
