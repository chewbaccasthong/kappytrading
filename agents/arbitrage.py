"""Arbitrage agent - compares Kalshi prices to external implied probabilities.

Cross-references Kalshi prediction market prices with:
- Crypto derivatives (options implied volatility)
- Other prediction markets (Polymarket, etc.)
- Crypto futures basis/funding rates
to find pricing discrepancies.
"""

import re
from datetime import datetime

import httpx
import ccxt.async_support as ccxt
import numpy as np

from agents.base import BaseAgent
from core.models import Market, Signal, Side, SignalStrength


class ArbitrageAgent(BaseAgent):
    """Finds arbitrage opportunities between Kalshi prices and external
    implied probabilities from derivatives and other sources."""

    def __init__(self, weight: float = 1.5):
        super().__init__(name="arbitrage", weight=weight)
        self._exchange = ccxt.binance({"enableRateLimit": True})
        self._http: httpx.AsyncClient | None = None

    async def _ensure_client(self):
        if not self._http:
            self._http = httpx.AsyncClient(timeout=15.0)

    def _extract_target(self, market: Market) -> tuple[str, float | None, str | None]:
        """Extract crypto pair, target price, and direction from market title."""
        title = market.title.lower()
        symbol_map = {
            "bitcoin": "BTC", "btc": "BTC",
            "ethereum": "ETH", "eth": "ETH",
            "solana": "SOL", "sol": "SOL",
        }

        symbol = "BTC"
        for name, sym in symbol_map.items():
            if name in title:
                symbol = sym
                break

        price_match = re.search(r"\$?([\d,]+\.?\d*)\s*k?", title)
        target = None
        if price_match:
            target = float(price_match.group(1).replace(",", ""))
            if "k" in title[price_match.end() - 1 : price_match.end() + 1]:
                target *= 1000

        direction = None
        if any(w in title for w in ["above", "over", "exceed", "higher", "reach"]):
            direction = "above"
        elif any(w in title for w in ["below", "under", "lower", "drop"]):
            direction = "below"

        return (symbol, target, direction)

    async def _get_futures_basis(self, symbol: str) -> dict:
        """Analyze futures basis and funding rate for directional signal.

        Positive funding = longs pay shorts = bullish consensus.
        Positive basis = futures > spot = bullish.
        """
        try:
            # Get spot price
            ticker = await self._exchange.fetch_ticker(f"{symbol}/USDT")
            spot_price = ticker["last"]

            # Get funding rate from Binance futures
            await self._ensure_client()
            resp = await self._http.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": f"{symbol}USDT", "limit": 1},
            )
            funding_rate = 0.0
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    funding_rate = float(data[0].get("fundingRate", 0))

            # Get futures price
            resp2 = await self._http.get(
                "https://fapi.binance.com/fapi/v1/ticker/price",
                params={"symbol": f"{symbol}USDT"},
            )
            futures_price = spot_price
            if resp2.status_code == 200:
                futures_price = float(resp2.json().get("price", spot_price))

            basis = (futures_price - spot_price) / spot_price if spot_price > 0 else 0

            return {
                "spot_price": spot_price,
                "futures_price": futures_price,
                "basis": basis,
                "funding_rate": funding_rate,
                "bullish_signal": basis > 0.001 or funding_rate > 0.0005,
                "bearish_signal": basis < -0.001 or funding_rate < -0.0005,
            }
        except Exception as e:
            self.logger.warning("futures_basis_failed", symbol=symbol, error=str(e))
            return {"spot_price": 0, "basis": 0, "funding_rate": 0}

    async def _get_options_implied_vol(self, symbol: str) -> dict:
        """Fetch implied volatility from Deribit options (via public API)."""
        await self._ensure_client()
        try:
            currency = symbol.upper()
            url = f"https://www.deribit.com/api/v2/public/get_historical_volatility"
            resp = await self._http.get(url, params={"currency": currency})
            if resp.status_code != 200:
                return {"implied_vol": 0.80}

            data = resp.json()
            vols = data.get("result", [])
            if vols:
                latest_vol = vols[-1][1] / 100.0 if isinstance(vols[-1], list) else 0.80
                return {"implied_vol": latest_vol}
            return {"implied_vol": 0.80}
        except Exception as e:
            self.logger.warning("options_iv_failed", error=str(e))
            return {"implied_vol": 0.80}

    def _compute_arb_probability(
        self,
        futures_data: dict,
        options_data: dict,
        target_price: float | None,
        direction: str | None,
        market: Market,
    ) -> float:
        """Estimate probability using derivatives pricing data.

        Uses a simplified Black-Scholes-like approach with implied vol
        and spot/futures basis to estimate target probability.
        """
        spot = futures_data.get("spot_price", 0)
        if spot == 0 or target_price is None:
            # Can't compute arb without target - use directional signals
            base = market.implied_probability
            if futures_data.get("bullish_signal"):
                return min(base + 0.06, 0.95)
            elif futures_data.get("bearish_signal"):
                return max(base - 0.06, 0.05)
            return base

        iv = options_data.get("implied_vol", 0.80)
        days_to_expiry = max((market.end_date - datetime.utcnow()).total_seconds() / 86400, 0.1)
        time_sqrt = np.sqrt(days_to_expiry / 365.0)

        # Drift adjustment from futures basis (annualized)
        basis = futures_data.get("basis", 0)
        drift = basis * (365 / max(days_to_expiry, 1))

        # Log-normal probability calculation
        if direction == "above":
            d = (np.log(spot / target_price) + (drift + 0.5 * iv**2) * (days_to_expiry / 365)) / (
                iv * time_sqrt
            ) if iv * time_sqrt > 0 else 0
            from scipy.stats import norm
            prob = norm.cdf(d)
        elif direction == "below":
            d = (np.log(target_price / spot) + (-drift + 0.5 * iv**2) * (days_to_expiry / 365)) / (
                iv * time_sqrt
            ) if iv * time_sqrt > 0 else 0
            from scipy.stats import norm
            prob = norm.cdf(d)
        else:
            prob = market.implied_probability

        # Funding rate adjustment
        funding = futures_data.get("funding_rate", 0)
        if funding > 0.001:
            prob += 0.02 if direction == "above" else -0.02
        elif funding < -0.001:
            prob -= 0.02 if direction == "above" else 0.02

        return max(0.02, min(0.98, prob))

    async def analyze(self, market: Market) -> Signal | None:
        """Find arbitrage between Kalshi prices and derivatives-implied probabilities."""
        symbol, target_price, direction = self._extract_target(market)

        futures_data = await self._get_futures_basis(symbol)
        options_data = await self._get_options_implied_vol(symbol)

        estimated_prob = self._compute_arb_probability(
            futures_data, options_data, target_price, direction, market
        )
        market_prob = market.implied_probability
        edge = estimated_prob - market_prob

        if abs(edge) < 0.04:  # higher threshold for arb - need larger edge
            return None

        side = Side.YES if edge > 0 else Side.NO
        confidence = min(abs(edge) / 0.15, 1.0)

        # Arb signals are higher conviction when derivatives agree
        if futures_data.get("bullish_signal") and side == Side.YES:
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
                f"Arb: spot={futures_data.get('spot_price', 0):.0f}, "
                f"basis={futures_data.get('basis', 0):.4f}, "
                f"funding={futures_data.get('funding_rate', 0):.5f}, "
                f"IV={options_data.get('implied_vol', 0):.0%}, "
                f"est_prob={estimated_prob:.2%} vs market={market_prob:.2%}"
            ),
            metadata={"futures": futures_data, "options": options_data},
        )

    async def cleanup(self):
        await self._exchange.close()
        if self._http:
            await self._http.aclose()
