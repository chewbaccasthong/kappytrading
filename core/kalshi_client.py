"""Kalshi API client for trading prediction markets.

Handles authentication, market data, order placement, and position management.
Uses Kalshi's v2 REST API with HMAC-SHA256 authentication.
"""

import hashlib
import hmac
import time
import base64
from datetime import datetime
from typing import Optional

import httpx
import structlog

from config.settings import settings
from core.models import Market, Order, Position, Side, OrderStatus

logger = structlog.get_logger(__name__)


class KalshiClient:
    """Async client for Kalshi's trading API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.kalshi.api_key
        self.api_secret = api_secret or settings.kalshi.api_secret
        self.base_url = (base_url or settings.kalshi.base_url).rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def _get_auth_headers(self, method: str, path: str, body: str = "") -> dict:
        """Generate HMAC-SHA256 authentication headers for Kalshi API."""
        timestamp = str(int(time.time() * 1000))
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    async def _request(
        self, method: str, path: str, params: Optional[dict] = None, json: Optional[dict] = None
    ) -> dict:
        """Make an authenticated request to the Kalshi API."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        url = f"{self.base_url}{path}"
        body = ""
        if json:
            import json as json_lib
            body = json_lib.dumps(json, separators=(",", ":"))

        headers = self._get_auth_headers(method, path, body)

        response = await self._client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            content=body if body else None,
        )

        if response.status_code >= 400:
            logger.error(
                "kalshi_api_error",
                status=response.status_code,
                path=path,
                body=response.text,
            )
            response.raise_for_status()

        return response.json()

    # ── Market Data ──────────────────────────────────────────────

    async def get_markets(
        self,
        category: str = "Crypto",
        status: str = "open",
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> list[Market]:
        """Fetch available crypto prediction markets."""
        params = {
            "status": status,
            "limit": limit,
            "series_ticker": category,
        }
        if cursor:
            params["cursor"] = cursor

        data = await self._request("GET", "/markets", params=params)
        markets = []

        for m in data.get("markets", []):
            try:
                markets.append(
                    Market(
                        ticker=m["ticker"],
                        title=m.get("title", ""),
                        category=m.get("category", ""),
                        end_date=datetime.fromisoformat(
                            m.get("close_time", m.get("expiration_time", "2099-01-01"))
                        ),
                        yes_price=m.get("yes_ask", 0) / 100.0 if m.get("yes_ask") else 0.50,
                        no_price=m.get("no_ask", 0) / 100.0 if m.get("no_ask") else 0.50,
                        volume=m.get("volume", 0),
                        open_interest=m.get("open_interest", 0),
                        status=m.get("status", "open"),
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning("failed_to_parse_market", error=str(e), market=m.get("ticker"))

        return markets

    async def get_market(self, ticker: str) -> Market:
        """Fetch a single market by ticker."""
        data = await self._request("GET", f"/markets/{ticker}")
        m = data["market"]
        return Market(
            ticker=m["ticker"],
            title=m.get("title", ""),
            category=m.get("category", ""),
            end_date=datetime.fromisoformat(
                m.get("close_time", m.get("expiration_time", "2099-01-01"))
            ),
            yes_price=m.get("yes_ask", 50) / 100.0,
            no_price=m.get("no_ask", 50) / 100.0,
            volume=m.get("volume", 0),
            open_interest=m.get("open_interest", 0),
            status=m.get("status", "open"),
        )

    async def get_orderbook(self, ticker: str) -> dict:
        """Fetch the orderbook for a market."""
        return await self._request("GET", f"/markets/{ticker}/orderbook")

    async def get_market_history(self, ticker: str, limit: int = 100) -> list[dict]:
        """Fetch trade history for a market."""
        data = await self._request(
            "GET", f"/markets/{ticker}/trades", params={"limit": limit}
        )
        return data.get("trades", [])

    async def search_crypto_markets(self) -> list[Market]:
        """Search for all open crypto-related prediction markets."""
        all_markets = []
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol"]

        data = await self._request(
            "GET", "/markets", params={"status": "open", "limit": 200}
        )

        for m in data.get("markets", []):
            title_lower = m.get("title", "").lower()
            ticker_lower = m.get("ticker", "").lower()
            if any(kw in title_lower or kw in ticker_lower for kw in crypto_keywords):
                try:
                    all_markets.append(
                        Market(
                            ticker=m["ticker"],
                            title=m.get("title", ""),
                            category=m.get("category", ""),
                            end_date=datetime.fromisoformat(
                                m.get("close_time", m.get("expiration_time", "2099-01-01"))
                            ),
                            yes_price=m.get("yes_ask", 50) / 100.0,
                            no_price=m.get("no_ask", 50) / 100.0,
                            volume=m.get("volume", 0),
                            open_interest=m.get("open_interest", 0),
                            status=m.get("status", "open"),
                        )
                    )
                except (KeyError, ValueError):
                    continue

        return all_markets

    # ── Trading ──────────────────────────────────────────────────

    async def place_order(
        self,
        ticker: str,
        side: Side,
        quantity: int,
        price: int,  # price in cents (1-99)
    ) -> Order:
        """Place a limit order on a market.

        Args:
            ticker: Market ticker
            side: 'yes' or 'no'
            quantity: Number of contracts
            price: Limit price in cents (1-99)
        """
        payload = {
            "ticker": ticker,
            "action": "buy",
            "side": side.value,
            "count": quantity,
            "type": "limit",
            "yes_price" if side == Side.YES else "no_price": price,
        }

        logger.info("placing_order", ticker=ticker, side=side.value, qty=quantity, price=price)

        data = await self._request("POST", "/portfolio/orders", json=payload)
        order_data = data.get("order", {})

        return Order(
            market_ticker=ticker,
            side=side,
            quantity=quantity,
            price=price / 100.0,
            order_id=order_data.get("order_id"),
            status=OrderStatus.PENDING,
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            await self._request("DELETE", f"/portfolio/orders/{order_id}")
            logger.info("order_cancelled", order_id=order_id)
            return True
        except httpx.HTTPStatusError:
            logger.warning("cancel_order_failed", order_id=order_id)
            return False

    # ── Portfolio ────────────────────────────────────────────────

    async def get_balance(self) -> float:
        """Get current account balance in dollars."""
        data = await self._request("GET", "/portfolio/balance")
        return data.get("balance", 0) / 100.0  # cents to dollars

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        data = await self._request("GET", "/portfolio/positions")
        positions = []

        for p in data.get("market_positions", []):
            qty = p.get("position", 0)
            if qty == 0:
                continue
            side = Side.YES if qty > 0 else Side.NO
            positions.append(
                Position(
                    market_ticker=p["ticker"],
                    side=side,
                    quantity=abs(qty),
                    avg_price=p.get("average_price", 0) / 100.0,
                )
            )

        return positions

    async def get_open_orders(self) -> list[Order]:
        """Get all open/pending orders."""
        data = await self._request("GET", "/portfolio/orders", params={"status": "resting"})
        orders = []

        for o in data.get("orders", []):
            orders.append(
                Order(
                    market_ticker=o["ticker"],
                    side=Side.YES if o.get("side") == "yes" else Side.NO,
                    quantity=o.get("remaining_count", 0),
                    price=o.get("yes_price", o.get("no_price", 0)) / 100.0,
                    order_id=o.get("order_id"),
                    status=OrderStatus.PENDING,
                )
            )

        return orders
