"""Kalshi API client for trading prediction markets.

Handles authentication, market data, order placement, and position management.
Uses Kalshi's v2 REST API with RSA-PSS request signing.

Auth spec: https://trading-api.readme.io/reference/authentication
"""

import json as json_lib
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config.settings import settings
from core.models import Market, Order, Position, Side, OrderStatus

logger = structlog.get_logger(__name__)


class KalshiClient:
    """Async client for Kalshi's trading API with RSA-PSS auth."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key_path: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.kalshi_api_key
        self._private_key_path = private_key_path or settings.kalshi_private_key_path
        self.base_url = (base_url or settings.kalshi_base_url).rstrip("/")
        self._private_key = None
        self._client: Optional[httpx.AsyncClient] = None

    def _load_private_key(self):
        """Load RSA private key from PEM file, auto-fixing common format issues."""
        if self._private_key is not None:
            return
        path = Path(self._private_key_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Kalshi private key not found at: {path}\n"
                "Set KALSHI_PRIVATE_KEY_PATH in your .env file."
            )

        raw = path.read_text(encoding="utf-8")

        # Fix common issues: extra blank lines, Windows \r\n, trailing whitespace
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        cleaned = "\n".join(lines) + "\n"

        try:
            self._private_key = serialization.load_pem_private_key(
                cleaned.encode("utf-8"), password=None
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load private key from {path}.\n"
                f"Make sure the file contains a valid PEM key with no extra blank lines.\n"
                f"Error: {e}"
            ) from e

    def _sign_request(self, method: str, path: str) -> dict:
        """Generate RSA-PSS signed request headers.

        Kalshi signing format:
            message = timestamp (ms) + method.upper() + path (no query string)
        """
        self._load_private_key()
        timestamp_ms = str(int(time.time() * 1000))
        message = (timestamp_ms + method.upper() + path).encode("utf-8")

        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        import base64
        sig_b64 = base64.b64encode(signature).decode("utf-8")

        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "Content-Type": "application/json",
        }

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> dict:
        """Make a signed request to the Kalshi API."""
        if not self._client:
            raise RuntimeError("Client not started. Use 'async with KalshiClient()' context manager.")

        url = f"{self.base_url}{path}"
        headers = self._sign_request(method, path)
        body = json_lib.dumps(json, separators=(",", ":")) if json else None

        response = await self._client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            content=body,
        )

        if response.status_code >= 400:
            logger.error(
                "kalshi_api_error",
                status=response.status_code,
                path=path,
                body=response.text[:300],
            )
            response.raise_for_status()

        return response.json()

    # ── Market Data ──────────────────────────────────────────────

    async def search_crypto_markets(self) -> list[Market]:
        """Search for all open crypto-related prediction markets."""
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol"]

        try:
            data = await self._request(
                "GET", "/markets", params={"status": "open", "limit": 200}
            )
        except Exception as e:
            logger.error("market_search_failed", error=str(e))
            return []

        markets = []
        for m in data.get("markets", []):
            title_lower = m.get("title", "").lower()
            ticker_lower = m.get("ticker", "").lower()
            if not any(kw in title_lower or kw in ticker_lower for kw in crypto_keywords):
                continue
            market = self._parse_market(m)
            if market:
                markets.append(market)

        logger.info("crypto_markets_found", count=len(markets))
        return markets

    async def get_market(self, ticker: str) -> Optional[Market]:
        """Fetch a single market by ticker."""
        try:
            data = await self._request("GET", f"/markets/{ticker}")
            return self._parse_market(data.get("market", {}))
        except Exception as e:
            logger.warning("get_market_failed", ticker=ticker, error=str(e))
            return None

    async def get_orderbook(self, ticker: str) -> dict:
        """Fetch the orderbook for a market."""
        try:
            return await self._request("GET", f"/markets/{ticker}/orderbook")
        except Exception as e:
            logger.warning("orderbook_failed", ticker=ticker, error=str(e))
            return {}

    async def get_market_history(self, ticker: str, limit: int = 50) -> list[dict]:
        """Fetch recent trades for a market."""
        try:
            data = await self._request(
                "GET", f"/markets/{ticker}/trades", params={"limit": limit}
            )
            return data.get("trades", [])
        except Exception as e:
            logger.warning("trade_history_failed", ticker=ticker, error=str(e))
            return []

    def _parse_market(self, m: dict) -> Optional[Market]:
        """Parse a market dict from the API into a Market model."""
        if not m.get("ticker"):
            return None
        try:
            # Kalshi v2 returns prices in cents (1–99)
            yes_ask = m.get("yes_ask", 50)
            no_ask = m.get("no_ask", 50)
            close_time = m.get("close_time") or m.get("expiration_time") or "2099-01-01T00:00:00Z"
            # Strip microseconds Kalshi sometimes returns with 7 digits
            if "." in close_time:
                close_time = close_time[:26] + "Z" if close_time.endswith("Z") else close_time[:26]

            return Market(
                ticker=m["ticker"],
                title=m.get("title", ""),
                category=m.get("category", ""),
                end_date=datetime.fromisoformat(close_time.replace("Z", "+00:00")),
                yes_price=yes_ask / 100.0,
                no_price=no_ask / 100.0,
                volume=m.get("volume", 0) or 0,
                open_interest=m.get("open_interest", 0) or 0,
                status=m.get("status", "open"),
            )
        except Exception as e:
            logger.warning("market_parse_failed", ticker=m.get("ticker"), error=str(e))
            return None

    # ── Trading ──────────────────────────────────────────────────

    async def place_order(
        self,
        ticker: str,
        side: Side,
        quantity: int,
        price: int,  # cents (1–99)
    ) -> Order:
        """Place a limit order on a market."""
        payload = {
            "ticker": ticker,
            "action": "buy",
            "side": side.value,
            "count": quantity,
            "type": "limit",
        }
        if side == Side.YES:
            payload["yes_price"] = price
        else:
            payload["no_price"] = price

        logger.info(
            "placing_order",
            ticker=ticker,
            side=side.value,
            qty=quantity,
            price_cents=price,
        )

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
        except Exception as e:
            logger.warning("cancel_failed", order_id=order_id, error=str(e))
            return False

    # ── Portfolio ────────────────────────────────────────────────

    async def get_balance(self) -> float:
        """Get current account balance in dollars."""
        try:
            data = await self._request("GET", "/portfolio/balance")
            # Kalshi returns balance in cents
            return data.get("balance", 0) / 100.0
        except Exception as e:
            logger.error("balance_fetch_failed", error=str(e))
            return 0.0

    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        try:
            data = await self._request("GET", "/portfolio/positions")
        except Exception as e:
            logger.error("positions_fetch_failed", error=str(e))
            return []

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
        """Get all resting (open) orders."""
        try:
            data = await self._request(
                "GET", "/portfolio/orders", params={"status": "resting"}
            )
        except Exception as e:
            logger.error("open_orders_fetch_failed", error=str(e))
            return []

        orders = []
        for o in data.get("orders", []):
            price_cents = o.get("yes_price") or o.get("no_price") or 50
            orders.append(
                Order(
                    market_ticker=o["ticker"],
                    side=Side.YES if o.get("side") == "yes" else Side.NO,
                    quantity=o.get("remaining_count", 0),
                    price=price_cents / 100.0,
                    order_id=o.get("order_id"),
                    status=OrderStatus.PENDING,
                )
            )
        return orders
