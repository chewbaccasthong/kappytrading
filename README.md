# Kappy Trading Bot

Automated trading bot for Kalshi crypto prediction markets. Uses multiple research agents to find edges, aggregates signals, applies risk management, and executes trades.

## Architecture

```
Market Discovery → Agent Analysis (parallel) → Signal Aggregation → Risk Check → Execution
```

### Research Agents

| Agent | What It Does | Data Source |
|-------|-------------|-------------|
| **Technical Analysis** | RSI, MACD, Bollinger Bands, trend detection | Binance OHLCV via ccxt |
| **Sentiment** | News & community sentiment scoring | CryptoPanic, CoinGecko |
| **On-Chain** | Exchange flows, volume analysis, momentum | CoinGecko market data |
| **Microstructure** | Kalshi orderbook depth, imbalance, trade flow | Kalshi API |
| **Arbitrage** | Cross-venue probability comparison (options IV, futures basis) | Binance futures, Deribit |

### Strategy Engine

- Weighted signal aggregation across all agents
- Requires minimum 2 agents to agree on direction
- Filters by minimum edge threshold (default 5%)
- Ranks opportunities by expected value

### Risk Management

- Half-Kelly criterion position sizing
- Maximum position size per market
- Daily loss limit circuit breaker
- Maximum open positions cap
- Portfolio exposure limits

## Setup

```bash
# Clone and install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your Kalshi API credentials
```

## Usage

```bash
# Dry run (no real trades - recommended to start)
python main.py

# Single cycle dry run
python main.py --once

# Live trading (real money!)
python main.py --live

# Custom interval (seconds between cycles)
python main.py --interval 120

# View performance stats
python main.py --status

# Run backtest
python -m scripts.backtest --days 30

# Live monitoring dashboard
python -m scripts.monitor
```

## Configuration

All settings via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `KALSHI_API_KEY` | - | Your Kalshi API key |
| `KALSHI_API_SECRET` | - | Your Kalshi API secret |
| `MAX_POSITION_SIZE` | 50 | Max contracts per market |
| `MAX_DAILY_LOSS` | 100 | Daily loss limit ($) |
| `MAX_OPEN_POSITIONS` | 10 | Max concurrent positions |
| `MIN_EDGE_THRESHOLD` | 0.05 | Min edge to trade (5%) |

## Project Structure

```
kappytrading/
├── main.py                  # Entry point
├── config/
│   └── settings.py          # Configuration via pydantic-settings
├── core/
│   ├── models.py            # Data models (Market, Signal, Order, etc.)
│   ├── kalshi_client.py     # Kalshi API client
│   └── orchestrator.py      # Main trading loop
├── agents/
│   ├── base.py              # Agent base class
│   ├── technical.py         # Technical analysis
│   ├── sentiment.py         # Sentiment analysis
│   ├── onchain.py           # On-chain data
│   ├── microstructure.py    # Orderbook analysis
│   └── arbitrage.py         # Cross-venue arbitrage
├── strategies/
│   ├── signal_aggregator.py # Signal combination
│   └── risk_manager.py      # Risk management
├── utils/
│   ├── logger.py            # Structured logging
│   └── database.py          # Trade history DB
├── scripts/
│   ├── backtest.py          # Strategy backtester
│   └── monitor.py           # Live dashboard
└── data/                    # SQLite database storage
```

## How It Works

1. **Discovery**: Scans Kalshi for open crypto prediction markets (BTC, ETH, SOL, etc.)
2. **Analysis**: Runs all 5 research agents in parallel on each market
3. **Aggregation**: Combines agent signals using weighted averaging with confidence scaling
4. **Filtering**: Only proceeds when 2+ agents agree and edge exceeds threshold
5. **Sizing**: Calculates position size using half-Kelly criterion
6. **Risk Check**: Validates against daily loss limits, position limits, and exposure caps
7. **Execution**: Places limit orders on Kalshi (or logs them in dry-run mode)
8. **Monitoring**: Records all signals and trades to SQLite for analysis

## Disclaimer

This software is for educational and research purposes. Trading prediction markets involves risk of loss. Always start with dry-run mode and small positions. Past performance does not guarantee future results.
