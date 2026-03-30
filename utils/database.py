"""SQLite database for trade history and performance tracking."""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import settings

Base = declarative_base()


class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_ticker = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    edge = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    signals_json = Column(Text, nullable=True)
    status = Column(String, default="open")
    created_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)


class SignalRecord(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_name = Column(String, nullable=False, index=True)
    market_ticker = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    estimated_probability = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PerformanceRecord(Base):
    __tablename__ = "performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, nullable=False, index=True)
    balance = Column(Float, nullable=False)
    daily_pnl = Column(Float, nullable=False)
    trades_count = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    avg_edge = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Database:
    """Trade database manager."""

    def __init__(self, url: str | None = None):
        db_url = url or settings.database_url
        # Ensure data directory exists
        if "sqlite" in db_url:
            db_path = db_url.split("///")[-1]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            # Use sync sqlite for simplicity
            db_url = db_url.replace("sqlite+aiosqlite", "sqlite")

        self.engine = create_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

    def record_trade(self, trade_data: dict):
        """Record a new trade."""
        with self.Session() as session:
            record = TradeRecord(
                market_ticker=trade_data["market_ticker"],
                side=trade_data["side"],
                quantity=trade_data["quantity"],
                entry_price=trade_data["entry_price"],
                edge=trade_data["edge"],
                confidence=trade_data["confidence"],
                signals_json=json.dumps(trade_data.get("signals", [])),
            )
            session.add(record)
            session.commit()
            return record.id

    def record_signal(self, signal_data: dict):
        """Record a signal for analysis."""
        with self.Session() as session:
            record = SignalRecord(**signal_data)
            session.add(record)
            session.commit()

    def close_trade(self, trade_id: int, exit_price: float, pnl: float):
        """Close a trade with result."""
        with self.Session() as session:
            trade = session.query(TradeRecord).filter_by(id=trade_id).first()
            if trade:
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.status = "closed"
                trade.closed_at = datetime.utcnow()
                session.commit()

    def record_daily_performance(self, stats: dict):
        """Record end-of-day performance."""
        with self.Session() as session:
            record = PerformanceRecord(**stats)
            session.add(record)
            session.commit()

    def get_recent_trades(self, limit: int = 50) -> list[dict]:
        """Get recent trades."""
        with self.Session() as session:
            trades = (
                session.query(TradeRecord)
                .order_by(TradeRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": t.id,
                    "ticker": t.market_ticker,
                    "side": t.side,
                    "qty": t.quantity,
                    "entry": t.entry_price,
                    "exit": t.exit_price,
                    "pnl": t.pnl,
                    "edge": t.edge,
                    "confidence": t.confidence,
                    "status": t.status,
                    "created": str(t.created_at),
                }
                for t in trades
            ]

    def get_performance_summary(self) -> dict:
        """Get overall performance summary."""
        with self.Session() as session:
            trades = session.query(TradeRecord).filter_by(status="closed").all()
            if not trades:
                return {"total_trades": 0, "total_pnl": 0, "win_rate": 0}

            total_pnl = sum(t.pnl or 0 for t in trades)
            winners = [t for t in trades if (t.pnl or 0) > 0]
            losers = [t for t in trades if (t.pnl or 0) < 0]

            return {
                "total_trades": len(trades),
                "total_pnl": total_pnl,
                "win_count": len(winners),
                "loss_count": len(losers),
                "win_rate": len(winners) / len(trades) if trades else 0,
                "avg_win": sum(t.pnl for t in winners) / len(winners) if winners else 0,
                "avg_loss": sum(t.pnl for t in losers) / len(losers) if losers else 0,
                "avg_edge": sum(t.edge for t in trades) / len(trades),
            }
