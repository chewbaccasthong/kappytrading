"""Web dashboard for monitoring the Kappy Trading Bot.

Tracks trades, signals, edges, agent performance, and lets you
adjust risk parameters live.

Usage:
    python -m dashboard.app
    python -m dashboard.app --port 8080
"""

import argparse
import json
from datetime import datetime, date

from flask import Flask, render_template, jsonify, request

from config.settings import settings
from utils.database import Database

app = Flask(__name__, template_folder="templates")
db = Database()


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/performance")
def api_performance():
    """Get overall performance summary."""
    perf = db.get_performance_summary()
    return jsonify(perf)


@app.route("/api/trades")
def api_trades():
    """Get recent trades."""
    limit = request.args.get("limit", 50, type=int)
    trades = db.get_recent_trades(limit)
    return jsonify(trades)


@app.route("/api/signals")
def api_signals():
    """Get recent signals from all agents."""
    limit = request.args.get("limit", 100, type=int)
    with db.Session() as session:
        from utils.database import SignalRecord
        records = (
            session.query(SignalRecord)
            .order_by(SignalRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        signals = [
            {
                "id": r.id,
                "agent": r.agent_name,
                "market": r.market_ticker,
                "side": r.side,
                "confidence": r.confidence,
                "probability": r.estimated_probability,
                "edge": r.edge,
                "reasoning": r.reasoning,
                "time": str(r.created_at),
            }
            for r in records
        ]
    return jsonify(signals)


@app.route("/api/agents")
def api_agents():
    """Get per-agent performance breakdown."""
    with db.Session() as session:
        from utils.database import SignalRecord, TradeRecord
        from sqlalchemy import func

        # Signal counts per agent
        signal_stats = (
            session.query(
                SignalRecord.agent_name,
                func.count(SignalRecord.id).label("signal_count"),
                func.avg(SignalRecord.confidence).label("avg_confidence"),
                func.avg(SignalRecord.edge).label("avg_edge"),
            )
            .group_by(SignalRecord.agent_name)
            .all()
        )

        agents = []
        for stat in signal_stats:
            agents.append({
                "name": stat.agent_name,
                "signal_count": stat.signal_count,
                "avg_confidence": round(float(stat.avg_confidence or 0), 4),
                "avg_edge": round(float(stat.avg_edge or 0), 4),
            })

    return jsonify(agents)


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    """Get current bot settings."""
    return jsonify({
        "max_position_size": settings.max_position_size,
        "max_daily_loss": settings.max_daily_loss,
        "max_open_positions": settings.max_open_positions,
        "min_edge_threshold": settings.min_edge_threshold,
        "kalshi_demo": settings.kalshi_demo,
        "kalshi_base_url": settings.kalshi_base_url,
        "log_level": settings.log_level,
    })


@app.route("/api/settings", methods=["POST"])
def api_update_settings():
    """Update bot settings (in memory only — restart to reset)."""
    data = request.json
    if "max_position_size" in data:
        settings.max_position_size = int(data["max_position_size"])
    if "max_daily_loss" in data:
        settings.max_daily_loss = int(data["max_daily_loss"])
    if "max_open_positions" in data:
        settings.max_open_positions = int(data["max_open_positions"])
    if "min_edge_threshold" in data:
        settings.min_edge_threshold = float(data["min_edge_threshold"])
    return jsonify({"status": "ok"})


@app.route("/api/equity")
def api_equity():
    """Get equity curve data from performance records."""
    with db.Session() as session:
        from utils.database import PerformanceRecord
        records = (
            session.query(PerformanceRecord)
            .order_by(PerformanceRecord.date.asc())
            .all()
        )
        curve = [
            {"date": r.date, "balance": r.balance, "pnl": r.daily_pnl}
            for r in records
        ]
    return jsonify(curve)


def main():
    parser = argparse.ArgumentParser(description="Kappy Trading Dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"\n  Kappy Trading Dashboard")
    print(f"  http://{args.host}:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
