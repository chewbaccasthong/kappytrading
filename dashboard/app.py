"""Web dashboard for monitoring the Kappy Trading Bot.

Tracks trades, signals, edges, agent performance, trade grades,
live balance, open positions, and lets you adjust parameters.

Usage:
    python -m dashboard.app
    python -m dashboard.app --port 8080
"""

import argparse
from datetime import datetime

from flask import Flask, render_template, jsonify, request

from config.settings import settings
from utils.database import Database
from strategies.trade_grader import TradeGrader

app = Flask(__name__, template_folder="templates")
db = Database()
grader = TradeGrader(db)

# The orchestrator can register itself here for live data
_orchestrator = None


def set_orchestrator(orch):
    """Called by main.py to share the running orchestrator instance."""
    global _orchestrator
    _orchestrator = orch


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/performance")
def api_performance():
    perf = db.get_performance_summary()
    return jsonify(perf)


@app.route("/api/trades")
def api_trades():
    limit = request.args.get("limit", 50, type=int)
    trades = db.get_recent_trades(limit)
    return jsonify(trades)


@app.route("/api/signals")
def api_signals():
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
    """Per-agent performance with grading data."""
    # DB signal stats
    with db.Session() as session:
        from utils.database import SignalRecord
        from sqlalchemy import func
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
        db_agents = {s.agent_name: s for s in signal_stats}

    # Grader report (accuracy, recommended weights)
    grader_report = {r["agent"]: r for r in grader.get_agent_report()}

    agents = []
    all_names = set(list(db_agents.keys()) + list(grader_report.keys()))
    for name in all_names:
        db_s = db_agents.get(name)
        gr = grader_report.get(name, {})
        agents.append({
            "name": name,
            "signal_count": db_s.signal_count if db_s else gr.get("total_signals", 0),
            "avg_confidence": round(float(db_s.avg_confidence or 0), 4) if db_s else 0,
            "avg_edge": round(float(db_s.avg_edge or 0), 4) if db_s else 0,
            "accuracy": gr.get("accuracy", 0),
            "correct": gr.get("correct", 0),
            "incorrect": gr.get("incorrect", 0),
            "recommended_weight": gr.get("recommended_weight", 1.0),
        })

    return jsonify(agents)


@app.route("/api/live")
def api_live():
    """Live bot state — balance, positions, current opportunities, markets scanned."""
    if _orchestrator:
        status = _orchestrator.get_status()
        return jsonify(status)

    # Fallback when orchestrator is not connected (dashboard-only mode)
    return jsonify({
        "mode": "dashboard_only",
        "balance": 0,
        "positions": [],
        "markets_scanned": 0,
        "active_opportunities": [],
        "agent_weights": {},
        "grade_summary": grader.get_grade_summary(),
        "agent_report": grader.get_agent_report(),
    })


@app.route("/api/grades")
def api_grades():
    """Trade grade summary and distribution."""
    summary = grader.get_grade_summary()
    return jsonify(summary)


@app.route("/api/research")
def api_research():
    """Current research: all signals grouped by market with edge analysis."""
    if _orchestrator and _orchestrator._last_signals:
        research = []
        for ticker, signals in _orchestrator._last_signals.items():
            market_data = None
            for m in _orchestrator._last_markets:
                if m.ticker == ticker:
                    market_data = m
                    break

            research.append({
                "ticker": ticker,
                "title": market_data.title if market_data else ticker,
                "market_price": market_data.yes_price if market_data else 0,
                "signals": [
                    {
                        "agent": s.agent_name,
                        "side": s.side.value,
                        "confidence": s.confidence,
                        "estimated_prob": s.estimated_probability,
                        "edge": s.edge,
                        "strength": s.strength.value,
                        "reasoning": s.reasoning,
                    }
                    for s in signals
                ],
                "consensus": _get_consensus(signals),
            })
        return jsonify(research)

    return jsonify([])


def _get_consensus(signals):
    """Compute consensus from a list of signals."""
    if not signals:
        return {"direction": "none", "avg_edge": 0, "agreement": 0}
    yes_count = sum(1 for s in signals if s.side.value == "yes")
    no_count = len(signals) - yes_count
    direction = "yes" if yes_count > no_count else "no" if no_count > yes_count else "split"
    avg_edge = sum(abs(s.edge) for s in signals) / len(signals)
    agreement = max(yes_count, no_count) / len(signals)
    return {"direction": direction, "avg_edge": round(avg_edge, 4), "agreement": round(agreement, 2)}


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
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
