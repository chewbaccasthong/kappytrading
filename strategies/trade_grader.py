"""Trade grading system — scores every trade entry and exit to improve over time.

After each trade resolves, the grader:
1. Scores the entry (was the edge real? was timing good?)
2. Scores each agent's signal accuracy
3. Tracks which agents are reliable vs which produce noise
4. Adjusts agent weights based on historical accuracy
5. Identifies patterns in winning vs losing trades

This creates a feedback loop: better grades → better weights → better trades.
"""

from datetime import datetime
from dataclasses import dataclass, field

import structlog

from core.models import Signal, TradeOpportunity, Side
from utils.database import Database, TradeRecord, SignalRecord

logger = structlog.get_logger(__name__)


@dataclass
class TradeGrade:
    """Grade assigned to a completed trade."""
    trade_id: int
    market_ticker: str
    # Entry grading
    entry_grade: str  # A, B, C, D, F
    entry_score: float  # 0-100
    edge_accuracy: float  # how close was estimated edge to actual outcome
    timing_score: float  # did price move favorably after entry?
    # Agent grading
    agent_grades: dict[str, float] = field(default_factory=dict)  # agent -> score 0-100
    correct_agents: list[str] = field(default_factory=list)
    wrong_agents: list[str] = field(default_factory=list)
    # Outcome
    pnl: float = 0.0
    was_profitable: bool = False
    # Feedback
    lessons: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TradeGrader:
    """Grades trades and adjusts agent weights based on outcomes."""

    def __init__(self, db: Database):
        self.db = db
        # Track agent accuracy over time
        self._agent_stats: dict[str, dict] = {}
        self._load_historical_stats()

    def _load_historical_stats(self):
        """Load agent performance stats from DB."""
        with self.db.Session() as session:
            agents = session.query(SignalRecord.agent_name).distinct().all()
            for (name,) in agents:
                signals = (
                    session.query(SignalRecord)
                    .filter_by(agent_name=name)
                    .order_by(SignalRecord.created_at.desc())
                    .limit(200)
                    .all()
                )
                if signals:
                    self._agent_stats[name] = {
                        "total_signals": len(signals),
                        "avg_confidence": sum(s.confidence for s in signals) / len(signals),
                        "avg_edge": sum(s.edge for s in signals) / len(signals),
                        "correct": 0,
                        "incorrect": 0,
                        "accuracy": 0.5,
                    }

    def grade_trade(
        self,
        trade_id: int,
        entry_side: str,
        entry_price: float,
        exit_price: float | None,
        pnl: float,
        estimated_edge: float,
        confidence: float,
        agent_signals: list[dict],
        market_resolved_yes: bool | None = None,
    ) -> TradeGrade:
        """Grade a completed trade on multiple dimensions.

        Args:
            trade_id: Database trade ID
            entry_side: 'yes' or 'no'
            entry_price: Price paid per contract
            exit_price: Price received (or resolution price)
            pnl: Actual profit/loss
            estimated_edge: Edge estimated at entry time
            confidence: Confidence at entry time
            agent_signals: List of signal dicts from the trade
            market_resolved_yes: True if market resolved YES, False for NO, None if still open
        """
        was_profitable = pnl > 0

        # 1. Edge accuracy: how close was our edge estimate to reality?
        if exit_price is not None:
            if entry_side == "yes":
                actual_edge = (exit_price - entry_price)  # simplified
            else:
                actual_edge = (entry_price - exit_price)
            edge_accuracy = max(0, 100 - abs(estimated_edge - actual_edge) * 500)
        else:
            edge_accuracy = 50.0  # unknown

        # 2. Entry timing score
        timing_score = self._score_timing(was_profitable, pnl, entry_price, confidence)

        # 3. Agent scoring
        agent_grades = {}
        correct_agents = []
        wrong_agents = []

        for sig in agent_signals:
            agent_name = sig.get("agent", sig.get("agent_name", "unknown"))
            sig_side = sig.get("side", "")
            sig_confidence = sig.get("confidence", 0.5)

            if market_resolved_yes is not None:
                # We know the outcome — grade definitively
                agent_was_right = (
                    (sig_side == "yes" and market_resolved_yes)
                    or (sig_side == "no" and not market_resolved_yes)
                )
            else:
                # Use PnL as proxy
                agent_agreed_with_trade = sig_side == entry_side
                agent_was_right = (agent_agreed_with_trade and was_profitable) or (
                    not agent_agreed_with_trade and not was_profitable
                )

            if agent_was_right:
                agent_grades[agent_name] = min(100, 60 + sig_confidence * 40)
                correct_agents.append(agent_name)
            else:
                agent_grades[agent_name] = max(0, 40 - sig_confidence * 30)
                wrong_agents.append(agent_name)

            # Update running stats
            self._update_agent_stats(agent_name, agent_was_right, sig_confidence)

        # 4. Overall entry grade
        entry_score = self._compute_entry_score(
            edge_accuracy, timing_score, was_profitable, confidence, estimated_edge
        )
        entry_grade = self._score_to_letter(entry_score)

        # 5. Generate lessons
        lessons = self._generate_lessons(
            entry_grade, was_profitable, estimated_edge, pnl,
            correct_agents, wrong_agents, confidence
        )

        grade = TradeGrade(
            trade_id=trade_id,
            market_ticker="",
            entry_grade=entry_grade,
            entry_score=entry_score,
            edge_accuracy=edge_accuracy,
            timing_score=timing_score,
            agent_grades=agent_grades,
            correct_agents=correct_agents,
            wrong_agents=wrong_agents,
            pnl=pnl,
            was_profitable=was_profitable,
            lessons=lessons,
        )

        logger.info(
            "trade_graded",
            trade_id=trade_id,
            grade=entry_grade,
            score=f"{entry_score:.0f}",
            pnl=f"${pnl:.2f}",
            correct_agents=correct_agents,
            wrong_agents=wrong_agents,
        )

        return grade

    def _score_timing(
        self, profitable: bool, pnl: float, entry_price: float, confidence: float
    ) -> float:
        """Score how good the entry timing was."""
        if profitable:
            # Good trade — score based on how much we made relative to risk
            return_pct = pnl / max(entry_price, 0.01)
            return min(100, 50 + return_pct * 200)
        else:
            # Bad trade — score based on how much we lost
            loss_pct = abs(pnl) / max(entry_price, 0.01)
            return max(0, 50 - loss_pct * 200)

    def _compute_entry_score(
        self,
        edge_accuracy: float,
        timing_score: float,
        profitable: bool,
        confidence: float,
        estimated_edge: float,
    ) -> float:
        """Compute overall entry score (0-100)."""
        score = 0.0

        # Outcome (40% weight)
        if profitable:
            score += 40
        else:
            score += 10  # some credit for trying

        # Edge accuracy (25% weight)
        score += edge_accuracy * 0.25

        # Timing (20% weight)
        score += timing_score * 0.20

        # Confidence calibration (15% weight)
        # High confidence + win = good, high confidence + loss = bad
        if profitable:
            score += confidence * 15
        else:
            score += (1 - confidence) * 15  # low confidence loss is less bad

        return max(0, min(100, score))

    def _score_to_letter(self, score: float) -> str:
        if score >= 85:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 55:
            return "C"
        elif score >= 40:
            return "D"
        return "F"

    def _update_agent_stats(self, agent_name: str, was_correct: bool, confidence: float):
        """Update running agent accuracy stats."""
        if agent_name not in self._agent_stats:
            self._agent_stats[agent_name] = {
                "total_signals": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.5,
                "avg_confidence": 0.5,
                "avg_edge": 0.0,
            }

        stats = self._agent_stats[agent_name]
        stats["total_signals"] += 1
        if was_correct:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1

        total = stats["correct"] + stats["incorrect"]
        stats["accuracy"] = stats["correct"] / total if total > 0 else 0.5

    def _generate_lessons(
        self,
        grade: str,
        profitable: bool,
        edge: float,
        pnl: float,
        correct_agents: list[str],
        wrong_agents: list[str],
        confidence: float,
    ) -> list[str]:
        """Generate actionable lessons from a trade."""
        lessons = []

        if not profitable and confidence > 0.7:
            lessons.append("High confidence loss — agents may be overconfident. Consider reducing position size on high-confidence signals.")

        if not profitable and edge < 0.06:
            lessons.append(f"Loss on thin edge ({edge:.1%}). Consider raising MIN_EDGE_THRESHOLD.")

        if profitable and len(correct_agents) >= 3:
            lessons.append(f"Strong multi-agent consensus paid off ({len(correct_agents)} agents correct). These setups are high quality.")

        if len(wrong_agents) > len(correct_agents) and profitable:
            lessons.append("Won despite majority of agents being wrong — could be luck. Watch for similar setups.")

        if not profitable and len(wrong_agents) >= 3:
            lessons.append(f"Multiple agents wrong ({', '.join(wrong_agents)}). Check if a common data source failed.")

        for agent in wrong_agents:
            stats = self._agent_stats.get(agent, {})
            if stats.get("accuracy", 1) < 0.4:
                lessons.append(f"Agent '{agent}' has low accuracy ({stats['accuracy']:.0%}). Consider reducing its weight.")

        return lessons

    def get_recommended_weights(self) -> dict[str, float]:
        """Calculate recommended agent weights based on historical accuracy.

        Agents that are more accurate get higher weights.
        Uses a softmax-like scaling so no agent gets zero weight.
        """
        if not self._agent_stats:
            return {}

        weights = {}
        for name, stats in self._agent_stats.items():
            accuracy = stats.get("accuracy", 0.5)
            total = stats.get("total_signals", 0)

            if total < 5:
                # Not enough data — use default weight
                weights[name] = 1.0
            else:
                # Scale weight: accuracy 0.3 → 0.5, accuracy 0.7 → 1.5
                weights[name] = max(0.3, accuracy * 2.0)

        return weights

    def get_agent_report(self) -> list[dict]:
        """Get a detailed report on each agent's performance."""
        report = []
        for name, stats in self._agent_stats.items():
            total = stats.get("correct", 0) + stats.get("incorrect", 0)
            report.append({
                "agent": name,
                "total_signals": stats.get("total_signals", 0),
                "graded_trades": total,
                "correct": stats.get("correct", 0),
                "incorrect": stats.get("incorrect", 0),
                "accuracy": stats.get("accuracy", 0),
                "recommended_weight": self.get_recommended_weights().get(name, 1.0),
            })
        return sorted(report, key=lambda x: x["accuracy"], reverse=True)

    def get_grade_summary(self) -> dict:
        """Get summary of all trade grades."""
        with self.db.Session() as session:
            trades = session.query(TradeRecord).filter_by(status="closed").all()

        if not trades:
            return {"total_graded": 0, "avg_score": 0, "grade_distribution": {}}

        grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        total_score = 0
        count = 0

        for t in trades:
            pnl = t.pnl or 0
            score = self._compute_entry_score(
                edge_accuracy=50,  # approximate
                timing_score=70 if pnl > 0 else 30,
                profitable=pnl > 0,
                confidence=t.confidence,
                estimated_edge=t.edge,
            )
            letter = self._score_to_letter(score)
            grades[letter] += 1
            total_score += score
            count += 1

        return {
            "total_graded": count,
            "avg_score": total_score / count if count else 0,
            "avg_grade": self._score_to_letter(total_score / count if count else 0),
            "grade_distribution": grades,
        }
