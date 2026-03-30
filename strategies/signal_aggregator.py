"""Signal aggregator - combines signals from multiple agents into trade decisions.

Uses weighted averaging with confidence-based scaling to produce
a single probability estimate and trade recommendation per market.
"""

from datetime import datetime

import structlog

from core.models import Market, Signal, Side, TradeOpportunity, SignalStrength
from config.settings import settings

logger = structlog.get_logger(__name__)


class SignalAggregator:
    """Aggregates signals from multiple agents into actionable trade opportunities."""

    def __init__(self, min_agents: int = 2, min_edge: float | None = None):
        """
        Args:
            min_agents: Minimum number of agents that must agree for a trade
            min_edge: Minimum edge required (defaults to settings)
        """
        self.min_agents = min_agents
        self.min_edge = min_edge or settings.min_edge_threshold

    def aggregate(
        self, market: Market, signals: list[Signal], agent_weights: dict[str, float] | None = None
    ) -> TradeOpportunity | None:
        """Combine signals into a single trade opportunity.

        Uses weighted probability averaging where weights come from:
        1. Agent weight (how reliable the agent's edge detection is)
        2. Signal confidence (how confident the agent is in this specific signal)

        Returns None if no sufficient edge is found.
        """
        if not signals:
            return None

        agent_weights = agent_weights or {}

        # Separate yes and no signals
        yes_signals = [s for s in signals if s.side == Side.YES]
        no_signals = [s for s in signals if s.side == Side.NO]

        # Check if enough agents agree on a direction
        if len(yes_signals) >= self.min_agents:
            consensus_signals = yes_signals
            opposing_signals = no_signals
            consensus_side = Side.YES
        elif len(no_signals) >= self.min_agents:
            consensus_signals = no_signals
            opposing_signals = yes_signals
            consensus_side = Side.NO
        else:
            logger.debug(
                "no_consensus",
                market=market.ticker,
                yes=len(yes_signals),
                no=len(no_signals),
            )
            return None

        # Weighted average probability estimate
        total_weight = 0.0
        weighted_prob = 0.0

        for signal in consensus_signals:
            w = agent_weights.get(signal.agent_name, 1.0) * signal.confidence
            weighted_prob += signal.estimated_probability * w
            total_weight += w

        # Include opposing signals with reduced weight (they temper the estimate)
        for signal in opposing_signals:
            w = agent_weights.get(signal.agent_name, 1.0) * signal.confidence * 0.3
            weighted_prob += signal.estimated_probability * w
            total_weight += w

        if total_weight == 0:
            return None

        estimated_prob = weighted_prob / total_weight

        # Calculate edge
        market_prob = market.implied_probability
        if consensus_side == Side.YES:
            edge = estimated_prob - market_prob
            entry_price = market.yes_price
        else:
            edge = market_prob - estimated_prob  # for NO, edge is inverted
            entry_price = market.no_price

        if edge < self.min_edge:
            logger.debug(
                "edge_too_small",
                market=market.ticker,
                edge=f"{edge:.4f}",
                threshold=f"{self.min_edge:.4f}",
            )
            return None

        # Aggregate confidence
        avg_confidence = sum(s.confidence for s in consensus_signals) / len(consensus_signals)

        # Agreement bonus: more agents agreeing = higher confidence
        agreement_bonus = min((len(consensus_signals) - self.min_agents) * 0.1, 0.2)
        adjusted_confidence = min(avg_confidence + agreement_bonus, 1.0)

        # Disagreement penalty
        if opposing_signals:
            penalty = len(opposing_signals) / len(signals) * 0.15
            adjusted_confidence = max(adjusted_confidence - penalty, 0.1)

        # Expected value calculation
        if consensus_side == Side.YES:
            win_amount = 1.0 - entry_price  # profit per contract if YES resolves
            lose_amount = entry_price  # loss per contract if NO resolves
            ev = (estimated_prob * win_amount) - ((1 - estimated_prob) * lose_amount)
        else:
            win_amount = 1.0 - entry_price
            lose_amount = entry_price
            ev = ((1 - estimated_prob) * win_amount) - (estimated_prob * lose_amount)

        if ev <= 0:
            return None

        return TradeOpportunity(
            market=market,
            side=consensus_side,
            entry_price=entry_price,
            estimated_probability=estimated_prob,
            edge=edge,
            confidence=adjusted_confidence,
            signals=consensus_signals + opposing_signals,
            expected_value=ev,
        )

    def rank_opportunities(
        self, opportunities: list[TradeOpportunity]
    ) -> list[TradeOpportunity]:
        """Rank trade opportunities by expected value and confidence."""
        # Score = EV * confidence * edge
        def score(opp: TradeOpportunity) -> float:
            return opp.expected_value * opp.confidence * opp.edge

        return sorted(opportunities, key=score, reverse=True)
