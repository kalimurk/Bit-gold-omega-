"""Confluence engine: combine strategy score, chain state, HTF bias,
liquidity, FVG alignment and timing into a single 0-100 A+ verdict.
"""
from __future__ import annotations

from obtc.chain.models import Bias, ChainState, Direction, FullChain, ScoredSetup, Signal


class ConfluenceEngine:
    """Score a Signal against its FullChain.

    Component weights:
      htf_bias        25 - HTF bias vs signal direction
      chain_alignment 25 - chain state maturity + strategy score
      liquidity       20 - aligned sweep, else opposing-liquidity magnet
      fvg             15 - aligned FVG presence
      time            15 - session/killzone gate (always full when the
                          strategy has no timing requirement)
    """

    WEIGHTS = {
        "htf_bias": 25.0,
        "chain_alignment": 25.0,
        "liquidity": 20.0,
        "fvg": 15.0,
        "time": 15.0,
    }

    def __init__(self, threshold: float = 80.0):
        self.threshold = float(threshold)

    # ------------------------------------------------------------- scoring
    def score(self, signal: Signal, chain: FullChain, ctx: dict) -> ScoredSetup:
        """ctx = {"now": datetime, "strategy": BaseStrategy}."""
        strategy = (ctx or {}).get("strategy")
        now = (ctx or {}).get("now")

        components = {
            "htf_bias": self._htf_bias(signal, chain),
            "chain_alignment": self._chain_alignment(signal, chain),
            "liquidity": self._liquidity(signal, chain),
            "fvg": self._fvg(signal, chain),
            "time": self._time(strategy, now, chain),
        }
        total = min(100.0, sum(components.values()))
        return ScoredSetup(
            signal=signal,
            score=total,
            components=components,
            is_a_plus=total >= self.threshold,
        )

    # ------------------------------------------------------------ components
    @staticmethod
    def _bias_aligned(chain: FullChain, direction: Direction):
        bias = chain.htf_bias
        if direction == Direction.BULLISH:
            if bias == Bias.LONG:
                return True
            if bias == Bias.NEUTRAL:
                return None
            return False
        if bias == Bias.SHORT:
            return True
        if bias == Bias.NEUTRAL:
            return None
        return False

    def _htf_bias(self, signal: Signal, chain: FullChain) -> float:
        a = self._bias_aligned(chain, signal.direction)
        if a is True:
            return 25.0
        if a is None:
            return 10.0
        return 0.0

    @staticmethod
    def _chain_alignment(signal: Signal, chain: FullChain) -> float:
        if chain.state == ChainState.TRIGGERED:
            state_pts = 15.0
        elif chain.state == ChainState.VALID:
            state_pts = 10.0
        else:
            state_pts = 4.0
        try:
            strategy_score = float(signal.rationale.get("strategy_score", 50.0))
        except (TypeError, ValueError):
            strategy_score = 50.0
        strategy_score = max(0.0, min(100.0, strategy_score))
        return state_pts + 10.0 * (strategy_score / 100.0)

    @staticmethod
    def _liquidity(signal: Signal, chain: FullChain) -> float:
        sweep = chain.liquidity_sweep
        if sweep is not None and sweep.implied_direction == signal.direction:
            return 20.0
        if signal.opposing_liquidity:
            return 12.0
        return 5.0

    @staticmethod
    def _fvg(signal: Signal, chain: FullChain) -> float:
        fvg = chain.ltf_fvg
        if fvg is not None and fvg.direction == signal.direction:
            return 15.0
        for c in chain.mtf_fvgs or []:
            if c is not None and c.direction == signal.direction:
                return 15.0
        return 0.0

    @staticmethod
    def _time(strategy, now, chain: FullChain) -> float:
        if strategy is not None and getattr(strategy, "needs_timing", False):
            if now is None:
                return 0.0
            return 15.0 if strategy.timing_ok(now, chain) else 0.0
        return 15.0
