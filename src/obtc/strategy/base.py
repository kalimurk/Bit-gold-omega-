"""OBTC strategy base class and shared signal-construction helpers.

Every concrete strategy subclasses :class:`BaseStrategy`, implements
``check_setup`` (chain -> Signal | None) and ``confluence_score``
(signal -> 0-100), and reuses the helpers below so entry/stop/target
math and rationale bookkeeping are identical across strategies.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import ClassVar

from obtc.chain.models import (
    Bias,
    Direction,
    FullChain,
    Signal,
    utcnow,
)


def _valid_price(p) -> bool:
    try:
        return math.isfinite(p) and p > 0
    except TypeError:
        return False


class BaseStrategy(ABC):
    """Abstract base for all OBTC entry strategies."""

    name: ClassVar[str] = "base"
    label: ClassVar[str] = "Base"
    timeframe_set: ClassVar[tuple] = ("4H", "1H", "5m")
    needs_timing: ClassVar[bool] = False

    # ------------------------------------------------------------------ API
    @abstractmethod
    def check_setup(self, chain: FullChain) -> Signal | None:
        """Inspect a FullChain; return a Signal when the setup is present."""

    @abstractmethod
    def confluence_score(self, signal: Signal, ctx: dict) -> float:
        """Score 0-100 for a signal. ``ctx`` carries at least ``"chain"``."""

    def timing_ok(self, now: datetime, chain: FullChain) -> bool:
        """Session/killzone gate. Strategies without a timing edge return True."""
        return True

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _num(value, default: float = 0.0) -> float:
        """Coerce to finite float, falling back to ``default``."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default

    @staticmethod
    def _as_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _atr(self, chain: FullChain, ref_high: float, ref_low: float) -> float:
        """ATR from chain notes, else half the reference zone range."""
        atr = chain.notes.get("atr_ltf")
        if atr is None or not _valid_price(atr):
            atr = (ref_high - ref_low) * 0.5
        return max(float(atr), 1e-9)

    @staticmethod
    def _stop_beyond_far_side(ob_high: float, ob_low: float,
                              direction: Direction, atr: float) -> float:
        """Stop just past the block's far side: low - 0.25*ATR for longs,
        high + 0.25*ATR for shorts."""
        buf = 0.25 * atr
        if direction == Direction.BULLISH:
            return ob_low - buf
        return ob_high + buf

    def _targets(self, entry: float, stop: float, direction: Direction,
                 opposing_liquidity) -> tuple:
        """Return (target_2r, target_3r, opposing_liquidity)."""
        risk = abs(entry - stop)
        if direction == Direction.BULLISH:
            return (entry + 2.0 * risk, entry + 3.0 * risk, opposing_liquidity)
        return (entry - 2.0 * risk, entry - 3.0 * risk, opposing_liquidity)

    def _bias_alignment(self, chain: FullChain, direction: Direction):
        """True = HTF bias aligned, None = neutral, False = opposed."""
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

    def _htf_points(self, chain: FullChain, direction: Direction,
                    aligned_pts: float) -> float:
        """HTF agreement points: full when aligned, half when neutral, 0 opposed."""
        a = self._bias_alignment(chain, direction)
        if a is True:
            return aligned_pts
        if a is None:
            return aligned_pts / 2.0
        return 0.0

    def _build_signal(self, chain: FullChain, direction: Direction,
                      entry: float, stop: float, opposing_liquidity,
                      timeframe: str, rationale_extra: dict | None) -> Signal | None:
        """Assemble a Signal, or None when prices are invalid.

        Guarantees: risk > 0, entry/stop/targets finite and positive, stop on
        the correct side of entry for the direction, rationale carrying
        ob_high, ob_low, entry, stop, target_2r, target_3r (required by the
        trade screenshotter) plus strategy-specific keys, and
        rationale["strategy_score"] set from confluence_score.
        """
        t2, t3, opp = self._targets(entry, stop, direction, opposing_liquidity)
        if not all(_valid_price(p) for p in (entry, stop, t2, t3)):
            return None
        if abs(entry - stop) <= 0:
            return None
        if direction == Direction.BULLISH and not entry > stop:
            return None
        if direction == Direction.BEARISH and not entry < stop:
            return None
        if opp is not None and not _valid_price(opp):
            opp = None
        rationale = dict(rationale_extra or {})
        if "ob_high" not in rationale or "ob_low" not in rationale:
            return None
        if not _valid_price(rationale["ob_high"]) or not _valid_price(rationale["ob_low"]):
            return None
        rationale.update({
            "entry": entry,
            "stop": stop,
            "target_2r": t2,
            "target_3r": t3,
        })
        signal = Signal(
            strategy_name=self.name,
            symbol=chain.symbol,
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            target_2r=t2,
            target_3r=t3,
            opposing_liquidity=opp,
            chain_id=chain.id,
            timeframe=timeframe,
            created_at=utcnow(),
            rationale=rationale,
        )
        score = self._num(self.confluence_score(signal, {"chain": chain}), 0.0)
        score = max(0.0, min(100.0, score))
        signal.score = score
        signal.rationale["strategy_score"] = score
        return signal
