"""S5 — Turtle Soup.

An Asia-session liquidity sweep in the chain's direction, followed by an MSS
and a LTF OB tap in that direction. Entry at the OB 50%, stop beyond the OB
far side. The opposing liquidity target is the Asia range extreme
(notes["asia_high"] for longs, notes["asia_low"] for shorts).

Timing gate: sweep.swept_at in UTC must fall between 00:00 and 09:00.
"""
from __future__ import annotations

from datetime import time
from typing import ClassVar

from obtc.chain.models import BlockState, Direction, FullChain, Signal
from obtc.strategy.base import BaseStrategy


class TurtleSoup(BaseStrategy):
    """Asia sweep -> MSS -> OB tap (turtle soup).

    confluence_score weights (0-100):
      sweep_quality 30 - sweep wick extension in ATR units
                        (min(1, ext/atr) * 30)
      session_purity 20 - 20 when the sweep session is asia, else 5
      mss 20           - MSS displacement from shift meta
                        (min(1, displacement_atr/2) * 20)
      ob 15            - 15 if the OB is fresh (ACTIVE), 9 if mitigated
      htf 15           - 15 if HTF bias aligned, 7.5 neutral, 0 opposed
    """

    name: ClassVar[str] = "s5_turtle_soup"
    label: ClassVar[str] = "Turtle Soup"
    needs_timing: ClassVar[bool] = True

    def timing_ok(self, now, chain: FullChain) -> bool:
        sweep = chain.liquidity_sweep
        if sweep is None or sweep.swept_at is None:
            return False
        t = self._as_utc(sweep.swept_at).time()
        return time(0, 0) <= t <= time(9, 0)

    def check_setup(self, chain: FullChain) -> Signal | None:
        sweep = chain.liquidity_sweep
        if (sweep is None or (sweep.meta or {}).get("session") != "asia"
                or sweep.implied_direction != chain.direction):
            return None
        shift = chain.ltf_shift
        if shift is None or shift.kind != "MSS" or shift.direction != chain.direction:
            return None
        ob = chain.ltf_ob
        if (ob is None or ob.direction != chain.direction
                or ob.state not in (BlockState.ACTIVE, BlockState.MITIGATED)):
            return None
        atr = self._atr(chain, ob.high, ob.low)
        entry = ob.fifty_pct
        stop = self._stop_beyond_far_side(ob.high, ob.low, chain.direction, atr)
        if chain.direction == Direction.BULLISH:
            opposing = chain.notes.get("asia_high")
        else:
            opposing = chain.notes.get("asia_low")
        return self._build_signal(
            chain, chain.direction, entry, stop, opposing, chain.ltf_timeframe,
            {
                "ob_high": ob.high,
                "ob_low": ob.low,
                "ob_state": ob.state.value,
                "sweep_level": sweep.level,
                "sweep_side": sweep.side,
                "sweep_session": "asia",
                "mss_broken_level": shift.broken_level,
                "atr": atr,
            },
        )

    def confluence_score(self, signal: Signal, ctx: dict) -> float:
        chain = ctx.get("chain") if isinstance(ctx, dict) else None
        if chain is None:
            return 0.0
        sweep = chain.liquidity_sweep
        shift = chain.ltf_shift
        ob = chain.ltf_ob
        if sweep is None or shift is None or ob is None:
            return 0.0
        atr = self._atr(chain, ob.high, ob.low)

        if sweep.side == "BUY_SIDE":
            ext = max(0.0, self._num(sweep.sweep_high) - self._num(sweep.level))
        else:
            ext = max(0.0, self._num(sweep.level) - self._num(sweep.sweep_low))
        sweep_quality = min(1.0, ext / atr) * 30.0

        session_purity = 20.0 if (sweep.meta or {}).get("session") == "asia" else 5.0

        disp = self._num((shift.meta or {}).get("displacement_atr"), 1.0)
        mss = min(1.0, max(0.0, disp) / 2.0) * 20.0

        ob_score = 15.0 if ob.state == BlockState.ACTIVE else 9.0

        htf = self._htf_points(chain, signal.direction, 15.0)

        return min(100.0, sweep_quality + session_purity + mss + ob_score + htf)
