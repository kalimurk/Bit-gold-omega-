"""S1 — Liquidity Sweep Reversal.

A liquidity sweep in the chain's direction, followed by a market-structure
shift (MSS) in the same direction, with price tapping a LTF order block in
that direction. Entry at the block's 50%, stop beyond the block's far side.
"""
from __future__ import annotations

from typing import ClassVar

from obtc.chain.models import BlockState, Direction, FullChain, Signal
from obtc.strategy.base import BaseStrategy


class LiquiditySweepReversal(BaseStrategy):
    """Sweep -> MSS -> OB tap reversal.

    confluence_score weights (0-100):
      sweep_wick 30    - sweep wick extension beyond the level, in ATR units
                        (min(1, ext/atr) * 30)
      mss_displacement 25 - MSS displacement from shift meta
                        (min(1, displacement_atr/2) * 25)
      ob_tap 20        - 20 if the OB is fresh (ACTIVE), 12 if already
                        mitigated (MITIGATED)
      htf 15           - 15 if HTF bias aligned, 7.5 neutral, 0 opposed
      volume/session 10 - 5 * volume participation (ob.volume / 500 capped at
                        1) + 5 for a london/new_york sweep session, else 2
    """

    name: ClassVar[str] = "s1_liquidity_sweep_reversal"
    label: ClassVar[str] = "Liquidity Sweep Reversal"

    def check_setup(self, chain: FullChain) -> Signal | None:
        sweep = chain.liquidity_sweep
        if sweep is None or sweep.implied_direction != chain.direction:
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
        return self._build_signal(
            chain, chain.direction, entry, stop,
            chain.notes.get("target_liquidity"), chain.ltf_timeframe,
            {
                "ob_high": ob.high,
                "ob_low": ob.low,
                "ob_state": ob.state.value,
                "sweep_level": sweep.level,
                "sweep_side": sweep.side,
                "sweep_session": (sweep.meta or {}).get("session"),
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
        sweep_wick = min(1.0, ext / atr) * 30.0

        disp = self._num((shift.meta or {}).get("displacement_atr"), 1.0)
        mss_displacement = min(1.0, max(0.0, disp) / 2.0) * 25.0

        ob_tap = 20.0 if ob.state == BlockState.ACTIVE else 12.0

        htf = self._htf_points(chain, signal.direction, 15.0)

        vol = 5.0 * min(1.0, max(0.0, self._num(ob.volume)) / 500.0)
        sess = 5.0 if (sweep.meta or {}).get("session") in ("london", "new_york") else 2.0

        return min(100.0, sweep_wick + mss_displacement + ob_tap + htf + vol + sess)
