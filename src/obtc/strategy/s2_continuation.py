"""S2 — Continuation.

HTF bias aligned with the chain direction, a BOS in that direction, a fresh
LTF order block in that direction, and a fair value gap in that direction
(ltf_fvg preferred, else the first aligned MTF FVG). Entry at the FVG
midpoint (or the OB 50% when the FVG is absent), stop beyond the OB far side.
"""
from __future__ import annotations

from typing import ClassVar

from obtc.chain.models import BlockState, Direction, FullChain, Signal
from obtc.strategy.base import BaseStrategy


class Continuation(BaseStrategy):
    """HTF-aligned BOS continuation into a fresh OB + FVG.

    confluence_score weights (0-100):
      bos_strength 30  - BOS displacement from shift meta
                        (min(1, displacement_atr/2) * 30)
      fvg_quality 25   - FVG size in ATR units (min(1, size/atr) * 25)
      ob_freshness 20   - 20 if the OB is fresh (ACTIVE), 12 if mitigated
      htf 15           - 15 if HTF bias aligned, 7.5 neutral, 0 opposed
      liquidity_target 10 - 10 when notes carry a target_liquidity magnet,
                        else 3
    """

    name: ClassVar[str] = "s2_continuation"
    label: ClassVar[str] = "Continuation"

    def _aligned_fvg(self, chain: FullChain):
        fvg = chain.ltf_fvg
        if (fvg is not None and fvg.direction == chain.direction
                and fvg.state == BlockState.ACTIVE):
            return fvg
        for c in chain.mtf_fvgs or []:
            if (c is not None and c.direction == chain.direction
                    and c.state == BlockState.ACTIVE):
                return c
        return None

    def check_setup(self, chain: FullChain) -> Signal | None:
        if self._bias_alignment(chain, chain.direction) is not True:
            return None
        shift = chain.ltf_shift
        if shift is None or shift.kind != "BOS" or shift.direction != chain.direction:
            return None
        ob = chain.ltf_ob
        if (ob is None or ob.direction != chain.direction
                or ob.state != BlockState.ACTIVE):
            return None
        fvg = self._aligned_fvg(chain)
        if fvg is None:
            return None
        entry = fvg.midpoint
        atr = self._atr(chain, ob.high, ob.low)
        stop = self._stop_beyond_far_side(ob.high, ob.low, chain.direction, atr)
        return self._build_signal(
            chain, chain.direction, entry, stop,
            chain.notes.get("target_liquidity"), chain.ltf_timeframe,
            {
                "ob_high": ob.high,
                "ob_low": ob.low,
                "ob_state": ob.state.value,
                "fvg_top": fvg.top,
                "fvg_bottom": fvg.bottom,
                "fvg_timeframe": fvg.timeframe,
                "bos_broken_level": shift.broken_level,
                "atr": atr,
            },
        )

    def confluence_score(self, signal: Signal, ctx: dict) -> float:
        chain = ctx.get("chain") if isinstance(ctx, dict) else None
        if chain is None:
            return 0.0
        shift = chain.ltf_shift
        ob = chain.ltf_ob
        fvg = self._aligned_fvg(chain)
        if shift is None or ob is None or fvg is None:
            return 0.0
        atr = self._atr(chain, ob.high, ob.low)

        disp = self._num((shift.meta or {}).get("displacement_atr"), 1.0)
        bos_strength = min(1.0, max(0.0, disp) / 2.0) * 30.0

        fvg_quality = min(1.0, max(0.0, self._num(fvg.size)) / atr) * 25.0

        ob_freshness = 20.0 if ob.state == BlockState.ACTIVE else 12.0

        htf = self._htf_points(chain, signal.direction, 15.0)

        liquidity_target = 10.0 if chain.notes.get("target_liquidity") else 3.0

        return min(100.0, bos_strength + fvg_quality + ob_freshness + htf + liquidity_target)
