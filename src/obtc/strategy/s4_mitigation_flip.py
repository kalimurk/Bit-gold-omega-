"""S4 — Mitigation Flip.

The LTF order block is MITIGATED (direction == chain direction) and an
inversion FVG prints after mitigation in the opposite direction. The signal
flips: direction = the inversion FVG's direction. Entry at the FVG midpoint,
stop beyond the FVG far side.
"""
from __future__ import annotations

from typing import ClassVar

from obtc.chain.models import BlockState, Direction, FullChain, Signal, utcnow
from obtc.strategy.base import BaseStrategy


class MitigationFlip(BaseStrategy):
    """Post-mitigation inversion: trade the flip, not the chain direction.

    confluence_score weights (0-100):
      mitigation_depth 30 - mitigated OB zone range in ATR units
                          (min(1, zone_range/atr) * 30)
      inversion_fvg_size 25 - inversion FVG size in ATR units
                          (min(1, size/atr) * 25)
      displacement 20     - flip displacement from notes
                          (min(1, flip_displacement_atr/2) * 20)
      htf_agreement 15    - 15 if HTF bias agrees with the FLIP direction,
                          7.5 neutral, 0 opposed
      timing 10           - 10 when timing_ok (no session gate for S4)
    """

    name: ClassVar[str] = "s4_mitigation_flip"
    label: ClassVar[str] = "Mitigation Flip"

    def _inversion_fvg(self, chain: FullChain):
        ob = chain.ltf_ob
        if ob is None or ob.mitigated_at is None:
            return None
        for fvg in list(chain.mtf_fvgs or []) + [chain.ltf_fvg]:
            if (fvg is not None and fvg.direction != chain.direction
                    and fvg.created_at is not None
                    and fvg.created_at > ob.mitigated_at):
                return fvg
        return None

    def check_setup(self, chain: FullChain) -> Signal | None:
        ob = chain.ltf_ob
        if (ob is None or ob.state != BlockState.MITIGATED
                or ob.direction != chain.direction or ob.mitigated_at is None):
            return None
        inv = self._inversion_fvg(chain)
        if inv is None:
            return None
        direction = inv.direction
        atr = self._atr(chain, inv.top, inv.bottom)
        entry = inv.midpoint
        stop = self._stop_beyond_far_side(inv.top, inv.bottom, direction, atr)
        return self._build_signal(
            chain, direction, entry, stop,
            chain.notes.get("target_liquidity"), chain.ltf_timeframe,
            {
                "ob_high": ob.high,
                "ob_low": ob.low,
                "ob_state": ob.state.value,
                "flip": True,
                "chain_direction": chain.direction.value,
                "inversion_fvg_id": inv.id,
                "inversion_fvg_top": inv.top,
                "inversion_fvg_bottom": inv.bottom,
                "inversion_fvg_timeframe": inv.timeframe,
                "mitigated_at": ob.mitigated_at.isoformat(),
                "atr": atr,
            },
        )

    def confluence_score(self, signal: Signal, ctx: dict) -> float:
        chain = ctx.get("chain") if isinstance(ctx, dict) else None
        if chain is None:
            return 0.0
        ob = chain.ltf_ob
        inv = self._inversion_fvg(chain)
        if ob is None or inv is None:
            return 0.0
        atr = self._atr(chain, inv.top, inv.bottom)

        mitigation_depth = min(1.0, max(0.0, self._num(ob.zone_range)) / atr) * 30.0

        inversion_fvg_size = min(1.0, max(0.0, self._num(inv.size)) / atr) * 25.0

        disp = self._num(chain.notes.get("flip_displacement_atr"), 1.0)
        displacement = min(1.0, max(0.0, disp) / 2.0) * 20.0

        htf_agreement = self._htf_points(chain, signal.direction, 15.0)

        now = ctx.get("now") if isinstance(ctx, dict) else None
        timing = 10.0 if self.timing_ok(now if now is not None else utcnow(), chain) else 0.0

        return min(100.0, mitigation_depth + inversion_fvg_size + displacement + htf_agreement + timing)
