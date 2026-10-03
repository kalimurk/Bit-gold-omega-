"""S6 — Silver Bullet.

The 10:00-11:00 America/New_York killzone: chain TRIGGERED, LTF FVG in the
chain direction, LTF OB tap in that direction. Entry at the FVG midpoint,
stop beyond the OB far side.

Timing gate: ``now`` in America/New_York between 10:00 (inclusive) and
11:00 (exclusive).
"""
from __future__ import annotations

from datetime import time
from typing import ClassVar
from zoneinfo import ZoneInfo

from obtc.chain.models import BlockState, ChainState, Direction, FullChain, Signal
from obtc.strategy.base import BaseStrategy

NEW_YORK = ZoneInfo("America/New_York")


class SilverBullet(BaseStrategy):
    """10:00-11:00 NY killzone FVG entry on a triggered chain.

    confluence_score weights (0-100):
      killzone_timing 25 - 25 when timing_ok (inside the killzone), else 0
      fvg_alignment 25   - 25 when the LTF FVG aligns with the signal
                          direction, else 0
      ob_confluence 20   - 20 if the OB is fresh (ACTIVE), 12 if mitigated
      htf 15             - 15 if HTF bias aligned, 7.5 neutral, 0 opposed
      liquidity 15       - 15 when notes carry a target_liquidity magnet,
                          else 5
    """

    name: ClassVar[str] = "s6_silver_bullet"
    label: ClassVar[str] = "Silver Bullet"
    needs_timing: ClassVar[bool] = True

    def timing_ok(self, now, chain: FullChain) -> bool:
        if now is None:
            return False
        ny_time = self._as_utc(now).astimezone(NEW_YORK).time()
        return time(10, 0) <= ny_time < time(11, 0)

    def check_setup(self, chain: FullChain) -> Signal | None:
        if chain.state != ChainState.TRIGGERED:
            return None
        fvg = chain.ltf_fvg
        if fvg is None or fvg.direction != chain.direction:
            return None
        ob = chain.ltf_ob
        if (ob is None or ob.direction != chain.direction
                or ob.state not in (BlockState.ACTIVE, BlockState.MITIGATED)):
            return None
        atr = self._atr(chain, ob.high, ob.low)
        entry = fvg.midpoint
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
                "chain_state": chain.state.value,
                "atr": atr,
            },
        )

    def confluence_score(self, signal: Signal, ctx: dict) -> float:
        chain = ctx.get("chain") if isinstance(ctx, dict) else None
        if chain is None:
            return 0.0
        fvg = chain.ltf_fvg
        ob = chain.ltf_ob
        if fvg is None or ob is None:
            return 0.0

        now = ctx.get("now") if isinstance(ctx, dict) else None
        killzone_timing = 25.0 if (now is not None and self.timing_ok(now, chain)) else 0.0

        fvg_alignment = 25.0 if fvg.direction == signal.direction else 0.0

        ob_confluence = 20.0 if ob.state == BlockState.ACTIVE else 12.0

        htf = self._htf_points(chain, signal.direction, 15.0)

        liquidity = 15.0 if chain.notes.get("target_liquidity") else 5.0

        return min(100.0, killzone_timing + fvg_alignment + ob_confluence + htf + liquidity)
