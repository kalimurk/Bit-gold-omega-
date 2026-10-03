"""S3 — Breaker Reclaim.

An MTF breaker block in the chain's direction that has FLIPPED, with a
confirmed retest (notes["breaker_retest"] is True). Entry at the breaker
50%, stop beyond the breaker far side.
"""
from __future__ import annotations

from typing import ClassVar

from obtc.chain.models import BlockState, Direction, FullChain, Signal
from obtc.strategy.base import BaseStrategy


class BreakerReclaim(BaseStrategy):
    """Flipped MTF breaker + confirmed retest.

    confluence_score weights (0-100):
      flip_strength 35   - breaker zone range in ATR units
                          (min(1, zone_range/atr) * 35)
      retest_precision 25 - how tight the retest was: 25 * max(0, 1 -
                          retest_deviation_atr) from notes
      htf 20             - 20 if HTF bias aligned, 10 neutral, 0 opposed
      displacement 10    - flip displacement from notes
                          (min(1, breaker_displacement_atr/2) * 10)
      structure 10       - 10 if a LTF shift confirms the direction, else 4
    """

    name: ClassVar[str] = "s3_breaker_reclaim"
    label: ClassVar[str] = "Breaker Reclaim"

    def check_setup(self, chain: FullChain) -> Signal | None:
        br = chain.mtf_breaker
        if (br is None or br.state != BlockState.FLIPPED
                or br.direction != chain.direction):
            return None
        if chain.notes.get("breaker_retest") is not True:
            return None
        atr = self._atr(chain, br.high, br.low)
        entry = br.fifty_pct
        stop = self._stop_beyond_far_side(br.high, br.low, chain.direction, atr)
        return self._build_signal(
            chain, chain.direction, entry, stop,
            chain.notes.get("target_liquidity"), chain.mtf_timeframe,
            {
                "ob_high": br.high,
                "ob_low": br.low,
                "breaker_state": br.state.value,
                "flipped_from": br.flipped_from.value if br.flipped_from else None,
                "breaker_retest": True,
                "atr": atr,
            },
        )

    def confluence_score(self, signal: Signal, ctx: dict) -> float:
        chain = ctx.get("chain") if isinstance(ctx, dict) else None
        if chain is None:
            return 0.0
        br = chain.mtf_breaker
        if br is None:
            return 0.0
        atr = self._atr(chain, br.high, br.low)

        flip_strength = min(1.0, max(0.0, self._num(br.zone_range)) / atr) * 35.0

        dev = self._num(chain.notes.get("retest_deviation_atr"), 0.5)
        retest_precision = max(0.0, 1.0 - max(0.0, dev)) * 25.0

        htf = self._htf_points(chain, signal.direction, 20.0)

        disp = self._num(chain.notes.get("breaker_displacement_atr"), 1.0)
        displacement = min(1.0, max(0.0, disp) / 2.0) * 10.0

        shift = chain.ltf_shift
        structure = 10.0 if (shift is not None and shift.direction == chain.direction) else 4.0

        return min(100.0, flip_strength + retest_precision + htf + displacement + structure)
