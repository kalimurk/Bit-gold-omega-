"""S7 — Orderflow Stacking.

Two or more stacked order blocks (notes["stacked_obs"]) in the chain's
direction, with price in the discount (long) / premium (short) side of the
dealing range (notes["dealing_range"] = (range_low, range_high),
notes["last_price"]). Entry at the deepest (last) stacked OB's 50%, stop
beyond its far side.

stacked_obs entries may be OrderBlock objects or plain dicts with
high/low/fifty_pct/direction keys.
"""
from __future__ import annotations

from typing import ClassVar

from obtc.chain.models import Direction, FullChain, Signal
from obtc.strategy.base import BaseStrategy


class OrderflowStacking(BaseStrategy):
    """Stacked OBs + dealing-range discount/premium entry.

    confluence_score weights (0-100):
      stack_count 30   - 30 * min(1, (n-1)/3): 2 blocks = 10, 3 = 20, 4+ = 30
      discount_depth 25 - how deep price sits in discount (long) / premium
                         (short), as a fraction of the range half, * 25
      ob_quality 20    - mean stacked zone range in ATR units
                         (min(1, mean_range/atr) * 20)
      htf 15           - 15 if HTF bias aligned, 7.5 neutral, 0 opposed
      liquidity 10     - 10 when notes carry a target_liquidity magnet,
                         else 3
    """

    name: ClassVar[str] = "s7_orderflow_stacking"
    label: ClassVar[str] = "Orderflow Stacking"

    # ------------------------------------------------------------ internals
    @staticmethod
    def _field(ob, name):
        if isinstance(ob, dict):
            return ob.get(name)
        return getattr(ob, name, None)

    def _ob_direction(self, ob):
        d = self._field(ob, "direction")
        if isinstance(d, Direction):
            return d
        try:
            return Direction(str(d))
        except (ValueError, TypeError):
            return None

    def _stacked(self, chain: FullChain) -> list:
        raw = chain.notes.get("stacked_obs") or []
        return [ob for ob in raw if self._ob_direction(ob) == chain.direction]

    def _dealing_range(self, chain: FullChain):
        dr = chain.notes.get("dealing_range")
        try:
            low, high = dr
            low, high = float(low), float(high)
        except (TypeError, ValueError):
            return None
        if not (low < high):
            return None
        return (low, high)

    # ------------------------------------------------------------------ API
    def check_setup(self, chain: FullChain) -> Signal | None:
        stacked = self._stacked(chain)
        if len(stacked) < 2:
            return None
        dr = self._dealing_range(chain)
        if dr is None:
            return None
        range_low, range_high = dr
        equilibrium = (range_low + range_high) / 2.0
        last_price = self._num(chain.notes.get("last_price"), None)
        if last_price is None:
            return None
        if chain.direction == Direction.BULLISH and not last_price < equilibrium:
            return None
        if chain.direction == Direction.BEARISH and not last_price > equilibrium:
            return None
        deepest = stacked[-1]
        ob_high = self._num(self._field(deepest, "high"), None)
        ob_low = self._num(self._field(deepest, "low"), None)
        fifty = self._num(self._field(deepest, "fifty_pct"), None)
        if ob_high is None or ob_low is None or fifty is None or not ob_high > ob_low:
            return None
        atr = self._atr(chain, ob_high, ob_low)
        entry = fifty
        stop = self._stop_beyond_far_side(ob_high, ob_low, chain.direction, atr)
        return self._build_signal(
            chain, chain.direction, entry, stop,
            chain.notes.get("target_liquidity"), chain.ltf_timeframe,
            {
                "ob_high": ob_high,
                "ob_low": ob_low,
                "stack_count": len(stacked),
                "dealing_range": [range_low, range_high],
                "equilibrium": equilibrium,
                "last_price": last_price,
                "atr": atr,
            },
        )

    def confluence_score(self, signal: Signal, ctx: dict) -> float:
        chain = ctx.get("chain") if isinstance(ctx, dict) else None
        if chain is None:
            return 0.0
        stacked = self._stacked(chain)
        dr = self._dealing_range(chain)
        if len(stacked) < 2 or dr is None:
            return 0.0
        range_low, range_high = dr
        equilibrium = (range_low + range_high) / 2.0
        last_price = self._num(chain.notes.get("last_price"), None)
        if last_price is None:
            return 0.0

        stack_count = min(1.0, (len(stacked) - 1) / 3.0) * 30.0

        if chain.direction == Direction.BULLISH:
            depth = (equilibrium - last_price) / max(1e-9, equilibrium - range_low)
        else:
            depth = (last_price - equilibrium) / max(1e-9, range_high - equilibrium)
        discount_depth = max(0.0, min(1.0, depth)) * 25.0

        ranges = []
        for ob in stacked:
            h = self._num(self._field(ob, "high"), None)
            l = self._num(self._field(ob, "low"), None)
            if h is not None and l is not None and h > l:
                ranges.append(h - l)
        atr = self._atr(chain, range_high, range_low)
        mean_range = sum(ranges) / len(ranges) if ranges else 0.0
        ob_quality = min(1.0, mean_range / atr) * 20.0

        htf = self._htf_points(chain, signal.direction, 15.0)

        liquidity = 10.0 if chain.notes.get("target_liquidity") else 3.0

        return min(100.0, stack_count + discount_depth + ob_quality + htf + liquidity)
