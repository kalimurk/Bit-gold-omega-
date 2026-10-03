"""Market structure: BOS / MSS from swing breaks, trend state.

Owns the liquidity detector's update cycle for its timeframe: call
``structure.update(bar)`` (which forwards the bar to liquidity) rather than
calling ``liquidity.update(bar)`` separately, or swings will double-count.

Reference levels: the most recent confirmed swing high/low that is neither
swept nor already broken. A break marks its level broken, so the reference
advances to the next latest swing and consecutive breaks keep emitting.
"""
from __future__ import annotations

from obtc.chain.models import Direction, MarketShift, new_id
from obtc.detection.liquidity import LiquidityDetector


class StructureDetector:
    def __init__(self, liquidity: LiquidityDetector, symbol: str, timeframe: str) -> None:
        self._liquidity = liquidity
        self.symbol = symbol
        self.timeframe = timeframe
        self._trend = "NEUTRAL"
        self._index = 0

    @property
    def trend(self) -> str:
        return self._trend

    @property
    def liquidity(self) -> LiquidityDetector:
        return self._liquidity

    def update(self, bar: dict) -> list[MarketShift]:
        liq_events = self._liquidity.update(bar)
        for _ in liq_events:
            pass  # swing bookkeeping lives in the liquidity detector
        shifts: list[MarketShift] = []
        close = bar["close"]
        ref_high = self._latest_level("high")
        ref_low = self._latest_level("low")
        if ref_high is not None and close > ref_high["price"]:
            kind = "BOS" if self._trend == "LONG" else "MSS"
            shifts.append(self._make_shift(kind, Direction.BULLISH,
                                          ref_high["price"], bar))
            ref_high["broken"] = True
            self._trend = "LONG"
        elif ref_low is not None and close < ref_low["price"]:
            kind = "BOS" if self._trend == "SHORT" else "MSS"
            shifts.append(self._make_shift(kind, Direction.BEARISH,
                                          ref_low["price"], bar))
            ref_low["broken"] = True
            self._trend = "SHORT"
        self._index += 1
        return shifts

    def _latest_level(self, kind: str) -> dict | None:
        cands = [s for s in self._liquidity._swings
                 if s["kind"] == kind and not s["swept"] and not s["broken"]]
        if not cands:
            return None
        return max(cands, key=lambda s: s["at"])

    def _make_shift(self, kind: str, direction: Direction,
                    level: float, bar: dict) -> MarketShift:
        return MarketShift(
            id=new_id("ms"),
            symbol=self.symbol,
            timeframe=self.timeframe,
            kind=kind,
            direction=direction,
            broken_level=float(level),
            broken_at=bar["timestamp"],
            origin_index=self._index,
        )
