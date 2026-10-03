"""Fair-value-gap detection: 3-candle displacement imbalances."""
from __future__ import annotations

from collections import deque

import numpy as np

from obtc.chain.models import BlockState, Direction, FairValueGap, new_id


class FVGDetector:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        min_gap_atr_mult: float = 0.25,
        atr_period: int = 14,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.min_gap_atr_mult = min_gap_atr_mult
        self.atr_period = atr_period

        self._bars: deque = deque(maxlen=64)
        self._fvgs: list[FairValueGap] = []
        self._atr: float | None = None
        self._tr_seed: list[float] = []
        self._prev_close: float | None = None

    @property
    def last_atr(self) -> float | None:
        return self._atr

    def _update_atr(self, bar: dict) -> None:
        high, low, close = bar["high"], bar["low"], bar["close"]
        tr = high - low if self._prev_close is None else max(
            high - low, abs(high - self._prev_close), abs(low - self._prev_close)
        )
        self._prev_close = close
        if self._atr is None:
            self._tr_seed.append(tr)
            if len(self._tr_seed) >= self.atr_period:
                self._atr = float(np.mean(np.asarray(self._tr_seed, dtype=float)))
        else:
            self._atr = (self._atr * (self.atr_period - 1) + tr) / self.atr_period

    def update(self, bar: dict) -> list[dict]:
        events: list[dict] = []
        self._bars.append(bar)
        self._update_atr(bar)

        if len(self._bars) >= 3 and self._atr and self._atr > 0:
            fvg = self._try_create()
            if fvg is not None:
                self._fvgs.append(fvg)
                events.append({"type": "fvg_created", "fvg": fvg,
                               "at": bar["timestamp"]})

        for fvg in list(self._fvgs):
            if fvg.state != BlockState.ACTIVE:
                continue
            ev = self._process(fvg, bar)
            if ev is not None:
                events.append(ev)
        return events

    def _try_create(self) -> FairValueGap | None:
        bars = list(self._bars)
        b0, b2 = bars[-3], bars[-1]
        min_gap = self.min_gap_atr_mult * self._atr
        if (
            b2["low"] > b0["high"]
            and b2["close"] > b2["open"]
            and (b2["low"] - b0["high"]) >= min_gap
        ):
            return FairValueGap(
                id=new_id("fvg"),
                symbol=self.symbol,
                timeframe=self.timeframe,
                direction=Direction.BULLISH,
                top=float(b2["low"]),
                bottom=float(b0["high"]),
                created_at=b2["timestamp"],
                origin_index=len(self._bars) - 1,
            )
        if (
            b2["high"] < b0["low"]
            and b2["close"] < b2["open"]
            and (b0["low"] - b2["high"]) >= min_gap
        ):
            return FairValueGap(
                id=new_id("fvg"),
                symbol=self.symbol,
                timeframe=self.timeframe,
                direction=Direction.BEARISH,
                top=float(b0["low"]),
                bottom=float(b2["high"]),
                created_at=b2["timestamp"],
                origin_index=len(self._bars) - 1,
            )
        return None

    def _process(self, fvg: FairValueGap, bar: dict) -> dict | None:
        at = bar["timestamp"]
        if fvg.is_bullish:
            if bar["close"] < fvg.bottom:
                fvg.state = BlockState.INVALIDATED
                fvg.invalidated_at = at
                return {"type": "fvg_invalidated", "fvg": fvg, "at": at}
            if bar["low"] <= fvg.bottom:
                fvg.state = BlockState.FILLED
                fvg.filled_at = at
                return {"type": "fvg_filled", "fvg": fvg, "at": at}
        else:
            if bar["close"] > fvg.top:
                fvg.state = BlockState.INVALIDATED
                fvg.invalidated_at = at
                return {"type": "fvg_invalidated", "fvg": fvg, "at": at}
            if bar["high"] >= fvg.top:
                fvg.state = BlockState.FILLED
                fvg.filled_at = at
                return {"type": "fvg_filled", "fvg": fvg, "at": at}
        return None

    def get_active(self) -> list[FairValueGap]:
        return [f for f in self._fvgs if f.state == BlockState.ACTIVE]
