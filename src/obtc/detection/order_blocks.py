"""Order-block detection: displacement impulses, mitigation, breakers.

Lifecycle per block: ACTIVE -> MITIGATED -> (FLIPPED | INVALIDATED).
A block that closes through the far side without ever being touched is
INVALIDATED; one that was mitigated first and then closes through flips
into a breaker (FLIPPED). A flipped block that closes through its new far
side is INVALIDATED.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from obtc.chain.models import BlockState, Direction, OrderBlock, new_id


class OrderBlockDetector:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        displacement_atr_mult: float = 1.5,
        range_mult: float = 1.2,
        atr_period: int = 14,
        range_period: int = 20,
        volume_imbalance_mult: float = 1.5,
        volume_period: int = 20,
        max_wick_pct: float = 0.40,
        lookback_ob: int = 5,
        max_active: int = 100,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.displacement_atr_mult = displacement_atr_mult
        self.range_mult = range_mult
        self.atr_period = atr_period
        self.range_period = range_period
        self.volume_imbalance_mult = volume_imbalance_mult
        self.volume_period = volume_period
        self.max_wick_pct = max_wick_pct
        self.lookback_ob = lookback_ob
        self.max_active = max_active

        self._bars: deque = deque(maxlen=512)
        self._obs: list[OrderBlock] = []
        self._atr: float | None = None
        self._tr_seed: list[float] = []
        self._prev_close: float | None = None

    # ------------------------------------------------------------------ ATR
    @property
    def last_atr(self) -> float | None:
        """Wilder ATR (numpy SMA seed); None until atr_period bars seen."""
        return self._atr

    def _update_atr(self, bar: dict) -> None:
        high, low, close = bar["high"], bar["low"], bar["close"]
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        self._prev_close = close
        if self._atr is None:
            self._tr_seed.append(tr)
            if len(self._tr_seed) >= self.atr_period:
                self._atr = float(np.mean(np.asarray(self._tr_seed, dtype=float)))
        else:
            self._atr = (self._atr * (self.atr_period - 1) + tr) / self.atr_period

    # ---------------------------------------------------------------- update
    def update(self, bar: dict) -> list[dict]:
        """Process one bar dict; return event dicts {"type","ob","at"}."""
        events: list[dict] = []
        self._bars.append(bar)
        self._update_atr(bar)
        idx = len(self._bars) - 1

        impulse_dir = self._detect_impulse(idx)
        if impulse_dir is not None:
            self._extend_blocks(impulse_dir, bar, events)
            ob = self._try_create(idx, impulse_dir, bar)
            if ob is not None:
                self._obs.append(ob)
                self._prune()
                events.append({"type": "ob_created", "ob": ob, "at": bar["timestamp"]})

        new_ob = self._obs[-1] if impulse_dir is not None and events and events[-1]["type"] == "ob_created" else None
        for ob in list(self._obs):
            if ob is new_ob:
                continue  # a block is never mitigated by its own impulse bar
            if ob.state not in (BlockState.ACTIVE, BlockState.MITIGATED):
                continue
            ev = self._process_touch(ob, bar)
            if ev is not None:
                events.append(ev)
        return events

    # -------------------------------------------------------------- impulse
    def _detect_impulse(self, idx: int) -> Direction | None:
        if self._atr is None or self._atr <= 0:
            return None
        bars = self._bars
        if idx < max(self.range_period, self.volume_period):
            return None
        bar = bars[idx]
        prev_ranges = np.array(
            [b["high"] - b["low"] for b in list(bars)[idx - self.range_period: idx]],
            dtype=float,
        )
        prev_vols = np.array(
            [b["volume"] for b in list(bars)[idx - self.volume_period: idx]],
            dtype=float,
        )
        mean_range = float(np.mean(prev_ranges))
        mean_vol = float(np.mean(prev_vols))
        if mean_range <= 0 or mean_vol <= 0:
            return None
        body = bar["close"] - bar["open"]
        candle_range = bar["high"] - bar["low"]
        volume_ok = bar["volume"] >= self.volume_imbalance_mult * mean_vol
        displacement_ok = abs(body) >= self.displacement_atr_mult * self._atr
        range_ok = candle_range > self.range_mult * mean_range
        if not (volume_ok and displacement_ok and range_ok):
            return None
        return Direction.BULLISH if body > 0 else Direction.BEARISH

    def _find_ob_candle(self, idx: int, impulse_dir: Direction) -> dict | None:
        """Nearest opposite-direction candle within lookback_ob bars."""
        bars = list(self._bars)
        for j in range(idx - 1, max(idx - 1 - self.lookback_ob, -1), -1):
            b = bars[j]
            if impulse_dir == Direction.BULLISH and b["close"] < b["open"]:
                return b
            if impulse_dir == Direction.BEARISH and b["close"] > b["open"]:
                return b
        return None

    def _try_create(self, idx: int, impulse_dir: Direction, impulse_bar: dict):
        ob_candle = self._find_ob_candle(idx, impulse_dir)
        if ob_candle is None:
            return None
        candle_range = ob_candle["high"] - ob_candle["low"]
        if candle_range <= 0:
            return None
        body = abs(ob_candle["close"] - ob_candle["open"])
        wick_pct = (candle_range - body) / candle_range
        if wick_pct > self.max_wick_pct:
            return None
        return OrderBlock(
            id=new_id("ob"),
            symbol=self.symbol,
            timeframe=self.timeframe,
            direction=impulse_dir,
            high=float(ob_candle["high"]),
            low=float(ob_candle["low"]),
            fifty_pct=(float(ob_candle["high"]) + float(ob_candle["low"])) / 2.0,
            open=float(ob_candle["open"]),
            close=float(ob_candle["close"]),
            created_at=impulse_bar["timestamp"],
            state=BlockState.ACTIVE,
            origin_index=idx,
            volume=float(impulse_bar["volume"]),
            atr_at_creation=float(self._atr) if self._atr else 0.0,
        )

    # ------------------------------------------------------------- extend
    def _extend_blocks(self, impulse_dir: Direction, impulse_bar: dict, events: list) -> None:
        for ob in self._obs:
            if ob.state not in (BlockState.ACTIVE, BlockState.MITIGATED):
                continue
            if ob.direction != impulse_dir:
                continue
            if ob.is_bullish:
                if ob.high < impulse_bar["high"]:
                    ob.high = float(impulse_bar["high"])
                    ob.fifty_pct = (ob.high + ob.low) / 2.0
                    ob.extended = True
                    events.append({"type": "ob_extended", "ob": ob,
                                   "at": impulse_bar["timestamp"]})
            else:
                if ob.low > impulse_bar["low"]:
                    ob.low = float(impulse_bar["low"])
                    ob.fifty_pct = (ob.high + ob.low) / 2.0
                    ob.extended = True
                    events.append({"type": "ob_extended", "ob": ob,
                                   "at": impulse_bar["timestamp"]})

    # -------------------------------------------------------------- touch
    def _process_touch(self, ob: OrderBlock, bar: dict) -> dict | None:
        at = bar["timestamp"]
        if ob.is_bullish:
            if bar["close"] < ob.low:
                if ob.state == BlockState.ACTIVE:
                    ob.state = BlockState.INVALIDATED
                    ob.invalidated_at = at
                    ob.invalidation_wick = ob.low - bar["low"]
                    return {"type": "ob_invalidated", "ob": ob, "at": at}
                ob.state = BlockState.FLIPPED
                ob.flipped_from = Direction.BULLISH
                ob.direction = Direction.BEARISH
                ob.flipped_at = at
                return {"type": "ob_flipped", "ob": ob, "at": at}
            if ob.state == BlockState.ACTIVE and bar["low"] <= ob.high and bar["close"] >= ob.low:
                ob.state = BlockState.MITIGATED
                ob.mitigated_at = at
                return {"type": "ob_mitigated", "ob": ob, "at": at}
        else:
            if bar["close"] > ob.high:
                if ob.state == BlockState.ACTIVE:
                    ob.state = BlockState.INVALIDATED
                    ob.invalidated_at = at
                    ob.invalidation_wick = bar["high"] - ob.high
                    return {"type": "ob_invalidated", "ob": ob, "at": at}
                ob.state = BlockState.FLIPPED
                ob.flipped_from = Direction.BEARISH
                ob.direction = Direction.BULLISH
                ob.flipped_at = at
                return {"type": "ob_flipped", "ob": ob, "at": at}
            if ob.state == BlockState.ACTIVE and bar["high"] >= ob.low and bar["close"] <= ob.high:
                ob.state = BlockState.MITIGATED
                ob.mitigated_at = at
                return {"type": "ob_mitigated", "ob": ob, "at": at}
        return None

    # -------------------------------------------------------------- access
    def get_active(self) -> list[OrderBlock]:
        return [
            ob for ob in self._obs
            if ob.state in (BlockState.ACTIVE, BlockState.MITIGATED, BlockState.FLIPPED)
        ]

    @property
    def order_blocks(self) -> list[OrderBlock]:
        return list(self._obs)

    def _prune(self) -> None:
        if len(self._obs) > self.max_active:
            self._obs.sort(key=lambda ob: ob.created_at)
            self._obs = self._obs[-self.max_active:]
