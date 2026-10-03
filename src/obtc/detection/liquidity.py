"""Swing points, equal highs/lows and liquidity sweeps.

A swing is confirmed only once ``swing_lookback`` bars have printed on each
side, so swing events arrive lagged with the swing bar's own timestamp.
"""
from __future__ import annotations

from collections import deque

from obtc.chain.models import Direction, Sweep, new_id


class LiquidityDetector:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        swing_lookback: int = 5,
        equal_tol_pct: float = 0.05,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.swing_lookback = swing_lookback
        self.equal_tol_pct = equal_tol_pct

        self._buffer: deque = deque(maxlen=2 * swing_lookback + 8)
        self._swings: list[dict] = []  # {"price","at","kind","swept","equal","broken"}
        self.sweeps: list[Sweep] = []

    # ---------------------------------------------------------------- update
    def update(self, bar: dict) -> list[dict]:
        events: list[dict] = []
        self._buffer.append(bar)
        confirmed = self._confirm_swing()
        if confirmed is not None:
            events.append(confirmed)
        events.extend(self._check_sweeps(bar))
        return events

    def _confirm_swing(self) -> dict | None:
        n = self.swing_lookback
        if len(self._buffer) < 2 * n + 1:
            return None
        bars = list(self._buffer)
        center = bars[len(bars) - n - 1]
        left = bars[len(bars) - 2 * n - 1: len(bars) - n - 1]
        right = bars[len(bars) - n: len(bars)]
        highs = [b["high"] for b in left + right]
        lows = [b["low"] for b in left + right]
        swing = None
        if center["high"] >= max(highs):
            swing = {"price": float(center["high"]), "at": center["timestamp"],
                     "kind": "high", "swept": False, "equal": False, "broken": False}
        elif center["low"] <= min(lows):
            swing = {"price": float(center["low"]), "at": center["timestamp"],
                     "kind": "low", "swept": False, "equal": False, "broken": False}
        if swing is None:
            return None
        self._swings.append(swing)
        self._mark_equal(swing)
        return {"type": "swing_high" if swing["kind"] == "high" else "swing_low",
                "level": swing, "at": swing["at"]}

    def _mark_equal(self, swing: dict) -> None:
        for other in self._swings:
            if other is swing or other["kind"] != swing["kind"]:
                continue
            if other["price"] <= 0:
                continue
            if abs(swing["price"] - other["price"]) / other["price"] <= self.equal_tol_pct / 100.0:
                swing["equal"] = True
                other["equal"] = True

    def _check_sweeps(self, bar: dict) -> list[dict]:
        events: list[dict] = []
        at = bar["timestamp"]
        for level in self._swings:
            if level["swept"]:
                continue
            if level["kind"] == "high":
                if bar["high"] > level["price"] and bar["close"] < level["price"]:
                    sweep = Sweep(
                        id=new_id("sweep"), symbol=self.symbol,
                        timeframe=self.timeframe, level=level["price"],
                        side="BUY_SIDE", swept_at=at,
                        sweep_high=float(bar["high"]), sweep_low=float(bar["low"]),
                        closed_back_inside=True,
                        implied_direction=Direction.BEARISH,
                    )
                    level["swept"] = True
                    self.sweeps.append(sweep)
                    events.append({"type": "sweep", "sweep": sweep, "at": at})
            else:
                if bar["low"] < level["price"] and bar["close"] > level["price"]:
                    sweep = Sweep(
                        id=new_id("sweep"), symbol=self.symbol,
                        timeframe=self.timeframe, level=level["price"],
                        side="SELL_SIDE", swept_at=at,
                        sweep_high=float(bar["high"]), sweep_low=float(bar["low"]),
                        closed_back_inside=True,
                        implied_direction=Direction.BULLISH,
                    )
                    level["swept"] = True
                    self.sweeps.append(sweep)
                    events.append({"type": "sweep", "sweep": sweep, "at": at})
        return events

    # ---------------------------------------------------------------- access
    def get_swings(self) -> list[dict]:
        return [dict(s) for s in self._swings]

    def get_levels(self) -> list[dict]:
        return [
            {"price": s["price"], "kind": s["kind"],
             "equal": s["equal"], "swept": s["swept"]}
            for s in self._swings
        ]
