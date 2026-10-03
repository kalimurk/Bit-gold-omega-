"""FullChain assembly across HTF / MTF / LTF detector sets.

Pipeline per timeframe:
- HTF: structure bias -> get-or-create directional chain, attach HTF order
  block + recent liquidity sweep, invalidate stale-direction chains.
- MTF: attach active FVGs / flipped breaker OBs -> chain VALID.
- LTF: MSS shift + touched OB + active FVG -> chain TRIGGERED; session
  (Asia) and dealing-range context in notes.

Call ``update`` with bars in chronological order. Bars whose timeframe is
not one of the three configured timeframes are ignored.
"""
from __future__ import annotations

from collections import deque
from datetime import time as dtime

from obtc.chain.models import (
    Bias,
    BlockState,
    ChainState,
    Direction,
    FullChain,
    new_id,
)
from obtc.detection.fvg import FVGDetector
from obtc.detection.liquidity import LiquidityDetector
from obtc.detection.order_blocks import OrderBlockDetector
from obtc.detection.structure import StructureDetector

_ACTIVE_CHAIN_STATES = (
    ChainState.FORMING,
    ChainState.VALID,
    ChainState.TRIGGERED,
)

_DEFAULT_CONFIG = {
    "max_chain_age_hours": 48,
    "asia_start": "00:00",
    "asia_end": "07:00",
    "dealing_range_lookback": 50,
    "sweep_lookback_bars": 30,
}


def _parse_hhmm(value: str) -> dtime:
    hh, mm = value.split(":")
    return dtime(int(hh), int(mm))


class ChainBuilder:
    def __init__(
        self,
        symbol: str,
        htf_timeframe: str = "4H",
        mtf_timeframe: str = "1H",
        ltf_timeframe: str = "5m",
        config: dict | None = None,
    ) -> None:
        self.symbol = symbol
        self.htf_timeframe = htf_timeframe
        self.mtf_timeframe = mtf_timeframe
        self.ltf_timeframe = ltf_timeframe
        self.config = {**_DEFAULT_CONFIG, **(config or {})}

        self._tf = {
            htf_timeframe: self._make_detectors(htf_timeframe),
            mtf_timeframe: self._make_detectors(mtf_timeframe),
            ltf_timeframe: self._make_detectors(ltf_timeframe),
        }
        self._bar_index = {htf_timeframe: 0, mtf_timeframe: 0, ltf_timeframe: 0}
        # timeframe -> list of (Sweep, bar_index)
        self._sweep_log: dict[str, list] = {tf: [] for tf in self._tf}

        self._chains: dict[str, FullChain] = {}
        self._prev_htf_bias = Bias.NEUTRAL

        self._asia = {"day": None, "high": None, "low": None}
        self._ltf_recent: deque = deque(maxlen=3)
        self._ltf_window: deque = deque(
            maxlen=int(self.config["dealing_range_lookback"])
        )

    def _make_detectors(self, timeframe: str) -> dict:
        liq = LiquidityDetector(self.symbol, timeframe)
        return {
            "ob": OrderBlockDetector(self.symbol, timeframe),
            "fvg": FVGDetector(self.symbol, timeframe),
            "liq": liq,
            "struct": StructureDetector(liq, self.symbol, timeframe),
        }

    # ------------------------------------------------------------------ api
    async def update(self, bar: dict) -> list[FullChain]:
        """Process one bar; return chains that changed state in this call."""
        tf = bar.get("timeframe")
        if tf not in self._tf:
            return []
        changed: list[FullChain] = []
        at = bar["timestamp"]
        self._expire_chains(at, changed)

        if tf == self.htf_timeframe:
            self._update_htf(bar, changed)
        elif tf == self.mtf_timeframe:
            self._update_mtf(bar, changed)
        else:
            self._update_ltf(bar, changed)
        return changed

    def get_active_chains(
        self, states: tuple = _ACTIVE_CHAIN_STATES
    ) -> list[FullChain]:
        return [c for c in self._chains.values() if c.state in states]

    def get_chain(self, chain_id: str) -> FullChain | None:
        return self._chains.get(chain_id)

    # ---------------------------------------------------------------- helpers
    def _det(self, tf: str) -> dict:
        return self._tf[tf]

    def _record_sweeps(self, tf: str) -> list:
        """Return sweeps that appeared on this bar, logging their bar index."""
        det = self._det(tf)
        log = self._sweep_log[tf]
        prev = len(log)
        new = det["liq"].sweeps[prev:]
        idx = self._bar_index[tf]
        for sweep in new:
            log.append((sweep, idx))
        return new

    def _link(self, chain: FullChain, timeframe: str, kind: str,
              ref_id: str, note: str = "") -> None:
        if any(l.kind == kind and l.ref_id == ref_id for l in chain.links):
            return
        chain.add_link(timeframe, kind, ref_id, note=note)

    def _invalidate(self, chain: FullChain, at, changed: list) -> None:
        if chain.state in _ACTIVE_CHAIN_STATES:
            chain.state = ChainState.INVALIDATED
            chain.invalidated_at = at
            chain.updated_at = at
            changed.append(chain)

    def _expire_chains(self, at, changed: list) -> None:
        max_age_s = float(self.config["max_chain_age_hours"]) * 3600.0
        for chain in self._chains.values():
            if chain.state in _ACTIVE_CHAIN_STATES:
                age = (at - chain.created_at).total_seconds()
                if age > max_age_s:
                    chain.state = ChainState.EXPIRED
                    chain.updated_at = at
                    changed.append(chain)

    def _active_chain_for(self, direction: Direction) -> FullChain | None:
        for chain in self._chains.values():
            if chain.direction == direction and chain.state in _ACTIVE_CHAIN_STATES:
                return chain
        return None

    # ------------------------------------------------------------------- HTF
    def _update_htf(self, bar: dict, changed: list) -> None:
        tf = self.htf_timeframe
        det = self._det(tf)
        self._bar_index[tf] += 1
        det["ob"].update(bar)
        det["fvg"].update(bar)
        det["struct"].update(bar)
        self._record_sweeps(tf)
        at = bar["timestamp"]

        trend = det["struct"].trend
        bias = {"LONG": Bias.LONG, "SHORT": Bias.SHORT}.get(trend, Bias.NEUTRAL)

        if bias != Bias.NEUTRAL:
            bias_dir = Direction.BULLISH if bias == Bias.LONG else Direction.BEARISH
            if bias != self._prev_htf_bias:
                for chain in self._chains.values():
                    if (chain.direction != bias_dir
                            and chain.state in _ACTIVE_CHAIN_STATES):
                        self._invalidate(chain, at, changed)
            self._prev_htf_bias = bias

            chain = self._active_chain_for(bias_dir)
            if chain is None:
                chain = FullChain(
                    id=new_id("chain"),
                    symbol=self.symbol,
                    direction=bias_dir,
                    htf_bias=bias,
                    htf_timeframe=self.htf_timeframe,
                    mtf_timeframe=self.mtf_timeframe,
                    ltf_timeframe=self.ltf_timeframe,
                    state=ChainState.FORMING,
                    created_at=at,
                    updated_at=at,
                )
                self._chains[chain.id] = chain
                self._link(chain, tf, "chain_created", chain.id,
                           note=f"htf bias {bias.value}")

            cands = [ob for ob in det["ob"].order_blocks
                     if ob.direction == chain.direction
                     and ob.state != BlockState.INVALIDATED]
            if cands:
                latest = max(cands, key=lambda ob: ob.created_at)
                if chain.htf_ob is None or chain.htf_ob.id != latest.id:
                    chain.htf_ob = latest
                    chain.updated_at = at
                    self._link(chain, tf, "htf_ob", latest.id,
                               note=f"{latest.state.value} {latest.timeframe} OB")

            lookback = int(self.config["sweep_lookback_bars"])
            idx = self._bar_index[tf]
            recent = [s for s, s_idx in self._sweep_log[tf]
                      if idx - s_idx <= lookback]
            if recent:
                sweep = recent[-1]
                if chain.liquidity_sweep is None or chain.liquidity_sweep.id != sweep.id:
                    chain.liquidity_sweep = sweep
                    chain.updated_at = at
                    self._link(chain, tf, "liquidity_sweep", sweep.id,
                               note=f"{sweep.side} sweep @ {sweep.level:g}")

            chain.notes["target_liquidity"] = self._target_liquidity(det, chain, bar)
            chain.notes["atr_htf"] = det["ob"].last_atr

            if (chain.htf_ob is not None
                    and chain.htf_ob.state == BlockState.INVALIDATED):
                self._invalidate(chain, at, changed)

    def _target_liquidity(self, det: dict, chain: FullChain, bar: dict):
        want = "low" if chain.direction == Direction.BULLISH else "high"
        cands = [l["price"] for l in det["liq"].get_levels()
                 if l["kind"] == want and not l["swept"]]
        if not cands:
            return None
        return min(cands, key=lambda p: abs(p - bar["close"]))

    # ------------------------------------------------------------------- MTF
    def _update_mtf(self, bar: dict, changed: list) -> None:
        tf = self.mtf_timeframe
        det = self._det(tf)
        self._bar_index[tf] += 1
        det["ob"].update(bar)
        det["fvg"].update(bar)
        det["struct"].update(bar)
        self._record_sweeps(tf)
        at = bar["timestamp"]

        for chain in self._chains.values():
            if chain.state not in (ChainState.FORMING, ChainState.VALID):
                continue
            fvgs = [f for f in det["fvg"].get_active()
                    if f.direction == chain.direction]
            chain.mtf_fvgs = list(fvgs)
            flipped = [ob for ob in det["ob"].get_active()
                       if ob.direction == chain.direction
                       and ob.state == BlockState.FLIPPED]
            breaker = max(flipped, key=lambda ob: ob.created_at) if flipped else None
            chain.mtf_breaker = breaker

            if chain.htf_ob and (fvgs or breaker):
                for fvg in fvgs:
                    self._link(chain, tf, "mtf_fvg", fvg.id,
                               note=f"{fvg.timeframe} FVG {fvg.bottom:g}-{fvg.top:g}")
                if breaker is not None:
                    self._link(chain, tf, "mtf_breaker", breaker.id,
                               note=f"flipped {breaker.timeframe} OB")
                if chain.state != ChainState.VALID:
                    chain.state = ChainState.VALID
                    chain.updated_at = at
                    changed.append(chain)

            if breaker is not None:
                in_zone = (bar["low"] <= breaker.high
                           and bar["high"] >= breaker.low)
                chain.notes["breaker_retest"] = bool(in_zone)
            chain.notes["atr_mtf"] = det["ob"].last_atr

    # ------------------------------------------------------------------- LTF
    def _update_ltf(self, bar: dict, changed: list) -> None:
        tf = self.ltf_timeframe
        det = self._det(tf)
        self._bar_index[tf] += 1
        det["ob"].update(bar)
        det["fvg"].update(bar)
        shifts = det["struct"].update(bar)
        new_sweeps = self._record_sweeps(tf)
        at = bar["timestamp"]

        self._track_asia(bar)
        self._tag_asia_sweeps(new_sweeps)
        self._ltf_recent.append(bar)
        self._ltf_window.append(bar)

        for chain in self._chains.values():
            if chain.state != ChainState.VALID:
                continue
            mss = [s for s in shifts
                   if s.kind == "MSS" and s.direction == chain.direction]
            if mss:
                chain.ltf_shift = mss[-1]
                chain.updated_at = at
                self._link(chain, tf, "ltf_shift", mss[-1].id,
                           note=f"MSS {mss[-1].broken_level:g}")

            recent = list(self._ltf_recent)
            ob_cands = [ob for ob in det["ob"].get_active()
                        if ob.direction == chain.direction
                        and ob.state in (BlockState.ACTIVE, BlockState.MITIGATED)
                        and any(b["low"] <= ob.high and b["high"] >= ob.low
                                for b in recent)]
            if ob_cands:
                latest_ob = max(ob_cands, key=lambda ob: ob.created_at)
                if chain.ltf_ob is None or chain.ltf_ob.id != latest_ob.id:
                    chain.ltf_ob = latest_ob
                    chain.updated_at = at
                    self._link(chain, tf, "ltf_ob", latest_ob.id,
                               note=f"{latest_ob.state.value} {latest_ob.timeframe} OB")

            fvg_cands = [f for f in det["fvg"].get_active()
                         if f.direction == chain.direction]
            if fvg_cands:
                latest_fvg = max(fvg_cands, key=lambda f: f.created_at)
                if chain.ltf_fvg is None or chain.ltf_fvg.id != latest_fvg.id:
                    chain.ltf_fvg = latest_fvg
                    chain.updated_at = at
                    self._link(chain, tf, "ltf_fvg", latest_fvg.id,
                               note=f"{latest_fvg.timeframe} FVG")

            if chain.ltf_shift and chain.ltf_ob and chain.ltf_fvg:
                if chain.state != ChainState.TRIGGERED:
                    chain.state = ChainState.TRIGGERED
                    chain.updated_at = at
                    changed.append(chain)

            chain.notes["asia_high"] = self._asia["high"]
            chain.notes["asia_low"] = self._asia["low"]
            if self._ltf_window:
                chain.notes["dealing_range"] = (
                    min(b["low"] for b in self._ltf_window),
                    max(b["high"] for b in self._ltf_window),
                )
            chain.notes["stacked_obs"] = [
                ob for ob in det["ob"].get_active()
                if ob.direction == chain.direction
            ]
            chain.notes["last_price"] = bar["close"]
            chain.notes["atr_ltf"] = det["ob"].last_atr

    # ------------------------------------------------------------------ asia
    def _track_asia(self, bar: dict) -> None:
        ts = bar["timestamp"]
        day = ts.date()
        if self._asia["day"] != day:
            self._asia = {"day": day, "high": None, "low": None}
        start, end = (_parse_hhmm(self.config["asia_start"]),
                      _parse_hhmm(self.config["asia_end"]))
        t = ts.time()
        in_window = (start <= t < end) if start <= end else (t >= start or t < end)
        if in_window:
            if self._asia["high"] is None or bar["high"] > self._asia["high"]:
                self._asia["high"] = bar["high"]
            if self._asia["low"] is None or bar["low"] < self._asia["low"]:
                self._asia["low"] = bar["low"]

    def _tag_asia_sweeps(self, sweeps: list) -> None:
        high, low = self._asia["high"], self._asia["low"]
        for sweep in sweeps:
            if high and abs(sweep.level - high) / high <= 0.001:
                sweep.meta["session"] = "asia"
            elif low and abs(sweep.level - low) / low <= 0.001:
                sweep.meta["session"] = "asia"
