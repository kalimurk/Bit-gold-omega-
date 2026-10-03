"""Tests for OB / FVG / liquidity / structure detectors (synthetic only)."""
from datetime import timedelta

import pandas as pd

from conftest import feed_bars, make_bars, make_impulse
from obtc.chain.models import BlockState, Direction
from obtc.data.normalize import bar_to_dict
from obtc.detection.fvg import FVGDetector
from obtc.detection.liquidity import LiquidityDetector
from obtc.detection.order_blocks import OrderBlockDetector
from obtc.detection.structure import StructureDetector

SYMBOL = "BTC/USD"


def _ob_detector(**kw):
    return OrderBlockDetector(SYMBOL, "1m", **kw)


def _next_ts(df):
    return df["timestamp"].iloc[-1] + timedelta(minutes=1)


def _mkbar(ts, o, h, l, c, v=12.0):
    return {
        "timestamp": ts,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "timeframe": "1m",
        "symbol": SYMBOL,
    }


def test_ob_created_with_correct_zone():
    df = make_impulse("bull")
    det = _ob_detector()
    events = feed_bars(det, df)
    created = [e for e in events if e["type"] == "ob_created"]
    assert len(created) == 1, f"expected 1 OB, got {len(created)}"
    ob = created[0]["ob"]
    opp = df.iloc[25]  # the crafted opposite candle
    assert ob.direction == Direction.BULLISH
    assert ob.high == opp["high"]
    assert ob.low == opp["low"]
    assert ob.fifty_pct == (opp["high"] + opp["low"]) / 2
    assert ob.state == BlockState.ACTIVE
    assert ob.volume == df.iloc[26]["volume"]  # impulse volume
    assert det.last_atr is not None and det.last_atr > 0


def test_ob_mitigation_on_zone_touch():
    df = make_impulse("bull")
    det = _ob_detector()
    feed_bars(det, df)
    ob = det.get_active()[0]
    ts = _next_ts(df)
    # dip into the zone but close back above its low
    bar = _mkbar(ts, ob.high + 20, ob.high + 30, ob.low - 5, ob.low + 10)
    events = det.update(bar)
    mitig = [e for e in events if e["type"] == "ob_mitigated"]
    assert len(mitig) == 1
    assert ob.state == BlockState.MITIGATED
    assert ob.mitigated_at == ts


def test_ob_breaker_flip_after_mitigation():
    df = make_impulse("bull")
    det = _ob_detector()
    feed_bars(det, df)
    ob = det.get_active()[0]
    ts = _next_ts(df)
    det.update(_mkbar(ts, ob.high + 20, ob.high + 30, ob.low - 5, ob.low + 10))
    assert ob.state == BlockState.MITIGATED
    ts2 = ts + timedelta(minutes=1)
    events = det.update(_mkbar(ts2, ob.low + 5, ob.low + 8, ob.low - 40, ob.low - 20))
    flipped = [e for e in events if e["type"] == "ob_flipped"]
    assert len(flipped) == 1
    assert ob.state == BlockState.FLIPPED
    assert ob.flipped_from == Direction.BULLISH
    assert ob.direction == Direction.BEARISH
    assert ob.flipped_at == ts2


def test_ob_invalidation_close_through_untouched():
    df = make_impulse("bull")
    det = _ob_detector()
    feed_bars(det, df)
    ob = det.get_active()[0]
    ts = _next_ts(df)
    bar = _mkbar(ts, ob.low + 50, ob.low + 60, ob.low - 30, ob.low - 10)
    events = det.update(bar)
    inv = [e for e in events if e["type"] == "ob_invalidated"]
    assert len(inv) == 1
    assert ob.state == BlockState.INVALIDATED
    assert ob.invalidated_at == ts
    assert ob.invalidation_wick == ob.low - bar["low"]


def test_ob_filter_rejects_low_volume_impulse():
    df = make_impulse("bull").copy()
    df.loc[26, "volume"] = 5.0  # starve the impulse of volume
    det = _ob_detector()
    events = feed_bars(det, df)
    assert not [e for e in events if e["type"] == "ob_created"]


def test_ob_filter_rejects_big_wick_ob_candle():
    df = make_impulse("bull").copy()
    opp = df.loc[25]
    df.loc[25, "high"] = opp["open"] + 100  # wick_pct = 0.8
    df.loc[25, "low"] = opp["close"] - 100
    det = _ob_detector()
    events = feed_bars(det, df)
    assert not [e for e in events if e["type"] == "ob_created"]


def test_ob_extension_on_new_impulse():
    df = make_impulse("bull")
    det = _ob_detector()
    feed_bars(det, df)
    ob = det.get_active()[0]
    old_high = ob.high
    ts = _next_ts(df)
    # fresh bullish impulse printing a higher high
    det.update(_mkbar(ts, ob.high, ob.high + 500, ob.low + 50, ob.high + 400, v=2000))
    assert ob.high > old_high
    assert ob.extended is True


def _fvg_setup(det, base=100.0):
    """Warm up ATR then return timestamps for crafted FVG bars."""
    warm = make_bars(16, seed=3, base=base, vol=2.0)
    feed_bars(det, warm)
    ts = _next_ts(warm)
    b0 = _mkbar(ts, 98.0, 100.0, 97.0, 99.0)  # high 100
    b1 = _mkbar(ts + timedelta(minutes=1), 99.0, 102.0, 98.5, 101.0)
    b2 = _mkbar(ts + timedelta(minutes=2), 101.5, 112.0, 110.0, 111.5)  # gap 10
    return det, [b0, b1, b2], ts


def test_fvg_created_filled_invalidated():
    det = FVGDetector(SYMBOL, "1m")
    det, bars, _ = _fvg_setup(det)
    events = []
    for b in bars:
        events.extend(det.update(b))
    created = [e for e in events if e["type"] == "fvg_created"]
    assert len(created) == 1
    fvg = created[0]["fvg"]
    assert fvg.direction == Direction.BULLISH
    assert fvg.top == 110.0 and fvg.bottom == 100.0

    ts = bars[-1]["timestamp"] + timedelta(minutes=1)
    events = det.update(_mkbar(ts, 105.0, 106.0, 99.0, 104.0))
    filled = [e for e in events if e["type"] == "fvg_filled"]
    assert len(filled) == 1
    assert fvg.state == BlockState.FILLED
    assert fvg.filled_at == ts

    # fresh detector: close through the far side invalidates instead
    det2 = FVGDetector(SYMBOL, "1m")
    det2, bars2, _ = _fvg_setup(det2)
    for b in bars2:
        det2.update(b)
    fvg2 = det2.get_active()[0]
    ts2 = bars2[-1]["timestamp"] + timedelta(minutes=1)
    events = det2.update(_mkbar(ts2, 105.0, 106.0, 98.0, 99.0))
    inv = [e for e in events if e["type"] == "fvg_invalidated"]
    assert len(inv) == 1
    assert fvg2.state == BlockState.INVALIDATED
    assert fvg2.invalidated_at == ts2


def _liq_bar(ts, o, h, l, c, v=10.0):
    return _mkbar(ts, o, h, l, c, v)


def test_swing_and_sweep():
    det = LiquidityDetector(SYMBOL, "1m", swing_lookback=5)
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10]
    events = []
    for i, h in enumerate(highs):
        events.extend(
            det.update(_liq_bar(ts + timedelta(minutes=i), h - 2, h, h - 4, h - 1))
        )
    swings = [e for e in events if e["type"] == "swing_high"]
    assert len(swings) == 1
    assert swings[0]["level"]["price"] == 20.0
    assert swings[0]["at"] == ts + timedelta(minutes=5)

    sweep_bar = _liq_bar(
        ts + timedelta(minutes=11), 20.5, 21.0, 18.0, 19.0
    )
    events = det.update(sweep_bar)
    sweeps = [e for e in events if e["type"] == "sweep"]
    assert len(sweeps) == 1
    sw = sweeps[0]["sweep"]
    assert sw.side == "BUY_SIDE"
    assert sw.level == 20.0
    assert sw.implied_direction == Direction.BEARISH
    assert sw.closed_back_inside is True
    levels = det.get_levels()
    assert levels[0]["swept"] is True


def test_equal_highs_marked():
    det = LiquidityDetector(SYMBOL, "1m", swing_lookback=5)
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    # two swing highs at the same price, 12 bars apart
    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 13, 14, 20, 14, 13, 12, 11, 10]
    for i, h in enumerate(highs):
        det.update(_liq_bar(ts + timedelta(minutes=i), h - 2, h, h - 4, h - 1))
    levels = [l for l in det.get_levels() if l["kind"] == "high"]
    assert len(levels) == 2
    assert all(l["equal"] for l in levels)


def test_structure_mss_flips_trend():
    liq = LiquidityDetector(SYMBOL, "1m", swing_lookback=5)
    det = StructureDetector(liq, SYMBOL, "1m")
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    def sbar(i, h, c):
        return _liq_bar(ts + timedelta(minutes=i), h - 2, h, h - 4, c)

    shifts = []
    # swing high 110 confirmed at bar 10
    highs = [100, 102, 104, 106, 108, 110, 108, 106, 104, 102, 100]
    for i, h in enumerate(highs):
        shifts.extend(det.update(sbar(i, h, h - 1)))
    # break above -> MSS (trend was NEUTRAL); clean bars, no wick-sweep first
    for i, c in enumerate([108, 109, 110, 112], start=11):
        shifts.extend(det.update(sbar(i, c + 1, c)))
    mss_up = [s for s in shifts if s.kind == "MSS" and s.direction == Direction.BULLISH]
    assert len(mss_up) >= 1
    assert det.trend == "LONG"

    # swing low 90 confirmed at bar 25, then break below -> MSS bearish
    lows = [106, 104, 102, 100, 98, 96, 94, 92, 90, 92, 94]
    for j, lo in enumerate(lows):
        i = 15 + j
        shifts.extend(det.update(sbar(i, lo + 4, lo + 1)))
    for i, c in enumerate([92, 91, 89], start=26):
        shifts.extend(det.update(sbar(i, 94, c)))
    mss_dn = [s for s in shifts if s.kind == "MSS" and s.direction == Direction.BEARISH]
    assert len(mss_dn) >= 1
    assert det.trend == "SHORT"
