"""End-to-end ChainBuilder test on engineered deterministic 1m bars.

Timeline (UTC, 2024-01-05):
- 00:00-04:00 warmup chop (6000 bars back to 2024-01-01)
- 04:00-08:00 bearish 4H candle (OB origin)
- 08:00-12:00 bullish 4H impulse -> HTF OB + HTF bias LONG -> chain FORMING
- 13:00-14:00 1H spike with bullish FVG -> chain VALID
- 14:00-15:10 pullback (5m trend SHORT), 15:45-15:50 5m impulse with
  MSS + OB + FVG -> chain TRIGGERED
- 16:00 4H bar (left-labeled bin covering the 17:00-20:00 crash) closes
  through the HTF OB far side -> chain INVALIDATED
"""
import pandas as pd

from obtc.chain.builder import ChainBuilder
from obtc.chain.models import ChainState, Direction
from obtc.data.normalize import bar_to_dict
from obtc.data.resample import resample_bars

SYMBOL = "BTC/USD"


def build_1m() -> pd.DataFrame:
    import math

    import numpy as np

    rows = []
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    rng = np.random.default_rng(7)
    price = 60000.0

    def bar(o, h, l, c, v):
        nonlocal ts, price
        rows.append((ts, o, h, l, c, v))
        ts += pd.Timedelta(minutes=1)
        price = c
        return c

    # 1. warmup chop: 6000 x 1m oscillating (48h sine) so that clean swing
    # pivots confirm on 4H/1H/5m (a pure random walk grinds with no pivots)
    for k in range(6000):
        target = 60000.0 + 500.0 * math.sin(2 * math.pi * k / 2880.0)
        o = price
        c = target + rng.normal(0, 12)
        price = bar(o, max(o, c) + abs(rng.normal(0, 5)),
                    min(o, c) - abs(rng.normal(0, 5)), c,
                    8 + abs(rng.normal(0, 3)))

    # 2. bearish 4H candle 04:00-08:00: 60200 -> 60000, tiny wicks
    o = 60200.0
    step = -200.0 / 240
    for _ in range(240):
        c = o + step
        price = bar(o, o + 2, c - 2, c, 10.0)
        o = c

    # 3. bullish 4H impulse 08:00-12:00: 60000 -> 62500
    o = 60000.0
    step = 2500.0 / 240
    for _ in range(240):
        c = o + step
        price = bar(o, c + 5, o - 5, c, 30.0)
        o = c

    # 4. hold 12:00-13:00
    for _ in range(60):
        o = price
        c = o + rng.normal(0, 10)
        price = bar(o, max(o, c) + 8, min(o, c) - 8, c, 12.0)
    c_high = max(r[2] for r in rows[-60:])

    # 5. 1H spike 13:00-14:00: gap 600 over C high, then +300
    d_open = c_high + 600.0
    o = d_open
    for i in range(60):
        c = o + (30.0 if i < 10 else rng.normal(0, 8))
        h = c + 6
        l = (o - 2) if i == 0 else min(o, c) - 6
        price = bar(o, h, l, c, 25.0)
        o = c
    p0 = price

    # 6. pullback 14:00-15:10 in two legs with a bounce between them, so a
    # 5m swing low forms (leg-1 bottom) and the second leg breaks it,
    # flipping 5m trend SHORT via bearish MSS
    o = price
    for _ in range(20):  # leg 1: -25/min
        c = o - 25.0
        price = bar(o, max(o, c) + 6, min(o, c) - 6, c, 12.0)
        o = c
    for _ in range(30):  # bounce: +8/min
        c = o + 8.0
        price = bar(o, max(o, c) + 6, min(o, c) - 6, c, 12.0)
        o = c
    for _ in range(20):  # leg 2: -30/min, breaks the leg-1 swing low
        c = o - 30.0
        price = bar(o, max(o, c) + 6, min(o, c) - 6, c, 12.0)
        o = c

    # 7. base 15:10-15:40
    for _ in range(30):
        o = price
        c = o + rng.normal(0, 8)
        price = bar(o, max(o, c) + 5, min(o, c) - 5, c, 10.0)
        o = c
    p1 = price

    # 8a. G1 bearish 5m candle 15:40-15:45 (OB origin, wick_pct ~0.14)
    o = p1
    for i in range(5):
        c = o - 6.0
        if i == 0:
            h, l = o + 4, c - 4
        else:
            h, l = max(o, c) + 1, min(o, c) - 1
        price = bar(o, h, l, c, 12.0)
        o = c

    # 8b. G2 bullish 5m impulse 15:45-15:50, gapped up (MSS + FVG + OB)
    o = p1 + 100.0
    for _ in range(5):
        c = o + 215.0
        price = bar(o, c + 5, o - 3, c, 60.0)
        o = c

    # 9. hold 15:50-17:00
    for _ in range(70):
        o = price
        c = o + rng.normal(0, 8)
        price = bar(o, max(o, c) + 6, min(o, c) - 6, c, 10.0)

    # 10. crash 17:00-20:00 -> 59000 (through the HTF OB far side)
    o = price
    step = (59000.0 - o) / 180
    for _ in range(180):
        c = o + step
        price = bar(o, max(o, c) + 8, min(o, c) - 8, c, 40.0)
        o = c

    return pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )


def merged_bars() -> pd.DataFrame:
    df1m = build_1m()
    frames = []
    rank = {"5m": 0, "1H": 1, "4H": 2}
    for tf in ("5m", "1H", "4H"):
        f = resample_bars(df1m, tf)
        f["timeframe"] = tf
        f["rank"] = rank[tf]
        frames.append(f)
    allb = pd.concat(frames, ignore_index=True)
    return allb.sort_values(["timestamp", "rank"]).reset_index(drop=True)


async def test_chain_full_lifecycle():
    allb = merged_bars()
    builder = ChainBuilder(SYMBOL)
    # Split BEFORE the left-labeled 16:00 4H bin: in the merged replay that
    # bin is processed at 16:00 but its close already reflects the crash, so
    # the invalidation belongs to the "post" section.
    split = pd.Timestamp("2024-01-05 16:00", tz="UTC")
    pre = allb[allb["timestamp"] < split]
    post = allb[allb["timestamp"] >= split]

    for _, row in pre.iterrows():
        await builder.update(bar_to_dict(row, row["timeframe"], SYMBOL))

    triggered = [
        c for c in builder.get_active_chains()
        if c.state == ChainState.TRIGGERED and c.direction == Direction.BULLISH
    ]
    assert triggered, "no bullish chain reached TRIGGERED"
    chain = max(triggered, key=lambda c: c.created_at)
    assert chain.htf_ob is not None
    assert chain.ltf_shift is not None
    assert chain.ltf_shift.kind == "MSS"
    assert chain.ltf_shift.direction == Direction.BULLISH
    assert chain.ltf_ob is not None
    assert chain.ltf_fvg is not None
    assert chain.ltf_fvg.direction == Direction.BULLISH
    assert len(chain.links) > 0
    assert chain.notes.get("atr_htf")
    assert chain.notes.get("last_price")

    for _, row in post.iterrows():
        await builder.update(bar_to_dict(row, row["timeframe"], SYMBOL))

    final = builder.get_chain(chain.id)
    assert final.state == ChainState.INVALIDATED
    assert final.invalidated_at is not None


async def test_chain_ignores_unknown_timeframe():
    builder = ChainBuilder(SYMBOL)
    bar = bar_to_dict(
        pd.Series(
            {
                "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 7.0,
            }
        ),
        "15m",
        SYMBOL,
    )
    assert await builder.update(bar) == []
    assert builder.get_active_chains() == []
