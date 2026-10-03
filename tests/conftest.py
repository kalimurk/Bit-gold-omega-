"""Shared fixtures: deterministic synthetic OHLCV, no network."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd


def make_bars(n, start="2024-01-01", seed=7, base=60000.0, drift=0.0, vol=50.0):
    """Deterministic 1m random-walk bars, tz-aware UTC."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, periods=n, freq="1min", tz="UTC")
    rows = []
    price = base
    for i in range(n):
        o = price
        c = o + drift + rng.normal(0, vol)
        h = max(o, c) + abs(rng.normal(0, vol * 0.25))
        l = min(o, c) - abs(rng.normal(0, vol * 0.25))
        v = 10.0 + abs(rng.normal(0, 3))
        rows.append((o, h, l, c, v))
        price = c
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [r[4] for r in rows],
        }
    )


def make_impulse(direction="bull", n_before=25, n_after=12, seed=11, base=60000.0):
    """Chop + crafted opposite candle + displacement impulse + follow-ups.

    Bullish: bearish opposite candle (wick_pct ~0.29), then a +600 body
    impulse on 1000 volume. Bearish mirrors it.
    """
    before = make_bars(n_before, seed=seed, base=base)
    prev_close = float(before["close"].iloc[-1])
    last_ts = before["timestamp"].iloc[-1]
    opp_ts = last_ts + pd.Timedelta(minutes=1)
    imp_ts = last_ts + pd.Timedelta(minutes=2)

    if direction == "bull":
        opp = (prev_close, prev_close + 10, prev_close - 60, prev_close - 50, 12.0)
        imp = (
            prev_close - 50,
            prev_close + 560,
            prev_close - 60,
            prev_close + 550,
            1000.0,
        )
        after_base, after_drift = prev_close + 550, 2.0
    else:
        opp = (prev_close, prev_close + 60, prev_close - 10, prev_close + 50, 12.0)
        imp = (
            prev_close + 50,
            prev_close + 60,
            prev_close - 560,
            prev_close - 550,
            1000.0,
        )
        after_base, after_drift = prev_close - 550, -2.0

    extra = pd.DataFrame(
        {
            "timestamp": [opp_ts, imp_ts],
            "open": [opp[0], imp[0]],
            "high": [opp[1], imp[1]],
            "low": [opp[2], imp[2]],
            "close": [opp[3], imp[3]],
            "volume": [opp[4], imp[4]],
        }
    )
    after = make_bars(
        n_after,
        start=imp_ts + pd.Timedelta(minutes=1),
        seed=seed + 1,
        base=after_base,
        drift=after_drift,
        vol=20.0,
    )
    return pd.concat([before, extra, after], ignore_index=True)


def feed_bars(detector, df, timeframe="1m", symbol="BTC/USD"):
    """Push DataFrame rows through a detector; return all events."""
    from obtc.data.normalize import bar_to_dict

    events = []
    for _, row in df.iterrows():
        events.extend(detector.update(bar_to_dict(row, timeframe, symbol)))
    return events
