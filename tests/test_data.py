"""Tests for data plumbing: resample, normalize, cache."""
import pandas as pd

from obtc.data.cache import BarCache
from obtc.data.normalize import bar_to_dict, normalize_bars
from obtc.data.resample import resample_bars


def _five_minute_bars():
    ts = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
    opens = [100.0, 101.0, 102.0, 103.0, 104.0]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": [o + 2 for o in opens],
            "low": [o - 1 for o in opens],
            "close": [o + 1 for o in opens],
            "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


def test_resample_1m_to_5m_hand_computed():
    out = resample_bars(_five_minute_bars(), "5m")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["timestamp"] == pd.Timestamp("2024-01-01", tz="UTC")
    assert row["open"] == 100.0
    assert row["high"] == 106.0  # max of open+2
    assert row["low"] == 99.0  # min of open-1
    assert row["close"] == 105.0
    assert row["volume"] == 150.0


def test_resample_alias_and_unknown():
    out = resample_bars(_five_minute_bars(), "5min")
    assert len(out) == 1
    try:
        resample_bars(_five_minute_bars(), "2W")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown timeframe")


def test_resample_keeps_tz_and_sorts():
    df = _five_minute_bars().iloc[::-1].reset_index(drop=True)
    out = resample_bars(df, "1m")
    assert str(out["timestamp"].dt.tz) == "UTC"
    assert out["timestamp"].is_monotonic_increasing


def test_normalize_drops_bad_rows_dedupes_sorts():
    ts = pd.date_range("2024-01-01", periods=6, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": [ts[2], ts[0], ts[1], ts[3], ts[4], ts[4]],
            "open": [100.0, 100.0, float("nan"), 0.0, 100.0, 101.0],
            "high": [102.0, 102.0, 102.0, 102.0, 99.0, 103.0],
            "low": [99.0, 99.0, 99.0, 99.0, 98.0, 100.0],
            "close": [101.0, 101.0, 101.0, 101.0, 100.5, 102.0],
            "volume": [10.0, "10", 10.0, 10.0, 10.0, 11.0],
        }
    )
    out = normalize_bars(df)
    # kept: ts[2] (valid), ts[0] (valid, volume coerced from str),
    # ts[4] deduped keep-last (open 101). Dropped: NaN row, zero-open row,
    # high<low row.
    assert list(out["timestamp"]) == [ts[0], ts[2], ts[4]]
    assert out.iloc[2]["open"] == 101.0
    assert out["volume"].dtype == float


def test_bar_to_dict_contract():
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
            "open": 1,
            "high": 2,
            "low": 0.5,
            "close": 1.5,
            "volume": 7,
        }
    )
    bar = bar_to_dict(row, "5m", "BTC/USD")
    assert bar == {
        "timestamp": pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(),
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "volume": 7.0,
        "timeframe": "5m",
        "symbol": "BTC/USD",
    }
    assert bar["timestamp"].tzinfo is not None


def test_cache_put_get_roundtrip(tmp_path):
    cache = BarCache(tmp_path)
    df = _five_minute_bars()
    cache.put(df, "coinbase", "BTC/USD", "1m")
    got = cache.get("coinbase", "BTC/USD", "1m")
    assert got is not None
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), df.reset_index(drop=True)
    )


def test_cache_miss_and_ttl_expiry(tmp_path):
    cache = BarCache(tmp_path, default_ttl_s=3600)
    assert cache.get("coinbase", "BTC/USD", "1m") is None
    cache.put(_five_minute_bars(), "coinbase", "BTC/USD", "1m")
    assert cache.get("coinbase", "BTC/USD", "1m", ttl_s=0) is None
    assert cache.get("coinbase", "BTC/USD", "1m", ttl_s=3600) is not None


async def test_cache_get_or_fetch_read_through(tmp_path):
    cache = BarCache(tmp_path)
    df = _five_minute_bars()
    calls = 0

    async def fetch_fn():
        nonlocal calls
        calls += 1
        return df

    first = await cache.get_or_fetch(
        "coinbase", "BTC/USD", "1m", None, 3600, fetch_fn
    )
    second = await cache.get_or_fetch(
        "coinbase", "BTC/USD", "1m", None, 3600, fetch_fn
    )
    assert calls == 1
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), df.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        second.reset_index(drop=True), df.reset_index(drop=True)
    )
