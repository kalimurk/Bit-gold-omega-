"""Timeframe resampling for tz-aware UTC OHLCV frames."""
from __future__ import annotations

import pandas as pd

_RULES = {
    "1m": "1min",
    "1min": "1min",
    "5m": "5min",
    "5min": "5min",
    "15m": "15min",
    "15min": "15min",
    "1H": "1h",
    "1h": "1h",
    "4H": "4h",
    "4h": "4h",
    "1D": "1D",
    "1d": "1D",
}


def resample_bars(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate 1m (or finer) bars into ``timeframe``.

    OHLC = first/max/min/last, volume = sum. Result is sorted by timestamp,
    tz-aware UTC, with a reset RangeIndex. Raises ValueError on unknown
    timeframe.
    """
    if timeframe not in _RULES:
        raise ValueError(
            f"unknown timeframe {timeframe!r}; supported: {sorted(_RULES)}"
        )
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
    agg = (
        work.set_index("timestamp")
        .resample(_RULES[timeframe])
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    agg = agg.reset_index().sort_values("timestamp").reset_index(drop=True)
    return agg[["timestamp", "open", "high", "low", "close", "volume"]]
