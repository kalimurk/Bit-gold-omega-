"""Bar normalization and the shared bar-dict contract.

Every module in the pipeline speaks the same bar dict::

    {"timestamp": datetime tz-aware UTC, "open": float, "high": float,
     "low": float, "close": float, "volume": float,
     "timeframe": str like "1m", "symbol": str like "BTC/USD"}
"""
from __future__ import annotations

import pandas as pd

_OHLCV_COLS = ("open", "high", "low", "close", "volume")


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce, clean, sort and dedupe a raw OHLCV frame.

    - timestamp coerced with ``pd.to_datetime(..., utc=True)`` (tz-aware UTC)
    - o/h/l/c/volume coerced to float
    - drops rows with NaN, any of o/h/l/c <= 0, or high < low
    - sorts by timestamp, dedupes timestamp keeping the last row
    - returns a fresh frame with a reset RangeIndex
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in _OHLCV_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", *_OHLCV_COLS])
    for col in ("open", "high", "low", "close"):
        out = out[out[col] > 0]
    out = out[out["high"] >= out["low"]]
    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset="timestamp", keep="last")
    return out.reset_index(drop=True)


def bar_to_dict(row, timeframe: str, symbol: str) -> dict:
    """Build a bar-dict-contract dict from a DataFrame row/Series."""
    ts = pd.to_datetime(row["timestamp"], utc=True)
    return {
        "timestamp": ts.to_pydatetime(),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
        "timeframe": timeframe,
        "symbol": symbol,
    }
