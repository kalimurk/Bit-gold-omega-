"""Disk cache for OHLCV frames with TTL and async read-through."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

import pandas as pd

try:
    import pyarrow  # noqa: F401

    _PARQUET_OK = True
except ImportError:  # pragma: no cover - depends on installed extras
    _PARQUET_OK = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BarCache:
    """File-backed bar cache.

    Key ``f"{exchange}_{symbol}_{timeframe}"`` sanitized to alphanumerics.
    Uses parquet when pyarrow is importable, otherwise CSV, plus a sidecar
    ``.meta.json`` storing the save time for TTL checks.
    """

    def __init__(self, cache_dir, default_ttl_s: float = 3600) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_s = default_ttl_s
        self._use_parquet = _PARQUET_OK

    @staticmethod
    def _key(exchange: str, symbol: str, timeframe: str) -> str:
        raw = f"{exchange}_{symbol}_{timeframe}"
        return re.sub(r"[^A-Za-z0-9]", "_", raw)

    def _data_path(self, key: str) -> Path:
        ext = ".parquet" if self._use_parquet else ".csv"
        return self.cache_dir / f"{key}{ext}"

    def _meta_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.meta.json"

    def put(self, df: pd.DataFrame, exchange: str, symbol: str, timeframe: str) -> Path:
        key = self._key(exchange, symbol, timeframe)
        data_path = self._data_path(key)
        out = df.copy()
        if self._use_parquet:
            out.to_parquet(data_path, index=False)
        else:
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            out.to_csv(data_path, index=False)
        meta = {
            "saved_at": _utcnow().isoformat(),
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "rows": int(len(out)),
            "format": "parquet" if self._use_parquet else "csv",
        }
        self._meta_path(key).write_text(json.dumps(meta), encoding="utf-8")
        return data_path

    def get(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        ttl_s: float | None = None,
    ) -> pd.DataFrame | None:
        """Return cached frame, or None when missing or older than TTL."""
        ttl = self.default_ttl_s if ttl_s is None else ttl_s
        key = self._key(exchange, symbol, timeframe)
        data_path = self._data_path(key)
        meta_path = self._meta_path(key)
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            saved_at = datetime.fromisoformat(meta["saved_at"])
        except (KeyError, ValueError, OSError):
            return None
        age = (_utcnow() - saved_at).total_seconds()
        if age > ttl:
            return None
        if self._use_parquet:
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.reset_index(drop=True)

    async def get_or_fetch(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        since,
        ttl_s: float | None,
        fetch_fn: Callable[[], Awaitable[pd.DataFrame]],
    ) -> pd.DataFrame:
        """Read-through cache: return fresh-enough cached data or fetch+store."""
        cached = self.get(exchange, symbol, timeframe, ttl_s=ttl_s)
        if cached is not None:
            return cached
        df = await fetch_fn()
        await asyncio.to_thread(self.put, df, exchange, symbol, timeframe)
        return df
