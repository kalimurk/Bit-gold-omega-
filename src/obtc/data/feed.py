"""Async market-data feed: REST history + websocket live bars."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import pandas as pd
import websockets

from obtc.data.normalize import normalize_bars

_INTERVAL_S = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1H": 3600,
    "4H": 14400,
    "1D": 86400,
}


class AsyncFeed:
    """ccxt-based history fetcher and websocket live-bar builder."""

    def __init__(self, exchange_id: str = "coinbase", symbol: str = "BTC/USD") -> None:
        self.exchange_id = exchange_id
        self.symbol = symbol
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ history
    async def fetch_history(self, timeframe: str = "1m", days: int = 7) -> pd.DataFrame:
        """Paginate ``fetch_ohlcv`` until caught up; returns normalized frame."""
        exchange_cls = getattr(ccxt, self.exchange_id)
        exchange = exchange_cls({"enableRateLimit": True})
        try:
            if not exchange.has.get("fetchOHLCV"):
                raise RuntimeError(f"{self.exchange_id} does not support fetchOHLCV")
            limit = 1000
            since = int(
                (
                    datetime.now(timezone.utc)
                    - pd.Timedelta(days=days)
                ).timestamp()
                * 1000
            )
            rows: list[list] = []
            while True:
                batch = await exchange.fetch_ohlcv(
                    self.symbol, timeframe=timeframe, since=since, limit=limit
                )
                if not batch:
                    break
                rows.extend(batch)
                since = batch[-1][0] + 1
                if len(batch) < limit:
                    break
                await asyncio.sleep(0.25)
            df = pd.DataFrame(
                rows,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return normalize_bars(df)
        finally:
            await exchange.close()

    # ------------------------------------------------------------------- live
    def _ws_url(self) -> str:
        if self.exchange_id == "coinbase":
            return "wss://ws-feed.exchange.coinbase.com"
        if self.exchange_id == "binance":
            stream = self.symbol.replace("/", "").lower() + "@trade"
            return f"wss://stream.binance.com:9443/ws/{stream}"
        raise ValueError(f"no websocket mapping for exchange {self.exchange_id!r}")

    def _subscribe_message(self) -> str | None:
        if self.exchange_id == "coinbase":
            product_id = self.symbol.replace("/", "-")
            return json.dumps(
                {
                    "type": "subscribe",
                    "product_ids": [product_id],
                    "channels": ["ticker", "matches"],
                }
            )
        return None  # binance encodes the stream in the URL

    @staticmethod
    def _parse_trade(exchange_id: str, msg: dict):
        """Return (price, size, datetime) or None for non-trade messages."""
        try:
            if exchange_id == "coinbase":
                if msg.get("type") != "matches":
                    return None
                ts = datetime.fromisoformat(msg["time"].replace("Z", "+00:00"))
                return float(msg["price"]), float(msg["size"]), ts
            if exchange_id == "binance":
                if "p" not in msg or "T" not in msg:
                    return None
                ts = datetime.fromtimestamp(msg["T"] / 1000, tz=timezone.utc)
                return float(msg["p"]), float(msg["q"]), ts
        except (KeyError, ValueError, TypeError):
            return None
        return None

    async def subscribe(self, on_bar, bar_interval: str = "1m") -> None:
        """Stream trades, aggregate into bars, emit completed bars via on_bar.

        Reconnects with exponential backoff (1s doubling, 60s cap) until
        :meth:`close` is called.
        """
        if bar_interval not in _INTERVAL_S:
            raise ValueError(f"unsupported bar_interval {bar_interval!r}")
        interval_s = _INTERVAL_S[bar_interval]
        url = self._ws_url()
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    sub = self._subscribe_message()
                    if sub:
                        await ws.send(sub)
                    backoff = 1.0
                    await self._consume(ws, on_bar, interval_s, bar_interval)
            except asyncio.CancelledError:
                raise
            except Exception:
                if self._stop.is_set():
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)

    async def _consume(self, ws, on_bar, interval_s: int, bar_interval: str) -> None:
        cur: dict | None = None
        cur_bucket: int | None = None
        while not self._stop.is_set():
            raw = await asyncio.wait_for(ws.recv(), timeout=60)
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue
            trade = self._parse_trade(self.exchange_id, msg)
            if trade is None:
                continue
            price, size, ts = trade
            bucket = int(ts.timestamp() // interval_s) * interval_s
            if cur is None:
                cur_bucket = bucket
                cur = self._new_bar(bucket, price, size)
            elif bucket != cur_bucket:
                await on_bar(self._finalize_bar(cur, cur_bucket, bar_interval))
                cur_bucket = bucket
                cur = self._new_bar(bucket, price, size)
            else:
                cur["high"] = max(cur["high"], price)
                cur["low"] = min(cur["low"], price)
                cur["close"] = price
                cur["volume"] += size

    def _new_bar(self, bucket: int, price: float, size: float) -> dict:
        return {"open": price, "high": price, "low": price, "close": price,
                "volume": size}

    def _finalize_bar(self, cur: dict, bucket: int, bar_interval: str) -> dict:
        return {
            "timestamp": datetime.fromtimestamp(bucket, tz=timezone.utc),
            "open": cur["open"],
            "high": cur["high"],
            "low": cur["low"],
            "close": cur["close"],
            "volume": cur["volume"],
            "timeframe": bar_interval,
            "symbol": self.symbol,
        }

    async def close(self) -> None:
        """Stop the subscribe loop."""
        self._stop.set()
