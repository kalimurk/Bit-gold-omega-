"""Webhook alerting with retries and exponential backoff.

Failures are logged and reported as ``False`` — alerting never raises into
the trading loop.
"""
from __future__ import annotations

import asyncio

import aiohttp

from obtc.chain.models import ChainEvent, utcnow
from obtc.utils.logging import get_logger


class WebhookAlerter:
    def __init__(
        self,
        url: str,
        timeout_s: float = 10.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        logger=None,
    ) -> None:
        self.url = url
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.logger = logger or get_logger("obtc.alerts")
        self._session: aiohttp.ClientSession | None = None

    async def _session_open(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def notify(self, event_type: str, payload: dict) -> bool:
        """POST one alert; retry with exponential backoff.

        Returns True on the first 2xx response, False after retries are
        exhausted. Never raises.
        """
        body = {
            "type": event_type,
            "at": utcnow().isoformat(),
            "payload": payload,
        }
        attempt = 0
        while True:
            try:
                session = await self._session_open()
                async with session.post(self.url, json=body) as response:
                    if 200 <= response.status < 300:
                        return True
                    self.logger.warning(
                        "webhook %s -> HTTP %s (attempt %d)",
                        event_type, response.status, attempt + 1,
                    )
            except Exception as exc:  # network errors, timeouts, DNS, ...
                self.logger.warning(
                    "webhook %s failed on attempt %d: %s",
                    event_type, attempt + 1, exc,
                )
            attempt += 1
            if attempt > self.max_retries:
                self.logger.error(
                    "webhook %s giving up after %d attempts", event_type, attempt
                )
                return False
            await asyncio.sleep(self.backoff_s * (2 ** (attempt - 1)))

    async def notify_chain(self, event: ChainEvent) -> bool:
        return await self.notify(event.type, event.to_dict())

    async def notify_setup(self, setup) -> bool:
        signal = setup.signal
        direction = getattr(signal.direction, "value", signal.direction)
        payload = {
            "strategy": signal.strategy_name,
            "symbol": signal.symbol,
            "direction": direction,
            "chain_id": signal.chain_id,
            "timeframe": signal.timeframe,
            "entry": signal.entry_price,
            "stop": signal.stop_price,
            "target_2r": signal.target_2r,
            "target_3r": signal.target_3r,
            "score": setup.score,
            "is_a_plus": setup.is_a_plus,
            "rationale": signal.rationale,
        }
        return await self.notify("A_PLUS_SETUP", payload)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
