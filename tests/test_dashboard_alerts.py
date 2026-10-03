"""Dashboard payload + app tests and webhook alerter tests.

The webhook tests spin up a local aiohttp test server — no external network.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from obtc.chain.models import (
    Bias,
    ChainEvent,
    ChainState,
    Direction,
    FullChain,
    OrderBlock,
    BlockState,
    ScoredSetup,
    Signal,
    new_id,
    utcnow,
)
from obtc.dashboard.alerts import WebhookAlerter
from obtc.dashboard.app import (
    DashboardState,
    chains_payload,
    create_app,
    equity_payload,
    setups_payload,
    to_jsonable,
)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def _make_chain(chain_id: str = "chain-abc") -> FullChain:
    now = utcnow()
    htf_ob = OrderBlock(
        id=new_id("ob"),
        symbol="BTC/USD",
        timeframe="4H",
        direction=Direction.BULLISH,
        high=102.0,
        low=100.0,
        fifty_pct=101.0,
        open=100.2,
        close=101.8,
        created_at=now,
    )
    ltf_ob = OrderBlock(
        id=new_id("ob"),
        symbol="BTC/USD",
        timeframe="5m",
        direction=Direction.BULLISH,
        high=100.8,
        low=100.3,
        fifty_pct=100.55,
        open=100.4,
        close=100.7,
        created_at=now,
        state=BlockState.ACTIVE,
    )
    return FullChain(
        id=chain_id,
        symbol="BTC/USD",
        direction=Direction.BULLISH,
        htf_bias=Bias.LONG,
        htf_timeframe="4H",
        mtf_timeframe="1H",
        ltf_timeframe="5m",
        state=ChainState.VALID,
        htf_ob=htf_ob,
        ltf_ob=ltf_ob,
    )


def _make_setup() -> ScoredSetup:
    signal = Signal(
        strategy_name="s1_liquidity_sweep_reversal",
        symbol="BTC/USD",
        direction=Direction.BULLISH,
        entry_price=100.5,
        stop_price=99.5,
        target_2r=102.5,
        target_3r=103.5,
        opposing_liquidity=105.0,
        chain_id="chain-abc",
        timeframe="5m",
        rationale={"ob_high": 101.0, "ob_low": 100.2},
    )
    return ScoredSetup(signal=signal, score=87.5, is_a_plus=True)


def _make_state() -> DashboardState:
    base = datetime(2026, 1, 5, tzinfo=timezone.utc)
    equity = pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(10)],
            "equity": [10000.0 + i * 25.0 for i in range(10)],
        }
    )
    return DashboardState(chains=[_make_chain()], setups=[_make_setup()], equity=equity)


# ----------------------------------------------------------------------
# payload tests
# ----------------------------------------------------------------------
def test_to_jsonable_handles_models():
    chain = _make_chain()
    data = to_jsonable(chain)
    json.dumps(data)  # serializable
    assert data["direction"] == "BULLISH"
    assert data["htf_ob"]["high"] == 102.0
    assert data["created_at"].endswith("+00:00")


def test_chains_payload_is_json_serializable():
    payload = chains_payload(_make_state())
    text = json.dumps(payload)
    assert "chain-abc" in text
    assert payload["chains"][0]["state"] == "VALID"


def test_setups_payload_is_json_serializable():
    payload = setups_payload(_make_state())
    text = json.dumps(payload)
    assert "s1_liquidity_sweep_reversal" in text
    assert payload["setups"][0]["is_a_plus"] is True
    assert payload["setups"][0]["score"] == 87.5


def test_equity_payload_is_json_serializable():
    payload = equity_payload(_make_state())
    json.dumps(payload)
    assert len(payload["equity"]) == 10
    assert payload["equity"][0]["equity"] == 10000.0
    assert "timestamp" in payload["equity"][0]


def test_equity_payload_empty_state():
    payload = equity_payload(DashboardState())
    assert payload == {"equity": []}
    json.dumps(payload)


def test_snapshot_combines_payloads():
    snapshot = _make_state().snapshot()
    json.dumps(snapshot)
    assert set(snapshot) == {"chains", "setups", "equity"}


# ----------------------------------------------------------------------
# app tests
# ----------------------------------------------------------------------
def test_create_app_routes():
    app = create_app(_make_state())
    paths = {route.path for route in app.routes}
    for expected in ("/chains", "/setups", "/equity", "/events", "/"):
        assert expected in paths


def test_static_index_served():
    app = create_app(_make_state())
    index = next(
        route for route in app.routes
        if getattr(route, "path", "") == "/static"
    )
    assert index is not None


@pytest.mark.asyncio
async def test_publish_and_events_queue():
    state = DashboardState()
    await state.publish({"type": "CHAIN_VALID", "chain_id": "c1"})
    event = await state.events.get()
    assert event["chain_id"] == "c1"


# ----------------------------------------------------------------------
# webhook alerter tests (local test server only)
# ----------------------------------------------------------------------
async def _start_server(handler):
    app = web.Application()
    app.router.add_post("/hook", handler)
    server = TestServer(app)
    await server.start_server()
    return server


@pytest.mark.asyncio
async def test_notify_success():
    received = []

    async def handler(request):
        received.append(await request.json())
        return web.json_response({"ok": True})

    server = await _start_server(handler)
    try:
        alerter = WebhookAlerter(str(server.make_url("/hook")))
        assert await alerter.notify("PING", {"a": 1}) is True
        await alerter.close()
    finally:
        await server.close()
    assert len(received) == 1
    body = received[0]
    assert body["type"] == "PING"
    assert "at" in body
    assert body["payload"] == {"a": 1}


@pytest.mark.asyncio
async def test_notify_retries_then_succeeds():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return web.Response(status=500)
        return web.json_response({"ok": True})

    server = await _start_server(handler)
    try:
        alerter = WebhookAlerter(str(server.make_url("/hook")),
                                 backoff_s=0.01, max_retries=5)
        assert await alerter.notify("PING", {}) is True
        await alerter.close()
    finally:
        await server.close()
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_notify_gives_up_without_raising():
    async def handler(request):
        return web.Response(status=500)

    server = await _start_server(handler)
    try:
        alerter = WebhookAlerter(str(server.make_url("/hook")),
                                 backoff_s=0.01, max_retries=2)
        assert await alerter.notify("PING", {}) is False  # no raise
        await alerter.close()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_notify_connection_error_returns_false():
    # Nothing listens on this port: must return False, never raise.
    alerter = WebhookAlerter("http://127.0.0.1:9/hook",
                             timeout_s=1, backoff_s=0.01, max_retries=1)
    assert await alerter.notify("PING", {}) is False
    await alerter.close()


@pytest.mark.asyncio
async def test_notify_chain_and_setup():
    received = []

    async def handler(request):
        received.append(await request.json())
        return web.json_response({"ok": True})

    server = await _start_server(handler)
    try:
        alerter = WebhookAlerter(str(server.make_url("/hook")))
        event = ChainEvent(type="CHAIN_VALID", chain_id="chain-abc",
                           payload={"direction": "BULLISH"})
        assert await alerter.notify_chain(event) is True
        assert await alerter.notify_setup(_make_setup()) is True
        await alerter.close()
    finally:
        await server.close()
    assert [b["type"] for b in received] == ["CHAIN_VALID", "A_PLUS_SETUP"]
    setup_body = received[1]["payload"]
    assert setup_body["strategy"] == "s1_liquidity_sweep_reversal"
    assert setup_body["score"] == 87.5
    assert setup_body["is_a_plus"] is True
