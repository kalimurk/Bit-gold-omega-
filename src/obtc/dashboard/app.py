"""Dashboard HTTP API.

Serves chain state, A+ setups and the equity curve as JSON, plus a
server-sent-events stream of chain events and a static canvas frontend.
"""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

EQUITY_POINT_LIMIT = 500


def to_jsonable(obj):
    """Recursively convert dataclasses/Enums/datetimes to JSON-safe values."""
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return str(obj)


class DashboardState:
    """Mutable snapshot of what the dashboard serves."""

    def __init__(self, chains=None, setups=None, equity=None, max_events: int = 1000):
        self.chains: list = list(chains or [])
        self.setups: list = list(setups or [])
        self.equity = equity
        self.events: asyncio.Queue = asyncio.Queue(maxsize=max_events)

    def get_chains(self) -> list:
        return list(self.chains)

    def get_setups(self) -> list:
        return list(self.setups)

    def get_equity(self):
        return self.equity

    async def publish(self, event: dict) -> None:
        try:
            self.events.put_nowait(event)
        except asyncio.QueueFull:
            # Drop the oldest event to make room; the stream must not block.
            try:
                self.events.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.events.put_nowait(event)

    def snapshot(self) -> dict:
        return {
            "chains": chains_payload(self),
            "setups": setups_payload(self),
            "equity": equity_payload(self),
        }


def chains_payload(state: DashboardState) -> dict:
    return {"chains": [to_jsonable(chain) for chain in state.get_chains()]}


def setups_payload(state: DashboardState) -> dict:
    return {"setups": [to_jsonable(setup) for setup in state.get_setups()]}


def equity_payload(state: DashboardState) -> dict:
    equity = state.get_equity()
    if equity is None or getattr(equity, "empty", True):
        return {"equity": []}
    frame = equity.tail(EQUITY_POINT_LIMIT)
    points = []
    for _, row in frame.iterrows():
        ts = row["timestamp"]
        if isinstance(ts, pd.Timestamp):
            ts = ts.isoformat()
        elif isinstance(ts, (datetime, date)):
            ts = ts.isoformat()
        else:
            ts = str(ts)
        value = row["equity"]
        points.append(
            {
                "timestamp": ts,
                "equity": None if pd.isna(value) else float(value),
            }
        )
    return {"equity": points}


def create_app(state: DashboardState) -> FastAPI:
    app = FastAPI(title="OBTC Dashboard")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/chains")
    def get_chains():
        return chains_payload(state)

    @app.get("/setups")
    def get_setups():
        return setups_payload(state)

    @app.get("/equity")
    def get_equity():
        return equity_payload(state)

    @app.get("/events")
    async def get_events():
        async def event_stream():
            while True:
                event = await state.events.get()
                yield f"data: {json.dumps(to_jsonable(event))}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/")
    def index():
        return FileResponse(str(static_dir / "index.html"))

    return app
