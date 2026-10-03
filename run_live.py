#!/usr/bin/env python3
"""Run OBTC live: stream bars, build chains, trade A+ setups.

Modes:
  paper (default) — PaperBroker simulates fills; nothing touches an exchange.
  live            — CcxtLiveBroker places real orders. HARD-GATED: refuses to
                    start unless config sets mode: live AND the environment
                    provides OBTC_EXCHANGE_API_KEY / OBTC_EXCHANGE_API_SECRET.

Also serves the dashboard (default http://127.0.0.1:8405) and posts webhook
alerts when configured.

Usage:
    python run_live.py --mode paper
    python run_live.py --mode live   # gated; see above
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve() / "src"))

import uvicorn

from obtc.chain.builder import ChainBuilder
from obtc.chain.models import ChainEvent, ChainState
from obtc.confluence.engine import ConfluenceEngine
from obtc.dashboard.alerts import WebhookAlerter
from obtc.dashboard.app import DashboardState, create_app
from obtc.data.feed import AsyncFeed
from obtc.execution.executor import AsyncExecutor, CcxtLiveBroker, PaperBroker
from obtc.execution.risk import RiskManager
from obtc.strategy import STRATEGIES
from obtc.utils.config import load_config
from obtc.utils.logging import get_logger

log = get_logger("obtc.run_live")


def build_components(cfg):
    builder = ChainBuilder(
        symbol=cfg.data.symbol,
        htf_timeframe="4H",
        mtf_timeframe="1H",
        ltf_timeframe="5m",
    )
    enabled = set(cfg.strategies.enabled)
    strategies = [cls() for cls in STRATEGIES if cls.name in enabled]
    if not strategies:
        raise RuntimeError("no strategies enabled in config")
    confluence = ConfluenceEngine(threshold=cfg.confluence.threshold)
    risk = RiskManager(
        risk_per_trade_pct=cfg.risk.risk_per_trade_pct,
        daily_loss_limit_pct=cfg.risk.daily_loss_limit_pct,
        max_concurrent=cfg.risk.max_concurrent,
        per_trade_risk_cap_pct=cfg.risk.per_trade_risk_cap_pct,
    )
    if cfg.execution.mode == "live":
        # Hard gate lives inside CcxtLiveBroker: raises RuntimeError unless
        # mode == "live" in the passed config AND env keys are present.
        broker = CcxtLiveBroker({"mode": "live",
                                 "exchange": cfg.data.exchange,
                                 "symbol": cfg.data.symbol})
    else:
        broker = PaperBroker(
            starting_equity=cfg.execution.starting_equity,
            fee_bps=cfg.execution.fee_bps,
            slippage_bps=cfg.execution.slippage_bps,
        )
    executor = AsyncExecutor(broker, risk)
    return builder, strategies, confluence, executor, broker


def serve_dashboard(state: DashboardState, host: str, port: int) -> None:
    """Run uvicorn in a daemon thread so the feed loop owns the main thread."""
    app = create_app(state)
    thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": host, "port": port, "log_level": "warning"},
        daemon=True,
    )
    thread.start()
    log.info("dashboard at http://%s:%d", host, port)


async def main_async(args) -> int:
    cfg = load_config(args.config)
    if args.mode:
        if args.mode not in ("paper", "live"):
            raise ValueError("mode must be paper or live")
        cfg.execution.mode = args.mode

    builder, strategies, confluence, executor, broker = build_components(cfg)
    state = DashboardState()
    serve_dashboard(state, cfg.dashboard.host, cfg.dashboard.port)

    alerter = WebhookAlerter(cfg.alerts.webhook_url) if cfg.alerts.webhook_url else None
    fired: set[tuple[str, str]] = set()
    equity_points: list[dict] = []

    def equity_now() -> float:
        eq = getattr(broker, "equity", None)
        if callable(eq):
            return float(eq())
        if eq is not None:
            return float(eq)
        return float(cfg.execution.starting_equity)

    async def on_bar(bar: dict) -> None:
        changed = await builder.update(bar)
        for chain in changed:
            state.events.put_nowait({
                "type": f"CHAIN_{chain.state.value}",
                "chain_id": chain.id,
            })
            if alerter and chain.state in (ChainState.VALID, ChainState.TRIGGERED,
                                           ChainState.INVALIDATED):
                await alerter.notify_chain(ChainEvent(
                    type=f"CHAIN_{chain.state.value}",
                    chain_id=chain.id,
                    payload={"direction": chain.direction.value,
                             "htf_bias": chain.htf_bias.value},
                ))
        for chain in builder.get_active_chains():
            if chain.state not in (ChainState.VALID, ChainState.TRIGGERED):
                continue
            for strat in strategies:
                key = (strat.name, chain.id)
                if key in fired:
                    continue
                sig = strat.check_setup(chain)
                if sig is None:
                    continue
                setup = confluence.score(
                    sig, chain, {"now": bar["timestamp"], "strategy": strat})
                sig.score = setup.score
                if not setup.is_a_plus:
                    continue
                order = await executor.submit(setup, equity_now())
                if order is None:
                    continue
                fired.add(key)
                state.setups.append(setup)
                state.events.put_nowait({
                    "type": "A_PLUS_SETUP",
                    "chain_id": chain.id,
                    "strategy": strat.name,
                    "score": round(setup.score, 1),
                })
                if alerter:
                    await alerter.notify_setup(setup)
                log.info("A+ %s %s score=%.1f entry=%g",
                         strat.name, sig.direction.value, setup.score,
                         sig.entry_price)
        await executor.on_bar(bar)
        state.chains = builder.get_active_chains()
        equity_points.append({"timestamp": bar["timestamp"],
                              "equity": equity_now()})
        state.equity = equity_points

    feed = AsyncFeed(exchange_id=cfg.data.exchange, symbol=cfg.data.symbol)
    log.info("starting %s feed for %s", cfg.execution.mode, cfg.data.symbol)
    try:
        await feed.subscribe(on_bar, bar_interval="1m")
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        await feed.close()
        if alerter:
            await alerter.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="OBTC live runner")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", default=None, choices=["paper", "live"])
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
