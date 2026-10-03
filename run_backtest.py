#!/usr/bin/env python3
"""Run a deterministic OBTC backtest over historical OHLCV.

Fetches 1m history via the async ccxt feed (cached on disk), resamples to
the configured timeframes, replays every bar through the ChainBuilder, all 7
strategies and the confluence engine, and executes A+ setups on the paper
broker. Writes equity curve, trade log and per-trade screenshots to the
output directory and prints a metrics summary.

Usage:
    python run_backtest.py --days 30 --symbol BTC/USD --output outputs
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve() / "src"))

from obtc.chain.builder import ChainBuilder
from obtc.confluence.engine import ConfluenceEngine
from obtc.data.cache import BarCache
from obtc.data.feed import AsyncFeed
from obtc.data.resample import resample_bars
from obtc.execution.executor import AsyncExecutor, PaperBroker
from obtc.execution.risk import RiskManager
from obtc.backtest.replay import ReplayEngine
from obtc.strategy import STRATEGIES
from obtc.utils.config import load_config
from obtc.utils.logging import get_logger

log = get_logger("obtc.run_backtest")


def build_components(cfg):
    """Instantiate builder, strategies, confluence, executor from config."""
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
    broker = PaperBroker(
        starting_equity=cfg.execution.starting_equity,
        fee_bps=cfg.execution.fee_bps,
        slippage_bps=cfg.execution.slippage_bps,
    )
    executor = AsyncExecutor(broker, risk)
    return builder, strategies, confluence, executor


async def load_bars(cfg, days: int) -> dict:
    """Fetch (cached) 1m history and resample to all configured timeframes."""
    cache = BarCache(cfg.cache.dir, default_ttl_s=cfg.cache.ttl_s)
    feed = AsyncFeed(exchange_id=cfg.data.exchange, symbol=cfg.data.symbol)

    async def fetch():
        return await feed.fetch_history(timeframe="1m", days=days)

    df_1m = await cache.get_or_fetch(
        cfg.data.exchange, cfg.data.symbol, "1m", cfg.cache.ttl_s, fetch
    )
    if df_1m is None or df_1m.empty:
        raise RuntimeError("no 1m history available")
    bars = {"1m": df_1m}
    for tf in cfg.data.timeframes:
        if tf == "1m":
            continue
        bars[tf] = resample_bars(df_1m, tf)
    return bars


async def main_async(args) -> int:
    cfg = load_config(args.config)
    if args.symbol:
        cfg.data.symbol = args.symbol
    builder, strategies, confluence, executor = build_components(cfg)
    bars = await load_bars(cfg, args.days)
    engine = ReplayEngine(
        builder, strategies, confluence, executor, output_dir=args.output
    )
    result = await engine.run(bars, symbol=cfg.data.symbol)
    m = result.metrics
    print(f"trades={m['num_trades']} win_rate={m['win_rate']:.1%} "
          f"expectancy={m['expectancy']:.2f} profit_factor={m['profit_factor']:.2f} "
          f"max_dd={m['max_drawdown_pct']:.1%} return={m['total_return_pct']:.1%}")
    print(f"output: {result.output_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="OBTC backtest runner")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--output", default="outputs")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
