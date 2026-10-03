"""Backtest pipeline tests.

Synthetic fixtures only — no network. Sibling modules (chain builder,
strategies, confluence, executor) are accessed exclusively through
``pytest.importorskip``: if a sibling is not built yet the wired end-to-end
test skips cleanly and the unit tests below still verify metrics, the trade
log and screenshots.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from obtc.backtest.metrics import compute_metrics, save_equity_csv, save_equity_plot
from obtc.backtest.replay import BacktestResult, ReplayEngine
from obtc.backtest.trade_log import TradeLogger
from obtc.chain.models import (
    Direction,
    Fill,
    Order,
    OrderStatus,
    Signal,
    utcnow,
)

STRATEGY_MODULES = [
    "s1_liquidity_sweep_reversal",
    "s2_continuation",
    "s3_breaker_reclaim",
    "s4_mitigation_flip",
    "s5_turtle_soup",
    "s6_silver_bullet",
    "s7_orderflow_stacking",
]

EXPECTED_METRIC_KEYS = {
    "num_trades",
    "wins",
    "losses",
    "win_rate",
    "total_pnl",
    "expectancy",
    "avg_r",
    "profit_factor",
    "max_drawdown",
    "max_drawdown_pct",
    "total_return_pct",
    "avg_win",
    "avg_loss",
}


# ----------------------------------------------------------------------
# synthetic fixtures
# ----------------------------------------------------------------------
def make_bars(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Seeded random walk with engineered impulse legs so structure can form."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    rets = rng.normal(0.0, 0.0012, n)
    for leg_start, leg_end, drift in (
        (300, 340, 0.004),
        (900, 930, -0.0045),
        (1500, 1545, 0.005),
    ):
        rets[leg_start:leg_end] += drift
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wiggle = np.abs(rng.normal(0.0, 0.0008, n)) * close
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    volume = rng.integers(50, 500, n).astype(float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Tiny local resampler for the test (independent of the data package)."""
    grouped = (
        df.set_index("timestamp")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna()
    )
    return grouped.reset_index()


def _fake_signal(**overrides) -> Signal:
    params = {
        "strategy_name": "s1_liquidity_sweep_reversal",
        "symbol": "BTC/USD",
        "direction": Direction.BULLISH,
        "entry_price": 100.0,
        "stop_price": 99.0,
        "target_2r": 102.0,
        "target_3r": 103.0,
        "opposing_liquidity": 104.0,
        "chain_id": "chain-1",
        "timeframe": "5m",
        "rationale": {"ob_high": 101.0, "ob_low": 100.2, "note": "sweep + reclaim"},
    }
    params.update(overrides)
    return Signal(**params)


def _fake_order(order_id: str = "ord-1", pnl: float = 150.0,
                opened_at=None, closed_at=None) -> Order:
    opened_at = opened_at or datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    closed_at = closed_at or datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    return Order(
        id=order_id,
        signal=_fake_signal(),
        size=1.0,
        status=OrderStatus.CLOSED,
        entry_fill=100.0,
        stop_price=99.0,
        opened_at=opened_at,
        closed_at=closed_at,
        fills=[
            Fill(price=100.0, size=1.0, at=opened_at, kind="ENTRY"),
            Fill(price=102.0, size=1.0, at=closed_at, kind="TARGET"),
        ],
        realized_pnl=pnl,
        fees_paid=1.0,
        notes={"risk_amt": 100.0},
    )


def _equity_curve(n: int = 10, start: float = 10000.0) -> pd.DataFrame:
    base = datetime(2026, 1, 5, tzinfo=timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(n)],
            "equity": [start + i * 10.0 for i in range(n)],
        }
    )


# ----------------------------------------------------------------------
# unit tests: metrics
# ----------------------------------------------------------------------
def test_metrics_zero_trades():
    metrics = compute_metrics([], _equity_curve(), 10000.0)
    assert EXPECTED_METRIC_KEYS <= set(metrics)
    assert metrics["num_trades"] == 0
    assert metrics["total_pnl"] == 0.0
    assert metrics["win_rate"] == 0.0
    assert metrics["profit_factor"] == 0.0
    assert metrics["max_drawdown"] == 0.0


def test_metrics_with_trades():
    orders = [_fake_order("ord-1", pnl=150.0), _fake_order("ord-2", pnl=-60.0)]
    metrics = compute_metrics(orders, _equity_curve(), 10000.0)
    assert metrics["num_trades"] == 2
    assert metrics["wins"] == 1
    assert metrics["losses"] == 1
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["total_pnl"] == pytest.approx(90.0)
    assert metrics["expectancy"] == pytest.approx(45.0)
    assert metrics["avg_win"] == pytest.approx(150.0)
    assert metrics["avg_loss"] == pytest.approx(-60.0)
    assert metrics["profit_factor"] == pytest.approx(150.0 / 60.0)
    # risk_amt=100 per order -> R multiples 1.5 and -0.6
    assert metrics["avg_r"] == pytest.approx(0.45)


def test_metrics_drawdown_and_return():
    base = datetime(2026, 1, 5, tzinfo=timezone.utc)
    equity = pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(4)],
            "equity": [10000.0, 11000.0, 10500.0, 12000.0],
        }
    )
    metrics = compute_metrics([_fake_order()], equity, 10000.0)
    assert metrics["max_drawdown"] == pytest.approx(500.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(500.0 / 11000.0 * 100.0)
    assert metrics["total_return_pct"] == pytest.approx(20.0)


def test_save_equity_artifacts(tmp_path):
    equity = _equity_curve()
    csv_path = tmp_path / "equity.csv"
    png_path = tmp_path / "equity.png"
    save_equity_csv(equity, str(csv_path))
    save_equity_plot(equity, str(png_path))
    assert csv_path.exists()
    assert png_path.exists() and png_path.stat().st_size > 0
    reloaded = pd.read_csv(csv_path)
    assert list(reloaded.columns) == ["timestamp", "equity"]
    assert len(reloaded) == len(equity)


# ----------------------------------------------------------------------
# unit tests: trade log + screenshots
# ----------------------------------------------------------------------
def test_trade_logger_jsonl(tmp_path):
    logger = TradeLogger(str(tmp_path))
    order = _fake_order()
    logger.log(order)
    logger.finalize()
    logger.close()
    lines = (tmp_path / "trades.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])  # valid JSONL
    assert record["order_id"] == "ord-1"
    assert record["strategy"] == "s1_liquidity_sweep_reversal"
    assert record["direction"] == "BULLISH"
    assert record["entry"] == 100.0
    assert record["realized_pnl"] == 150.0
    assert "ob_high" in record["rationale_summary"]


def test_screenshot_with_ob_zone(tmp_path):
    bars = make_bars(300, seed=11)
    order = _fake_order(
        opened_at=bars["timestamp"].iloc[100].to_pydatetime(),
        closed_at=bars["timestamp"].iloc[200].to_pydatetime(),
    )
    logger = TradeLogger(str(tmp_path))
    shot = tmp_path / "shot.png"
    logger.screenshot(order, bars, str(shot))
    logger.close()
    assert shot.exists() and shot.stat().st_size > 0


def test_screenshot_without_ob_keys_does_not_crash(tmp_path):
    bars = make_bars(300, seed=12)
    signal = _fake_signal(rationale={"note": "no zone keys here"})
    order = Order(
        id="ord-x",
        signal=signal,
        size=1.0,
        status=OrderStatus.OPEN,
        opened_at=bars["timestamp"].iloc[150].to_pydatetime(),
        realized_pnl=0.0,
    )
    logger = TradeLogger(str(tmp_path))
    shot = tmp_path / "shot2.png"
    logger.screenshot(order, bars, str(shot))
    logger.close()
    assert shot.exists()


# ----------------------------------------------------------------------
# wired end-to-end replay (skips cleanly when siblings are missing)
# ----------------------------------------------------------------------
def _first_setup_class(module):
    import inspect as _inspect

    for attr in vars(module).values():
        if (
            isinstance(attr, type)
            and hasattr(attr, "check_setup")
            and not _inspect.isabstract(attr)
        ):
            return attr
    raise AssertionError(f"no concrete strategy class with check_setup in {module.__name__}")


def _wire_pipeline():
    builder_mod = pytest.importorskip("obtc.chain.builder")
    builder_cls = getattr(builder_mod, "ChainBuilder", None)
    if builder_cls is None:
        pytest.skip("obtc.chain.builder has no ChainBuilder yet")

    strategies = []
    for name in STRATEGY_MODULES:
        mod = pytest.importorskip(f"obtc.strategy.{name}")
        strategies.append(_first_setup_class(mod)())

    confluence_mod = pytest.importorskip("obtc.confluence.engine")
    confluence_cls = next(
        (attr for attr in vars(confluence_mod).values()
         if isinstance(attr, type)
         and callable(getattr(attr, "score", None))
         and attr.__name__ != "Signal"),
        None,
    )
    if confluence_cls is None:
        pytest.skip("obtc.confluence.engine has no scoring class yet")

    execution_mod = pytest.importorskip("obtc.execution.executor")
    broker_cls = getattr(execution_mod, "PaperBroker", None)
    risk_cls = getattr(execution_mod, "RiskManager", None)
    executor_cls = getattr(execution_mod, "AsyncExecutor", None)
    if broker_cls is None or executor_cls is None:
        pytest.skip("obtc.execution.executor missing PaperBroker/AsyncExecutor")

    try:
        builder = builder_cls(symbol="BTC/USD")
    except TypeError:
        builder = builder_cls()
    try:
        broker = broker_cls(starting_equity=10000.0)
    except TypeError:
        broker = broker_cls()
    risk = None
    if risk_cls is not None:
        try:
            risk = risk_cls(risk_per_trade_pct=1.0, max_concurrent=3)
        except TypeError:
            risk = risk_cls()
    for make in (
        lambda: executor_cls(broker, risk) if risk is not None else executor_cls(broker),
        lambda: executor_cls(broker=broker, risk=risk),
        lambda: executor_cls(broker),
    ):
        try:
            executor = make()
            break
        except TypeError:
            continue
    else:
        pytest.skip("AsyncExecutor constructor signature not recognized")
    return builder, strategies, confluence_cls(), executor


@pytest.mark.asyncio
async def test_replay_end_to_end(tmp_path):
    builder, strategies, confluence, executor = _wire_pipeline()

    df1 = make_bars(2000, seed=7)
    bars = {
        "1m": df1,
        "5m": resample(df1, "5min"),
        "1H": resample(df1, "1h"),
        "4H": resample(df1, "4h"),
    }
    out = tmp_path / "outputs"
    engine = ReplayEngine(builder, strategies, confluence, executor,
                          output_dir=str(out))
    result = await engine.run(bars)

    assert isinstance(result, BacktestResult)
    assert not result.equity_curve.empty
    assert list(result.equity_curve.columns) == ["timestamp", "equity"]
    assert EXPECTED_METRIC_KEYS <= set(result.metrics)
    assert result.metrics["num_trades"] >= 0  # pipeline integrity, not profit

    trade_log = Path(result.trade_log_path)
    assert trade_log.exists()
    for line in trade_log.read_text(encoding="utf-8").splitlines():
        json.loads(line)  # every line is valid JSON
    assert (out / "equity.csv").exists()
    assert (out / "equity.png").exists()
    assert Path(result.screenshot_dir).is_dir()
    assert result.start < result.end
    assert result.output_dir == str(out)
    assert len(result.trades) == result.metrics["num_trades"]


@pytest.mark.asyncio
async def test_replay_rejects_empty_bars(tmp_path):
    # Raises before touching any component, so a dummy builder is fine.
    engine = ReplayEngine(
        chain_builder=object(),
        strategies=[],
        confluence=None,
        executor=None,
        output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError):
        await engine.run({})
