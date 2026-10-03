"""Execution tests: risk manager, paper broker lifecycle, live gate."""
from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest

from obtc.chain.models import (
    Direction,
    Order,
    OrderStatus,
    ScoredSetup,
    Signal,
    new_id,
    utcnow,
)
from obtc.execution import AsyncExecutor, CcxtLiveBroker, PaperBroker, RiskManager


def long_signal(entry=100.0, stop=95.0, chain_id="chain-1"):
    risk = entry - stop
    return Signal(
        strategy_name="s1_liquidity_sweep_reversal", symbol="BTC/USDT",
        direction=Direction.BULLISH, entry_price=entry, stop_price=stop,
        target_2r=entry + 2 * risk, target_3r=entry + 3 * risk,
        opposing_liquidity=None, chain_id=chain_id, timeframe="5m",
    )


def short_signal(entry=100.0, stop=105.0, chain_id="chain-1"):
    risk = stop - entry
    return Signal(
        strategy_name="s1_liquidity_sweep_reversal", symbol="BTC/USDT",
        direction=Direction.BEARISH, entry_price=entry, stop_price=stop,
        target_2r=entry - 2 * risk, target_3r=entry - 3 * risk,
        opposing_liquidity=None, chain_id=chain_id, timeframe="5m",
    )


def bar(high, low, close):
    return SimpleNamespace(high=high, low=low, close=close, at=utcnow())


# ------------------------------------------------------------ RiskManager
def test_position_sizing_math():
    rm = RiskManager(risk_per_trade_pct=1.0, per_trade_risk_cap_pct=2.0)
    sig = long_signal(entry=150.0, stop=100.0)  # risk = 50
    # min(10000*1%, 10000*2%) / 50 = 100/50
    assert rm.position_size(sig, 10000.0) == pytest.approx(2.0)


def test_position_size_zero_risk():
    rm = RiskManager()
    sig = long_signal(entry=100.0, stop=100.0)
    assert rm.position_size(sig, 10000.0) == 0.0
    assert rm.position_size(long_signal(), 0.0) == 0.0


def test_daily_loss_limit_halts():
    rm = RiskManager(daily_loss_limit_pct=3.0)
    rm.set_day_start(10000.0)
    rm.register_close(-350.0, utcnow())  # beyond the 300 limit
    assert rm.halted is True
    ok, reason = rm.can_open("chain-1", 0, rm.day_pnl, 10000.0)
    assert ok is False
    assert "daily loss limit" in reason


def test_day_rollover_resets_halt():
    rm = RiskManager(daily_loss_limit_pct=3.0)
    rm.set_day_start(10000.0)
    rm.register_close(-350.0, utcnow())
    assert rm.halted is True
    rm.register_close(10.0, utcnow() + timedelta(days=1))
    assert rm.halted is False
    assert rm.day_pnl == pytest.approx(10.0)


def test_max_concurrent_blocks():
    rm = RiskManager(max_concurrent=3)
    rm.set_day_start(10000.0)
    ok, reason = rm.can_open("chain-9", 3, 0.0, 10000.0)
    assert ok is False
    assert "concurrent" in reason
    ok, _ = rm.can_open("chain-9", 2, 0.0, 10000.0)
    assert ok is True


def test_chain_already_open_blocks():
    rm = RiskManager()
    rm.set_day_start(10000.0)
    ok, reason = rm.can_open("chain-1", 0, 0.0, 10000.0, {"chain-1"})
    assert ok is False
    assert "already open" in reason


def test_trail_to_breakeven():
    rm = RiskManager()
    sig = long_signal()
    order = Order(id=new_id("order"), signal=sig, size=2.0, status=OrderStatus.OPEN,
                  entry_fill=100.0, stop_price=95.0, remaining=2.0)
    assert rm.trail_to_breakeven(order, True) is True
    assert order.stop_price == pytest.approx(100.0)
    assert order.notes["trailed_to_breakeven"] is True
    assert rm.trail_to_breakeven(order, True) is False  # already at breakeven
    order2 = Order(id=new_id("order"), signal=sig, size=2.0, status=OrderStatus.OPEN,
                   entry_fill=100.0, stop_price=95.0, remaining=2.0)
    assert rm.trail_to_breakeven(order2, False) is False
    assert order2.stop_price == pytest.approx(95.0)


# ------------------------------------------------------------ PaperBroker
def test_paper_broker_full_lifecycle():
    broker = PaperBroker(starting_equity=10000.0, fee_bps=5.0, slippage_bps=2.0)
    sig = long_signal()
    order = asyncio.run(broker.place_order(sig, 2.0))
    assert order.status == OrderStatus.OPEN
    assert order.entry_fill == pytest.approx(100.0 * 1.0002)  # slippage against trader
    assert order.stop_price == pytest.approx(95.0)
    assert order.remaining == pytest.approx(2.0)
    entry_fee = 2.0 * 100.02 * 0.0005
    assert broker.cash == pytest.approx(10000.0 - entry_fee)

    # bar 1: taps 2R -> 50% partial, stop moves to breakeven
    changed = asyncio.run(broker.update(bar(111.0, 99.0, 108.0)))
    assert len(changed) == 1 and changed[0] is order
    assert order.status == OrderStatus.PARTIAL
    assert order.remaining == pytest.approx(1.0)
    assert order.stop_price == pytest.approx(order.entry_fill)
    assert [f.kind for f in order.fills] == ["ENTRY", "PARTIAL_2R"]
    assert order.fills[1].price == pytest.approx(110.0 * 0.9998)
    assert order.notes["partial_2r_taken"] is True

    # bar 2: taps 3R -> close the rest
    asyncio.run(broker.update(bar(116.0, 112.0, 114.0)))
    assert order.status == OrderStatus.CLOSED
    assert order.remaining == pytest.approx(0.0)
    assert order.closed_at is not None
    assert [f.kind for f in order.fills] == ["ENTRY", "PARTIAL_2R", "PARTIAL_3R"]

    expected_pnl = (110 * 0.9998 - 100 * 1.0002) + (115 * 0.9998 - 100 * 1.0002)
    assert order.realized_pnl == pytest.approx(expected_pnl)
    expected_fees = entry_fee + (110 * 0.9998 * 0.0005) + (115 * 0.9998 * 0.0005)
    assert order.fees_paid == pytest.approx(expected_fees)
    assert broker.cash == pytest.approx(10000.0 + expected_pnl - expected_fees)
    assert broker.equity == pytest.approx(broker.cash)  # nothing open
    assert broker.open_orders() == []


def test_paper_broker_stop_out():
    broker = PaperBroker(starting_equity=10000.0, fee_bps=5.0, slippage_bps=2.0)
    sig = short_signal()
    order = asyncio.run(broker.place_order(sig, 2.0))
    assert order.entry_fill == pytest.approx(100.0 * 0.9998)
    asyncio.run(broker.update(bar(106.0, 98.0, 99.0)))
    assert order.status == OrderStatus.CLOSED
    assert order.fills[-1].kind == "STOP"
    assert order.fills[-1].price == pytest.approx(105.0 * 1.0002)
    assert order.realized_pnl == pytest.approx((99.98 - 105.021) * 2.0)


def test_stop_wins_over_target_in_one_bar():
    broker = PaperBroker()
    order = asyncio.run(broker.place_order(long_signal(), 2.0))
    asyncio.run(broker.update(bar(112.0, 94.0, 100.0)))  # both stop and 2R inside
    assert [f.kind for f in order.fills] == ["ENTRY", "STOP"]
    assert order.status == OrderStatus.CLOSED


def test_paper_broker_unrealized_equity():
    broker = PaperBroker(starting_equity=10000.0)
    order = asyncio.run(broker.place_order(long_signal(), 2.0))
    asyncio.run(broker.update(bar(108.0, 99.0, 107.0)))  # 2R not hit (108 < 110)
    assert order.status == OrderStatus.OPEN
    unrealized = (107.0 - order.entry_fill) * 2.0
    assert broker.equity == pytest.approx(broker.cash + unrealized)


def test_paper_broker_close_all():
    broker = PaperBroker()
    order = asyncio.run(broker.place_order(long_signal(), 2.0))
    closed = asyncio.run(broker.close_all(105.0, utcnow(), reason="EOD"))
    assert closed == [order]
    assert order.status == OrderStatus.CLOSED
    assert order.fills[-1].kind == "EOD"
    assert order.fills[-1].price == pytest.approx(105.0 * 0.9998)


# ------------------------------------------------------------ CcxtLiveBroker gate
def test_ccxt_live_broker_gate_mode():
    with pytest.raises(RuntimeError, match="Live execution disabled"):
        CcxtLiveBroker({})


def test_ccxt_live_broker_gate_credentials(monkeypatch):
    monkeypatch.delenv("OBTC_EXCHANGE_API_KEY", raising=False)
    monkeypatch.delenv("OBTC_EXCHANGE_API_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="missing live credentials"):
        CcxtLiveBroker({"mode": "live"})


# ------------------------------------------------------------ AsyncExecutor
def test_async_executor_submit_and_reject():
    broker = PaperBroker()
    risk = RiskManager()
    risk.set_day_start(10000.0)
    ex = AsyncExecutor(broker, risk)
    setup = ScoredSetup(signal=long_signal(chain_id="chain-1"), score=90.0,
                        components={}, is_a_plus=True)
    order = asyncio.run(ex.submit(setup, 10000.0))
    assert order is not None
    assert order.size == pytest.approx(20.0)  # min(1%, 2%) of 10k / 5 risk
    assert ex.open_chain_ids() == {"chain-1"}
    # same chain already open -> rejected
    assert asyncio.run(ex.submit(setup, 10000.0)) is None
    # on_bar delegates to the broker
    changed = asyncio.run(ex.on_bar(bar(101.0, 99.5, 100.5)))
    assert changed == []
    assert len(ex.open_orders()) == 1


def test_async_executor_rejects_when_halted():
    broker = PaperBroker()
    risk = RiskManager()
    risk.set_day_start(10000.0)
    risk.register_close(-400.0, utcnow())
    ex = AsyncExecutor(broker, risk)
    setup = ScoredSetup(signal=long_signal(chain_id="chain-2"), score=95.0,
                        components={}, is_a_plus=True)
    assert asyncio.run(ex.submit(setup, 10000.0)) is None
    assert broker.orders == []
