"""Confluence engine tests: hand-built signals and chains."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from obtc.chain.models import (
    Bias,
    BlockState,
    ChainState,
    Direction,
    FairValueGap,
    FullChain,
    OrderBlock,
    Signal,
    Sweep,
    new_id,
    utcnow,
)
from obtc.confluence import ConfluenceEngine
from obtc.strategy import LiquiditySweepReversal, SilverBullet

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


def make_signal(**kw):
    base = dict(
        strategy_name="s1_liquidity_sweep_reversal", symbol="BTC/USDT",
        direction=Direction.BULLISH, entry_price=101.0, stop_price=99.5,
        target_2r=104.0, target_3r=105.5, opposing_liquidity=106.0,
        chain_id="chain-1", timeframe="5m",
        rationale={"strategy_score": 90.0},
    )
    base.update(kw)
    return Signal(**base)


def perfect_chain():
    return FullChain(
        id="chain-1", symbol="BTC/USDT", direction=Direction.BULLISH,
        htf_bias=Bias.LONG, htf_timeframe="4H", mtf_timeframe="1H",
        ltf_timeframe="5m", state=ChainState.TRIGGERED,
        ltf_fvg=FairValueGap(
            id=new_id("fvg"), symbol="BTC/USDT", timeframe="5m",
            direction=Direction.BULLISH, top=101.5, bottom=100.5,
            created_at=NOW, state=BlockState.ACTIVE),
        liquidity_sweep=Sweep(
            id=new_id("sweep"), symbol="BTC/USDT", timeframe="5m", level=99.5,
            side="SELL_SIDE", swept_at=NOW, sweep_low=98.5,
            implied_direction=Direction.BULLISH, meta={"session": "london"}),
        notes={"target_liquidity": 106.0},
    )


def test_perfect_setup_is_a_plus():
    engine = ConfluenceEngine(threshold=80.0)
    sig = make_signal()
    setup = engine.score(sig, perfect_chain(),
                         {"now": NOW, "strategy": LiquiditySweepReversal()})
    # 25 (htf) + 24 (15 state + 9 strategy) + 20 (liquidity) + 15 (fvg) + 15 (time)
    assert setup.score == pytest.approx(99.0)
    assert setup.is_a_plus is True
    assert set(setup.components) == {"htf_bias", "chain_alignment", "liquidity", "fvg", "time"}
    assert setup.components["htf_bias"] == pytest.approx(25.0)
    assert setup.components["chain_alignment"] == pytest.approx(24.0)
    assert setup.components["liquidity"] == pytest.approx(20.0)
    assert setup.components["fvg"] == pytest.approx(15.0)
    assert setup.components["time"] == pytest.approx(15.0)
    assert abs(sum(setup.components.values()) - setup.score) < 1e-9


def test_opposed_bias_scores_low():
    engine = ConfluenceEngine(threshold=80.0)
    sig = make_signal(opposing_liquidity=None, rationale={"strategy_score": 50.0})
    chain = FullChain(
        id="chain-2", symbol="BTC/USDT", direction=Direction.BULLISH,
        htf_bias=Bias.SHORT, htf_timeframe="4H", mtf_timeframe="1H",
        ltf_timeframe="5m", state=ChainState.FORMING, notes={},
    )
    setup = engine.score(sig, chain, {"now": NOW, "strategy": LiquiditySweepReversal()})
    # 0 (htf opposed) + 9 (4 state + 5 strategy) + 5 (liquidity) + 0 (fvg) + 15 (time)
    assert setup.score == pytest.approx(29.0)
    assert setup.is_a_plus is False
    assert setup.components["htf_bias"] == pytest.approx(0.0)


def test_neutral_bias_partial_credit():
    engine = ConfluenceEngine()
    sig = make_signal()
    chain = perfect_chain()
    chain.htf_bias = Bias.NEUTRAL
    setup = engine.score(sig, chain, {"now": NOW, "strategy": LiquiditySweepReversal()})
    assert setup.components["htf_bias"] == pytest.approx(10.0)


def test_timing_failure_zeroes_time_component():
    engine = ConfluenceEngine(threshold=80.0)
    strat = SilverBullet()
    ny = ZoneInfo("America/New_York")
    outside = datetime(2026, 10, 2, 9, 30, tzinfo=ny)  # before the killzone
    sig = make_signal(strategy_name="s6_silver_bullet",
                      rationale={"strategy_score": 90.0})
    setup = engine.score(sig, perfect_chain(),
                         {"now": outside, "strategy": strat})
    assert setup.components["time"] == pytest.approx(0.0)
    # 25 (htf) + 24 (alignment) + 20 (liquidity) + 15 (fvg) + 0 (time) = 84
    assert setup.score == pytest.approx(84.0)


def test_timing_success_keeps_time_component():
    engine = ConfluenceEngine()
    strat = SilverBullet()
    ny = ZoneInfo("America/New_York")
    inside = datetime(2026, 10, 2, 10, 30, tzinfo=ny)
    sig = make_signal(strategy_name="s6_silver_bullet")
    setup = engine.score(sig, perfect_chain(), {"now": inside, "strategy": strat})
    assert setup.components["time"] == pytest.approx(15.0)


def test_score_capped_at_100():
    engine = ConfluenceEngine()
    sig = make_signal(rationale={"strategy_score": 100.0})
    setup = engine.score(sig, perfect_chain(),
                         {"now": NOW, "strategy": LiquiditySweepReversal()})
    assert setup.score <= 100.0


def test_custom_threshold():
    engine = ConfluenceEngine(threshold=99.5)
    sig = make_signal()
    setup = engine.score(sig, perfect_chain(),
                         {"now": NOW, "strategy": LiquiditySweepReversal()})
    assert setup.score == pytest.approx(99.0)
    assert setup.is_a_plus is False
