"""Strategy tests: hand-built FullChain fixtures, no network, no detectors."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from obtc.chain.models import (
    Bias,
    BlockState,
    ChainState,
    Direction,
    FairValueGap,
    FullChain,
    MarketShift,
    OrderBlock,
    Sweep,
    new_id,
    utcnow,
)
from obtc.strategy import (
    STRATEGIES,
    BaseStrategy,
    BreakerReclaim,
    Continuation,
    LiquiditySweepReversal,
    MitigationFlip,
    OrderflowStacking,
    SilverBullet,
    TurtleSoup,
)

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
RATIONALE_KEYS = ("ob_high", "ob_low", "entry", "stop", "target_2r", "target_3r", "strategy_score")


# --------------------------------------------------------------- builders
def make_ob(direction, high, low, state=BlockState.ACTIVE, timeframe="5m",
            volume=300.0, created_at=None, mitigated_at=None, **kw):
    return OrderBlock(
        id=new_id("ob"), symbol="BTC/USDT", timeframe=timeframe, direction=direction,
        high=high, low=low, fifty_pct=(high + low) / 2.0, open=low, close=high,
        created_at=created_at or NOW, state=state, mitigated_at=mitigated_at,
        volume=volume, **kw,
    )


def make_fvg(direction, top, bottom, timeframe="5m", created_at=None,
             state=BlockState.ACTIVE):
    return FairValueGap(
        id=new_id("fvg"), symbol="BTC/USDT", timeframe=timeframe, direction=direction,
        top=top, bottom=bottom, created_at=created_at or NOW, state=state,
    )


def make_sweep(direction, level, side, swept_at=None, session="london",
               sweep_high=0.0, sweep_low=0.0):
    return Sweep(
        id=new_id("sweep"), symbol="BTC/USDT", timeframe="5m", level=level,
        side=side, swept_at=swept_at or NOW, sweep_high=sweep_high,
        sweep_low=sweep_low, closed_back_inside=True,
        implied_direction=direction, meta={"session": session},
    )


def make_shift(kind, direction, broken_level, displacement_atr=1.5, broken_at=None):
    return MarketShift(
        id=new_id("shift"), symbol="BTC/USDT", timeframe="5m", kind=kind,
        direction=direction, broken_level=broken_level, broken_at=broken_at or NOW,
        origin_index=0, meta={"displacement_atr": displacement_atr},
    )


def make_chain(**kw):
    base = dict(
        id=new_id("chain"), symbol="BTC/USDT", direction=Direction.BULLISH,
        htf_bias=Bias.LONG, htf_timeframe="4H", mtf_timeframe="1H",
        ltf_timeframe="5m", state=ChainState.VALID,
        notes={"atr_ltf": 2.0, "target_liquidity": 106.0},
    )
    base.update(kw)
    return FullChain(**base)


def assert_signal_sane(sig, entry, stop):
    assert sig is not None
    risk = abs(entry - stop)
    assert sig.risk == pytest.approx(risk)
    assert sig.risk > 0
    if sig.is_long:
        assert sig.target_2r == pytest.approx(entry + 2 * risk)
        assert sig.target_3r == pytest.approx(entry + 3 * risk)
    else:
        assert sig.target_2r == pytest.approx(entry - 2 * risk)
        assert sig.target_3r == pytest.approx(entry - 3 * risk)
    for key in RATIONALE_KEYS:
        assert key in sig.rationale, f"missing rationale key {key}"
    assert sig.rationale["entry"] == pytest.approx(entry)
    assert sig.rationale["stop"] == pytest.approx(stop)
    assert 0.0 <= sig.rationale["strategy_score"] <= 100.0
    assert sig.score == pytest.approx(sig.rationale["strategy_score"])


# ------------------------------------------------------------------ registry
def test_strategy_registry():
    assert len(STRATEGIES) == 7
    names = [s.name for s in STRATEGIES]
    assert len(set(names)) == 7
    for cls in STRATEGIES:
        assert issubclass(cls, BaseStrategy)
        assert cls().timeframe_set == ("4H", "1H", "5m")


# ------------------------------------------------------------------ S1
def s1_chain(**kw):
    base = dict(
        liquidity_sweep=make_sweep(Direction.BULLISH, level=99.5, side="SELL_SIDE",
                                  sweep_low=98.5, session="london"),
        ltf_shift=make_shift("MSS", Direction.BULLISH, broken_level=101.5,
                             displacement_atr=2.0),
        ltf_ob=make_ob(Direction.BULLISH, 102.0, 100.0, state=BlockState.ACTIVE),
    )
    base.update(kw)
    return make_chain(**base)


def test_s1_positive():
    sig = LiquiditySweepReversal().check_setup(s1_chain())
    assert sig is not None and sig.is_long
    assert sig.entry_price == pytest.approx(101.0)
    assert sig.stop_price == pytest.approx(99.5)  # 100 - 0.25 * 2
    assert sig.opposing_liquidity == pytest.approx(106.0)
    assert_signal_sane(sig, 101.0, 99.5)


def test_s1_negative_no_sweep():
    assert LiquiditySweepReversal().check_setup(s1_chain(liquidity_sweep=None)) is None


def test_s1_negative_wrong_shift_kind():
    c = s1_chain(ltf_shift=make_shift("BOS", Direction.BULLISH, 101.5))
    assert LiquiditySweepReversal().check_setup(c) is None


# ------------------------------------------------------------------ S2
def s2_chain(**kw):
    base = dict(
        ltf_shift=make_shift("BOS", Direction.BULLISH, broken_level=101.5,
                             displacement_atr=2.0),
        ltf_ob=make_ob(Direction.BULLISH, 102.0, 100.0, state=BlockState.ACTIVE),
        ltf_fvg=make_fvg(Direction.BULLISH, top=101.5, bottom=100.5),
    )
    base.update(kw)
    return make_chain(**base)


def test_s2_positive():
    sig = Continuation().check_setup(s2_chain())
    assert sig is not None and sig.is_long
    assert sig.entry_price == pytest.approx(101.0)  # FVG midpoint
    assert sig.stop_price == pytest.approx(99.5)
    assert_signal_sane(sig, 101.0, 99.5)


def test_s2_negative_opposed_htf_bias():
    assert Continuation().check_setup(s2_chain(htf_bias=Bias.SHORT)) is None


def test_s2_negative_no_fvg():
    assert Continuation().check_setup(s2_chain(ltf_fvg=None)) is None


# ------------------------------------------------------------------ S3
def s3_chain(**kw):
    br = make_ob(Direction.BULLISH, 103.0, 101.0, state=BlockState.FLIPPED,
                 timeframe="1H", flipped_from=Direction.BEARISH)
    base = dict(
        mtf_breaker=br,
        notes={"atr_ltf": 2.0, "target_liquidity": 107.0, "breaker_retest": True},
    )
    base.update(kw)
    return make_chain(**base)


def test_s3_positive():
    sig = BreakerReclaim().check_setup(s3_chain())
    assert sig is not None and sig.is_long
    assert sig.entry_price == pytest.approx(102.0)
    assert sig.stop_price == pytest.approx(100.5)
    assert sig.rationale["flipped_from"] == Direction.BEARISH.value
    assert_signal_sane(sig, 102.0, 100.5)


def test_s3_negative_no_retest():
    c = s3_chain(notes={"atr_ltf": 2.0, "target_liquidity": 107.0})
    assert BreakerReclaim().check_setup(c) is None


def test_s3_negative_not_flipped():
    br = make_ob(Direction.BULLISH, 103.0, 101.0, state=BlockState.ACTIVE,
                 timeframe="1H")
    c = s3_chain(mtf_breaker=br)
    assert BreakerReclaim().check_setup(c) is None


# ------------------------------------------------------------------ S4
def s4_chain(**kw):
    mit_at = NOW - timedelta(hours=1)
    base = dict(
        ltf_ob=make_ob(Direction.BULLISH, 105.0, 103.0, state=BlockState.MITIGATED,
                       mitigated_at=mit_at, created_at=NOW - timedelta(hours=3)),
        mtf_fvgs=[make_fvg(Direction.BEARISH, top=102.0, bottom=100.0,
                           timeframe="1H", created_at=NOW - timedelta(minutes=30))],
        notes={"target_liquidity": 96.0},
    )
    base.update(kw)
    return make_chain(**base)


def test_s4_positive_flip():
    sig = MitigationFlip().check_setup(s4_chain())
    assert sig is not None
    assert sig.direction == Direction.BEARISH  # the flip
    assert not sig.is_long
    assert sig.entry_price == pytest.approx(101.0)
    assert sig.stop_price == pytest.approx(102.25)  # 102 + 0.25 * 1.0 (fallback ATR)
    assert sig.rationale["flip"] is True
    assert_signal_sane(sig, 101.0, 102.25)


def test_s4_negative_no_inversion():
    assert MitigationFlip().check_setup(s4_chain(mtf_fvgs=[])) is None


# ------------------------------------------------------------------ S5
def s5_chain(**kw):
    swept = datetime(2026, 10, 2, 3, 0, tzinfo=timezone.utc)
    base = dict(
        liquidity_sweep=make_sweep(Direction.BULLISH, level=99.5, side="SELL_SIDE",
                                   swept_at=swept, session="asia", sweep_low=98.5),
        ltf_shift=make_shift("MSS", Direction.BULLISH, broken_level=101.5),
        ltf_ob=make_ob(Direction.BULLISH, 102.0, 100.0, state=BlockState.ACTIVE),
        notes={"atr_ltf": 2.0, "asia_high": 104.0, "asia_low": 98.0},
    )
    base.update(kw)
    return make_chain(**base)


def test_s5_positive():
    strat = TurtleSoup()
    c = s5_chain()
    assert strat.needs_timing is True
    assert strat.timing_ok(NOW, c) is True
    sig = strat.check_setup(c)
    assert sig is not None and sig.is_long
    assert sig.entry_price == pytest.approx(101.0)
    assert sig.stop_price == pytest.approx(99.5)
    assert sig.opposing_liquidity == pytest.approx(104.0)  # asia_high for longs
    assert_signal_sane(sig, 101.0, 99.5)


def test_s5_negative_wrong_session():
    c = s5_chain(liquidity_sweep=make_sweep(
        Direction.BULLISH, level=99.5, side="SELL_SIDE", session="london",
        swept_at=datetime(2026, 10, 2, 3, 0, tzinfo=timezone.utc), sweep_low=98.5))
    assert TurtleSoup().check_setup(c) is None


def test_s5_negative_timing():
    bad = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
    c = s5_chain(liquidity_sweep=make_sweep(
        Direction.BULLISH, level=99.5, side="SELL_SIDE", swept_at=bad,
        session="asia", sweep_low=98.5))
    assert TurtleSoup().timing_ok(bad, c) is False


# ------------------------------------------------------------------ S6
def s6_chain(**kw):
    base = dict(
        state=ChainState.TRIGGERED,
        ltf_fvg=make_fvg(Direction.BULLISH, top=101.5, bottom=100.5),
        ltf_ob=make_ob(Direction.BULLISH, 102.0, 100.0, state=BlockState.ACTIVE),
    )
    base.update(kw)
    return make_chain(**base)


def test_s6_positive():
    strat = SilverBullet()
    ny = ZoneInfo("America/New_York")
    now_kz = datetime(2026, 10, 2, 10, 30, tzinfo=ny)
    c = s6_chain()
    assert strat.needs_timing is True
    assert strat.timing_ok(now_kz, c) is True
    sig = strat.check_setup(c)
    assert sig is not None and sig.is_long
    assert sig.entry_price == pytest.approx(101.0)  # FVG midpoint
    assert sig.stop_price == pytest.approx(99.5)
    assert_signal_sane(sig, 101.0, 99.5)


def test_s6_negative_wrong_timing():
    ny = ZoneInfo("America/New_York")
    now_early = datetime(2026, 10, 2, 9, 30, tzinfo=ny)
    assert SilverBullet().timing_ok(now_early, s6_chain()) is False


def test_s6_negative_not_triggered():
    assert SilverBullet().check_setup(s6_chain(state=ChainState.VALID)) is None


# ------------------------------------------------------------------ S7
def s7_chain(**kw):
    stacked = [
        make_ob(Direction.BULLISH, 100.0, 98.0, state=BlockState.ACTIVE),
        make_ob(Direction.BULLISH, 99.0, 97.0, state=BlockState.ACTIVE),
    ]
    base = dict(
        notes={
            "stacked_obs": stacked,
            "dealing_range": (95.0, 105.0),
            "last_price": 97.5,  # below equilibrium 100 -> discount
            "target_liquidity": 104.0,
        },
    )
    base.update(kw)
    return make_chain(**base)


def test_s7_positive():
    sig = OrderflowStacking().check_setup(s7_chain())
    assert sig is not None and sig.is_long
    assert sig.entry_price == pytest.approx(98.0)  # deepest stacked OB 50%
    assert sig.stop_price == pytest.approx(96.75)  # 97 - 0.25 * 1.0 (fallback ATR)
    assert sig.rationale["stack_count"] == 2
    assert_signal_sane(sig, 98.0, 96.75)


def test_s7_negative_no_discount():
    c = s7_chain(notes={
        "stacked_obs": [
            make_ob(Direction.BULLISH, 100.0, 98.0),
            make_ob(Direction.BULLISH, 99.0, 97.0),
        ],
        "dealing_range": (95.0, 105.0),
        "last_price": 102.0,  # premium -> no long
        "target_liquidity": 104.0,
    })
    assert OrderflowStacking().check_setup(c) is None


def test_s7_negative_single_block():
    c = s7_chain(notes={
        "stacked_obs": [make_ob(Direction.BULLISH, 100.0, 98.0)],
        "dealing_range": (95.0, 105.0),
        "last_price": 97.5,
    })
    assert OrderflowStacking().check_setup(c) is None


def test_s7_dict_obs_supported():
    notes = {
        "stacked_obs": [
            {"direction": "BULLISH", "high": 100.0, "low": 98.0, "fifty_pct": 99.0},
            {"direction": "BULLISH", "high": 99.0, "low": 97.0, "fifty_pct": 98.0},
        ],
        "dealing_range": (95.0, 105.0),
        "last_price": 97.5,
    }
    sig = OrderflowStacking().check_setup(s7_chain(notes=notes))
    assert sig is not None
    assert sig.entry_price == pytest.approx(98.0)


# ------------------------------------------------------- confluence_score API
def test_confluence_score_returns_0_100_with_chain_ctx():
    c = s1_chain()
    strat = LiquiditySweepReversal()
    sig = strat.check_setup(c)
    score = strat.confluence_score(sig, {"chain": c})
    assert 0.0 <= score <= 100.0
    assert strat.confluence_score(sig, {}) == 0.0  # missing chain -> fail closed
