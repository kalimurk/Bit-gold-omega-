"""OBTC entry strategies."""
from obtc.strategy.base import BaseStrategy
from obtc.strategy.s1_liquidity_sweep_reversal import LiquiditySweepReversal
from obtc.strategy.s2_continuation import Continuation
from obtc.strategy.s3_breaker_reclaim import BreakerReclaim
from obtc.strategy.s4_mitigation_flip import MitigationFlip
from obtc.strategy.s5_turtle_soup import TurtleSoup
from obtc.strategy.s6_silver_bullet import SilverBullet
from obtc.strategy.s7_orderflow_stacking import OrderflowStacking

STRATEGIES = [
    LiquiditySweepReversal,
    Continuation,
    BreakerReclaim,
    MitigationFlip,
    TurtleSoup,
    SilverBullet,
    OrderflowStacking,
]

__all__ = [
    "BaseStrategy",
    "LiquiditySweepReversal",
    "Continuation",
    "BreakerReclaim",
    "MitigationFlip",
    "TurtleSoup",
    "SilverBullet",
    "OrderflowStacking",
    "STRATEGIES",
]
