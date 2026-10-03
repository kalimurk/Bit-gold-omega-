"""OBTC — Order Block Trading Chain.

A full-chain order-block trading system: multi-timeframe data, order-block /
FVG / liquidity / structure detection, HTF→MTF→LTF chain assembly, seven
pluggable master strategies, confluence scoring, risk-managed execution
(paper by default), a deterministic replay backtester, dashboard and alerts.
"""

__version__ = "0.1.0"

from obtc.chain.models import (
    Bias,
    BlockState,
    ChainEvent,
    ChainState,
    Direction,
    FairValueGap,
    Fill,
    FullChain,
    MarketShift,
    Order,
    OrderBlock,
    OrderStatus,
    ScoredSetup,
    Signal,
    Sweep,
    new_id,
    utcnow,
)
from obtc.chain.builder import ChainBuilder
from obtc.strategy import STRATEGIES, BaseStrategy
from obtc.confluence.engine import ConfluenceEngine
from obtc.execution.risk import RiskManager
from obtc.execution.executor import AsyncExecutor, CcxtLiveBroker, PaperBroker
from obtc.backtest.replay import BacktestResult, ReplayEngine
from obtc.dashboard.alerts import WebhookAlerter
from obtc.utils.config import AppConfig, load_config

__all__ = [
    "__version__",
    "Bias",
    "BlockState",
    "ChainEvent",
    "ChainState",
    "Direction",
    "FairValueGap",
    "Fill",
    "FullChain",
    "MarketShift",
    "Order",
    "OrderBlock",
    "OrderStatus",
    "ScoredSetup",
    "Signal",
    "Sweep",
    "new_id",
    "utcnow",
    "ChainBuilder",
    "STRATEGIES",
    "BaseStrategy",
    "ConfluenceEngine",
    "RiskManager",
    "AsyncExecutor",
    "CcxtLiveBroker",
    "PaperBroker",
    "BacktestResult",
    "ReplayEngine",
    "WebhookAlerter",
    "AppConfig",
    "load_config",
]
