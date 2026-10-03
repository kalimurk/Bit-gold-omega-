"""OBTC shared domain models — single source of truth.

This file is owned by the build coordinator. Every agent imports domain types
from ``obtc.chain.models``. Do not redefine these types elsewhere and do not
modify this file without coordinator approval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class Bias(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class BlockState(str, Enum):
    ACTIVE = "ACTIVE"
    MITIGATED = "MITIGATED"
    INVALIDATED = "INVALIDATED"
    FLIPPED = "FLIPPED"
    FILLED = "FILLED"


class ChainState(str, Enum):
    FORMING = "FORMING"
    VALID = "VALID"
    TRIGGERED = "TRIGGERED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class OrderStatus(str, Enum):
    NEW = "NEW"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


@dataclass
class OrderBlock:
    id: str
    symbol: str
    timeframe: str
    direction: Direction
    high: float
    low: float
    fifty_pct: float
    open: float
    close: float
    created_at: datetime
    state: BlockState = BlockState.ACTIVE
    mitigated_at: datetime | None = None
    invalidated_at: datetime | None = None
    invalidation_wick: float | None = None
    flipped_from: Direction | None = None
    flipped_at: datetime | None = None
    origin_index: int = 0
    volume: float = 0.0
    atr_at_creation: float = 0.0
    extended: bool = False

    @property
    def is_bullish(self) -> bool:
        return self.direction == Direction.BULLISH

    @property
    def zone_range(self) -> float:
        return self.high - self.low


@dataclass
class FairValueGap:
    id: str
    symbol: str
    timeframe: str
    direction: Direction
    top: float
    bottom: float
    created_at: datetime
    state: BlockState = BlockState.ACTIVE
    filled_at: datetime | None = None
    invalidated_at: datetime | None = None
    origin_index: int = 0

    @property
    def is_bullish(self) -> bool:
        return self.direction == Direction.BULLISH

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def size(self) -> float:
        return self.top - self.bottom


@dataclass
class Sweep:
    id: str
    symbol: str
    timeframe: str
    level: float
    side: str  # "BUY_SIDE" (wick above old highs) or "SELL_SIDE" (wick below old lows)
    swept_at: datetime
    sweep_high: float = 0.0
    sweep_low: float = 0.0
    closed_back_inside: bool = True
    implied_direction: Direction = Direction.BULLISH
    meta: dict = field(default_factory=dict)


@dataclass
class MarketShift:
    id: str
    symbol: str
    timeframe: str
    kind: str  # "BOS" or "MSS"
    direction: Direction
    broken_level: float
    broken_at: datetime
    origin_index: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class ChainLink:
    timeframe: str
    kind: str
    ref_id: str
    created_at: datetime
    note: str = ""


@dataclass
class FullChain:
    id: str
    symbol: str
    direction: Direction
    htf_bias: Bias
    htf_timeframe: str
    mtf_timeframe: str
    ltf_timeframe: str
    state: ChainState = ChainState.FORMING
    htf_ob: OrderBlock | None = None
    mtf_fvgs: list = field(default_factory=list)
    mtf_breaker: OrderBlock | None = None
    ltf_shift: MarketShift | None = None
    ltf_ob: OrderBlock | None = None
    ltf_fvg: FairValueGap | None = None
    liquidity_sweep: Sweep | None = None
    links: list = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    invalidated_at: datetime | None = None
    notes: dict = field(default_factory=dict)

    def add_link(self, timeframe: str, kind: str, ref_id: str, note: str = "") -> None:
        self.links.append(
            ChainLink(timeframe=timeframe, kind=kind, ref_id=ref_id,
                      created_at=utcnow(), note=note)
        )
        self.updated_at = utcnow()


@dataclass
class Signal:
    strategy_name: str
    symbol: str
    direction: Direction
    entry_price: float
    stop_price: float
    target_2r: float
    target_3r: float
    opposing_liquidity: float | None
    chain_id: str
    timeframe: str
    created_at: datetime = field(default_factory=utcnow)
    rationale: dict = field(default_factory=dict)
    score: float = 0.0
    partials: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.partials:
            self.partials = [
                {"at": "2R", "close_pct": 50.0},
                {"at": "3R", "close_pct": 100.0},
            ]

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop_price)

    @property
    def is_long(self) -> bool:
        return self.direction == Direction.BULLISH


@dataclass
class ScoredSetup:
    signal: Signal
    score: float
    components: dict = field(default_factory=dict)
    is_a_plus: bool = False
    scored_at: datetime = field(default_factory=utcnow)


@dataclass
class Fill:
    price: float
    size: float
    at: datetime
    kind: str  # ENTRY, PARTIAL_2R, PARTIAL_3R, STOP, TARGET, BREAKEVEN, EOD


@dataclass
class Order:
    id: str
    signal: Signal
    size: float
    status: OrderStatus = OrderStatus.NEW
    entry_fill: float | None = None
    stop_price: float | None = None
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    fills: list = field(default_factory=list)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    remaining: float = 0.0
    notes: dict = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        return self.signal.is_long


@dataclass
class ChainEvent:
    type: str  # CHAIN_VALID, CHAIN_TRIGGERED, CHAIN_INVALIDATED, A_PLUS_SETUP
    chain_id: str
    at: datetime = field(default_factory=utcnow)
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "chain_id": self.chain_id,
            "at": self.at.isoformat(),
            "payload": self.payload,
        }
