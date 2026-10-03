"""Execution: paper broker for backtests, gated live broker, async executor."""
from __future__ import annotations

import logging
import os

from obtc.chain.models import Fill, Order, OrderStatus, ScoredSetup, new_id, utcnow
from obtc.execution.risk import RiskManager

log = logging.getLogger(__name__)


def _bar_field(bar, name, default=None):
    if isinstance(bar, dict):
        return bar.get(name, default)
    return getattr(bar, name, default)


class PaperBroker:
    """Immediate-fill paper broker.

    Fills market orders at the signal entry adjusted for slippage against
    the trader, charges fees on every fill, manages the 2R partial (moving
    the stop to breakeven) and the 3R final close. When a bar contains both
    the stop and a target, the stop wins (conservative).

    cash: starting equity - fees + realized PnL.
    equity: cash + unrealized PnL of open positions (marked on each update).
    """

    def __init__(self, starting_equity: float = 10000.0,
                 fee_bps: float = 5.0, slippage_bps: float = 2.0):
        self.starting_equity = float(starting_equity)
        self.cash = float(starting_equity)
        self.realized_pnl = 0.0
        self._fee = float(fee_bps) / 10000.0
        self._slip = float(slippage_bps) / 10000.0
        self.orders: list[Order] = []
        self._marks: dict = {}

    # -------------------------------------------------------------- orders
    def open_orders(self) -> list[Order]:
        return [o for o in self.orders
                if o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)]

    async def place_order(self, signal, size: float) -> Order:
        slip = self._slip
        px = signal.entry_price * (1.0 + slip) if signal.is_long else signal.entry_price * (1.0 - slip)
        fee = size * px * self._fee
        now = utcnow()
        order = Order(
            id=new_id("order"),
            signal=signal,
            size=float(size),
            status=OrderStatus.OPEN,
            entry_fill=px,
            stop_price=signal.stop_price,
            opened_at=now,
            remaining=float(size),
            fees_paid=fee,
            fills=[Fill(price=px, size=float(size), at=now, kind="ENTRY")],
        )
        self.cash -= fee
        self.orders.append(order)
        self._marks[order.id] = px
        return order

    def _close_qty(self, order: Order, qty: float, price: float, at, kind: str) -> None:
        qty = min(qty, order.remaining)
        if qty <= 0:
            return
        fee = qty * price * self._fee
        if order.is_long:
            pnl = (price - order.entry_fill) * qty
        else:
            pnl = (order.entry_fill - price) * qty
        order.fills.append(Fill(price=price, size=qty, at=at, kind=kind))
        order.fees_paid += fee
        order.realized_pnl += pnl
        order.remaining = max(0.0, order.remaining - qty)
        self.cash += pnl - fee
        self.realized_pnl += pnl
        if order.remaining <= 1e-9:
            order.remaining = 0.0
            order.status = OrderStatus.CLOSED
            order.closed_at = at

    async def update(self, bar) -> list[Order]:
        """Process one bar; return orders whose state changed."""
        high = _bar_field(bar, "high")
        low = _bar_field(bar, "low")
        close = _bar_field(bar, "close")
        at = _bar_field(bar, "at") or utcnow()
        changed: list[Order] = []
        for order in self.open_orders():
            sig = order.signal
            long = order.is_long
            slip = self._slip
            touched = False

            stop_hit = (low <= order.stop_price) if long else (high >= order.stop_price)
            if stop_hit:
                px = order.stop_price * (1.0 - slip) if long else order.stop_price * (1.0 + slip)
                self._close_qty(order, order.remaining, px, at, "STOP")
                touched = True
            elif long and order.status == OrderStatus.OPEN \
                    and not order.notes.get("partial_2r_taken") and high >= sig.target_2r:
                pct = (sig.partials[0].get("close_pct", 50.0)
                       if sig.partials else 50.0)
                qty = order.size * pct / 100.0
                px = sig.target_2r * (1.0 - slip)
                self._close_qty(order, qty, px, at, "PARTIAL_2R")
                order.status = OrderStatus.PARTIAL
                order.stop_price = order.entry_fill  # breakeven
                order.notes["partial_2r_taken"] = True
                touched = True
            elif (not long) and order.status == OrderStatus.OPEN \
                    and not order.notes.get("partial_2r_taken") and low <= sig.target_2r:
                pct = (sig.partials[0].get("close_pct", 50.0)
                       if sig.partials else 50.0)
                qty = order.size * pct / 100.0
                px = sig.target_2r * (1.0 + slip)
                self._close_qty(order, qty, px, at, "PARTIAL_2R")
                order.status = OrderStatus.PARTIAL
                order.stop_price = order.entry_fill  # breakeven
                order.notes["partial_2r_taken"] = True
                touched = True
            elif long and order.status == OrderStatus.PARTIAL and high >= sig.target_3r:
                px = sig.target_3r * (1.0 - slip)
                self._close_qty(order, order.remaining, px, at, "PARTIAL_3R")
                touched = True
            elif (not long) and order.status == OrderStatus.PARTIAL and low <= sig.target_3r:
                px = sig.target_3r * (1.0 + slip)
                self._close_qty(order, order.remaining, px, at, "PARTIAL_3R")
                touched = True

            self._marks[order.id] = close
            if touched:
                changed.append(order)
        return changed

    async def close_all(self, price: float, at, reason: str = "EOD") -> list[Order]:
        closed: list[Order] = []
        for order in self.open_orders():
            px = price * (1.0 - self._slip) if order.is_long else price * (1.0 + self._slip)
            self._close_qty(order, order.remaining, px, at, reason)
            closed.append(order)
        return closed

    # ------------------------------------------------------------ accounting
    def unrealized(self) -> float:
        total = 0.0
        for order in self.open_orders():
            mark = self._marks.get(order.id, order.entry_fill)
            if order.is_long:
                total += (mark - order.entry_fill) * order.remaining
            else:
                total += (order.entry_fill - mark) * order.remaining
        return total

    @property
    def equity(self) -> float:
        return self.cash + self.unrealized()


class CcxtLiveBroker:
    """Live broker over ccxt. HARD GATE: constructing with anything other
    than ``mode="live"`` in config raises, and live credentials must be
    present in the environment.

    Real order routing code; never exercised in tests beyond the gate.
    """

    def __init__(self, config: dict):
        config = dict(config or {})
        if config.get("mode") != "live":
            raise RuntimeError(
                "Live execution disabled: set mode='live' in config AND export "
                "OBTC_EXCHANGE_API_KEY/OBTC_EXCHANGE_API_SECRET"
            )
        self.api_key = os.environ.get("OBTC_EXCHANGE_API_KEY")
        self.api_secret = os.environ.get("OBTC_EXCHANGE_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise RuntimeError(
                "missing live credentials: export OBTC_EXCHANGE_API_KEY and "
                "OBTC_EXCHANGE_API_SECRET"
            )
        import ccxt.async_support as ccxt_async

        exchange_id = config.get("exchange", "binance")
        exchange_cls = getattr(ccxt_async, exchange_id)
        params = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
        }
        params.update(config.get("exchange_params", {}) or {})
        self.exchange = exchange_cls(params)
        self.orders: list[Order] = []

    def open_orders(self) -> list[Order]:
        return [o for o in self.orders
                if o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)]

    async def place_order(self, signal, size: float) -> Order:
        side = "buy" if signal.is_long else "sell"
        resp = await self.exchange.create_market_order(signal.symbol, side, float(size))
        await self.exchange.load_markets()
        px = float(resp.get("average") or resp.get("price") or signal.entry_price)
        filled = float(resp.get("filled") or resp.get("amount") or size)
        fee_cost = 0.0
        fee = resp.get("fee") or {}
        if fee.get("cost"):
            fee_cost = float(fee["cost"])
        now = utcnow()
        order = Order(
            id=str(resp.get("id") or new_id("live-order")),
            signal=signal,
            size=filled,
            status=OrderStatus.OPEN,
            entry_fill=px,
            stop_price=signal.stop_price,
            opened_at=now,
            remaining=filled,
            fees_paid=fee_cost,
            fills=[Fill(price=px, size=filled, at=now, kind="ENTRY")],
            notes={"exchange": self.exchange.id, "raw_id": resp.get("id")},
        )
        self.orders.append(order)
        return order

    async def update(self, bar) -> list[Order]:
        """Poll the exchange for open-order status changes."""
        changed: list[Order] = []
        for order in self.open_orders():
            raw_id = order.notes.get("raw_id")
            if not raw_id:
                continue
            try:
                resp = await self.exchange.fetch_order(raw_id, order.signal.symbol)
            except Exception as exc:  # network/exchange hiccup: try next bar
                log.warning("live poll failed for %s: %s", raw_id, exc)
                continue
            status = str(resp.get("status") or "").lower()
            if status in ("closed", "filled") and order.status != OrderStatus.CLOSED:
                filled = float(resp.get("filled") or order.size)
                avg = float(resp.get("average") or order.entry_fill or 0.0)
                if order.is_long:
                    pnl = (avg - order.entry_fill) * filled
                else:
                    pnl = (order.entry_fill - avg) * filled
                order.fills.append(Fill(price=avg, size=filled, at=utcnow(), kind="TARGET"))
                order.realized_pnl += pnl
                order.remaining = 0.0
                order.status = OrderStatus.CLOSED
                order.closed_at = utcnow()
                changed.append(order)
            elif status in ("canceled", "cancelled") and order.status != OrderStatus.CANCELLED:
                order.status = OrderStatus.CANCELLED
                order.closed_at = utcnow()
                changed.append(order)
        return changed

    async def close_all(self, price: float, at, reason: str = "EOD") -> list[Order]:
        closed: list[Order] = []
        for order in self.open_orders():
            side = "sell" if order.is_long else "buy"
            try:
                resp = await self.exchange.create_market_order(
                    order.signal.symbol, side, order.remaining)
                px = float(resp.get("average") or resp.get("price") or price)
            except Exception as exc:
                log.warning("live close failed for %s: %s", order.id, exc)
                continue
            if order.is_long:
                pnl = (px - order.entry_fill) * order.remaining
            else:
                pnl = (order.entry_fill - px) * order.remaining
            order.fills.append(Fill(price=px, size=order.remaining, at=at, kind=reason))
            order.realized_pnl += pnl
            order.remaining = 0.0
            order.status = OrderStatus.CLOSED
            order.closed_at = at
            closed.append(order)
        return closed

    async def aclose(self) -> None:
        await self.exchange.close()


class AsyncExecutor:
    """Risk-gated submission + bar delegation over any broker exposing
    place_order / update / close_all / open_orders."""

    def __init__(self, broker, risk: RiskManager):
        self.broker = broker
        self.risk = risk

    def open_orders(self) -> list[Order]:
        return self.broker.open_orders()

    def open_chain_ids(self) -> set:
        return {o.signal.chain_id for o in self.open_orders()}

    async def submit(self, setup: ScoredSetup, equity: float) -> Order | None:
        signal = setup.signal
        ok, reason = self.risk.can_open(
            signal.chain_id, len(self.open_orders()), self.risk.day_pnl,
            equity, self.open_chain_ids(),
        )
        if not ok:
            log.info("order rejected: %s", reason)
            return None
        size = self.risk.position_size(signal, equity)
        if size <= 0:
            log.info("order rejected: non-positive size")
            return None
        return await self.broker.place_order(signal, size)

    async def on_bar(self, bar) -> list[Order]:
        return await self.broker.update(bar)
