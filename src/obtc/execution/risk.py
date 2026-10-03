"""Position sizing, exposure guards, daily loss limits, breakeven trailing."""
from __future__ import annotations

from datetime import datetime, timezone

from obtc.chain.models import Order, utcnow


class RiskManager:
    """Pre-trade risk gate and post-trade day-PnL accounting.

    - position_size: risk_per_trade_pct of equity (capped by
      per_trade_risk_cap_pct), divided by the signal's risk distance.
    - can_open: blocks when the daily loss limit is hit, max concurrent
      positions are open, or this chain already has an open position.
    - register_close: accrues day PnL; the day rolls automatically on the
      UTC date of ``at``.
    """

    def __init__(self, risk_per_trade_pct: float = 1.0,
                 daily_loss_limit_pct: float = 3.0,
                 max_concurrent: int = 3,
                 per_trade_risk_cap_pct: float = 2.0):
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.daily_loss_limit_pct = float(daily_loss_limit_pct)
        self.max_concurrent = int(max_concurrent)
        self.per_trade_risk_cap_pct = float(per_trade_risk_cap_pct)
        self.day_pnl = 0.0
        self.day_start_equity = 0.0
        self._day = None
        self._halted = False

    # ------------------------------------------------------------- sizing
    def position_size(self, signal, equity: float) -> float:
        """Units to trade. 0.0 when the signal risk or equity is non-positive."""
        risk = signal.risk
        if risk is None or risk <= 0 or equity is None or equity <= 0:
            return 0.0
        risk_budget = min(equity * self.risk_per_trade_pct / 100.0,
                          equity * self.per_trade_risk_cap_pct / 100.0)
        return max(0.0, risk_budget / risk)

    # -------------------------------------------------------------- gating
    def set_day_start(self, equity: float) -> None:
        """Open a fresh trading day with a known starting equity."""
        self.day_start_equity = float(equity)
        self.day_pnl = 0.0
        self._day = utcnow().date()
        self._halted = False

    def register_close(self, pnl: float, at: datetime, equity: float | None = None) -> None:
        """Accrue a closed-trade PnL. Rolls to a new day automatically; the
        halt latch resets on roll. Pass ``equity`` to rebase day_start_equity
        on the roll."""
        day = at.astimezone(timezone.utc).date() if at.tzinfo else at.date()
        if self._day is None or day != self._day:
            self._day = day
            self.day_pnl = 0.0
            self._halted = False
            if equity is not None:
                self.day_start_equity = float(equity)
        self.day_pnl += float(pnl)
        if self._loss_limit_hit():
            self._halted = True

    def _loss_limit_hit(self) -> bool:
        if self.day_start_equity <= 0:
            return False
        limit = self.daily_loss_limit_pct / 100.0 * self.day_start_equity
        return self.day_pnl <= -limit

    @property
    def halted(self) -> bool:
        """True once the daily loss limit is hit (latched for the day)."""
        return self._halted or self._loss_limit_hit()

    def can_open(self, chain_id: str, open_count: int, day_pnl: float,
                 equity: float, open_chain_ids: set | None = None) -> tuple:
        """Return (allowed, reason)."""
        if self.halted:
            return (False, "daily loss limit")
        if open_count >= self.max_concurrent:
            return (False, f"max concurrent ({self.max_concurrent}) reached")
        if open_chain_ids and chain_id in open_chain_ids:
            return (False, "chain already open")
        return (True, "ok")

    # ------------------------------------------------------------ trailing
    @staticmethod
    def trail_to_breakeven(order: Order, mss_confirmed: bool) -> bool:
        """Move the stop to the entry fill once an MSS confirms the trade.

        Returns True when the stop moved.
        """
        if not mss_confirmed:
            return False
        if order.entry_fill is None or order.stop_price is None:
            return False
        if order.stop_price == order.entry_fill:
            return False
        order.stop_price = order.entry_fill
        order.notes["trailed_to_breakeven"] = True
        order.notes["breakeven_at"] = utcnow().isoformat()
        return True
