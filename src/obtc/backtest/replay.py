"""Deterministic event replay engine.

Feeds a merged chronological bar stream through the chain builder, strategy
checks, confluence scoring and the executor — the same call order the live
loop uses, so backtest behaviour mirrors live behaviour.

Sibling components (chain builder, strategies, confluence, executor) are
duck-typed on purpose: this module never imports them at module level so it
keeps working while they are built in parallel.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import pandas as pd

from obtc.backtest import metrics as metrics_mod
from obtc.backtest.trade_log import TradeLogger
from obtc.chain.models import ChainState, utcnow
from obtc.utils.logging import get_logger

if TYPE_CHECKING:  # annotations only; runtime stays duck-typed
    from obtc.chain.models import Order

# Lower timeframes stream first when timestamps tie, so LTF structure is
# visible before the HTF bar that contains it closes.
_TIMEFRAME_RANK = {"1m": 0, "5m": 1, "15m": 2, "1H": 3, "4H": 4, "1D": 5}

_REQUIRED_BAR_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
_ACTIVE_CHAIN_STATES = (ChainState.VALID, ChainState.TRIGGERED)


@dataclass
class BacktestResult:
    trades: list
    equity_curve: pd.DataFrame
    metrics: dict
    trade_log_path: str
    screenshot_dir: str
    start: datetime
    end: datetime
    output_dir: str


class ReplayEngine:
    """Replays multi-timeframe bars through the full chain pipeline."""

    def __init__(
        self,
        chain_builder,
        strategies: list,
        confluence,
        executor,
        output_dir: str = "outputs",
        logger=None,
    ) -> None:
        self.chain_builder = chain_builder
        self.strategies = list(strategies)
        self.confluence = confluence
        self.executor = executor
        self.output_dir = Path(output_dir)
        self.logger = logger or get_logger("obtc.backtest")
        self._fired: set[tuple[str, str]] = set()
        self._trades: list = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    async def _maybe_await(value):
        if inspect.isawaitable(value):
            return await value
        return value

    def _broker(self):
        return getattr(self.executor, "broker", None)

    def _equity(self) -> float:
        broker = self._broker()
        if broker is None:
            return 0.0
        equity = getattr(broker, "equity", None)
        if callable(equity):
            equity = equity()
        try:
            return float(equity)
        except (TypeError, ValueError):
            return 0.0

    def _merged_stream(self, bars: dict[str, pd.DataFrame], symbol: str) -> list[dict]:
        """Merge all timeframe DataFrames into one chronological bar stream.

        Stable sort by (timestamp, timeframe rank): lower timeframes come
        first on timestamp ties. Fully deterministic — no randomness.
        """
        rows: list[tuple[Any, int, str, Any]] = []
        for timeframe, frame in bars.items():
            missing = [c for c in _REQUIRED_BAR_COLUMNS if c not in frame.columns]
            if missing:
                raise ValueError(
                    f"bars[{timeframe!r}] missing columns: {missing}"
                )
            rank = _TIMEFRAME_RANK.get(timeframe, 99)
            stamps = pd.to_datetime(frame["timestamp"], utc=True)
            for pos in range(len(frame)):
                rows.append((stamps.iloc[pos], rank, timeframe, pos, frame))
        rows.sort(key=lambda r: (r[0], r[1], r[3]))
        stream: list[dict] = []
        for stamp, _rank, timeframe, pos, frame in rows:
            row = frame.iloc[pos]
            ts = stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp
            stream.append(
                {
                    "timestamp": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "timeframe": timeframe,
                    "symbol": symbol,
                }
            )
        return stream

    # ------------------------------------------------------------------
    # main run
    # ------------------------------------------------------------------
    async def run(
        self,
        bars: dict[str, pd.DataFrame],
        progress: Optional[Callable[[int, int], None]] = None,
        symbol: str = "BTC/USD",
    ) -> BacktestResult:
        if not bars:
            raise ValueError("bars is empty")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        screenshots_dir = self.output_dir / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        stream = self._merged_stream(bars, symbol)
        if not stream:
            raise ValueError("bars contain no rows")
        total = len(stream)
        self.logger.info("replaying %d bars across %s", total, sorted(bars))

        equity_points: list[dict] = []
        last_bar: Optional[dict] = None
        for index, bar in enumerate(stream):
            last_bar = bar
            await self._maybe_await(self.chain_builder.update(bar))
            for chain in self.chain_builder.get_active_chains():
                if chain.state not in _ACTIVE_CHAIN_STATES:
                    continue
                for strategy in self.strategies:
                    key = (strategy.name, chain.id)
                    if key in self._fired:
                        continue
                    signal = await self._maybe_await(strategy.check_setup(chain))
                    if signal is None:
                        continue
                    setup = await self._maybe_await(
                        self.confluence.score(
                            signal,
                            chain,
                            {"now": bar["timestamp"], "strategy": strategy},
                        )
                    )
                    signal.score = setup.score
                    if setup.is_a_plus:
                        order = await self._maybe_await(
                            self.executor.submit(setup, self._equity())
                        )
                        if order is not None:
                            self._fired.add(key)
                            self._trades.append(order)
            await self._maybe_await(self.executor.on_bar(bar))
            equity_points.append(
                {"timestamp": bar["timestamp"], "equity": self._equity()}
            )
            if progress is not None:
                progress(index + 1, total)

        # End-of-data: flatten anything still open.
        broker = self._broker()
        close_all = getattr(broker, "close_all", None) if broker is not None else None
        if callable(close_all) and last_bar is not None:
            await self._maybe_await(
                close_all(last_bar["close"], last_bar["timestamp"], reason="EOD")
            )
            equity_points.append(
                {"timestamp": last_bar["timestamp"], "equity": self._equity()}
            )

        equity_df = pd.DataFrame(equity_points, columns=["timestamp", "equity"])
        starting_equity = (
            float(equity_df["equity"].iloc[0]) if not equity_df.empty else 0.0
        )
        metrics = metrics_mod.compute_metrics(
            self._trades, equity_df, starting_equity
        )
        self.logger.info(
            "replay done: %d trades, pnl=%+.2f, win_rate=%.1f%%",
            metrics["num_trades"],
            metrics["total_pnl"],
            metrics["win_rate"] * 100.0,
        )

        # Artifacts: trade log, per-trade screenshots, equity CSV + PNG.
        trade_logger = TradeLogger(str(self.output_dir))
        try:
            for order in self._trades:
                trade_logger.log(order)
            trade_logger.finalize()
            first_frame = next(iter(bars.values()))
            for order in self._trades:
                signal = getattr(order, "signal", None)
                trade_tf = getattr(signal, "timeframe", None) if signal else None
                frame = bars.get(trade_tf, first_frame)
                safe_id = "".join(
                    ch if ch.isalnum() or ch in "-_" else "_" for ch in str(getattr(order, "id", "order"))
                )
                trade_logger.screenshot(
                    order, frame, str(screenshots_dir / f"{safe_id}.png")
                )
        finally:
            trade_logger.close()

        metrics_mod.save_equity_csv(equity_df, str(self.output_dir / "equity.csv"))
        metrics_mod.save_equity_plot(equity_df, str(self.output_dir / "equity.png"))

        return BacktestResult(
            trades=list(self._trades),
            equity_curve=equity_df,
            metrics=metrics,
            trade_log_path=str(self.output_dir / "trades.jsonl"),
            screenshot_dir=str(screenshots_dir),
            start=stream[0]["timestamp"],
            end=stream[-1]["timestamp"],
            output_dir=str(self.output_dir),
        )
