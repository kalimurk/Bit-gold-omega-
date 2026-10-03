"""Backtest performance metrics and equity-curve artifacts.

All functions are deterministic and handle the zero-trade case without
raising.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

METRIC_KEYS = (
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
)


def _initial_risk(order) -> float:
    """Per-order initial risk in account currency.

    Prefers ``order.notes["risk_amt"]`` (recorded by the executor); falls
    back to ``|entry - stop| * size``; returns 0.0 when neither is available.
    """
    notes = getattr(order, "notes", None) or {}
    if isinstance(notes, dict) and notes.get("risk_amt"):
        try:
            return float(notes["risk_amt"])
        except (TypeError, ValueError):
            pass
    signal = getattr(order, "signal", None)
    entry = getattr(order, "entry_fill", None)
    if entry is None and signal is not None:
        entry = getattr(signal, "entry_price", None)
    stop = getattr(order, "stop_price", None)
    if stop is None and signal is not None:
        stop = getattr(signal, "stop_price", None)
    size = getattr(order, "size", 0.0) or 0.0
    try:
        if entry is not None and stop is not None:
            return abs(float(entry) - float(stop)) * float(size)
    except (TypeError, ValueError):
        pass
    return 0.0


def compute_metrics(orders: list, equity: pd.DataFrame, starting_equity: float) -> dict:
    """Compute the standard backtest scorecard.

    ``orders`` are executor ``Order`` objects (duck-typed), ``equity`` is a
    DataFrame with ``timestamp``/``equity`` columns.
    """
    metrics: dict = {key: 0.0 for key in METRIC_KEYS}
    metrics["num_trades"] = 0

    pnls = [float(getattr(order, "realized_pnl", 0.0) or 0.0) for order in orders]
    n_trades = len(pnls)
    metrics["num_trades"] = n_trades
    if n_trades == 0:
        return metrics

    wins = [p for p in pnls if p > 0.0]
    losses = [p for p in pnls if p < 0.0]
    metrics["wins"] = len(wins)
    metrics["losses"] = len(losses)
    metrics["win_rate"] = len(wins) / n_trades

    total_pnl = float(sum(pnls))
    metrics["total_pnl"] = total_pnl
    metrics["expectancy"] = total_pnl / n_trades
    metrics["avg_win"] = float(sum(wins) / len(wins)) if wins else 0.0
    metrics["avg_loss"] = float(sum(losses) / len(losses)) if losses else 0.0

    gross_profit = float(sum(wins))
    gross_loss = float(abs(sum(losses)))
    if gross_loss > 0.0:
        metrics["profit_factor"] = gross_profit / gross_loss
    elif gross_profit > 0.0:
        metrics["profit_factor"] = math.inf
    else:
        metrics["profit_factor"] = 0.0

    r_multiples = []
    for order, pnl in zip(orders, pnls):
        risk = _initial_risk(order)
        if risk > 0.0:
            r_multiples.append(pnl / risk)
    metrics["avg_r"] = float(sum(r_multiples) / len(r_multiples)) if r_multiples else 0.0

    if equity is not None and not equity.empty and "equity" in equity.columns:
        eq = pd.to_numeric(equity["equity"], errors="coerce").dropna()
        if len(eq) > 0:
            running_peak = eq.cummax()
            drawdown = running_peak - eq
            metrics["max_drawdown"] = float(drawdown.max())
            safe_peak = running_peak.replace(0.0, float("nan"))
            dd_pct = (drawdown / safe_peak * 100.0).dropna()
            metrics["max_drawdown_pct"] = float(dd_pct.max()) if len(dd_pct) else 0.0
            try:
                start = float(starting_equity)
                last = float(eq.iloc[-1])
                metrics["total_return_pct"] = (
                    (last - start) / start * 100.0 if start != 0.0 else 0.0
                )
            except (TypeError, ValueError):
                metrics["total_return_pct"] = 0.0

    return metrics


def save_equity_csv(equity_df: pd.DataFrame, path: str) -> None:
    frame = equity_df.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    frame.to_csv(path, index=False)


def save_equity_plot(equity_df: pd.DataFrame, path: str) -> None:
    """Equity curve with drawdown shading. Never raises on empty input."""
    fig, ax = plt.subplots(figsize=(12, 5))
    try:
        if equity_df is not None and not equity_df.empty and "equity" in equity_df.columns:
            eq = pd.to_numeric(equity_df["equity"], errors="coerce").dropna().reset_index(drop=True)
            if len(eq):
                peak = eq.cummax()
                ax.plot(eq.index, eq.values, color="#1f6feb", linewidth=1.5, label="Equity")
                ax.plot(peak.index, peak.values, color="#6e7681", linewidth=1.0,
                        linestyle="--", label="Peak")
                ax.fill_between(eq.index, eq.values, peak.values,
                                where=(peak.values > eq.values),
                                color="#d1242f", alpha=0.25, label="Drawdown")
                ax.legend(loc="upper left")
        ax.set_title("Equity curve")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Equity")
        ax.grid(True, alpha=0.3)
    finally:
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
