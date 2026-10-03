"""Backtesting: deterministic replay, metrics, trade log, screenshots."""
from obtc.backtest.metrics import compute_metrics, save_equity_csv, save_equity_plot
from obtc.backtest.replay import BacktestResult, ReplayEngine
from obtc.backtest.trade_log import TradeLogger

__all__ = [
    "BacktestResult",
    "ReplayEngine",
    "TradeLogger",
    "compute_metrics",
    "save_equity_csv",
    "save_equity_plot",
]
