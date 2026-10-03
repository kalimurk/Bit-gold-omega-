"""Trade log (JSONL) and per-trade chart screenshots."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from obtc.utils.logging import get_logger  # noqa: E402


def _iso(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _jsonable(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    return str(value)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _rationale_summary(rationale: Any) -> str:
    if not rationale:
        return ""
    if isinstance(rationale, str):
        return rationale[:300]
    if isinstance(rationale, dict):
        parts = []
        for key, val in rationale.items():
            text = str(val)
            parts.append(f"{key}={text[:60]}")
            if len(parts) >= 8:
                break
        return "; ".join(parts)
    return str(rationale)[:300]


class TradeLogger:
    """Appends one JSON record per order to ``trades.jsonl`` and renders
    per-trade chart screenshots."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "trades.jsonl"
        self._handle = open(self.path, "a", encoding="utf-8")
        self.logger = get_logger("obtc.backtest.tradelog")

    def log(self, order) -> None:
        signal = getattr(order, "signal", None)
        rationale = getattr(signal, "rationale", None) if signal is not None else None
        record = {
            "order_id": getattr(order, "id", None),
            "strategy": getattr(signal, "strategy_name", None),
            "symbol": getattr(signal, "symbol", None),
            "direction": _enum_value(getattr(signal, "direction", None)),
            "timeframe": getattr(signal, "timeframe", None),
            "chain_id": getattr(signal, "chain_id", None),
            "entry": getattr(signal, "entry_price", None),
            "stop": getattr(signal, "stop_price", None),
            "target_2r": getattr(signal, "target_2r", None),
            "target_3r": getattr(signal, "target_3r", None),
            "size": getattr(order, "size", None),
            "entry_fill": getattr(order, "entry_fill", None),
            "fills": _jsonable(getattr(order, "fills", None) or []),
            "realized_pnl": getattr(order, "realized_pnl", 0.0),
            "fees_paid": getattr(order, "fees_paid", 0.0),
            "status": _enum_value(getattr(order, "status", None)),
            "opened_at": _iso(getattr(order, "opened_at", None)),
            "closed_at": _iso(getattr(order, "closed_at", None)),
            "rationale_summary": _rationale_summary(rationale),
        }
        self._handle.write(json.dumps(record, default=str) + "\n")
        self._handle.flush()

    def finalize(self) -> None:
        try:
            if not self._handle.closed:
                self._handle.flush()
        except Exception as exc:  # never break a backtest on log flush
            self.logger.warning("trade log finalize failed: %s", exc)

    def close(self) -> None:
        try:
            if not self._handle.closed:
                self._handle.close()
        except Exception as exc:
            self.logger.warning("trade log close failed: %s", exc)

    # ------------------------------------------------------------------
    # screenshots
    # ------------------------------------------------------------------
    def screenshot(self, order, bars, path: str) -> None:
        """Render up to the last 80 bars ending at the order's close/open.

        ``bars`` is a DataFrame with timestamp/open/high/low/close/volume.
        Draws candles, the OB zone (from ``signal.rationale`` ``ob_high`` /
        ``ob_low`` when present), entry/stop/target lines and fill markers.
        """
        import pandas as pd

        try:
            self._render_screenshot(order, bars, path, pd)
        except Exception as exc:  # a chart must never break the backtest
            self.logger.warning("screenshot failed for order %s: %s",
                                getattr(order, "id", "?"), exc)

    def _render_screenshot(self, order, bars, path: str, pd) -> None:
        signal = getattr(order, "signal", None)
        rationale = getattr(signal, "rationale", None) or {}
        if not isinstance(rationale, dict):
            rationale = {}

        frame = bars.copy()
        if frame.empty:
            raise ValueError("no bars available for screenshot")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

        anchor = getattr(order, "closed_at", None) or getattr(order, "opened_at", None)
        if anchor is not None:
            window = frame[frame["timestamp"] <= pd.Timestamp(anchor)]
            if not window.empty:
                frame = window
        frame = frame.tail(80).reset_index(drop=True)

        xs = list(range(len(frame)))
        opens = frame["open"].astype(float).tolist()
        highs = frame["high"].astype(float).tolist()
        lows = frame["low"].astype(float).tolist()
        closes = frame["close"].astype(float).tolist()
        times = pd.to_datetime(frame["timestamp"], utc=True).tolist()

        fig, ax = plt.subplots(figsize=(12, 6))
        for x, o, h, low, c in zip(xs, opens, highs, lows, closes):
            color = "#2ea043" if c >= o else "#d1242f"
            ax.plot([x, x], [low, h], color=color, linewidth=1.0)
            body_low, body_high = (o, c) if c >= o else (c, o)
            ax.add_patch(Rectangle((x - 0.35, body_low), 0.7,
                                   max(body_high - body_low, 1e-9),
                                   facecolor=color, edgecolor=color, alpha=0.9))

        # OB zone from rationale (skipped silently when keys are absent)
        ob_high = rationale.get("ob_high")
        ob_low = rationale.get("ob_low")
        if ob_high is not None and ob_low is not None:
            try:
                ax.axhspan(float(ob_low), float(ob_high), xmin=0.0, xmax=1.0,
                           color="#1f6feb", alpha=0.18, label="OB zone")
            except (TypeError, ValueError):
                pass

        def _hline(price, color, style, label):
            if price is None:
                return
            try:
                ax.axhline(float(price), color=color, linestyle=style,
                           linewidth=1.2, label=label)
            except (TypeError, ValueError):
                pass

        _hline(getattr(signal, "entry_price", None), "#1f6feb", "-", "entry")
        _hline(getattr(order, "stop_price", None) or
               getattr(signal, "stop_price", None), "#d1242f", "-", "stop")
        _hline(getattr(signal, "target_2r", None), "#2ea043", "--", "target 2R")
        _hline(getattr(signal, "target_3r", None), "#2ea043", ":", "target 3R")

        # fill markers mapped to the nearest bar
        for fill in getattr(order, "fills", None) or []:
            price = getattr(fill, "price", None)
            at = getattr(fill, "at", None)
            if price is None or at is None:
                continue
            try:
                at_ts = pd.Timestamp(at)
                idx = min(range(len(times)),
                          key=lambda i: abs((times[i] - at_ts).total_seconds()))
                ax.scatter([idx], [float(price)], s=70, marker="v",
                           color="#a371f7", zorder=5)
            except (TypeError, ValueError):
                continue

        strategy = getattr(signal, "strategy_name", "?")
        direction = _enum_value(getattr(signal, "direction", "?"))
        pnl = getattr(order, "realized_pnl", 0.0)
        ax.set_title(f"{strategy} {direction}  pnl={pnl:+.2f}")
        ax.set_xlabel("Bar")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
