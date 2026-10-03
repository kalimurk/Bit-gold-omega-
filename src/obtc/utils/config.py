"""Application configuration.

Single source of config truth: pydantic models with sensible defaults,
optional YAML file, and ``OBTC_``-prefixed environment variable overrides.
Nesting uses a double underscore, e.g. ``OBTC_EXECUTION__MODE=backtest``.
A bare name such as ``OBTC_SYMBOL`` maps to the matching sub-config field
(``data.symbol``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    exchange: str = "coinbase"
    symbol: str = "BTC/USD"
    timeframes: list[str] = ["1m", "5m", "15m", "1H", "4H", "1D"]
    history_days: int = 30


class DetectionConfig(BaseModel):
    displacement_atr_mult: float = 1.5
    range_mult: float = 1.2
    atr_period: int = 14
    volume_imbalance_mult: float = 1.5
    max_wick_pct: float = 0.40


class StrategyConfig(BaseModel):
    enabled: list[str] = [
        "s1_liquidity_sweep_reversal",
        "s2_continuation",
        "s3_breaker_reclaim",
        "s4_mitigation_flip",
        "s5_turtle_soup",
        "s6_silver_bullet",
        "s7_orderflow_stacking",
    ]


class ConfluenceConfig(BaseModel):
    threshold: float = 80.0


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = 1.0
    daily_loss_limit_pct: float = 3.0
    max_concurrent: int = 3
    per_trade_risk_cap_pct: float = 2.0


class ExecutionConfig(BaseModel):
    mode: str = "paper"
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    starting_equity: float = 10000.0

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        allowed = {"paper", "backtest", "live"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}, got {v!r}")
        return v


class DashboardConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8405


class AlertsConfig(BaseModel):
    webhook_url: Optional[str] = None


class BacktestConfig(BaseModel):
    output_dir: str = "outputs"


class CacheConfig(BaseModel):
    dir: str = ".cache"
    ttl_s: int = 3600


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    confluence: ConfluenceConfig = Field(default_factory=ConfluenceConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


_SECTION_MODELS: dict[str, type[BaseModel]] = {
    "data": DataConfig,
    "detection": DetectionConfig,
    "strategy": StrategyConfig,
    "confluence": ConfluenceConfig,
    "risk": RiskConfig,
    "execution": ExecutionConfig,
    "dashboard": DashboardConfig,
    "alerts": AlertsConfig,
    "backtest": BacktestConfig,
    "cache": CacheConfig,
}

# Single-part env names map to (section, field), e.g. OBTC_SYMBOL -> data.symbol.
_FIELD_LOOKUP: dict[str, tuple[str, str, type[BaseModel]]] = {}
for _section, _model in _SECTION_MODELS.items():
    for _fname in _model.model_fields:
        _FIELD_LOOKUP.setdefault(_fname.upper(), (_section, _fname, _model))


def _coerce(raw: str, model: type[BaseModel], field_name: str) -> Any:
    """Best-effort scalar coercion; lists come in as comma-separated strings."""
    annotation = model.model_fields[field_name].annotation
    default = model.model_fields[field_name].default
    value: Any = raw.strip()
    if isinstance(default, list) or "list" in str(annotation):
        return [part.strip() for part in value.split(",") if part.strip()]
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_env_overrides(data: dict) -> None:
    for key, raw in os.environ.items():
        if not key.startswith("OBTC_"):
            continue
        rest = key[len("OBTC_"):]
        if "__" in rest:
            section, _, field = rest.partition("__")
            section = section.lower()
            field = field.lower()
            model = _SECTION_MODELS.get(section)
            if model is None or field not in model.model_fields:
                continue
        else:
            hit = _FIELD_LOOKUP.get(rest)
            if hit is None:
                continue
            section, field, model = hit
        section_dict = data.get(section)
        if not isinstance(section_dict, dict):
            section_dict = {}
            data[section] = section_dict
        section_dict[field] = _coerce(raw, model, field)


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load configuration from YAML (when present) with OBTC_ env overrides."""
    data: dict = {}
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            data = loaded
    _apply_env_overrides(data)
    return AppConfig.model_validate(data)
