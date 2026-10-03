"""Config loading tests: YAML file, OBTC_ env overlay, validation."""
from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from obtc.utils.config import AppConfig, load_config


def _write_yaml(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def _clear_obtc_env(monkeypatch):
    for key in [k for k in os.environ if k.startswith("OBTC_")]:
        monkeypatch.delenv(key, raising=False)


def test_defaults_without_file(tmp_path, monkeypatch):
    # Keep the real environment from leaking OBTC_ vars into the test.
    _clear_obtc_env(monkeypatch)
    cfg = load_config(str(tmp_path / "missing.yaml"))
    assert isinstance(cfg, AppConfig)
    assert cfg.data.symbol == "BTC/USD"
    assert cfg.execution.mode == "paper"
    assert cfg.dashboard.port == 8405
    assert cfg.confluence.threshold == 80.0
    assert cfg.alerts.webhook_url is None


def test_yaml_values_applied(tmp_path, monkeypatch):
    _clear_obtc_env(monkeypatch)
    path = _write_yaml(
        tmp_path / "config.yaml",
        "data:\n"
        "  symbol: ETH/USD\n"
        "execution:\n"
        "  mode: backtest\n"
        "  starting_equity: 5000.0\n",
    )
    cfg = load_config(path)
    assert cfg.data.symbol == "ETH/USD"
    assert cfg.execution.mode == "backtest"
    assert cfg.execution.starting_equity == 5000.0
    # untouched sections keep defaults
    assert cfg.dashboard.port == 8405
    assert cfg.risk.max_concurrent == 3


def test_env_overlay_single_part(tmp_path, monkeypatch):
    path = _write_yaml(tmp_path / "config.yaml", "data:\n  symbol: BTC/USD\n")
    monkeypatch.setenv("OBTC_SYMBOL", "SOL/USD")
    cfg = load_config(path)
    assert cfg.data.symbol == "SOL/USD"


def test_env_overlay_nested_double_underscore(tmp_path, monkeypatch):
    path = _write_yaml(tmp_path / "config.yaml", "{}\n")
    monkeypatch.setenv("OBTC_EXECUTION__MODE", "backtest")
    monkeypatch.setenv("OBTC_DASHBOARD__PORT", "9000")
    cfg = load_config(path)
    assert cfg.execution.mode == "backtest"
    assert cfg.dashboard.port == 9000


def test_env_overrides_yaml(tmp_path, monkeypatch):
    path = _write_yaml(tmp_path / "config.yaml", "data:\n  symbol: BTC/USD\n")
    monkeypatch.setenv("OBTC_SYMBOL", "ETH/USD")
    cfg = load_config(path)
    assert cfg.data.symbol == "ETH/USD"


def test_invalid_mode_raises(tmp_path, monkeypatch):
    _clear_obtc_env(monkeypatch)
    path = _write_yaml(tmp_path / "config.yaml", "execution:\n  mode: hyperdrive\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_invalid_mode_via_env_raises(tmp_path, monkeypatch):
    path = _write_yaml(tmp_path / "config.yaml", "{}\n")
    monkeypatch.setenv("OBTC_EXECUTION__MODE", "turbo")
    with pytest.raises(ValidationError):
        load_config(path)
