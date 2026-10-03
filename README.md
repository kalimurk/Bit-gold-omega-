# OBTC — Order Block Trading Chain

OBTC detects institutional order-flow structure (order blocks, fair value gaps,
liquidity sweeps, breakers, market-structure shifts) across multiple
timeframes, links the structures into a validated full chain, scores trading
setups through a confluence engine, and executes the A+ ones — in backtest,
paper, or live mode.

## Architecture

```
 bars (1m/5m/15m/1H/4H/1D)
   │
   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  data        │──▶│  detection   │──▶│  chain       │
│  fetch/cache │   │  OB/FVG/     │   │  builder:    │
│              │   │  sweep/BOS    │   │  HTF→MTF→LTF │
└──────────────┘   └──────────────┘   └──────┬───────┘
                                            │ FullChain (VALID/TRIGGERED)
                                            ▼
                                     ┌──────────────┐   ┌──────────────┐
                                     │  strategy    │──▶│  confluence  │
                                     │  S1–S7       │   │  score ≥ 80  │
                                     └──────────────┘   └──────┬───────┘
                                                              │ A+ ScoredSetup
                                                              ▼
                                                       ┌──────────────┐
                                                       │  execution   │
                                                       │  risk +      │
                                                       │  paper/live  │
                                                       └──────┬───────┘
                                                              │
                    ┌──────────────────┬──────────────────────┼──────────────────┐
                    ▼                  ▼                      ▼                  ▼
              backtest replay    live paper loop         dashboard         webhook alerts
              (ReplayEngine)     (run_live.py)           (FastAPI :8405)   (WebhookAlerter)
```

The replay engine drives the *same* call sequence as the live loop
(`chain_builder.update → strategy.check_setup → confluence.score →
executor.submit → executor.on_bar`), so backtest behaviour mirrors live
behaviour.

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# backtest (writes outputs/equity.csv, outputs/equity.png,
# outputs/trades.jsonl, outputs/screenshots/*.png)
PYTHONPATH=src python run_backtest.py

# paper trading (live bars, no real orders)
PYTHONPATH=src python run_live.py --mode paper

# dashboard (http://127.0.0.1:8405)
PYTHONPATH=src python run_dashboard.py
```

Docker:

```bash
docker compose --profile backtest up --build    # backtest
docker compose --profile paper up --build       # paper trading
docker compose --profile dashboard up --build   # dashboard on :8405
```

## Configuration

`config.yaml` holds every setting; any value can be overridden with an
`OBTC_`-prefixed environment variable. Nesting uses a double underscore
(`OBTC_EXECUTION__MODE=backtest`); a bare name such as `OBTC_SYMBOL` maps to
the matching field (`data.symbol`).

| Section     | Key                     | Default                                          | Meaning                                |
|-------------|-------------------------|--------------------------------------------------|----------------------------------------|
| data        | exchange                | coinbase                                         | ccxt exchange id                       |
| data        | symbol                  | BTC/USD                                          | market                                 |
| data        | timeframes              | 1m, 5m, 15m, 1H, 4H, 1D                          | replayed/live timeframes               |
| data        | history_days            | 30                                               | warmup history                         |
| detection   | displacement_atr_mult   | 1.5                                              | displacement threshold (ATR multiple)  |
| detection   | range_mult              | 1.2                                              | candle-range expansion multiple        |
| detection   | atr_period              | 14                                               | ATR lookback                           |
| detection   | volume_imbalance_mult   | 1.5                                              | volume imbalance multiple              |
| detection   | max_wick_pct            | 0.40                                             | max wick share for a valid OB          |
| strategy    | enabled                 | s1…s7                                            | active strategy modules                |
| confluence  | threshold               | 80.0                                             | minimum score for an A+ setup          |
| risk        | risk_per_trade_pct      | 1.0                                              | equity risked per trade (%)            |
| risk        | daily_loss_limit_pct    | 3.0                                              | daily stop (%)                         |
| risk        | max_concurrent          | 3                                                | max open positions                     |
| risk        | per_trade_risk_cap_pct  | 2.0                                              | hard cap per trade (%)                 |
| execution   | mode                    | paper                                            | paper \| backtest \| live              |
| execution   | fee_bps                 | 5.0                                              | commission per side                    |
| execution   | slippage_bps            | 2.0                                              | slippage model                         |
| execution   | starting_equity         | 10000.0                                          | account size                           |
| dashboard   | host / port             | 127.0.0.1 / 8405                                 | dashboard bind                         |
| alerts      | webhook_url             | null                                             | POST target for chain/setup alerts     |
| backtest    | output_dir              | outputs                                          | replay artifact directory              |
| cache       | dir / ttl_s             | .cache / 3600                                    | bar cache                              |

## Strategies

| ID | Name                      | Idea                                             |
|----|---------------------------|--------------------------------------------------|
| S1 | s1_liquidity_sweep_reversal | sweep of HTF liquidity + LTF reversal into the OB |
| S2 | s2_continuation           | BOS continuation riding the MTF leg              |
| S3 | s3_breaker_reclaim        | reclaimed breaker block as the trigger           |
| S4 | s4_mitigation_flip        | mitigated OB flip to the opposing side           |
| S5 | s5_turtle_soup            | failed breakout / turtle-soup fade               |
| S6 | s6_silver_bullet          | killzone-timed displacement entry                |
| S7 | s7_orderflow_stacking     | stacked OB + FVG confluence entry                |

## Testing

```bash
cd ~/workspace/obtc
PYTHONPATH=src python -m pytest tests/test_backtest.py \
  tests/test_dashboard_alerts.py tests/test_config.py -q
```

- `test_backtest.py` — deterministic synthetic 1m bars (seeded random walk
  with engineered impulse legs), resampled in-test to 5m/1H/4H; unit-tests
  metrics, trade log and screenshots; the wired end-to-end replay runs when
  the sibling `chain.builder`, `strategy.*`, `confluence.engine` and
  `execution.executor` modules exist, otherwise it skips cleanly via
  `pytest.importorskip`.
- `test_dashboard_alerts.py` — payload serialisation, FastAPI routes, and
  the webhook alerter against a local aiohttp test server (success, retry
  then success, give-up without raising).
- `test_config.py` — YAML loading, `OBTC_` env overlay (including nested
  `OBTC_EXECUTION__MODE`), and mode validation.

## Backtest outputs

```
outputs/
  equity.csv        # timestamp,equity per bar
  equity.png        # equity curve with drawdown shading
  trades.jsonl      # one JSON record per order
  screenshots/      # per-trade candle chart with OB zone, entry/stop/targets, fills
```
