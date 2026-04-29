# Simulation design

This file records the generator parameters used by `src/gate_ablation_study.py`.

## Replication design

- Base seed: `42`
- Replications: `100`
- Replication seeds: `42` through `141`
- Synthetic symbols per replication: `20`
- Strategy templates: `50`
- Regimes: `TREND`, `RANGE`, `BREAKOUT`, `CHOP`
- Timeframes: `M5`, `M15`, `M30`, `H1`, `H4`, `D1`
- Seed `42` produces `6,625` candidates.

For each strategy-regime-timeframe combination, the generator samples `n_active` symbols uniformly from `3` to `min(8, n_symbols)`, then creates one candidate for each active symbol.

## Latent quality model

For each candidate:

```text
quality ~ Normal(0, 1)
q = quality + regime_bonus[regime]
```

Regime bonuses:

| Regime | Bonus |
| --- | ---: |
| TREND | 0.30 |
| BREAKOUT | 0.15 |
| RANGE | -0.10 |
| CHOP | -0.40 |

Base validation trade counts by timeframe:

| Timeframe | Base trades |
| --- | ---: |
| M5 | 80 |
| M15 | 55 |
| M30 | 40 |
| H1 | 28 |
| H4 | 16 |
| D1 | 8 |

Candidate metrics are generated as:

```text
val_trades = max(1, int(Normal(base_trades, 0.4 * base_trades)))
val_return = Normal(3.0 + 6.0*q, 8.0)
val_drawdown = clamp(Normal(14.0 - 3.0*q, 6.0), 1.0, 50.0)
val_profit_factor = clamp(Normal(1.05 + 0.3*q, 0.35), 0.3, 4.0)
val_sharpe = clamp(Normal(0.15 + 0.25*q, 0.3), -1.0, 3.0)
val_win_rate = clamp(Normal(48.0 + 5.0*q, 10.0), 15.0, 85.0)
val_return_dd_ratio = val_return / val_drawdown

train_return = Normal(1.2 * val_return + 2.0, 5.0)
train_profit_factor = clamp(Normal(1.1 * val_profit_factor, 0.25), 0.3, 5.0)
train_drawdown = clamp(Normal(0.9 * val_drawdown, 4.0), 1.0, 45.0)

gap = abs(train_return - val_return)
gap_penalty = 1 / (1 + 0.05 * gap)
rho = clamp((val_profit_factor / (train_profit_factor + 0.1)) * gap_penalty, 0.0, 2.0)
```

Values are rounded inside the candidate record exactly as implemented in the Python script.

## Empirical gate profile

The aligned paper run uses the `historical` threshold profile:

| Gate | Predicate |
| --- | --- |
| G1 | `val_trades >= 5` |
| G2 | `val_drawdown < 20%` |
| G3 | `train_return >= 0%` |
| G4 | `val_return >= 5%` |
| G5 | `val_return / val_drawdown >= 1.0` |
| G6 | `val_profit_factor >= 1.0` |
| G7 | `rho >= 0.75` |

The active gate presets are:

| Preset | Active gates |
| --- | --- |
| all_gates | G1-G7 |
| no_gates | none |
| minimal | G1, G2 |
| drawdown_only | G2 |
| profitability_only | G3, G6 |
| quality_focused | G2, G5, G6 |
| robustness_focused | G1, G5, G7 |

## Statistical analysis

Each preset is applied to the same candidates within a replication. Reported simulation tables use medians across the `100` paired replications. The script also computes paired Wilcoxon signed-rank tests versus `all_gates`, Bonferroni correction within each metric family, and Friedman omnibus tests for median robustness and FAR.

