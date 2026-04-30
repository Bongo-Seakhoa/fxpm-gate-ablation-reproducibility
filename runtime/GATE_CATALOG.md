<!--
FXPM Formal Gate Catalog
Author: Bongo Bokoa Kosa | WD42M3
Institution: University of Debrecen, Hungary
Programme: BSc Engineering Management
Created: 2026-03-10
Research Log Entry: R-007
Purpose: Supervisor-readable governance documentation for the seven-gate
         validation pipeline. Written for an Engineering Management audience.
-->

# FXPM Validation Gate Catalog

## Purpose

This document formalises the seven validation gates that govern whether a
candidate strategy configuration is approved for live deployment. These gates
form the core of the FXPM governance framework and are the primary subject
of Study A (Validation Governance) in the dissertation programme.

Every candidate strategy must pass all active gates to be accepted. A failure
at any gate results in rejection. This is a strict stage-gate model: no
override mechanism exists, and no candidate can bypass the pipeline.

---

## Gate Summary

| Gate | Name | Criterion | Default Threshold | Governance Function |
|---|---|---|---|---|
| G1 | Minimum Trade Count | Candidate must produce a minimum number of trades during validation | 25 trades | Prevents approval of strategies with insufficient statistical evidence |
| G2 | Maximum Drawdown | Peak-to-trough equity decline must not exceed a threshold | 20% | Capital protection; prevents deployment of strategies with unacceptable loss potential |
| G3 | Train Profitability | Strategy must be profitable on the training data window | > 0% return | Basic viability check; eliminates candidates that fail even on in-sample data |
| G4 | Validation Return | Strategy must achieve a minimum return on out-of-sample data | > 5% return | Generalisation check; ensures in-sample success transfers to unseen data |
| G5 | Return-to-Drawdown Ratio | Ratio of validation return to maximum drawdown must exceed threshold | > 1.0 | Risk-efficiency check; ensures returns justify the risk exposure |
| G6 | Validation Profit Factor | Ratio of gross profit to gross loss on validation data must exceed threshold | > 1.0 | Quality-of-wins check; ensures profitable trades outweigh losses in magnitude |
| G7 | Robustness with Gap Penalty | Combined robustness score with penalty for train-to-validation performance gap | > 0.2 | Overfitting detection; penalises candidates that degrade sharply out-of-sample |

---

## Detailed Gate Specifications

### G1: Minimum Trade Count

- **Criterion:** `validation_trade_count >= min_trades`
- **Default threshold:** 25 trades
- **Config key:** `pipeline.min_trades`
- **Code location:** `pm_pipeline.py`, validation logic
- **Rationale:** A strategy that produces fewer than 25 trades in the validation
  window provides insufficient statistical evidence to judge its quality. Approval
  based on 3-5 trades would be unreliable regardless of the returns achieved.
- **Governance function:** Ensures that approval decisions are grounded in
  statistically meaningful sample sizes.

### G2: Maximum Drawdown

- **Criterion:** `validation_max_drawdown_pct < max_drawdown_threshold`
- **Default threshold:** 20%
- **Config key:** `pipeline.fx_val_max_drawdown`
- **Code location:** `pm_pipeline.py`, validation and selection logic
- **Rationale:** A strategy that draws down more than 20% of equity during
  validation represents an unacceptable capital risk. This gate exists independently
  of profitability: a strategy can be profitable but still rejected for excessive
  drawdown.
- **Governance function:** Absolute capital protection. This is a hard safety gate
  that cannot be overridden by strong performance in other dimensions.

### G3: Train Profitability

- **Criterion:** `train_return_pct > 0`
- **Default threshold:** 0% (must be positive)
- **Code location:** `pm_pipeline.py`, strategy scoring
- **Rationale:** If a strategy cannot produce a positive return on the data it was
  optimised against, it provides no basis for expecting out-of-sample success.
- **Governance function:** Eliminates fundamentally non-viable candidates early in
  the pipeline, reducing computational waste on validation of hopeless strategies.

### G4: Validation Return

- **Criterion:** `validation_return_pct >= min_val_return`
- **Default threshold:** 5%
- **Config key:** `pipeline.fx_val_min_return`
- **Code location:** `pm_pipeline.py`, validation logic
- **Rationale:** A strategy must demonstrate meaningful positive performance on
  data it has never seen during optimisation. The 5% threshold ensures that
  approved strategies show genuine generalisation, not just noise.
- **Governance function:** Generalisation assurance. Prevents deployment of
  strategies that overfit to training data.

### G5: Return-to-Drawdown Ratio

- **Criterion:** `validation_return_pct / max_drawdown_pct >= min_return_dd_ratio`
- **Default threshold:** 1.0
- **Config key:** `pipeline.fx_val_min_return_dd_ratio`
- **Code location:** `pm_pipeline.py`, `SymbolConfig.passes_live_quality_gate()`
- **Rationale:** Absolute return alone is insufficient. A strategy returning 10%
  with a 10% drawdown is qualitatively different from one returning 10% with a 2%
  drawdown. This gate ensures that returns are commensurate with the risk taken.
- **Governance function:** Risk-efficiency governance. Ensures the reward-to-risk
  profile is acceptable before deployment.

### G6: Validation Profit Factor

- **Criterion:** `validation_profit_factor >= min_profit_factor`
- **Default threshold:** 1.0
- **Config key:** `pipeline.regime_min_val_profit_factor`
- **Code location:** `pm_pipeline.py`, regime validation logic
- **Rationale:** Profit factor measures the ratio of total gross profit to total
  gross loss. A profit factor below 1.0 means the strategy loses more money on
  losing trades than it makes on winning trades, regardless of win rate.
- **Governance function:** Win-quality governance. Ensures that the magnitude
  of wins exceeds the magnitude of losses.

### G7: Robustness with Gap Penalty

- **Criterion:** `robustness_score >= min_robustness` (with gap penalty applied)
- **Default threshold:** 0.2
- **Config key:** `pipeline.min_robustness`
- **Code location:** `pm_pipeline.py`, robustness scoring
- **Rationale:** The robustness score combines multiple validation metrics into a
  single composite measure and applies a penalty when the gap between training and
  validation performance is large. This gate specifically targets overfitting:
  a strategy that performs brilliantly in-sample but degrades out-of-sample
  will receive a low robustness score.
- **Governance function:** Overfitting detection. The most sophisticated gate in
  the pipeline, designed to catch candidates that pass individual gates but show
  signs of being over-tuned to historical data.

---

## Gate Interactions and Ordering

The gates are evaluated in sequence. A failure at any gate terminates evaluation
immediately (short-circuit rejection). The ordering is designed to reject the
cheapest-to-evaluate failures first:

1. **G1** (trade count) eliminates statistically insufficient candidates
2. **G2** (drawdown) eliminates dangerous candidates
3. **G3** (train profitability) eliminates fundamentally non-viable candidates
4. **G4** (validation return) eliminates non-generalising candidates
5. **G5** (return/DD ratio) eliminates risk-inefficient candidates
6. **G6** (profit factor) eliminates win-quality-deficient candidates
7. **G7** (robustness) eliminates overfitting candidates

This ordering means that a candidate rejected at G2 never reaches G7,
saving computational effort and providing clear rejection reasons.

---

## Research Implications

The gate pipeline is the primary subject of Study A (Validation Governance).
Key research questions include:

- Which gates contribute the most to out-of-sample decision quality?
- What is the marginal contribution of each gate when added to a baseline?
- How do gate thresholds interact (e.g., does a stricter G2 make G5 redundant)?
- What is the trade-off between gate strictness and strategy acceptance rate?

The `pm_research.py` module provides `GateConfig` and `ABLATION_PRESETS` for
conducting these studies systematically.

---

## Prior Art Acknowledgement

Stage-gate methodology was introduced by Robert G. Cooper (1990) for
new-product development. Kumiega and Van Vliet (2008) applied stage-gate
thinking to trading-system development. The FXPM gate pipeline extends
this tradition by implementing empirically testable, quantitative validation
gates inside a working adaptive decision system.
