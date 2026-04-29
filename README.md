# FXPM gate-ablation reproducibility package

This repository contains the public reproduction package for the simulation layer of:

**Simulation and Historical Evaluation of Validation-Gate Design in a Regime-Adaptive Decision System**

The package is intentionally small. It lets reviewers inspect and rerun the calibrated Monte Carlo gate-ablation study without access to the private FXPM workflow or broker data.

## Contents

- `src/gate_ablation_study.py` - standalone Python simulation and paired statistical analysis.
- `results/ablation_results_expected.json` - expected output for the paper run.
- `scripts/verify_results.py` - compares a reproduced JSON file against the expected medians and key test statistics.
- `docs/simulation_design.md` - compact description of the generator, thresholds, presets, and seeds.
- `docs/historical_far_g7_note.md` - clarification of the historical FAR/G7 relationship.

The simulation uses only the Python standard library.

## Reproduce the simulation

From the repository root:

```powershell
python src/gate_ablation_study.py --symbols 20 --replications 100 --output ../results/ablation_results_reproduced.json
python scripts/verify_results.py results/ablation_results_expected.json results/ablation_results_reproduced.json
```

The first command reruns the 100 paired replications with seeds `42` through `141`. The second command checks that the reproduced medians and selected inferential statistics match the archived expected output.

## Expected headline values

The paper's simulation table is based on `results/ablation_results_expected.json`:

| Preset | Median acceptance % | Median rho | Median FAR % |
| --- | ---: | ---: | ---: |
| all_gates | 4.62 | 0.836 | 0.00 |
| no_gates | 100.00 | 0.677 | 12.61 |
| minimal | 77.52 | 0.680 | 11.77 |
| drawdown_only | 81.25 | 0.680 | 11.79 |
| profitability_only | 41.39 | 0.686 | 7.36 |
| quality_focused | 16.09 | 0.674 | 8.74 |
| robustness_focused | 6.11 | 0.844 | 0.00 |

## Data availability

This repository does not include the broker historical M5 data used for the paper's frozen historical cross-check. That data is broker-environment research data rather than a public benchmark. The repository supports reproduction of the calibrated simulation layer and inspection of the exact gate predicates used for the aligned simulation profile.

