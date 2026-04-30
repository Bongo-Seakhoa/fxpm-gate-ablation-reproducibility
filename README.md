# FXPM gate-ablation reproducibility package

This repository contains the public reproduction package for the simulation
and historical ablation layers of:

**Simulation and Historical Evaluation of Validation-Gate Design in a Regime-Adaptive Decision System**

It lets reviewers inspect the gate logic, rerun the calibrated Monte Carlo
study, rerun the historical materialisation/analysis from archived candidate
evidence, and rerun the full historical evidence build from the committed
frozen broker CSV files.

## Contents

- `src/gate_ablation_study.py` - standalone Python simulation and paired statistical analysis.
- `results/ablation_results_expected.json` - expected output for the paper run.
- `scripts/verify_results.py` - compares a reproduced JSON file against the expected medians and key test statistics.
- `scripts/verify_historical_results.py` - checks the archived historical evidence manifest and paper-facing table.
- `historical/` - Section 6 historical pipeline: verify, evidence, materialise, analyse.
- `runtime/` - FXPM runtime modules required by the historical pipeline.
- `historical_results/` - archived historical candidate evidence, preset outputs, rejection logs, and analysis tables.
- `data/frozen_dataset/` - frozen historical CSV data plus manifest/checksums.
- `docs/simulation_design.md` - compact description of the generator, thresholds, presets, and seeds.
- `docs/historical_pipeline_runbook.md` - commands for cached and full historical reproduction.
- `docs/historical_far_g7_note.md` - clarification of the historical FAR/G7 relationship.

The Monte Carlo simulation uses only the Python standard library. The
historical layer requires pandas/numpy/scipy/pyarrow and the included runtime
modules.

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

## Reproduce the historical ablation layer

Install the historical environment:

```powershell
python -m pip install -r requirements-historical.txt
```

Check the committed historical artifacts without rerunning the expensive
evidence build:

```powershell
python scripts/verify_historical_results.py
```

Rerun the historical gate materialisation and analysis from the archived
candidate evidence:

```powershell
python historical/section6_run.py --stage materialise --output-root historical_results
python historical/section6_run.py --stage analyse --output-root historical_results
python scripts/verify_historical_results.py
```

For the full historical rerun, use the committed frozen CSV files in
`data/frozen_dataset/` and run:

```powershell
python historical/section6_run.py --stage verify --data-dir data/frozen_dataset --output-root outputs/empirical_results_v2
python historical/section6_run.py --stage evidence --workers 4 --data-dir data/frozen_dataset --output-root outputs/empirical_results_v2
python historical/section6_run.py --stage materialise --output-root outputs/empirical_results_v2
python historical/section6_run.py --stage analyse --output-root outputs/empirical_results_v2
```

See `docs/historical_pipeline_runbook.md` for the full runbook.

## Expected historical headline values

The paper's historical table is based on `historical_results/analysis/manuscript_table_7.csv`:

| Preset | Accepted slots | Acceptance % | Median rho | FAR % |
| --- | ---: | ---: | ---: | ---: |
| all_gates | 526 | 43.8 | 1.117 | 14.07 |
| no_gates | 1111 | 92.6 | 0.939 | 30.87 |
| drawdown_only | 1110 | 92.5 | 0.941 | 30.81 |
| profitability_only | 925 | 77.1 | 0.989 | 16.43 |
| minimal | 900 | 75.0 | 1.037 | 17.11 |
| quality_focused | 923 | 76.9 | 1.183 | 9.21 |
| robustness_focused | 704 | 58.7 | 1.199 | 11.08 |

## Data availability

This repository includes the raw frozen M5 CSV files used by the paper, the
frozen-data manifest/checksums, cached historical candidate evidence,
historical preset outputs, rejection logs, and analysis exports. A clone of
the repository is therefore sufficient to rerun both the Monte Carlo layer and
the historical ablation pipeline.
