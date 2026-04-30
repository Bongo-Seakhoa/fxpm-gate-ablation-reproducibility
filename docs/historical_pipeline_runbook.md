# Historical ablation runbook

This repository now supports both levels of historical reproducibility used
for the paper:

1. Rerun the cheap historical materialisation and analysis from archived
   candidate evidence.
2. Rerun the full historical evidence build if the frozen broker CSV files
   are available locally.

The raw broker M5 CSV files are not committed. The dataset manifest and
checksums are committed in `data/frozen_dataset/MANIFEST.json`.

## Repository layout

```text
historical/                         Section 6 orchestration code
runtime/                            FXPM runtime modules used by Section 6
historical_results/evidence_build/  archived build manifest, slot index, candidates
historical_results/presets/         archived preset winners and rejection logs
historical_results/analysis/        archived paper-facing historical tables
data/frozen_dataset/MANIFEST.json   frozen-data identity and checksums
```

The archived evidence omits `evidence_build/features/` because that cache is
large and can be regenerated from the raw CSVs. The committed candidate
Parquet files are enough to rerun Stage C and Stage D/E.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-historical.txt
```

## Quick historical artifact check

This validates the committed historical table and evidence manifest without
running the long evidence build:

```powershell
python scripts/verify_historical_results.py
```

Expected headline:

```text
all_gates accepts 526/1200 slots, median rho 1.117, FAR 14.07%
```

## Rerun historical materialisation and analysis from cached evidence

This reproduces the historical preset winners and analysis tables from the
committed candidate cache:

```powershell
python historical/section6_run.py --stage materialise --output-root historical_results
python historical/section6_run.py --stage analyse --output-root historical_results
python scripts/verify_historical_results.py
```

To rerun one preset only:

```powershell
python historical/section6_run.py --stage materialise --preset all_gates --output-root historical_results
```

## Rerun the full historical evidence build

Place the 50 manifest-matching CSV files beside the committed manifest:

```text
data/frozen_dataset/AU200_M5.csv
data/frozen_dataset/AUDCAD_M5.csv
...
data/frozen_dataset/XTIUSD_M5.csv
```

Then run:

```powershell
python historical/section6_run.py --stage verify --data-dir data/frozen_dataset --output-root outputs/empirical_results_v2
python historical/section6_run.py --stage evidence --workers 4 --data-dir data/frozen_dataset --output-root outputs/empirical_results_v2
python historical/section6_run.py --stage materialise --output-root outputs/empirical_results_v2
python historical/section6_run.py --stage analyse --output-root outputs/empirical_results_v2
```

Or run all stages end to end:

```powershell
python historical/section6_run.py --stage all --workers 4 --data-dir data/frozen_dataset --output-root outputs/empirical_results_v2
```

The archived run completed the 300 symbol-timeframe evidence slots with no
failures. Runtime depends on hardware and worker count; the paper run was
multi-hour and should be treated as the expensive reproduction path.

## Stage definitions

- `verify`: checks each raw CSV against `MANIFEST.json`.
- `evidence`: builds gate-neutral candidates over 50 symbols x 6 timeframes.
- `materialise`: applies each gate preset to the frozen candidate evidence.
- `analyse`: computes fill rates, convergence, FAR sensitivity, statistical
  tests, and manuscript-ready tables.

## FAR/G7 note

Historical G7 is applied at materialisation time using the evidence-cache
score ratio `validation_score / training_score`. Historical FAR is computed
after materialisation using the paper-facing robustness diagnostic over the
selected winners. This is why `all_gates` can have non-zero historical FAR
while G7 is active. See `docs/historical_far_g7_note.md`.
