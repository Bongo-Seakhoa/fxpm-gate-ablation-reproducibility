# Frozen historical data

This directory contains the paper's frozen historical M5 OHLCV CSV files and
the matching `MANIFEST.json` checksum manifest. It is the self-contained input
dataset for rerunning the historical ablation pipeline.

```text
data/frozen_dataset/
  MANIFEST.json
  AU200_M5.csv
  AUDCAD_M5.csv
  ...
  XTIUSD_M5.csv
```

Verify the checksums before building evidence:

```powershell
python historical/section6_run.py --stage verify --data-dir data/frozen_dataset
```

The expected dataset identity is `FBS-M5-HISTORICAL-v1`, covering
2013-06-11 through 2026-02-11 across the 50 symbols listed in the manifest.
