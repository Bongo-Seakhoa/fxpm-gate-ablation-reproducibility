# Frozen historical data placement

The public repository includes `MANIFEST.json` for the paper's frozen
historical dataset, but it does not include the broker M5 OHLCV CSV files.

To rerun the long historical evidence build, place the 50 manifest-matching
files in this directory:

```text
data/frozen_dataset/
  MANIFEST.json
  AU200_M5.csv
  AUDCAD_M5.csv
  ...
  XTIUSD_M5.csv
```

Then verify the checksums before building evidence:

```powershell
python historical/section6_run.py --stage verify --data-dir data/frozen_dataset
```

The expected dataset identity is `FBS-M5-HISTORICAL-v1`, covering
2013-06-11 through 2026-02-11 across the 50 symbols listed in the manifest.
