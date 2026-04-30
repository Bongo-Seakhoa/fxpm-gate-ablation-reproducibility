"""Verify the archived historical ablation artifacts.

This is a lightweight audit helper. It does not rerun the long historical
evidence build; it checks that the committed cached outputs contain the
expected Section 6 headline table and evidence manifest.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RESULTS = ROOT / "historical_results"


EXPECTED_TABLE = {
    "all_gates": {
        "active_gates": "G1, G2, G3, G4, G5, G6, G7",
        "accepted_slots": "526",
        "acceptance_pct": "43.8",
        "median_rho": "1.117",
        "far_pct": "14.07",
        "median_val_return": "10.86",
    },
    "no_gates": {
        "active_gates": "none",
        "accepted_slots": "1111",
        "acceptance_pct": "92.6",
        "median_rho": "0.939",
        "far_pct": "30.87",
        "median_val_return": "3.99",
    },
    "drawdown_only": {
        "active_gates": "G2",
        "accepted_slots": "1110",
        "acceptance_pct": "92.5",
        "median_rho": "0.941",
        "far_pct": "30.81",
        "median_val_return": "3.97",
    },
    "profitability_only": {
        "active_gates": "G3, G6",
        "accepted_slots": "925",
        "acceptance_pct": "77.1",
        "median_rho": "0.989",
        "far_pct": "16.43",
        "median_val_return": "4.93",
    },
    "minimal": {
        "active_gates": "G1, G2",
        "accepted_slots": "900",
        "acceptance_pct": "75.0",
        "median_rho": "1.037",
        "far_pct": "17.11",
        "median_val_return": "5.57",
    },
    "quality_focused": {
        "active_gates": "G2, G5, G6",
        "accepted_slots": "923",
        "acceptance_pct": "76.9",
        "median_rho": "1.183",
        "far_pct": "9.21",
        "median_val_return": "5.45",
    },
    "robustness_focused": {
        "active_gates": "G1, G5, G7",
        "accepted_slots": "704",
        "acceptance_pct": "58.7",
        "median_rho": "1.199",
        "far_pct": "11.08",
        "median_val_return": "7.58",
    },
}


def fail(message: str) -> int:
    print(f"FAIL: {message}")
    return 1


def main() -> int:
    manifest_path = HISTORICAL_RESULTS / "evidence_build" / "build_manifest.json"
    table_path = HISTORICAL_RESULTS / "analysis" / "manuscript_table_7.csv"

    if not manifest_path.exists():
        return fail(f"missing {manifest_path.relative_to(ROOT)}")
    if not table_path.exists():
        return fail(f"missing {table_path.relative_to(ROOT)}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("completed_slots") != 300 or manifest.get("total_slots") != 300:
        return fail("historical evidence manifest does not show 300/300 completed slots")
    if manifest.get("failed_slots"):
        return fail("historical evidence manifest contains failed slots")

    with table_path.open("r", encoding="utf-8", newline="") as fh:
        rows = {row["preset"]: row for row in csv.DictReader(fh)}

    missing = sorted(set(EXPECTED_TABLE) - set(rows))
    if missing:
        return fail(f"missing table rows: {', '.join(missing)}")

    errors = []
    for preset, expected in EXPECTED_TABLE.items():
        row = rows[preset]
        for key, expected_value in expected.items():
            observed = row.get(key, "")
            if observed != expected_value:
                errors.append(f"{preset}.{key}: expected {expected_value}, observed {observed}")

    if errors:
        print("FAIL: historical table mismatch")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("OK: historical artifacts match expected Section 6 outputs.")
    print(f"Evidence cache key: {manifest.get('cache_key')}")
    print("Headline: all_gates accepts 526/1200 slots, median rho 1.117, FAR 14.07%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
