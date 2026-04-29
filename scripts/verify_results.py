"""Verify reproduced simulation output against the archived expected JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PRESETS = [
    "all_gates",
    "no_gates",
    "minimal",
    "drawdown_only",
    "profitability_only",
    "quality_focused",
    "robustness_focused",
]

SUMMARY_METRICS = [
    "acceptance_rate_pct",
    "median_robustness",
    "false_acceptance_rate_pct",
    "median_val_return_pct",
    "median_val_drawdown_pct",
    "median_val_pf",
    "median_val_return_dd",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_close(label: str, expected: float, actual: float, tolerance: float = 1e-9) -> None:
    if abs(expected - actual) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def verify(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    if expected["metadata"]["threshold_profile"] != actual["metadata"]["threshold_profile"]:
        raise AssertionError("threshold_profile mismatch")
    if expected["metadata"]["thresholds"] != actual["metadata"]["thresholds"]:
        raise AssertionError("threshold dictionary mismatch")

    for preset in PRESETS:
        for metric in SUMMARY_METRICS:
            label = f"{preset}.{metric}.median"
            _assert_close(
                label,
                expected["replication_summary"][preset][metric]["median"],
                actual["replication_summary"][preset][metric]["median"],
            )

    for test_name in ["friedman_median_robustness", "friedman_false_acceptance_rate_pct"]:
        for field in ["chi2", "df", "n", "k"]:
            _assert_close(
                f"{test_name}.{field}",
                expected["statistical_tests"][test_name][field],
                actual["statistical_tests"][test_name][field],
            )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: python scripts/verify_results.py EXPECTED_JSON ACTUAL_JSON", file=sys.stderr)
        return 2

    expected = _load(Path(argv[1]))
    actual = _load(Path(argv[2]))
    verify(expected, actual)
    print("OK: reproduced output matches expected simulation medians and selected tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

