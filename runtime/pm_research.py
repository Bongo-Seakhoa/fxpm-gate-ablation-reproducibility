"""
FXPM Research Infrastructure
==============================

Provides the academic research layer on top of the operational FXPM system.
This module does NOT modify the core trading logic. It adds:

1. Dataset freezing and verification (provenance, checksums)
2. Experiment manifests (full state capture per study run)
3. Gate ablation configuration (selective gate enable/disable)
4. Research-mode utilities

Academic context:
    Author: Bongo Bokoa Kosa (WD42M3)
    Institution: University of Debrecen, Hungary
    Programme: BSc Engineering Management
    Audit gaps addressed:
        - No frozen dataset (High)
        - No experiment registry (High)
        - No ablation programme (Medium)

Created: 2026-03-10
Research Log Entries: R-005, R-006
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pm_version import (
    ARTIFACT_VERSION,
    CONFIG_SCHEMA_VERSION,
    RESEARCH_TRACK_VERSION,
    get_artifact_stamp,
    get_environment_fingerprint,
    get_git_commit,
    hash_config,
)

RUNTIME_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = RUNTIME_ROOT / "pm_outputs" / "research"
EXPERIMENT_MANIFEST_VERSION = "1.0.0"


# =============================================================================
# 1. DATASET MANIFEST AND FREEZING
# =============================================================================

@dataclass
class DatasetManifest:
    """
    Provenance record for a frozen research dataset.

    Every frozen dataset must be accompanied by a manifest that records
    exactly what data it contains, when and how it was collected, and
    a checksum for integrity verification.
    """
    # Identity
    dataset_id: str = ""
    dataset_version: str = "1.0"
    description: str = ""

    # Content scope
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    date_range_start: str = ""
    date_range_end: str = ""

    # Quality metrics
    total_files: int = 0
    total_rows: int = 0
    file_checksums: Dict[str, str] = field(default_factory=dict)

    # Provenance
    collection_protocol: str = ""
    source: str = ""
    collected_by: str = ""
    collection_date: str = ""
    freeze_date: str = ""

    # Integrity
    manifest_checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path):
        """Save manifest as JSON."""
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        """Load manifest from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


def _sha256_file(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_dataset(
    data_dir: Path,
    output_dir: Path,
    dataset_id: str,
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    description: str = "",
    collection_protocol: str = "MT5 historical data export",
    source: str = "MetaTrader 5",
) -> DatasetManifest:
    """
    Freeze a dataset: copy CSV files, compute checksums, write manifest.

    Args:
        data_dir: Source directory containing CSV data files.
        output_dir: Destination directory for the frozen bundle.
        dataset_id: Unique identifier for this dataset.
        symbols: Optional filter for specific symbols. None = all.
        timeframes: Optional filter for specific timeframes. None = all.
        description: Human-readable description.
        collection_protocol: How the data was collected.
        source: Data source name.

    Returns:
        DatasetManifest with complete provenance information.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Regenerating a frozen bundle should not leave stale CSVs from an older
    # symbol universe in place. Clear prior dataset artifacts before copying.
    for stale_path in output_dir.glob("*.csv"):
        stale_path.unlink()
    manifest_path = output_dir / "MANIFEST.json"
    if manifest_path.exists():
        manifest_path.unlink()

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        description=description,
        collection_protocol=collection_protocol,
        source=source,
        collected_by="Bongo Bokoa Kosa",
        freeze_date=datetime.now(timezone.utc).isoformat(),
    )

    csv_files = sorted(data_dir.glob("*.csv"))
    found_symbols: Set[str] = set()
    found_timeframes: Set[str] = set()
    total_rows = 0
    file_count = 0
    selected_files: List[Path] = []

    for csv_path in csv_files:
        stem = csv_path.stem
        # Parse symbol and timeframe from filename (e.g., EURUSD_M5.csv)
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            sym, tf = parts
        else:
            sym, tf = stem, "unknown"

        # Apply filters
        if symbols and sym not in symbols:
            continue
        if timeframes and tf not in timeframes:
            continue

        # Copy file
        dest = output_dir / csv_path.name
        shutil.copy2(csv_path, dest)
        selected_files.append(dest)

        # Compute checksum and row count
        checksum = _sha256_file(dest)
        with open(dest, "r", encoding="utf-8") as f:
            rows = sum(1 for _ in f) - 1  # Subtract header
        total_rows += max(rows, 0)

        manifest.file_checksums[csv_path.name] = checksum
        found_symbols.add(sym)
        found_timeframes.add(tf)
        file_count += 1

    manifest.symbols = sorted(found_symbols)
    manifest.timeframes = sorted(found_timeframes)
    manifest.total_files = file_count
    manifest.total_rows = total_rows

    # Detect date range from the selected files
    if selected_files:
        try:
            import pandas as pd
            sample = pd.read_csv(selected_files[0], nrows=1)
            date_col = [c for c in sample.columns if "date" in c.lower() or "time" in c.lower()]
            if date_col:
                first_df = pd.read_csv(selected_files[0], usecols=[date_col[0]])
                manifest.date_range_start = str(first_df.iloc[0, 0])
                last_df = pd.read_csv(selected_files[-1], usecols=[date_col[0]])
                manifest.date_range_end = str(last_df.iloc[-1, 0])
        except Exception:
            pass  # Date range detection is best-effort

    # Save manifest
    manifest.save(manifest_path)

    return manifest


def verify_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """
    Verify a frozen dataset against its manifest.

    Returns a dict with verification results: overall pass/fail,
    and per-file status.
    """
    manifest_path = dataset_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return {"passed": False, "error": "No MANIFEST.json found"}

    manifest = DatasetManifest.load(manifest_path)
    results = {"passed": True, "files": {}, "manifest_id": manifest.dataset_id}

    for filename, expected_hash in manifest.file_checksums.items():
        filepath = dataset_dir / filename
        if not filepath.exists():
            results["files"][filename] = {"status": "MISSING"}
            results["passed"] = False
        else:
            actual_hash = _sha256_file(filepath)
            if actual_hash == expected_hash:
                results["files"][filename] = {"status": "OK"}
            else:
                results["files"][filename] = {
                    "status": "CHECKSUM_MISMATCH",
                    "expected": expected_hash[:16] + "...",
                    "actual": actual_hash[:16] + "...",
                }
                results["passed"] = False

    return results


# =============================================================================
# 2. EXPERIMENT MANIFEST
# =============================================================================

@dataclass
class ExperimentManifest:
    """
    Complete state capture for a single experiment run.

    This is the atomic unit of research reproducibility: every claim
    in the dissertation should trace back to an experiment manifest
    that fully specifies how the evidence was generated.
    """
    manifest_version: str = EXPERIMENT_MANIFEST_VERSION

    # Identity
    experiment_id: str = ""
    study_name: str = ""
    description: str = ""
    created: str = ""

    # Artifact state
    artifact_version: str = ""
    git_commit: str = ""
    config_hash: str = ""
    config_schema_version: str = ""

    # Data state
    dataset_id: str = ""
    dataset_version: str = ""
    dataset_path: str = ""

    # Experiment parameters
    random_seed: int = 42
    gate_config: str = "all_gates"
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Environment
    environment: Dict[str, Any] = field(default_factory=dict)

    # Results (filled after run)
    status: str = "pending"
    output_root: str = ""
    results_path: str = ""
    verification_path: str = ""
    config_path: str = ""
    completed: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ExperimentManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _resolve_path(path_value: Optional[str], base_dir: Optional[Path] = None) -> Path:
    base = Path(base_dir or RUNTIME_ROOT)
    if not path_value:
        return base
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _config_to_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        payload = config.to_dict()
        if isinstance(payload, dict):
            return payload
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError(f"Unsupported config payload: {type(config)!r}")


def _build_config_payload(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    for symbol, result in results.items():
        config = getattr(result, "config", None)
        if bool(getattr(result, "success", False)) and config is not None:
            configs[symbol] = _config_to_dict(config)
    return configs


def _count_regime_winners(configs: Dict[str, Dict[str, Any]]) -> int:
    total = 0
    for cfg in configs.values():
        regime_configs = cfg.get("regime_configs", {}) if isinstance(cfg, dict) else {}
        if isinstance(regime_configs, dict):
            total += sum(
                len(regimes) for regimes in regime_configs.values() if isinstance(regimes, dict)
            )
    return total


def _write_config_payload(configs: Dict[str, Dict[str, Any]], path: Path) -> Dict[str, Any]:
    _write_json(path, configs)
    reloaded = _load_json(path)
    saved_symbols = sorted(reloaded.keys())
    expected_symbols = sorted(configs.keys())

    if saved_symbols != expected_symbols:
        missing = sorted(set(expected_symbols) - set(saved_symbols))
        extra = sorted(set(saved_symbols) - set(expected_symbols))
        raise RuntimeError(
            "Final pm_configs.json verification failed: "
            f"expected {len(expected_symbols)} symbols, saved {len(saved_symbols)} "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )

    saved_regime_winners = _count_regime_winners(reloaded)
    expected_regime_winners = _count_regime_winners(configs)

    if saved_regime_winners != expected_regime_winners:
        raise RuntimeError(
            "Final pm_configs.json verification failed: "
            f"expected {expected_regime_winners} regime winners, saved {saved_regime_winners}"
        )

    return {
        "saved_config_count": len(saved_symbols),
        "saved_regime_winner_count": saved_regime_winners,
    }


def _filter_configs_to_canonical_successes(
    configs: Dict[str, Dict[str, Any]],
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    winner_symbols = set(_winner_row_symbols(rows))
    return {
        symbol: payload
        for symbol, payload in configs.items()
        if symbol in winner_symbols
    }


def _discover_repair_symbols(
    requested_symbols: List[str],
    existing_rows: List[Dict[str, Any]],
    existing_configs: Dict[str, Dict[str, Any]],
) -> List[str]:
    row_by_symbol = {row["symbol"]: row for row in existing_rows if row.get("symbol")}
    retry: List[str] = []
    for symbol in requested_symbols:
        row = row_by_symbol.get(symbol)
        if row is None:
            retry.append(symbol)
            continue
        if not _is_truthy(row.get("success")):
            retry.append(symbol)
            continue
        if _to_int(row.get("regime_winners")) > 0 and symbol not in existing_configs:
            retry.append(symbol)
    return sorted(set(retry))


def _write_final_configs(results: Dict[str, Any], path: Path) -> Dict[str, Any]:
    return _write_config_payload(_build_config_payload(results), path)


def _filter_dataclass_kwargs(cls: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {field_info.name for field_info in fields(cls)}
    return {key: value for key, value in (data or {}).items() if key in allowed}


def _load_dataset_manifest(dataset_dir: Path) -> Optional[DatasetManifest]:
    manifest_path = dataset_dir / "MANIFEST.json"
    if manifest_path.exists():
        return DatasetManifest.load(manifest_path)
    return None


def create_experiment_manifest(
    experiment_id: str,
    study_name: str,
    description: str = "",
    dataset_id: str = "",
    dataset_version: str = "",
    dataset_path: str = "",
    random_seed: int = 42,
    gate_config: str = "all_gates",
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    config_path: str = "config.json",
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
) -> ExperimentManifest:
    """Create a preregistered experiment manifest capturing the artifact state."""
    dataset_dir = _resolve_path(dataset_path, RUNTIME_ROOT) if dataset_path else None
    config_file = _resolve_path(config_path, RUNTIME_ROOT)
    output_root_path = _resolve_path(output_root, RUNTIME_ROOT)

    dataset_manifest = _load_dataset_manifest(dataset_dir) if dataset_dir else None
    resolved_symbols = list(symbols or [])
    resolved_timeframes = list(timeframes or [])

    if dataset_manifest:
        if not dataset_id:
            dataset_id = dataset_manifest.dataset_id
        if not dataset_version:
            dataset_version = dataset_manifest.dataset_version
        if not resolved_symbols:
            resolved_symbols = list(dataset_manifest.symbols)
        if not resolved_timeframes:
            resolved_timeframes = list(dataset_manifest.timeframes)

    return ExperimentManifest(
        experiment_id=experiment_id,
        study_name=study_name,
        description=description,
        created=datetime.now(timezone.utc).isoformat(),
        artifact_version=ARTIFACT_VERSION,
        git_commit=get_git_commit() or "unknown",
        config_hash=hash_config(str(config_file)),
        config_schema_version=CONFIG_SCHEMA_VERSION,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        dataset_path=str(dataset_dir) if dataset_dir else "",
        random_seed=random_seed,
        gate_config=gate_config,
        symbols=resolved_symbols,
        timeframes=resolved_timeframes,
        parameters=parameters or {},
        environment=get_environment_fingerprint(),
        status="pending",
        output_root=str(output_root_path),
        config_path=str(config_file),
    )


def write_sample_manifest(
    manifest_path: Path,
    dataset_path: str,
    experiment_id: str = "EXP-SAMPLE-001",
    study_name: str = "FXPM Research Study",
    description: str = "Manifest-driven offline benchmark run",
    random_seed: int = 42,
    gate_config: str = "all_gates",
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    config_path: str = "config.json",
    output_root: str = str(DEFAULT_OUTPUT_ROOT),
) -> ExperimentManifest:
    manifest = create_experiment_manifest(
        experiment_id=experiment_id,
        study_name=study_name,
        description=description,
        dataset_path=dataset_path,
        random_seed=random_seed,
        gate_config=gate_config,
        symbols=symbols,
        timeframes=timeframes,
        parameters=parameters,
        config_path=config_path,
        output_root=output_root,
    )
    manifest.save(Path(manifest_path))
    return manifest


def _build_pipeline_config(manifest: ExperimentManifest, config_file: Path, dataset_dir: Path, run_dir: Path) -> Tuple[Any, Dict[str, Any], GateConfig]:
    from pm_core import PipelineConfig

    config_data = _load_json(config_file)
    pipeline_data = _filter_dataclass_kwargs(PipelineConfig, config_data.get("pipeline", {}))
    pipeline_config = PipelineConfig(**pipeline_data)
    pipeline_config.data_dir = dataset_dir
    pipeline_config.output_dir = run_dir
    pipeline_config.optimization_max_workers = 1
    if manifest.timeframes:
        pipeline_config.timeframes = list(manifest.timeframes)
    setattr(pipeline_config, "data_cache_dir", run_dir / ".cache")

    for key, value in manifest.parameters.items():
        if hasattr(pipeline_config, key):
            setattr(pipeline_config, key, value)

    gate_config = _resolve_gate_config(manifest.gate_config)
    _apply_gate_config(pipeline_config, gate_config)
    return pipeline_config, config_data, gate_config


def _resolve_symbols(manifest: ExperimentManifest, config_data: Dict[str, Any], dataset_manifest: Optional[DatasetManifest]) -> List[str]:
    if manifest.symbols:
        return list(manifest.symbols)
    if dataset_manifest and dataset_manifest.symbols:
        return list(dataset_manifest.symbols)
    return list(config_data.get("symbols", []))


def _is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _serialise_result_rows(symbols: List[str], results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for symbol in symbols:
        result = results.get(symbol)
        config = getattr(result, "config", None) if result is not None else None
        rows.append(
            {
                "symbol": symbol,
                "success": bool(getattr(result, "success", False)) if result is not None else False,
                "is_validated": bool(getattr(config, "is_validated", False)) if config is not None else False,
                "strategy_name": getattr(config, "strategy_name", "") if config is not None else "",
                "timeframe": getattr(config, "timeframe", "") if config is not None else "",
                "error_message": getattr(result, "error_message", "") if result is not None else "",
                "duration_seconds": getattr(result, "duration_seconds", 0.0) if result is not None else 0.0,
                "strategies_tested": getattr(result, "strategies_tested", 0) if result is not None else 0,
                "timeframes_tested": getattr(result, "timeframes_tested", 0) if result is not None else 0,
                "param_combos_tested": getattr(result, "param_combos_tested", 0) if result is not None else 0,
                "regime_winners": getattr(result, "regime_winners", 0) if result is not None else 0,
            }
        )
    return rows


def _load_results_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_results_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "symbol",
        "success",
        "is_validated",
        "strategy_name",
        "timeframe",
        "error_message",
        "duration_seconds",
        "strategies_tested",
        "timeframes_tested",
        "param_combos_tested",
        "regime_winners",
    ]
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _successful_row_symbols(rows: List[Dict[str, Any]]) -> List[str]:
    return sorted(
        row["symbol"]
        for row in rows
        if row.get("symbol") and _is_truthy(row.get("success"))
    )


def _winner_row_symbols(rows: List[Dict[str, Any]]) -> List[str]:
    return sorted(
        row["symbol"]
        for row in rows
        if row.get("symbol")
        and _is_truthy(row.get("success"))
        and _to_int(row.get("regime_winners")) > 0
    )


def _merge_result_rows(
    requested_symbols: List[str],
    existing_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    row_by_symbol: Dict[str, Dict[str, Any]] = {
        row["symbol"]: dict(row)
        for row in existing_rows
        if row.get("symbol")
    }
    for row in new_rows:
        if row.get("symbol"):
            row_by_symbol[row["symbol"]] = dict(row)

    merged_rows: List[Dict[str, Any]] = []
    for symbol in requested_symbols:
        if symbol not in row_by_symbol:
            raise RuntimeError(
                f"Repair merge failed: no canonical optimization_results.csv row for {symbol}"
            )
        merged_rows.append(row_by_symbol[symbol])
    return merged_rows


def _snapshot_repair_state(run_dir: Path, repair_event_id: str, artifact_paths: List[Path]) -> Path:
    snapshot_dir = run_dir / "repairs" / repair_event_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for source in artifact_paths:
        if source.exists():
            shutil.copy2(source, snapshot_dir / source.name)
    return snapshot_dir


def _rollback_repair_state(run_dir: Path, snapshot_dir: Path, artifact_names: List[str]) -> None:
    """Restore pre-repair canonical artifacts from a snapshot directory."""
    for name in artifact_names:
        snapshot_file = snapshot_dir / name
        target_file = run_dir / name
        if snapshot_file.exists():
            shutil.copy2(snapshot_file, target_file)


def _append_repair_log(path: Path, event: Dict[str, Any]) -> int:
    payload = _load_json(path)
    events = payload.get("events")
    if not isinstance(events, list):
        events = []
    events.append(event)
    _write_json(path, {"events": events})
    return len(events)


def _build_experiment_summary(
    manifest: ExperimentManifest,
    dataset_dir: Path,
    requested_symbols: List[str],
    requested_timeframes: List[str],
    optimized_symbols: List[str],
    rows: List[Dict[str, Any]],
    config_integrity: Dict[str, Any],
    gate_config: Any,
    run_dir: Path,
    results_csv_path: Path,
    verification_path: Path,
    config_snapshot_path: Path,
    artifact_stamp_path: Path,
    completed: str,
    repair_event: Optional[Dict[str, Any]] = None,
    repair_count: int = 0,
) -> Dict[str, Any]:
    summary = {
        "experiment_id": manifest.experiment_id,
        "study_name": manifest.study_name,
        "status": "completed",
        "dataset_id": manifest.dataset_id,
        "dataset_version": manifest.dataset_version,
        "dataset_path": str(dataset_dir),
        "requested_symbols": requested_symbols,
        "requested_timeframes": requested_timeframes,
        "optimized_symbols": sorted(optimized_symbols),
        "optimized_count": len(optimized_symbols),
        "success_count": sum(1 for row in rows if _is_truthy(row.get("success"))),
        "validated_count": sum(1 for row in rows if _is_truthy(row.get("is_validated"))),
        "saved_config_count": config_integrity["saved_config_count"],
        "saved_regime_winner_count": config_integrity["saved_regime_winner_count"],
        "gate_config": manifest.gate_config,
        "gate_config_applied": True,
        "active_gates": gate_config.active_gate_names(),
        "inactive_gates": [name for name in [
            "G1_min_trades", "G2_max_drawdown", "G3_train_profitability",
            "G4_val_return", "G5_return_dd_ratio", "G6_val_profit_factor",
            "G7_robustness_gap",
        ] if name not in gate_config.active_gate_names()],
        "output_dir": str(run_dir),
        "results_csv": str(results_csv_path),
        "verification_path": str(verification_path),
        "config_snapshot_path": str(config_snapshot_path),
        "artifact_stamp_path": str(artifact_stamp_path),
        "completed": completed,
    }
    if repair_event is not None:
        summary.update(
            {
                "repair_mode": True,
                "repair_count": repair_count,
                "repair_log_path": str(run_dir / "repair_log.json"),
                "last_repair_event_id": repair_event["repair_event_id"],
                "last_repair_timestamp": repair_event["repair_timestamp"],
                "repair_retry_symbols": list(repair_event["retry_symbols"]),
                "repair_reason": repair_event["repair_reason"],
            }
        )
    return summary


def run_experiment_manifest(
    manifest_path: Path,
    config_path: Optional[str] = None,
    output_root: Optional[str] = None,
    overwrite: bool = False,
    repair: bool = False,
    repair_symbols: Optional[List[str]] = None,
    repair_reason: str = "",
    manager_cls: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a preregistered offline experiment against a frozen dataset."""
    manifest_path = Path(manifest_path)
    manifest = ExperimentManifest.load(manifest_path)

    dataset_dir = _resolve_path(manifest.dataset_path, manifest_path.parent)
    config_file = _resolve_path(config_path or manifest.config_path or "config.json", manifest_path.parent)
    output_root_path = _resolve_path(output_root or manifest.output_root or str(DEFAULT_OUTPUT_ROOT), manifest_path.parent)
    run_dir = output_root_path / manifest.experiment_id

    if repair and overwrite:
        raise ValueError("repair=True cannot be combined with overwrite=True")
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite and not repair:
        raise FileExistsError(f"Run directory already exists for {manifest.experiment_id}: {run_dir}. Use overwrite=True to reuse it.")
    if repair and (not run_dir.exists() or not any(run_dir.iterdir())):
        raise FileNotFoundError(
            f"Repair mode requires an existing populated run directory for {manifest.experiment_id}: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    registration_path = run_dir / "registration_manifest.json"
    final_manifest_path = run_dir / "experiment_manifest.json"
    verification_path = run_dir / "dataset_verification.json"
    artifact_stamp_path = run_dir / "artifact_stamp.json"
    summary_path = run_dir / "experiment_summary.json"
    results_csv_path = run_dir / "optimization_results.csv"
    config_snapshot_path = run_dir / "config_snapshot.json"
    repair_log_path = run_dir / "repair_log.json"

    if not repair or not registration_path.exists():
        registration_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    verification: Dict[str, Any] = {}
    existing_rows: List[Dict[str, Any]] = []
    existing_configs: Dict[str, Dict[str, Any]] = {}
    requested_symbols: List[str] = []
    _repair_rolled_back = False
    requested_timeframes: List[str] = []
    retry_symbols_to_run: List[str] = []
    repair_snapshot_dir: Optional[Path] = None
    repair_event_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    try:
        verification = verify_dataset(dataset_dir)
        _write_json(verification_path, verification)
        if not verification.get("passed"):
            raise RuntimeError(f"Dataset verification failed for {dataset_dir}")

        dataset_manifest = _load_dataset_manifest(dataset_dir)
        if dataset_manifest and manifest.dataset_id and manifest.dataset_id != dataset_manifest.dataset_id:
            raise ValueError(f"Dataset ID mismatch: manifest expects {manifest.dataset_id}, frozen data contains {dataset_manifest.dataset_id}")
        if dataset_manifest and manifest.dataset_version and manifest.dataset_version != dataset_manifest.dataset_version:
            raise ValueError(f"Dataset version mismatch: manifest expects {manifest.dataset_version}, frozen data contains {dataset_manifest.dataset_version}")

        actual_config_hash = hash_config(str(config_file))
        if manifest.config_hash and manifest.config_hash != actual_config_hash:
            raise ValueError("Config hash mismatch between the preregistered manifest and the supplied configuration file.")

        pipeline_config, config_data, gate_config = _build_pipeline_config(manifest, config_file, dataset_dir, run_dir)
        dataset_manifest = dataset_manifest or _load_dataset_manifest(dataset_dir)
        symbols = _resolve_symbols(manifest, config_data, dataset_manifest)
        if not symbols:
            raise ValueError("No symbols were resolved for the experiment run.")

        if not getattr(pipeline_config, "timeframes", None):
            if manifest.timeframes:
                pipeline_config.timeframes = list(manifest.timeframes)
            elif dataset_manifest and dataset_manifest.timeframes:
                pipeline_config.timeframes = list(dataset_manifest.timeframes)

        requested_symbols = list(symbols)
        requested_timeframes = list(getattr(pipeline_config, "timeframes", []))

        if repair:
            if not summary_path.exists() or not results_csv_path.exists():
                raise FileNotFoundError(
                    f"Repair mode requires existing experiment_summary.json and optimization_results.csv in {run_dir}"
                )
            existing_summary = _load_json(summary_path)
            existing_rows = _load_results_csv(results_csv_path)
            existing_configs = _filter_configs_to_canonical_successes(
                _load_json(run_dir / "pm_configs.json"),
                existing_rows,
            )
            requested_symbols = list(existing_summary.get("requested_symbols") or requested_symbols)
            requested_timeframes = list(existing_summary.get("requested_timeframes") or requested_timeframes)

            discovered_retry_symbols = _discover_repair_symbols(
                requested_symbols=requested_symbols,
                existing_rows=existing_rows,
                existing_configs=existing_configs,
            )
            if repair_symbols:
                invalid = sorted(set(repair_symbols) - set(discovered_retry_symbols))
                if invalid:
                    raise ValueError(
                        "Repair symbols must come from the failed or missing symbol set. "
                        f"Invalid selections: {invalid[:5]}"
                    )
                retry_symbols_to_run = sorted(dict.fromkeys(repair_symbols))
            else:
                retry_symbols_to_run = discovered_retry_symbols

            repair_snapshot_dir = _snapshot_repair_state(
                run_dir,
                repair_event_id,
                [
                    registration_path,
                    final_manifest_path,
                    summary_path,
                    results_csv_path,
                    run_dir / "pm_configs.json",
                ],
            )

        manifest.status = "running"
        manifest.output_root = str(output_root_path)
        manifest.dataset_path = str(dataset_dir)
        manifest.config_path = str(config_file)
        manifest.error_message = ""
        manifest.completed = ""
        manifest.save(final_manifest_path)

        if config_file.exists():
            shutil.copy2(config_file, config_snapshot_path)
        else:
            _write_json(config_snapshot_path, config_data)

        # Persist the applied gate flags and thresholds into the config snapshot
        # so the receipt chain records exactly which gates were active.
        try:
            snap_data = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
            snap_data["_applied_gate_config"] = {
                "preset_name": gate_config.name,
                "active_gates": gate_config.active_gate_names(),
                "flags": {
                    "g1_min_trades": gate_config.g1_min_trades,
                    "g2_max_drawdown": gate_config.g2_max_drawdown,
                    "g3_train_profitability": gate_config.g3_train_profitability,
                    "g4_val_return": gate_config.g4_val_return,
                    "g5_return_dd_ratio": gate_config.g5_return_dd_ratio,
                    "g6_val_profit_factor": gate_config.g6_val_profit_factor,
                    "g7_robustness_gap": gate_config.g7_robustness_gap,
                },
                "thresholds": {
                    "g1_min_trades": gate_config.g1_threshold,
                    "g2_max_drawdown": gate_config.g2_threshold,
                    "g3_train_return": gate_config.g3_threshold,
                    "g4_val_return": gate_config.g4_threshold,
                    "g5_return_dd_ratio": gate_config.g5_threshold,
                    "g6_val_profit_factor": gate_config.g6_threshold,
                    "g7_robustness": gate_config.g7_threshold,
                },
            }
            config_snapshot_path.write_text(
                json.dumps(snap_data, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to persist gate config to snapshot: %s", exc)

        _write_json(artifact_stamp_path, get_artifact_stamp())

        if manager_cls is None:
            from pm_pipeline import PortfolioManager
            manager_cls = PortfolioManager

        manager_symbols = list(symbols)
        manager_config_path = run_dir / "pm_configs.json"
        manager_overwrite = overwrite
        if repair:
            manager_symbols = list(retry_symbols_to_run)
            manager_config_path = (repair_snapshot_dir or run_dir) / "repair_work_pm_configs.json"
            manager_overwrite = False
            preserved_configs = {
                symbol: payload
                for symbol, payload in existing_configs.items()
                if symbol not in retry_symbols_to_run
            }
            _write_json(manager_config_path, preserved_configs)

        manager = manager_cls(
            config=pipeline_config,
            symbols=manager_symbols,
            config_file=str(manager_config_path),
        )
        results = manager.initial_optimization(overwrite=manager_overwrite)
        if repair:
            merged_configs = dict(existing_configs)
            merged_configs.update(_build_config_payload(results))
            rows = _merge_result_rows(
                requested_symbols=requested_symbols,
                existing_rows=existing_rows,
                new_rows=_serialise_result_rows(manager_symbols, results),
            )

            # Post-merge cross-artifact integrity: verify consistency
            # BEFORE writing canonical files so rollback is unnecessary
            # if the check itself fails.
            csv_winner_syms = set(_winner_row_symbols(rows))
            cfg_syms = set(merged_configs.keys())
            if csv_winner_syms != cfg_syms:
                missing_in_cfg = sorted(csv_winner_syms - cfg_syms)
                extra_in_cfg = sorted(cfg_syms - csv_winner_syms)
                raise RuntimeError(
                    "Post-merge cross-artifact integrity failure: pm_configs.json symbols "
                    f"do not match optimization_results.csv winner symbols "
                    f"(missing_in_configs={missing_in_cfg[:5]}, extra_in_configs={extra_in_cfg[:5]})"
                )

            # --- Transactional repair finalization ---
            # All canonical artifact writes are grouped here. If any step
            # fails after the first write, the pre-repair snapshots are
            # restored so the preset folder is never left half-merged.
            # experiment_manifest.json is included so the outer exception
            # handler cannot leave it in a "failed" state while the other
            # canonical artifacts reflect the pre-repair "completed" state.
            _rollback_artifacts = ["pm_configs.json", "optimization_results.csv",
                                   "experiment_summary.json", "experiment_manifest.json"]
            try:
                config_integrity = _write_config_payload(merged_configs, run_dir / "pm_configs.json")
                _write_results_csv(rows, results_csv_path)

                completed = datetime.now(timezone.utc).isoformat()
                post_row_map = {row["symbol"]: row for row in rows if row.get("symbol")}
                repair_event = {
                    "repair_event_id": repair_event_id,
                    "experiment_id": manifest.experiment_id,
                    "repair_timestamp": completed,
                    "repair_reason": repair_reason or "in-place completion",
                    "repair_mode_version": "1.0",
                    "retry_symbols": retry_symbols_to_run,
                    "requested_symbols": requested_symbols,
                    "requested_timeframes": requested_timeframes,
                    "repaired_success_symbols": sorted(
                        symbol for symbol in retry_symbols_to_run
                        if _is_truthy(post_row_map.get(symbol, {}).get("success"))
                    ),
                    "preserved_symbols": sorted(set(requested_symbols) - set(retry_symbols_to_run)),
                    "symbols_still_failed": sorted(
                        symbol for symbol in retry_symbols_to_run
                        if not _is_truthy(post_row_map.get(symbol, {}).get("success"))
                    ),
                    "pre_repair_success_count": sum(1 for row in existing_rows if _is_truthy(row.get("success"))),
                    "post_repair_success_count": sum(1 for row in rows if _is_truthy(row.get("success"))),
                    "pre_repair_validated_count": sum(1 for row in existing_rows if _is_truthy(row.get("is_validated"))),
                    "post_repair_validated_count": sum(1 for row in rows if _is_truthy(row.get("is_validated"))),
                    "pre_repair_saved_config_count": len(existing_configs),
                    "post_repair_saved_config_count": config_integrity["saved_config_count"],
                    "pre_repair_regime_winner_count": _count_regime_winners(existing_configs),
                    "post_repair_regime_winner_count": config_integrity["saved_regime_winner_count"],
                    "dataset_id": manifest.dataset_id,
                    "dataset_version": manifest.dataset_version,
                    "dataset_verification_passed": bool(verification.get("passed")),
                    "config_hash_verified": bool(manifest.config_hash == actual_config_hash),
                    "config_hash": actual_config_hash,
                    "gate_preset": manifest.gate_config,
                    "artifact_stamp": get_artifact_stamp(),
                    "output_dir": str(run_dir),
                    "snapshot_dir": str(repair_snapshot_dir) if repair_snapshot_dir else "",
                    "optimization_max_workers": int(getattr(pipeline_config, "optimization_max_workers", 1) or 1),
                }
                repair_count = _append_repair_log(repair_log_path, repair_event)

                # In repair mode, optimized_symbols must reflect the full
                # canonical run — every symbol that has been through the
                # pipeline, whether preserved, retried, or still failed.
                # This is exactly `requested_symbols`, not just the
                # successful-config union, because a staged repair may
                # leave un-retried failed symbols in the merged CSV.
                canonical_optimized = sorted(requested_symbols)

                summary = _build_experiment_summary(
                    manifest=manifest,
                    dataset_dir=dataset_dir,
                    requested_symbols=requested_symbols,
                    requested_timeframes=requested_timeframes,
                    optimized_symbols=canonical_optimized,
                    rows=rows,
                    config_integrity=config_integrity,
                    gate_config=gate_config,
                    run_dir=run_dir,
                    results_csv_path=results_csv_path,
                    verification_path=verification_path,
                    config_snapshot_path=config_snapshot_path,
                    artifact_stamp_path=artifact_stamp_path,
                    completed=completed,
                    repair_event=repair_event,
                    repair_count=repair_count,
                )
                _write_json(summary_path, summary)
            except Exception:
                # Roll back ALL canonical artifacts to pre-repair state,
                # including experiment_manifest.json, so the preset folder
                # is left in a fully consistent pre-repair condition.
                if repair_snapshot_dir:
                    _rollback_repair_state(run_dir, repair_snapshot_dir, _rollback_artifacts)
                    _repair_rolled_back = True
                raise
        else:
            config_integrity = _write_final_configs(results, run_dir / "pm_configs.json")
            rows = _serialise_result_rows(symbols, results)
            _write_results_csv(rows, results_csv_path)

            # Post-write cross-artifact integrity check (non-repair)
            csv_winner_syms = set(_winner_row_symbols(rows))
            cfg_syms = set(_build_config_payload(results).keys())
            if csv_winner_syms != cfg_syms:
                missing_in_cfg = sorted(csv_winner_syms - cfg_syms)
                extra_in_cfg = sorted(cfg_syms - csv_winner_syms)
                raise RuntimeError(
                    "Post-write cross-artifact integrity failure: pm_configs.json symbols "
                    f"do not match optimization_results.csv winner symbols "
                    f"(missing_in_configs={missing_in_cfg[:5]}, extra_in_configs={extra_in_cfg[:5]})"
                )

            completed = datetime.now(timezone.utc).isoformat()
            repair_event: Optional[Dict[str, Any]] = None
            repair_count = 0
            summary = _build_experiment_summary(
                manifest=manifest,
                dataset_dir=dataset_dir,
                requested_symbols=requested_symbols,
                requested_timeframes=requested_timeframes,
                optimized_symbols=sorted(results.keys()),
                rows=rows,
                config_integrity=config_integrity,
                gate_config=gate_config,
                run_dir=run_dir,
                results_csv_path=results_csv_path,
                verification_path=verification_path,
                config_snapshot_path=config_snapshot_path,
                artifact_stamp_path=artifact_stamp_path,
                completed=completed,
                repair_event=repair_event,
                repair_count=repair_count,
            )
            _write_json(summary_path, summary)

        manifest.status = "completed"
        manifest.results_path = str(summary_path)
        manifest.verification_path = str(verification_path)
        manifest.completed = completed
        manifest.error_message = ""
        manifest.save(final_manifest_path)
        return summary
    except Exception as exc:
        if _repair_rolled_back:
            # Repair rollback already restored all canonical artifacts
            # including experiment_manifest.json to the pre-repair state.
            # Do NOT overwrite the manifest here — the folder must remain
            # in a fully consistent pre-repair condition.
            pass
        else:
            manifest.status = "failed"
            manifest.results_path = ""
            manifest.verification_path = str(verification_path) if verification_path.exists() else ""
            manifest.completed = datetime.now(timezone.utc).isoformat()
            manifest.error_message = str(exc)
            manifest.save(final_manifest_path)
        raise


# =============================================================================
# 3. GATE ABLATION CONFIGURATION
# =============================================================================

@dataclass
class GateConfig:
    """
    Configuration for validation gate ablation studies.

    Each boolean flag controls whether a specific validation gate
    is active. Thresholds can be overridden for sensitivity analysis.

    Gate identifiers match the seven gates documented in GATE_CATALOG.md:
        G1: Minimum trade count
        G2: Maximum drawdown
        G3: Train profitability
        G4: Validation return
        G5: Return-to-drawdown ratio
        G6: Validation profit factor
        G7: Robustness with gap penalty
    """
    name: str = "all_gates"
    description: str = ""

    # Gate enable/disable flags
    g1_min_trades: bool = True
    g2_max_drawdown: bool = True
    g3_train_profitability: bool = True
    g4_val_return: bool = True
    g5_return_dd_ratio: bool = True
    g6_val_profit_factor: bool = True
    g7_robustness_gap: bool = True

    # Threshold overrides (None = use default from config.json)
    g1_threshold: Optional[int] = None        # min trades
    g2_threshold: Optional[float] = None      # max drawdown %
    g3_threshold: Optional[float] = None      # min train return %
    g4_threshold: Optional[float] = None      # min val return %
    g5_threshold: Optional[float] = None      # min return/DD ratio
    g6_threshold: Optional[float] = None      # min profit factor
    g7_threshold: Optional[float] = None      # min robustness score

    def active_gate_count(self) -> int:
        """Number of active gates."""
        return sum([
            self.g1_min_trades, self.g2_max_drawdown,
            self.g3_train_profitability, self.g4_val_return,
            self.g5_return_dd_ratio, self.g6_val_profit_factor,
            self.g7_robustness_gap,
        ])

    def active_gate_names(self) -> List[str]:
        """List of active gate identifiers."""
        gates = []
        if self.g1_min_trades: gates.append("G1_min_trades")
        if self.g2_max_drawdown: gates.append("G2_max_drawdown")
        if self.g3_train_profitability: gates.append("G3_train_profitability")
        if self.g4_val_return: gates.append("G4_val_return")
        if self.g5_return_dd_ratio: gates.append("G5_return_dd_ratio")
        if self.g6_val_profit_factor: gates.append("G6_val_profit_factor")
        if self.g7_robustness_gap: gates.append("G7_robustness_gap")
        return gates

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Pre-built ablation presets for standard studies
ABLATION_PRESETS: Dict[str, GateConfig] = {
    "all_gates": GateConfig(
        name="all_gates",
        description="Full seven-gate validation pipeline (baseline configuration)",
    ),
    "no_gates": GateConfig(
        name="no_gates",
        description="All gates disabled (negative baseline for ablation comparison)",
        g1_min_trades=False, g2_max_drawdown=False,
        g3_train_profitability=False, g4_val_return=False,
        g5_return_dd_ratio=False, g6_val_profit_factor=False,
        g7_robustness_gap=False,
    ),
    "drawdown_only": GateConfig(
        name="drawdown_only",
        description="Only the maximum drawdown gate is active",
        g1_min_trades=False, g3_train_profitability=False,
        g4_val_return=False, g5_return_dd_ratio=False,
        g6_val_profit_factor=False, g7_robustness_gap=False,
    ),
    "profitability_only": GateConfig(
        name="profitability_only",
        description="Only profitability gates active (G3 train + G6 profit factor)",
        g1_min_trades=False, g2_max_drawdown=False,
        g4_val_return=False, g5_return_dd_ratio=False,
        g7_robustness_gap=False,
    ),
    "minimal": GateConfig(
        name="minimal",
        description="Minimal gate set: trade count + drawdown only",
        g3_train_profitability=False, g4_val_return=False,
        g5_return_dd_ratio=False, g6_val_profit_factor=False,
        g7_robustness_gap=False,
    ),
    "quality_focused": GateConfig(
        name="quality_focused",
        description="Quality-focused gates: drawdown, return/DD ratio, profit factor",
        g1_min_trades=False, g3_train_profitability=False,
        g4_val_return=False, g7_robustness_gap=False,
    ),
    "robustness_focused": GateConfig(
        name="robustness_focused",
        description="Robustness-focused gates: trade count, return/DD ratio, robustness gap",
        g2_max_drawdown=False, g3_train_profitability=False,
        g4_val_return=False, g6_val_profit_factor=False,
    ),
}


# =============================================================================
# 4. UTILITY: LIST ALL ABLATION VARIANTS FOR STUDY DESIGN
# =============================================================================

def generate_single_gate_variants() -> Dict[str, GateConfig]:
    """
    Generate all single-gate-active variants for complete ablation.

    Returns 7 configs, each with exactly one gate active.
    Useful for measuring the marginal contribution of each gate.
    """
    gate_fields = [
        ("g1_min_trades", "G1: Minimum trade count only"),
        ("g2_max_drawdown", "G2: Maximum drawdown only"),
        ("g3_train_profitability", "G3: Train profitability only"),
        ("g4_val_return", "G4: Validation return only"),
        ("g5_return_dd_ratio", "G5: Return-to-drawdown ratio only"),
        ("g6_val_profit_factor", "G6: Validation profit factor only"),
        ("g7_robustness_gap", "G7: Robustness with gap penalty only"),
    ]

    variants = {}
    for field_name, desc in gate_fields:
        # Start with all gates off
        kwargs = {f: False for f, _ in gate_fields}
        # Enable only this one
        kwargs[field_name] = True
        gate_id = field_name.split("_", 1)[0]  # e.g., "g1"
        config = GateConfig(name=f"single_{gate_id}", description=desc, **kwargs)
        variants[f"single_{gate_id}"] = config

    return variants


def _resolve_gate_config(name: str) -> GateConfig:
    available = dict(ABLATION_PRESETS)
    available.update(generate_single_gate_variants())
    try:
        return available[name]
    except KeyError as exc:
        raise ValueError(f"Unknown gate configuration: {name}") from exc


def _apply_gate_config(pipeline_config: Any, gate_config: GateConfig) -> None:
    gate_flags = {
        "research_gate_g1_min_trades": gate_config.g1_min_trades,
        "research_gate_g2_max_drawdown": gate_config.g2_max_drawdown,
        "research_gate_g3_train_profitability": gate_config.g3_train_profitability,
        "research_gate_g4_val_return": gate_config.g4_val_return,
        "research_gate_g5_return_dd_ratio": gate_config.g5_return_dd_ratio,
        "research_gate_g6_val_profit_factor": gate_config.g6_val_profit_factor,
        "research_gate_g7_robustness_gap": gate_config.g7_robustness_gap,
    }
    for attr, value in gate_flags.items():
        setattr(pipeline_config, attr, value)

    # Research-only thresholds are kept separate from the core pipeline thresholds
    # so candidate generation and tuning stay constant across ablation runs.
    setattr(pipeline_config, "research_gate_g1_threshold", int(gate_config.g1_threshold) if gate_config.g1_threshold is not None else None)
    setattr(pipeline_config, "research_gate_g2_threshold", float(gate_config.g2_threshold) if gate_config.g2_threshold is not None else None)
    setattr(pipeline_config, "research_gate_g3_threshold", float(gate_config.g3_threshold) if gate_config.g3_threshold is not None else None)
    setattr(pipeline_config, "research_gate_g4_threshold", float(gate_config.g4_threshold) if gate_config.g4_threshold is not None else None)
    setattr(pipeline_config, "research_gate_g5_threshold", float(gate_config.g5_threshold) if gate_config.g5_threshold is not None else None)
    setattr(pipeline_config, "research_gate_g6_threshold", float(gate_config.g6_threshold) if gate_config.g6_threshold is not None else None)
    setattr(pipeline_config, "research_gate_g7_threshold", float(gate_config.g7_threshold) if gate_config.g7_threshold is not None else None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FXPM research tooling")
    parser.add_argument("--verify-dataset", dest="verify_dataset_path", help="Verify a frozen dataset directory")
    parser.add_argument("--write-sample-manifest", dest="manifest_out", help="Write a sample experiment manifest JSON file")
    parser.add_argument("--run-manifest", dest="run_manifest_path", help="Run a preregistered manifest against a frozen dataset")
    parser.add_argument("--dataset-path", help="Path to a frozen dataset directory")
    parser.add_argument("--config", dest="config_path", default="config.json", help="Path to config.json")
    parser.add_argument("--output-root", help="Root directory for research run outputs")
    parser.add_argument("--experiment-id", default="EXP-SAMPLE-001", help="Experiment identifier for sample manifest creation")
    parser.add_argument("--study-name", default="FXPM Research Study", help="Study name for sample manifest creation")
    parser.add_argument("--description", default="Manifest-driven offline benchmark run", help="Manifest description")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol override")
    parser.add_argument("--timeframes", nargs="*", help="Optional timeframe override")
    parser.add_argument("--gate-config", default="all_gates", help="Gate preset to record in the manifest")
    parser.add_argument("--overwrite", action="store_true", help="Allow reuse of an existing run directory")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verify_dataset_path:
        result = verify_dataset(_resolve_path(args.verify_dataset_path, RUNTIME_ROOT))
        print(json.dumps(result, indent=2))
        return 0 if result.get("passed") else 1

    if args.manifest_out:
        if not args.dataset_path:
            parser.error("--dataset-path is required when using --write-sample-manifest")
        manifest = write_sample_manifest(
            manifest_path=Path(args.manifest_out),
            dataset_path=args.dataset_path,
            experiment_id=args.experiment_id,
            study_name=args.study_name,
            description=args.description,
            gate_config=args.gate_config,
            symbols=args.symbols,
            timeframes=args.timeframes,
            config_path=args.config_path,
            output_root=args.output_root or str(DEFAULT_OUTPUT_ROOT),
        )
        print(json.dumps(manifest.to_dict(), indent=2))
        return 0

    if args.run_manifest_path:
        summary = run_experiment_manifest(
            manifest_path=Path(args.run_manifest_path),
            config_path=args.config_path,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2))
        return 0

    print("=== ABLATION PRESETS ===")
    for name, gc in ABLATION_PRESETS.items():
        print(f"  {name}: {gc.active_gate_count()} gates active")
        print(f"    -> {gc.active_gate_names()}")

    print("\n=== SINGLE-GATE VARIANTS ===")
    for name, gc in generate_single_gate_variants().items():
        print(f"  {name}: {gc.active_gate_names()}")

    print("\n=== EXPERIMENT MANIFEST (sample) ===")
    em = create_experiment_manifest(
        experiment_id="EXP-001",
        study_name="Gate Ablation Study A",
        description="Research manifest creation smoke test",
        random_seed=42,
        gate_config="all_gates",
        symbols=["EURUSD", "GBPUSD"],
        timeframes=["H1"],
    )
    print(json.dumps(em.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
