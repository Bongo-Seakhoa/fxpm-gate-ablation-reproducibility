"""
Section 6 Manifest — Versioning, Cache Keys, Deterministic Seeds
================================================================
Handles provenance, cache invalidation, and reproducibility infrastructure
for the Section 6 evidence-first empirical ablation study.

Every other section6_* module imports from here.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "2.0.0"
STUDY_ID = "ISM2026-SECTION6-REWORK"
DATASET_ID = "FBS-M5-HISTORICAL-v1"

REGIMES = ["TREND", "RANGE", "BREAKOUT", "CHOP"]
TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "D1"]

PRESET_ORDER = [
    "all_gates",
    "no_gates",
    "drawdown_only",
    "profitability_only",
    "minimal",
    "quality_focused",
    "robustness_focused",
]

# Integrity-only thresholds (non-ablatable, deliberately loose)
INTEGRITY_THRESHOLDS = {
    "train_min_profit_factor": 0.5,
    "train_min_return_pct": -30.0,
    "train_max_drawdown": 60.0,
    "regime_min_train_trades": 25,
    "regime_min_val_trades": 10,
}

# Symbols excluded from the 50-symbol universe
EXCLUDE_SYMBOLS = {
    "TONUSD", "AUDCHF", "CADCHF", "NZDCHF", "NZDCAD", "GBPNZD",
    "EURNZD", "GBPZAR", "EURZAR", "USDPLN", "USDSEK", "FR40",
}

MIN_ROWS_PER_SYMBOL = 200_000

# Files whose content determines the code hash (evidence invalidation)
CODE_HASH_MODULES = [
    "pm_pipeline.py",
    "pm_core.py",
    "pm_optuna.py",
    "pm_strategies.py",
    "pm_regime.py",
    "pm_research.py",
]


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------
def _sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file, read in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_code_hash(artifact_dir: Path) -> str:
    """SHA-256 over sorted source files that affect evidence generation.

    If any of these files change, the evidence build must be re-run.
    """
    h = hashlib.sha256()
    for name in sorted(CODE_HASH_MODULES):
        p = artifact_dir / name
        if p.exists():
            h.update(name.encode("utf-8"))
            h.update(_sha256_file(p).encode("utf-8"))
    return h.hexdigest()


def compute_config_hash(config_dict: dict) -> str:
    """SHA-256 of a deterministic JSON serialisation of pipeline config."""
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return _sha256_bytes(canonical.encode("utf-8"))


def compute_dataset_hash(manifest_path: Path) -> str:
    """SHA-256 of the frozen MANIFEST.json file itself."""
    return _sha256_file(manifest_path)


def build_cache_key(dataset_hash: str, code_hash: str, config_hash: str) -> str:
    """Composite cache key.  If this matches, evidence build can be skipped."""
    combined = f"{dataset_hash}:{code_hash}:{config_hash}"
    return _sha256_bytes(combined.encode("utf-8"))


# ---------------------------------------------------------------------------
# Deterministic per-slot seed
# ---------------------------------------------------------------------------
def slot_seed(dataset_id: str, symbol: str, timeframe: str,
              strategy_name: str, search_version: int = 1) -> int:
    """Deterministic seed for Optuna and any random tiebreaks.

    Returns an unsigned 31-bit integer derived from the concatenation of
    all input fields.  Identical inputs → identical seed regardless of
    worker count or processing order.
    """
    payload = f"{dataset_id}|{symbol}|{timeframe}|{strategy_name}|{search_version}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    # Unpack first 4 bytes as unsigned int, mask to 31 bits
    raw = struct.unpack(">I", digest[:4])[0]
    return raw & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Candidate tiebreak key
# ---------------------------------------------------------------------------
def candidate_sort_key(quality_score: float, strategy_name: str,
                       param_hash: str):
    """Return a tuple suitable for sorted(..., key=...).

    Order: quality_score DESC, strategy_name ASC, param_hash ASC.
    Using -quality_score for descending numeric sort.
    """
    return (-quality_score, strategy_name, param_hash)


def compute_param_hash(params: dict) -> str:
    """SHA-256 of a strategy's parameter dict (sorted, deterministic)."""
    canonical = json.dumps(params, sort_keys=True, default=str)
    return _sha256_bytes(canonical.encode("utf-8"))[:16]  # 16-char prefix


# ---------------------------------------------------------------------------
# Build Manifest
# ---------------------------------------------------------------------------
@dataclass
class BuildManifest:
    """Tracks the state of an evidence build (Stage B)."""
    study_id: str = STUDY_ID
    schema_version: str = SCHEMA_VERSION
    dataset_hash: str = ""
    code_hash: str = ""
    config_hash: str = ""
    cache_key: str = ""
    started_at: str = ""
    completed_at: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    total_slots: int = 0
    completed_slots: int = 0
    failed_slots: List[Dict[str, str]] = field(default_factory=list)
    worker_count: int = 3
    # Per-slot status: {(symbol, tf) -> "complete"/"failed"/"pending"}
    slot_status: Dict[str, str] = field(default_factory=dict)

    def slot_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}/{timeframe}"

    def mark_complete(self, symbol: str, timeframe: str) -> None:
        self.slot_status[self.slot_key(symbol, timeframe)] = "complete"
        self.completed_slots = sum(
            1 for v in self.slot_status.values() if v == "complete"
        )

    def mark_failed(self, symbol: str, timeframe: str, error: str) -> None:
        key = self.slot_key(symbol, timeframe)
        self.slot_status[key] = "failed"
        self.failed_slots = [
            e for e in self.failed_slots if e.get("slot") != key
        ]
        self.failed_slots.append({"slot": key, "symbol": symbol,
                                   "timeframe": timeframe, "error": error})

    def is_slot_done(self, symbol: str, timeframe: str) -> bool:
        return self.slot_status.get(self.slot_key(symbol, timeframe)) == "complete"

    def pending_slots(self, symbols: List[str],
                      timeframes: List[str]) -> List[tuple]:
        """Return (symbol, tf) pairs not yet completed."""
        out = []
        for s in symbols:
            for tf in timeframes:
                status = self.slot_status.get(self.slot_key(s, tf), "pending")
                if status != "complete":
                    out.append((s, tf))
        return out

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Path) -> "BuildManifest":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Remove any extra keys that aren't in the dataclass
        valid_keys = {fld.name for fld in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Preset Manifest
# ---------------------------------------------------------------------------
@dataclass
class PresetManifest:
    """Tracks the output of one preset materialisation (Stage C)."""
    study_id: str = STUDY_ID
    preset_name: str = ""
    gate_config: Dict[str, Any] = field(default_factory=dict)
    evidence_cache_key: str = ""
    total_slots: int = 0
    filled_slots: int = 0
    no_trade_slots: int = 0
    materialised_at: str = ""

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Path) -> "PresetManifest":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid_keys = {fld.name for fld in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Helper: now() in ISO format
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
