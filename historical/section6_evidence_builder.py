"""
Section 6 Evidence Builder — Gate-Neutral Candidate Generation
==============================================================
Builds the widest possible candidate evidence base with only loose
integrity rules.  All gate-specific filtering is deferred to
section6_preset_materializer.py (Stage C).

This module implements:
  Stage A — Frozen dataset verification
  Stage B — Evidence build (the expensive ~25-hour pass)

Design principles:
  - Build once, filter many: every candidate that *could* win under any
    preset is generated and persisted.
  - Deterministic: per-slot seeds ensure identical results regardless of
    worker count.
  - Resumable: interrupted builds pick up where they left off.
  - Auditable: every candidate record is versioned and traceable.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve bundled runtime for imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ARTIFACT_RUNTIME = (REPO_ROOT / "runtime").resolve()
if str(ARTIFACT_RUNTIME) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_RUNTIME))

from pm_core import (                         # type: ignore
    PipelineConfig,
    DataLoader,
    DataSplitter,
    FeatureComputer,
)
from pm_pipeline import (                     # type: ignore
    RegimeOptimizer,
)
from pm_strategies import StrategyRegistry    # type: ignore
from pm_research import (                     # type: ignore
    verify_dataset,
)

from section6_manifest import (
    SCHEMA_VERSION,
    STUDY_ID,
    DATASET_ID,
    REGIMES,
    TIMEFRAMES,
    INTEGRITY_THRESHOLDS,
    EXCLUDE_SYMBOLS,
    MIN_ROWS_PER_SYMBOL,
    BuildManifest,
    compute_code_hash,
    compute_config_hash,
    compute_dataset_hash,
    build_cache_key,
    slot_seed,
    compute_param_hash,
    now_iso,
)

logger = logging.getLogger("section6.evidence")

# ---------------------------------------------------------------------------
# Evidence-build PipelineConfig factory
# ---------------------------------------------------------------------------

def make_evidence_config(
    frozen_dir: Path,
    config_path: Optional[Path] = None,
    workers: int = 3,
) -> PipelineConfig:
    """Create a PipelineConfig tuned for gate-neutral evidence generation.

    Key overrides vs. production config:
      - All research gates DISABLED (evidence is gate-neutral).
      - Training eligibility thresholds set deliberately LOOSE so that the
        ablatable gates (G1–G7) become the binding constraints in Stage C.
      - Scoring mode = fx_backtester for consistent quality scores.
    """
    # Start from disk config if available, else defaults
    base_kwargs: Dict[str, Any] = {}
    resolved = config_path or (ARTIFACT_RUNTIME / "config.json")
    if resolved.exists():
        try:
            raw = json.loads(resolved.read_text(encoding="utf-8"))
            pipeline_payload = raw.get("pipeline", {})
            allowed = {f.name for f in dc_fields(PipelineConfig)}
            base_kwargs = {k: v for k, v in pipeline_payload.items() if k in allowed}
        except Exception as exc:
            logger.warning("Could not load config from %s: %s", resolved, exc)

    # Force overrides for evidence build
    base_kwargs.update({
        "data_dir":                   str(frozen_dir),
        "optimization_max_workers":   workers,
        "scoring_mode":               "fx_backtester",
        "use_regime_optimization":    True,
        "regime_enable_hyperparam_tuning": True,
        "regime_hyperparam_top_k":    3,
        "regime_hyperparam_max_combos": 30,
        "regime_validation_top_k":    5,
        # Loose integrity thresholds (non-ablatable)
        "train_min_profit_factor":    INTEGRITY_THRESHOLDS["train_min_profit_factor"],
        "train_min_return_pct":       INTEGRITY_THRESHOLDS["train_min_return_pct"],
        "train_max_drawdown":         INTEGRITY_THRESHOLDS["train_max_drawdown"],
        "regime_min_train_trades":    INTEGRITY_THRESHOLDS["regime_min_train_trades"],
        "regime_min_val_trades":      INTEGRITY_THRESHOLDS["regime_min_val_trades"],
        "fx_val_max_drawdown":        INTEGRITY_THRESHOLDS["train_max_drawdown"],
        # Allow losing winners in evidence (gates will filter in Stage C)
        #
        # Gate-neutrality rationale:
        #   - Research gates (G1–G7) are read by RegimeOptimizer.__init__ from
        #     PipelineConfig via getattr(..., 'research_gate_g*', True).  Since
        #     PipelineConfig has no such attributes, they default to True inside
        #     the optimizer.  However, these gates are only enforced in
        #     _select_best_for_regime(), which Stage B never calls.  Stage B
        #     calls _collect_candidates() → _apply_training_eligibility_gates()
        #     only, so the research gate defaults are inert.
        #   - The thresholds below widen PipelineConfig's own eligibility so
        #     _collect_candidates() retains the broadest possible candidate pool.
        #   - regime_min_val_return_dd_ratio is set to a small positive value
        #     because PipelineConfig.__post_init__ clamps non-positive values
        #     back to 1.0.  0.01 survives the clamp and is effectively disabled.
        "regime_allow_losing_winners":  True,
        "regime_min_val_profit_factor": 0.0,
        "regime_min_val_return_pct":    -100.0,
        "regime_min_val_return_dd_ratio": 0.01,
    })

    config = PipelineConfig(**base_kwargs)
    gate_overrides = {
        "research_gate_g1_min_trades": False,
        "research_gate_g2_max_drawdown": False,
        "research_gate_g3_train_profitability": False,
        "research_gate_g4_val_return": False,
        "research_gate_g5_return_dd_ratio": False,
        "research_gate_g6_val_profit_factor": False,
        "research_gate_g7_robustness_gap": False,
        "research_gate_g1_threshold": int(getattr(config, "regime_min_val_trades", INTEGRITY_THRESHOLDS["regime_min_val_trades"])),
        "research_gate_g2_threshold": float(getattr(config, "fx_val_max_drawdown", INTEGRITY_THRESHOLDS["train_max_drawdown"])),
        "research_gate_g3_threshold": 0.0,
        "research_gate_g4_threshold": float(getattr(config, "regime_min_val_return_pct", -100.0)),
        "research_gate_g5_threshold": float(getattr(config, "regime_min_val_return_dd_ratio", 0.01)),
        "research_gate_g6_threshold": float(getattr(config, "regime_min_val_profit_factor", 0.0)),
        "research_gate_g7_threshold": 0.0,
    }
    for name, value in gate_overrides.items():
        setattr(config, name, value)
    return config


def evidence_config_hash_payload(config: PipelineConfig) -> Dict[str, Any]:
    """Return the config fields that materially shape Stage B evidence."""
    return {
        "optimization_max_workers": getattr(config, "optimization_max_workers", 1),
        "use_regime_optimization": getattr(config, "use_regime_optimization", True),
        "regime_enable_hyperparam_tuning": getattr(config, "regime_enable_hyperparam_tuning", True),
        "regime_hyperparam_top_k": getattr(config, "regime_hyperparam_top_k", 3),
        "regime_hyperparam_max_combos": getattr(config, "regime_hyperparam_max_combos", 30),
        "regime_validation_top_k": getattr(config, "regime_validation_top_k", 5),
        "train_min_profit_factor": getattr(config, "train_min_profit_factor", 0.5),
        "train_min_return_pct": getattr(config, "train_min_return_pct", -30.0),
        "train_max_drawdown": getattr(config, "train_max_drawdown", 60.0),
        "regime_min_train_trades": getattr(config, "regime_min_train_trades", 25),
        "regime_min_val_trades": getattr(config, "regime_min_val_trades", 10),
        "fx_val_max_drawdown": getattr(config, "fx_val_max_drawdown", 60.0),
        "regime_allow_losing_winners": getattr(config, "regime_allow_losing_winners", False),
        "regime_min_val_profit_factor": getattr(config, "regime_min_val_profit_factor", 0.0),
        "regime_min_val_return_pct": getattr(config, "regime_min_val_return_pct", -100.0),
        "regime_min_val_return_dd_ratio": getattr(config, "regime_min_val_return_dd_ratio", 1.0),
        "research_gate_g1_min_trades": getattr(config, "research_gate_g1_min_trades", True),
        "research_gate_g2_max_drawdown": getattr(config, "research_gate_g2_max_drawdown", True),
        "research_gate_g3_train_profitability": getattr(config, "research_gate_g3_train_profitability", True),
        "research_gate_g4_val_return": getattr(config, "research_gate_g4_val_return", True),
        "research_gate_g5_return_dd_ratio": getattr(config, "research_gate_g5_return_dd_ratio", True),
        "research_gate_g6_val_profit_factor": getattr(config, "research_gate_g6_val_profit_factor", True),
        "research_gate_g7_robustness_gap": getattr(config, "research_gate_g7_robustness_gap", True),
        "research_gate_g1_threshold": getattr(config, "research_gate_g1_threshold", None),
        "research_gate_g2_threshold": getattr(config, "research_gate_g2_threshold", None),
        "research_gate_g3_threshold": getattr(config, "research_gate_g3_threshold", None),
        "research_gate_g4_threshold": getattr(config, "research_gate_g4_threshold", None),
        "research_gate_g5_threshold": getattr(config, "research_gate_g5_threshold", None),
        "research_gate_g6_threshold": getattr(config, "research_gate_g6_threshold", None),
        "research_gate_g7_threshold": getattr(config, "research_gate_g7_threshold", None),
        "scoring_mode": getattr(config, "scoring_mode", "fx_backtester"),
        "initial_capital": getattr(config, "initial_capital", 10000.0),
        "train_pct": getattr(config, "train_pct", 80.0),
        "val_pct": getattr(config, "val_pct", 30.0),
        "overlap_pct": getattr(config, "overlap_pct", 10.0),
    }


# ---------------------------------------------------------------------------
# Per-symbol worker function (runs in subprocess)
# ---------------------------------------------------------------------------

def _build_symbol(
    frozen_dir: Path,
    output_dir: Path,
    symbol: str,
    timeframes: List[str],
    config_path: Optional[Path],
    workers: int,
    code_hash: str,
    config_hash: str,
) -> Dict[str, Any]:
    """Process one symbol: load data, compute features, collect candidates.

    This function runs in a subprocess.  It returns a summary dict per
    timeframe with candidate counts and timings.

    Returns:
        {timeframe: {"status": "complete"/"failed", "candidates": N,
                      "duration": secs, "error": ""}}
    """
    results: Dict[str, Any] = {}
    config = make_evidence_config(frozen_dir, config_path, workers=1)

    # Initialise heavy objects inside the worker
    data_loader = DataLoader(config.data_dir)
    splitter = DataSplitter(config)
    strategies = StrategyRegistry.get_all_instances()
    regime_optimizer = RegimeOptimizer(config)
    regime_params_file = getattr(config, "regime_params_file", "regime_params.json")

    for tf in timeframes:
        t0 = time.time()
        tf_out = output_dir / "candidates" / symbol
        tf_out.mkdir(parents=True, exist_ok=True)
        pq_path = tf_out / f"{tf}.parquet"

        try:
            # 1. Load data
            raw_data = data_loader.get_data(symbol, tf)
            if raw_data is None or len(raw_data) < 200:
                results[tf] = {
                    "status": "skipped",
                    "candidates": 0,
                    "duration": time.time() - t0,
                    "error": f"Insufficient data ({len(raw_data) if raw_data is not None else 0} rows)",
                }
                continue

            # 2. Split train / val
            split_idx = splitter.get_split_indices(len(raw_data))
            train_start, train_end = split_idx["train"]
            val_start, val_end = split_idx["val"]

            # 3. Compute features (includes regime detection)
            full_features = FeatureComputer.compute_all(
                raw_data, symbol=symbol, timeframe=tf,
                regime_params_file=regime_params_file,
            )
            train_features = full_features.iloc[train_start:train_end]
            val_features = full_features.iloc[val_start:val_end]

            # 4. Cache features to Parquet
            feat_dir = output_dir / "features" / symbol
            feat_dir.mkdir(parents=True, exist_ok=True)
            feat_path = feat_dir / f"{tf}.parquet"
            if not feat_path.exists():
                full_features.to_parquet(feat_path, index=True)

            # 5. Set per-slot seed on the Optuna optimizer
            if regime_optimizer.optuna_optimizer is not None:
                base_seed = slot_seed(DATASET_ID, symbol, tf, "combined")
                regime_optimizer.optuna_optimizer.config.seed = base_seed

            # 6. Collect ALL candidates (screening + Optuna tuning)
            candidates = regime_optimizer._collect_candidates(
                symbol, tf, train_features, val_features, strategies
            )

            if not candidates:
                # Write empty Parquet (header only)
                _write_empty_parquet(pq_path, symbol, tf)
                results[tf] = {
                    "status": "complete",
                    "candidates": 0,
                    "duration": time.time() - t0,
                    "error": "",
                }
                continue

            # 7. Score every candidate per regime and flatten to rows
            rows = _candidates_to_rows(
                candidates, symbol, tf,
                regime_optimizer, code_hash, config_hash,
            )

            # 8. Persist to Parquet
            df = pd.DataFrame(rows)
            df.to_parquet(pq_path, index=False)

            results[tf] = {
                "status": "complete",
                "candidates": len(rows),
                "duration": time.time() - t0,
                "error": "",
            }
            logger.info(
                "[%s] [%s] %d candidates persisted (%.1fs)",
                symbol, tf, len(rows), time.time() - t0,
            )

        except MemoryError:
            results[tf] = {
                "status": "failed",
                "candidates": 0,
                "duration": time.time() - t0,
                "error": "MemoryError — slot too large for available RAM",
            }
            logger.error("[%s] [%s] MemoryError", symbol, tf)

        except Exception as exc:
            results[tf] = {
                "status": "failed",
                "candidates": 0,
                "duration": time.time() - t0,
                "error": str(exc)[:500],
            }
            logger.error("[%s] [%s] Failed: %s", symbol, tf, exc, exc_info=True)

    return results


# ---------------------------------------------------------------------------
# Candidate → flat row conversion
# ---------------------------------------------------------------------------

def _extract_regime_metrics(
    regime_metrics: Dict[str, Dict[str, Any]],
    regime: str,
    prefix: str,
) -> Dict[str, Any]:
    """Pull per-regime metrics into prefixed flat columns."""
    m = regime_metrics.get(regime, {})
    out: Dict[str, Any] = {}
    out[f"{prefix}_trades"] = m.get("total_trades", 0)
    out[f"{prefix}_pf"] = m.get("profit_factor", 0.0)
    out[f"{prefix}_return"] = m.get("total_return_pct", 0.0)
    out[f"{prefix}_dd"] = m.get("max_drawdown_pct", 0.0)
    out[f"{prefix}_sharpe"] = m.get("sharpe_approx", 0.0)
    out[f"{prefix}_win_rate"] = m.get("win_rate", 0.0)
    return out


def _candidates_to_rows(
    candidates: List[Dict[str, Any]],
    symbol: str,
    timeframe: str,
    optimizer: RegimeOptimizer,
    code_hash: str,
    config_hash: str,
) -> List[Dict[str, Any]]:
    """Convert the raw candidate list into flat dicts ready for Parquet.

    Each row includes per-regime train+val metrics and a preset-independent
    quality score computed by _compute_regime_score.
    """
    rows: List[Dict[str, Any]] = []

    for cand in candidates:
        params = cand.get("params", {})
        p_hash = compute_param_hash(params)
        base_seed = slot_seed(
            DATASET_ID, symbol, timeframe,
            cand["strategy_name"],
        )

        row: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy_name": cand["strategy_name"],
            "param_hash": p_hash,
            "params_json": json.dumps(params, sort_keys=True, default=str),
            "is_tuned": bool(cand.get("is_tuned", False)),
            "slot_seed": base_seed,
            "code_hash": code_hash,
            "config_hash": config_hash,
        }

        train_regime = cand.get("train_regime_metrics", {})
        val_regime = cand.get("val_regime_metrics", {})

        for regime in REGIMES:
            r_prefix = regime  # e.g. "TREND"

            # Train metrics
            train_cols = _extract_regime_metrics(
                train_regime, regime, f"{r_prefix}_train",
            )
            row.update(train_cols)

            # Val metrics
            val_cols = _extract_regime_metrics(
                val_regime, regime, f"{r_prefix}_val",
            )
            row.update(val_cols)

            # Compute preset-independent quality score
            t_metrics = train_regime.get(regime, {})
            v_metrics = val_regime.get(regime, {})
            try:
                score = optimizer._compute_regime_score(t_metrics, v_metrics)
            except Exception:
                score = -float("inf")

            # Also store train-only and val-only scores for G7 robustness
            try:
                t_full = optimizer._bucket_to_full_metrics(t_metrics)
                v_full = optimizer._bucket_to_full_metrics(v_metrics)
                t_score = optimizer.scorer.score(t_full, purpose="selection")
                v_score = optimizer.scorer.score(v_full, purpose="selection")
            except Exception:
                t_score = 0.0
                v_score = 0.0

            row[f"{r_prefix}_train_score"] = float(t_score) if math.isfinite(t_score) else 0.0
            row[f"{r_prefix}_val_score"] = float(v_score) if math.isfinite(v_score) else 0.0
            row[f"{r_prefix}_quality_score"] = float(score) if math.isfinite(score) else -999.0

        rows.append(row)

    return rows


def _write_empty_parquet(path: Path, symbol: str, timeframe: str) -> None:
    """Write a header-only Parquet for a slot with zero candidates."""
    cols: Dict[str, Any] = {
        "symbol": pd.Series(dtype="str"),
        "timeframe": pd.Series(dtype="str"),
        "strategy_name": pd.Series(dtype="str"),
        "param_hash": pd.Series(dtype="str"),
        "params_json": pd.Series(dtype="str"),
        "is_tuned": pd.Series(dtype="bool"),
        "slot_seed": pd.Series(dtype="int64"),
        "code_hash": pd.Series(dtype="str"),
        "config_hash": pd.Series(dtype="str"),
    }
    for regime in REGIMES:
        for split in ("train", "val"):
            for metric in ("trades", "pf", "return", "dd", "sharpe", "win_rate"):
                cols[f"{regime}_{split}_{metric}"] = pd.Series(dtype="float64")
        cols[f"{regime}_train_score"] = pd.Series(dtype="float64")
        cols[f"{regime}_val_score"] = pd.Series(dtype="float64")
        cols[f"{regime}_quality_score"] = pd.Series(dtype="float64")
    df = pd.DataFrame(cols)
    df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Main EvidenceBuilder class
# ---------------------------------------------------------------------------

class EvidenceBuilder:
    """Orchestrates the full evidence build (Stages A + B)."""

    def __init__(
        self,
        frozen_dir: Path,
        output_dir: Path,
        config_path: Optional[Path] = None,
        workers: int = 3,
        resume: bool = True,
    ):
        self.frozen_dir = Path(frozen_dir)
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        self.workers = workers
        self.resume = resume

        # Will be computed during build
        self._code_hash: str = ""
        self._config_hash: str = ""
        self._dataset_hash: str = ""
        self._cache_key: str = ""

    # ------------------------------------------------------------------
    # Stage A: data verification
    # ------------------------------------------------------------------
    def verify_frozen_data(self) -> Dict[str, Any]:
        """Verify frozen dataset integrity against MANIFEST.json.

        Raises RuntimeError if verification fails.
        Returns the verification result dict.
        """
        manifest_path = self.frozen_dir / "MANIFEST.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"No MANIFEST.json in frozen directory: {self.frozen_dir}"
            )

        result = verify_dataset(self.frozen_dir)
        if not result.get("passed", False):
            failed_files = [
                fn for fn, info in result.get("files", {}).items()
                if info.get("status") != "OK"
            ]
            raise RuntimeError(
                f"Dataset verification FAILED. Bad files: {failed_files}"
            )

        logger.info("Stage A: Dataset verification PASSED (%s)",
                     result.get("manifest_id", "?"))

        # Persist verification result
        ver_path = self.output_dir / "dataset_verification.json"
        ver_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ver_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result

    # ------------------------------------------------------------------
    # Compute hashes and cache key
    # ------------------------------------------------------------------
    def _compute_hashes(self) -> None:
        """Compute code, config, and dataset hashes for cache validation."""
        self._code_hash = compute_code_hash(ARTIFACT_RUNTIME)
        config = make_evidence_config(self.frozen_dir, self.config_path, self.workers)
        config_dict = evidence_config_hash_payload(config)
        self._config_hash = compute_config_hash(config_dict)
        self._dataset_hash = compute_dataset_hash(self.frozen_dir / "MANIFEST.json")
        self._cache_key = build_cache_key(
            self._dataset_hash, self._code_hash, self._config_hash
        )
        logger.info("Cache key: %s", self._cache_key[:16])

    # ------------------------------------------------------------------
    # Discover symbols from frozen data
    # ------------------------------------------------------------------
    def _discover_symbols(self) -> List[str]:
        """Find all symbols available in the frozen dataset directory."""
        seen: set = set()
        for csv_path in sorted(self.frozen_dir.glob("*.csv")):
            parts = csv_path.stem.rsplit("_", 1)
            symbol = parts[0] if len(parts) == 2 else csv_path.stem
            if symbol not in EXCLUDE_SYMBOLS:
                seen.add(symbol)
        symbols = sorted(seen)
        logger.info("Discovered %d symbols in frozen dataset", len(symbols))
        return symbols

    # ------------------------------------------------------------------
    # Load or create build manifest
    # ------------------------------------------------------------------
    def _load_or_create_manifest(
        self, symbols: List[str], timeframes: List[str],
    ) -> BuildManifest:
        """Load an existing manifest (for resume) or create a new one."""
        manifest_path = self.output_dir / "build_manifest.json"

        if self.resume and manifest_path.exists():
            try:
                existing = BuildManifest.from_json(manifest_path)
                if existing.cache_key == self._cache_key:
                    logger.info(
                        "Resuming build: %d/%d slots complete",
                        existing.completed_slots, existing.total_slots,
                    )
                    return existing
                else:
                    logger.warning(
                        "Cache key mismatch (code/config/data changed) — "
                        "starting fresh build"
                    )
            except Exception as exc:
                logger.warning("Could not load manifest: %s — starting fresh", exc)

        manifest = BuildManifest(
            study_id=STUDY_ID,
            schema_version=SCHEMA_VERSION,
            dataset_hash=self._dataset_hash,
            code_hash=self._code_hash,
            config_hash=self._config_hash,
            cache_key=self._cache_key,
            started_at=now_iso(),
            symbols=symbols,
            timeframes=timeframes,
            total_slots=len(symbols) * len(timeframes),
            worker_count=self.workers,
        )
        return manifest

    # ------------------------------------------------------------------
    # Stage B: evidence build
    # ------------------------------------------------------------------
    def build_evidence(self) -> BuildManifest:
        """Run the full evidence build.

        Flow:
          1. verify_frozen_data()        (Stage A)
          2. Compute hashes + cache key
          3. If build already complete with matching key → skip
          4. Enumerate symbol × timeframe slots
          5. Dispatch symbols to worker pool
          6. Persist build_manifest.json + slot_index.csv

        Returns: BuildManifest with final completion status.
        """
        # Stage A
        self.verify_frozen_data()

        # Hashes
        self._compute_hashes()

        # Check if build is already complete
        manifest_path = self.output_dir / "build_manifest.json"
        if self.resume and manifest_path.exists():
            try:
                existing = BuildManifest.from_json(manifest_path)
                if (existing.cache_key == self._cache_key
                        and existing.completed_at is not None
                        and existing.completed_slots == existing.total_slots):
                    logger.info(
                        "Evidence build already complete (cache key match). "
                        "Skipping."
                    )
                    return existing
            except Exception:
                pass

        # Discover symbols and build manifest
        symbols = self._discover_symbols()
        timeframes = TIMEFRAMES
        manifest = self._load_or_create_manifest(symbols, timeframes)

        # Determine pending work
        pending = manifest.pending_slots(symbols, timeframes)
        if not pending:
            logger.info("All slots already complete — nothing to do")
            manifest.completed_at = now_iso()
            manifest.to_json(manifest_path)
            self._build_slot_index(manifest)
            return manifest

        # Group pending slots by symbol for efficient processing
        symbol_tfs: Dict[str, List[str]] = {}
        for sym, tf in pending:
            symbol_tfs.setdefault(sym, []).append(tf)

        pending_symbols = sorted(symbol_tfs.keys())
        logger.info(
            "Evidence build: %d pending slots across %d symbols "
            "(workers=%d)",
            len(pending), len(pending_symbols), self.workers,
        )

        # Create output directories
        (self.output_dir / "candidates").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "features").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Dispatch to workers (parallelise over symbols)
        build_log: List[Dict[str, Any]] = []
        total_t0 = time.time()

        if self.workers <= 1:
            # Sequential execution — simpler debugging
            for sym in pending_symbols:
                tfs = symbol_tfs[sym]
                result = _build_symbol(
                    self.frozen_dir, self.output_dir, sym, tfs,
                    self.config_path, 1, self._code_hash, self._config_hash,
                )
                self._update_manifest_from_result(manifest, sym, result, build_log)
                manifest.to_json(manifest_path)  # Save after each symbol
        else:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                futures = {}
                for sym in pending_symbols:
                    tfs = symbol_tfs[sym]
                    f = pool.submit(
                        _build_symbol,
                        self.frozen_dir, self.output_dir, sym, tfs,
                        self.config_path, 1,
                        self._code_hash, self._config_hash,
                    )
                    futures[f] = sym

                for f in as_completed(futures):
                    sym = futures[f]
                    try:
                        result = f.result()
                    except Exception as exc:
                        # Whole-symbol failure
                        result = {
                            tf: {
                                "status": "failed",
                                "candidates": 0,
                                "duration": 0,
                                "error": str(exc)[:500],
                            }
                            for tf in symbol_tfs[sym]
                        }
                        logger.error("[%s] Worker crashed: %s", sym, exc)

                    self._update_manifest_from_result(
                        manifest, sym, result, build_log,
                    )
                    manifest.to_json(manifest_path)  # Save after each symbol

        # Finalise
        total_duration = time.time() - total_t0
        manifest.completed_at = now_iso()
        manifest.to_json(manifest_path)

        # Build slot index
        self._build_slot_index(manifest)

        # Write build log
        log_path = self.output_dir / "logs" / "build_log.jsonl"
        with open(log_path, "w", encoding="utf-8") as f:
            for entry in build_log:
                f.write(json.dumps(entry, default=str) + "\n")

        logger.info(
            "Evidence build complete: %d/%d slots in %.1f hours",
            manifest.completed_slots, manifest.total_slots,
            total_duration / 3600,
        )

        return manifest

    # ------------------------------------------------------------------
    # Manifest updates
    # ------------------------------------------------------------------
    def _update_manifest_from_result(
        self,
        manifest: BuildManifest,
        symbol: str,
        result: Dict[str, Dict[str, Any]],
        build_log: List[Dict[str, Any]],
    ) -> None:
        """Update BuildManifest from a _build_symbol result."""
        for tf, info in result.items():
            status = info.get("status", "failed")
            if status == "complete":
                manifest.mark_complete(symbol, tf)
            elif status == "failed":
                manifest.mark_failed(symbol, tf, info.get("error", "unknown"))
            elif status == "skipped":
                # Skipped slots (insufficient data) count as complete
                manifest.mark_complete(symbol, tf)

            build_log.append({
                "symbol": symbol,
                "timeframe": tf,
                "status": status,
                "candidates": info.get("candidates", 0),
                "duration_seconds": info.get("duration", 0),
                "error": info.get("error", ""),
            })

    # ------------------------------------------------------------------
    # Slot index
    # ------------------------------------------------------------------
    def _build_slot_index(self, manifest: BuildManifest) -> None:
        """Create slot_index.csv summarising all slots."""
        rows = []
        for sym in manifest.symbols:
            for tf in manifest.timeframes:
                key = manifest.slot_key(sym, tf)
                status = manifest.slot_status.get(key, "pending")
                pq_path = f"candidates/{sym}/{tf}.parquet"
                pq_full = self.output_dir / pq_path

                candidate_count = 0
                if pq_full.exists():
                    try:
                        df = pd.read_parquet(pq_full)
                        candidate_count = len(df)
                    except Exception:
                        pass

                error = ""
                for fe in manifest.failed_slots:
                    if fe.get("slot") == key:
                        error = fe.get("error", "")
                        break

                rows.append({
                    "symbol": sym,
                    "timeframe": tf,
                    "candidate_count": candidate_count,
                    "parquet_path": pq_path,
                    "status": status,
                    "error_message": error,
                })

        idx_df = pd.DataFrame(rows)
        idx_path = self.output_dir / "slot_index.csv"
        idx_df.to_csv(idx_path, index=False)
        logger.info("Slot index: %s (%d rows)", idx_path, len(rows))
