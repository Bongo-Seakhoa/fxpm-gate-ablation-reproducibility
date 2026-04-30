"""
Section 6 Preset Materialiser — Gate Filtering Over Frozen Evidence
===================================================================
Reads the candidate evidence built by section6_evidence_builder.py and
applies one gate preset as a pure post-processing filter.

This is Stage C: cheap, deterministic, and fully auditable.
Each preset run produces winners, a rejection ledger, and a manifest.

Design:
  - Never re-runs backtests or Optuna — reads Parquet only.
  - Gates are implemented as simple predicates over stored metrics.
  - Deterministic candidate ordering via (quality_score DESC,
    strategy_name ASC, param_hash ASC).
  - Rejection ledger records every decision for auditability.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Artifact imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ARTIFACT_RUNTIME = (REPO_ROOT / "runtime").resolve()
if str(ARTIFACT_RUNTIME) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_RUNTIME))

from pm_research import GateConfig, ABLATION_PRESETS  # type: ignore

from section6_manifest import (
    REGIMES,
    PRESET_ORDER,
    BuildManifest,
    PresetManifest,
    now_iso,
)

logger = logging.getLogger("section6.materializer")


# ---------------------------------------------------------------------------
# Gate checks — clean predicates over stored metrics
# ---------------------------------------------------------------------------

def check_gates(
    row: dict,
    regime: str,
    gc: GateConfig,
) -> Tuple[bool, str, List[str]]:
    """Apply all enabled gates for one candidate in one regime.

    Args:
        row: flat candidate dict (from Parquet row)
        regime: e.g. "TREND"
        gc: GateConfig defining which gates are active and thresholds

    Returns:
        (passed, reason_string, list_of_failed_gate_ids)
    """
    r = f"{regime}_"  # column prefix
    failed: List[str] = []

    # G1: Minimum validation trades
    if gc.g1_min_trades:
        threshold = gc.g1_threshold if gc.g1_threshold is not None else 5
        val_trades = row.get(f"{r}val_trades", 0)
        if val_trades < threshold:
            failed.append(f"G1: val_trades {val_trades} < {threshold}")

    # G2: Maximum validation drawdown
    if gc.g2_max_drawdown:
        threshold = gc.g2_threshold if gc.g2_threshold is not None else 20.0
        val_dd = row.get(f"{r}val_dd", 100.0)
        if val_dd >= threshold:
            failed.append(f"G2: val_dd {val_dd:.1f}% >= {threshold:.1f}%")

    # G3: Training profitability floor
    if gc.g3_train_profitability:
        threshold = gc.g3_threshold if gc.g3_threshold is not None else 0.0
        train_return = row.get(f"{r}train_return", -999.0)
        if train_return < threshold:
            failed.append(f"G3: train_return {train_return:.2f}% < {threshold:.1f}%")

    # G4: Validation return minimum
    if gc.g4_val_return:
        threshold = gc.g4_threshold if gc.g4_threshold is not None else 5.0
        val_return = row.get(f"{r}val_return", -999.0)
        if val_return < threshold:
            failed.append(f"G4: val_return {val_return:.2f}% < {threshold:.1f}%")

    # G5: Return-to-drawdown ratio
    if gc.g5_return_dd_ratio:
        threshold = gc.g5_threshold if gc.g5_threshold is not None else 1.0
        val_dd = max(row.get(f"{r}val_dd", 0.01), 0.01)
        val_return = row.get(f"{r}val_return", 0.0)
        ratio = val_return / val_dd
        if ratio < threshold:
            failed.append(f"G5: ret/dd {ratio:.2f} < {threshold:.1f}")

    # G6: Validation profit factor
    if gc.g6_val_profit_factor:
        threshold = gc.g6_threshold if gc.g6_threshold is not None else 1.0
        val_pf = row.get(f"{r}val_pf", 0.0)
        if val_pf < threshold:
            failed.append(f"G6: val_pf {val_pf:.2f} < {threshold:.2f}")

    # G7: Robustness gap (val_score / train_score)
    if gc.g7_robustness_gap:
        threshold = gc.g7_threshold if gc.g7_threshold is not None else 0.75
        train_score = row.get(f"{r}train_score", 0.001)
        val_score = row.get(f"{r}val_score", 0.0)
        if train_score > 0.001:
            robustness = val_score / train_score
        else:
            robustness = 0.0
        if robustness < threshold:
            failed.append(f"G7: robustness {robustness:.2f} < {threshold:.2f}")

    if failed:
        return False, "; ".join(failed), [f.split(":")[0] for f in failed]
    return True, "PASS", []


# ---------------------------------------------------------------------------
# Active gate list (for manifest / ledger metadata)
# ---------------------------------------------------------------------------

def active_gate_ids(gc: GateConfig) -> List[str]:
    """Return list of active gate identifiers for a GateConfig."""
    gates = []
    if gc.g1_min_trades:         gates.append("G1")
    if gc.g2_max_drawdown:       gates.append("G2")
    if gc.g3_train_profitability: gates.append("G3")
    if gc.g4_val_return:         gates.append("G4")
    if gc.g5_return_dd_ratio:    gates.append("G5")
    if gc.g6_val_profit_factor:  gates.append("G6")
    if gc.g7_robustness_gap:     gates.append("G7")
    return gates


# ---------------------------------------------------------------------------
# Preset Materializer
# ---------------------------------------------------------------------------

class PresetMaterializer:
    """Apply one gate preset over the frozen evidence, emitting winners
    and a rejection ledger.
    """

    def __init__(
        self,
        evidence_dir: Path,
        output_dir: Path,
        gate_config: GateConfig,
    ):
        self.evidence_dir = Path(evidence_dir)
        self.output_dir = Path(output_dir)
        self.gate_config = gate_config

    def materialise(self) -> PresetManifest:
        """Process all slots for this preset.

        Flow:
          1. Load slot_index.csv
          2. Load build_manifest.json → extract cache_key
          3. For each completed slot, load candidates and select winner
          4. Write winners.json, winner_table.csv, rejection_log.jsonl
          5. Write preset_manifest.json

        Returns: PresetManifest with fill statistics.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load evidence metadata
        slot_index = pd.read_csv(self.evidence_dir / "slot_index.csv")
        build_manifest = BuildManifest.from_json(
            self.evidence_dir / "build_manifest.json"
        )

        active_gates = active_gate_ids(self.gate_config)
        logger.info(
            "Materialising preset '%s' — gates: %s",
            self.gate_config.name, active_gates or ["none"],
        )

        winners: List[Dict[str, Any]] = []
        rejection_log: List[Dict[str, Any]] = []
        total_slots = 0
        filled_slots = 0
        no_trade_slots = 0

        for _, idx_row in slot_index.iterrows():
            symbol = idx_row["symbol"]
            tf = idx_row["timeframe"]
            status = idx_row["status"]

            if status not in ("complete",):
                # Failed/pending slots → NO_TRADE for all regimes
                slot_total, slot_no_trade = self._append_no_trade_entries(
                    rejection_log, symbol, tf, active_gates,
                    f"evidence slot {status}",
                )
                total_slots += slot_total
                no_trade_slots += slot_no_trade
                continue

            # Load candidate Parquet
            pq_path = self.evidence_dir / idx_row["parquet_path"]
            if not pq_path.exists():
                slot_total, slot_no_trade = self._append_no_trade_entries(
                    rejection_log, symbol, tf, active_gates,
                    "parquet file missing",
                )
                total_slots += slot_total
                no_trade_slots += slot_no_trade
                continue

            try:
                candidates_df = pd.read_parquet(pq_path)
            except Exception as exc:
                logger.error("Failed to read %s: %s", pq_path, exc)
                slot_total, slot_no_trade = self._append_no_trade_entries(
                    rejection_log, symbol, tf, active_gates,
                    f"parquet read failed: {str(exc)[:240]}",
                )
                total_slots += slot_total
                no_trade_slots += slot_no_trade
                continue

            if len(candidates_df) == 0:
                slot_total, slot_no_trade = self._append_no_trade_entries(
                    rejection_log, symbol, tf, active_gates,
                    "zero candidates in evidence",
                )
                total_slots += slot_total
                no_trade_slots += slot_no_trade
                continue

            # Process each regime
            for regime in REGIMES:
                total_slots += 1
                winner, slot_rejections = self._select_winner(
                    candidates_df, symbol, tf, regime,
                )
                train_trades_col = f"{regime}_train_trades"
                regime_candidate_count = int((candidates_df[train_trades_col] > 0).sum())
                if winner is not None:
                    candidates_considered = int(winner.get("rank", 1))
                    slot_reason = ""
                elif regime_candidate_count == 0:
                    candidates_considered = 0
                    slot_reason = "no candidates with training trades for regime"
                elif slot_rejections:
                    candidates_considered = min(regime_candidate_count, 20)
                    slot_reason = "all considered candidates failed active gates"
                else:
                    candidates_considered = min(regime_candidate_count, 20)
                    slot_reason = "no winner selected"
                if winner is not None:
                    filled_slots += 1
                    winners.append(winner)
                else:
                    no_trade_slots += 1

                # Build rejection log entry
                binding = set()
                for rej in slot_rejections:
                    for gid in rej.get("failed_gates", []):
                        binding.add(gid)

                rejection_log.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "regime": regime,
                    "candidate_count": int(len(candidates_df)),
                    "candidates_considered": candidates_considered,
                    "winner": winner,
                    "rejections": slot_rejections,
                    "gates_applied": active_gates,
                    "binding_gates": sorted(binding),
                    "outcome": "FILLED" if winner else "NO_TRADE",
                    "reason": slot_reason,
                })

        # Write outputs
        self._write_winners(winners)
        self._write_rejection_log(rejection_log)

        manifest = PresetManifest(
            preset_name=self.gate_config.name,
            gate_config=asdict(self.gate_config),
            evidence_cache_key=build_manifest.cache_key,
            total_slots=total_slots,
            filled_slots=filled_slots,
            no_trade_slots=no_trade_slots,
            materialised_at=now_iso(),
        )
        manifest.to_json(self.output_dir / "preset_manifest.json")

        logger.info(
            "Preset '%s': %d/%d filled, %d NO_TRADE",
            self.gate_config.name, filled_slots, total_slots, no_trade_slots,
        )
        return manifest

    # ------------------------------------------------------------------
    # Winner selection for one slot
    # ------------------------------------------------------------------
    def _select_winner(
        self,
        candidates_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        regime: str,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Select best passing candidate for one (symbol, timeframe, regime).

        Candidates are sorted by deterministic tiebreak order, then walked
        until one passes all enabled gates.

        Returns: (winner_dict or None, list of rejection records)
        """
        score_col = f"{regime}_quality_score"
        train_trades_col = f"{regime}_train_trades"

        # Filter to candidates with data for this regime
        mask = candidates_df[train_trades_col] > 0
        regime_cands = candidates_df[mask].copy()

        if len(regime_cands) == 0:
            return None, []

        # Sort by deterministic tiebreak
        regime_cands = regime_cands.sort_values(
            by=[score_col, "strategy_name", "param_hash"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

        rejections: List[Dict[str, Any]] = []

        for rank, (_, row) in enumerate(regime_cands.iterrows(), start=1):
            row_dict = row.to_dict()
            passed, reason, failed_gates = check_gates(
                row_dict, regime, self.gate_config,
            )

            if passed:
                winner = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "regime": regime,
                    "strategy_name": row_dict["strategy_name"],
                    "param_hash": row_dict["param_hash"],
                    "params": json.loads(row_dict.get("params_json", "{}")),
                    "quality_score": row_dict.get(score_col, 0.0),
                    "rank": rank,
                    "val_trades": row_dict.get(f"{regime}_val_trades", 0),
                    "val_pf": row_dict.get(f"{regime}_val_pf", 0.0),
                    "val_return": row_dict.get(f"{regime}_val_return", 0.0),
                    "val_dd": row_dict.get(f"{regime}_val_dd", 0.0),
                    "val_sharpe": row_dict.get(f"{regime}_val_sharpe", 0.0),
                    "val_win_rate": row_dict.get(f"{regime}_val_win_rate", 0.0),
                    "train_trades": row_dict.get(f"{regime}_train_trades", 0),
                    "train_pf": row_dict.get(f"{regime}_train_pf", 0.0),
                    "train_return": row_dict.get(f"{regime}_train_return", 0.0),
                    "is_tuned": bool(row_dict.get("is_tuned", False)),
                }
                return winner, rejections

            # Record rejection
            rejections.append({
                "rank": rank,
                "strategy_name": row_dict["strategy_name"],
                "param_hash": row_dict["param_hash"],
                "quality_score": row_dict.get(score_col, 0.0),
                "reason": reason,
                "failed_gates": failed_gates,
            })

            # Only examine top candidates to avoid massive logs
            if rank >= 20:
                break

        return None, rejections

    # ------------------------------------------------------------------
    # Shared NO_TRADE logging
    # ------------------------------------------------------------------
    def _append_no_trade_entries(
        self,
        rejection_log: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
        active_gates: List[str],
        reason: str,
    ) -> Tuple[int, int]:
        """Append one NO_TRADE rejection-log entry per regime."""
        for regime in REGIMES:
            rejection_log.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": regime,
                "candidates_considered": 0,
                "winner": None,
                "rejections": [],
                "gates_applied": active_gates,
                "binding_gates": [],
                "outcome": "NO_TRADE",
                "reason": reason,
            })
        return len(REGIMES), len(REGIMES)

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------
    def _write_winners(self, winners: List[Dict[str, Any]]) -> None:
        """Write winners.json and winner_table.csv."""
        # JSON
        output = {
            "preset": self.gate_config.name,
            "total_winners": len(winners),
            "winners": winners,
        }
        with open(self.output_dir / "winners.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        # CSV (flat table for easy inspection)
        if winners:
            flat_rows = []
            for w in winners:
                flat = {k: v for k, v in w.items() if k != "params"}
                flat_rows.append(flat)
            df = pd.DataFrame(flat_rows)
            df.to_csv(self.output_dir / "winner_table.csv", index=False)
        else:
            pd.DataFrame().to_csv(self.output_dir / "winner_table.csv", index=False)

    def _write_rejection_log(self, entries: List[Dict[str, Any]]) -> None:
        """Write rejection_log.jsonl (one JSON object per line per slot)."""
        with open(self.output_dir / "rejection_log.jsonl", "w", encoding="utf-8") as f:
            for entry in entries:
                # Simplify winner dict for the log (remove params to save space)
                log_entry = dict(entry)
                if log_entry.get("winner") and "params" in log_entry["winner"]:
                    w = dict(log_entry["winner"])
                    del w["params"]
                    log_entry["winner"] = w
                f.write(json.dumps(log_entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Convenience: materialise all presets
# ---------------------------------------------------------------------------

def materialise_all_presets(
    evidence_dir: Path,
    presets_dir: Path,
    preset_names: Optional[List[str]] = None,
) -> Dict[str, PresetManifest]:
    """Run materialisation for all (or selected) presets.

    Args:
        evidence_dir: Path to evidence_build/ directory
        presets_dir: Parent directory for preset output folders
        preset_names: Subset of PRESET_ORDER to run (default: all)

    Returns: {preset_name: PresetManifest}
    """
    names = preset_names or PRESET_ORDER
    results: Dict[str, PresetManifest] = {}

    for name in names:
        if name not in ABLATION_PRESETS:
            logger.warning("Unknown preset '%s' — skipping", name)
            continue

        gc = ABLATION_PRESETS[name]
        out_dir = presets_dir / name
        materializer = PresetMaterializer(evidence_dir, out_dir, gc)
        results[name] = materializer.materialise()

    return results
