"""
Section 6 Analysis — Comparative Metrics, Statistical Tests, Paper Export
=========================================================================
Stage D: reads all 7 preset outputs and produces the comparative metrics,
statistical tests, and paper-ready tables for the ISM 2026 conference paper.

Stage E: generates ready-to-inject JSON for Tables 7–9.

All operations are read-only over the preset materialisation outputs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from section6_manifest import (
    PRESET_ORDER,
    REGIMES,
    PresetManifest,
    now_iso,
)

logger = logging.getLogger("section6.analysis")


# ---------------------------------------------------------------------------
# Helper: load preset data
# ---------------------------------------------------------------------------

def _load_preset_winners(preset_dir: Path) -> pd.DataFrame:
    """Load winner_table.csv for one preset."""
    csv_path = preset_dir / "winner_table.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return pd.DataFrame()


def _load_preset_manifest(preset_dir: Path) -> Optional[PresetManifest]:
    """Load preset_manifest.json for one preset."""
    mp = preset_dir / "preset_manifest.json"
    if mp.exists():
        return PresetManifest.from_json(mp)
    return None


def _load_rejection_log(preset_dir: Path) -> List[dict]:
    """Load rejection_log.jsonl for one preset."""
    path = preset_dir / "rejection_log.jsonl"
    entries = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def _prepare_winners_metrics(winners_df: pd.DataFrame) -> pd.DataFrame:
    """Add manuscript-facing metrics derived from winner_table.csv."""
    if len(winners_df) == 0:
        return winners_df.copy()

    df = winners_df.copy()
    numeric_cols = [
        "val_pf", "train_pf", "val_return", "train_return", "val_dd",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    val_dd_safe = df["val_dd"].clip(lower=0.5)
    df["robustness"] = (
        df["val_pf"] / (df["train_pf"] + 0.1)
    ) * (1.0 / (1.0 + 0.05 * (df["train_return"] - df["val_return"]).abs()))
    df["far_flag"] = df["robustness"] < 0.5
    df["val_return_dd"] = df["val_return"] / val_dd_safe
    return df


def _active_gate_summary(gate_config: Dict[str, Any]) -> str:
    """Human-readable active-gate list from PresetManifest.gate_config."""
    mapping = [
        ("g1_min_trades", "G1"),
        ("g2_max_drawdown", "G2"),
        ("g3_train_profitability", "G3"),
        ("g4_val_return", "G4"),
        ("g5_return_dd_ratio", "G5"),
        ("g6_val_profit_factor", "G6"),
        ("g7_robustness_gap", "G7"),
    ]
    active = [gid for key, gid in mapping if gate_config.get(key)]
    return ", ".join(active) if active else "none"


# ---------------------------------------------------------------------------
# Analysis class
# ---------------------------------------------------------------------------

class Section6Analysis:
    """Comparative analysis across all preset materialisations."""

    def __init__(
        self,
        presets_dir: Path,
        output_dir: Path,
        preset_order: Optional[List[str]] = None,
    ):
        self.presets_dir = Path(presets_dir)
        self.output_dir = Path(output_dir)
        self.preset_order = preset_order or PRESET_ORDER

    def run_analysis(self) -> Dict[str, Any]:
        """Full analysis pipeline (Stages D + E).

        Returns: complete analysis dict (also saved to JSON).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load all preset data
        presets_data: Dict[str, Dict[str, Any]] = {}
        for name in self.preset_order:
            pdir = self.presets_dir / name
            if not pdir.exists():
                logger.warning("Preset directory not found: %s", pdir)
                continue
            presets_data[name] = {
                "winners": _load_preset_winners(pdir),
                "manifest": _load_preset_manifest(pdir),
                "rejections": _load_rejection_log(pdir),
            }

        if not presets_data:
            logger.error("No preset data found in %s", self.presets_dir)
            return {}

        # Compute analyses
        fill_rates = self._compute_fill_rates(presets_data)
        convergence = self._compute_convergence_matrix(presets_data)
        binding = self._compute_binding_gate_analysis(presets_data)
        quality = self._compute_quality_comparison(presets_data)
        manuscript_results = self._compute_manuscript_results(presets_data)
        far_sensitivity = self._compute_far_sensitivity(presets_data)
        gate_overlap = self._compute_gate_failure_overlap(
            presets_data, preset_name="all_gates", gate_a="G2", gate_b="G5",
        )
        symbol_aggregates = self._compute_symbol_aggregates(presets_data)
        stats = self._run_statistical_tests(presets_data, symbol_aggregates)
        tables = self._export_paper_tables(
            fill_rates, convergence, binding, quality, manuscript_results,
            far_sensitivity, gate_overlap, stats,
        )

        # Save outputs
        analysis = {
            "generated_at": now_iso(),
            "presets_analysed": list(presets_data.keys()),
            "fill_rates": fill_rates,
            "convergence_matrix": convergence,
            "binding_gate_analysis": binding,
            "quality_comparison": quality,
            "manuscript_results": manuscript_results,
            "far_sensitivity": far_sensitivity,
            "gate_failure_overlap": gate_overlap,
            "symbol_aggregates": symbol_aggregates,
            "statistical_tests": stats,
            "tables_for_paper": tables,
        }

        with open(self.output_dir / "comparative_metrics.json", "w",
                   encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, default=_json_default)

        with open(self.output_dir / "statistical_tests.json", "w",
                   encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=_json_default)

        with open(self.output_dir / "tables_for_paper.json", "w",
                   encoding="utf-8") as f:
            json.dump(tables, f, indent=2, default=_json_default)

        # CSV exports
        if isinstance(fill_rates, list):
            pd.DataFrame(fill_rates).to_csv(
                self.output_dir / "fill_rates.csv", index=False)
        if isinstance(convergence, list):
            conv_df = pd.DataFrame(convergence)
            conv_df.to_csv(self.output_dir / "convergence_matrix.csv", index=False)
        if isinstance(quality, list):
            pd.DataFrame(quality).to_csv(
                self.output_dir / "quality_by_regime.csv", index=False)
        if isinstance(manuscript_results, list):
            pd.DataFrame(manuscript_results).to_csv(
                self.output_dir / "manuscript_table_7.csv", index=False)
        if isinstance(far_sensitivity, dict) and isinstance(
            far_sensitivity.get("rows"), list
        ):
            pd.DataFrame(far_sensitivity["rows"]).to_csv(
                self.output_dir / "far_sensitivity.csv", index=False)
        if isinstance(symbol_aggregates, list):
            pd.DataFrame(symbol_aggregates).to_csv(
                self.output_dir / "symbol_aggregates.csv", index=False)

        # Export audit
        audit = {
            "generated_at": now_iso(),
            "source_presets": {
                name: str(self.presets_dir / name)
                for name in presets_data
            },
            "output_files": [
                "comparative_metrics.json",
                "statistical_tests.json",
                "tables_for_paper.json",
                "fill_rates.csv",
                "convergence_matrix.csv",
                "quality_by_regime.csv",
                "manuscript_table_7.csv",
                "far_sensitivity.csv",
                "symbol_aggregates.csv",
            ],
        }
        with open(self.output_dir / "export_audit.json", "w",
                   encoding="utf-8") as f:
            json.dump(audit, f, indent=2, default=_json_default)

        logger.info("Analysis complete → %s", self.output_dir)
        return analysis

    # ------------------------------------------------------------------
    # Fill rates (Table 7)
    # ------------------------------------------------------------------
    def _compute_fill_rates(
        self, presets_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute fill rates per preset, broken down by regime."""
        rows = []
        for name in self.preset_order:
            if name not in presets_data:
                continue
            manifest = presets_data[name].get("manifest")
            rejections = presets_data[name].get("rejections", [])

            total = manifest.total_slots if manifest else 0
            filled = manifest.filled_slots if manifest else 0
            no_trade = manifest.no_trade_slots if manifest else 0

            # Per-regime breakdown from rejection log
            regime_filled = {r: 0 for r in REGIMES}
            regime_total = {r: 0 for r in REGIMES}
            for entry in rejections:
                r = entry.get("regime", "")
                if r in regime_total:
                    regime_total[r] += 1
                    if entry.get("outcome") == "FILLED":
                        regime_filled[r] += 1

            row = {
                "preset": name,
                "total_slots": total,
                "filled_slots": filled,
                "no_trade_slots": no_trade,
                "fill_rate_pct": round(100 * filled / max(total, 1), 1),
            }
            for r in REGIMES:
                rt = regime_total.get(r, 0)
                rf = regime_filled.get(r, 0)
                row[f"{r}_filled"] = rf
                row[f"{r}_total"] = rt
                row[f"{r}_fill_pct"] = round(100 * rf / max(rt, 1), 1)

            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Convergence matrix (pairwise Jaccard similarity)
    # ------------------------------------------------------------------
    def _compute_convergence_matrix(
        self, presets_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Pairwise Jaccard similarity of winner sets across presets.

        Two presets share a slot outcome if the same (symbol, tf, regime)
        selects the same strategy_name + param_hash.
        """
        # Build winner fingerprints per preset
        fingerprints: Dict[str, set] = {}
        for name in self.preset_order:
            if name not in presets_data:
                continue
            winners_df = presets_data[name].get("winners", pd.DataFrame())
            fps = set()
            if len(winners_df) > 0:
                for _, w in winners_df.iterrows():
                    key = (
                        str(w.get("symbol", "")),
                        str(w.get("timeframe", "")),
                        str(w.get("regime", "")),
                        str(w.get("strategy_name", "")),
                        str(w.get("param_hash", "")),
                    )
                    fps.add(key)
            fingerprints[name] = fps

        # Pairwise Jaccard
        rows = []
        names = [n for n in self.preset_order if n in fingerprints]
        for a in names:
            row = {"preset": a}
            for b in names:
                sa, sb = fingerprints[a], fingerprints[b]
                union = len(sa | sb)
                inter = len(sa & sb)
                jaccard = round(inter / max(union, 1), 4)
                row[b] = jaccard
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Binding gate analysis (Table 8)
    # ------------------------------------------------------------------
    def _compute_binding_gate_analysis(
        self, presets_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """For each preset, determine which gates were binding."""
        rows = []
        for name in self.preset_order:
            if name not in presets_data:
                continue
            rejections = presets_data[name].get("rejections", [])

            gate_binding_count = {}  # gate_id → count of slots where it rejected ≥1
            slots_with_rejection = 0
            total_filled = 0
            total_no_trade = 0
            winner_rank_sum = 0
            winner_count = 0

            for entry in rejections:
                outcome = entry.get("outcome", "NO_TRADE")
                if outcome == "FILLED":
                    total_filled += 1
                    winner = entry.get("winner", {})
                    rank = winner.get("rank", 1) if winner else 1
                    winner_rank_sum += rank
                    winner_count += 1
                    if rank > 1:
                        slots_with_rejection += 1
                else:
                    total_no_trade += 1

                for gid in entry.get("binding_gates", []):
                    gate_binding_count[gid] = gate_binding_count.get(gid, 0) + 1

            avg_winner_rank = round(
                winner_rank_sum / max(winner_count, 1), 2
            )

            row = {
                "preset": name,
                "filled": total_filled,
                "no_trade": total_no_trade,
                "slots_with_top1_rejected": slots_with_rejection,
                "avg_winner_rank": avg_winner_rank,
            }
            for gid in ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]:
                row[f"{gid}_binding_slots"] = gate_binding_count.get(gid, 0)

            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Quality comparison (Table 9)
    # ------------------------------------------------------------------
    def _compute_quality_comparison(
        self, presets_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Average quality_score of winners per preset, by regime."""
        rows = []
        for name in self.preset_order:
            if name not in presets_data:
                continue
            winners_df = presets_data[name].get("winners", pd.DataFrame())

            row: Dict[str, Any] = {"preset": name}

            if len(winners_df) == 0:
                row["overall_mean_quality"] = 0.0
                row["overall_median_quality"] = 0.0
                for r in REGIMES:
                    row[f"{r}_mean_quality"] = 0.0
                    row[f"{r}_count"] = 0
                rows.append(row)
                continue

            all_scores = winners_df["quality_score"].dropna()
            row["overall_mean_quality"] = round(float(all_scores.mean()), 4)
            row["overall_median_quality"] = round(float(all_scores.median()), 4)
            row["winner_count"] = len(winners_df)

            for r in REGIMES:
                regime_winners = winners_df[winners_df["regime"] == r]
                if len(regime_winners) > 0:
                    scores = regime_winners["quality_score"].dropna()
                    row[f"{r}_mean_quality"] = round(float(scores.mean()), 4)
                    row[f"{r}_count"] = len(regime_winners)
                else:
                    row[f"{r}_mean_quality"] = 0.0
                    row[f"{r}_count"] = 0

            # Also compute mean validation metrics
            for metric in ["val_pf", "val_return", "val_dd"]:
                if metric in winners_df.columns:
                    vals = winners_df[metric].dropna()
                    row[f"mean_{metric}"] = round(float(vals.mean()), 4)

            rows.append(row)
        return rows

    def _compute_manuscript_results(
        self, presets_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Paper-facing per-preset metrics for Section 6 Table 7."""
        rows = []
        for name in self.preset_order:
            if name not in presets_data:
                continue

            manifest = presets_data[name].get("manifest")
            winners_df = _prepare_winners_metrics(
                presets_data[name].get("winners", pd.DataFrame())
            )

            filled = manifest.filled_slots if manifest else len(winners_df)
            total = manifest.total_slots if manifest else 0
            gate_config = manifest.gate_config if manifest else {}
            active_gates = _active_gate_summary(gate_config)

            if len(winners_df) == 0:
                row = {
                    "preset": name,
                    "active_gates": active_gates,
                    "gate_count": len([g for g in active_gates.split(", ") if g and g != "none"]),
                    "accepted_slots": int(filled),
                    "acceptance_pct": round(100 * filled / max(total, 1), 1),
                    "median_rho": 0.0,
                    "far_pct": 0.0,
                    "median_val_return": 0.0,
                    "median_val_dd": 0.0,
                    "median_pf": 0.0,
                    "median_return_dd": 0.0,
                }
                rows.append(row)
                continue

            row = {
                "preset": name,
                "active_gates": active_gates,
                "gate_count": len([g for g in active_gates.split(", ") if g and g != "none"]),
                "accepted_slots": int(filled),
                "acceptance_pct": round(100 * filled / max(total, 1), 1),
                "median_rho": round(float(winners_df["robustness"].median()), 3),
                "far_pct": round(float(winners_df["far_flag"].mean() * 100.0), 2),
                "median_val_return": round(float(winners_df["val_return"].median()), 2),
                "median_val_dd": round(float(winners_df["val_dd"].median()), 2),
                "median_pf": round(float(winners_df["val_pf"].median()), 3),
                "median_return_dd": round(float(winners_df["val_return_dd"].median()), 3),
            }
            rows.append(row)
        return rows

    def _compute_far_sensitivity(
        self,
        presets_data: Dict[str, Dict[str, Any]],
        thresholds: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Recalculate FAR under alternative robustness-failure thresholds."""
        thresholds = thresholds or [0.4, 0.5, 0.6]
        rows: List[Dict[str, Any]] = []

        for name in self.preset_order:
            if name not in presets_data:
                continue

            winners_df = _prepare_winners_metrics(
                presets_data[name].get("winners", pd.DataFrame())
            )
            row: Dict[str, Any] = {"preset": name}

            for threshold in thresholds:
                key = f"far_pct_rho_lt_{str(threshold).replace('.', '_')}"
                if len(winners_df) == 0:
                    row[key] = 0.0
                else:
                    far_pct = float((winners_df["robustness"] < threshold).mean() * 100.0)
                    row[key] = round(far_pct, 2)

            rows.append(row)

        rankings: Dict[str, List[str]] = {}
        for threshold in thresholds:
            key = f"far_pct_rho_lt_{str(threshold).replace('.', '_')}"
            ranked_rows = sorted(rows, key=lambda item: (item.get(key, 0.0), item["preset"]))
            rankings[f"rho_lt_{str(threshold).replace('.', '_')}"] = [
                row["preset"] for row in ranked_rows
            ]

        ranking_values = list(rankings.values())
        ranking_stable = (
            len(ranking_values) > 0
            and all(order == ranking_values[0] for order in ranking_values[1:])
        )

        return {
            "thresholds": thresholds,
            "rows": rows,
            "rankings": rankings,
            "ranking_stable": ranking_stable,
        }

    def _compute_gate_failure_overlap(
        self,
        presets_data: Dict[str, Dict[str, Any]],
        preset_name: str,
        gate_a: str,
        gate_b: str,
    ) -> Dict[str, Any]:
        """Quantify overlap between two gates in the rejection ledger."""
        preset = presets_data.get(preset_name, {})
        rejections = preset.get("rejections", [])

        cand_both = cand_a_only = cand_b_only = cand_neither = 0
        slot_both = slot_a_only = slot_b_only = slot_neither = 0

        for entry in rejections:
            binding = set(entry.get("binding_gates", []))
            slot_has_a = gate_a in binding
            slot_has_b = gate_b in binding
            if slot_has_a and slot_has_b:
                slot_both += 1
            elif slot_has_a:
                slot_a_only += 1
            elif slot_has_b:
                slot_b_only += 1
            else:
                slot_neither += 1

            for rejection in entry.get("rejections", []):
                failed = set(rejection.get("failed_gates", []))
                cand_has_a = gate_a in failed
                cand_has_b = gate_b in failed
                if cand_has_a and cand_has_b:
                    cand_both += 1
                elif cand_has_a:
                    cand_a_only += 1
                elif cand_has_b:
                    cand_b_only += 1
                else:
                    cand_neither += 1

        def _jaccard(both: int, only_a: int, only_b: int) -> float:
            union = both + only_a + only_b
            return round(both / union, 4) if union else 0.0

        def _phi(both: int, only_a: int, only_b: int, neither: int) -> float:
            denom = np.sqrt(
                (both + only_a) * (both + only_b)
                * (only_a + neither) * (only_b + neither)
            )
            if denom <= 0:
                return 0.0
            num = (both * neither) - (only_a * only_b)
            return round(float(num / denom), 4)

        cand_total_a = cand_both + cand_a_only
        cand_total_b = cand_both + cand_b_only
        slot_total_a = slot_both + slot_a_only
        slot_total_b = slot_both + slot_b_only

        return {
            "preset": preset_name,
            "gate_a": gate_a,
            "gate_b": gate_b,
            "candidate_level": {
                "both_failed": cand_both,
                "gate_a_only": cand_a_only,
                "gate_b_only": cand_b_only,
                "neither": cand_neither,
                "jaccard": _jaccard(cand_both, cand_a_only, cand_b_only),
                "phi": _phi(cand_both, cand_a_only, cand_b_only, cand_neither),
                "p_gate_b_given_gate_a_pct": round(
                    100.0 * cand_both / max(cand_total_a, 1), 2
                ),
                "p_gate_a_given_gate_b_pct": round(
                    100.0 * cand_both / max(cand_total_b, 1), 2
                ),
            },
            "slot_level": {
                "both_binding": slot_both,
                "gate_a_only": slot_a_only,
                "gate_b_only": slot_b_only,
                "neither": slot_neither,
                "jaccard": _jaccard(slot_both, slot_a_only, slot_b_only),
                "phi": _phi(slot_both, slot_a_only, slot_b_only, slot_neither),
                "p_gate_b_given_gate_a_pct": round(
                    100.0 * slot_both / max(slot_total_a, 1), 2
                ),
                "p_gate_a_given_gate_b_pct": round(
                    100.0 * slot_both / max(slot_total_b, 1), 2
                ),
            },
        }

    def _compute_symbol_aggregates(
        self, presets_data: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Per-symbol aggregates for symbol-blocked non-parametric tests."""
        rows: List[Dict[str, Any]] = []
        for name in self.preset_order:
            if name not in presets_data:
                continue

            winners_df = _prepare_winners_metrics(
                presets_data[name].get("winners", pd.DataFrame())
            )
            rejections = pd.DataFrame(presets_data[name].get("rejections", []))
            if len(rejections) == 0:
                continue

            for symbol, sym_rej in rejections.groupby("symbol"):
                total_slots = int(len(sym_rej))
                filled_slots = int((sym_rej["outcome"] == "FILLED").sum())
                sym_winners = winners_df[winners_df["symbol"] == symbol]

                if len(sym_winners) > 0:
                    median_rho = float(sym_winners["robustness"].median())
                    far_pct = float(sym_winners["far_flag"].mean() * 100.0)
                    median_val_return = float(sym_winners["val_return"].median())
                    median_val_dd = float(sym_winners["val_dd"].median())
                    median_pf = float(sym_winners["val_pf"].median())
                    median_return_dd = float(sym_winners["val_return_dd"].median())
                else:
                    median_rho = 0.0
                    far_pct = 0.0
                    median_val_return = 0.0
                    median_val_dd = 0.0
                    median_pf = 0.0
                    median_return_dd = 0.0

                rows.append({
                    "preset": name,
                    "symbol": symbol,
                    "total_slots": total_slots,
                    "filled_slots": filled_slots,
                    "acceptance_pct": round(100.0 * filled_slots / max(total_slots, 1), 4),
                    "median_rho": round(median_rho, 6),
                    "far_pct": round(far_pct, 6),
                    "median_val_return": round(median_val_return, 6),
                    "median_val_dd": round(median_val_dd, 6),
                    "median_pf": round(median_pf, 6),
                    "median_return_dd": round(median_return_dd, 6),
                })
        return rows

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------
    def _run_statistical_tests(
        self,
        presets_data: Dict[str, Dict[str, Any]],
        symbol_aggregates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Paired statistical tests across presets.

        Tests:
          - McNemar's test on fill/no-fill per slot
          - Wilcoxon signed-rank on quality scores (paired by slot)
          - Cohen's d effect size
          - Symbol-blocked Friedman and Wilcoxon tests on aggregate metrics
        """
        results: Dict[str, Any] = {}

        # Build slot-level fill/score maps per preset
        slot_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for name in self.preset_order:
            if name not in presets_data:
                continue
            winners_df = presets_data[name].get("winners", pd.DataFrame())
            rejections = presets_data[name].get("rejections", [])

            fill_map: Dict[str, bool] = {}
            score_map: Dict[str, float] = {}

            for entry in rejections:
                slot_key = f"{entry['symbol']}/{entry['timeframe']}/{entry['regime']}"
                fill_map[slot_key] = entry.get("outcome") == "FILLED"

            if len(winners_df) > 0:
                for _, w in winners_df.iterrows():
                    slot_key = f"{w['symbol']}/{w['timeframe']}/{w['regime']}"
                    score_map[slot_key] = float(w.get("quality_score", 0))

            slot_maps[name] = {"fill": fill_map, "score": score_map}

        # Pairwise tests: all_gates vs each other preset
        baseline = "all_gates"
        if baseline not in slot_maps:
            return {"error": "all_gates preset not found"}

        pairwise: List[Dict[str, Any]] = []
        for name in self.preset_order:
            if name == baseline or name not in slot_maps:
                continue

            pair_result = self._paired_test(
                baseline, name,
                slot_maps[baseline], slot_maps[name],
            )
            pairwise.append(pair_result)

        results["baseline"] = baseline
        results["pairwise_vs_baseline"] = pairwise
        results["symbol_blocked"] = self._run_symbol_blocked_tests(
            symbol_aggregates, baseline
        )
        return results

    def _run_symbol_blocked_tests(
        self,
        symbol_aggregates: List[Dict[str, Any]],
        baseline: str,
    ) -> Dict[str, Any]:
        """Per-symbol omnibus and pairwise tests for manuscript-facing metrics."""
        if not symbol_aggregates:
            return {"error": "no symbol aggregates available"}

        df = pd.DataFrame(symbol_aggregates)
        metrics = [
            ("acceptance_pct", "Acceptance %"),
            ("median_rho", "Median rho"),
            ("far_pct", "FAR %"),
        ]
        results: Dict[str, Any] = {"symbol_count": int(df["symbol"].nunique())}

        for metric_key, metric_label in metrics:
            metric_block: Dict[str, Any] = {"label": metric_label}
            matrix = (
                df.pivot(index="symbol", columns="preset", values=metric_key)
                .reindex(columns=[p for p in self.preset_order if p in df["preset"].unique()])
                .fillna(0.0)
            )
            available_presets = list(matrix.columns)

            if len(available_presets) >= 3:
                try:
                    from scipy.stats import friedmanchisquare
                    stat, p_value = friedmanchisquare(
                        *[matrix[p].values for p in available_presets]
                    )
                    metric_block["friedman"] = {
                        "chi2": round(float(stat), 4),
                        "p_value": float(p_value),
                        "k": len(available_presets),
                        "n": int(len(matrix)),
                    }
                except Exception as exc:
                    metric_block["friedman"] = {"note": f"failed: {exc}"}
            else:
                metric_block["friedman"] = {"note": "insufficient presets"}

            if baseline in matrix.columns:
                pairwise = []
                for name in available_presets:
                    if name == baseline:
                        continue
                    diffs = matrix[baseline] - matrix[name]
                    med_delta = float(np.median(diffs))
                    try:
                        from scipy.stats import wilcoxon
                        stat, p_value = wilcoxon(diffs)
                        raw_p = float(p_value)
                        bonf_p = min(raw_p * 6.0, 1.0)
                        pairwise.append({
                            "preset_a": baseline,
                            "preset_b": name,
                            "metric": metric_key,
                            "wilcoxon_statistic": float(stat),
                            "p_value": raw_p,
                            "bonferroni_p": bonf_p,
                            "median_delta": round(med_delta, 4),
                        })
                    except Exception as exc:
                        pairwise.append({
                            "preset_a": baseline,
                            "preset_b": name,
                            "metric": metric_key,
                            "note": f"wilcoxon failed: {exc}",
                            "median_delta": round(med_delta, 4),
                        })
                metric_block["pairwise_vs_baseline"] = pairwise

            results[metric_key] = metric_block

        return results

    def _paired_test(
        self,
        name_a: str,
        name_b: str,
        maps_a: Dict[str, Dict[str, Any]],
        maps_b: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run paired tests between two presets."""
        fill_a = maps_a["fill"]
        fill_b = maps_b["fill"]
        score_a = maps_a["score"]
        score_b = maps_b["score"]

        # Common slots
        all_slots = set(fill_a.keys()) | set(fill_b.keys())

        # McNemar's contingency table
        # a=filled, b=filled -> n11
        # a=filled, b=not   -> n10
        # a=not,   b=filled -> n01
        # a=not,   b=not    -> n00
        n11 = n10 = n01 = n00 = 0
        for s in all_slots:
            fa = fill_a.get(s, False)
            fb = fill_b.get(s, False)
            if fa and fb:     n11 += 1
            elif fa and not fb: n10 += 1
            elif not fa and fb: n01 += 1
            else:              n00 += 1

        # McNemar's test (without scipy dependency: chi-squared approximation)
        discordant = n10 + n01
        if discordant > 0:
            mcnemar_chi2 = (abs(n10 - n01) - 1) ** 2 / discordant
        else:
            mcnemar_chi2 = 0.0

        # Paired quality scores (for slots filled in BOTH presets)
        common_filled = set(score_a.keys()) & set(score_b.keys())
        paired_diff = []
        for s in sorted(common_filled):
            paired_diff.append(score_a[s] - score_b[s])

        # Cohen's d
        if len(paired_diff) > 1:
            diffs = np.array(paired_diff)
            mean_d = float(np.mean(diffs))
            std_d = float(np.std(diffs, ddof=1))
            cohens_d = mean_d / std_d if std_d > 0 else 0.0

            # Wilcoxon signed-rank (simplified: just report stats)
            try:
                from scipy.stats import wilcoxon
                stat, p_value = wilcoxon(diffs)
                wilcoxon_result = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                }
            except ImportError:
                # No scipy — compute basic descriptive stats only
                wilcoxon_result = {
                    "note": "scipy not available; descriptive stats only",
                    "mean_diff": round(mean_d, 6),
                    "std_diff": round(std_d, 6),
                }
            except Exception:
                wilcoxon_result = {"note": "test failed"}
        else:
            cohens_d = 0.0
            wilcoxon_result = {"note": "insufficient paired observations"}

        return {
            "preset_a": name_a,
            "preset_b": name_b,
            "mcnemar": {
                "n11_both_filled": n11,
                "n10_a_only": n10,
                "n01_b_only": n01,
                "n00_neither": n00,
                "chi2_approx": round(mcnemar_chi2, 4),
                "discordant_pairs": discordant,
            },
            "quality_paired": {
                "common_filled_slots": len(common_filled),
                "cohens_d": round(cohens_d, 4),
                "wilcoxon": wilcoxon_result,
            },
        }

    # ------------------------------------------------------------------
    # Paper tables (Stage E)
    # ------------------------------------------------------------------
    def _export_paper_tables(
        self,
        fill_rates: List[Dict],
        convergence: List[Dict],
        binding: List[Dict],
        quality: List[Dict],
        manuscript_results: List[Dict],
        far_sensitivity: Dict[str, Any],
        gate_overlap: Dict[str, Any],
        stats: Dict,
    ) -> Dict[str, Any]:
        """Generate exact cell values for manuscript-facing tables."""
        return {
            "table_7_fill_rates": fill_rates,
            "table_7_manuscript_metrics": manuscript_results,
            "table_8_binding_gates": binding,
            "table_9_quality_comparison": quality,
            "table_10_far_sensitivity": far_sensitivity,
            "gate_overlap_g2_g5": gate_overlap,
            "convergence_matrix": convergence,
            "statistical_tests_summary": stats,
        }


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Handle numpy/pandas types in JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)
