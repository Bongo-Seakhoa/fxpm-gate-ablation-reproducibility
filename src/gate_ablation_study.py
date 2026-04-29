"""
Replicated gate ablation study for the standalone paper.

The study is deliberately framed as a calibrated simulation, not a replay of a
frozen artifact dataset. Inference is based on paired replication-level
summaries across repeated runs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Tuple

BASE_SEED = 42
THRESHOLD_PROFILE_NAME = "historical"


@dataclass
class GateConfig:
    name: str = "all_gates"
    description: str = ""
    g1_min_trades: bool = True
    g2_max_drawdown: bool = True
    g3_train_profitability: bool = True
    g4_val_return: bool = True
    g5_return_dd_ratio: bool = True
    g6_val_profit_factor: bool = True
    g7_robustness_gap: bool = True

    def active_gate_count(self) -> int:
        return sum(
            [
                self.g1_min_trades,
                self.g2_max_drawdown,
                self.g3_train_profitability,
                self.g4_val_return,
                self.g5_return_dd_ratio,
                self.g6_val_profit_factor,
                self.g7_robustness_gap,
            ]
        )

    def active_gate_names(self) -> List[str]:
        names: List[str] = []
        if self.g1_min_trades:
            names.append("G1")
        if self.g2_max_drawdown:
            names.append("G2")
        if self.g3_train_profitability:
            names.append("G3")
        if self.g4_val_return:
            names.append("G4")
        if self.g5_return_dd_ratio:
            names.append("G5")
        if self.g6_val_profit_factor:
            names.append("G6")
        if self.g7_robustness_gap:
            names.append("G7")
        return names


ABLATION_PRESETS: Dict[str, GateConfig] = {
    "all_gates": GateConfig(
        name="all_gates",
        description="Full seven-gate validation pipeline (baseline)",
    ),
    "no_gates": GateConfig(
        name="no_gates",
        description="All gates disabled (negative control)",
        g1_min_trades=False,
        g2_max_drawdown=False,
        g3_train_profitability=False,
        g4_val_return=False,
        g5_return_dd_ratio=False,
        g6_val_profit_factor=False,
        g7_robustness_gap=False,
    ),
    "drawdown_only": GateConfig(
        name="drawdown_only",
        description="Only G2 active",
        g1_min_trades=False,
        g3_train_profitability=False,
        g4_val_return=False,
        g5_return_dd_ratio=False,
        g6_val_profit_factor=False,
        g7_robustness_gap=False,
    ),
    "profitability_only": GateConfig(
        name="profitability_only",
        description="Only G3 + G6 active",
        g1_min_trades=False,
        g2_max_drawdown=False,
        g4_val_return=False,
        g5_return_dd_ratio=False,
        g7_robustness_gap=False,
    ),
    "minimal": GateConfig(
        name="minimal",
        description="G1 + G2 only",
        g3_train_profitability=False,
        g4_val_return=False,
        g5_return_dd_ratio=False,
        g6_val_profit_factor=False,
        g7_robustness_gap=False,
    ),
    "quality_focused": GateConfig(
        name="quality_focused",
        description="G2 + G5 + G6",
        g1_min_trades=False,
        g3_train_profitability=False,
        g4_val_return=False,
        g7_robustness_gap=False,
    ),
    "robustness_focused": GateConfig(
        name="robustness_focused",
        description="G1 + G5 + G7",
        g2_max_drawdown=False,
        g3_train_profitability=False,
        g4_val_return=False,
        g6_val_profit_factor=False,
    ),
}

PRODUCTION_THRESHOLDS = {
    "g1_min_trades": 15,
    "g2_max_drawdown": 18.0,
    "g3_train_min_pf": 1.0,
    "g3_train_min_return": 0.0,
    "g3_exceptional_val_pf": 1.15,
    "g3_exceptional_val_return": 8.0,
    "g4_min_val_return": 5.0,
    "g5_min_return_dd_ratio": 1.0,
    "g6_min_val_profit_factor": 1.05,
    "g7_min_robustness": 0.80,
    "g7_sharpe_override": 0.30,
}

HISTORICAL_MATERIALISATION_THRESHOLDS = {
    "g1_min_trades": 5,
    "g2_max_drawdown": 20.0,
    "g3_train_min_pf": None,
    "g3_train_min_return": 0.0,
    "g3_exceptional_val_pf": None,
    "g3_exceptional_val_return": None,
    "g4_min_val_return": 5.0,
    "g5_min_return_dd_ratio": 1.0,
    "g6_min_val_profit_factor": 1.0,
    "g7_min_robustness": 0.75,
    "g7_sharpe_override": None,
}

THRESHOLD_PROFILES = {
    "production": PRODUCTION_THRESHOLDS,
    "historical": HISTORICAL_MATERIALISATION_THRESHOLDS,
}

THRESHOLDS = dict(HISTORICAL_MATERIALISATION_THRESHOLDS)


def set_threshold_profile(profile: str) -> None:
    """Select the gate predicate profile used by the simulation."""
    global THRESHOLD_PROFILE_NAME, THRESHOLDS
    if profile not in THRESHOLD_PROFILES:
        raise ValueError(f"Unknown threshold profile: {profile}")
    THRESHOLD_PROFILE_NAME = profile
    THRESHOLDS = dict(THRESHOLD_PROFILES[profile])

STRATEGIES = [
    "BB_Squeeze",
    "Bollinger_Trend",
    "CCI_Momentum",
    "Chaikin_Vol",
    "DEMA_Cross",
    "Donchian_Break",
    "EMA_Cross",
    "EMA_Ribbon",
    "Engulfing_Pattern",
    "FRAMA_Trend",
    "Fractal_Break",
    "HMA_Momentum",
    "Heikin_Ashi",
    "Hull_MA",
    "Ichimoku_Cloud",
    "KAMA_Trend",
    "Keltner_Break",
    "MACD_Cross",
    "MACD_Histogram",
    "MFI_Divergence",
    "Momentum_Burst",
    "OBV_Trend",
    "Parabolic_SAR",
    "RSI_Divergence",
    "RSI_Mean_Rev",
    "Range_Break",
    "SMA_Cross",
    "SMMA_Trend",
    "Stochastic_Cross",
    "Stochastic_RSI",
    "SuperTrend",
    "TEMA_Cross",
    "TRIX_Signal",
    "VWAP_Deviation",
    "Vortex_Cross",
    "WMA_Cross",
    "Williams_R",
    "ZigZag_Swing",
    "ADX_Trend",
    "Aroon_Cross",
    "ATR_Trail",
    "CMO_Momentum",
    "DPO_Cycle",
    "Elder_Ray",
    "Force_Index",
    "KST_Signal",
    "Mass_Index",
    "PPO_Signal",
    "RVI_Signal",
    "Squeeze_Mom",
]

REGIMES = ["TREND", "RANGE", "BREAKOUT", "CHOP"]
TIMEFRAMES = ["M5", "M15", "M30", "H1", "H4", "D1"]
PRESET_ORDER = [
    "all_gates",
    "no_gates",
    "minimal",
    "drawdown_only",
    "profitability_only",
    "quality_focused",
    "robustness_focused",
]


@dataclass
class PresetResult:
    preset_name: str
    description: str
    active_gates: List[str]
    active_gate_count: int
    total_candidates: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    acceptance_rate: float = 0.0
    gate_rejections: Dict[str, int] = field(default_factory=dict)
    gate_rejection_rates: Dict[str, float] = field(default_factory=dict)
    first_gate_rejections: Dict[str, int] = field(default_factory=dict)
    accepted_robustness_values: List[float] = field(default_factory=list)
    accepted_robustness_median: float = 0.0
    accepted_robustness_mean: float = 0.0
    accepted_robustness_iqr: Tuple[float, float] = (0.0, 0.0)
    accepted_val_return_median: float = 0.0
    accepted_val_drawdown_median: float = 0.0
    accepted_val_pf_median: float = 0.0
    accepted_val_return_dd_median: float = 0.0
    false_acceptance_count: int = 0
    false_acceptance_rate: float = 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _gauss(mu: float, sigma: float, rng: Random) -> float:
    return rng.gauss(mu, sigma)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    return statistics.median(values)


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = q * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _distribution_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(_median(values), 4),
        "q25": round(_quantile(values, 0.25), 4),
        "q75": round(_quantile(values, 0.75), 4),
        "ci95_low": round(_quantile(values, 0.025), 4),
        "ci95_high": round(_quantile(values, 0.975), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _compact_float(value: float) -> float:
    return float(f"{value:.10g}")


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _chi2_survival(x: float, df: int) -> float:
    if x <= 0:
        return 1.0
    if df <= 0:
        return 0.0
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(
        2.0 / (9.0 * df)
    )
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _wilcoxon_signed_rank(x: List[float], y: List[float]) -> Dict[str, Any]:
    if len(x) != len(y) or len(x) < 10:
        return {
            "W": None,
            "z": None,
            "p_value": None,
            "n": len(x),
            "note": "Insufficient paired replication samples",
        }

    diffs = [a - b for a, b in zip(x, y) if abs(a - b) > 1e-12]
    n = len(diffs)
    if n < 10:
        return {
            "W": None,
            "z": None,
            "p_value": None,
            "n": n,
            "note": "Too few non-zero paired differences",
        }

    abs_diffs = [(abs(diff), idx) for idx, diff in enumerate(diffs)]
    abs_diffs.sort(key=lambda item: item[0])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs(abs_diffs[j + 1][0] - abs_diffs[i][0]) < 1e-12:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[abs_diffs[k][1]] = avg_rank
        i = j + 1

    w_plus = sum(ranks[idx] for idx in range(n) if diffs[idx] > 0)
    w_minus = sum(ranks[idx] for idx in range(n) if diffs[idx] < 0)
    w_stat = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if std_w < 1e-12:
        return {"W": round(w_stat, 4), "z": 0.0, "p_value": 1.0, "n": n}

    z = (w_stat - mean_w) / std_w
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return {"W": round(w_stat, 4), "z": round(z, 4), "p_value": _compact_float(p_value), "n": n}


def _friedman_test(groups: List[List[float]]) -> Dict[str, Any]:
    k = len(groups)
    if k < 3:
        return {"chi2": None, "df": None, "p_value": None, "note": "Need at least 3 groups"}

    n = min(len(group) for group in groups)
    if n < 10:
        return {
            "chi2": None,
            "df": None,
            "p_value": None,
            "note": "Insufficient paired replications",
        }

    aligned = [group[:n] for group in groups]
    rank_sums = [0.0] * k
    for row_idx in range(n):
        row = [(aligned[col_idx][row_idx], col_idx) for col_idx in range(k)]
        row.sort(key=lambda item: item[0])
        start = 0
        while start < k:
            end = start
            while end < k - 1 and abs(row[end + 1][0] - row[start][0]) < 1e-12:
                end += 1
            avg_rank = (start + 1 + end + 1) / 2.0
            for idx in range(start, end + 1):
                rank_sums[row[idx][1]] += avg_rank
            start = end + 1

    grand_mean = n * (k + 1) / 2.0
    ss = sum((rank_sum - grand_mean) ** 2 for rank_sum in rank_sums)
    chi2 = 12.0 * ss / (n * k * (k + 1))
    df = k - 1
    p_value = _chi2_survival(chi2, df)
    return {"chi2": round(chi2, 4), "df": df, "p_value": _compact_float(p_value), "n": n, "k": k}


def _generate_single_candidate(
    strategy: str,
    regime: str,
    timeframe: str,
    symbol: str,
    candidate_id: int,
    rng: Random,
) -> Dict[str, Any]:
    quality = _gauss(0.0, 1.0, rng)
    regime_bonus = {"TREND": 0.3, "BREAKOUT": 0.15, "RANGE": -0.1, "CHOP": -0.4}
    q = quality + regime_bonus.get(regime, 0.0)

    tf_trades = {"M5": 80, "M15": 55, "M30": 40, "H1": 28, "H4": 16, "D1": 8}
    base_trades = tf_trades.get(timeframe, 30)

    val_trades = max(1, int(_gauss(base_trades, base_trades * 0.4, rng)))
    val_return = _gauss(3.0 + q * 6.0, 8.0, rng)
    val_drawdown = _clamp(_gauss(14.0 - q * 3.0, 6.0, rng), 1.0, 50.0)
    val_profit_factor = _clamp(_gauss(1.05 + q * 0.3, 0.35, rng), 0.3, 4.0)
    val_sharpe = _clamp(_gauss(0.15 + q * 0.25, 0.3, rng), -1.0, 3.0)
    val_win_rate = _clamp(_gauss(48.0 + q * 5.0, 10.0, rng), 15.0, 85.0)
    val_return_dd_ratio = val_return / val_drawdown if val_drawdown > 0 else 0.0

    train_return = _gauss(val_return * 1.2 + 2.0, 5.0, rng)
    train_pf = _clamp(_gauss(val_profit_factor * 1.1, 0.25, rng), 0.3, 5.0)
    train_drawdown = _clamp(_gauss(val_drawdown * 0.9, 4.0, rng), 1.0, 45.0)

    gap = abs(train_return - val_return)
    gap_penalty = 1.0 / (1.0 + gap * 0.05)
    raw_robustness = (val_profit_factor / (train_pf + 0.1)) * gap_penalty
    robustness = _clamp(raw_robustness, 0.0, 2.0)

    return {
        "candidate_id": candidate_id,
        "strategy": strategy,
        "regime": regime,
        "timeframe": timeframe,
        "symbol": symbol,
        "val_trades": val_trades,
        "val_return_pct": round(val_return, 2),
        "val_drawdown_pct": round(val_drawdown, 2),
        "val_profit_factor": round(val_profit_factor, 3),
        "val_return_dd_ratio": round(val_return_dd_ratio, 3),
        "val_sharpe": round(val_sharpe, 3),
        "val_win_rate": round(val_win_rate, 1),
        "train_return_pct": round(train_return, 2),
        "train_profit_factor": round(train_pf, 3),
        "train_drawdown_pct": round(train_drawdown, 2),
        "robustness_ratio": round(robustness, 3),
    }


def generate_candidate_pool(n_symbols: int, seed: int) -> List[Dict[str, Any]]:
    rng = Random(seed)
    symbols = [f"SYM_{idx:03d}" for idx in range(1, n_symbols + 1)]
    candidates: List[Dict[str, Any]] = []
    candidate_id = 0

    for strategy in STRATEGIES:
        for regime in REGIMES:
            for timeframe in TIMEFRAMES:
                n_active = rng.randint(3, min(8, n_symbols))
                active_symbols = rng.sample(symbols, n_active)
                for symbol in active_symbols:
                    candidate_id += 1
                    candidates.append(
                        _generate_single_candidate(
                            strategy=strategy,
                            regime=regime,
                            timeframe=timeframe,
                            symbol=symbol,
                            candidate_id=candidate_id,
                            rng=rng,
                        )
                    )

    return candidates


def evaluate_gate(candidate: Dict[str, Any], gate_config: GateConfig) -> Tuple[bool, str, List[str]]:
    failed: List[str] = []
    thresholds = THRESHOLDS

    if THRESHOLD_PROFILE_NAME == "historical":
        if gate_config.g1_min_trades and candidate["val_trades"] < thresholds["g1_min_trades"]:
            failed.append("G1")

        if gate_config.g2_max_drawdown and candidate["val_drawdown_pct"] >= thresholds["g2_max_drawdown"]:
            failed.append("G2")

        if gate_config.g3_train_profitability and candidate["train_return_pct"] < thresholds["g3_train_min_return"]:
            failed.append("G3")

        if gate_config.g4_val_return and candidate["val_return_pct"] < thresholds["g4_min_val_return"]:
            failed.append("G4")

        if gate_config.g5_return_dd_ratio and candidate["val_return_dd_ratio"] < thresholds["g5_min_return_dd_ratio"]:
            failed.append("G5")

        if gate_config.g6_val_profit_factor and candidate["val_profit_factor"] < thresholds["g6_min_val_profit_factor"]:
            failed.append("G6")

        if gate_config.g7_robustness_gap and candidate["robustness_ratio"] < thresholds["g7_min_robustness"]:
            failed.append("G7")

        accepted = len(failed) == 0
        reason = "Accepted" if accepted else f"Rejected by: {', '.join(failed)}"
        return accepted, reason, failed

    if gate_config.g1_min_trades and candidate["val_trades"] < thresholds["g1_min_trades"]:
        failed.append("G1")

    if gate_config.g2_max_drawdown:
        if candidate["val_drawdown_pct"] > thresholds["g2_max_drawdown"]:
            failed.append("G2")
        if candidate["train_drawdown_pct"] > thresholds["g2_max_drawdown"] * 1.25:
            failed.append("G2")

    if gate_config.g3_train_profitability:
        weak_train = (
            candidate["train_profit_factor"] < thresholds["g3_train_min_pf"]
            or candidate["train_return_pct"] < thresholds["g3_train_min_return"]
        )
        if weak_train:
            exceptional = (
                candidate["val_profit_factor"] >= thresholds["g3_exceptional_val_pf"]
                and candidate["val_return_pct"] >= thresholds["g3_exceptional_val_return"]
                and candidate["val_trades"] >= thresholds["g1_min_trades"] * 2
                and candidate["val_drawdown_pct"] < thresholds["g2_max_drawdown"] * 0.75
                and candidate["val_win_rate"] > 50.0
                and candidate["val_return_dd_ratio"] >= thresholds["g5_min_return_dd_ratio"]
            )
            if not exceptional:
                failed.append("G3")

    if gate_config.g4_val_return and candidate["val_return_pct"] < thresholds["g4_min_val_return"]:
        failed.append("G4")

    if gate_config.g5_return_dd_ratio and candidate["val_return_dd_ratio"] < thresholds["g5_min_return_dd_ratio"]:
        failed.append("G5")

    if gate_config.g6_val_profit_factor and candidate["val_profit_factor"] < thresholds["g6_min_val_profit_factor"]:
        failed.append("G6")

    if gate_config.g7_robustness_gap:
        if (
            candidate["robustness_ratio"] < thresholds["g7_min_robustness"]
            and candidate["val_sharpe"] <= thresholds["g7_sharpe_override"]
        ):
            failed.append("G7")

    accepted = len(failed) == 0
    reason = "Accepted" if accepted else f"Rejected by: {', '.join(failed)}"
    return accepted, reason, failed


def run_ablation_preset(candidates: List[Dict[str, Any]], gate_config: GateConfig) -> PresetResult:
    result = PresetResult(
        preset_name=gate_config.name,
        description=gate_config.description,
        active_gates=gate_config.active_gate_names(),
        active_gate_count=gate_config.active_gate_count(),
        total_candidates=len(candidates),
    )

    gate_rejections = {f"G{idx}": 0 for idx in range(1, 8)}
    first_gate_rejections = {f"G{idx}": 0 for idx in range(1, 8)}
    robustness_values: List[float] = []
    returns: List[float] = []
    drawdowns: List[float] = []
    profit_factors: List[float] = []
    return_dd_values: List[float] = []

    for candidate in candidates:
        accepted, _reason, failed_gates = evaluate_gate(candidate, gate_config)
        if accepted:
            result.accepted_count += 1
            robustness_values.append(candidate["robustness_ratio"])
            returns.append(candidate["val_return_pct"])
            drawdowns.append(candidate["val_drawdown_pct"])
            profit_factors.append(candidate["val_profit_factor"])
            return_dd_values.append(candidate["val_return_dd_ratio"])
            if candidate["robustness_ratio"] < 0.5:
                result.false_acceptance_count += 1
        else:
            result.rejected_count += 1
            for gate in failed_gates:
                gate_rejections[gate] += 1
            if failed_gates:
                first_gate_rejections[failed_gates[0]] += 1

    result.acceptance_rate = result.accepted_count / max(1, result.total_candidates)
    result.false_acceptance_rate = result.false_acceptance_count / max(1, result.accepted_count)
    result.gate_rejections = gate_rejections
    result.gate_rejection_rates = {
        gate: count / max(1, result.total_candidates) for gate, count in gate_rejections.items()
    }
    result.first_gate_rejections = first_gate_rejections
    result.accepted_robustness_values = robustness_values
    result.accepted_robustness_median = _median(robustness_values)
    result.accepted_robustness_mean = statistics.mean(robustness_values) if robustness_values else 0.0
    result.accepted_robustness_iqr = (
        _quantile(robustness_values, 0.25),
        _quantile(robustness_values, 0.75),
    )
    result.accepted_val_return_median = _median(returns)
    result.accepted_val_drawdown_median = _median(drawdowns)
    result.accepted_val_pf_median = _median(profit_factors)
    result.accepted_val_return_dd_median = _median(return_dd_values)
    return result


def run_single_replication(seed: int, n_symbols: int) -> Dict[str, Any]:
    candidates = generate_candidate_pool(n_symbols=n_symbols, seed=seed)
    preset_results = {
        name: run_ablation_preset(candidates, config)
        for name, config in ABLATION_PRESETS.items()
    }
    return {"seed": seed, "candidates": candidates, "results": preset_results}


def _build_reference_summary(replication: Dict[str, Any]) -> Dict[str, Any]:
    candidates = replication["candidates"]
    results: Dict[str, PresetResult] = replication["results"]

    results_table = []
    for name in PRESET_ORDER:
        result = results[name]
        results_table.append(
            {
                "preset": name,
                "active_gates": result.active_gates,
                "active_gate_count": result.active_gate_count,
                "accepted": result.accepted_count,
                "rejected": result.rejected_count,
                "acceptance_rate_pct": round(result.acceptance_rate * 100, 1),
                "median_robustness": round(result.accepted_robustness_median, 3),
                "mean_robustness": round(result.accepted_robustness_mean, 3),
                "robustness_q25": round(result.accepted_robustness_iqr[0], 3),
                "robustness_q75": round(result.accepted_robustness_iqr[1], 3),
                "median_val_return_pct": round(result.accepted_val_return_median, 2),
                "median_val_drawdown_pct": round(result.accepted_val_drawdown_median, 2),
                "median_val_pf": round(result.accepted_val_pf_median, 3),
                "median_val_return_dd": round(result.accepted_val_return_dd_median, 3),
                "false_acceptance_count": result.false_acceptance_count,
                "false_acceptance_rate_pct": round(result.false_acceptance_rate * 100, 1),
            }
        )

    gate_rejection_heatmap = {}
    for name, result in results.items():
        gate_rejection_heatmap[name] = {
            "rejections": result.gate_rejections,
            "rejection_rates": {
                gate: round(rate * 100, 1) for gate, rate in result.gate_rejection_rates.items()
            },
            "first_rejections": result.first_gate_rejections,
        }

    robustness_box_plots = {}
    for name, result in results.items():
        values = sorted(result.accepted_robustness_values)
        if values:
            robustness_box_plots[name] = {
                "min": round(min(values), 3),
                "q25": round(_quantile(values, 0.25), 3),
                "median": round(_median(values), 3),
                "q75": round(_quantile(values, 0.75), 3),
                "max": round(max(values), 3),
                "mean": round(statistics.mean(values), 3),
                "n": len(values),
            }

    return {
        "seed": replication["seed"],
        "total_candidates": len(candidates),
        "results_table": results_table,
        "gate_rejection_heatmap": gate_rejection_heatmap,
        "robustness_box_plots": robustness_box_plots,
    }


def _aggregate_replications(replications: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name in PRESET_ORDER:
        acceptance_rates = []
        accepted_counts = []
        rejected_counts = []
        median_robustness = []
        false_acceptance_rates = []
        median_val_returns = []
        median_val_drawdowns = []
        median_val_pfs = []
        median_val_return_dd = []
        gate_rejection_rates: Dict[str, List[float]] = {f"G{idx}": [] for idx in range(1, 8)}
        first_gate_rejections: Dict[str, List[float]] = {f"G{idx}": [] for idx in range(1, 8)}

        for replication in replications:
            result: PresetResult = replication["results"][name]
            acceptance_rates.append(result.acceptance_rate * 100)
            accepted_counts.append(float(result.accepted_count))
            rejected_counts.append(float(result.rejected_count))
            median_robustness.append(result.accepted_robustness_median)
            false_acceptance_rates.append(result.false_acceptance_rate * 100)
            median_val_returns.append(result.accepted_val_return_median)
            median_val_drawdowns.append(result.accepted_val_drawdown_median)
            median_val_pfs.append(result.accepted_val_pf_median)
            median_val_return_dd.append(result.accepted_val_return_dd_median)
            for gate in gate_rejection_rates:
                gate_rejection_rates[gate].append(result.gate_rejection_rates.get(gate, 0.0) * 100)
                first_gate_rejections[gate].append(float(result.first_gate_rejections.get(gate, 0)))

        summary[name] = {
            "n_replications": len(replications),
            "accepted_count": _distribution_summary(accepted_counts),
            "rejected_count": _distribution_summary(rejected_counts),
            "acceptance_rate_pct": _distribution_summary(acceptance_rates),
            "median_robustness": _distribution_summary(median_robustness),
            "false_acceptance_rate_pct": _distribution_summary(false_acceptance_rates),
            "median_val_return_pct": _distribution_summary(median_val_returns),
            "median_val_drawdown_pct": _distribution_summary(median_val_drawdowns),
            "median_val_pf": _distribution_summary(median_val_pfs),
            "median_val_return_dd": _distribution_summary(median_val_return_dd),
            "gate_rejection_rate_pct": {
                gate: _distribution_summary(values) for gate, values in gate_rejection_rates.items()
            },
            "first_gate_rejections": {
                gate: _distribution_summary(values) for gate, values in first_gate_rejections.items()
            },
        }
    return summary


def _build_replication_level_tests(replications: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_extractors = {
        "acceptance_rate_pct": lambda result: result.acceptance_rate * 100,
        "median_robustness": lambda result: result.accepted_robustness_median,
        "false_acceptance_rate_pct": lambda result: result.false_acceptance_rate * 100,
        "median_val_return_pct": lambda result: result.accepted_val_return_median,
        "median_val_return_dd": lambda result: result.accepted_val_return_dd_median,
    }

    tests: Dict[str, Any] = {"pairwise_wilcoxon": {}, "bonferroni_correction": {}}
    baseline_name = "all_gates"
    comparison_names = [name for name in PRESET_ORDER if name != baseline_name]

    for metric_name, extractor in metric_extractors.items():
        metric_tests: Dict[str, Any] = {}
        baseline_values = [extractor(replication["results"][baseline_name]) for replication in replications]
        for comparison_name in comparison_names:
            comparison_values = [
                extractor(replication["results"][comparison_name]) for replication in replications
            ]
            metric_tests[f"{baseline_name}_vs_{comparison_name}"] = _wilcoxon_signed_rank(
                baseline_values, comparison_values
            )
        tests["pairwise_wilcoxon"][metric_name] = metric_tests

        valid_tests = [result for result in metric_tests.values() if result.get("p_value") is not None]
        corrected: Dict[str, Any] = {}
        if valid_tests:
            n_tests = len(valid_tests)
            for key, result in metric_tests.items():
                if result.get("p_value") is None:
                    continue
                raw_p = result["p_value"]
                bonferroni_p = min(1.0, raw_p * n_tests)
                corrected[key] = {
                    "raw_p": _compact_float(raw_p),
                    "bonferroni_p": _compact_float(bonferroni_p),
                    "significant_at_005": bonferroni_p < 0.05,
                    "significant_at_001": bonferroni_p < 0.01,
                }
        tests["bonferroni_correction"][metric_name] = corrected

    robustness_groups = [
        [replication["results"][name].accepted_robustness_median for replication in replications]
        for name in PRESET_ORDER
    ]
    tests["friedman_median_robustness"] = _friedman_test(robustness_groups)

    false_acceptance_groups = [
        [replication["results"][name].false_acceptance_rate * 100 for replication in replications]
        for name in PRESET_ORDER
    ]
    tests["friedman_false_acceptance_rate_pct"] = _friedman_test(false_acceptance_groups)
    return tests


def _build_export(
    reference_replication: Dict[str, Any],
    replications: List[Dict[str, Any]],
    n_symbols: int,
    n_replications: int,
) -> Dict[str, Any]:
    reference_summary = _build_reference_summary(reference_replication)
    replication_summary = _aggregate_replications(replications)
    statistical_tests = _build_replication_level_tests(replications)
    metadata = {
        "experiment_id": f"ISM2026-ABLATION-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "study_type": "replicated_calibrated_simulation",
        "base_seed": BASE_SEED,
        "reference_seed": reference_replication["seed"],
        "n_replications": n_replications,
        "n_symbols": n_symbols,
        "n_strategies": len(STRATEGIES),
        "n_regimes": len(REGIMES),
        "n_timeframes": len(TIMEFRAMES),
        "n_presets": len(ABLATION_PRESETS),
        "threshold_profile": THRESHOLD_PROFILE_NAME,
        "thresholds": THRESHOLDS,
        "generated": datetime.now().isoformat(),
        "method_note": (
            f"Results are from a replicated simulation using the {THRESHOLD_PROFILE_NAME} gate-threshold profile. "
            "Inference is based on paired replication-level summaries rather than pseudo-paired "
            "accepted-candidate slices from a single run."
        ),
    }
    return {
        "metadata": metadata,
        "reference_run": reference_summary,
        "results_table": reference_summary["results_table"],
        "gate_rejection_heatmap": reference_summary["gate_rejection_heatmap"],
        "robustness_box_plots": reference_summary["robustness_box_plots"],
        "replication_summary": replication_summary,
        "statistical_tests": statistical_tests,
    }


def run_full_experiment(n_symbols: int, n_replications: int) -> Dict[str, Any]:
    print(f"[1/4] Running reference replication (seed={BASE_SEED}, symbols={n_symbols})...")
    reference_replication = run_single_replication(seed=BASE_SEED, n_symbols=n_symbols)
    print(
        f"       Reference run produced {len(reference_replication['candidates'])} candidates "
        f"across {len(STRATEGIES)} strategies, {len(REGIMES)} regimes, and {len(TIMEFRAMES)} timeframes."
    )

    print(f"[2/4] Running paired replication suite (n={n_replications})...")
    replications: List[Dict[str, Any]] = []
    for offset in range(n_replications):
        seed = BASE_SEED + offset
        replications.append(run_single_replication(seed=seed, n_symbols=n_symbols))
        if (offset + 1) % max(1, n_replications // 5) == 0 or offset == n_replications - 1:
            print(f"       Completed {offset + 1}/{n_replications} replications")

    print("[3/4] Aggregating replication summaries and paired tests...")
    export = _build_export(
        reference_replication=reference_replication,
        replications=replications,
        n_symbols=n_symbols,
        n_replications=n_replications,
    )
    print("[4/4] Complete.")
    return export


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replicated gate ablation study")
    parser.add_argument(
        "--symbols",
        type=int,
        default=20,
        help="Number of synthetic symbols per replication (default: 20)",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=100,
        help="Number of paired simulation replications (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="ablation_results.json",
        help="Output JSON filename (default: ablation_results.json)",
    )
    parser.add_argument(
        "--threshold-profile",
        choices=sorted(THRESHOLD_PROFILES),
        default="historical",
        help="Gate predicate profile to use (default: historical)",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.symbols < 3:
        parser.error("--symbols must be at least 3")
    if args.replications < 10:
        parser.error("--replications must be at least 10 for paired inference")

    set_threshold_profile(args.threshold_profile)
    output_path = Path(__file__).parent / args.output
    export = run_full_experiment(n_symbols=args.symbols, n_replications=args.replications)
    output_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    print(f"\nResults written to: {output_path}")

    print("\nReplication-level summary (median across replications)")
    print("=" * 110)
    print(
        f"{'Preset':<22} {'Acc%':>8} {'Med.Rob':>10} {'FalseAcc%':>11} "
        f"{'MedRet%':>10} {'MedDD%':>9} {'MedPF':>8} {'MedRet/DD':>11}"
    )
    print("-" * 110)
    for name in PRESET_ORDER:
        row = export["replication_summary"][name]
        print(
            f"{name:<22} "
            f"{row['acceptance_rate_pct']['median']:>8.2f} "
            f"{row['median_robustness']['median']:>10.3f} "
            f"{row['false_acceptance_rate_pct']['median']:>11.2f} "
            f"{row['median_val_return_pct']['median']:>10.2f} "
            f"{row['median_val_drawdown_pct']['median']:>9.2f} "
            f"{row['median_val_pf']['median']:>8.3f} "
            f"{row['median_val_return_dd']['median']:>11.3f}"
        )
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
