"""
FX Portfolio Manager - Core Components
=======================================

Core classes and utilities for the FX Portfolio Manager.
Contains: Configuration, Data Loading, Feature Engineering, Data Splitting,
and Backtesting Engine with complete performance metrics.

This is the foundation module that provides:
- PipelineConfig: Global configuration management
- DataLoader: Load and resample OHLCV data from MT5 or CSV
- FeatureComputer: Technical indicator calculations
- DataSplitter: 80/30 split with 10% overlap for training/validation
- Backtester: Full backtesting engine matching original implementation
  - Includes Numba JIT-compiled main loop for 3-10x speedup

Version: 3.1 (Portfolio Manager - Numba backtester optimization)
"""

import os
import time
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Set

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# NUMBA JIT COMPILATION FOR BACKTESTER (optional - graceful fallback)
# =============================================================================

try:
    from numba import jit
    NUMBA_AVAILABLE = True
    logger.debug("Numba JIT compilation available for backtester")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.debug("Numba not available - using pure Python backtester")
    
    # Create no-op decorator for graceful fallback
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Exit reason codes for JIT kernel (cannot use Enum in Numba)
EXIT_NONE = 0
EXIT_SL = 1
EXIT_TP = 2
EXIT_EOD = 3


@jit(nopython=True, cache=True)
def _round_volume_numba(volume: float, min_lot: float, max_lot: float, step: float) -> float:
    """
    Numba-safe volume rounding helper.
    
    Replicates broker volume "floor to step" logic without calling Python methods.
    
    Args:
        volume: Raw position size
        min_lot: Minimum lot size
        max_lot: Maximum lot size
        step: Volume step (e.g., 0.01)
        
    Returns:
        Rounded volume clamped to min/max
    """
    if step <= 0.0:
        step = 0.01
    # Floor to step (risk-safe - never round up)
    rounded = np.floor(volume / step) * step
    # Clamp to min/max
    if rounded < min_lot:
        rounded = min_lot
    if rounded > max_lot:
        rounded = max_lot
    # Stabilize float noise
    return np.round(rounded, 8)


@jit(nopython=True, cache=True)
def _backtest_loop_numba(
    # OHLC arrays (float64 for precision)
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    # Signal array (int32)
    sig_arr: np.ndarray,
    # Pre-computed stop/entry prices (SL/TP must be in Python due to strategy.calculate_stops)
    sl_prices: np.ndarray,
    tp_prices: np.ndarray,
    entry_prices: np.ndarray,
    # Config values
    initial_capital: float,
    position_size_pct: float,
    half_spread: float,
    slippage_price: float,
    use_spread: bool,
    use_slippage: bool,
    use_commission: bool,
    commission_per_lot: float,
    # Tick calculation params
    tick_size: float,
    tick_value: float,
    pip_size: float,
    pip_value: float,
    # Volume rounding params (for live-equity sizing)
    min_lot: float,
    max_lot: float,
    volume_step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Numba JIT-compiled backtest loop for 3-10x speedup.
    
    CRITICAL: This function preserves EXACT semantics of the Python backtester:
    - SL is checked BEFORE TP (order matters for same-bar hits)
    - Long: exits at BID, checks if bid_low <= SL first, then if bid_high >= TP
    - Short: exits at ASK, checks if ask_high >= SL first, then if ask_low <= TP
    - Position sizing uses LIVE EQUITY (compounding preserved)
    - No fastmath to preserve floating-point precision
    - All arrays are float64 for precision
    
    Args:
        OHLC arrays, signals, pre-computed stops (entry_prices, sl_prices, tp_prices)
        Config params including position_size_pct and volume rounding params
        
    Returns:
        Tuple of arrays for trade reconstruction:
        - trade_signal_bars, trade_entry_bars, trade_exit_bars
        - trade_directions, trade_entry_prices, trade_exit_prices
        - trade_sl_prices, trade_tp_prices, trade_sizes
        - trade_pnl_dollars, trade_pnl_pips, trade_exit_reasons, trade_risk_amounts
        - final_equity, max_drawdown
    """
    n_bars = len(open_arr)
    
    # Pre-allocate trade arrays (max possible = n_bars trades)
    # We'll return only the filled portion
    max_trades = n_bars
    trade_signal_bars = np.zeros(max_trades, dtype=np.int64)
    trade_entry_bars = np.zeros(max_trades, dtype=np.int64)
    trade_exit_bars = np.zeros(max_trades, dtype=np.int64)
    trade_directions = np.zeros(max_trades, dtype=np.int32)
    trade_entry_prices = np.zeros(max_trades, dtype=np.float64)
    trade_exit_prices = np.zeros(max_trades, dtype=np.float64)
    trade_sl_prices = np.zeros(max_trades, dtype=np.float64)
    trade_tp_prices = np.zeros(max_trades, dtype=np.float64)
    trade_sizes = np.zeros(max_trades, dtype=np.float64)
    trade_pnl_dollars = np.zeros(max_trades, dtype=np.float64)
    trade_pnl_pips = np.zeros(max_trades, dtype=np.float64)
    trade_exit_reasons = np.zeros(max_trades, dtype=np.int32)
    trade_risk_amounts = np.zeros(max_trades, dtype=np.float64)
    
    # Equity curve
    equity_curve = np.zeros(n_bars, dtype=np.float64)
    equity_curve[0] = initial_capital
    
    # State
    equity = initial_capital
    peak_equity = equity
    max_drawdown = 0.0
    
    in_position = False
    position_direction = 0
    entry_price = 0.0
    entry_bar = 0
    signal_bar = 0
    stop_loss = 0.0
    take_profit = 0.0
    position_size = 0.0
    risk_amount_at_entry = 0.0
    
    trade_count = 0
    
    # Main loop
    for i in range(1, n_bars):
        open_price = open_arr[i]
        high_price = high_arr[i]
        low_price = low_arr[i]
        close_price = close_arr[i]
        
        # ================================================================
        # CHECK EXITS FOR OPEN POSITIONS
        # ================================================================
        if in_position:
            exit_price = -1.0  # Sentinel for "no exit"
            exit_reason = EXIT_NONE
            
            # Calculate bid/ask for this bar
            if use_spread:
                bid_high = high_price - half_spread
                bid_low = low_price - half_spread
                ask_high = high_price + half_spread
                ask_low = low_price + half_spread
            else:
                bid_high = high_price
                bid_low = low_price
                ask_high = high_price
                ask_low = low_price
            
            if position_direction == 1:  # Long position
                # CRITICAL: Check SL FIRST (exact same order as Python version)
                # Exit at BID price
                if bid_low <= stop_loss:
                    # Apply adverse slippage on stop exit only
                    if use_slippage:
                        exit_price = stop_loss - slippage_price
                    else:
                        exit_price = stop_loss
                    exit_reason = EXIT_SL
                # Check TP (only if SL not hit)
                elif bid_high >= take_profit:
                    exit_price = take_profit
                    exit_reason = EXIT_TP
                    
            else:  # Short position (position_direction == -1)
                # CRITICAL: Check SL FIRST (exact same order as Python version)
                # Exit at ASK price
                if ask_high >= stop_loss:
                    # Apply adverse slippage on stop exit only
                    if use_slippage:
                        exit_price = stop_loss + slippage_price
                    else:
                        exit_price = stop_loss
                    exit_reason = EXIT_SL
                # Check TP (only if SL not hit)
                elif ask_low <= take_profit:
                    exit_price = take_profit
                    exit_reason = EXIT_TP
            
            if exit_reason != EXIT_NONE:
                # Calculate P&L using tick-based math (MT5 parity)
                if tick_size > 0 and tick_value > 0:
                    price_diff = (exit_price - entry_price) * position_direction
                    ticks = price_diff / tick_size
                    pnl_dollars = ticks * tick_value * position_size
                else:
                    # Fallback to pip-based calculation
                    if position_direction == 1:
                        pnl_pips_calc = (exit_price - entry_price) / pip_size
                    else:
                        pnl_pips_calc = (entry_price - exit_price) / pip_size
                    pnl_dollars = pnl_pips_calc * position_size * pip_value
                
                # Calculate pips for reporting
                if position_direction == 1:
                    pnl_pips = (exit_price - entry_price) / pip_size
                else:
                    pnl_pips = (entry_price - exit_price) / pip_size
                
                # Deduct commission
                if use_commission:
                    pnl_dollars -= commission_per_lot * position_size
                
                equity += pnl_dollars
                
                # Record trade
                trade_signal_bars[trade_count] = signal_bar
                trade_entry_bars[trade_count] = entry_bar
                trade_exit_bars[trade_count] = i
                trade_directions[trade_count] = position_direction
                trade_entry_prices[trade_count] = entry_price
                trade_exit_prices[trade_count] = exit_price
                trade_sl_prices[trade_count] = stop_loss
                trade_tp_prices[trade_count] = take_profit
                trade_sizes[trade_count] = position_size
                trade_pnl_dollars[trade_count] = pnl_dollars
                trade_pnl_pips[trade_count] = pnl_pips
                trade_exit_reasons[trade_count] = exit_reason
                trade_risk_amounts[trade_count] = risk_amount_at_entry
                
                trade_count += 1
                in_position = False
        
        # ================================================================
        # CHECK ENTRIES
        # ================================================================
        if not in_position:
            # Use signal from PREVIOUS bar (i-1)
            signal = sig_arr[i - 1]
            
            if signal != 0:
                direction = signal
                signal_bar_index = i - 1
                
                # Get pre-computed entry/stop prices
                entry_price_candidate = entry_prices[i]
                stop_loss_candidate = sl_prices[i]
                take_profit_candidate = tp_prices[i]
                
                # Skip if any pre-computed value is NaN (indicates invalid entry)
                if np.isnan(entry_price_candidate) or np.isnan(stop_loss_candidate) or np.isnan(take_profit_candidate):
                    # Update equity curve even when skipping
                    equity_curve[i] = equity
                    if equity > peak_equity:
                        peak_equity = equity
                    if peak_equity > 0:
                        dd = (peak_equity - equity) / peak_equity * 100.0
                        if dd > max_drawdown:
                            max_drawdown = dd
                    continue
                
                # ============================================================
                # POSITION SIZING FROM LIVE EQUITY (compounding preserved)
                # ============================================================
                # Risk amount based on CURRENT equity (not initial capital)
                risk_amount_at_entry = equity * (position_size_pct / 100.0)
                
                # Calculate loss per lot at stop using tick math (or pip fallback)
                # distance to stop in price terms (should be positive)
                if direction == 1:  # Long: SL is below entry
                    dist = entry_price_candidate - stop_loss_candidate
                else:  # Short: SL is above entry
                    dist = stop_loss_candidate - entry_price_candidate
                
                loss_per_lot = 0.0
                if tick_size > 0.0 and tick_value > 0.0:
                    ticks = dist / tick_size
                    loss_per_lot = ticks * tick_value  # Loss for 1.0 lot
                else:
                    if pip_size > 0.0 and pip_value > 0.0:
                        pips = dist / pip_size
                        loss_per_lot = pips * pip_value
                
                if loss_per_lot <= 0.0:
                    # Cannot size safely - skip this entry
                    equity_curve[i] = equity
                    if equity > peak_equity:
                        peak_equity = equity
                    if peak_equity > 0:
                        dd = (peak_equity - equity) / peak_equity * 100.0
                        if dd > max_drawdown:
                            max_drawdown = dd
                    continue
                
                # Calculate position size
                raw_size = risk_amount_at_entry / loss_per_lot
                position_size = _round_volume_numba(raw_size, min_lot, max_lot, volume_step)
                
                if position_size <= 0.0:
                    # Invalid position size - skip
                    equity_curve[i] = equity
                    if equity > peak_equity:
                        peak_equity = equity
                    if peak_equity > 0:
                        dd = (peak_equity - equity) / peak_equity * 100.0
                        if dd > max_drawdown:
                            max_drawdown = dd
                    continue
                
                # Valid entry - set position state
                entry_price = entry_price_candidate
                stop_loss = stop_loss_candidate
                take_profit = take_profit_candidate
                
                in_position = True
                position_direction = direction
                entry_bar = i
                signal_bar = signal_bar_index
        
        # Track equity
        equity_curve[i] = equity
        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity * 100.0
            if dd > max_drawdown:
                max_drawdown = dd
    
    # ================================================================
    # CLOSE ANY REMAINING POSITION AT END
    # ================================================================
    if in_position:
        # Exit at close of last bar
        if use_spread:
            if position_direction == 1:
                exit_price = close_arr[n_bars - 1] - half_spread
            else:
                exit_price = close_arr[n_bars - 1] + half_spread
        else:
            exit_price = close_arr[n_bars - 1]
        
        # Calculate P&L
        if tick_size > 0 and tick_value > 0:
            price_diff = (exit_price - entry_price) * position_direction
            ticks = price_diff / tick_size
            pnl_dollars = ticks * tick_value * position_size
        else:
            if position_direction == 1:
                pnl_pips_calc = (exit_price - entry_price) / pip_size
            else:
                pnl_pips_calc = (entry_price - exit_price) / pip_size
            pnl_dollars = pnl_pips_calc * position_size * pip_value
        
        if position_direction == 1:
            pnl_pips = (exit_price - entry_price) / pip_size
        else:
            pnl_pips = (entry_price - exit_price) / pip_size
        
        if use_commission:
            pnl_dollars -= commission_per_lot * position_size
        
        equity += pnl_dollars
        
        # Record final trade
        trade_signal_bars[trade_count] = signal_bar
        trade_entry_bars[trade_count] = entry_bar
        trade_exit_bars[trade_count] = n_bars - 1
        trade_directions[trade_count] = position_direction
        trade_entry_prices[trade_count] = entry_price
        trade_exit_prices[trade_count] = exit_price
        trade_sl_prices[trade_count] = stop_loss
        trade_tp_prices[trade_count] = take_profit
        trade_sizes[trade_count] = position_size
        trade_pnl_dollars[trade_count] = pnl_dollars
        trade_pnl_pips[trade_count] = pnl_pips
        trade_exit_reasons[trade_count] = EXIT_EOD
        trade_risk_amounts[trade_count] = risk_amount_at_entry
        
        trade_count += 1
        
        # ============================================================
        # FIX: Update equity curve and drawdown after EOD close
        # ============================================================
        equity_curve[n_bars - 1] = equity
        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0.0:
            dd = (peak_equity - equity) / peak_equity * 100.0
            if dd > max_drawdown:
                max_drawdown = dd
    
    # Return only filled portion of arrays
    return (
        trade_signal_bars[:trade_count],
        trade_entry_bars[:trade_count],
        trade_exit_bars[:trade_count],
        trade_directions[:trade_count],
        trade_entry_prices[:trade_count],
        trade_exit_prices[:trade_count],
        trade_sl_prices[:trade_count],
        trade_tp_prices[:trade_count],
        trade_sizes[:trade_count],
        trade_pnl_dollars[:trade_count],
        trade_pnl_pips[:trade_count],
        trade_exit_reasons[:trade_count],
        trade_risk_amounts[:trade_count],
        equity,
        max_drawdown
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Global pipeline configuration.
    
    Controls all aspects of the Portfolio Manager including:
    - Data directories and paths
    - Train/validation split ratios
    - Backtest settings (capital, risk, costs)
    - Optimization parameters
    - Evaluation thresholds
    """
    # Data directories
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    output_dir: Path = field(default_factory=lambda: Path("./pm_outputs"))
    
    # Data split ratios (with overlap)
    # Training: 0-80%, Validation: 70-100% (10% overlap at 70-80%)
    # NOTE: val_pct is informational only; actual validation window is
    # derived from train_pct and overlap_pct.
    train_pct: float = 80.0
    val_pct: float = 30.0
    overlap_pct: float = 10.0
    
    # Backtest settings
    initial_capital: float = 10000.0
    risk_per_trade_pct: float = 1.0
    position_size_pct: float = 1.0  # Alias for risk_per_trade_pct
    
    # Cost modeling
    use_spread: bool = True
    use_commission: bool = True
    use_slippage: bool = True
    slippage_pips: float = 0.5
    
    # Optimization settings
    max_param_combos: int = 150
    min_trades: int = 25
    min_robustness: float = 0.20
    optimization_max_workers: int = 1  # 1 = sequential
    
    # Evaluation thresholds
    min_win_rate: float = 40.0
    min_profit_factor: float = 1.1
    min_sharpe: float = 0.5
    max_drawdown: float = 18.0
    


    # Scoring / validation mode
    # - "pm_weighted": original PM behavior (weighted composite + strict minimum criteria)
    # - "fx_backtester": align selection/tuning/validation with the simulator/backtester
    scoring_mode: str = "fx_backtester"

    # FX backtester-aligned optimization/validation thresholds
    fx_opt_min_trades: int = 15          # used during param search (train only)
    fx_val_min_trades: int = 15          # minimum validation trades required
    fx_val_max_drawdown: float = 18.0    # maximum validation drawdown (%)
    fx_val_sharpe_override: float = 0.3  # validation Sharpe can override robustness threshold
    # Generalization controls (fx_backtester)
    fx_selection_top_k: int = 5          # validate only top-K strategy/timeframe candidates
    fx_opt_top_k: int = 5                # validate only top-K parameter combos
    fx_gap_penalty_lambda: float = 0.70  # penalty strength for train->val score gaps
    fx_robustness_boost: float = 0.15    # robustness multiplier weight (0..0.5 recommended)
    fx_min_robustness_ratio: float = 0.80  # minimum val_score/train_score ratio for validation

    # Optuna objective behavior
    # If True, Optuna's trial objective includes validation metrics (may overfit to holdout split).
    # Default False: tune on train-only, validate on val during selection.
    optuna_use_val_in_objective: bool = False
    # Blended objective: train_weight * train_score + val_weight * val_score
    # Active only when optuna_use_val_in_objective is False and val data exists.
    # This provides bounded val influence without full val optimization.
    optuna_objective_blend_enabled: bool = True
    optuna_objective_train_weight: float = 0.80
    optuna_objective_val_weight: float = 0.20

    # Timeframes to evaluate
    timeframes: List[str] = field(default_factory=lambda: ['M5', 'M15', 'M30', 'H1', 'H4', 'D1'])
    
    
    # Live trading data window
    live_bars_count: int = 1500      # bars loaded per timeframe during live trading
    live_min_bars: int = 300         # minimum bars required to evaluate a timeframe in live trading
    actionable_score_margin: float = 0.92  # best actionable must be >= best_overall * margin
    # Optional manual backfill for legacy open positions lacking timeframe metadata.
    # Keys supported: "<ticket>", "ticket:<ticket>", "<magic>", "magic:<magic>", "<symbol>:<magic>"
    # Values: timeframe string (e.g., "D1", "H4", "M30")
    position_timeframe_overrides: Dict[str, str] = field(default_factory=dict)

    # Retrain periods to evaluate (in days)
    retrain_periods: List[int] = field(default_factory=lambda: [7, 14, 30, 60, 90])
    
    # Maximum bars to load per symbol (5 years of M5 data = ~500k bars)
    max_bars: int = 500000
    
    # Regime-aware optimization settings
    use_regime_optimization: bool = True
    regime_min_train_trades: int = 25  # Minimum trades per regime bucket in training
    regime_min_val_trades: int = 15    # Minimum trades per regime bucket in validation
    regime_freshness_decay: float = 0.85  # Freshness decay for stale timeframe signals
    regime_chop_no_trade: bool = False  # Hard no-trade when in CHOP with no winner
    regime_params_file: str = "regime_params.json"  # Path to tuned regime params
    
    # Hyperparameter tuning settings for regime optimization
    regime_enable_hyperparam_tuning: bool = True  # Enable hyperparameter tuning in regime optimization
    regime_hyperparam_top_k: int = 3  # Top K strategies to tune per regime (screening phase)
    regime_hyperparam_max_combos: int = 150  # Max param combinations to test per strategy
    
    # ===== REGIME WINNER PROFITABILITY GATES (FIX #1) =====
    # These thresholds ensure regime winners are actually profitable, not just "best loser"
    regime_min_val_profit_factor: float = 1.05   # Minimum validation PF to be stored as winner
    regime_min_val_return_pct: float = 5.0       # Minimum validation return % floor for validated winners
    regime_allow_losing_winners: bool = False    # If True, allows PF < 1 (not recommended)
    regime_no_winner_marker: str = "NO_TRADE"    # Strategy name for "no valid winner" state
    regime_validation_top_k: int = 5             # Max candidates to validate in descent order
    regime_min_val_return_dd_ratio: float = 1.0  # Hard gate: val return must >= val DD (ratio >= 1.0)

    # ===== SCORING CALIBRATION FLAGS (Scoring Audit Workstream C) =====
    # Feature flags for new scoring terms; all operate inside DD-passing set only.
    scoring_use_continuous_dd: bool = True        # Smooth exponential DD penalty (replaces buckets)
    scoring_use_sortino_blend: bool = True        # Blend Sortino with Sharpe in risk-adjusted score
    scoring_use_tail_risk: bool = True            # Penalize extreme worst-case R-multiples
    scoring_use_consistency: bool = True          # Penalize excessive losing streaks
    scoring_use_trade_frequency_bonus: bool = True  # Reward higher trade counts (log-scaled)

    # Pre-tuning training eligibility gates (lenient by design).
    train_min_profit_factor: float = 0.75
    train_min_return_pct: float = -20.0
    train_max_drawdown: float = 25.0

    # Optional exceptional-validation overrides for weak-train candidates.
    exceptional_val_profit_factor: float = 1.50  # was 1.15
    exceptional_val_return_pct: float = 10.0     # was 3.0
    
    # ===== LIVE RISK POLICY =====
    # Single-path risk sizing for winners-only live trading.
    live_risk_multiplier: float = 1.0      # multiplies position.risk_per_trade_pct
    live_max_risk_pct: float = 2.0         # hard cap before PositionConfig max_risk_pct
    min_trade_risk_pct: float = 0.1        # minimum non-zero risk for a placed trade

    # ===== MARGIN PROTECTION (BLACK SWAN GUARD) =====
    # Cycle-based margin protection layer (thresholds are config-driven).
    # Uses MT5-native margin_level (equity/margin*100). No sleep/cooldown logic.
    margin_entry_block_level: float = 100.0       # block new entries below this margin level %
    margin_recovery_start_level: float = 80.0     # start forced closures below this %
    margin_panic_level: float = 65.0              # aggressive forced closures below this %
    margin_reopen_level: float = 100.0            # resume entries above this %
    margin_recovery_closes_per_cycle: int = 1     # max forced closes per cycle in RECOVERY
    margin_panic_closes_per_cycle: int = 3        # max forced closes per cycle in PANIC

    
    # ===== DUAL-TRADE D1 + LOWER-TF SETTINGS (FIX #2) =====
    # Allow up to 2 concurrent trades: one D1 + one lower-TF
    allow_d1_plus_lower_tf: bool = True          # Enable D1 + lower-TF concurrent trades
    d1_secondary_risk_multiplier: float = 1.0   # Risk for second trade when D1 is open
    max_combined_risk_pct: float = 3.0          # Max combined risk per symbol (D1 + lower)
    secondary_trade_max_risk_pct: float = 0.9  # hard cap for the secondary (non-D1) trade
    
    # Optimization validity and persistence settings
    optimization_valid_days: int = 14  # Default validity period for optimized configs
    
    # Scoring weights for strategy selection
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        'sharpe': 0.25,
        'profit_factor': 0.20,
        'win_rate': 0.15,
        'total_return': 0.15,
        'max_drawdown': 0.15,
        'trade_count': 0.10
    })
    
    def __post_init__(self):
        """Ensure directories exist and sync aliases."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Clamp workers to sane minimum
        try:
            self.optimization_max_workers = max(1, int(self.optimization_max_workers))
        except Exception:
            self.optimization_max_workers = 1
        # Normalize scoring mode
        self.scoring_mode = str(self.scoring_mode).strip().lower()
        if self.scoring_mode not in {"pm_weighted", "fx_backtester"}:
            raise ValueError(f"Invalid scoring_mode: {self.scoring_mode}. Use 'pm_weighted' or 'fx_backtester'.")

        # Clamp regime validation descent depth to a safe minimum.
        try:
            self.regime_validation_top_k = max(1, int(self.regime_validation_top_k))
        except Exception:
            self.regime_validation_top_k = 5

        # Clamp return/DD hard gate threshold to a safe positive value.
        try:
            ratio = float(self.regime_min_val_return_dd_ratio)
            self.regime_min_val_return_dd_ratio = ratio if math.isfinite(ratio) and ratio > 0 else 1.0
        except Exception:
            self.regime_min_val_return_dd_ratio = 1.0

        # Normalize Optuna objective blend weights for robustness against config typos.
        try:
            tw = float(self.optuna_objective_train_weight)
            vw = float(self.optuna_objective_val_weight)
            total = tw + vw
            if total > 0:
                self.optuna_objective_train_weight = tw / total
                self.optuna_objective_val_weight = vw / total
            else:
                self.optuna_objective_train_weight = 0.80
                self.optuna_objective_val_weight = 0.20
        except Exception:
            self.optuna_objective_train_weight = 0.80
            self.optuna_objective_val_weight = 0.20
        self.position_size_pct = self.risk_per_trade_pct


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SignalType(Enum):
    """Trading signal types."""
    LONG = 1
    SHORT = -1
    FLAT = 0


class StrategyCategory(Enum):
    """Strategy categories."""
    TREND_FOLLOWING = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT_MOMENTUM = "breakout"
    VOLATILITY = "volatility"
    HYBRID = "hybrid"


class TradeStatus(Enum):
    """Trade exit reasons."""
    OPEN = "open"
    CLOSED_TP = "closed_tp"
    CLOSED_SL = "closed_sl"
    CLOSED_SIGNAL = "closed_signal"
    CLOSED_TIME = "closed_time"
    CLOSED_EOD = "end_of_data"


# =============================================================================
# UTILITY CLASSES
# =============================================================================

class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.elapsed = 0
        
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        logger.info(f"[TIME] {self.name}: {self.elapsed:.2f}s")


# =============================================================================
# INSTRUMENT SPECIFICATIONS
# =============================================================================

@dataclass
class InstrumentSpec:
    """
    Specification for a trading instrument.
    
    Contains all instrument-specific parameters needed for:
    - Pip calculations
    - Spread/commission modeling
    - Position sizing
    - P&L calculation (tick-based for broker parity)
    
    Note: pip_value represents USD per pip per STANDARD lot.
    - For FX pairs: 1 standard lot = 100,000 units
    - For Gold (XAUUSD): 1 lot = 100 troy oz
    - For Silver (XAGUSD): 1 lot = 5,000 oz
    
    Broker-real fields (from MT5 snapshot):
    - tick_size, tick_value: For tick-based P&L calculation
    - contract_size: For proper lot sizing
    - volume_step: For proper volume rounding
    - stops_level: Minimum stop distance in points
    """
    symbol: str
    pip_position: int  # Decimal position (4 for EURUSD, 2 for USDJPY/XAUUSD, 3 for XAGUSD)
    pip_value: float = 10.0  # USD per pip per standard lot
    spread_avg: float = 1.0  # Average spread in pips
    min_lot: float = 0.01
    max_lot: float = 100.0
    commission_per_lot: float = 7.0  # Commission in USD per round trip per lot
    swap_long: float = 0.0  # Daily swap for long positions
    swap_short: float = 0.0  # Daily swap for short positions
    
    # Broker-real fields (populated from MT5 snapshot or broker_specs.json)
    tick_size: float = 0.0  # trade_tick_size from MT5
    tick_value: float = 0.0  # trade_tick_value from MT5 (value of 1 tick in account currency)
    contract_size: float = 100000.0  # trade_contract_size from MT5
    volume_step: float = 0.01  # volume_step from MT5
    stops_level: int = 0  # trade_stops_level from MT5 (in points)
    point: float = 0.0  # point from MT5 (minimum price change)
    digits: int = 5  # digits from MT5
    
    def __post_init__(self):
        """Initialize derived values if not set."""
        # Derive point from pip_position if not explicitly set
        if self.point == 0.0:
            self.point = 10 ** (-self.digits) if self.digits > 0 else 10 ** (-self.pip_position)
        # Derive tick_size from point if not set
        if self.tick_size == 0.0:
            self.tick_size = self.point
        # Derive tick_value from pip_value if not set
        if self.tick_value == 0.0:
            # tick_value = pip_value * (tick_size / pip_size)
            pip_sz = 10 ** (-self.pip_position)
            if pip_sz > 0:
                self.tick_value = self.pip_value * (self.tick_size / pip_sz)
    
    @property
    def pip_size(self) -> float:
        """Price change for one pip."""
        return 10 ** (-self.pip_position)
    
    def price_to_pips(self, price_diff: float) -> float:
        """Convert price difference to pips."""
        return price_diff / self.pip_size
    
    def pips_to_price(self, pips: float) -> float:
        """Convert pips to price difference."""
        return pips * self.pip_size
    
    def get_half_spread_price(self) -> float:
        """Get half spread as price (for bid/ask calculations)."""
        return self.pips_to_price(self.spread_avg / 2.0)
    
    def get_entry_price(self, mid_price: float, is_long: bool) -> float:
        """
        Calculate entry price including spread.
        - Longs enter at ASK (mid + half_spread)
        - Shorts enter at BID (mid - half_spread)
        """
        half_spread = self.get_half_spread_price()
        return mid_price + half_spread if is_long else mid_price - half_spread
    
    def get_exit_price(self, mid_price: float, is_long: bool) -> float:
        """
        Calculate exit price including spread.
        - Longs exit at BID (mid - half_spread)
        - Shorts exit at ASK (mid + half_spread)
        """
        half_spread = self.get_half_spread_price()
        return mid_price - half_spread if is_long else mid_price + half_spread
    
    def calculate_tick_profit(self, entry_price: float, exit_price: float, 
                              volume: float, direction: int) -> float:
        """
        Calculate P&L using tick-based math (MT5 parity).
        
        This is the same calculation MT5 uses internally:
        profit = ticks * tick_value * volume
        where ticks = (exit - entry) / tick_size * direction_sign
        
        Args:
            entry_price: Entry price
            exit_price: Exit price  
            volume: Position size in lots
            direction: 1 for long, -1 for short
            
        Returns:
            Profit/loss in account currency
        """
        if self.tick_size <= 0 or self.tick_value <= 0:
            # Fallback to pip-based calculation
            if direction == 1:
                pnl_pips = self.price_to_pips(exit_price - entry_price)
            else:
                pnl_pips = self.price_to_pips(entry_price - exit_price)
            return pnl_pips * volume * self.pip_value
        
        # Tick-based calculation (MT5 parity)
        price_diff = (exit_price - entry_price) * direction
        ticks = price_diff / self.tick_size
        profit = ticks * self.tick_value * volume
        return profit
    
    def calculate_loss_at_stop(self, entry_price: float, stop_price: float,
                               volume: float, direction: int) -> float:
        """
        Calculate loss amount at stop price using tick math.
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            volume: Position size in lots
            direction: 1 for long, -1 for short
            
        Returns:
            Absolute loss amount in account currency
        """
        profit = self.calculate_tick_profit(entry_price, stop_price, volume, direction)
        return abs(profit)
    
    def round_volume(self, volume: float) -> float:
        """
        Round volume to valid lot size using broker volume_step.
        
        Uses floor rounding to avoid risk increases.
        
        Args:
            volume: Raw volume to round
            
        Returns:
            Volume rounded to volume_step, clamped to min/max
        """
        step = self.volume_step if self.volume_step > 0 else 0.01
        # Floor to step (risk-safe)
        rounded = math.floor(volume / step) * step
        # Clamp to min/max
        rounded = max(self.min_lot, min(self.max_lot, rounded))
        # Round to avoid floating point issues
        return round(rounded, 8)
    
    def get_min_stop_distance_price(self) -> float:
        """
        Get minimum stop distance as price distance.
        
        Returns:
            Minimum distance in price units
        """
        if self.stops_level > 0 and self.point > 0:
            return self.stops_level * self.point
        return 0.0


def sync_instrument_spec_from_mt5(spec: InstrumentSpec, mt5_symbol_info) -> InstrumentSpec:
    """
    Update InstrumentSpec with live MT5 broker values.
    Call this during LiveTrader initialization for all symbols.

    This keeps sizing and stop-distance logic aligned with broker reality.
    """
    if not mt5_symbol_info:
        return spec

    # Tick + volume sizing primitives
    spec.tick_value = mt5_symbol_info.trade_tick_value
    spec.tick_size = mt5_symbol_info.trade_tick_size
    spec.volume_step = mt5_symbol_info.volume_step
    spec.min_lot = mt5_symbol_info.volume_min
    spec.max_lot = mt5_symbol_info.volume_max

    # Spread (MT5 points -> pips)
    if mt5_symbol_info.spread > 0 and spec.pip_size > 0:
        spec.spread_avg = mt5_symbol_info.spread * mt5_symbol_info.point / spec.pip_size

    if mt5_symbol_info.trade_contract_size > 0:
        spec.contract_size = mt5_symbol_info.trade_contract_size

    spec.point = mt5_symbol_info.point
    spec.digits = mt5_symbol_info.digits
    spec.stops_level = mt5_symbol_info.trade_stops_level
    spec.swap_long = mt5_symbol_info.swap_long
    spec.swap_short = mt5_symbol_info.swap_short

    return spec


# Default instrument specifications
# These are used when MT5 data is not available (e.g., backtesting from CSV)
# Live trading will override these with actual broker values
INSTRUMENT_SPECS = {
    # Major FX Pairs
    'EURUSD': InstrumentSpec('EURUSD', 4, 10.0, 1.0, 0.01, 100.0, 7.0, -6.5, 1.2),
    'GBPUSD': InstrumentSpec('GBPUSD', 4, 10.0, 1.2, 0.01, 100.0, 7.0, -5.0, 0.8),
    'AUDUSD': InstrumentSpec('AUDUSD', 4, 10.0, 1.2, 0.01, 100.0, 7.0, -4.0, 0.5),
    'NZDUSD': InstrumentSpec('NZDUSD', 4, 10.0, 1.5, 0.01, 100.0, 7.0, -3.5, 0.3),
    'USDJPY': InstrumentSpec('USDJPY', 2, 9.0, 1.0, 0.01, 100.0, 7.0, 8.5, -15.0),
    'USDCAD': InstrumentSpec('USDCAD', 4, 7.5, 1.5, 0.01, 100.0, 7.0, 2.5, -8.0),
    'USDCHF': InstrumentSpec('USDCHF', 4, 11.0, 1.5, 0.01, 100.0, 7.0, 5.0, -10.0),
    
    # Cross pairs
    'AUDNZD': InstrumentSpec('AUDNZD', 4, 6.0, 2.5, 0.01, 100.0, 8.0, -2.0, -1.5),
    'EURGBP': InstrumentSpec('EURGBP', 4, 12.5, 1.5, 0.01, 100.0, 7.0, -4.0, 0.5),
    'EURJPY': InstrumentSpec('EURJPY', 2, 9.0, 1.5, 0.01, 100.0, 7.0, 3.0, -10.0),
    'GBPJPY': InstrumentSpec('GBPJPY', 2, 9.0, 2.0, 0.01, 100.0, 7.0, 5.0, -12.0),
    
    # Exotic FX Pairs
    'USDBRL': InstrumentSpec('USDBRL', 4, 2.0, 50.0, 0.01, 100.0, 15.0, 15.0, -40.0),
    'USDMXN': InstrumentSpec('USDMXN', 4, 0.55, 30.0, 0.01, 100.0, 12.0, 20.0, -45.0),
    'USDPLN': InstrumentSpec('USDPLN', 4, 2.5, 25.0, 0.01, 100.0, 10.0, 3.0, -12.0),
    'USDNOK': InstrumentSpec('USDNOK', 4, 0.95, 30.0, 0.01, 100.0, 10.0, 2.0, -10.0),
    'USDSEK': InstrumentSpec('USDSEK', 4, 0.95, 30.0, 0.01, 100.0, 10.0, 2.0, -10.0),
    'USDSGD': InstrumentSpec('USDSGD', 4, 7.5, 2.0, 0.01, 100.0, 8.0, 1.5, -6.0),
    'USDTRY': InstrumentSpec('USDTRY', 4, 0.035, 100.0, 0.01, 100.0, 20.0, 50.0, -150.0),
    'USDZAR': InstrumentSpec('USDZAR', 4, 0.55, 50.0, 0.01, 100.0, 15.0, 12.0, -35.0),
    'EURZAR': InstrumentSpec('EURZAR', 4, 0.55, 70.0, 0.01, 100.0, 15.0, 10.0, -30.0),
    'GBPZAR': InstrumentSpec('GBPZAR', 4, 0.55, 80.0, 0.01, 100.0, 15.0, 12.0, -35.0),
    'USDCNH': InstrumentSpec('USDCNH', 4, 1.4, 20.0, 0.01, 100.0, 10.0, 2.0, -10.0),
    'EURCNH': InstrumentSpec('EURCNH', 4, 1.4, 10.0, 0.01, 100.0, 10.0, 2.0, -10.0),
    'EURPLN': InstrumentSpec('EURPLN', 4, 2.5, 25.0, 0.01, 100.0, 10.0, 3.0, -12.0),
    'EURNOK': InstrumentSpec('EURNOK', 4, 0.95, 30.0, 0.01, 100.0, 10.0, 2.0, -10.0),
    'EURSEK': InstrumentSpec('EURSEK', 4, 0.95, 30.0, 0.01, 100.0, 10.0, 2.0, -10.0),
    'GBPNOK': InstrumentSpec('GBPNOK', 4, 0.95, 35.0, 0.01, 100.0, 10.0, 2.0, -12.0),
    'GBPSEK': InstrumentSpec('GBPSEK', 4, 0.95, 35.0, 0.01, 100.0, 10.0, 2.0, -12.0),
    'EURTRY': InstrumentSpec('EURTRY', 4, 0.035, 120.0, 0.01, 100.0, 20.0, 45.0, -130.0),
    'Platinum': InstrumentSpec('Platinum', 2, 1.0, 8.0, 0.01, 100.0, 5.0, -5.0, -3.0),
    'Palladium': InstrumentSpec('Palladium', 2, 1.0, 8.0, 0.01, 100.0, 5.0, -6.0, -4.0),
    'EURMXN': InstrumentSpec('EURMXN', 4, 0.55, 140.0, 0.01, 100.0, 12.0, 20.0, -45.0),
    'NOKJPY': InstrumentSpec('NOKJPY', 2, 9.0, 2.0, 0.01, 100.0, 8.0, -1.0, -2.0),
    
    # Commodities
    # XAUUSD: 1 lot = 100 troy oz, pip_size = 0.01, pip_value = $1.00
    'XAUUSD': InstrumentSpec('XAUUSD', 2, 1.0, 3.0, 0.01, 100.0, 5.0, -8.0, -5.0),
    # XAGUSD: 1 lot = 5,000 oz, pip_size = 0.001, pip_value = $5.00
    'XAGUSD': InstrumentSpec('XAGUSD', 3, 5.0, 2.0, 0.01, 100.0, 5.0, -3.0, -2.0),
    # Metal crosses
    'XAUEUR': InstrumentSpec('XAUEUR', 2, 1.0, 3.0, 0.01, 100.0, 5.0, -8.0, -5.0),
    'XAUGBP': InstrumentSpec('XAUGBP', 2, 1.0, 3.0, 0.01, 100.0, 5.0, -8.0, -5.0),
    'XAUAUD': InstrumentSpec('XAUAUD', 2, 1.0, 3.0, 0.01, 100.0, 5.0, -8.0, -5.0),
    'XAGEUR': InstrumentSpec('XAGEUR', 3, 5.0, 2.0, 0.01, 100.0, 5.0, -3.0, -2.0),
    'XRX': InstrumentSpec('XRX', 3, 1.0, 5.0, 0.01, 100.0, 5.0, 0.0, 0.0),
    # Energy CFDs
    'XTIUSD': InstrumentSpec('XTIUSD', 2, 1.0, 3.0, 0.01, 100.0, 5.0, 0.0, 0.0),
    'XBRUSD': InstrumentSpec('XBRUSD', 2, 1.0, 3.0, 0.01, 100.0, 5.0, 0.0, 0.0),
    'XNGUSD': InstrumentSpec('XNGUSD', 3, 1.0, 4.0, 0.01, 100.0, 5.0, 0.0, 0.0),

    # Additional FX Pairs (aligned with config)
    'AUDJPY': InstrumentSpec('AUDJPY', 2, 9.0, 2.0, 0.01, 100.0, 7.0, -4.0, 1.0),
    'EURAUD': InstrumentSpec('EURAUD', 4, 7.0, 2.0, 0.01, 100.0, 7.0, -4.0, 0.5),
    'EURCHF': InstrumentSpec('EURCHF', 4, 11.0, 2.0, 0.01, 100.0, 7.0, -3.0, 0.5),
    'EURCAD': InstrumentSpec('EURCAD', 4, 7.5, 2.0, 0.01, 100.0, 7.0, -4.0, 0.5),
    'EURNZD': InstrumentSpec('EURNZD', 4, 6.0, 3.0, 0.01, 100.0, 7.0, -4.0, 0.5),
    'GBPAUD': InstrumentSpec('GBPAUD', 4, 7.0, 2.5, 0.01, 100.0, 7.0, -5.0, 0.8),
    'GBPCAD': InstrumentSpec('GBPCAD', 4, 7.5, 2.5, 0.01, 100.0, 7.0, -5.0, 0.8),
    'GBPCHF': InstrumentSpec('GBPCHF', 4, 11.0, 2.5, 0.01, 100.0, 7.0, -5.0, 0.8),
    'CADJPY': InstrumentSpec('CADJPY', 2, 9.0, 2.0, 0.01, 100.0, 7.0, -2.0, 0.5),
    'NZDJPY': InstrumentSpec('NZDJPY', 2, 9.0, 2.0, 0.01, 100.0, 7.0, -2.0, 0.5),
    'AUDCAD': InstrumentSpec('AUDCAD', 4, 7.5, 1.5, 0.01, 100.0, 7.0, 2.5, -8.0),
    'AUDCHF': InstrumentSpec('AUDCHF', 4, 11.0, 2.0, 0.01, 100.0, 7.0, -3.0, 0.5),
    'CADCHF': InstrumentSpec('CADCHF', 4, 11.0, 2.0, 0.01, 100.0, 7.0, -3.0, 0.5),
    'CHFJPY': InstrumentSpec('CHFJPY', 2, 9.0, 1.0, 0.01, 100.0, 7.0, 8.5, -15.0),
    'NZDCAD': InstrumentSpec('NZDCAD', 4, 7.5, 1.5, 0.01, 100.0, 7.0, 2.5, -8.0),
    'NZDCHF': InstrumentSpec('NZDCHF', 4, 11.0, 2.0, 0.01, 100.0, 7.0, -3.0, 0.5),
    'GBPNZD': InstrumentSpec('GBPNZD', 4, 6.0, 3.0, 0.01, 100.0, 7.0, -4.0, 0.5),

    # Crypto (CFDs) - defaults used mainly for offline backtests; live trading overrides via broker specs
    'BTCUSD': InstrumentSpec('BTCUSD', 2, 1.0, 20.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'ETHUSD': InstrumentSpec('ETHUSD', 2, 1.0, 8.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'LTCUSD': InstrumentSpec('LTCUSD', 2, 1.0, 15.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'SOLUSD': InstrumentSpec('SOLUSD', 2, 1.0, 15.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'BCHUSD': InstrumentSpec('BCHUSD', 2, 1.0, 20.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'DOGUSD': InstrumentSpec('DOGUSD', 4, 1.0, 20.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'TRXUSD': InstrumentSpec('TRXUSD', 4, 1.0, 15.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'XRPUSD': InstrumentSpec('XRPUSD', 4, 1.0, 12.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'TONUSD': InstrumentSpec('TONUSD', 3, 1.0, 12.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'BTCETH': InstrumentSpec('BTCETH', 2, 1.0, 8.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'BTCXAU': InstrumentSpec('BTCXAU', 2, 1.0, 10.0, 0.01, 100.0, 0.0, 0.0, 0.0),

    # Indices - point-based instruments; defaults used mainly for offline backtests; live trading overrides via broker specs
    'US100': InstrumentSpec('US100', 0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'US30':  InstrumentSpec('US30',  0, 1.0, 3.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'DE30':  InstrumentSpec('DE30',  0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'EU50':  InstrumentSpec('EU50',  0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'UK100': InstrumentSpec('UK100', 0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'JP225': InstrumentSpec('JP225', 0, 1.0, 8.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'US500': InstrumentSpec('US500', 0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'FR40':  InstrumentSpec('FR40',  0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'ES35':  InstrumentSpec('ES35',  0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'HK50':  InstrumentSpec('HK50',  0, 1.0, 8.0, 0.01, 100.0, 0.0, 0.0, 0.0),
    'AU200': InstrumentSpec('AU200', 0, 1.0, 2.0, 0.01, 100.0, 0.0, 0.0, 0.0),
}


# Broker specs cache (loaded from broker_specs.json if available)
_BROKER_SPECS_CACHE: Dict[str, Dict[str, Any]] = {}
_BROKER_SPECS_LOADED = False
_BROKER_SPECS_PATH = "broker_specs.json"
_BROKER_SPECS_LOADED_PATH: Optional[str] = None

# Config-provided instrument spec overrides (single source of truth)
_CONFIG_INSTRUMENT_SPECS: Dict[str, InstrumentSpec] = {}
_CONFIG_SPEC_DEFAULTS: Dict[str, Any] = {}
_CONFIG_SPECS_LOADED: bool = False
_CONFIG_SPECS_LOADED_PATH: Optional[str] = None

# Warn once per symbol when spec is missing to avoid log spam
_MISSING_SPEC_WARNED: Set[str] = set()


def load_broker_specs(filepath: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load broker specifications from JSON file.
    
    The JSON file should contain MT5 symbol info captured via save_broker_specs().
    
    Args:
        filepath: Path to broker_specs.json
        
    Returns:
        Dict mapping symbol -> broker spec dict
    """
    global _BROKER_SPECS_CACHE, _BROKER_SPECS_LOADED, _BROKER_SPECS_LOADED_PATH
    
    if filepath is None:
        filepath = _BROKER_SPECS_PATH

    # Return cached if already loaded from the same path
    if _BROKER_SPECS_LOADED and _BROKER_SPECS_LOADED_PATH == filepath:
        return _BROKER_SPECS_CACHE
    
    try:
        import json
        with open(filepath, 'r') as f:
            specs = json.load(f)
        _BROKER_SPECS_CACHE = specs
        _BROKER_SPECS_LOADED = True
        _BROKER_SPECS_LOADED_PATH = filepath
        logger.info(f"Loaded broker specs for {len(specs)} symbols from {filepath}")
    except FileNotFoundError:
        logger.debug(f"No broker specs file found at {filepath}, using defaults")
        _BROKER_SPECS_LOADED = True
        _BROKER_SPECS_LOADED_PATH = filepath
    except Exception as e:
        logger.warning(f"Failed to load broker_specs.json: {e}")
        _BROKER_SPECS_LOADED = True
        _BROKER_SPECS_LOADED_PATH = filepath
    
    return _BROKER_SPECS_CACHE


def set_broker_specs_path(filepath: str) -> None:
    """
    Set the broker specs JSON path and force reload on next access.

    Args:
        filepath: Path to broker specs JSON file
    """
    global _BROKER_SPECS_PATH, _BROKER_SPECS_LOADED, _BROKER_SPECS_LOADED_PATH
    if filepath:
        _BROKER_SPECS_PATH = filepath
    _BROKER_SPECS_LOADED = False
    _BROKER_SPECS_LOADED_PATH = None


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol for lookup (remove suffixes, uppercase)."""
    if not symbol:
        return ""
    return symbol.split('.')[0].split('#')[0].upper()


def _maybe_load_config_specs(config_path: Optional[str] = None) -> None:
    """Lazy-load instrument specs from config.json for child processes."""
    global _CONFIG_SPECS_LOADED, _CONFIG_SPECS_LOADED_PATH
    if _CONFIG_INSTRUMENT_SPECS:
        return

    path = config_path or os.environ.get("PM_CONFIG_PATH") or "config.json"
    if not path:
        _CONFIG_SPECS_LOADED = True
        _CONFIG_SPECS_LOADED_PATH = path
        return

    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if _CONFIG_SPECS_LOADED and _CONFIG_SPECS_LOADED_PATH == path:
        return

    try:
        import json
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        _CONFIG_SPECS_LOADED = True
        _CONFIG_SPECS_LOADED_PATH = path
        return
    except Exception as exc:
        logger.warning(f"Failed to load config specs from {path}: {exc}")
        _CONFIG_SPECS_LOADED = True
        _CONFIG_SPECS_LOADED_PATH = path
        return

    broker_specs_path = data.get("broker_specs_path")
    if broker_specs_path:
        if not os.path.isabs(broker_specs_path):
            broker_specs_path = os.path.join(os.path.dirname(path), broker_specs_path)
        set_broker_specs_path(broker_specs_path)

    set_instrument_specs(
        specs=data.get("instrument_specs"),
        defaults=data.get("instrument_spec_defaults"),
    )
    _CONFIG_SPECS_LOADED = True
    _CONFIG_SPECS_LOADED_PATH = path


def _spec_from_config_dict(symbol: str, data: Optional[Dict[str, Any]],
                           defaults: Optional[Dict[str, Any]] = None) -> InstrumentSpec:
    """
    Build an InstrumentSpec from config dict with optional defaults and inheritance.
    """
    data = data or {}
    merged: Dict[str, Any] = {}

    # Inherit from another symbol if requested
    inherit_symbol = data.get("inherit")
    if inherit_symbol:
        parent_key = _normalize_symbol(str(inherit_symbol))
        parent_spec = _CONFIG_INSTRUMENT_SPECS.get(parent_key)
        if parent_spec is None:
            broker_specs = load_broker_specs()
            if parent_key in broker_specs:
                parent_spec = _create_spec_from_broker_data(parent_key, broker_specs[parent_key])
            elif parent_key in INSTRUMENT_SPECS:
                parent_spec = INSTRUMENT_SPECS[parent_key]
        if parent_spec is not None:
            merged.update(asdict(parent_spec))

    # Apply defaults, then explicit overrides
    if defaults:
        merged.update(defaults)
    merged.update({k: v for k, v in data.items() if k != "inherit"})

    # Required fields
    merged["symbol"] = symbol

    # Infer pip_position if not provided
    if "pip_position" not in merged or merged["pip_position"] in (None, ""):
        digits = merged.get("digits")
        if digits is not None:
            try:
                digits_i = int(digits)
                merged["pip_position"] = digits_i - 1 if digits_i in (3, 5) else digits_i
            except Exception:
                merged["pip_position"] = 4
        else:
            merged["pip_position"] = 4

    # Filter only valid InstrumentSpec fields
    valid_keys = {f.name for f in fields(InstrumentSpec)}
    filtered = {k: v for k, v in merged.items() if k in valid_keys}

    # Coerce numeric types lightly (avoid strings in config)
    float_fields = {
        "pip_value", "spread_avg", "min_lot", "max_lot", "commission_per_lot",
        "swap_long", "swap_short", "tick_size", "tick_value", "contract_size",
        "volume_step", "point"
    }
    int_fields = {"pip_position", "stops_level", "digits"}
    for k in float_fields:
        if k in filtered and filtered[k] is not None:
            try:
                filtered[k] = float(filtered[k])
            except Exception:
                pass
    for k in int_fields:
        if k in filtered and filtered[k] is not None:
            try:
                filtered[k] = int(filtered[k])
            except Exception:
                pass

    return InstrumentSpec(**filtered)


def set_instrument_specs(specs: Optional[Dict[str, Any]] = None,
                         defaults: Optional[Dict[str, Any]] = None) -> int:
    """
    Set config-provided instrument specs and defaults.

    Args:
        specs: Dict of symbol -> spec dict (may include 'inherit')
        defaults: Dict of default values applied to config specs

    Returns:
        Number of specs loaded
    """
    global _CONFIG_INSTRUMENT_SPECS, _CONFIG_SPEC_DEFAULTS
    _CONFIG_SPEC_DEFAULTS = defaults or {}
    _CONFIG_INSTRUMENT_SPECS = {}
    if not specs:
        return 0

    # Two-pass load: first specs without inherit, then inherit-based
    pending: Dict[str, Any] = {}
    for sym, data in specs.items():
        key = _normalize_symbol(sym)
        if isinstance(data, dict) and data.get("inherit"):
            pending[key] = data
        else:
            _CONFIG_INSTRUMENT_SPECS[key] = _spec_from_config_dict(key, data, _CONFIG_SPEC_DEFAULTS)

    for sym, data in pending.items():
        _CONFIG_INSTRUMENT_SPECS[sym] = _spec_from_config_dict(sym, data, _CONFIG_SPEC_DEFAULTS)

    return len(_CONFIG_INSTRUMENT_SPECS)


def _clone_spec(spec: InstrumentSpec, symbol_override: Optional[str] = None) -> InstrumentSpec:
    """Return a shallow copy of an InstrumentSpec, optionally overriding symbol."""
    spec_dict = asdict(spec)
    if symbol_override:
        spec_dict["symbol"] = symbol_override
    return InstrumentSpec(**spec_dict)


def _apply_spec_defaults(spec: InstrumentSpec) -> InstrumentSpec:
    """Apply config-level default overrides to a spec (non-broker only)."""
    if not _CONFIG_SPEC_DEFAULTS:
        return spec
    spec_dict = asdict(spec)
    for k, v in _CONFIG_SPEC_DEFAULTS.items():
        if k in spec_dict and v is not None:
            spec_dict[k] = v
    return InstrumentSpec(**spec_dict)


def save_broker_specs(specs: Dict[str, Dict[str, Any]], filepath: str = "broker_specs.json"):
    """
    Save broker specifications to JSON file.
    
    Args:
        specs: Dict mapping symbol -> broker spec dict
        filepath: Output file path
    """
    import json
    with open(filepath, 'w') as f:
        json.dump(specs, f, indent=2)
    logger.info(f"Saved broker specs for {len(specs)} symbols to {filepath}")


def _create_spec_from_broker_data(symbol: str, broker_data: Dict[str, Any]) -> InstrumentSpec:
    """Create an InstrumentSpec from broker spec data.

    This function must be robust to broker/MT5 edge-cases where certain fields
    can be missing or zero for CFDs/crypto/indices (e.g., point=0, tick_size=0,
    pip_size=0). When this happens we should *not* crash; we should sanitize
    values and allow the system to fall back to safe defaults.

    Args:
        symbol: Symbol name
        broker_data: Dict with MT5 symbol info fields

    Returns:
        InstrumentSpec populated with broker-real values (sanitized)
    """
    # Digits / point
    digits = int(broker_data.get('digits', 5))
    point = float(broker_data.get('point', 0.0) or 0.0)

    # Derive point if broker returned 0/None
    if point <= 0.0:
        # Best-effort: MT5 point is usually 10**(-digits)
        try:
            point = float(10 ** (-digits)) if digits > 0 else 0.0001
        except Exception:
            point = 0.0001

    # Determine pip_position from digits
    if digits in (3, 5):
        pip_position = digits - 1
    else:
        pip_position = digits

    # Compute a safe pip_size for spread conversion
    # Priority:
    # 1) explicit broker pip_size if valid
    # 2) derive from digits/point (MT5-style)
    # 3) derive from pip_position
    pip_size = float(broker_data.get('pip_size', 0.0) or 0.0)
    if pip_size <= 0.0:
        if digits in (3, 5):
            pip_size = point * 10.0
        else:
            pip_size = point
    if pip_size <= 0.0:
        pip_size = float(10 ** (-pip_position)) if pip_position >= 0 else 0.0001

    # Tick fields (sanitize zeros)
    tick_size = float(broker_data.get('tick_size', 0.0) or 0.0)
    if tick_size <= 0.0:
        tick_size = point

    tick_value = float(broker_data.get('tick_value', 0.0) or 0.0)
    contract_size = float(broker_data.get('contract_size', 0.0) or 0.0)

    # Pip value (if broker provides it, trust it; otherwise infer from tick fields)
    pip_value = float(broker_data.get('pip_value', 0.0) or 0.0)
    if pip_value <= 0.0:
        if tick_size > 0.0 and tick_value > 0.0:
            pip_value = (pip_size / tick_size) * tick_value
        else:
            # Safe fallback used historically in the PM
            pip_value = 10.0

    # Spread conversion to pips (guard against division by zero)
    spread_raw = float(broker_data.get('spread', 2.0) or 2.0)
    if pip_size > 0.0:
        spread_avg = spread_raw * point / pip_size
    else:
        spread_avg = 2.0

    return InstrumentSpec(
        symbol=symbol,
        pip_position=pip_position,
        pip_value=pip_value,
        spread_avg=spread_avg,
        min_lot=float(broker_data.get('volume_min', 0.01) or 0.01),
        max_lot=float(broker_data.get('volume_max', 100.0) or 100.0),
        volume_step=float(broker_data.get('volume_step', 0.01) or 0.01),
        commission_per_lot=float(broker_data.get('commission_per_lot', 7.0) or 7.0),
        swap_long=float(broker_data.get('swap_long', 0.0) or 0.0),
        swap_short=float(broker_data.get('swap_short', 0.0) or 0.0),
        # Broker-real fields (used for MT5-parity sizing/P&L)
        digits=digits,
        point=point,
        tick_size=tick_size,
        tick_value=tick_value,
        contract_size=contract_size,
        stops_level=int(broker_data.get('stops_level', broker_data.get('trade_stops_level', 0)) or 0),
    )


def get_instrument_spec(symbol: str) -> InstrumentSpec:
    """
    Get instrument specification for a symbol.
    
    Priority:
    1. Config-provided instrument_specs overrides
    2. Broker specs from broker_specs.json (if available)
    3. Default hardcoded specs
    4. Generic fallback spec
    
    Args:
        symbol: Symbol name
        
    Returns:
        InstrumentSpec for the symbol
    """
    if not _CONFIG_INSTRUMENT_SPECS:
        _maybe_load_config_specs()
    symbol_key = _normalize_symbol(symbol)

    # 1) Config-provided overrides
    if symbol_key in _CONFIG_INSTRUMENT_SPECS:
        return _apply_spec_defaults(_clone_spec(_CONFIG_INSTRUMENT_SPECS[symbol_key], symbol_override=symbol))
    
    # 2) Broker specs
    broker_specs = load_broker_specs()
    if symbol in broker_specs:
        return _create_spec_from_broker_data(symbol, broker_specs[symbol])
    
    if symbol_key in broker_specs:
        return _create_spec_from_broker_data(symbol, broker_specs[symbol_key])
    
    # 3) Hardcoded defaults
    if symbol in INSTRUMENT_SPECS:
        return _apply_spec_defaults(_clone_spec(INSTRUMENT_SPECS[symbol]))
    
    if symbol_key in INSTRUMENT_SPECS:
        return _apply_spec_defaults(_clone_spec(INSTRUMENT_SPECS[symbol_key], symbol_override=symbol))
    
    # 4) Generic fallback (warn only once per symbol to avoid spam)
    if symbol not in _MISSING_SPEC_WARNED:
        logger.warning(
            f"No spec found for {symbol}, using default. "
            f"Add to config.json instrument_specs or provide broker specs."
        )
        _MISSING_SPEC_WARNED.add(symbol)
    return _apply_spec_defaults(InstrumentSpec(symbol, 4, 10.0, 2.0))


# =============================================================================
# DATA LOADING
# =============================================================================

class DataLoader:
    """
    Loads and resamples OHLCV data.
    
    Supports:
    - Loading from CSV files
    - Resampling M5 data to higher timeframes
    - Data validation and cleaning
    """
    
    # Timeframe resampling rules
    RESAMPLE_MAP = {
        'M1': '1min',
        'M5': '5min',
        'M15': '15min',
        'M30': '30min',
        'H1': '1h',
        'H4': '4h',
        'D1': '1D',
        'W1': '1W',
    }
    
    # Minimum bars required for each timeframe (increased for 5-year data)
    MIN_BARS = {
        'M5': 50000,   # ~6 months of M5 data minimum
        'M15': 20000,  # ~6 months of M15 data minimum
        'M30': 10000,  # ~6 months of M30 data minimum
        'H1': 5000,    # ~6 months of H1 data minimum
        'H4': 2000,    # ~6 months of H4 data minimum  
        'D1': 500,     # ~2 years of D1 data minimum
    }
    
    def __init__(self, data_dir: Path, cache_resampled: bool = True, cache_dir: Optional[Path] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing CSV data files
        """
        self.data_dir = Path(data_dir)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._resample_cache: Dict[str, pd.DataFrame] = {}
        self._resample_meta: Dict[str, Dict[str, Any]] = {}
        self._source_meta: Dict[str, Dict[str, Any]] = {}

        self.cache_resampled = bool(cache_resampled)
        self.cache_dir = Path(cache_dir) if cache_dir else (self.data_dir / ".cache")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If cache dir can't be created, fall back to in-memory only
            self.cache_resampled = False
    
    def load_symbol(self, symbol: str, base_timeframe: str = 'M5') -> Optional[pd.DataFrame]:
        """
        Load data for a symbol from CSV.
        
        Args:
            symbol: Symbol name
            base_timeframe: Base timeframe of the data file
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        # Check cache
        cache_key = f"{symbol}_{base_timeframe}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Find data file
        patterns = [
            f"{symbol}_{base_timeframe}*.csv",
            f"{symbol.upper()}_{base_timeframe}*.csv",
            f"{symbol.lower()}_{base_timeframe}*.csv",
            f"{symbol}*.csv",
        ]
        
        data_file = None
        for pattern in patterns:
            matches = list(self.data_dir.glob(pattern))
            if matches:
                data_file = matches[0]
                break
        
        if not data_file:
            logger.warning(f"No data file found for {symbol} in {self.data_dir}")
            return None
        
        try:
            df = pd.read_csv(data_file)
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Parse and set datetime index
            df = self._parse_datetime_index(df)
            
            # Validate data
            if not self._validate_data(df, symbol):
                return None
            
            # Cache and return
            self._cache[cache_key] = df

            # Track source metadata for resample cache validation
            try:
                stat = data_file.stat()
                self._source_meta[cache_key] = {
                    "path": str(data_file),
                    "mtime": stat.st_mtime,
                    "rows": len(df),
                    "last_index": df.index[-1].isoformat() if len(df) else None,
                }
            except Exception:
                self._source_meta[cache_key] = {
                    "path": str(data_file),
                    "mtime": None,
                    "rows": len(df),
                    "last_index": df.index[-1].isoformat() if len(df) else None,
                }
            logger.info(f"Loaded {symbol}: {len(df)} bars from {data_file.name}")
            
            return df.copy()
            
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
            return None
    
    def resample(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample data to a higher timeframe.
        
        Args:
            df: Source DataFrame (must have datetime index)
            target_timeframe: Target timeframe (e.g., 'H1', 'D1')
            
        Returns:
            Resampled DataFrame
        """
        if target_timeframe not in self.RESAMPLE_MAP:
            raise ValueError(f"Unknown timeframe: {target_timeframe}")
        
        resample_rule = self.RESAMPLE_MAP[target_timeframe]
        
        # Resample OHLCV
        resampled = df.resample(resample_rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # Handle Spread if present
        if 'Spread' in df.columns:
            spread_resampled = df['Spread'].resample(resample_rule).mean()
            resampled['Spread'] = spread_resampled
        
        return resampled

    def _resample_cache_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"

    def _resample_cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.cache_dir / f"{safe_symbol}_{timeframe}.pkl"

    def _is_resample_cache_valid(self, meta: Dict[str, Any], source_meta: Dict[str, Any]) -> bool:
        if not meta or not source_meta:
            return False
        return (
            meta.get("source_mtime") == source_meta.get("mtime") and
            meta.get("source_rows") == source_meta.get("rows") and
            meta.get("source_last_index") == source_meta.get("last_index")
        )

    def _load_resample_cache(self, symbol: str, timeframe: str, source_meta: Dict[str, Any]) -> Optional[pd.DataFrame]:
        cache_key = self._resample_cache_key(symbol, timeframe)

        # In-memory cache
        meta = self._resample_meta.get(cache_key)
        if cache_key in self._resample_cache and self._is_resample_cache_valid(meta, source_meta):
            return self._resample_cache[cache_key].copy()

        if not self.cache_resampled:
            return None

        # Disk cache
        path = self._resample_cache_path(symbol, timeframe)
        if not path.exists():
            return None

        try:
            payload = pd.read_pickle(path)
            if not isinstance(payload, dict):
                return None
            meta = payload.get("meta", {})
            if not self._is_resample_cache_valid(meta, source_meta):
                return None
            df = payload.get("data")
            if isinstance(df, pd.DataFrame):
                self._resample_cache[cache_key] = df.copy()
                self._resample_meta[cache_key] = meta
                return df.copy()
        except Exception:
            return None

        return None

    def _save_resample_cache(self, symbol: str, timeframe: str, source_meta: Dict[str, Any],
                             df: pd.DataFrame) -> None:
        cache_key = self._resample_cache_key(symbol, timeframe)
        meta = {
            "source_mtime": source_meta.get("mtime"),
            "source_rows": source_meta.get("rows"),
            "source_last_index": source_meta.get("last_index"),
        }
        df_copy = df.copy()
        self._resample_cache[cache_key] = df_copy
        self._resample_meta[cache_key] = meta

        if not self.cache_resampled:
            return
        try:
            path = self._resample_cache_path(symbol, timeframe)
            payload = {"meta": meta, "data": df_copy}
            pd.to_pickle(payload, path)
        except Exception:
            pass
    
    def get_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get data for a symbol at a specific timeframe.
        
        Loads base M5 data and resamples if needed.
        
        Args:
            symbol: Symbol name
            timeframe: Target timeframe
            
        Returns:
            DataFrame with OHLCV data or None
        """
        # Load base data
        base_df = self.load_symbol(symbol, 'M5')
        if base_df is None:
            return None
        
        # Resample if needed
        if timeframe == 'M5':
            return base_df.copy()
        
        # Try cache (memory/disk) keyed to the M5 source file state
        base_meta = self._source_meta.get(f"{symbol}_M5", {})
        cached = self._load_resample_cache(symbol, timeframe, base_meta)
        if cached is not None:
            resampled = cached
        else:
            resampled = self.resample(base_df, timeframe)
            self._save_resample_cache(symbol, timeframe, base_meta, resampled)
        
        # Check minimum bars
        min_bars = self.MIN_BARS.get(timeframe, 100)
        if len(resampled) < min_bars:
            logger.warning(f"{symbol} {timeframe}: Only {len(resampled)} bars, need {min_bars}")
            return None
        
        return resampled.copy()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to Open, High, Low, Close, Volume."""
        column_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'tick_volume': 'Volume',
            'tickvolume': 'Volume',
            'spread': 'Spread',
            'real_volume': 'RealVolume',
        }
        
        df.columns = df.columns.str.lower()
        df = df.rename(columns=column_map)
        
        return df
    
    def _parse_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetime and set as index."""
        time_col = None
        for col in ['time', 'datetime', 'date', 'timestamp']:
            if col in df.columns.str.lower():
                time_col = [c for c in df.columns if c.lower() == col][0]
                break
        
        if time_col is None:
            raise ValueError("No datetime column found")
        
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        df = df.sort_index()
        
        return df
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate data quality."""
        required_cols = ['Open', 'High', 'Low', 'Close']
        
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"{symbol}: Missing column {col}")
                return False

        # Coerce OHLC (and optional numeric columns) to numeric to avoid mixed-dtype issues
        # when reading from CSVs. This is especially important for ATR and other features.
        for ncol in ['Open', 'High', 'Low', 'Close', 'Volume', 'Spread']:
            if ncol in df.columns:
                df[ncol] = pd.to_numeric(df[ncol], errors='coerce')

        # Drop rows with missing critical OHLC values (use drop instead of inplace to avoid modifying cached data)
        before = len(df)
        mask = df[['Open', 'High', 'Low', 'Close']].notna().all(axis=1)
        rows_to_drop = (~mask).sum()
        if rows_to_drop > 0:
            # Get indices to drop and drop them
            indices_to_drop = df.index[~mask]
            df.drop(indices_to_drop, inplace=True)
            logger.warning(f"{symbol}: Dropped {rows_to_drop} rows with non-numeric/missing OHLC")

        
        # Check for valid OHLC relationships
        invalid_bars = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        ).sum()
        
        if invalid_bars > 0:
            logger.warning(f"{symbol}: {invalid_bars} bars with invalid OHLC")
        
        # Check for Volume
        if 'Volume' not in df.columns:
            df['Volume'] = 0
            logger.warning(f"{symbol}: No volume data, using 0")
        
        return True
    
    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
        self._resample_cache.clear()
        self._resample_meta.clear()
        self._source_meta.clear()


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

class FeatureComputer:
    """
    Computes technical indicators and features.
    
    Adds common indicators used by strategies:
    - Moving averages (SMA, EMA)
    - ATR (Average True Range)
    - RSI (Relative Strength Index)
    - Bollinger Bands
    - MACD
    - Stochastic
    - And more...
    
    Supports lazy feature computation for efficiency when only specific
    features are needed.
    """

    # In-memory feature cache (bounded to avoid memory bloat)
    _FEATURE_CACHE: Dict[Tuple[Any, ...], pd.DataFrame] = {}
    _FEATURE_CACHE_ORDER: List[Tuple[Any, ...]] = []
    _FEATURE_CACHE_MAX = 6

    @classmethod
    def clear_cache(cls) -> None:
        """Clear in-memory feature cache (memory hygiene between symbols)."""
        cls._FEATURE_CACHE.clear()
        cls._FEATURE_CACHE_ORDER.clear()

    @staticmethod
    def _make_cache_key(df: pd.DataFrame, symbol: str, timeframe: str,
                        regime_params_file: str, tag: str = "all") -> Tuple[Any, ...]:
        try:
            first_idx = df.index[0]
            last_idx = df.index[-1]
            first_key = first_idx.isoformat() if hasattr(first_idx, "isoformat") else str(first_idx)
            last_key = last_idx.isoformat() if hasattr(last_idx, "isoformat") else str(last_idx)
        except Exception:
            first_key = "na"
            last_key = "na"
        return (tag, symbol, timeframe, regime_params_file, len(df), first_key, last_key)

    @classmethod
    def _cache_get(cls, key: Tuple[Any, ...]) -> Optional[pd.DataFrame]:
        if key in cls._FEATURE_CACHE:
            return cls._FEATURE_CACHE[key].copy()
        return None

    @classmethod
    def _cache_put(cls, key: Tuple[Any, ...], value: pd.DataFrame) -> None:
        cls._FEATURE_CACHE[key] = value.copy()
        cls._FEATURE_CACHE_ORDER.append(key)
        if len(cls._FEATURE_CACHE_ORDER) > cls._FEATURE_CACHE_MAX:
            oldest = cls._FEATURE_CACHE_ORDER.pop(0)
            cls._FEATURE_CACHE.pop(oldest, None)
    
    # Feature dependencies - maps feature names to their computation requirements
    _FEATURE_DEPS = {
        'ATR_7': set(),
        'ATR_10': set(),
        'ATR_14': set(),
        'ATR_20': set(),
        'SMA_5': set(), 'SMA_8': set(), 'SMA_10': set(), 'SMA_13': set(),
        'SMA_20': set(), 'SMA_21': set(), 'SMA_30': set(), 'SMA_50': set(),
        'SMA_100': set(), 'SMA_200': set(),
        'EMA_5': set(), 'EMA_8': set(), 'EMA_10': set(), 'EMA_13': set(),
        'EMA_20': set(), 'EMA_21': set(), 'EMA_30': set(), 'EMA_50': set(),
        'EMA_100': set(), 'EMA_200': set(),
        'RSI_7': set(), 'RSI_14': set(), 'RSI_21': set(),
        'BB_MID_20': set(), 'BB_UPPER_20': set(), 'BB_LOWER_20': set(),
        'MACD': set(), 'MACD_SIGNAL': set(), 'MACD_HIST': set(),
        'STOCH_K': set(), 'STOCH_D': set(),
        'ADX': {'ATR_14'}, 'PLUS_DI': {'ATR_14'}, 'MINUS_DI': {'ATR_14'},
        'CCI': set(),
        'WILLR': set(),
        'DONCHIAN_HIGH_20': set(), 'DONCHIAN_LOW_20': set(),
        'DONCHIAN_HIGH_50': set(), 'DONCHIAN_LOW_50': set(),
        'KC_MID': {'ATR_20'}, 'KC_UPPER': {'ATR_20'}, 'KC_LOWER': {'ATR_20'},
        'HULL_MA': set(),
        'CHANGE': set(), 'CHANGE_PCT': set(),
        'VOLATILITY': set(),
    }
    
    @staticmethod
    def compute_required(df: pd.DataFrame, required_features: set,
                        symbol: str = "", timeframe: str = "H1") -> pd.DataFrame:
        """
        Compute only the required features (lazy computation).
        
        This is more efficient than compute_all() when only specific
        features are needed, such as when evaluating a single strategy.
        
        Args:
            df: DataFrame with OHLCV data
            required_features: Set of feature column names needed
            symbol: Symbol name (for regime parameter lookup if needed)
            timeframe: Timeframe string
            
        Returns:
            DataFrame with only the required features computed
        """
        features = df.copy()
        computed = set(features.columns)
        
        # Expand dependencies
        to_compute = set(required_features)
        for feat in list(to_compute):
            deps = FeatureComputer._FEATURE_DEPS.get(feat, set())
            to_compute.update(deps)
        
        # Compute ATRs if needed
        for period in [7, 10, 14, 20]:
            col = f'ATR_{period}'
            if col in to_compute and col not in computed:
                features[col] = FeatureComputer.atr(features, period)
                computed.add(col)
        
        # Compute SMAs if needed
        for period in [5, 8, 10, 13, 20, 21, 30, 50, 100, 200]:
            col = f'SMA_{period}'
            if col in to_compute and col not in computed:
                features[col] = features['Close'].rolling(period).mean()
                computed.add(col)
        
        # Compute EMAs if needed
        for period in [5, 8, 10, 13, 20, 21, 30, 50, 100, 200]:
            col = f'EMA_{period}'
            if col in to_compute and col not in computed:
                features[col] = features['Close'].ewm(span=period, adjust=False).mean()
                computed.add(col)
        
        # RSI
        for period in [7, 14, 21]:
            col = f'RSI_{period}'
            if col in to_compute and col not in computed:
                features[col] = FeatureComputer.rsi(features['Close'], period)
                computed.add(col)
        
        # Bollinger Bands
        bb_cols = {'BB_MID_20', 'BB_UPPER_20', 'BB_LOWER_20'}
        if bb_cols & to_compute:
            bb_mid, bb_upper, bb_lower = FeatureComputer.bollinger_bands(features['Close'], 20, 2.0)
            if 'BB_MID_20' in to_compute:
                features['BB_MID_20'] = bb_mid
            if 'BB_UPPER_20' in to_compute:
                features['BB_UPPER_20'] = bb_upper
            if 'BB_LOWER_20' in to_compute:
                features['BB_LOWER_20'] = bb_lower
            computed.update(bb_cols)
        
        # MACD
        macd_cols = {'MACD', 'MACD_SIGNAL', 'MACD_HIST'}
        if macd_cols & to_compute:
            macd, signal, hist = FeatureComputer.macd(features['Close'])
            features['MACD'] = macd
            features['MACD_SIGNAL'] = signal
            features['MACD_HIST'] = hist
            computed.update(macd_cols)
        
        # Stochastic
        stoch_cols = {'STOCH_K', 'STOCH_D'}
        if stoch_cols & to_compute:
            stoch_k, stoch_d = FeatureComputer.stochastic(features)
            features['STOCH_K'] = stoch_k
            features['STOCH_D'] = stoch_d
            computed.update(stoch_cols)
        
        # ADX family (uses cached ATR)
        adx_cols = {'ADX', 'PLUS_DI', 'MINUS_DI'}
        if adx_cols & to_compute:
            atr_14 = features.get('ATR_14', FeatureComputer.atr(features, 14))
            if 'ADX' in to_compute:
                features['ADX'] = FeatureComputer.adx(features, 14, atr_cache=atr_14)
            if 'PLUS_DI' in to_compute:
                features['PLUS_DI'] = FeatureComputer.plus_di(features, 14, atr_cache=atr_14)
            if 'MINUS_DI' in to_compute:
                features['MINUS_DI'] = FeatureComputer.minus_di(features, 14, atr_cache=atr_14)
            computed.update(adx_cols)
        
        # CCI
        if 'CCI' in to_compute and 'CCI' not in computed:
            features['CCI'] = FeatureComputer.cci(features, 20)
            computed.add('CCI')
        
        # Williams %R
        if 'WILLR' in to_compute and 'WILLR' not in computed:
            features['WILLR'] = FeatureComputer.williams_r(features, 14)
            computed.add('WILLR')
        
        # Donchian
        for period in [20, 50]:
            high_col = f'DONCHIAN_HIGH_{period}'
            low_col = f'DONCHIAN_LOW_{period}'
            if high_col in to_compute and high_col not in computed:
                features[high_col] = features['High'].rolling(period).max()
                computed.add(high_col)
            if low_col in to_compute and low_col not in computed:
                features[low_col] = features['Low'].rolling(period).min()
                computed.add(low_col)
        
        # Hull MA
        if 'HULL_MA' in to_compute and 'HULL_MA' not in computed:
            features['HULL_MA'] = FeatureComputer.hull_ma(features['Close'], 20)
            computed.add('HULL_MA')
        
        # Simple derived features
        if 'CHANGE' in to_compute and 'CHANGE' not in computed:
            features['CHANGE'] = features['Close'].diff()
        if 'CHANGE_PCT' in to_compute and 'CHANGE_PCT' not in computed:
            features['CHANGE_PCT'] = features['Close'].pct_change() * 100
        if 'VOLATILITY' in to_compute and 'VOLATILITY' not in computed:
            features['VOLATILITY'] = features['Close'].rolling(20).std()
        
        return features
    
    @staticmethod
    def compute_all(df: pd.DataFrame, symbol: str = "", timeframe: str = "H1",
                    regime_params_file: str = "regime_params.json") -> pd.DataFrame:
        """
        Compute all standard features including regime detection.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name (for regime parameter lookup)
            timeframe: Timeframe string (for regime parameter lookup)
            regime_params_file: Path to regime parameters JSON
            
        Returns:
            DataFrame with added feature columns including regime columns:
            - TREND_SCORE, RANGE_SCORE, BREAKOUT_SCORE, CHOP_SCORE
            - REGIME_RAW, REGIME, REGIME_STRENGTH, REGIME_GAP
            - REGIME_LIVE, REGIME_STRENGTH_LIVE
        """
        # In-memory cache lookup
        cache_key = FeatureComputer._make_cache_key(df, symbol, timeframe, regime_params_file, tag="all")
        cached = FeatureComputer._cache_get(cache_key)
        if cached is not None:
            return cached

        features = df.copy()
        
        # ATR (multiple periods)
        for period in [7, 10, 14, 20]:
            features[f'ATR_{period}'] = FeatureComputer.atr(features, period)
        
        # Moving Averages
        for period in [5, 8, 10, 13, 20, 21, 30, 50, 100, 200]:
            features[f'SMA_{period}'] = features['Close'].rolling(period).mean()
            features[f'EMA_{period}'] = features['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        for period in [7, 14, 21]:
            features[f'RSI_{period}'] = FeatureComputer.rsi(features['Close'], period)
        
        # Bollinger Bands
        for period in [20]:
            bb_mid, bb_upper, bb_lower = FeatureComputer.bollinger_bands(
                features['Close'], period, 2.0
            )
            features[f'BB_MID_{period}'] = bb_mid
            features[f'BB_UPPER_{period}'] = bb_upper
            features[f'BB_LOWER_{period}'] = bb_lower
        
        # MACD
        macd, signal, hist = FeatureComputer.macd(features['Close'])
        features['MACD'] = macd
        features['MACD_SIGNAL'] = signal
        features['MACD_HIST'] = hist
        
        # Stochastic
        stoch_k, stoch_d = FeatureComputer.stochastic(features)
        features['STOCH_K'] = stoch_k
        features['STOCH_D'] = stoch_d
        
        # ADX, +DI, -DI (optimized: compute ATR once, share across all three)
        atr_14_cached = features['ATR_14']  # Already computed above
        features['ADX'] = FeatureComputer.adx(features, 14, atr_cache=atr_14_cached)
        features['PLUS_DI'] = FeatureComputer.plus_di(features, 14, atr_cache=atr_14_cached)
        features['MINUS_DI'] = FeatureComputer.minus_di(features, 14, atr_cache=atr_14_cached)
        
        # CCI
        features['CCI'] = FeatureComputer.cci(features, 20)
        
        # Williams %R
        features['WILLR'] = FeatureComputer.williams_r(features, 14)
        
        # Donchian Channels
        for period in [20, 50]:
            features[f'DONCHIAN_HIGH_{period}'] = features['High'].rolling(period).max()
            features[f'DONCHIAN_LOW_{period}'] = features['Low'].rolling(period).min()
        
        # Keltner Channels (use cached ATR_20 if available)
        atr_20_cached = features.get('ATR_20')
        kc_mid, kc_upper, kc_lower = FeatureComputer.keltner_channels(features, 20, 2.0)
        features['KC_MID'] = kc_mid
        features['KC_UPPER'] = kc_upper
        features['KC_LOWER'] = kc_lower
        
        # Hull MA
        features['HULL_MA'] = FeatureComputer.hull_ma(features['Close'], 20)
        
        # Price changes
        features['CHANGE'] = features['Close'].diff()
        features['CHANGE_PCT'] = features['Close'].pct_change() * 100
        
        # Volatility
        features['VOLATILITY'] = features['Close'].rolling(20).std()
        
        # =====================================================================
        # REGIME DETECTION
        # Compute regime features using symbol/timeframe-specific parameters
        # =====================================================================
        try:
            from pm_regime import (
                MarketRegimeDetector, 
                load_regime_params,
                RegimeType
            )
            
            # Load symbol/timeframe-specific regime parameters
            params = load_regime_params(symbol, timeframe, regime_params_file)
            
            # Create detector and compute regime scores
            detector = MarketRegimeDetector(params)
            regime_df = detector.compute_regime_scores(features)
            
            # Merge all regime columns into features
            regime_columns = [
                'TREND_SCORE', 'RANGE_SCORE', 'BREAKOUT_SCORE', 'CHOP_SCORE',
                'REGIME_RAW', 'REGIME', 'REGIME_STRENGTH', 'REGIME_GAP',
                'REGIME_LIVE', 'REGIME_STRENGTH_LIVE'
            ]
            
            for col in regime_columns:
                if col in regime_df.columns:
                    features[col] = regime_df[col]
            
            logger.debug(f"Added regime features for {symbol} {timeframe}")
                    
        except ImportError:
            logger.warning("pm_regime module not available, skipping regime features")
        except Exception as e:
            logger.warning(f"Failed to compute regime features for {symbol} {timeframe}: {e}")
        
        # Cache and return
        FeatureComputer._cache_put(cache_key, features)
        return features
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range.

        Notes:
            We coerce OHLC columns to numeric to avoid dtype issues when data
            is loaded from CSVs (e.g., strings) or mixed dtypes occur.
        """
        if df is None or len(df) == 0:
            return pd.Series(dtype=float)

        high = pd.to_numeric(df['High'], errors='coerce')
        low = pd.to_numeric(df['Low'], errors='coerce')
        close = pd.to_numeric(df['Close'], errors='coerce')

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=True)
        atr = tr.rolling(window=period, min_periods=period).mean()

        return atr
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index using Wilder's smoothing method."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's exponential smoothing (alpha = 1/period)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)  # Add epsilon to prevent division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, 
                        std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        mid = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = mid + (std_dev * std)
        lower = mid - (std_dev * std)
        
        return mid, upper, lower
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, 
                   d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        
        stoch_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14, atr_cache: pd.Series = None) -> pd.Series:
        """Calculate Average Directional Index.
        
        Optimized to accept pre-computed ATR for efficiency when computing
        multiple directional indicators.
        
        Args:
            df: DataFrame with OHLCV
            period: ADX period
            atr_cache: Pre-computed ATR series (optional, computed if None)
        """
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff().abs() * -1
        
        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
        
        # Use cached ATR if provided, otherwise compute
        if atr_cache is not None:
            atr = atr_cache
        else:
            atr = FeatureComputer.atr(df, period)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def plus_di(df: pd.DataFrame, period: int = 14, atr_cache: pd.Series = None) -> pd.Series:
        """Calculate +DI.
        
        Optimized to accept pre-computed ATR.
        """
        plus_dm = df['High'].diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        atr = atr_cache if atr_cache is not None else FeatureComputer.atr(df, period)
        return 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
    
    @staticmethod
    def minus_di(df: pd.DataFrame, period: int = 14, atr_cache: pd.Series = None) -> pd.Series:
        """Calculate -DI.
        
        Optimized to accept pre-computed ATR.
        """
        minus_dm = df['Low'].diff().abs()
        atr = atr_cache if atr_cache is not None else FeatureComputer.atr(df, period)
        return 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
    
    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index.
        
        Optimized with vectorized MAD calculation using raw=True for 3-5x speedup.
        """
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma = tp.rolling(window=period).mean()
        # Vectorized MAD: use raw=True to pass numpy array directly (faster)
        mad = tp.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        # Protect against division by zero
        cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
        
        return cci
    
    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        
        wr = -100 * (high_max - df['Close']) / (high_max - low_min)
        
        return wr
    
    @staticmethod
    def keltner_channels(df: pd.DataFrame, period: int = 20, 
                         mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels."""
        mid = df['Close'].ewm(span=period, adjust=False).mean()
        atr = FeatureComputer.atr(df, period)
        upper = mid + (mult * atr)
        lower = mid - (mult * atr)
        
        return mid, upper, lower
    
    @staticmethod
    def hull_ma(series: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Hull Moving Average using proper Weighted Moving Averages.
        
        Optimized with pre-computed weights array for 2-3x speedup.
        """
        half_period = max(1, int(period / 2))
        sqrt_period = max(1, int(np.sqrt(period)))
        
        # Pre-compute weights outside the rolling function for efficiency
        def weighted_ma(s: pd.Series, p: int, weights_cache: dict = {}) -> pd.Series:
            """Calculate Weighted Moving Average with cached weights."""
            if p not in weights_cache:
                weights_cache[p] = np.arange(1, p + 1, dtype=np.float64)
                weights_cache[f'{p}_sum'] = weights_cache[p].sum()
            weights = weights_cache[p]
            weight_sum = weights_cache[f'{p}_sum']
            return s.rolling(p).apply(lambda x: np.dot(x, weights) / weight_sum, raw=True)
        
        wma_half = weighted_ma(series, half_period)
        wma_full = weighted_ma(series, period)
        
        raw_hma = 2 * wma_half - wma_full
        hma = weighted_ma(raw_hma, sqrt_period)
        
        return hma


# =============================================================================
# DATA SPLITTING
# =============================================================================

class DataSplitter:
    """
    Splits data into training and validation sets.
    
    Implements 80/30 split with 10% overlap:
    - Training: First 80% of data
    - Validation: Last 30% of data (overlaps with last 10% of training)
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize DataSplitter.
        
        Args:
            config: Pipeline configuration with split ratios
        """
        self.config = config
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets.
        
        Args:
            df: Full dataset
            
        Returns:
            Tuple of (train_df, val_df)
        """
        n = len(df)
        
        # Training: 0 to 80%
        train_end = int(n * (self.config.train_pct / 100.0))
        
        # Validation: 70% to 100% (10% overlap)
        val_start = int(n * ((self.config.train_pct - self.config.overlap_pct) / 100.0))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:].copy()
        
        logger.debug(f"Split: Total={n}, Train={len(train_df)} (0:{train_end}), "
                    f"Val={len(val_df)} ({val_start}:{n})")
        
        return train_df, val_df
    
    def get_split_indices(self, n: int) -> Dict[str, Tuple[int, int]]:
        """
        Get split indices without actually splitting.
        
        Args:
            n: Total number of rows
            
        Returns:
            Dict with 'train' and 'val' index tuples
        """
        train_end = int(n * (self.config.train_pct / 100.0))
        val_start = int(n * ((self.config.train_pct - self.config.overlap_pct) / 100.0))
        
        return {
            'train': (0, train_end),
            'val': (val_start, n)
        }


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class Backtester:
    """
    Full backtesting engine with complete performance metrics.
    
    Simulates trading with:
    - Tick-based P&L calculation (MT5 parity)
    - Proper entry/exit price calculation (spread)
    - Slippage applied only on STOP exits (not entries or TP)
    - Position sizing based on risk using tick math
    - Volume rounding using broker volume_step
    - Commission deduction
    - Complete performance metrics calculation
    
    Optimizations:
    - Pre-extracts OHLC to NumPy arrays
    - Caches spread price
    - Passes bar index to strategy.calculate_stops()
    - Numba JIT-compiled main loop (3-10x speedup when available)
    
    The Numba optimization preserves EXACT trade semantics:
    - SL checked before TP (same-bar hit order preserved)
    - Same floating-point precision (float64, no fastmath)
    - Same exit pricing rules (bid for long, ask for short)
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize Backtester.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
    
    def run(self, 
            features: pd.DataFrame,
            signals: pd.Series,
            symbol: str,
            strategy: Any,
            spec: Optional[InstrumentSpec] = None,
            timeframe: str = "") -> Dict[str, Any]:
        """
        Run backtest on a dataset with signals.
        
        Uses Numba JIT-compiled loop when available for 3-10x speedup.
        Falls back to pure Python if Numba is not installed.
        
        IMPORTANT: Position sizing uses LIVE EQUITY (compounding preserved).
        Only entry/SL/TP prices are precomputed; sizing happens inside the loop.
        
        Args:
            features: DataFrame with OHLCV and indicators
            signals: Series of signals (-1, 0, 1)
            symbol: Symbol being traded
            strategy: Strategy instance (for stop calculation)
            spec: Optional InstrumentSpec (uses default if None)
            
        Returns:
            Dict with complete backtest results and metrics
        """
        if spec is None:
            spec = get_instrument_spec(symbol)

        # ----------------------------------------------------------------
        # Signal contract validation (prevents silent misalignment)
        # ----------------------------------------------------------------
        if not isinstance(signals, pd.Series):
            signals = pd.Series(signals, index=features.index)

        # Align index strictly; if misaligned, reindex and fill flat.
        if not signals.index.equals(features.index):
            signals = signals.reindex(features.index)

        if signals.isna().any():
            signals = signals.fillna(0)

        # Ensure numeric int signals in {-1,0,1}
        signals = pd.to_numeric(signals, errors='coerce').fillna(0)
        signals = signals.clip(-1, 1).astype(int)
        
        # ----------------------------------------------------------------
        # Pre-extract arrays for performance (avoid repeated .iloc calls)
        # ----------------------------------------------------------------
        open_arr = features['Open'].to_numpy().astype(np.float64)
        high_arr = features['High'].to_numpy().astype(np.float64)
        low_arr = features['Low'].to_numpy().astype(np.float64)
        close_arr = features['Close'].to_numpy().astype(np.float64)
        sig_arr = signals.to_numpy().astype(np.int32)
        
        # Cache spread-related constants
        half_spread = spec.get_half_spread_price() if self.config.use_spread else 0.0
        slippage_price = spec.pips_to_price(self.config.slippage_pips) if self.config.use_slippage else 0.0
        
        n_bars = len(features)
        
        # ----------------------------------------------------------------
        # PRE-COMPUTE ENTRY/STOP PRICES ONLY (NOT position sizes)
        # Position sizing must be done inside the loop to use live equity
        # ----------------------------------------------------------------
        sl_prices = np.full(n_bars, np.nan, dtype=np.float64)
        tp_prices = np.full(n_bars, np.nan, dtype=np.float64)
        entry_prices = np.full(n_bars, np.nan, dtype=np.float64)
        
        for i in range(1, n_bars):
            signal = sig_arr[i - 1]  # Signal from previous bar
            
            if signal != 0:
                direction = int(signal)
                is_long = direction == 1
                signal_bar_index = i - 1
                
                # Calculate stops using the SIGNAL bar index (i-1)
                try:
                    sl_pips, tp_pips = strategy.calculate_stops(
                        features, direction, symbol, spec=spec, bar_index=signal_bar_index
                    )
                except TypeError:
                    # Fallback for strategies without bar_index support
                    try:
                        sl_pips, tp_pips = strategy.calculate_stops(
                            features.iloc[:signal_bar_index + 1].copy(), direction, symbol
                        )
                    except Exception:
                        continue
                except Exception:
                    continue
                
                # Skip if stops are invalid (NaN from indicator warmup)
                if np.isnan(sl_pips) or np.isnan(tp_pips) or sl_pips <= 0 or tp_pips <= 0:
                    continue
                
                # Entry price: Open of ENTRY bar (i) + spread
                open_price = open_arr[i]
                if self.config.use_spread:
                    entry_price = open_price + half_spread if is_long else open_price - half_spread
                else:
                    entry_price = open_price
                
                # Set SL/TP prices
                if direction == 1:
                    stop_loss = entry_price - spec.pips_to_price(sl_pips)
                    take_profit = entry_price + spec.pips_to_price(tp_pips)
                else:
                    stop_loss = entry_price + spec.pips_to_price(sl_pips)
                    take_profit = entry_price - spec.pips_to_price(tp_pips)
                
                # Store pre-computed prices (NOT sizes - those use live equity)
                entry_prices[i] = entry_price
                sl_prices[i] = stop_loss
                tp_prices[i] = take_profit
        
        # ----------------------------------------------------------------
        # RUN BACKTEST LOOP (Numba JIT or Pure Python)
        # ----------------------------------------------------------------
        if NUMBA_AVAILABLE:
            # Use Numba JIT-compiled kernel with live-equity sizing
            (
                trade_signal_bars, trade_entry_bars, trade_exit_bars,
                trade_directions, trade_entry_prices, trade_exit_prices,
                trade_sl_prices, trade_tp_prices, trade_sizes,
                trade_pnl_dollars, trade_pnl_pips, trade_exit_reasons,
                trade_risk_amounts, final_equity, max_drawdown
            ) = _backtest_loop_numba(
                open_arr, high_arr, low_arr, close_arr, sig_arr,
                sl_prices, tp_prices, entry_prices,
                self.config.initial_capital,
                self.config.position_size_pct,
                half_spread, slippage_price,
                self.config.use_spread, self.config.use_slippage,
                self.config.use_commission, spec.commission_per_lot,
                spec.tick_size, spec.tick_value, spec.pip_size, spec.pip_value,
                spec.min_lot, spec.max_lot, spec.volume_step
            )
            
            # Convert JIT results to trade dicts
            exit_reason_map = {
                EXIT_NONE: "none",
                EXIT_SL: TradeStatus.CLOSED_SL.value,
                EXIT_TP: TradeStatus.CLOSED_TP.value,
                EXIT_EOD: TradeStatus.CLOSED_EOD.value
            }
            
            trades = []
            for j in range(len(trade_signal_bars)):
                r_multiple = trade_pnl_dollars[j] / trade_risk_amounts[j] if trade_risk_amounts[j] > 0 else 0.0
                
                trades.append({
                    'signal_bar': int(trade_signal_bars[j]),
                    'entry_bar': int(trade_entry_bars[j]),
                    'exit_bar': int(trade_exit_bars[j]),
                    'signal_time': features.index[int(trade_signal_bars[j])],
                    'entry_time': features.index[int(trade_entry_bars[j])],
                    'exit_time': features.index[int(trade_exit_bars[j])],
                    'direction': 'LONG' if trade_directions[j] == 1 else 'SHORT',
                    'entry_price': trade_entry_prices[j],
                    'exit_price': trade_exit_prices[j],
                    'stop_loss': trade_sl_prices[j],
                    'take_profit': trade_tp_prices[j],
                    'position_size': trade_sizes[j],
                    'pnl_pips': round(trade_pnl_pips[j], 2),
                    'pnl_dollars': round(trade_pnl_dollars[j], 2),
                    'exit_reason': exit_reason_map.get(trade_exit_reasons[j], "unknown"),
                    'risk_amount': round(trade_risk_amounts[j], 2),
                    'r_multiple': round(r_multiple, 3),
                })
            
            # Build equity curve from trades
            equity_curve = [self.config.initial_capital]
            running_equity = self.config.initial_capital
            for t in trades:
                running_equity += t['pnl_dollars']
                equity_curve.append(running_equity)
            
        else:
            # Fallback to pure Python implementation with live-equity sizing
            trades, final_equity, max_drawdown, equity_curve = self._run_python_loop(
                features, open_arr, high_arr, low_arr, close_arr, sig_arr,
                sl_prices, tp_prices, entry_prices,
                half_spread, slippage_price, spec
            )
        
        # Calculate full metrics
        return self._calculate_metrics(
            trades,
            final_equity,
            equity_curve,
            max_drawdown,
            features,
            timeframe,
        )
    
    def _run_python_loop(self, features, open_arr, high_arr, low_arr, close_arr, sig_arr,
                         sl_prices, tp_prices, entry_prices,
                         half_spread, slippage_price, spec) -> Tuple[List[Dict], float, float, List[float]]:
        """
        Pure Python backtest loop (fallback when Numba not available).
        
        Uses LIVE EQUITY for position sizing (compounding preserved).
        """
        equity = self.config.initial_capital
        peak_equity = equity
        max_drawdown = 0.0
        
        trades: List[Dict] = []
        equity_curve: List[float] = [equity]
        
        # Position state
        in_position = False
        position_direction = 0
        entry_price = 0.0
        entry_bar = 0
        signal_bar = 0
        stop_loss = 0.0
        take_profit = 0.0
        position_size = 0.0
        risk_amount_at_entry = 0.0
        
        n_bars = len(features)
        
        for i in range(1, n_bars):
            open_price = open_arr[i]
            high_price = high_arr[i]
            low_price = low_arr[i]
            close_price = close_arr[i]
            
            # CHECK EXITS FOR OPEN POSITIONS
            if in_position:
                exit_price = None
                exit_reason = None
                
                # Calculate bid/ask for this bar
                if self.config.use_spread:
                    bid_high = high_price - half_spread
                    bid_low = low_price - half_spread
                    ask_high = high_price + half_spread
                    ask_low = low_price + half_spread
                else:
                    bid_high = high_price
                    bid_low = low_price
                    ask_high = high_price
                    ask_low = low_price
                
                if position_direction == 1:  # Long position
                    # Check SL (exit at BID, apply adverse slippage)
                    if bid_low <= stop_loss:
                        exit_price = stop_loss - slippage_price
                        exit_reason = TradeStatus.CLOSED_SL.value
                    # Check TP (exit at BID, no slippage)
                    elif bid_high >= take_profit:
                        exit_price = take_profit
                        exit_reason = TradeStatus.CLOSED_TP.value
                        
                else:  # Short position
                    # Check SL (exit at ASK, apply adverse slippage)
                    if ask_high >= stop_loss:
                        exit_price = stop_loss + slippage_price
                        exit_reason = TradeStatus.CLOSED_SL.value
                    # Check TP (exit at ASK, no slippage)
                    elif ask_low <= take_profit:
                        exit_price = take_profit
                        exit_reason = TradeStatus.CLOSED_TP.value
                
                if exit_price is not None:
                    # Calculate P&L using tick-based math
                    pnl_dollars = spec.calculate_tick_profit(
                        entry_price, exit_price, position_size, position_direction
                    )
                    
                    # Calculate pips for reporting
                    if position_direction == 1:
                        pnl_pips = spec.price_to_pips(exit_price - entry_price)
                    else:
                        pnl_pips = spec.price_to_pips(entry_price - exit_price)
                    
                    # Deduct commission
                    if self.config.use_commission:
                        pnl_dollars -= spec.commission_per_lot * position_size
                    
                    # Calculate R-multiple
                    r_multiple = pnl_dollars / risk_amount_at_entry if risk_amount_at_entry > 0 else 0.0
                    
                    equity += pnl_dollars
                    
                    # Record trade
                    trades.append({
                        'signal_bar': signal_bar,
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'signal_time': features.index[signal_bar],
                        'entry_time': features.index[entry_bar],
                        'exit_time': features.index[i],
                        'direction': 'LONG' if position_direction == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'position_size': position_size,
                        'pnl_pips': round(pnl_pips, 2),
                        'pnl_dollars': round(pnl_dollars, 2),
                        'exit_reason': exit_reason,
                        'risk_amount': round(risk_amount_at_entry, 2),
                        'r_multiple': round(r_multiple, 3),
                    })
                    
                    in_position = False
            
            # CHECK ENTRIES
            if not in_position:
                signal = sig_arr[i - 1]
                
                if signal != 0:
                    # Use pre-computed entry/stop prices
                    if not np.isnan(entry_prices[i]) and not np.isnan(sl_prices[i]):
                        direction = int(signal)
                        
                        entry_price_candidate = entry_prices[i]
                        stop_loss_candidate = sl_prices[i]
                        take_profit_candidate = tp_prices[i]
                        
                        # ============================================================
                        # POSITION SIZING FROM LIVE EQUITY (compounding preserved)
                        # ============================================================
                        risk_amount_at_entry = equity * (self.config.position_size_pct / 100.0)
                        
                        # Calculate loss per lot at stop
                        loss_per_lot = spec.calculate_loss_at_stop(
                            entry_price_candidate, stop_loss_candidate, 1.0, direction
                        )
                        
                        if loss_per_lot > 0:
                            raw_size = risk_amount_at_entry / loss_per_lot
                        else:
                            # Fallback to pip-based sizing
                            if direction == 1:
                                dist_pips = spec.price_to_pips(entry_price_candidate - stop_loss_candidate)
                            else:
                                dist_pips = spec.price_to_pips(stop_loss_candidate - entry_price_candidate)
                            if dist_pips > 0 and spec.pip_value > 0:
                                raw_size = risk_amount_at_entry / (dist_pips * spec.pip_value)
                            else:
                                raw_size = spec.min_lot
                        
                        # Round volume
                        position_size = spec.round_volume(raw_size)
                        
                        if position_size > 0:
                            entry_price = entry_price_candidate
                            stop_loss = stop_loss_candidate
                            take_profit = take_profit_candidate
                            
                            in_position = True
                            position_direction = direction
                            entry_bar = i
                            signal_bar = i - 1
            
            # Track equity
            equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, dd)
        
        # CLOSE ANY REMAINING POSITION AT END
        if in_position:
            is_long = position_direction == 1
            if self.config.use_spread:
                exit_price = close_arr[-1] - half_spread if is_long else close_arr[-1] + half_spread
            else:
                exit_price = close_arr[-1]
            
            pnl_dollars = spec.calculate_tick_profit(
                entry_price, exit_price, position_size, position_direction
            )
            
            if position_direction == 1:
                pnl_pips = spec.price_to_pips(exit_price - entry_price)
            else:
                pnl_pips = spec.price_to_pips(entry_price - exit_price)
            
            if self.config.use_commission:
                pnl_dollars -= spec.commission_per_lot * position_size
            
            r_multiple = pnl_dollars / risk_amount_at_entry if risk_amount_at_entry > 0 else 0.0
            
            equity += pnl_dollars
            
            trades.append({
                'signal_bar': signal_bar,
                'entry_bar': entry_bar,
                'exit_bar': n_bars - 1,
                'signal_time': features.index[signal_bar],
                'entry_time': features.index[entry_bar],
                'exit_time': features.index[-1],
                'direction': 'LONG' if position_direction == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'pnl_pips': round(pnl_pips, 2),
                'pnl_dollars': round(pnl_dollars, 2),
                'exit_reason': TradeStatus.CLOSED_EOD.value,
                'risk_amount': round(risk_amount_at_entry, 2),
                'r_multiple': round(r_multiple, 3),
            })
            
            # Update equity curve and drawdown after EOD close
            equity_curve[-1] = equity  # Update last point with final equity
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            max_drawdown = max(max_drawdown, dd)
        
        return trades, equity, max_drawdown, equity_curve
    
    # Bars-per-year lookup for known timeframes (252 trading days assumed)
    _TF_BARS_PER_YEAR = {
        'M1': 252 * 24 * 60,
        'M5': 252 * 24 * 12,
        'M15': 252 * 24 * 4,
        'M30': 252 * 24 * 2,
        'H1': 252 * 24,
        'H4': 252 * 6,
        'D1': 252,
        'W1': 52,
        'MN1': 12,
    }

    def _calculate_metrics(self, 
                           trades: List[Dict],
                           final_equity: float,
                           equity_curve: List[float],
                           max_dd: float,
                           features: pd.DataFrame = None,
                           timeframe: str = "",
                           warmup_bars: int = 0) -> Dict[str, Any]:
        """
        Calculate complete performance metrics.
        
        Sharpe/Sortino ratios are now computed from equity curve returns (per bar),
        annualized based on timeframe. This provides more accurate risk-adjusted
        returns compared to per-trade calculations.
        
        Args:
            trades: List of trade dictionaries
            final_equity: Final account equity
            equity_curve: List of equity values
            max_dd: Maximum drawdown percentage
            features: Optional features DataFrame (for timeframe detection)
            timeframe: Optional timeframe string for annualization (e.g. 'H1', 'M15')
            warmup_bars: Exclude trades whose entry_bar < warmup_bars
            
        Returns:
            Dict with all performance metrics including R-multiple stats
        """
        if not trades:
            return self._empty_result()

        # Optional exclusion of trades entered during warmup.
        if warmup_bars > 0:
            trades = [t for t in trades if t.get('entry_bar', 0) >= warmup_bars]
            if not trades:
                return self._empty_result()
        
        # Basic counts
        total_trades = len(trades)
        wins = [t for t in trades if t['pnl_pips'] > 0]
        losses = [t for t in trades if t['pnl_pips'] < 0]
        breakeven = [t for t in trades if t['pnl_pips'] == 0]
        
        # P&L calculations
        total_pnl = sum(t['pnl_dollars'] for t in trades)
        win_pnl = sum(t['pnl_dollars'] for t in wins) if wins else 0
        loss_pnl = abs(sum(t['pnl_dollars'] for t in losses)) if losses else 0
        
        # Win rate
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        
        # Profit factor
        profit_factor = win_pnl / loss_pnl if loss_pnl > 0 else 99.0
        profit_factor = min(profit_factor, 99.0)  # Cap at 99
        
        # Average trade metrics
        avg_win_pips = sum(t['pnl_pips'] for t in wins) / len(wins) if wins else 0
        avg_loss_pips = abs(sum(t['pnl_pips'] for t in losses) / len(losses)) if losses else 0
        avg_trade_pips = sum(t['pnl_pips'] for t in trades) / total_trades
        
        avg_win_dollars = win_pnl / len(wins) if wins else 0
        avg_loss_dollars = loss_pnl / len(losses) if losses else 0
        avg_trade_dollars = total_pnl / total_trades
        
        # Expectancy
        if win_rate > 0:
            expectancy = (win_rate / 100 * avg_win_pips) - ((1 - win_rate / 100) * avg_loss_pips)
        else:
            expectancy = -avg_loss_pips
        
        # ================================================================
        # Sharpe and Sortino ratios from EQUITY CURVE returns (per bar)
        # Annualized based on timeframe for proper comparison
        # ================================================================
        sharpe = 0.0
        sortino = 0.0
        
        # Need at least 3 equity points to get 2 returns for meaningful std calculation
        if len(equity_curve) > 2:
            # Convert equity curve to returns
            equity_arr = np.array(equity_curve)
            # Per-bar returns (percentage)
            bar_returns = np.diff(equity_arr) / np.where(equity_arr[:-1] != 0, equity_arr[:-1], 1.0)
            
            # Determine annualization factor based on timeframe
            bars_per_year = self._TF_BARS_PER_YEAR.get(str(timeframe).upper(), 252 * 24)
            
            if features is not None and len(features) > 1:
                # Try to infer timeframe from data frequency
                try:
                    time_diff = (features.index[1] - features.index[0]).total_seconds()
                    bars_per_day = 86400 / max(time_diff, 60)  # Avoid div by zero
                    bars_per_year = bars_per_day * 252
                except:
                    pass  # Use default
            
            # Sharpe ratio (annualized from per-bar returns)
            # Need at least 2 returns for ddof=1 std calculation
            if len(bar_returns) >= 2:
                returns_mean = np.mean(bar_returns)
                returns_std = np.std(bar_returns, ddof=1) if len(bar_returns) > 1 else 0.0
                if returns_std > 1e-10:
                    sharpe = (returns_mean / returns_std) * np.sqrt(bars_per_year)
            
            # Sortino ratio (annualized, using downside deviation)
            # No artificial cap - let the metric speak for itself
            if len(bar_returns) >= 2:
                negative_returns = bar_returns[bar_returns < 0]
                # Need at least 2 negative returns for ddof=1 std calculation
                if len(negative_returns) >= 2:
                    downside_std = np.std(negative_returns, ddof=1)
                    if downside_std > 1e-10:
                        sortino = (np.mean(bar_returns) / downside_std) * np.sqrt(bars_per_year)
                elif len(negative_returns) == 1:
                    # Single negative return - use its absolute value as downside deviation proxy
                    downside_std = abs(negative_returns[0])
                    if downside_std > 1e-10:
                        sortino = (np.mean(bar_returns) / downside_std) * np.sqrt(bars_per_year)
                else:
                    # No negative returns - very high sortino but not infinity
                    if np.mean(bar_returns) > 0:
                        sortino = float(sharpe * 2) if sharpe > 0 else 0.0  # Rough approximation
        
        # ================================================================
        # R-Multiple Statistics (risk-normalized performance)
        # ================================================================
        r_multiples = [t.get('r_multiple', 0.0) for t in trades if 'r_multiple' in t]
        
        if r_multiples:
            mean_r = float(np.mean(r_multiples))
            median_r = float(np.median(r_multiples))
            pct_positive_r = len([r for r in r_multiples if r > 0]) / len(r_multiples) * 100
            worst_5pct_r = float(np.percentile(r_multiples, 5)) if len(r_multiples) >= 20 else min(r_multiples)
        else:
            mean_r = median_r = pct_positive_r = worst_5pct_r = 0.0
        
        # Calmar ratio
        total_return_pct = (final_equity - self.config.initial_capital) / self.config.initial_capital * 100
        calmar = total_return_pct / max_dd if max_dd > 0 else 0
        
        # Consecutive wins/losses
        max_consec_wins = self._max_consecutive(trades, win=True)
        max_consec_losses = self._max_consecutive(trades, win=False)
        
        # Trade duration statistics
        if trades:
            durations = [t['exit_bar'] - t['entry_bar'] for t in trades]
            avg_duration = np.mean(durations)
            max_duration = max(durations)
            min_duration = min(durations)
        else:
            avg_duration = max_duration = min_duration = 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            reason = t['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Long/Short breakdown
        long_trades = [t for t in trades if t['direction'] == 'LONG']
        short_trades = [t for t in trades if t['direction'] == 'SHORT']
        
        long_win_rate = len([t for t in long_trades if t['pnl_pips'] > 0]) / len(long_trades) * 100 if long_trades else 0
        short_win_rate = len([t for t in short_trades if t['pnl_pips'] > 0]) / len(short_trades) * 100 if short_trades else 0
        
        return {
            # Core metrics
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'breakeven_trades': len(breakeven),
            'win_rate': round(win_rate, 2),
            
            # P&L metrics
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'gross_profit': round(win_pnl, 2),
            'gross_loss': round(loss_pnl, 2),
            'profit_factor': round(profit_factor, 2),
            
            # Average trade metrics
            'avg_win_pips': round(avg_win_pips, 2),
            'avg_loss_pips': round(avg_loss_pips, 2),
            'avg_trade_pips': round(avg_trade_pips, 2),
            'avg_win_dollars': round(avg_win_dollars, 2),
            'avg_loss_dollars': round(avg_loss_dollars, 2),
            'avg_trade_dollars': round(avg_trade_dollars, 2),
            
            # Risk metrics
            'max_drawdown_pct': round(max_dd, 2),
            'expectancy_pips': round(expectancy, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),  # No cap - let the metric speak
            'calmar_ratio': round(calmar, 2),
            
            # R-Multiple metrics (risk-normalized)
            'mean_r': round(mean_r, 3),
            'median_r': round(median_r, 3),
            'pct_positive_r': round(pct_positive_r, 2),
            'worst_5pct_r': round(worst_5pct_r, 3),
            
            # Streak metrics
            'max_consecutive_wins': max_consec_wins,
            'max_consecutive_losses': max_consec_losses,
            
            # Duration metrics
            'avg_trade_duration': round(avg_duration, 1),
            'max_trade_duration': max_duration,
            'min_trade_duration': min_duration,
            
            # Breakdown metrics
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': round(long_win_rate, 2),
            'short_win_rate': round(short_win_rate, 2),
            'exit_reasons': exit_reasons,
            
            # Raw data
            'final_equity': round(final_equity, 2),
            'equity_curve': equity_curve,
            'trades': trades,
        }
    
    def _max_consecutive(self, trades: List[Dict], win: bool = True) -> int:
        """Calculate maximum consecutive wins or losses."""
        max_streak = 0
        current_streak = 0
        
        for trade in trades:
            is_win = trade['pnl_pips'] > 0
            if is_win == win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    @staticmethod
    def validate_execution_timing(results: Dict[str, Any], max_trades_to_check: int = 5) -> Tuple[bool, List[str]]:
        """
        Validate that backtest execution timing matches the live trading contract.
        
        The contract states:
        - Signal from bar i-1 triggers entry on bar i
        - entry_bar_index = signal_bar_index + 1
        - entry_time > signal_time
        
        Args:
            results: Backtest results dictionary containing 'trades'
            max_trades_to_check: Number of trades to validate (default: first 5)
            
        Returns:
            Tuple of (is_valid, list of validation messages)
        """
        trades = results.get('trades', [])
        
        if not trades:
            return True, ["No trades to validate"]
        
        messages = []
        all_valid = True
        
        num_to_check = min(len(trades), max_trades_to_check)
        messages.append(f"Validating execution timing for first {num_to_check} trade(s):")
        messages.append("-" * 60)
        
        for idx, trade in enumerate(trades[:num_to_check]):
            signal_bar = trade.get('signal_bar')
            entry_bar = trade.get('entry_bar')
            signal_time = trade.get('signal_time')
            entry_time = trade.get('entry_time')
            
            # Check if required fields exist
            if signal_bar is None or entry_bar is None:
                messages.append(f"  Trade {idx + 1}: MISSING signal_bar or entry_bar fields")
                all_valid = False
                continue
            
            # Validate: entry_bar = signal_bar + 1 (exactly 1 bar delay)
            bar_delay_valid = (entry_bar == signal_bar + 1)
            
            # Validate: entry_time > signal_time
            time_order_valid = True
            if signal_time is not None and entry_time is not None:
                time_order_valid = (entry_time > signal_time)
            
            trade_valid = bar_delay_valid and time_order_valid
            status = "✓ PASS" if trade_valid else "✗ FAIL"
            
            messages.append(f"  Trade {idx + 1}: {status}")
            messages.append(f"    signal_bar={signal_bar}, entry_bar={entry_bar} "
                          f"(expected entry_bar={signal_bar + 1})")
            if signal_time is not None and entry_time is not None:
                messages.append(f"    signal_time={signal_time}")
                messages.append(f"    entry_time ={entry_time}")
                messages.append(f"    entry_time > signal_time: {time_order_valid}")
            
            if not trade_valid:
                all_valid = False
        
        messages.append("-" * 60)
        overall_status = "PASS" if all_valid else "FAIL"
        messages.append(f"Execution timing validation: {overall_status}")
        
        return all_valid, messages
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result when no trades."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return_pct': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'profit_factor': 0,
            'avg_win_pips': 0,
            'avg_loss_pips': 0,
            'avg_trade_pips': 0,
            'avg_win_dollars': 0,
            'avg_loss_dollars': 0,
            'avg_trade_dollars': 0,
            'max_drawdown_pct': 0,
            'expectancy_pips': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'mean_r': 0,
            'median_r': 0,
            'pct_positive_r': 0,
            'worst_5pct_r': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_trade_duration': 0,
            'max_trade_duration': 0,
            'min_trade_duration': 0,
            'long_trades': 0,
            'short_trades': 0,
            'long_win_rate': 0,
            'short_win_rate': 0,
            'exit_reasons': {},
            'final_equity': self.config.initial_capital,
            'equity_curve': [self.config.initial_capital],
            'trades': [],
        }


# =============================================================================
# SCORING AND EVALUATION
# =============================================================================

class StrategyScorer:
    """
    Scores and ranks strategy performance.
    
    Uses weighted composite scoring to select the best strategy
    for a given symbol and timeframe.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize StrategyScorer.
        
        Args:
            config: Pipeline configuration with score weights
        """
        self.config = config
        self.weights = config.score_weights
    
    def calculate_composite_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate composite score from metrics.
        
        Args:
            metrics: Backtest result metrics
            
        Returns:
            Composite score (0-100)
        """
        if metrics['total_trades'] < self.config.min_trades:
            return 0.0
        
        # Normalize each metric to 0-100 scale
        scores = {}
        
        # Sharpe ratio: target 2.0 = 100
        scores['sharpe'] = min(100, max(0, metrics['sharpe_ratio'] / 2.0 * 100))
        
        # Profit factor: 1.0 = 0, 3.0+ = 100
        pf = metrics['profit_factor']
        scores['profit_factor'] = min(100, max(0, (pf - 1.0) / 2.0 * 100))
        
        # Win rate: 40% = 0, 70%+ = 100
        wr = metrics['win_rate']
        scores['win_rate'] = min(100, max(0, (wr - 40) / 30 * 100))
        
        # Total return: 0% = 0, 100%+ = 100
        ret = metrics['total_return_pct']
        scores['total_return'] = min(100, max(0, ret))
        
        # Max drawdown: 0% = 100, 30%+ = 0 (inverted)
        dd = metrics['max_drawdown_pct']
        scores['max_drawdown'] = max(0, 100 - (dd / 30 * 100))
        
        # Trade count: 20 = 0, 100+ = 100
        tc = metrics['total_trades']
        scores['trade_count'] = min(100, max(0, (tc - 20) / 80 * 100))
        
        # Calculate weighted composite
        composite = 0.0
        for metric, weight in self.weights.items():
            if metric in scores:
                composite += scores[metric] * weight
        
        return round(composite, 2)


    def calculate_fx_selection_score(self, metrics: Dict[str, Any]) -> float:
        """Backtester-aligned strategy selection score (training data only).

        Mirrors fx_pipeline_main._calculate_score() with calibration extensions:
        - Continuous DD penalty (replaces discrete buckets)
        - Sortino/Sharpe blend for downside-aware risk scoring
        - Tail-risk penalty via worst 5th-percentile R-multiple
        - Consistency penalty via max consecutive losses
        - Trade-frequency bonus (log-scaled statistical confidence)
        All extensions gated by PipelineConfig feature flags.
        """
        sharpe = min(max(metrics.get('sharpe_ratio', 0.0), -2.0), 5.0)
        pf = min(max(metrics.get('profit_factor', 0.0), 0.0), 10.0)
        win_rate = float(metrics.get('win_rate', 0.0)) / 100.0
        ret = float(metrics.get('total_return_pct', 0.0))
        dd = float(metrics.get('max_drawdown_pct', 100.0))

        return_dd = (ret / dd) if dd > 0 else ret
        return_dd = min(max(return_dd, -5.0), 20.0)

        expectancy = float(metrics.get('expectancy_pips', 0.0))

        # --- Sortino/Sharpe blend ---
        risk_adj = sharpe
        if getattr(self.config, 'scoring_use_sortino_blend', False):
            sortino = min(max(metrics.get('sortino_ratio', sharpe), -2.0), 5.0)
            risk_adj = 0.6 * sharpe + 0.4 * sortino

        score = (
            ((risk_adj + 2.0) / 7.0) * 25.0 +
            (pf / 10.0) * 20.0 +
            (win_rate * 15.0) +
            ((return_dd + 5.0) / 25.0) * 25.0 +
            min(expectancy / 10.0, 1.0) * 15.0
        )

        # --- Drawdown penalty ---
        if getattr(self.config, 'scoring_use_continuous_dd', False):
            # Smooth exponential: dd=0 → 1.0, dd=15 → 0.64, dd=25 → 0.47, dd=35 → 0.35
            dd_mult = math.exp(-0.03 * dd)
            score *= dd_mult
        else:
            # Legacy discrete buckets
            if dd > 30.0:
                score *= 0.6
            elif dd > 25.0:
                score *= 0.75
            elif dd > 20.0:
                score *= 0.9

        # --- Tail-risk penalty ---
        if getattr(self.config, 'scoring_use_tail_risk', False):
            worst_5 = float(metrics.get('worst_5pct_r', 0.0))
            if worst_5 < -3.0:
                # Severe tail: penalize proportionally, clamped
                tail_penalty = max(0.70, 1.0 + 0.05 * worst_5)  # worst_5=-5 → 0.75
                score *= tail_penalty

        # --- Consistency penalty ---
        if getattr(self.config, 'scoring_use_consistency', False):
            max_consec = int(metrics.get('max_consecutive_losses', 0))
            if max_consec > 8:
                # Penalize fragile strategies with very long losing streaks
                consec_penalty = max(0.75, 1.0 - 0.03 * (max_consec - 8))  # 12 → 0.88, 16+ → 0.75
                score *= consec_penalty

        # --- Trade-frequency bonus ---
        if getattr(self.config, 'scoring_use_trade_frequency_bonus', False):
            tc = int(metrics.get('total_trades', 0))
            if tc > 30:
                # Log-scaled bonus: 30 trades → 0%, 100 → ~3.5%, 300 → ~6.5%
                freq_bonus = min(0.08, 0.03 * math.log(tc / 30.0))
                score *= (1.0 + freq_bonus)

        return float(score)

    def calculate_fx_opt_score(self, metrics: Dict[str, Any]) -> float:
        """Backtester-aligned optimization score (train only).

        Mirrors fx_pipeline_main._calc_score() with calibration extensions:
        - Continuous DD penalty (replaces 3-tier discrete)
        - Profit factor and win rate components
        - Sortino blend for downside awareness
        All extensions gated by PipelineConfig feature flags.
        """
        sharpe = max(-2.0, min(5.0, float(metrics.get('sharpe_ratio', 0.0))))
        ret = float(metrics.get('total_return_pct', 0.0))
        dd = float(metrics.get('max_drawdown_pct', 100.0))
        pf = min(max(float(metrics.get('profit_factor', 0.0)), 0.0), 10.0)
        wr = float(metrics.get('win_rate', 0.0)) / 100.0

        # --- Sortino blend ---
        risk_adj = sharpe
        if getattr(self.config, 'scoring_use_sortino_blend', False):
            sortino = max(-2.0, min(5.0, float(metrics.get('sortino_ratio', sharpe))))
            risk_adj = 0.6 * sharpe + 0.4 * sortino

        # Base score: risk-adjusted return + return magnitude + PF + win rate
        base = (
            risk_adj * 30.0 +
            min(ret, 500.0) * 0.1 +
            min(pf, 5.0) * 2.0 +
            wr * 5.0
        )

        # --- DD penalty ---
        if getattr(self.config, 'scoring_use_continuous_dd', False):
            dd_penalty = math.exp(-0.03 * dd)
        else:
            dd_penalty = 1.0 if dd < 15.0 else 0.8 if dd < 25.0 else 0.5

        return float(base * dd_penalty)
    def calculate_return_robustness_ratio(self,
                                          train_metrics: Dict[str, Any],
                                          val_metrics: Dict[str, Any]) -> float:
        """Backtester-aligned robustness ratio.

        Historical versions used val_return / abs(train_return). In practice that ratio becomes
        uninformative under large compounded returns (it collapses toward ~0 even when validation
        is objectively strong).

        We now define robustness as a *score* generalization ratio:
            robustness = val_selection_score / max(train_selection_score, eps)

        This correlates with out-of-sample stability while remaining aligned to the PM objectives
        (Sharpe, PF, WR, return/DD, expectancy).
        """
        return self.calculate_fx_score_robustness_ratio(train_metrics, val_metrics, purpose="selection")

    def calculate_fx_score_robustness_ratio(self,
                                            train_metrics: Dict[str, Any],
                                            val_metrics: Dict[str, Any],
                                            purpose: str = "selection",
                                            eps: float = 1e-9) -> float:
        """Robustness ratio based on scorer outputs (val_score/train_score)."""
        train_score = float(self.score(train_metrics, purpose=purpose))
        val_score = float(self.score(val_metrics, purpose=purpose))
        denom = train_score if abs(train_score) > eps else eps
        ratio = val_score / denom
        return float(np.clip(ratio, 0.0, 2.0))

    def fx_generalization_score(self,
                                train_metrics: Dict[str, Any],
                                val_metrics: Dict[str, Any],
                                purpose: str = "selection") -> Tuple[float, float, float, float]:
        """Compute validation-aware score for fx_backtester mode.

        Returns:
            (final_score, train_score, val_score, robustness_ratio)
        """
        train_score = float(self.score(train_metrics, purpose=purpose))
        val_score = float(self.score(val_metrics, purpose=purpose))

        gap = max(0.0, train_score - val_score)
        lam = float(getattr(self.config, "fx_gap_penalty_lambda", 0.50))
        final_score = val_score - lam * gap

        rr = self.calculate_fx_score_robustness_ratio(train_metrics, val_metrics, purpose="selection")
        boost_w = float(getattr(self.config, "fx_robustness_boost", 0.15))
        rr_clipped = float(np.clip(rr, 0.0, 1.25))
        robust_mult = (1.0 - boost_w) + boost_w * rr_clipped
        final_score *= robust_mult

        return float(final_score), float(train_score), float(val_score), float(rr)

    def score(self, metrics: Dict[str, Any], purpose: str = "selection") -> float:
        """Unified scoring honoring config.scoring_mode.

        purpose:
          - "selection": strategy selection / rolling window scoring
          - "opt": hyperparameter tuning comparisons
        """
        mode = getattr(self.config, "scoring_mode", "pm_weighted")
        if mode == "fx_backtester":
            if purpose == "opt":
                return self.calculate_fx_opt_score(metrics)
            return self.calculate_fx_selection_score(metrics)
        return self.calculate_composite_score(metrics)
    
    def passes_minimum_criteria(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if metrics pass minimum criteria.
        
        Args:
            metrics: Backtest result metrics
            
        Returns:
            Tuple of (passes, reason_if_failed)
        """
        if metrics['total_trades'] < self.config.min_trades:
            return False, f"Insufficient trades: {metrics['total_trades']} < {self.config.min_trades}"
        
        if metrics['win_rate'] < self.config.min_win_rate:
            return False, f"Low win rate: {metrics['win_rate']}% < {self.config.min_win_rate}%"
        
        if metrics['profit_factor'] < self.config.min_profit_factor:
            return False, f"Low profit factor: {metrics['profit_factor']} < {self.config.min_profit_factor}"
        
        if metrics['sharpe_ratio'] < self.config.min_sharpe:
            return False, f"Low Sharpe ratio: {metrics['sharpe_ratio']} < {self.config.min_sharpe}"
        
        if metrics['max_drawdown_pct'] > self.config.max_drawdown:
            return False, f"High drawdown: {metrics['max_drawdown_pct']}% > {self.config.max_drawdown}%"
        
        return True, "OK"
    
    def calculate_robustness_ratio(self, 
                                    train_metrics: Dict[str, Any],
                                    val_metrics: Dict[str, Any]) -> float:
        """
        Calculate robustness ratio between training and validation.
        
        A ratio close to 1.0 indicates good generalization.
        
        Args:
            train_metrics: Training period metrics
            val_metrics: Validation period metrics
            
        Returns:
            Robustness ratio (val_sharpe / train_sharpe)
        """
        train_sharpe = train_metrics.get('sharpe_ratio', 0)
        val_sharpe = val_metrics.get('sharpe_ratio', 0)
        
        if train_sharpe <= 0:
            return 0.0
        
        ratio = val_sharpe / train_sharpe
        return round(min(ratio, 2.0), 3)  # Cap at 2.0


# =============================================================================
# MAIN EXPORTS
# =============================================================================

__all__ = [
    'PipelineConfig',
    'SignalType',
    'StrategyCategory',
    'TradeStatus',
    'Timer',
    'InstrumentSpec',
    'INSTRUMENT_SPECS',
    'get_instrument_spec',
    'sync_instrument_spec_from_mt5',
    'load_broker_specs',
    'set_broker_specs_path',
    'set_instrument_specs',
    'save_broker_specs',
    'DataLoader',
    'FeatureComputer',
    'DataSplitter',
    'Backtester',
    'StrategyScorer',
]


if __name__ == "__main__":
    # Basic tests
    logging.basicConfig(level=logging.INFO)
    
    config = PipelineConfig()
    print(f"Config: {config}")
    
    # Test instrument spec
    spec = get_instrument_spec('EURUSD')
    print(f"\nEURUSD spec: pip_size={spec.pip_size}, pip_value={spec.pip_value}")
    print(f"  10 pips in price: {spec.pips_to_price(10)}")
    print(f"  0.001 price in pips: {spec.price_to_pips(0.001)}")
    
    # Test data splitter
    splitter = DataSplitter(config)
    indices = splitter.get_split_indices(1000)
    print(f"\nSplit indices for 1000 rows: {indices}")
    
    print("\npm_core.py loaded successfully!")
