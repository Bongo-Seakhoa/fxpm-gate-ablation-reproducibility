"""
FX Portfolio Manager - Optuna TPE Optimization Module
======================================================

Production-ready implementation of Optuna-based hyperparameter optimization
using Tree-structured Parzen Estimators (TPE) for efficient search.

Key Features:
- TPE sampler optimized for discrete/categorical parameter spaces
- Parameter constraint handling (e.g., fast_period < slow_period)
- Multi-objective regime-aware optimization
- Robust logging with trial statistics
- Reproducible results via seed control
- Graceful fallback when Optuna unavailable

Integration Points:
- HyperparameterOptimizer: Single-strategy optimization
- RegimeOptimizer: Multi-regime simultaneous optimization
- RetrainPeriodSelector: Retrain period optimization

Version: 1.1 (Updated with ExperimentalWarning suppression)
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

# Optuna import with graceful fallback
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.trial import TrialState
    from optuna.exceptions import ExperimentalWarning  # Import for suppression
    import warnings  # Import for suppression
    
    OPTUNA_AVAILABLE = True
    
    # Configure Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Suppress Optuna ExperimentalWarning (specifically for multivariate TPE)
    # This keeps logs clean while using the advanced correlation features
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None
    MedianPruner = None
    HyperbandPruner = None
    TrialState = None

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptunaConfig:
    """Configuration for Optuna optimization."""
    # Trial budget
    n_trials: int = 100
    
    # TPE sampler settings
    n_startup_trials: int = 10  # Random trials before TPE kicks in
    seed: int = 42
    multivariate: bool = True  # Consider parameter correlations
    
    # Pruning settings
    use_pruning: bool = False  # Disabled by default - full eval needed for trading
    pruning_warmup_steps: int = 5
    
    # Timeout (optional)
    timeout_seconds: Optional[float] = None

    # Objective definition
    # If True, Optuna maximizes a score that includes validation metrics. This can overfit
    # to the holdout split because the same validation set is later used for selection.
    # Default is False to avoid train/val leakage: tune on train, validate on val.
    use_validation_in_objective: bool = False

    # Blended objective weights (train-first with bounded val influence)
    objective_blend_enabled: bool = True
    objective_train_weight: float = 0.80
    objective_val_weight: float = 0.20

    # Early rejection thresholds (configurable)
    train_dd_multiplier: float = 1.25
    val_dd_multiplier: float = 1.0
    penalty_dd_rejection: float = -500.0
    penalty_no_trades: float = -1000.0
    # Progressive rejection: relax thresholds during early exploration
    progressive_rejection: bool = True
    progressive_warmup_frac: float = 0.20
    progressive_warmup_multiplier: float = 1.5
    
    # Logging
    log_interval: int = 25  # Log progress every N trials
    
    @classmethod
    def from_pipeline_config(cls, config) -> 'OptunaConfig':
        """Create OptunaConfig from PipelineConfig."""
        return cls(
            n_trials=getattr(config, 'regime_hyperparam_max_combos', 100),
            n_startup_trials=min(10, getattr(config, 'regime_hyperparam_max_combos', 100) // 10),
            seed=42,
            multivariate=True,
            use_validation_in_objective=bool(getattr(config, 'optuna_use_val_in_objective', False)),
            objective_blend_enabled=bool(getattr(config, 'optuna_objective_blend_enabled', True)),
            objective_train_weight=float(getattr(config, 'optuna_objective_train_weight', 0.80)),
            objective_val_weight=float(getattr(config, 'optuna_objective_val_weight', 0.20)),
        )


# =============================================================================
# PARAMETER SPACE DEFINITION
# =============================================================================

class ParameterSpace:
    """
    Defines and samples from strategy parameter spaces for Optuna.
    
    Handles:
    - Discrete/categorical parameters from strategy param grids
    - Parameter constraints (e.g., slow_period > fast_period)
    - Type conversion for Optuna suggestions
    """
    
    # Known parameter constraints
    CONSTRAINTS = {
        # (param1, param2, relation): param1 relation param2
        ('fast_period', 'slow_period', 'lt'),  # fast_period < slow_period
        ('tenkan_period', 'kijun_period', 'lt'),  # tenkan < kijun (Ichimoku)
        ('short_period', 'long_period', 'lt'),  # short < long
        ('signal_period', 'slow_period', 'lt'),  # signal < slow (MACD)
    }
    
    def __init__(self, param_grid: Dict[str, List], default_params: Dict[str, Any]):
        """
        Initialize parameter space from strategy param grid.
        
        Args:
            param_grid: Dict mapping param names to lists of possible values
            default_params: Default parameter values from strategy
        """
        self.param_grid = param_grid
        self.default_params = default_params
        self.param_info = self._analyze_params()
    
    def _analyze_params(self) -> Dict[str, Dict[str, Any]]:
        """Analyze parameter grid to determine types and ranges."""
        info = {}
        
        for name, values in self.param_grid.items():
            if not values:
                continue
            
            sample = values[0]
            sorted_vals = sorted(values) if all(isinstance(v, (int, float)) for v in values) else values
            
            if isinstance(sample, bool):
                info[name] = {
                    'type': 'categorical',
                    'choices': values,
                }
            elif isinstance(sample, int) and not isinstance(sample, bool):
                # Check if values form a continuous range
                if len(values) >= 2 and all(isinstance(v, int) for v in values):
                    info[name] = {
                        'type': 'int',
                        'low': min(values),
                        'high': max(values),
                        'choices': sorted_vals,  # Keep for constraint checking
                    }
                else:
                    info[name] = {
                        'type': 'categorical',
                        'choices': values,
                    }
            elif isinstance(sample, float):
                info[name] = {
                    'type': 'float',
                    'low': min(values),
                    'high': max(values),
                    'choices': sorted_vals,
                }
            else:
                info[name] = {
                    'type': 'categorical',
                    'choices': values,
                }
        
        return info
    
    def suggest(self, trial: 'optuna.Trial') -> Dict[str, Any]:
        """
        Suggest parameters using Optuna trial.
        
        Handles parameter constraints by adjusting dependent parameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dict of suggested parameters merged with defaults
        """
        params = self.default_params.copy()
        
        # Suggest each parameter
        for name, info in self.param_info.items():
            if info['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, info['choices'])
            elif info['type'] == 'int':
                # For discrete integers, use categorical to match grid exactly
                params[name] = trial.suggest_categorical(name, info['choices'])
            elif info['type'] == 'float':
                # For floats, also use categorical to match grid
                params[name] = trial.suggest_categorical(name, info['choices'])
        
        # Apply constraints
        params = self._apply_constraints(params)
        
        return params
    
    def _apply_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter constraints to ensure valid configurations."""
        # EMA/MA crossover: slow > fast
        if 'fast_period' in params and 'slow_period' in params:
            if params['slow_period'] <= params['fast_period']:
                # Adjust slow_period to be greater
                fast = params['fast_period']
                slow_choices = self.param_info.get('slow_period', {}).get('choices', [])
                valid_slow = [s for s in slow_choices if s > fast]
                if valid_slow:
                    params['slow_period'] = min(valid_slow)
                else:
                    params['slow_period'] = fast + 5
        
        # Ichimoku: kijun > tenkan
        if 'tenkan_period' in params and 'kijun_period' in params:
            if params['kijun_period'] <= params['tenkan_period']:
                tenkan = params['tenkan_period']
                kijun_choices = self.param_info.get('kijun_period', {}).get('choices', [])
                valid_kijun = [k for k in kijun_choices if k > tenkan]
                if valid_kijun:
                    params['kijun_period'] = min(valid_kijun)
                else:
                    params['kijun_period'] = tenkan + 5
        
        # MACD: signal < slow
        if 'signal_period' in params and 'slow_period' in params:
            if params['signal_period'] >= params['slow_period']:
                slow = params['slow_period']
                signal_choices = self.param_info.get('signal_period', {}).get('choices', [])
                valid_signal = [s for s in signal_choices if s < slow]
                if valid_signal:
                    params['signal_period'] = max(valid_signal)
                else:
                    params['signal_period'] = max(1, slow - 1)
        
        # Stochastic: d <= k
        if 'k_period' in params and 'd_period' in params:
            if params['d_period'] > params['k_period']:
                params['d_period'] = params['k_period']
        
        return params
    
    def get_search_space_size(self) -> int:
        """Calculate total size of parameter search space."""
        size = 1
        for info in self.param_info.values():
            if 'choices' in info:
                size *= len(info['choices'])
        return size


# =============================================================================
# OPTIMIZATION RESULTS
# =============================================================================

@dataclass
class OptimizationStats:
    """Statistics from an optimization run."""
    n_trials: int = 0
    n_completed: int = 0
    n_pruned: int = 0
    n_failed: int = 0
    best_score: float = float('-inf')
    optimization_time_sec: float = 0.0
    method: str = "optuna_tpe"
    early_dd_rejections: int = 0  # Count of trials rejected due to high drawdown
    
    def __str__(self) -> str:
        base = (f"Trials: {self.n_trials} (completed={self.n_completed}, "
                f"pruned={self.n_pruned}, failed={self.n_failed}), "
                f"best_score={self.best_score:.4f}, time={self.optimization_time_sec:.1f}s")
        if self.early_dd_rejections > 0:
            base += f", early_dd_rejects={self.early_dd_rejections}"
        return base


@dataclass
class OptimizationResult:
    """Result from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    train_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    stats: OptimizationStats
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'stats': {
                'n_trials': self.stats.n_trials,
                'n_completed': self.stats.n_completed,
                'n_pruned': self.stats.n_pruned,
                'optimization_time_sec': self.stats.optimization_time_sec,
                'method': self.stats.method,
                'early_dd_rejections': self.stats.early_dd_rejections,
            }
        }


# =============================================================================
# CORE OPTIMIZER
# =============================================================================

class OptunaTPEOptimizer:
    """
    Core Optuna TPE optimizer for hyperparameter tuning.
    
    Uses Tree-structured Parzen Estimators (TPE) which are well-suited for:
    - Discrete/categorical parameter spaces
    - Mixed parameter types
    - Parameter constraints
    
    TPE models P(x|y) and P(y) separately, making it efficient for
    trading strategy optimization where evaluations are expensive.
    """
    
    def __init__(self, 
                 config: OptunaConfig,
                 backtester,
                 scorer,
                 strategy_registry):
        """
        Initialize the optimizer.
        
        Args:
            config: OptunaConfig with optimization settings
            backtester: Backtester instance for strategy evaluation
            scorer: StrategyScorer instance for computing scores
            strategy_registry: StrategyRegistry for creating strategy instances
        """
        self.config = config
        self.backtester = backtester
        self.scorer = scorer
        self.strategy_registry = strategy_registry
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Install with: pip install optuna")
    
    def optimize(self,
                 symbol: str,
                 strategy_name: str,
                 param_grid: Dict[str, List],
                 train_features: pd.DataFrame,
                 val_features: Optional[pd.DataFrame],
                 scoring_fn: Callable[[Dict, Dict], float],
                 min_trades: int = 10,
                 max_drawdown_pct: float = 30.0) -> OptimizationResult:
        """
        Optimize strategy parameters using Optuna TPE with early rejection.
        
        Args:
            symbol: Trading symbol
            strategy_name: Name of strategy to optimize
            param_grid: Parameter grid from strategy
            train_features: Training data
            val_features: Validation data (can be None)
            scoring_fn: Function(train_metrics, val_metrics) -> score
            min_trades: Minimum trades required for valid evaluation
            max_drawdown_pct: Maximum drawdown threshold for early rejection
            
        Returns:
            OptimizationResult with best parameters and metrics
        """
        if not OPTUNA_AVAILABLE:
            return self._fallback_random_search(
                symbol, strategy_name, param_grid, train_features, 
                val_features, scoring_fn, min_trades, max_drawdown_pct
            )
        
        start_time = time.time()
        
        # Get default params
        base_strategy = self.strategy_registry.get(strategy_name)
        default_params = base_strategy.get_params()
        
        # Handle empty param grid
        if not param_grid:
            return self._evaluate_default_only(
                symbol, strategy_name, default_params,
                train_features, val_features, scoring_fn
            )
        
        # Create parameter space
        param_space = ParameterSpace(param_grid, default_params)
        search_space_size = param_space.get_search_space_size()
        
        # Adjust n_trials based on search space
        n_trials = min(self.config.n_trials, search_space_size)
        
        logger.info(f"[{symbol}] Optuna TPE optimization for {strategy_name}: "
                   f"search_space={search_space_size}, trials={n_trials}")
        
        # Track best results
        best_result = {
            'params': default_params.copy(),
            'score': float('-inf'),
            'train_metrics': {},
            'val_metrics': {},
        }
        
        # Trial counter and early rejection counter for logging
        trial_count = [0]
        early_rejections = [0]
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function with drawdown-based early rejection."""
            trial_count[0] += 1
            
            # Suggest parameters
            params = param_space.suggest(trial)
            
            try:
                # Create strategy
                strategy = self.strategy_registry.get(strategy_name, **params)
                
                # Evaluate on training data
                signals = strategy.generate_signals(train_features, symbol)
                if signals.abs().sum() == 0:
                    return self.config.penalty_no_trades
                train_metrics = self.backtester.run(
                    train_features, signals, symbol, strategy
                )
                
                # EARLY REJECTION 1: Minimum trades
                if train_metrics.get('total_trades', 0) < min_trades:
                    return self.config.penalty_no_trades

                progress = trial_count[0] / max(n_trials, 1)
                if self.config.progressive_rejection and progress < self.config.progressive_warmup_frac:
                    dd_train_mult = self.config.train_dd_multiplier * self.config.progressive_warmup_multiplier
                    dd_val_mult = self.config.val_dd_multiplier * self.config.progressive_warmup_multiplier
                else:
                    dd_train_mult = self.config.train_dd_multiplier
                    dd_val_mult = self.config.val_dd_multiplier
                
                # EARLY REJECTION 2: Training drawdown
                train_dd = train_metrics.get('max_drawdown_pct', 100.0)
                if train_dd > max_drawdown_pct * dd_train_mult:
                    early_rejections[0] += 1
                    return self.config.penalty_dd_rejection
                
                # CONDITIONAL VALIDATION: Only run if training passed
                if val_features is not None and len(val_features) >= 50:
                    val_signals = strategy.generate_signals(val_features, symbol)
                    val_metrics = self.backtester.run(
                        val_features, val_signals, symbol, strategy
                    )
                    
                    # EARLY REJECTION 3: Validation drawdown
                    val_dd = val_metrics.get('max_drawdown_pct', 100.0)
                    if val_dd > max_drawdown_pct * dd_val_mult:
                        early_rejections[0] += 1
                        return self.config.penalty_dd_rejection
                else:
                    val_metrics = self._empty_val_metrics()
                
                # Compute score
                score = scoring_fn(train_metrics, val_metrics)
                
                # Update best if improved (only if passes DD check)
                if score > best_result['score']:
                    best_result['score'] = score
                    best_result['params'] = self._clean_params(params)
                    best_result['train_metrics'] = train_metrics
                    best_result['val_metrics'] = val_metrics
                
                # Log progress
                if trial_count[0] % self.config.log_interval == 0:
                    logger.debug(f"[{symbol}] Trial {trial_count[0]}/{n_trials}: "
                               f"score={score:.4f}, best={best_result['score']:.4f}, "
                               f"dd_rejects={early_rejections[0]}")
                
                return score
                
            except Exception as e:
                logger.debug(f"[{symbol}] Trial {trial_count[0]} failed: {e}")
                return -1000.0
        
        # Create sampler
        sampler = TPESampler(
            seed=self.config.seed,
            n_startup_trials=min(self.config.n_startup_trials, n_trials // 2),
            multivariate=self.config.multivariate,
        )
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.timeout_seconds,
            show_progress_bar=False,
            catch=(Exception,),
        )
        
        # Compile statistics
        elapsed = time.time() - start_time
        stats = self._compile_stats(study, elapsed)
        
        logger.info(f"[{symbol}] Optimization complete: {stats}")
        
        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['score'],
            train_metrics=best_result['train_metrics'],
            val_metrics=best_result['val_metrics'],
            stats=stats,
        )
    
    def optimize_for_regimes(self,
                             symbol: str,
                             timeframe: str,
                             strategy_name: str,
                             param_grid: Dict[str, List],
                             train_features: pd.DataFrame,
                             val_features: Optional[pd.DataFrame],
                             regimes: List[str],
                             bucket_trades_fn: Callable,
                             compute_score_fn: Callable,
                             min_train_trades: int = 25,
                             max_drawdown_pct: float = 25.0) -> Dict[str, Dict[str, Any]]:
        """
        Optimize strategy parameters for multiple regimes simultaneously.
        
        This is more efficient than per-regime optimization because:
        1. Each backtest produces trades for all regimes
        2. TPE learns parameter-regime relationships
        3. Early rejection of high-drawdown candidates (saves compute)
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe being optimized
            strategy_name: Strategy name
            param_grid: Parameter grid
            train_features: Training data with REGIME_LIVE column
            val_features: Validation data
            regimes: List of regime names
            bucket_trades_fn: Function to bucket trades by regime
            compute_score_fn: Function to compute regime score
            min_train_trades: Minimum trades per regime
            max_drawdown_pct: Maximum allowed drawdown (early rejection threshold)
            
        Returns:
            Dict mapping regime -> best candidate dict
        """
        if not OPTUNA_AVAILABLE:
            return self._fallback_regime_search(
                symbol, timeframe, strategy_name, param_grid,
                train_features, val_features, regimes,
                bucket_trades_fn, compute_score_fn, min_train_trades,
                max_drawdown_pct
            )
        
        start_time = time.time()
        
        # Get default params
        base_strategy = self.strategy_registry.get(strategy_name)
        default_params = base_strategy.get_params()
        
        if not param_grid:
            return {}
        
        # Create parameter space
        param_space = ParameterSpace(param_grid, default_params)
        search_space_size = param_space.get_search_space_size()
        n_trials = min(self.config.n_trials, search_space_size)
        
        # Track best per regime
        best_by_regime: Dict[str, Tuple[float, Dict[str, Any]]] = {
            r: (float('-inf'), None) for r in regimes
        }
        
        # Track early rejections for logging
        early_rejections = [0]
        trial_count = [0]
        
        def objective(trial: optuna.Trial) -> float:
            """Multi-regime objective with drawdown-based early rejection."""
            trial_count[0] += 1
            params = param_space.suggest(trial)
            
            try:
                strategy = self.strategy_registry.get(strategy_name, **params)
                
                # Backtest on training data
                signals = strategy.generate_signals(train_features, symbol)
                if signals.abs().sum() == 0:
                    return self.config.penalty_no_trades
                train_result = self.backtester.run(
                    train_features, signals, symbol, strategy
                )
                
                # ===== EARLY REJECTION 1: Minimum trades =====
                if train_result.get('total_trades', 0) < min_train_trades // 2:
                    return self.config.penalty_no_trades

                progress = trial_count[0] / max(n_trials, 1)
                if self.config.progressive_rejection and progress < self.config.progressive_warmup_frac:
                    dd_train_mult = self.config.train_dd_multiplier * self.config.progressive_warmup_multiplier
                    dd_val_mult = self.config.val_dd_multiplier * self.config.progressive_warmup_multiplier
                else:
                    dd_train_mult = self.config.train_dd_multiplier
                    dd_val_mult = self.config.val_dd_multiplier
                
                # ===== EARLY REJECTION 2: Training drawdown =====
                # If overall training DD exceeds threshold, skip validation entirely
                train_dd = train_result.get('max_drawdown_pct', 100.0)
                if train_dd > max_drawdown_pct * dd_train_mult:
                    early_rejections[0] += 1
                    return self.config.penalty_dd_rejection
                
                # Bucket trades by regime
                train_trades = train_result.get('trades', [])
                train_regime_metrics = bucket_trades_fn(train_trades, train_features)
                
                # ===== CONDITIONAL VALIDATION: Only if training passed DD check =====
                val_regime_metrics = {}
                if val_features is not None and len(val_features) >= 50:
                    val_signals = strategy.generate_signals(val_features, symbol)
                    val_result = self.backtester.run(
                        val_features, val_signals, symbol, strategy
                    )
                    
                    # Early rejection if validation DD too high
                    val_dd = val_result.get('max_drawdown_pct', 100.0)
                    if val_dd > max_drawdown_pct * dd_val_mult:
                        early_rejections[0] += 1
                        return self.config.penalty_dd_rejection
                    
                    val_trades = val_result.get('trades', [])
                    val_regime_metrics = bucket_trades_fn(val_trades, val_features)
                
                # Score each regime and track best
                regime_scores = []
                clean_params = self._clean_params(params)
                
                for regime in regimes:
                    train_m = train_regime_metrics.get(regime, {})
                    val_m = val_regime_metrics.get(regime, {})
                    
                    # ===== EARLY REJECTION 3: Per-regime minimum trades =====
                    if train_m.get('total_trades', 0) < min_train_trades:
                        continue
                    
                    # ===== EARLY REJECTION 4: Per-regime drawdown =====
                    regime_train_dd = train_m.get('max_drawdown_pct', 100.0)
                    regime_val_dd = val_m.get('max_drawdown_pct', 100.0)
                    
                    if regime_val_dd > max_drawdown_pct * dd_val_mult:
                        continue  # Skip this regime but continue with others
                    if regime_train_dd > max_drawdown_pct * dd_train_mult:
                        continue
                    
                    if self.config.use_validation_in_objective:
                        score = compute_score_fn(train_m, val_m)
                    elif self.config.objective_blend_enabled and val_m.get('total_trades', 0) >= min_train_trades:
                        train_score = compute_score_fn(train_m, {})
                        gen_score = compute_score_fn(train_m, val_m)
                        score = (self.config.objective_train_weight * train_score +
                                 self.config.objective_val_weight * gen_score)
                    else:
                        score = compute_score_fn(train_m, {})
                    regime_scores.append(score)
                    
                    # Update best for this regime (only if passes DD check)
                    if score > best_by_regime[regime][0]:
                        best_by_regime[regime] = (score, {
                            'strategy': strategy,
                            'strategy_name': strategy_name,
                            'params': clean_params,
                            'train_result': train_result,
                            'train_regime_metrics': train_regime_metrics,
                            'val_regime_metrics': val_regime_metrics,
                            'is_tuned': True,
                        })
                
                # Return max score for TPE guidance
                return max(regime_scores) if regime_scores else -1000.0
                
            except Exception as e:
                logger.debug(f"[{symbol}] [{timeframe}] Trial failed: {e}")
                return -1000.0
        
        # Create and run study
        sampler = TPESampler(
            seed=self.config.seed,
            n_startup_trials=min(self.config.n_startup_trials, n_trials // 2),
            multivariate=self.config.multivariate,
        )
        
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.timeout_seconds,
            show_progress_bar=False,
            catch=(Exception,),
        )
        
        elapsed = time.time() - start_time
        stats = self._compile_stats(study, elapsed)
        stats.early_dd_rejections = early_rejections[0]
        
        logger.debug(f"[{symbol}] [{timeframe}] {strategy_name} regime tuning: {stats}")
        
        # Return best candidates per regime
        results = {}
        for regime, (score, candidate) in best_by_regime.items():
            if candidate is not None:
                results[regime] = candidate
        
        return results
    
    def _evaluate_default_only(self,
                               symbol: str,
                               strategy_name: str,
                               default_params: Dict[str, Any],
                               train_features: pd.DataFrame,
                               val_features: Optional[pd.DataFrame],
                               scoring_fn: Callable) -> OptimizationResult:
        """Evaluate strategy with default parameters only."""
        start_time = time.time()
        
        strategy = self.strategy_registry.get(strategy_name, **default_params)
        
        signals = strategy.generate_signals(train_features, symbol)
        train_metrics = self.backtester.run(train_features, signals, symbol, strategy)
        
        if val_features is not None and len(val_features) >= 50:
            val_signals = strategy.generate_signals(val_features, symbol)
            val_metrics = self.backtester.run(val_features, val_signals, symbol, strategy)
        else:
            val_metrics = self._empty_val_metrics()
        
        score = scoring_fn(train_metrics, val_metrics)
        
        stats = OptimizationStats(
            n_trials=1,
            n_completed=1,
            best_score=score,
            optimization_time_sec=time.time() - start_time,
            method="default_params",
        )
        
        return OptimizationResult(
            best_params=self._clean_params(default_params),
            best_score=score,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            stats=stats,
        )
    
    def _fallback_random_search(self,
                                symbol: str,
                                strategy_name: str,
                                param_grid: Dict[str, List],
                                train_features: pd.DataFrame,
                                val_features: Optional[pd.DataFrame],
                                scoring_fn: Callable,
                                min_trades: int,
                                max_drawdown_pct: float = 30.0) -> OptimizationResult:
        """Fallback to random search when Optuna unavailable (with drawdown early rejection)."""
        from itertools import product
        
        start_time = time.time()
        
        base_strategy = self.strategy_registry.get(strategy_name)
        default_params = base_strategy.get_params()
        
        if not param_grid:
            return self._evaluate_default_only(
                symbol, strategy_name, default_params,
                train_features, val_features, scoring_fn
            )
        
        # Generate combinations with Latin Hypercube sampling
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(product(*param_values))
        
        n_trials = min(self.config.n_trials, len(all_combos))
        if len(all_combos) > n_trials:
            np.random.seed(self.config.seed)
            indices = np.random.choice(len(all_combos), n_trials, replace=False)
            combos = [all_combos[i] for i in indices]
        else:
            combos = all_combos
        
        best_score = float('-inf')
        best_params = default_params.copy()
        best_train_metrics = {}
        best_val_metrics = {}
        completed = 0
        early_rejections = 0
        
        for idx, combo in enumerate(combos):
            params = {**default_params, **dict(zip(param_names, combo))}

            progress = (idx + 1) / max(len(combos), 1)
            if self.config.progressive_rejection and progress < self.config.progressive_warmup_frac:
                dd_train_mult = self.config.train_dd_multiplier * self.config.progressive_warmup_multiplier
                dd_val_mult = self.config.val_dd_multiplier * self.config.progressive_warmup_multiplier
            else:
                dd_train_mult = self.config.train_dd_multiplier
                dd_val_mult = self.config.val_dd_multiplier
            
            try:
                strategy = self.strategy_registry.get(strategy_name, **params)
                
                signals = strategy.generate_signals(train_features, symbol)
                train_metrics = self.backtester.run(
                    train_features, signals, symbol, strategy
                )
                
                # EARLY REJECTION 1: Minimum trades
                if train_metrics.get('total_trades', 0) < min_trades:
                    continue
                
                # EARLY REJECTION 2: Training drawdown
                train_dd = train_metrics.get('max_drawdown_pct', 100.0)
                if train_dd > max_drawdown_pct * dd_train_mult:
                    early_rejections += 1
                    continue
                
                # CONDITIONAL VALIDATION
                if val_features is not None and len(val_features) >= 50:
                    val_signals = strategy.generate_signals(val_features, symbol)
                    val_metrics = self.backtester.run(
                        val_features, val_signals, symbol, strategy
                    )
                    
                    # EARLY REJECTION 3: Validation drawdown
                    val_dd = val_metrics.get('max_drawdown_pct', 100.0)
                    if val_dd > max_drawdown_pct * dd_val_mult:
                        early_rejections += 1
                        continue
                else:
                    val_metrics = self._empty_val_metrics()
                
                score = scoring_fn(train_metrics, val_metrics)
                completed += 1
                
                if score > best_score:
                    best_score = score
                    best_params = self._clean_params(params)
                    best_train_metrics = train_metrics
                    best_val_metrics = val_metrics
                    
            except Exception:
                continue
        
        if early_rejections > 0:
            logger.debug(f"[{symbol}] Random search: {early_rejections} early DD rejections")
        
        elapsed = time.time() - start_time
        
        stats = OptimizationStats(
            n_trials=len(combos),
            n_completed=completed,
            n_failed=len(combos) - completed,
            best_score=best_score,
            optimization_time_sec=elapsed,
            method="random_search_fallback",
        )
        
        logger.info(f"[{symbol}] Random search fallback: {stats}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            train_metrics=best_train_metrics,
            val_metrics=best_val_metrics,
            stats=stats,
        )
    
    def _fallback_regime_search(self,
                                symbol: str,
                                timeframe: str,
                                strategy_name: str,
                                param_grid: Dict[str, List],
                                train_features: pd.DataFrame,
                                val_features: Optional[pd.DataFrame],
                                regimes: List[str],
                                bucket_trades_fn: Callable,
                                compute_score_fn: Callable,
                                min_train_trades: int,
                                max_drawdown_pct: float = 25.0) -> Dict[str, Dict[str, Any]]:
        """Fallback regime search when Optuna unavailable (with drawdown-based early rejection)."""
        from itertools import product
        
        base_strategy = self.strategy_registry.get(strategy_name)
        default_params = base_strategy.get_params()
        
        if not param_grid:
            return {}
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(product(*param_values))
        
        n_trials = min(self.config.n_trials, len(all_combos))
        if len(all_combos) > n_trials:
            np.random.seed(self.config.seed)
            indices = np.random.choice(len(all_combos), n_trials, replace=False)
            combos = [all_combos[i] for i in indices]
        else:
            combos = all_combos
        
        best_by_regime = {r: (float('-inf'), None) for r in regimes}
        early_rejections = 0
        
        for idx, combo in enumerate(combos):
            params = {**default_params, **dict(zip(param_names, combo))}

            progress = (idx + 1) / max(len(combos), 1)
            if self.config.progressive_rejection and progress < self.config.progressive_warmup_frac:
                dd_train_mult = self.config.train_dd_multiplier * self.config.progressive_warmup_multiplier
                dd_val_mult = self.config.val_dd_multiplier * self.config.progressive_warmup_multiplier
            else:
                dd_train_mult = self.config.train_dd_multiplier
                dd_val_mult = self.config.val_dd_multiplier
            
            try:
                strategy = self.strategy_registry.get(strategy_name, **params)
                
                signals = strategy.generate_signals(train_features, symbol)
                train_result = self.backtester.run(
                    train_features, signals, symbol, strategy
                )
                
                # EARLY REJECTION 1: Minimum trades
                if train_result.get('total_trades', 0) < min_train_trades // 2:
                    continue
                
                # EARLY REJECTION 2: Training drawdown
                train_dd = train_result.get('max_drawdown_pct', 100.0)
                if train_dd > max_drawdown_pct * dd_train_mult:
                    early_rejections += 1
                    continue
                
                train_trades = train_result.get('trades', [])
                train_regime_metrics = bucket_trades_fn(train_trades, train_features)
                
                # CONDITIONAL VALIDATION: Only if training passed DD check
                val_regime_metrics = {}
                if val_features is not None and len(val_features) >= 50:
                    val_signals = strategy.generate_signals(val_features, symbol)
                    val_result = self.backtester.run(
                        val_features, val_signals, symbol, strategy
                    )
                    
                    # Early rejection if validation DD too high
                    val_dd = val_result.get('max_drawdown_pct', 100.0)
                    if val_dd > max_drawdown_pct * dd_val_mult:
                        early_rejections += 1
                        continue
                    
                    val_trades = val_result.get('trades', [])
                    val_regime_metrics = bucket_trades_fn(val_trades, val_features)
                
                clean_params = self._clean_params(params)
                
                for regime in regimes:
                    train_m = train_regime_metrics.get(regime, {})
                    val_m = val_regime_metrics.get(regime, {})
                    
                    # EARLY REJECTION 3: Per-regime minimum trades
                    if train_m.get('total_trades', 0) < min_train_trades:
                        continue
                    
                    # EARLY REJECTION 4: Per-regime drawdown
                    regime_train_dd = train_m.get('max_drawdown_pct', 100.0)
                    regime_val_dd = val_m.get('max_drawdown_pct', 100.0)
                    
                    if regime_val_dd > max_drawdown_pct * dd_val_mult:
                        continue
                    if regime_train_dd > max_drawdown_pct * dd_train_mult:
                        continue
                    
                    if self.config.use_validation_in_objective:
                        score = compute_score_fn(train_m, val_m)
                    elif self.config.objective_blend_enabled and val_m.get('total_trades', 0) >= min_train_trades:
                        train_score = compute_score_fn(train_m, {})
                        gen_score = compute_score_fn(train_m, val_m)
                        score = (self.config.objective_train_weight * train_score +
                                 self.config.objective_val_weight * gen_score)
                    else:
                        score = compute_score_fn(train_m, {})

                    if score > best_by_regime[regime][0]:
                        best_by_regime[regime] = (score, {
                            'strategy': strategy,
                            'strategy_name': strategy_name,
                            'params': clean_params,
                            'train_result': train_result,
                            'train_regime_metrics': train_regime_metrics,
                            'val_regime_metrics': val_regime_metrics,
                            'is_tuned': True,
                        })
                        
            except Exception:
                continue
        
        if early_rejections > 0:
            logger.debug(f"[{symbol}] [{timeframe}] Grid search: {early_rejections} early DD rejections")
        
        results = {}
        for regime, (score, candidate) in best_by_regime.items():
            if candidate is not None:
                results[regime] = candidate
        
        return results
    
    def _compile_stats(self, study: 'optuna.Study', elapsed: float) -> OptimizationStats:
        """Compile optimization statistics from Optuna study."""
        trials = study.trials
        
        n_completed = len([t for t in trials if t.state == TrialState.COMPLETE])
        n_pruned = len([t for t in trials if t.state == TrialState.PRUNED])
        n_failed = len([t for t in trials if t.state == TrialState.FAIL])
        
        best_score = study.best_value if study.best_trial else float('-inf')
        
        return OptimizationStats(
            n_trials=len(trials),
            n_completed=n_completed,
            n_pruned=n_pruned,
            n_failed=n_failed,
            best_score=best_score,
            optimization_time_sec=elapsed,
            method="optuna_tpe",
        )
    
    def _clean_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clean parameters for JSON serialization."""
        clean = {}
        for k, v in params.items():
            if hasattr(v, 'item'):
                v = v.item()
            elif isinstance(v, (np.integer, np.floating)):
                v = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            clean[k] = v
        return clean
    
    def _empty_val_metrics(self) -> Dict[str, Any]:
        """Return empty validation metrics."""
        return {
            'total_trades': 0,
            'total_return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_optuna_available() -> bool:
    """Check if Optuna is available."""
    return OPTUNA_AVAILABLE


def get_optimization_method() -> str:
    """Get current optimization method string."""
    if OPTUNA_AVAILABLE:
        return "Optuna TPE"
    return "Random Search (fallback)"


def create_optimizer(config, backtester, scorer, strategy_registry) -> OptunaTPEOptimizer:
    """
    Factory function to create optimizer with proper configuration.
    
    Args:
        config: PipelineConfig
        backtester: Backtester instance
        scorer: StrategyScorer instance
        strategy_registry: StrategyRegistry class
        
    Returns:
        Configured OptunaTPEOptimizer
    """
    optuna_config = OptunaConfig.from_pipeline_config(config)
    return OptunaTPEOptimizer(optuna_config, backtester, scorer, strategy_registry)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'OptunaConfig',
    'OptunaTPEOptimizer',
    'OptimizationResult',
    'OptimizationStats',
    'ParameterSpace',
    'is_optuna_available',
    'get_optimization_method',
    'create_optimizer',
    'OPTUNA_AVAILABLE',
]
