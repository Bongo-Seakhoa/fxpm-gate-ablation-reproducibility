"""
FX Portfolio Manager - Pipeline Module
=======================================

Main pipeline for the Portfolio Manager that handles:
- Strategy selection: Test all strategies, select best for each symbol
- Timeframe selection: Evaluate multiple timeframes, select optimal
- Hyperparameter optimization: Optuna TPE-based Bayesian optimization
- Retrain period selection: Determine optimal retraining frequency
- Validation: Out-of-sample testing with robustness checks
- Continuous loop: Retrain when period expires

Version: 3.1 (Portfolio Manager with Optuna TPE Optimization)
"""

import json
import logging
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from dataclasses import dataclass, field, fields

import pandas as pd
import numpy as np

from pm_core import (
    PipelineConfig,
    DataLoader,
    FeatureComputer,
    DataSplitter,
    Backtester,
    StrategyScorer,
    Timer,
    get_instrument_spec,
    InstrumentSpec
)
from pm_strategies import StrategyRegistry, BaseStrategy
from pm_position import PositionConfig, PositionManager
from pm_regime import load_regime_params

# Import Optuna TPE optimizer
try:
    from pm_optuna import (
        OptunaTPEOptimizer,
        OptunaConfig,
        OptimizationResult,
        OptimizationStats,
        is_optuna_available,
        get_optimization_method,
        create_optimizer,
        OPTUNA_AVAILABLE
    )
except ImportError:
    OPTUNA_AVAILABLE = False
    OptunaTPEOptimizer = None
    OptunaConfig = None
    OptimizationResult = None
    def is_optuna_available():
        return False
    def get_optimization_method():
        return "Grid Search (fallback)"
    def create_optimizer(*args, **kwargs):
        return None

# Configure module logger
logger = logging.getLogger(__name__)


def _pipeline_config_to_dict(config: PipelineConfig) -> Dict[str, Any]:
    """Serialize PipelineConfig to a plain dict for multiprocessing."""
    cfg = {f.name: getattr(config, f.name) for f in fields(PipelineConfig)}
    # Normalize Paths to strings for pickling and re-init
    if isinstance(cfg.get("data_dir"), Path):
        cfg["data_dir"] = str(cfg["data_dir"])
    if isinstance(cfg.get("output_dir"), Path):
        cfg["output_dir"] = str(cfg["output_dir"])
    return cfg


def _optimize_symbol_worker(config_dict: Dict[str, Any], symbol: str) -> Tuple[str, 'PipelineResult']:
    """Worker for parallel optimization."""
    try:
        cfg = PipelineConfig(**config_dict)
        pipeline = OptimizationPipeline(cfg)
        result = pipeline.run_for_symbol(symbol)
        return symbol, result
    except Exception as exc:
        return symbol, PipelineResult(symbol=symbol, success=False, error_message=str(exc))


# =============================================================================
# CONFIG LEDGER - STATEFUL OPTIMIZATION PERSISTENCE
# =============================================================================

class ConfigLedger:
    """
    Stateful optimization ledger for pm_configs.json.
    
    Provides:
    - Atomic writes (temp file + rename pattern)
    - Incremental persistence (save after each symbol)
    - Skip valid configs (re-optimize only if expired/missing/invalid)
    - Failure safety (never lose configs on errors)
    
    Usage:
        ledger = ConfigLedger(filepath="pm_configs.json")
        ledger.load()  # Load existing configs
        
        for symbol in symbols:
            if not ledger.should_optimize(symbol, overwrite=False):
                continue  # Skip valid
            
            # Run optimization...
            if success:
                ledger.update_symbol(symbol, config)  # Saves atomically
    """
    
    # Version for future migration support
    LEDGER_VERSION = "1.0"
    
    def __init__(self, filepath: str = "pm_configs.json"):
        """
        Initialize config ledger.
        
        Args:
            filepath: Path to config JSON file
        """
        self.filepath = Path(filepath)
        self.configs: Dict[str, SymbolConfig] = {}
        self._loaded = False
    
    def load(self) -> int:
        """
        Load existing configs from file.
        
        Returns:
            Number of configs loaded
            
        Raises:
            RuntimeError: If file is corrupted JSON
        """
        if not self.filepath.exists():
            logger.info(f"Config file not found: {self.filepath} (starting fresh)")
            self.configs = {}
            self._loaded = True
            return 0
        
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.configs = {}
            for symbol, config_dict in data.items():
                try:
                    self.configs[symbol] = SymbolConfig.from_dict(config_dict)
                except Exception as e:
                    logger.warning(f"Could not parse config for {symbol}: {e}")
            
            self._loaded = True
            logger.info(f"Loaded {len(self.configs)} existing configs from {self.filepath}")
            return len(self.configs)
            
        except json.JSONDecodeError as e:
            # Corrupted file - fail fast with clear message
            raise RuntimeError(
                f"Corrupted JSON in {self.filepath}: {e}\n"
                f"Please fix or remove the file manually, or backup to .corrupt and start fresh."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.filepath}: {e}")
    
    def _atomic_save(self):
        """
        Save configs atomically using temp file + rename pattern.
        
        This ensures configs are never corrupted if the process is interrupted.
        """
        temp_path = self.filepath.with_suffix('.json.tmp')
        
        # Serialize all configs
        data = {s: c.to_dict() for s, c in self.configs.items()}
        
        try:
            # Write to temp file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                f.flush()
                # fsync for durability (optional but recommended)
                import os
                os.fsync(f.fileno())
            
            # Atomic rename (on POSIX systems)
            temp_path.replace(self.filepath)
            
        except Exception as e:
            # Clean up temp file on failure
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise RuntimeError(f"Failed to save configs: {e}")
    
    def update_symbol(self, symbol: str, config: 'SymbolConfig') -> bool:
        """
        Update a single symbol's config and save atomically.
        
        This is the core incremental persistence method.
        
        Args:
            symbol: Symbol name
            config: SymbolConfig to save
            
        Returns:
            True if saved successfully
        """
        if not self._loaded:
            self.load()
        
        self.configs[symbol] = config
        
        try:
            self._atomic_save()
            logger.debug(f"Saved {symbol} to {self.filepath} (atomic)")
            return True
        except Exception as e:
            logger.error(f"Failed to save {symbol}: {e}")
            return False
    
    def get_config(self, symbol: str) -> Optional['SymbolConfig']:
        """Get config for a symbol."""
        if not self._loaded:
            self.load()
        return self.configs.get(symbol)
    
    def has_valid_config(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if a symbol has a valid (non-expired) config.
        
        Returns:
            Tuple of (is_valid, reason_string)
        """
        if not self._loaded:
            self.load()
        
        if symbol not in self.configs:
            return False, "missing"
        
        config = self.configs[symbol]
        
        # Check validation status
        if not config.is_validated:
            return False, f"invalid ({config.validation_reason or 'not validated'})"
        
        # Check expiry
        if config.valid_until is None:
            return False, "no expiry date"
        
        now = datetime.now()
        if now >= config.valid_until:
            days_expired = (now - config.valid_until).days
            return False, f"expired {days_expired} days ago"
        
        # Valid!
        days_remaining = (config.valid_until - now).days
        return True, f"valid until {config.valid_until.strftime('%Y-%m-%d')} ({days_remaining} days remaining)"
    
    def should_optimize(self,
                        symbol: str,
                        overwrite: bool = False,
                        current_regime_version: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Determine if a symbol should be optimized.
        
        Args:
            symbol: Symbol to check
            overwrite: If True, always optimize (ignore validity)
            
        Returns:
            Tuple of (should_optimize, reason_string)
        """
        if overwrite:
            return True, "overwrite enabled"
        
        is_valid, reason = self.has_valid_config(symbol)
        
        if is_valid:
            if current_regime_version and symbol in self.configs:
                stored = getattr(self.configs[symbol], "regime_detection_version", None)
                if stored and stored != current_regime_version:
                    return True, "regime detection params changed"
            return False, reason  # Skip - config is valid

        return True, reason  # Optimize - config is missing/expired/invalid
    
    def get_symbols_to_optimize(self,
                                symbols: List[str],
                                overwrite: bool = False,
                                current_regime_versions: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[List[str], List[str]]:
        """
        Partition symbols into those needing optimization and those to skip.
        
        Args:
            symbols: All symbols to consider
            overwrite: If True, optimize all
            
        Returns:
            Tuple of (symbols_to_optimize, symbols_to_skip)
        """
        if not self._loaded:
            self.load()
        
        to_optimize = []
        to_skip = []
        
        for symbol in symbols:
            version = (current_regime_versions or {}).get(symbol)
            should_opt, reason = self.should_optimize(symbol, overwrite, current_regime_version=version)
            if should_opt:
                to_optimize.append(symbol)
                logger.info(f"OPTIMIZE {symbol}: {reason}")
            else:
                to_skip.append(symbol)
                logger.info(f"SKIP {symbol}: {reason}")
        
        return to_optimize, to_skip
    
    def get_all_configs(self) -> Dict[str, 'SymbolConfig']:
        """Get all loaded configs."""
        if not self._loaded:
            self.load()
        return self.configs.copy()
    
    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol's config (use with caution).
        
        Args:
            symbol: Symbol to remove
            
        Returns:
            True if removed and saved
        """
        if not self._loaded:
            self.load()
        
        if symbol not in self.configs:
            return False
        
        del self.configs[symbol]
        
        try:
            self._atomic_save()
            logger.info(f"Removed {symbol} from {self.filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove {symbol}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ledger statistics."""
        if not self._loaded:
            self.load()
        
        now = datetime.now()
        valid = 0
        expired = 0
        invalid = 0
        
        for config in self.configs.values():
            if not config.is_validated:
                invalid += 1
            elif config.valid_until and now >= config.valid_until:
                expired += 1
            else:
                valid += 1
        
        return {
            'total': len(self.configs),
            'valid': valid,
            'expired': expired,
            'invalid': invalid,
            'filepath': str(self.filepath)
        }


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RegimeConfig:
    """
    Configuration for a single (timeframe, regime) winner.
    
    Stores the best strategy+params for a specific regime on a specific timeframe.
    """
    strategy_name: str
    parameters: Dict[str, Any]
    quality_score: float  # Normalized 0-1 for cross-TF comparison
    
    # Performance metrics for this regime bucket
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Regime-specific stats
    regime_train_trades: int = 0
    regime_val_trades: int = 0
    
    # Timestamps
    trained_at: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    def is_no_trade_marker(self) -> bool:
        """Check if this config is a NO_TRADE marker (no valid strategy found)."""
        return self.strategy_name == "NO_TRADE" or self.strategy_name == ""
    
    def is_valid_for_live(
        self,
        min_pf: float = 1.0,
        min_return: float = 5.0,
        max_dd: float = 35.0,
        min_return_dd_ratio: float = 1.0,
    ) -> bool:
        """
        Check if this config passes the live quality gate.
        
        A config is valid for live trading if:
        - It's not a NO_TRADE marker
        - Validation profit_factor >= min_pf
        - Validation return >= min_return
        - Validation drawdown <= max_dd
        - Validation return/DD ratio >= min_return_dd_ratio
        
        Args:
            min_pf: Minimum validation profit factor (default 1.0)
            min_return: Minimum validation return % (default 5.0)
            max_dd: Maximum validation drawdown % (default 35.0)
            min_return_dd_ratio: Minimum validation return/DD ratio (default 1.0)
            
        Returns:
            True if this config is valid for live trading
        """
        if self.is_no_trade_marker():
            return False
        
        val_pf = self.val_metrics.get('profit_factor', 0.0)
        val_return = self.val_metrics.get('total_return_pct', -100.0)
        val_dd = self.val_metrics.get('max_drawdown_pct', 100.0)

        ratio = val_return / max(val_dd, 0.5)
        return (
            val_pf >= min_pf and
            val_return >= min_return and
            val_dd <= max_dd and
            ratio >= min_return_dd_ratio
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'strategy_name': self.strategy_name,
            'parameters': self.parameters,
            'quality_score': self.quality_score,
            'regime_train_trades': self.regime_train_trades,
            'regime_val_trades': self.regime_val_trades,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'train_metrics': {k: v for k, v in self.train_metrics.items() 
                            if k not in ['equity_curve', 'trades']},
            'val_metrics': {k: v for k, v in self.val_metrics.items() 
                          if k not in ['equity_curve', 'trades']},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegimeConfig':
        """Create from dictionary."""
        config = cls(
            strategy_name=data['strategy_name'],
            parameters=data['parameters'],
            quality_score=data.get('quality_score', 0.0),
            train_metrics=data.get('train_metrics', {}),
            val_metrics=data.get('val_metrics', {}),
            regime_train_trades=data.get('regime_train_trades', 0),
            regime_val_trades=data.get('regime_val_trades', 0),
        )
        
        if data.get('trained_at'):
            config.trained_at = datetime.fromisoformat(data['trained_at'])
        if data.get('valid_until'):
            config.valid_until = datetime.fromisoformat(data['valid_until'])
        
        return config


@dataclass
class SymbolConfig:
    """
    Optimized configuration for a single symbol with regime-aware strategy selection.
    
    Structure:
    - regime_configs: Dict[timeframe][regime] -> RegimeConfig (winners per tf/regime)
    - default_config: RegimeConfig (fallback when no regime winner exists)
    """
    symbol: str
    
    # Per-(timeframe, regime) winners
    # Structure: {"H1": {"TREND": RegimeConfig, "RANGE": RegimeConfig, ...}, ...}
    regime_configs: Dict[str, Dict[str, RegimeConfig]] = field(default_factory=dict)
    
    # Default fallback (best overall, used when no regime winner exists)
    default_config: Optional[RegimeConfig] = None
    
    # Legacy fields (populated from default_config for compatibility)
    strategy_name: str = ""
    timeframe: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    retrain_days: int = 30
    
    # Performance metrics (from default)
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    composite_score: float = 0.0
    robustness_ratio: float = 0.0
    
    # Validation status
    is_validated: bool = False
    validation_reason: str = ""

    # Regime detection version stamp for invalidation when params change.
    regime_detection_version: Optional[Dict[str, Any]] = None
    
    # Timestamps
    optimized_at: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    def has_regime_configs(self) -> bool:
        """Check if this config has regime configs."""
        return bool(self.regime_configs)
    
    def get_regime_config(self, timeframe: str, regime: str) -> Optional[RegimeConfig]:
        """Get config for a specific (timeframe, regime) pair."""
        if timeframe in self.regime_configs:
            return self.regime_configs[timeframe].get(regime)
        return None
    
    def get_available_timeframes(self) -> List[str]:
        """Get list of timeframes with regime configs."""
        return list(self.regime_configs.keys())
    
    def get_regimes_for_timeframe(self, timeframe: str) -> List[str]:
        """Get list of regimes configured for a timeframe."""
        if timeframe in self.regime_configs:
            return list(self.regime_configs[timeframe].keys())
        return []
    
    def count_regime_winners(self) -> int:
        """Count total number of regime winners across all timeframes."""
        return sum(len(regimes) for regimes in self.regime_configs.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'symbol': self.symbol,
            'strategy_name': self.strategy_name,
            'timeframe': self.timeframe,
            'parameters': self.parameters,
            'retrain_days': self.retrain_days,
            'composite_score': self.composite_score,
            'robustness_ratio': self.robustness_ratio,
            'is_validated': self.is_validated,
            'validation_reason': self.validation_reason,
            'train_return': float(self.train_metrics.get('total_return_pct', 0.0)) if self.train_metrics else 0.0,
            'train_sharpe': float(self.train_metrics.get('sharpe_ratio', 0.0)) if self.train_metrics else 0.0,
            'val_return': float(self.val_metrics.get('total_return_pct', 0.0)) if self.val_metrics else 0.0,
            'val_sharpe': float(self.val_metrics.get('sharpe_ratio', 0.0)) if self.val_metrics else 0.0,
            'optimized_at': self.optimized_at.isoformat() if self.optimized_at else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'train_metrics': {k: v for k, v in self.train_metrics.items() 
                            if k not in ['equity_curve', 'trades']},
            'val_metrics': {k: v for k, v in self.val_metrics.items() 
                          if k not in ['equity_curve', 'trades']},
        }

        if self.regime_detection_version:
            result['regime_detection_version'] = self.regime_detection_version
        
        # Add regime configs
        if self.regime_configs:
            result['regime_configs'] = {}
            for tf, regime_dict in self.regime_configs.items():
                result['regime_configs'][tf] = {}
                for regime, cfg in regime_dict.items():
                    result['regime_configs'][tf][regime] = cfg.to_dict()
        
        # Default config
        if self.default_config:
            result['default_config'] = self.default_config.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymbolConfig':
        """Create from dictionary."""
        config = cls(
            symbol=data['symbol'],
            strategy_name=data.get('strategy_name', ''),
            timeframe=data.get('timeframe', ''),
            parameters=data.get('parameters', {}),
            retrain_days=data.get('retrain_days', 30),
            composite_score=data.get('composite_score', 0.0),
            robustness_ratio=data.get('robustness_ratio', 0.0),
            is_validated=data.get('is_validated', False),
            validation_reason=data.get('validation_reason', ''),
            train_metrics=data.get('train_metrics', {}),
            val_metrics=data.get('val_metrics', {}),
        )
        
        if data.get('optimized_at'):
            config.optimized_at = datetime.fromisoformat(data['optimized_at'])
        if data.get('valid_until'):
            config.valid_until = datetime.fromisoformat(data['valid_until'])
        config.regime_detection_version = data.get('regime_detection_version')
        
        # Load regime configs
        if 'regime_configs' in data and data['regime_configs']:
            config.regime_configs = {}
            for tf, regime_dict in data['regime_configs'].items():
                config.regime_configs[tf] = {}
                for regime, cfg_data in regime_dict.items():
                    config.regime_configs[tf][regime] = RegimeConfig.from_dict(cfg_data)
        
        # Default config
        if 'default_config' in data and data['default_config']:
            config.default_config = RegimeConfig.from_dict(data['default_config'])
        
        return config


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    symbol: str
    success: bool
    config: Optional[SymbolConfig] = None
    error_message: str = ""
    duration_seconds: float = 0.0
    
    # Intermediate results
    strategies_tested: int = 0
    timeframes_tested: int = 0
    param_combos_tested: int = 0
    regime_winners: int = 0


# =============================================================================
# PIPELINE STAGES
# =============================================================================


class StrategySelector:
    """
    Strategy selector for each symbol.

    Tests all strategies on training data for each timeframe.
    In fx_backtester mode, selection is validation-aware: evaluate top-K candidates on validation.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.backtester = Backtester(config)
        self.scorer = StrategyScorer(config)

    def select_best(self,
                    symbol: str,
                    train_data_by_tf: Dict[str, pd.DataFrame],
                    strategies: List[BaseStrategy],
                    val_data_by_tf: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[Optional[str], Optional[str], Dict]:
            """
            Select best strategy and timeframe for a symbol.

            In fx_backtester mode, this is validation-aware without being expensive:
              1) score all (strategy,timeframe) candidates on TRAIN
              2) take top-K by train score
              3) evaluate those K candidates on VALIDATION
              4) pick the winner by a generalization-aware score (val-first + gap penalty + robustness boost)

            Args:
                symbol: Symbol name
                train_data_by_tf: Dict mapping timeframe -> training DataFrame
                strategies: List of strategy instances to test
                val_data_by_tf: Optional dict mapping timeframe -> validation DataFrame (recommended)

            Returns:
                (best_strategy_name, best_timeframe, best_metrics_train)
            """
            best_strategy: Optional[str] = None
            best_timeframe: Optional[str] = None
            best_metrics: Dict[str, Any] = {}
            best_final_score: float = -float("inf")

            logger.info(f"[{symbol}] Testing {len(strategies)} strategies x {len(train_data_by_tf)} timeframes")

            # Feature cache: compute once per timeframe for train/val
            train_feat_by_tf: Dict[str, pd.DataFrame] = {}
            val_feat_by_tf: Dict[str, pd.DataFrame] = {}

            # ------------------------------------------------------------
            # Pass 1: evaluate TRAIN only (cheap) and collect candidates
            # ------------------------------------------------------------
            candidates: List[Dict[str, Any]] = []
            for timeframe, train_data in train_data_by_tf.items():
                if train_data is None or len(train_data) < 100:
                    logger.warning(f"[{symbol}] Skipping {timeframe}: insufficient training data")
                    continue

                if timeframe not in train_feat_by_tf:
                    train_feat_by_tf[timeframe] = FeatureComputer.compute_all(train_data)
                train_features = train_feat_by_tf[timeframe]

                for strategy in strategies:
                    try:
                        signals = strategy.generate_signals(train_features, symbol)
                        metrics = self.backtester.run(
                            features=train_features,
                            signals=signals,
                            symbol=symbol,
                            strategy=strategy
                        )

                        if self.config.scoring_mode == "fx_backtester":
                            if metrics.get('total_trades', 0) < self.config.min_trades:
                                continue
                            train_score = float(self.scorer.score(metrics, purpose="selection"))
                            candidates.append({
                                "strategy_name": strategy.name,
                                "timeframe": timeframe,
                                "train_metrics": metrics,
                                "train_score": train_score
                            })
                        else:
                            # Original behavior: strict minimum criteria + composite
                            passes, _ = self.scorer.passes_minimum_criteria(metrics)
                            if not passes:
                                continue
                            train_score = float(self.scorer.calculate_composite_score(metrics))
                            candidates.append({
                                "strategy_name": strategy.name,
                                "timeframe": timeframe,
                                "train_metrics": metrics,
                                "train_score": train_score
                            })
                    except Exception as e:
                        logger.warning(f"  [{symbol}] Error testing {strategy.name} {timeframe}: {e}")
                        continue

            if not candidates:
                logger.warning(f"[{symbol}] No valid strategy candidates found")
                return None, None, {}

            # ------------------------------------------------------------
            # Pass 2: validation-aware selection (fx_backtester)
            # ------------------------------------------------------------
            if self.config.scoring_mode == "fx_backtester" and val_data_by_tf is not None:
                top_k = int(getattr(self.config, "fx_selection_top_k", 5))
                top_k = max(1, top_k)

                candidates.sort(key=lambda d: d["train_score"], reverse=True)
                shortlist = candidates[:top_k]

                for cand in shortlist:
                    tf = cand["timeframe"]
                    val_df = val_data_by_tf.get(tf) if val_data_by_tf else None
                    if val_df is None or len(val_df) < 50:
                        continue

                    if tf not in val_feat_by_tf:
                        val_feat_by_tf[tf] = FeatureComputer.compute_all(val_df)
                    val_features = val_feat_by_tf[tf]

                    try:
                        strat = StrategyRegistry.get(cand["strategy_name"])
                        signals = strat.generate_signals(val_features, symbol)
                        val_metrics = self.backtester.run(val_features, signals, symbol, strat)

                        # Enforce basic validation guardrails
                        val_trades = int(val_metrics.get("total_trades", 0))
                        val_dd = float(val_metrics.get("max_drawdown_pct", 100.0))
                        if val_trades < self.config.fx_val_min_trades:
                            continue
                        if val_dd >= self.config.fx_val_max_drawdown:
                            continue

                        final_score, train_score, val_score, rr = self.scorer.fx_generalization_score(
                            cand["train_metrics"], val_metrics, purpose="selection"
                        )

                        if final_score > best_final_score:
                            best_final_score = final_score
                            best_strategy = cand["strategy_name"]
                            best_timeframe = tf
                            best_metrics = cand["train_metrics"]

                    except Exception as e:
                        logger.debug(f"[{symbol}] Validation eval failed for {cand['strategy_name']} {tf}: {e}")
                        continue

                if best_strategy:
                    logger.info(f"[{symbol}] Selected: {best_strategy} @ {best_timeframe} "
                                f"(GenScore: {best_final_score:.1f})")
                    return best_strategy, best_timeframe, best_metrics

                # If validation-aware selection couldn't pick, fall back to best train
                best_train = max(candidates, key=lambda d: d["train_score"])
                logger.info(f"[{symbol}] Selected (train fallback): {best_train['strategy_name']} @ {best_train['timeframe']} "
                            f"(TrainScore: {best_train['train_score']:.1f})")
                return best_train["strategy_name"], best_train["timeframe"], best_train["train_metrics"]

            # ------------------------------------------------------------
            # Non-fx_backtester: choose best train score (original)
            # ------------------------------------------------------------
            best_train = max(candidates, key=lambda d: d["train_score"])
            logger.info(f"[{symbol}] Selected: {best_train['strategy_name']} @ {best_train['timeframe']} "
                        f"(Score: {best_train['train_score']:.1f})")
            return best_train["strategy_name"], best_train["timeframe"], best_train["train_metrics"]

class HyperparameterOptimizer:
    """
    Stage 2: Hyperparameter Optimization
    
    Uses Optuna TPE (Tree-structured Parzen Estimator) for efficient
    hyperparameter search. Falls back to grid search when Optuna unavailable.
    
    TPE is well-suited for:
    - Discrete/categorical parameter spaces (period lengths, multipliers)
    - Parameter constraints (fast_period < slow_period)
    - Mixed parameter types
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.backtester = Backtester(config)
        self.scorer = StrategyScorer(config)
        
        # Initialize Optuna optimizer if available
        if OPTUNA_AVAILABLE and OptunaTPEOptimizer is not None:
            self.optuna_optimizer = create_optimizer(
                config, self.backtester, self.scorer, StrategyRegistry
            )
            logger.debug(f"HyperparameterOptimizer using: {get_optimization_method()}")
        else:
            self.optuna_optimizer = None
            logger.debug("HyperparameterOptimizer using: Grid Search (Optuna not available)")
    
    def optimize(self,
                 symbol: str,
                 strategy_name: str,
                 train_features: pd.DataFrame,
                 val_features: pd.DataFrame) -> Tuple[Dict[str, Any], Dict, Dict]:
        """
        Optimize strategy parameters using Optuna TPE.
        
        Args:
            symbol: Symbol name
            strategy_name: Name of strategy to optimize
            train_features: Training data with features
            val_features: Validation data with features
            
        Returns:
            Tuple of (best_params, train_metrics, val_metrics)
        """
        # Get strategy and param grid
        strategy = StrategyRegistry.get(strategy_name)
        param_grid = strategy.get_param_grid()
        
        if not param_grid:
            # No params to optimize, use defaults
            logger.info(f"[{symbol}] No param grid for {strategy_name}, using defaults")
            
            signals = strategy.generate_signals(train_features, symbol)
            train_metrics = self.backtester.run(train_features, signals, symbol, strategy)
            
            if val_features is not None and len(val_features) >= 50:
                signals = strategy.generate_signals(val_features, symbol)
                val_metrics = self.backtester.run(val_features, signals, symbol, strategy)
            else:
                val_metrics = {'total_return_pct': 0.0, 'sharpe_ratio': 0.0,
                               'max_drawdown_pct': 0.0, 'total_trades': 0}
            
            return strategy.get_params(), train_metrics, val_metrics
        
        # Use Optuna TPE if available
        if self.optuna_optimizer is not None:
            return self._optimize_with_optuna(
                symbol, strategy_name, param_grid, train_features, val_features
            )
        
        # Fallback to grid search
        return self._optimize_grid_search(
            symbol, strategy_name, param_grid, train_features, val_features
        )
    
    def _optimize_with_optuna(self,
                              symbol: str,
                              strategy_name: str,
                              param_grid: Dict[str, List],
                              train_features: pd.DataFrame,
                              val_features: pd.DataFrame) -> Tuple[Dict[str, Any], Dict, Dict]:
        """Optimize using Optuna TPE."""
        
        # Define scoring function based on config mode
        def scoring_fn(train_metrics: Dict, val_metrics: Dict) -> float:
            if self.config.scoring_mode == "fx_backtester":
                # Use fx_generalization_score for validation-aware scoring
                if val_metrics.get('total_trades', 0) >= self.config.fx_val_min_trades:
                    # Check validation guardrails
                    val_dd = float(val_metrics.get('max_drawdown_pct', 100.0))
                    if val_dd >= self.config.fx_val_max_drawdown:
                        return -500.0  # Penalty for excessive drawdown
                    
                    final_score, _, _, _ = self.scorer.fx_generalization_score(
                        train_metrics, val_metrics, purpose="opt"
                    )
                    return final_score
                else:
                    # Not enough validation trades - use train with discount
                    train_score = self.scorer.score(train_metrics, purpose="opt")
                    return train_score * 0.7
            else:
                # Legacy pm_weighted mode
                train_score = self.scorer.calculate_composite_score(train_metrics)
                val_score = self.scorer.calculate_composite_score(val_metrics)
                return 0.4 * train_score + 0.6 * val_score
        
        # Get n_trials from config
        n_trials = getattr(self.config, 'max_param_combos', 100)
        
        # Update optimizer config
        if hasattr(self.optuna_optimizer, 'config'):
            self.optuna_optimizer.config.n_trials = n_trials
        
        # Run optimization
        result = self.optuna_optimizer.optimize(
            symbol=symbol,
            strategy_name=strategy_name,
            param_grid=param_grid,
            train_features=train_features,
            val_features=val_features,
            scoring_fn=scoring_fn,
            min_trades=self.config.fx_opt_min_trades if self.config.scoring_mode == "fx_backtester" 
                       else self.config.min_trades
        )
        
        logger.info(f"[{symbol}] Optuna optimization for {strategy_name}: "
                   f"{result.stats.n_trials} trials, best_score={result.best_score:.2f}")
        
        return result.best_params, result.train_metrics, result.val_metrics
    
    def _optimize_grid_search(self,
                              symbol: str,
                              strategy_name: str,
                              param_grid: Dict[str, List],
                              train_features: pd.DataFrame,
                              val_features: pd.DataFrame) -> Tuple[Dict[str, Any], Dict, Dict]:
        """Fallback grid search optimization."""
        strategy = StrategyRegistry.get(strategy_name)
        default_params = strategy.get_params()
        
        # Generate combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        # Limit combinations
        if len(combinations) > self.config.max_param_combos:
            np.random.seed(42)
            indices = np.random.choice(len(combinations), self.config.max_param_combos, replace=False)
            combinations = [combinations[i] for i in indices]
        
        logger.info(f"[{symbol}] Grid search {strategy_name}: {len(combinations)} combinations")
        
        # fx_backtester mode with validation
        if self.config.scoring_mode == "fx_backtester":
            candidates: List[Dict[str, Any]] = []
            
            for combo in combinations:
                params = dict(zip(param_names, combo))
                test_params = {**default_params, **params}
                test_strategy = StrategyRegistry.get(strategy_name, **test_params)
                
                try:
                    signals = test_strategy.generate_signals(train_features, symbol)
                    train_metrics = self.backtester.run(train_features, signals, symbol, test_strategy)
                    
                    if train_metrics.get('total_trades', 0) < self.config.fx_opt_min_trades:
                        continue
                    
                    train_score = float(self.scorer.score(train_metrics, purpose="opt"))
                    candidates.append({
                        "params": dict(test_params),
                        "train_metrics": train_metrics,
                        "train_score": train_score
                    })
                except Exception:
                    continue
            
            if not candidates:
                candidates.append({
                    "params": dict(default_params),
                    "train_metrics": {},
                    "train_score": -float("inf")
                })
            
            candidates.sort(key=lambda d: d["train_score"], reverse=True)
            top_k = max(1, int(getattr(self.config, "fx_opt_top_k", 5)))
            shortlist = candidates[:top_k]
            
            best_final = -float("inf")
            best_params = dict(default_params)
            best_train_metrics: Dict[str, Any] = {}
            best_val_metrics: Dict[str, Any] = {}
            
            for cand in shortlist:
                test_params = cand["params"]
                test_strategy = StrategyRegistry.get(strategy_name, **test_params)
                train_metrics = cand.get("train_metrics") or {}
                
                if not train_metrics:
                    try:
                        signals = test_strategy.generate_signals(train_features, symbol)
                        train_metrics = self.backtester.run(train_features, signals, symbol, test_strategy)
                    except Exception:
                        continue
                
                if val_features is None or len(val_features) < 50:
                    val_metrics = {'total_return_pct': 0.0, 'sharpe_ratio': 0.0,
                                   'max_drawdown_pct': 0.0, 'total_trades': 0}
                else:
                    try:
                        signals = test_strategy.generate_signals(val_features, symbol)
                        val_metrics = self.backtester.run(val_features, signals, symbol, test_strategy)
                    except Exception:
                        continue
                
                val_trades = int(val_metrics.get("total_trades", 0))
                val_dd = float(val_metrics.get("max_drawdown_pct", 100.0))
                if val_trades < self.config.fx_val_min_trades:
                    continue
                if val_dd >= self.config.fx_val_max_drawdown:
                    continue
                
                final_score, _, _, _ = self.scorer.fx_generalization_score(
                    train_metrics, val_metrics, purpose="opt"
                )
                
                if final_score > best_final:
                    best_final = final_score
                    best_params = dict(test_params)
                    best_train_metrics = train_metrics
                    best_val_metrics = val_metrics
            
            if not best_train_metrics:
                best_params = dict(shortlist[0]["params"])
                best_strategy = StrategyRegistry.get(strategy_name, **best_params)
                signals = best_strategy.generate_signals(train_features, symbol)
                best_train_metrics = self.backtester.run(train_features, signals, symbol, best_strategy)
                if val_features is not None and len(val_features) >= 50:
                    signals = best_strategy.generate_signals(val_features, symbol)
                    best_val_metrics = self.backtester.run(val_features, signals, symbol, best_strategy)
                else:
                    best_val_metrics = {'total_return_pct': 0.0, 'sharpe_ratio': 0.0,
                                        'max_drawdown_pct': 0.0, 'total_trades': 0}
            
            return self._clean_params(best_params), best_train_metrics, best_val_metrics
        
        # Legacy pm_weighted mode
        best_score = -float('inf')
        best_params = default_params
        best_train_metrics = {}
        best_val_metrics = {}
        
        for combo in combinations:
            params = dict(zip(param_names, combo))
            test_strategy = StrategyRegistry.get(strategy_name, **params)
            
            try:
                signals = test_strategy.generate_signals(train_features, symbol)
                train_metrics = self.backtester.run(train_features, signals, symbol, test_strategy)
                
                passes, _ = self.scorer.passes_minimum_criteria(train_metrics)
                if not passes:
                    continue
                
                signals = test_strategy.generate_signals(val_features, symbol)
                val_metrics = self.backtester.run(val_features, signals, symbol, test_strategy)
                
                train_score = self.scorer.calculate_composite_score(train_metrics)
                val_score = self.scorer.calculate_composite_score(val_metrics)
                combined_score = 0.4 * train_score + 0.6 * val_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_params = test_strategy.get_params()
                    best_train_metrics = train_metrics
                    best_val_metrics = val_metrics
                    
            except Exception as e:
                logger.debug(f"[{symbol}] Param combo failed: {params} - {e}")
                continue
        
        if not best_train_metrics:
            logger.warning(f"[{symbol}] No valid params found, using defaults")
            signals = strategy.generate_signals(train_features, symbol)
            best_train_metrics = self.backtester.run(train_features, signals, symbol, strategy)
            signals = strategy.generate_signals(val_features, symbol)
            best_val_metrics = self.backtester.run(val_features, signals, symbol, strategy)
        
        return self._clean_params(best_params), best_train_metrics, best_val_metrics
    
    def _clean_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clean parameters for JSON serialization."""
        clean = {}
        for k, v in params.items():
            if hasattr(v, "item"):
                v = v.item()
            elif isinstance(v, (np.integer, np.floating)):
                v = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, (datetime, pd.Timestamp)):
                v = v.isoformat() if hasattr(v, 'isoformat') else str(v)
            clean[k] = v
        return clean


class RetrainPeriodSelector:
    """
    Stage 3: Retrain Period Selection
    
    Determines optimal retraining frequency by testing
    different lookback windows.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.backtester = Backtester(config)
        self.scorer = StrategyScorer(config)
    
    def select_period(self,
                      symbol: str,
                      strategy_name: str,
                      params: Dict[str, Any],
                      full_data: pd.DataFrame,
                      timeframe: str) -> int:
        """
        Select optimal retrain period.
        
        Tests different lookback windows and selects the one
        that gives most consistent out-of-sample performance.
        
        OPTIMIZED: Computes features once on full data, then slices for each window.
        This provides 3-5x speedup over recomputing features for each window.
        
        Args:
            symbol: Symbol name
            strategy_name: Strategy name
            params: Strategy parameters
            full_data: Full dataset
            timeframe: Timeframe for bar count conversion
            
        Returns:
            Optimal retrain period in days
        """
        # Convert retrain periods to approximate bar counts
        bars_per_day = {
            'M5': 288,
            'M15': 96,
            'M30': 48,
            'H1': 24,
            'H4': 6,
            'D1': 1
        }
        
        bpd = bars_per_day.get(timeframe, 24)
        
        # OPTIMIZATION: Compute features once on full data, then slice
        # This is much faster than recomputing for each window
        full_features = FeatureComputer.compute_all(full_data, symbol=symbol, timeframe=timeframe)
        
        results = {}
        
        for retrain_days in self.config.retrain_periods:
            lookback_bars = retrain_days * bpd
            
            if lookback_bars > len(full_data) * 0.8:
                continue
            
            # Simulate walk-forward validation
            # Use last 30% of data for testing
            test_start = int(len(full_data) * 0.7)
            step_size = max(1, lookback_bars // 4)  # Overlap windows
            
            scores = []
            
            for start in range(test_start, len(full_data) - lookback_bars, step_size):
                train_end = start
                train_start = max(0, train_end - lookback_bars)
                test_end = min(len(full_data), start + step_size)
                
                if train_end - train_start < 50 or test_end - start < 10:
                    continue
                
                try:
                    # OPTIMIZATION: Slice from precomputed features instead of recomputing
                    train_features = full_features.iloc[train_start:train_end]
                    test_features = full_features.iloc[start:test_end]
                    
                    strategy = StrategyRegistry.get(strategy_name, **params)
                    
                    signals = strategy.generate_signals(test_features, symbol)
                    metrics = self.backtester.run(test_features, signals, symbol, strategy)
                    
                    score = self.scorer.score(metrics, purpose="selection")
                    # Keep window scoring consistent with the selection metric for the active mode
                    if metrics.get('total_trades', 0) >= self.config.min_trades and score is not None:
                        scores.append(score)
                
                except Exception:
                    continue
            
            if scores:
                avg_score = np.mean(scores)
                # Need at least 2 scores for meaningful std calculation
                std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
                consistency = avg_score / (std_score + 1) if std_score > 0 else avg_score
                results[retrain_days] = {
                    'avg_score': avg_score,
                    'std_score': std_score,
                    'consistency': consistency,
                    'windows': len(scores)
                }
        
        # Select period with best consistency
        if not results:
            best_period = 30  # Default
        else:
            best_period = max(results.keys(), key=lambda k: results[k]['consistency'])
        
        logger.info(f"[{symbol}] Selected retrain period: {best_period} days")
        
        return best_period


class Validator:
    """
    Stage 4: Validation
    
    Performs final validation checks:
    - Minimum criteria check on validation data
    - Robustness ratio calculation
    - Final approval for live trading
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.scorer = StrategyScorer(config)
    
    def validate(self,
                 train_metrics: Dict[str, Any],
                 val_metrics: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Validate strategy configuration.
        
        Args:
            train_metrics: Training period metrics
            val_metrics: Validation period metrics
            
        Returns:
            Tuple of (is_valid, reason, robustness_ratio)
        """
        # Backtester-aligned validation rules
        if self.config.scoring_mode == "fx_backtester":
            robustness = self.scorer.calculate_fx_score_robustness_ratio(train_metrics, val_metrics, purpose="selection")

            val_sharpe = float(val_metrics.get('sharpe_ratio', 0.0))
            val_dd = float(val_metrics.get('max_drawdown_pct', 100.0))
            val_trades = int(val_metrics.get('total_trades', 0))

            is_valid = (
                (robustness >= getattr(self.config, "fx_min_robustness_ratio", 0.85) or val_sharpe > self.config.fx_val_sharpe_override)
                and val_dd < self.config.fx_val_max_drawdown
                and val_trades >= self.config.fx_val_min_trades
            )

            if not is_valid:
                if val_trades < self.config.fx_val_min_trades:
                    return False, f"Validation failed: Insufficient trades: {val_trades} < {self.config.fx_val_min_trades}", robustness
                if val_dd >= self.config.fx_val_max_drawdown:
                    return False, f"Validation failed: High drawdown: {val_dd:.2f}% >= {self.config.fx_val_max_drawdown}%", robustness
                return False, f"Validation failed: Robustness {robustness:.2f} < {getattr(self.config, 'fx_min_robustness_ratio', 0.85)} and Sharpe {val_sharpe:.2f} <= {self.config.fx_val_sharpe_override}", robustness

            return True, "Validated successfully", robustness

        # Check validation metrics pass minimum criteria
        passes, reason = self.scorer.passes_minimum_criteria(val_metrics)
        if not passes:
            return False, f"Validation failed: {reason}", 0.0
        
        # Calculate robustness ratio
        robustness = self.scorer.calculate_robustness_ratio(train_metrics, val_metrics)
        
        # Check robustness
        if robustness < self.config.min_robustness:
            return False, f"Low robustness: {robustness:.2f} < {self.config.min_robustness}", robustness
        
        return True, "Validated successfully", robustness


# =============================================================================
# REGIME-AWARE OPTIMIZATION
# =============================================================================

class RegimeOptimizer:
    """
    Regime-aware strategy selection and optimization.
    
    Finds the best (strategy, params) for each (timeframe, regime) combination.
    All strategies compete for all regimes - winners are selected per (tf, regime).
    
    Flow:
    1. Quick screening with default params to identify top-K strategies per regime
    2. Hyperparameter tuning using Optuna TPE (simultaneous multi-regime optimization)
    3. Final selection: pick best tuned config per regime
    4. Validation: apply fx_backtester validation rules to each winner
    
    Trade bucketing uses REGIME_LIVE at entry bar for backtest/live parity.
    """
    
    # Regime types
    REGIMES = ['TREND', 'RANGE', 'BREAKOUT', 'CHOP']
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.backtester = Backtester(config)
        self.scorer = StrategyScorer(config)
        
        # Minimum trades per regime for validation
        self.min_train_trades = getattr(config, 'regime_min_train_trades', 25)
        self.min_val_trades = getattr(config, 'regime_min_val_trades', 10)
        
        # Hyperparameter tuning settings
        self.enable_hyperparam_tuning = getattr(config, 'regime_enable_hyperparam_tuning', True)
        self.hyperparam_top_k = getattr(config, 'regime_hyperparam_top_k', 3)  # Top K strategies to tune per regime
        self.hyperparam_max_combos = getattr(config, 'regime_hyperparam_max_combos', 30)  # Max param combos per strategy
        
        # Validation thresholds (from fx_backtester config)
        self.val_max_drawdown = getattr(config, 'fx_val_max_drawdown', 20.0)
        self.val_min_sharpe_override = getattr(config, 'fx_val_sharpe_override', 0.3)
        self.min_robustness_ratio = getattr(config, 'fx_min_robustness_ratio', 0.75)
        
        # Max candidates to attempt validation on (descent through scored list)
        self.regime_validation_top_k = getattr(config, 'regime_validation_top_k', 5)

        # ===== PROFITABILITY GATES (FIX #1) =====
        # These prevent storing losing strategies as "regime winners"
        self.min_val_profit_factor = getattr(config, 'regime_min_val_profit_factor', 1.0)
        self.min_val_return_pct = getattr(config, 'regime_min_val_return_pct', 5.0)
        self.allow_losing_winners = getattr(config, 'regime_allow_losing_winners', False)
        self.no_winner_marker = getattr(config, 'regime_no_winner_marker', 'NO_TRADE')

        # Return-to-DD efficiency gate (unconditional hard gate)
        _raw_ratio = float(getattr(config, 'regime_min_val_return_dd_ratio', 1.0))
        self.min_val_return_dd_ratio = _raw_ratio if math.isfinite(_raw_ratio) and _raw_ratio > 0 else 1.0

        # Research-only gate toggles (default all enabled to preserve production behavior)
        self.g1_min_trades_enabled = bool(getattr(config, 'research_gate_g1_min_trades', True))
        self.g2_max_drawdown_enabled = bool(getattr(config, 'research_gate_g2_max_drawdown', True))
        self.g3_train_profitability_enabled = bool(getattr(config, 'research_gate_g3_train_profitability', True))
        self.g4_val_return_enabled = bool(getattr(config, 'research_gate_g4_val_return', True))
        self.g5_return_dd_ratio_enabled = bool(getattr(config, 'research_gate_g5_return_dd_ratio', True))
        self.g6_val_profit_factor_enabled = bool(getattr(config, 'research_gate_g6_val_profit_factor', True))
        self.g7_robustness_gap_enabled = bool(getattr(config, 'research_gate_g7_robustness_gap', True))
        self.g1_min_trades_threshold = int(getattr(config, 'research_gate_g1_threshold', self.min_val_trades) or self.min_val_trades)
        self.g2_max_drawdown_threshold = float(getattr(config, 'research_gate_g2_threshold', self.val_max_drawdown) or self.val_max_drawdown)
        self.g3_train_return_threshold = float(getattr(config, 'research_gate_g3_threshold', 0.0) or 0.0)
        self.g4_val_return_threshold = float(getattr(config, 'research_gate_g4_threshold', self.min_val_return_pct) or self.min_val_return_pct)
        self.g5_return_dd_ratio_threshold = float(getattr(config, 'research_gate_g5_threshold', self.min_val_return_dd_ratio) or self.min_val_return_dd_ratio)
        self.g6_val_profit_factor_threshold = float(getattr(config, 'research_gate_g6_threshold', self.min_val_profit_factor) or self.min_val_profit_factor)
        self.g7_robustness_threshold = float(getattr(config, 'research_gate_g7_threshold', self.min_robustness_ratio) or self.min_robustness_ratio)
        
        # Use fx_backtester scoring mode
        self.use_fx_scoring = getattr(config, 'scoring_mode', 'pm_weighted') == 'fx_backtester'
        
        # Initialize Optuna optimizer for regime tuning
        if OPTUNA_AVAILABLE and self.enable_hyperparam_tuning:
            self.optuna_optimizer = create_optimizer(
                config, self.backtester, self.scorer, StrategyRegistry
            )
            # Update n_trials for regime optimization
            if self.optuna_optimizer and hasattr(self.optuna_optimizer, 'config'):
                self.optuna_optimizer.config.n_trials = self.hyperparam_max_combos
            logger.debug(f"RegimeOptimizer using: {get_optimization_method()}")
        else:
            self.optuna_optimizer = None
            if self.enable_hyperparam_tuning:
                logger.debug("RegimeOptimizer using: Grid Search (Optuna not available)")
    
    @staticmethod
    def _compute_return_dd_ratio(val_return: float, val_dd: float, eps: float = 0.5) -> float:
        """Return-to-drawdown efficiency ratio with epsilon floor."""
        return val_return / max(val_dd, eps)

    def optimize_symbol(self,
                        symbol: str,
                        train_features_by_tf: Dict[str, pd.DataFrame],
                        val_features_by_tf: Dict[str, pd.DataFrame],
                        strategies: List[BaseStrategy]) -> Tuple[Dict[str, Dict[str, RegimeConfig]], Optional[RegimeConfig], int, int]:
        """
        Optimize for all (timeframe, regime) combinations with validation.
        
        Args:
            symbol: Symbol name
            train_features_by_tf: Dict mapping timeframe -> training features (with regime columns)
            val_features_by_tf: Dict mapping timeframe -> validation features (with regime columns)
            strategies: List of strategy instances
            
        Returns:
            Tuple of:
            - Dict[timeframe][regime] -> RegimeConfig (validated winners only)
            - Optional[RegimeConfig] for default fallback (best validated overall)
            - int: count of validated winners
            - int: count of unvalidated (rejected) winners
        """
        logger.info(f"[{symbol}] Regime Optimization: {len(strategies)} strategies x {len(train_features_by_tf)} timeframes x {len(self.REGIMES)} regimes")
        
        # Result structure: {timeframe: {regime: RegimeConfig}}
        regime_configs: Dict[str, Dict[str, RegimeConfig]] = {}
        
        # Track best overall for default fallback
        best_overall_score = -float('inf')
        best_overall_config: Optional[RegimeConfig] = None
        
        # Validation counters
        validated_winners = 0
        unvalidated_winners = 0
        
        for tf, train_features in train_features_by_tf.items():
            val_features = val_features_by_tf.get(tf)
            
            if train_features is None or len(train_features) < 200:
                logger.debug(f"[{symbol}] [{tf}] Skipping - insufficient data")
                continue
            
            # Check if regime columns exist
            if 'REGIME_LIVE' not in train_features.columns:
                logger.warning(f"[{symbol}] [{tf}] No REGIME_LIVE column - regime features not computed")
                continue
            
            regime_configs[tf] = {}
            
            # Collect all candidates for this timeframe
            tf_candidates = self._collect_candidates(
                symbol, tf, train_features, val_features, strategies
            )
            
            if not tf_candidates:
                logger.debug(f"[{symbol}] [{tf}] No valid candidates")
                continue
            
            # Select best per regime (with validation)
            for regime in self.REGIMES:
                best_config, is_valid, val_reason = self._select_best_for_regime(
                    symbol, tf, regime, tf_candidates
                )
                
                if best_config is not None:
                    # Only include validated winners (or all if validation disabled)
                    if is_valid:
                        regime_configs[tf][regime] = best_config
                        validated_winners += 1
                        
                        # Track best overall (only from validated)
                        if best_config.quality_score > best_overall_score:
                            best_overall_score = best_config.quality_score
                            best_overall_config = best_config
                        
                        # Check if this was a tuned candidate
                        is_tuned = any(
                            c.get('is_tuned', False) and 
                            c['strategy_name'] == best_config.strategy_name and
                            c['params'] == best_config.parameters
                            for c in tf_candidates
                        )
                        tuned_tag = " [TUNED]" if is_tuned else ""
                        
                        logger.info(f"[{symbol}] [{tf}] [{regime}] Winner: {best_config.strategy_name}{tuned_tag} "
                                   f"(quality={best_config.quality_score:.3f}, "
                                   f"train={best_config.regime_train_trades}, "
                                   f"val={best_config.regime_val_trades}) "
                                   f"[VALIDATED: {val_reason}]")
                    else:
                        unvalidated_winners += 1
                        logger.debug(f"[{symbol}] [{tf}] [{regime}] Candidate {best_config.strategy_name} "
                                    f"FAILED validation: {val_reason}")
                else:
                    unvalidated_winners += 1
                    logger.warning(
                        f"[{symbol}] [{tf}] [{regime}] No validated winner - {val_reason}. "
                        f"NO TRADE for this regime."
                    )
        
        # --- Reason-coded summary telemetry ---
        total_slots = sum(len(self.REGIMES) for _ in train_features_by_tf)
        logger.info(
            f"[{symbol}] Regime optimization summary: "
            f"{validated_winners}/{total_slots} validated, "
            f"{unvalidated_winners} unvalidated/no-winner, "
            f"{total_slots - validated_winners - unvalidated_winners} skipped (data)"
        )

        return regime_configs, best_overall_config, validated_winners, unvalidated_winners

    def _apply_training_eligibility_gates(self,
                                          candidates: List[Dict[str, Any]],
                                          symbol: str,
                                          timeframe: str) -> List[Dict[str, Any]]:
        """
        Filter catastrophic training candidates before expensive tuning.
        """
        eligible: List[Dict[str, Any]] = []
        rejected_reasons: List[str] = []

        train_pf_floor = float(getattr(self.config, "train_min_profit_factor", 0.5))
        train_return_floor = float(getattr(self.config, "train_min_return_pct", -30.0))
        train_dd_ceiling = float(getattr(self.config, "train_max_drawdown", 60.0))

        for cand in candidates:
            train_result = cand.get("train_result", {})
            train_pf = float(train_result.get("profit_factor", 0))
            train_return = float(train_result.get("total_return_pct", 0))
            train_dd = float(train_result.get("max_drawdown_pct", 100))

            rejected = False
            reason = ""
            if train_pf < train_pf_floor:
                rejected = True
                reason = f"train PF {train_pf:.2f} < {train_pf_floor:.2f}"
            elif train_return < train_return_floor:
                rejected = True
                reason = f"train return {train_return:.1f}% < {train_return_floor:.1f}%"
            elif train_dd > train_dd_ceiling:
                rejected = True
                reason = f"train DD {train_dd:.1f}% > {train_dd_ceiling:.1f}%"

            if rejected:
                rejected_reasons.append(f"{cand['strategy_name']}: {reason}")
            else:
                eligible.append(cand)

        if rejected_reasons:
            logger.info(
                f"[{symbol}] [{timeframe}] Training eligibility rejected "
                f"{len(rejected_reasons)}/{len(candidates)}: "
                f"{rejected_reasons[:3]}{'...' if len(rejected_reasons) > 3 else ''}"
            )

        return eligible
    
    def _collect_candidates(self,
                            symbol: str,
                            timeframe: str,
                            train_features: pd.DataFrame,
                            val_features: Optional[pd.DataFrame],
                            strategies: List[BaseStrategy]) -> List[Dict[str, Any]]:
        """
        Collect all strategy candidates with per-regime metrics.
        
        Phase 1 (Screening): Run all strategies with default params, bucket by regime.
        Phase 2 (Tuning): For top-K strategies per regime, run hyperparameter optimization.
        
        Returns list of candidate dicts with bucketed metrics (including tuned variants).
        """
        # Phase 1: Quick screening with default params
        screening_candidates = []
        
        for strategy in strategies:
            try:
                # Run backtest on full training data with default params
                signals = strategy.generate_signals(train_features, symbol)
                train_result = self.backtester.run(
                    train_features, signals, symbol, strategy
                )
                
                # Skip if no trades
                if train_result.get('total_trades', 0) == 0:
                    continue
                
                # Bucket trades by regime
                train_trades = train_result.get('trades', [])
                train_regime_metrics = self._bucket_trades_by_regime(
                    train_trades, train_features
                )
                
                # Run validation if available
                val_regime_metrics = {}
                if val_features is not None and len(val_features) >= 50:
                    val_signals = strategy.generate_signals(val_features, symbol)
                    val_result = self.backtester.run(
                        val_features, val_signals, symbol, strategy
                    )
                    val_trades = val_result.get('trades', [])
                    val_regime_metrics = self._bucket_trades_by_regime(
                        val_trades, val_features
                    )
                
                screening_candidates.append({
                    'strategy': strategy,
                    'strategy_name': strategy.name,
                    'params': strategy.get_params(),
                    'train_result': train_result,
                    'train_regime_metrics': train_regime_metrics,
                    'val_regime_metrics': val_regime_metrics,
                    'is_tuned': False,
                })
                
            except Exception as e:
                logger.debug(f"[{symbol}] [{timeframe}] {strategy.name} failed: {e}")
                continue

        eligible_candidates = self._apply_training_eligibility_gates(
            screening_candidates, symbol, timeframe
        )
        if not eligible_candidates:
            logger.warning(
                f"[{symbol}] [{timeframe}] No strategies passed training eligibility gates. "
                f"All {len(screening_candidates)} candidates rejected."
            )
            return []

        if not self.enable_hyperparam_tuning:
            return eligible_candidates
        
        # Phase 2: Hyperparameter tuning for top-K strategies per regime
        # Identify top-K strategies for each regime based on screening scores
        regime_top_strategies = self._identify_top_strategies_per_regime(
            eligible_candidates, top_k=self.hyperparam_top_k
        )
        
        # Collect all unique strategies that need tuning
        strategies_to_tune = set()
        for regime, strategy_names in regime_top_strategies.items():
            strategies_to_tune.update(strategy_names)
        
        if not strategies_to_tune:
            logger.debug(f"[{symbol}] [{timeframe}] No strategies eligible for hyperparameter tuning")
            return eligible_candidates
        
        logger.info(f"[{symbol}] [{timeframe}] Hyperparameter tuning {len(strategies_to_tune)} strategies")
        
        # Run hyperparameter tuning for each strategy
        tuned_candidates = []
        for strategy_name in strategies_to_tune:
            # Find the original strategy from eligible candidate set.
            original_cand = next((c for c in eligible_candidates if c['strategy_name'] == strategy_name), None)
            if original_cand is None:
                continue
            
            # Get param grid
            param_grid = original_cand['strategy'].get_param_grid()
            if not param_grid:
                # No params to tune, keep original
                continue
            
            # Run grid search
            tuned_results = self._tune_strategy_params(
                symbol, timeframe, strategy_name, param_grid,
                train_features, val_features
            )
            
            tuned_candidates.extend(tuned_results)
        
        # Combine eligibility-filtered candidates with tuned candidates.
        # The tuned candidates may replace or augment the defaults
        all_candidates = eligible_candidates + tuned_candidates
        
        return all_candidates
    
    def _identify_top_strategies_per_regime(self,
                                            candidates: List[Dict[str, Any]],
                                            top_k: int = 3) -> Dict[str, List[str]]:
        """
        Identify top-K strategies for each regime based on screening scores.
        
        Returns dict mapping regime -> list of strategy names to tune.
        """
        regime_scores: Dict[str, List[Tuple[str, float]]] = {r: [] for r in self.REGIMES}
        
        for cand in candidates:
            for regime in self.REGIMES:
                train_metrics = cand['train_regime_metrics'].get(regime, {})
                val_metrics = cand['val_regime_metrics'].get(regime, {})
                
                train_trades = train_metrics.get('total_trades', 0)
                
                # Only consider strategies with enough trades in this regime
                if train_trades < self.min_train_trades // 2:  # Relaxed threshold for screening
                    continue
                
                score = self._compute_regime_score(train_metrics, val_metrics)
                regime_scores[regime].append((cand['strategy_name'], score))
        
        # Select top-K for each regime
        result = {}
        for regime, scores in regime_scores.items():
            if scores:
                scores.sort(key=lambda x: x[1], reverse=True)
                result[regime] = [name for name, _ in scores[:top_k]]
            else:
                result[regime] = []
        
        return result
    
    def _tune_strategy_params(self,
                              symbol: str,
                              timeframe: str,
                              strategy_name: str,
                              param_grid: Dict[str, List],
                              train_features: pd.DataFrame,
                              val_features: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Tune hyperparameters for a strategy using Optuna TPE.
        
        Uses multi-regime optimization to find best params for each regime
        in a single optimization run.
        
        Returns list of tuned candidate dicts with regime metrics.
        """
        # Use Optuna if available
        if self.optuna_optimizer is not None:
            regime_results = self.optuna_optimizer.optimize_for_regimes(
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name,
                param_grid=param_grid,
                train_features=train_features,
                val_features=val_features,
                regimes=self.REGIMES,
                bucket_trades_fn=self._bucket_trades_by_regime,
                compute_score_fn=self._compute_regime_score,
                min_train_trades=self.min_train_trades,
                max_drawdown_pct=self.val_max_drawdown,  # Pass DD threshold for early rejection
            )
            
            # Convert to candidate list format
            tuned_candidates = []
            for regime, candidate in regime_results.items():
                candidate['tuned_for_regime'] = regime
                tuned_candidates.append(candidate)
            
            if tuned_candidates:
                logger.debug(f"[{symbol}] [{timeframe}] {strategy_name}: "
                           f"{len(tuned_candidates)} tuned variants (Optuna TPE)")
            
            return tuned_candidates
        
        # Fallback to grid search
        return self._tune_strategy_params_grid(
            symbol, timeframe, strategy_name, param_grid, train_features, val_features
        )
    
    def _tune_strategy_params_grid(self,
                                   symbol: str,
                                   timeframe: str,
                                   strategy_name: str,
                                   param_grid: Dict[str, List],
                                   train_features: pd.DataFrame,
                                   val_features: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Fallback grid search for hyperparameter tuning.
        
        Returns list of tuned candidate dicts with regime metrics.
        """
        from itertools import product
        
        # Generate param combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(product(*param_values))
        
        # Limit combinations
        if len(all_combos) > self.hyperparam_max_combos:
            np.random.seed(42)
            indices = np.random.choice(len(all_combos), self.hyperparam_max_combos, replace=False)
            all_combos = [all_combos[i] for i in indices]
        
        # Get default params as base
        base_strategy = StrategyRegistry.get(strategy_name)
        default_params = base_strategy.get_params()
        
        tuned_candidates = []
        best_scores_by_regime: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        
        for combo in all_combos:
            params = {**default_params, **dict(zip(param_names, combo))}
            
            try:
                # Create strategy with these params
                test_strategy = StrategyRegistry.get(strategy_name, **params)
                
                # Run backtest
                signals = test_strategy.generate_signals(train_features, symbol)
                train_result = self.backtester.run(
                    train_features, signals, symbol, test_strategy
                )
                
                if train_result.get('total_trades', 0) < self.config.min_trades // 2:
                    continue
                
                # Bucket by regime
                train_trades = train_result.get('trades', [])
                train_regime_metrics = self._bucket_trades_by_regime(train_trades, train_features)
                
                # Validation
                val_regime_metrics = {}
                if val_features is not None and len(val_features) >= 50:
                    val_signals = test_strategy.generate_signals(val_features, symbol)
                    val_result = self.backtester.run(
                        val_features, val_signals, symbol, test_strategy
                    )
                    val_trades = val_result.get('trades', [])
                    val_regime_metrics = self._bucket_trades_by_regime(val_trades, val_features)
                
                # Score per regime and track best
                for regime in self.REGIMES:
                    train_m = train_regime_metrics.get(regime, {})
                    val_m = val_regime_metrics.get(regime, {})
                    
                    if train_m.get('total_trades', 0) < self.min_train_trades:
                        continue
                    
                    score = self._compute_regime_score(train_m, val_m)
                    
                    if regime not in best_scores_by_regime or score > best_scores_by_regime[regime][0]:
                        # Clean params for JSON serialization
                        clean_params = {}
                        for k, v in params.items():
                            if hasattr(v, "item"):
                                v = v.item()
                            elif isinstance(v, (np.integer, np.floating)):
                                v = float(v) if isinstance(v, np.floating) else int(v)
                            clean_params[k] = v
                        
                        best_scores_by_regime[regime] = (score, {
                            'strategy': test_strategy,
                            'strategy_name': strategy_name,
                            'params': clean_params,
                            'train_result': train_result,
                            'train_regime_metrics': train_regime_metrics,
                            'val_regime_metrics': val_regime_metrics,
                            'is_tuned': True,
                            'tuned_for_regime': regime,
                        })
                
            except Exception as e:
                logger.debug(f"[{symbol}] [{timeframe}] {strategy_name} param combo failed: {e}")
                continue
        
        # Collect best tuned candidates (one per regime this strategy is good for)
        for regime, (score, candidate) in best_scores_by_regime.items():
            tuned_candidates.append(candidate)
        
        if tuned_candidates:
            logger.debug(f"[{symbol}] [{timeframe}] {strategy_name}: "
                        f"{len(tuned_candidates)} tuned variants (grid search)")
        
        return tuned_candidates
    
    def _bucket_trades_by_regime(self,
                                  trades: List[Dict],
                                  features: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Bucket trades by REGIME_LIVE at entry bar.
        
        Uses REGIME_LIVE which is the shifted regime (what the system knows at decision time).
        
        Returns dict mapping regime -> metrics for trades in that regime.
        """
        regime_buckets = {regime: [] for regime in self.REGIMES}
        
        for trade in trades:
            entry_bar = trade.get('entry_bar', trade.get('signal_bar', 0))
            
            # Get regime at entry using REGIME_LIVE (shifted version for decision-time parity)
            if entry_bar < len(features) and 'REGIME_LIVE' in features.columns:
                regime = features['REGIME_LIVE'].iloc[entry_bar]
                
                if pd.notna(regime) and regime in regime_buckets:
                    regime_buckets[regime].append(trade)
        
        # Compute metrics for each bucket
        regime_metrics = {}
        for regime, bucket_trades in regime_buckets.items():
            if bucket_trades:
                regime_metrics[regime] = self._compute_bucket_metrics(bucket_trades)
            else:
                regime_metrics[regime] = {'total_trades': 0}
        
        return regime_metrics
    
    def _compute_bucket_metrics(self, trades: List[Dict], initial_capital: float = None) -> Dict[str, Any]:
        """
        Compute metrics for a bucket of trades including proper drawdown.
        
        Args:
            trades: List of trade dicts with pnl_dollars, entry_bar, etc.
            initial_capital: Starting capital for return calculation (from config)
            
        Returns:
            Dict with comprehensive metrics including max_drawdown_pct
        """
        if not trades:
            return {'total_trades': 0, 'max_drawdown_pct': 0.0, 'total_return_pct': 0.0}
        
        # Get initial capital from config if not provided
        if initial_capital is None:
            initial_capital = getattr(self.config, 'initial_capital', 10000.0)
        
        n = len(trades)
        winners = [t for t in trades if t.get('pnl_dollars', 0) > 0]
        losers = [t for t in trades if t.get('pnl_dollars', 0) < 0]
        
        total_pnl = sum(t.get('pnl_dollars', 0) for t in trades)
        gross_profit = sum(t.get('pnl_dollars', 0) for t in winners) if winners else 0
        gross_loss = abs(sum(t.get('pnl_dollars', 0) for t in losers)) if losers else 0
        
        win_rate = (len(winners) / n * 100) if n > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (10.0 if gross_profit > 0 else 0)
        
        avg_win = (gross_profit / len(winners)) if winners else 0
        avg_loss = (gross_loss / len(losers)) if losers else 0
        
        # Calculate Sharpe-like metric
        returns = [t.get('pnl_dollars', 0) for t in trades]
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0
        sharpe_approx = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        
        # ===== FIX: Compute proper total_return_pct =====
        total_return_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0
        
        # ===== FIX: Compute proper max_drawdown_pct from bucket equity curve =====
        # Sort trades by entry time to build correct equity curve
        sorted_trades = sorted(trades, key=lambda t: t.get('entry_bar', t.get('signal_bar', 0)))
        
        # Build cumulative equity curve for this bucket
        equity = initial_capital
        peak_equity = initial_capital
        max_drawdown_pct = 0.0
        
        for trade in sorted_trades:
            pnl = trade.get('pnl_dollars', 0)
            equity += pnl
            
            # Track peak and drawdown
            if equity > peak_equity:
                peak_equity = equity
            
            if peak_equity > 0:
                current_dd_pct = ((peak_equity - equity) / peak_equity) * 100
                if current_dd_pct > max_drawdown_pct:
                    max_drawdown_pct = current_dd_pct

        # ===== Step D: Expanded regime bucket metrics for scoring calibration =====

        # Max consecutive losses
        max_consec_losses = 0
        current_streak = 0
        for trade in sorted_trades:
            if trade.get('pnl_dollars', 0) < 0:
                current_streak += 1
                max_consec_losses = max(max_consec_losses, current_streak)
            else:
                current_streak = 0

        # Calmar ratio
        calmar_ratio = (total_return_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0

        # R-multiple stats (from trade data if available)
        r_multiples = [t.get('r_multiple', 0.0) for t in trades if 'r_multiple' in t]
        if r_multiples:
            mean_r = float(np.mean(r_multiples))
            worst_5pct_r = float(np.percentile(r_multiples, 5)) if len(r_multiples) >= 20 else min(r_multiples)
        else:
            mean_r = 0.0
            worst_5pct_r = 0.0

        # Sortino ratio proxy (from trade returns)
        if len(returns) > 1:
            negative_returns = [r for r in returns if r < 0]
            if len(negative_returns) >= 2:
                downside_std = np.std(negative_returns, ddof=1)
                sortino_approx = (avg_return / downside_std * np.sqrt(252)) if downside_std > 1e-10 else 0.0
            else:
                sortino_approx = sharpe_approx * 1.5 if avg_return > 0 else 0.0
        else:
            sortino_approx = 0.0

        return {
            'total_trades': n,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 3),
            'total_pnl': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'sharpe_approx': round(sharpe_approx, 3),
            'max_drawdown_pct': round(max_drawdown_pct, 2),  # FIX: Now properly computed
            'total_return_pct': round(total_return_pct, 2),  # FIX: Now properly computed
            'max_consecutive_losses': max_consec_losses,
            'calmar_ratio': round(calmar_ratio, 2),
            'mean_r': round(mean_r, 3),
            'worst_5pct_r': round(worst_5pct_r, 3),
            'sortino_approx': round(sortino_approx, 3),
        }
    
    def _select_best_for_regime(self,
                                 symbol: str,
                                 timeframe: str,
                                 regime: str,
                                 candidates: List[Dict[str, Any]]) -> Tuple[Optional[RegimeConfig], bool, str]:
        """
        Select best candidate for a specific regime with validation.
        
        Applies minimum trade thresholds, profitability gates, scores candidates, 
        and validates winner.
        
        Returns:
            Tuple of (RegimeConfig or None, is_validated, validation_reason)
        """
        scored_candidates = []  # List of (score, candidate) tuples
        rejection_reasons = []  # Track why candidates were rejected at gate level

        for cand in candidates:
            train_metrics = cand['train_regime_metrics'].get(regime, {})
            val_metrics = cand['val_regime_metrics'].get(regime, {})

            train_trades = train_metrics.get('total_trades', 0)
            val_trades = val_metrics.get('total_trades', 0)

            # Check minimum training trades (not a gate — always enforced)
            if train_trades < self.min_train_trades:
                continue

            # G1: Check minimum validation trades (gate-controlled for ablation)
            if self.g1_min_trades_enabled and val_trades < self.g1_min_trades_threshold:
                continue

            # ===== EARLY REJECTION: Check drawdown before scoring =====
            train_dd = train_metrics.get('max_drawdown_pct', 100.0)
            val_dd = val_metrics.get('max_drawdown_pct', 100.0)

            # G2: Reject if validation drawdown exceeds threshold (gate-controlled)
            if self.g2_max_drawdown_enabled and val_dd > self.g2_max_drawdown_threshold:
                continue

            # G2: Reject if training drawdown is excessively high (gate-controlled)
            if self.g2_max_drawdown_enabled and train_dd > self.g2_max_drawdown_threshold * 1.25:
                continue

            # G5: Return-to-DD efficiency gate (gate-controlled for ablation)
            val_return_early = val_metrics.get('total_return_pct', -100.0)
            val_return_dd_ratio = self._compute_return_dd_ratio(val_return_early, val_dd)
            if self.g5_return_dd_ratio_enabled and val_return_dd_ratio < self.g5_return_dd_ratio_threshold:
                rejection_reasons.append(
                    f"{cand['strategy_name']}: return/DD={val_return_dd_ratio:.2f} < "
                    f"{self.g5_return_dd_ratio_threshold:.2f} (ret={val_return_early:.1f}%, dd={val_dd:.1f}%)"
                )
                continue

            # G4/G6: Profitability gates (gate-controlled for ablation)
            val_pf = val_metrics.get('profit_factor', 0.0)
            val_return = val_metrics.get('total_return_pct', -100.0)

            if not self.allow_losing_winners:
                if self.g6_val_profit_factor_enabled and val_pf < self.g6_val_profit_factor_threshold:
                    rejection_reasons.append(f"{cand['strategy_name']}: PF={val_pf:.2f} < {self.g6_val_profit_factor_threshold}")
                    continue
                if self.g4_val_return_enabled and val_return < self.g4_val_return_threshold:
                    rejection_reasons.append(f"{cand['strategy_name']}: return={val_return:.1f}% < {self.g4_val_return_threshold}")
                    continue

            # Score based on validation metrics (with train as fallback)
            score = self._compute_regime_score(train_metrics, val_metrics)
            scored_candidates.append((score, cand))

        if not scored_candidates:
            reason = "No candidates met minimum trade/drawdown/ratio/profitability thresholds"
            if rejection_reasons:
                reason += f" (rejected: {len(rejection_reasons)} candidates by gate checks)"
            return None, False, reason

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Cap the number of validation attempts
        top_k = min(self.regime_validation_top_k, len(scored_candidates))

        logger.debug(
            f"[{symbol}] [{timeframe}] [{regime}] Candidate descent: "
            f"{len(scored_candidates)} gate-passed, validating top-{top_k}"
        )

        # Iterate top-K candidates, return first that passes validation
        validation_failures = []
        for rank, (score, cand) in enumerate(scored_candidates[:top_k], start=1):
            train_metrics = cand['train_regime_metrics'].get(regime, {})
            val_metrics = cand['val_regime_metrics'].get(regime, {})
            strategy_name = cand.get('strategy_name', 'Unknown')

            is_validated, validation_reason = self._validate_regime_winner(
                train_metrics, val_metrics, regime, strategy_name
            )

            if is_validated:
                if rank > 1:
                    logger.info(
                        f"[{symbol}] [{timeframe}] [{regime}] Descent: "
                        f"rank #{rank} {strategy_name} passed validation "
                        f"(top {rank - 1} failed)"
                    )
                else:
                    logger.debug(
                        f"[{symbol}] [{timeframe}] [{regime}] Descent: "
                        f"rank #1 {strategy_name} passed validation"
                    )

                quality_score = self._normalize_score(score)
                now = datetime.now()

                config = RegimeConfig(
                    strategy_name=strategy_name,
                    parameters=cand['params'],
                    quality_score=quality_score,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    regime_train_trades=train_metrics.get('total_trades', 0),
                    regime_val_trades=val_metrics.get('total_trades', 0),
                    trained_at=now,
                    valid_until=now + timedelta(days=60),
                )

                return config, True, validation_reason
            else:
                validation_failures.append(f"#{rank} {strategy_name}: {validation_reason}")
                logger.debug(
                    f"[{symbol}] [{timeframe}] [{regime}] Descent: "
                    f"rank #{rank} {strategy_name} FAILED: {validation_reason}"
                )

        # All top-K candidates failed validation - return the best one as unvalidated
        best_score, best_candidate = scored_candidates[0]
        train_metrics = best_candidate['train_regime_metrics'].get(regime, {})
        val_metrics = best_candidate['val_regime_metrics'].get(regime, {})

        quality_score = self._normalize_score(best_score)
        now = datetime.now()

        config = RegimeConfig(
            strategy_name=best_candidate['strategy_name'],
            parameters=best_candidate['params'],
            quality_score=quality_score,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            regime_train_trades=train_metrics.get('total_trades', 0),
            regime_val_trades=val_metrics.get('total_trades', 0),
            trained_at=now,
            valid_until=now + timedelta(days=60),
        )

        descent_summary = "; ".join(validation_failures)
        final_reason = (
            f"All {len(validation_failures)} descent candidates failed validation "
            f"[{descent_summary}]"
        )
        logger.info(
            f"[{symbol}] [{timeframe}] [{regime}] Descent exhausted: "
            f"{len(validation_failures)}/{top_k} candidates failed validation"
        )

        return config, False, final_reason
    
    def _validate_regime_winner(self,
                                 train_metrics: Dict[str, Any],
                                 val_metrics: Dict[str, Any],
                                 regime: str,
                                 candidate_name: str = "Unknown") -> Tuple[bool, str]:
        """
        Validate a regime winner against fx_backtester thresholds.
        
        Checks:
        - Always enforced:
          - Validation trade count >= fx_val_min_trades
          - Validation drawdown < fx_val_max_drawdown
          - Validation return/DD ratio >= regime_min_val_return_dd_ratio
        - Enforced when allow_losing_winners=False:
          - Validation profit_factor >= regime_min_val_profit_factor (default 1.0)
          - Validation return_pct >= regime_min_val_return_pct (default 5.0)
        - Robustness ratio >= fx_min_robustness_ratio OR val_sharpe > override
        
        Returns:
            (is_valid, reason)
        """
        val_trades = val_metrics.get('total_trades', 0)
        val_sharpe = val_metrics.get('sharpe_approx', 0)
        val_drawdown = val_metrics.get('max_drawdown_pct', 100.0)  # Default to worst-case
        val_pf = val_metrics.get('profit_factor', 0.0)
        val_return = val_metrics.get('total_return_pct', -100.0)
        train_pf = train_metrics.get('profit_factor', 0.0)
        train_return = train_metrics.get('total_return_pct', -100.0)
        
        # Check minimum validation trades
        if self.g1_min_trades_enabled and val_trades < self.g1_min_trades_threshold:
            return False, f"Insufficient val trades: {val_trades} < {self.g1_min_trades_threshold}"

        # Check validation drawdown
        if self.g2_max_drawdown_enabled and val_drawdown > self.g2_max_drawdown_threshold:
            return False, f"Excessive val drawdown: {val_drawdown:.1f}% > {self.g2_max_drawdown_threshold}%"
        
        # Also check training drawdown as early rejection
        train_drawdown = train_metrics.get('max_drawdown_pct', 100.0)
        # Allow slightly higher train DD (1.25x) since we expect some overfitting
        train_dd_threshold = self.g2_max_drawdown_threshold * 1.25
        if self.g2_max_drawdown_enabled and train_drawdown > train_dd_threshold:
            return False, f"Excessive train drawdown: {train_drawdown:.1f}% > {train_dd_threshold:.1f}%"

        # Weak-train exception: only allow if validation is exceptional.
        weak_train = self.g3_train_profitability_enabled and (train_pf < 1.0 or train_return < self.g3_train_return_threshold)
        if weak_train:
            exceptional_val_pf = float(getattr(self.config, "exceptional_val_profit_factor", 1.15))
            exceptional_val_return = float(getattr(self.config, "exceptional_val_return_pct", 5.0))
            exceptional_val_min_trades = self.min_val_trades * 2

            val_dd_limit = self.val_max_drawdown * 0.75
            val_win_rate = val_metrics.get('win_rate', 0)
            wt_return_dd_ratio = self._compute_return_dd_ratio(val_return, val_drawdown)
            wt_min_ratio = self.g5_return_dd_ratio_threshold

            if (val_pf >= exceptional_val_pf and
                val_return >= exceptional_val_return and
                val_trades >= exceptional_val_min_trades and
                val_drawdown < val_dd_limit and
                val_win_rate > 50.0 and
                wt_return_dd_ratio >= wt_min_ratio):
                logger.info(
                    f"[{regime}] {candidate_name}: allowing weak-train winner "
                    f"(train PF {train_pf:.2f}, train return {train_return:.1f}%) "
                    f"due to exceptional validation "
                    f"(val PF {val_pf:.2f}, val return {val_return:.1f}%, val trades {val_trades}, "
                    f"val DD {val_drawdown:.1f}%, val WR {val_win_rate:.1f}%, "
                    f"return/DD {wt_return_dd_ratio:.2f})"
                )
            else:
                return False, (
                    f"Weak train rejected: train PF {train_pf:.2f}, train return {train_return:.1f}% "
                    f"(requires val PF>={exceptional_val_pf:.2f}, val return>={exceptional_val_return:.1f}%, "
                    f"val trades>={exceptional_val_min_trades}, "
                    f"val DD<{val_dd_limit:.1f}%, val WR>50%, "
                    f"return/DD>={wt_min_ratio:.2f})"
                )
        
        # ===== FIX #1: PROFITABILITY GATES (Critical new checks) =====
        # These prevent storing losing strategies as "regime winners"
        if not self.allow_losing_winners:
            # Check profit factor >= threshold (default 1.0 = must be profitable)
            if self.g6_val_profit_factor_enabled and val_pf < self.g6_val_profit_factor_threshold:
                return False, f"Unprofitable: val PF {val_pf:.3f} < {self.g6_val_profit_factor_threshold:.2f}"
            
            # Check return >= threshold (default 5.0)
            if self.g4_val_return_enabled and val_return < self.g4_val_return_threshold:
                return False, f"Negative return: val return {val_return:.2f}% < {self.g4_val_return_threshold:.1f}%"

        # Return-to-DD efficiency gate (unconditional risk-efficiency rule, same category as DD cap)
        val_return_dd_ratio = self._compute_return_dd_ratio(val_return, val_drawdown)
        min_return_dd = self.g5_return_dd_ratio_threshold
        if self.g5_return_dd_ratio_enabled and val_return_dd_ratio < min_return_dd:
            return False, (
                f"Poor return/DD efficiency: {val_return_dd_ratio:.2f} "
                f"(return {val_return:.1f}% / DD {val_drawdown:.1f}%) "
                f"< {min_return_dd:.2f}"
            )

        # Compute robustness ratio
        train_full = self._bucket_to_full_metrics(train_metrics)
        val_full = self._bucket_to_full_metrics(val_metrics)
        
        if self.use_fx_scoring:
            robustness = self.scorer.calculate_fx_score_robustness_ratio(
                train_full, val_full, purpose="selection"
            )
        else:
            # Simple profit factor ratio
            train_pf = train_metrics.get('profit_factor', 1.0)
            robustness = val_pf / (train_pf + 0.1) if train_pf > 0 else 0.5
        
        # Check robustness OR Sharpe override
        if self.g7_robustness_gap_enabled and robustness < self.g7_robustness_threshold and val_sharpe <= self.val_min_sharpe_override:
            return False, f"Low robustness {robustness:.2f} < {self.g7_robustness_threshold} and Sharpe {val_sharpe:.2f} <= {self.val_min_sharpe_override}"
        
        return True, f"Validated (PF={val_pf:.2f}, ret={val_return:.1f}%, robustness={robustness:.2f}, dd={val_drawdown:.1f}%)"
    
    def _compute_regime_score(self,
                               train_metrics: Dict[str, Any],
                               val_metrics: Dict[str, Any]) -> float:
        """
        Compute score for a regime bucket using fx_backtester generalization scoring.
        
        When scoring_mode='fx_backtester', uses:
        - Gap penalty for train->val degradation
        - Robustness ratio boost
        - Validation-first scoring
        - Trade count stability factor (Gap C fix)
        
        Falls back to simple composite for pm_weighted mode.
        """
        # ===== FIX Gap C: Trade count stability factor =====
        # Prefer candidates with more trades (more statistical confidence)
        # Uses log scaling: log1p(trades) / log1p(target) capped at 1.0
        train_trades = train_metrics.get('total_trades', 0)
        val_trades = val_metrics.get('total_trades', 0)
        
        # Target trade counts for full stability bonus
        # (2x minimum is considered "healthy")
        target_train = self.min_train_trades * 2
        target_val = self.min_val_trades * 2
        
        # Calculate stability factors using log scaling
        import math
        train_stability = min(1.0, math.log1p(train_trades) / math.log1p(target_train)) if target_train > 0 else 0.5
        val_stability = min(1.0, math.log1p(val_trades) / math.log1p(target_val)) if target_val > 0 else 0.5
        
        # Combined stability factor (weighted average favoring validation)
        stability_factor = 0.3 * train_stability + 0.7 * val_stability
        # Clamp to reasonable range [0.7, 1.0] - don't over-penalize low counts
        stability_factor = 0.7 + 0.3 * stability_factor
        
        if self.use_fx_scoring:
            # Build pseudo-metrics dicts for the scorer (regime bucket metrics -> full metrics format)
            train_full = self._bucket_to_full_metrics(train_metrics)
            val_full = self._bucket_to_full_metrics(val_metrics)
            
            # Use validation metrics if sufficient trades
            if val_trades >= self.min_val_trades:
                # Use fx_generalization_score for proper gap penalty + robustness boost
                final_score, train_score, val_score, rr = self.scorer.fx_generalization_score(
                    train_full, val_full, purpose="selection"
                )
                # Apply stability factor (Gap C fix)
                return final_score * stability_factor
            else:
                # Not enough validation trades - use train score with discount
                train_score = self.scorer.score(train_full, purpose="selection")
                return train_score * 0.7 * stability_factor  # Discount for no validation
        
        # Legacy pm_weighted scoring (original behavior)
        if val_trades >= self.min_val_trades:
            metrics = val_metrics
            train_pf = train_metrics.get('profit_factor', 1.0)
            val_pf = metrics.get('profit_factor', 1.0)
            robustness = min(1.0, val_pf / (train_pf + 0.1)) if train_pf > 0 else 0.5
        else:
            metrics = train_metrics
            robustness = 0.7
        
        if metrics.get('total_trades', 0) == 0:
            return -float('inf')
        
        pf = metrics.get('profit_factor', 0)
        wr = metrics.get('win_rate', 0) / 100
        sharpe = metrics.get('sharpe_approx', 0)
        total_pnl = metrics.get('total_pnl', 0)
        
        score = (
            min(pf, 5.0) * 20 +
            wr * 30 +
            min(max(sharpe, -2), 3) * 15 +
            min(total_pnl / 50, 20) +
            robustness * 20
        )
        
        # Apply stability factor (Gap C fix)
        return score * stability_factor
    
    def _bucket_to_full_metrics(self, bucket_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert regime bucket metrics to full metrics format expected by StrategyScorer.
        
        Maps bucket fields to standard backtest result fields.
        Now uses properly computed max_drawdown_pct and total_return_pct from bucket.
        """
        # Get properly computed values (now available from _compute_bucket_metrics)
        max_dd = bucket_metrics.get('max_drawdown_pct', 100.0)  # Default to worst-case, not 0
        total_return = bucket_metrics.get('total_return_pct', 0.0)  # Now properly computed
        
        return {
            'total_trades': bucket_metrics.get('total_trades', 0),
            'win_rate': bucket_metrics.get('win_rate', 0),
            'profit_factor': bucket_metrics.get('profit_factor', 0),
            'total_return_pct': total_return,  # FIX: Use properly computed value
            'sharpe_ratio': bucket_metrics.get('sharpe_approx', 0),
            'max_drawdown_pct': max_dd,  # FIX: Use properly computed value (default worst-case)
            'gross_profit': bucket_metrics.get('gross_profit', 0),
            'gross_loss': bucket_metrics.get('gross_loss', 0),
            'total_pnl': bucket_metrics.get('total_pnl', 0),
            'expectancy_pips': bucket_metrics.get('avg_win', 0) * bucket_metrics.get('win_rate', 0) / 100
                             - bucket_metrics.get('avg_loss', 0) * (1 - bucket_metrics.get('win_rate', 0) / 100),
            'sortino_ratio': bucket_metrics.get('sortino_approx', 0),
            'calmar_ratio': bucket_metrics.get('calmar_ratio', 0),
            'mean_r': bucket_metrics.get('mean_r', 0),
            'worst_5pct_r': bucket_metrics.get('worst_5pct_r', 0),
            'max_consecutive_losses': bucket_metrics.get('max_consecutive_losses', 0),
        }
    
    def _normalize_score(self, score: float) -> float:
        """
        Normalize score to 0-1 range using sigmoid mapping.

        This prevents saturation at extreme scores.
        Calibrated for the continuous-DD scoring distribution where
        typical scores land in the 25-55 range.
        """
        # Sigmoid centered at 45 with scale 30
        # score=15 -> ~0.27, score=30 -> ~0.38, score=45 -> 0.5,
        # score=60 -> ~0.62, score=80 -> ~0.76
        import math
        try:
            normalized = 1.0 / (1.0 + math.exp(-(score - 45) / 30))
        except OverflowError:
            normalized = 0.0 if score < 0 else 1.0

        return max(0.0, min(1.0, normalized))


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class OptimizationPipeline:
    """
    Main optimization pipeline with regime-aware strategy selection.
    
    Orchestrates all stages:
    1. Data loading and splitting
    2. Feature computation (including regime detection)
    3. Regime-aware strategy selection per (timeframe, regime)
    4. Retrain period selection
    5. Validation and config generation
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_loader = DataLoader(config.data_dir, cache_dir=getattr(config, "data_cache_dir", None))
        self.splitter = DataSplitter(config)
        
        # Pipeline stages
        self.selector = StrategySelector(config)
        self.optimizer = HyperparameterOptimizer(config)
        self.retrain_selector = RetrainPeriodSelector(config)
        self.validator = Validator(config)
        
        # Regime-aware optimizer
        self.regime_optimizer = RegimeOptimizer(config)

    def build_regime_detection_version(self, symbol: str) -> Dict[str, Any]:
        """
        Build a version stamp for regime detection inputs used by a symbol.

        Stored in SymbolConfig and compared in ConfigLedger to trigger
        re-optimization when regime logic inputs change.
        """
        regime_params_file = getattr(self.config, "regime_params_file", "regime_params.json")
        params_path = Path(regime_params_file)
        if not params_path.is_absolute():
            params_path = Path.cwd() / params_path

        file_meta: Dict[str, Any] = {"path": str(params_path)}
        try:
            if params_path.exists():
                file_meta["mtime_ns"] = int(params_path.stat().st_mtime_ns)
        except Exception:
            pass

        by_timeframe: Dict[str, Dict[str, Any]] = {}
        for tf in self.config.timeframes:
            try:
                params = load_regime_params(symbol, tf, regime_params_file)
                by_timeframe[tf] = params.to_dict()
            except Exception:
                by_timeframe[tf] = {}

        return {
            "file": file_meta,
            "timeframes": by_timeframe,
        }
    
    def run_for_symbol(self, symbol: str) -> PipelineResult:
        """
        Run regime-aware optimization pipeline for a single symbol.
        
        Finds best (strategy, params) for each (timeframe, regime) combination.
        
        Args:
            symbol: Symbol to optimize
            
        Returns:
            PipelineResult with SymbolConfig containing regime_configs
        """
        start_time = time.time()
        result = PipelineResult(symbol=symbol, success=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMIZING: {symbol} (Regime-Aware)")
        logger.info(f"{'='*60}")
        
        try:
            # Load data for all timeframes
            data_by_tf = {}
            for tf in self.config.timeframes:
                data = self.data_loader.get_data(symbol, tf)
                if data is not None and len(data) >= 200:
                    data_by_tf[tf] = data
            
            if not data_by_tf:
                result.error_message = "No valid data available"
                return result
            
            result.timeframes_tested = len(data_by_tf)
            
            # Get all strategies
            strategies = StrategyRegistry.get_all_instances()
            result.strategies_tested = len(strategies)
            
            # Split data and compute features (including regime)
            train_features_by_tf = {}
            val_features_by_tf = {}
            full_data_by_tf = {}
            
            regime_params_file = getattr(self.config, 'regime_params_file', 'regime_params.json')
            
            for tf, data in data_by_tf.items():
                # Compute features once on full data, then slice into train/val
                # This preserves causality and avoids recomputation overhead.
                split_idx = self.splitter.get_split_indices(len(data))
                train_start, train_end = split_idx['train']
                val_start, val_end = split_idx['val']

                full_features = FeatureComputer.compute_all(
                    data, symbol=symbol, timeframe=tf,
                    regime_params_file=regime_params_file
                )
                train_features_by_tf[tf] = full_features.iloc[train_start:train_end]
                val_features_by_tf[tf] = full_features.iloc[val_start:val_end]
                full_data_by_tf[tf] = data
            
            # Run regime-aware optimization
            with Timer(f"[{symbol}] Regime Optimization"):
                regime_configs, default_config, validated_count, unvalidated_count = self.regime_optimizer.optimize_symbol(
                    symbol, train_features_by_tf, val_features_by_tf, strategies
                )
            
            # Count total validated winners
            total_winners = sum(len(regimes) for regimes in regime_configs.values())
            result.regime_winners = total_winners
            
            if total_winners == 0 and default_config is None:
                if unvalidated_count > 0:
                    result.error_message = f"No validated regime winners ({unvalidated_count} candidates failed validation)"
                else:
                    result.error_message = "No valid regime winners found"
                return result
            
            # Determine best overall for legacy fields
            best_overall_tf = None
            best_overall_regime = None
            if default_config:
                # Find which tf/regime this came from
                for tf, regimes in regime_configs.items():
                    for regime, cfg in regimes.items():
                        if cfg.strategy_name == default_config.strategy_name:
                            best_overall_tf = tf
                            best_overall_regime = regime
                            break
                    if best_overall_tf:
                        break
            
            # Determine retrain period
            retrain_days = 60  # Default
            if best_overall_tf and best_overall_tf in full_data_by_tf and default_config:
                try:
                    retrain_days = self.retrain_selector.select_period(
                        symbol,
                        default_config.strategy_name,
                        default_config.parameters,
                        full_data_by_tf[best_overall_tf],
                        best_overall_tf
                    )
                except Exception as e:
                    logger.warning(f"[{symbol}] Retrain period selection failed: {e}")
            
            # Create SymbolConfig with regime configs
            # is_validated now reflects actual validation, not just "any winner exists"
            now = datetime.now()
            config = SymbolConfig(
                symbol=symbol,
                regime_configs=regime_configs,
                default_config=default_config,
                # Legacy fields (from default)
                strategy_name=default_config.strategy_name if default_config else "",
                timeframe=best_overall_tf or "",
                parameters=default_config.parameters if default_config else {},
                retrain_days=retrain_days,
                train_metrics=default_config.train_metrics if default_config else {},
                val_metrics=default_config.val_metrics if default_config else {},
                composite_score=default_config.quality_score * 100 if default_config else 0.0,
                robustness_ratio=0.0,
                is_validated=total_winners > 0,  # Now only True if at least one winner passed validation
                validation_reason=f"{total_winners} validated winners ({unvalidated_count} rejected) across {len(regime_configs)} timeframes",
                regime_detection_version=self.build_regime_detection_version(symbol),
                optimized_at=now,
                valid_until=now + timedelta(days=retrain_days),
            )
            
            result.success = True
            result.config = config
            result.duration_seconds = time.time() - start_time
            
            # Log summary
            logger.info(f"\n[{symbol}] OPTIMIZATION COMPLETE")
            logger.info(f"  Timeframes:  {len(regime_configs)}")
            logger.info(f"  Validated:   {total_winners}")
            logger.info(f"  Rejected:    {unvalidated_count}")
            if default_config:
                logger.info(f"  Best:        {default_config.strategy_name} @ {best_overall_tf}/{best_overall_regime}")
            logger.info(f"  Retrain:     Every {retrain_days} days")
            logger.info(f"  Duration:    {result.duration_seconds:.1f}s")
            
            # Log per-timeframe summary
            for tf in sorted(regime_configs.keys()):
                regimes = regime_configs[tf]
                if regimes:
                    regime_summary = ", ".join(f"{r}:{cfg.strategy_name[:12]}" for r, cfg in regimes.items())
                    logger.info(f"  [{tf}] {regime_summary}")
            
        except Exception as e:
            logger.exception(f"[{symbol}] Pipeline error: {e}")
            result.error_message = str(e)
            result.duration_seconds = time.time() - start_time
        
        return result
    
    def run_for_all(self, symbols: List[str]) -> Dict[str, PipelineResult]:
        """
        Run optimization for all symbols.
        
        Args:
            symbols: List of symbols to optimize
            
        Returns:
            Dict mapping symbol to PipelineResult
        """
        results = {}
        
        total_start = time.time()
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"PORTFOLIO MANAGER - OPTIMIZATION PIPELINE")
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Strategies: {StrategyRegistry.count()}")
        logger.info(f"Timeframes: {self.config.timeframes}")
        logger.info(f"{'#'*60}")
        
        for i, symbol in enumerate(symbols):
            logger.info(f"\nProgress: {i+1}/{len(symbols)}")
            result = self.run_for_symbol(symbol)
            results[symbol] = result
            
            # Clear data cache after each symbol to prevent memory leaks during long runs
            self.data_loader.clear_cache()
        
        # Summary
        total_time = time.time() - total_start
        successful = sum(1 for r in results.values() if r.success)
        validated = sum(1 for r in results.values() if r.config and r.config.is_validated)
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"OPTIMIZATION COMPLETE")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Successful: {successful}/{len(symbols)}")
        logger.info(f"Validated:  {validated}/{len(symbols)}")
        logger.info(f"{'#'*60}")
        
        return results
    
    def save_configs(self, results: Dict[str, PipelineResult], filepath: str):
        """Save validated configurations to JSON file."""
        configs = {}
        for symbol, result in results.items():
            if result.success and result.config:
                configs[symbol] = result.config.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(configs, f, indent=2)
        
        logger.info(f"Saved {len(configs)} configurations to {filepath}")
    
    def load_configs(self, filepath: str) -> Dict[str, SymbolConfig]:
        """Load configurations from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        configs = {}
        for symbol, config_dict in data.items():
            configs[symbol] = SymbolConfig.from_dict(config_dict)
        
        logger.info(f"Loaded {len(configs)} configurations from {filepath}")
        return configs


# =============================================================================
# PORTFOLIO MANAGER
# =============================================================================

class PortfolioManager:
    """
    Main Portfolio Manager class with stateful optimization ledger.
    
    Manages the complete lifecycle:
    1. Initial optimization of all symbols (with skip/resume support)
    2. Continuous monitoring and trading
    3. Automatic retraining when periods expire
    4. Incremental configuration persistence (atomic saves)
    
    Key Features:
    - Skips re-optimizing symbols with valid (non-expired) configs
    - Persists progress after each symbol (never loses work)
    - Atomic writes prevent config corruption
    - Explicit overwrite mode for full re-optimization
    """
    
    def __init__(self, 
                 config: PipelineConfig,
                 symbols: List[str],
                 config_file: str = "pm_configs.json"):
        """
        Initialize Portfolio Manager.
        
        Args:
            config: Pipeline configuration
            symbols: List of symbols to manage
            config_file: Path to configuration file
        """
        self.config = config
        self.symbols = symbols
        self.config_file = Path(config_file)
        
        # Pipeline
        self.pipeline = OptimizationPipeline(config)
        
        # Config ledger for stateful persistence
        self.ledger = ConfigLedger(str(self.config_file))
        
        # Active configurations (loaded from ledger)
        self.symbol_configs: Dict[str, SymbolConfig] = {}
        
        # Load existing configs via ledger
        self._load_configs()
    
    def _load_configs(self):
        """Load configurations from ledger."""
        try:
            self.ledger.load()
            self.symbol_configs = self.ledger.get_all_configs()
        except RuntimeError as e:
            # Corrupted JSON - propagate with clear message
            logger.error(f"Config load failed: {e}")
            raise
        except Exception as e:
            logger.warning(f"Could not load configs: {e}")
            self.symbol_configs = {}
    
    def _save_symbol_config(self, symbol: str, config: SymbolConfig):
        """
        Save a single symbol's config atomically.
        
        This is the incremental persistence method.
        """
        self.symbol_configs[symbol] = config
        if not self.ledger.update_symbol(symbol, config):
            logger.warning(f"Failed to persist config for {symbol}")
    
    def needs_retraining(self, symbol: str) -> bool:
        """
        Check if symbol needs retraining.
        
        Uses ledger validity check.
        """
        try:
            current_version = self.pipeline.build_regime_detection_version(symbol)
        except Exception:
            current_version = None
        should_optimize, _ = self.ledger.should_optimize(
            symbol,
            overwrite=False,
            current_regime_version=current_version,
        )
        return should_optimize
    
    def get_symbols_needing_retrain(self) -> List[str]:
        """Get list of symbols that need retraining."""
        return [s for s in self.symbols if self.needs_retraining(s)]
    
    def retrain_symbol(self, symbol: str, force: bool = False) -> bool:
        """
        Retrain a single symbol.
        
        Args:
            symbol: Symbol to retrain
            force: If True, retrain even if config is valid
            
        Returns:
            True if successful
        """
        # Check if we should skip
        if not force:
            try:
                current_version = self.pipeline.build_regime_detection_version(symbol)
            except Exception:
                current_version = None
            should_optimize, reason = self.ledger.should_optimize(
                symbol,
                overwrite=False,
                current_regime_version=current_version,
            )
            if not should_optimize:
                logger.info(f"SKIP {symbol}: {reason}")
                return True  # Already valid
        
        logger.info(f"RETRAIN {symbol}: running optimization...")
        
        result = self.pipeline.run_for_symbol(symbol)
        
        if result.success and result.config:
            self._save_symbol_config(symbol, result.config)
            logger.info(f"SAVED {symbol}: optimization complete")
            return True
        else:
            # Keep existing config on failure
            logger.warning(f"FAILED {symbol}: keeping existing config")
            return False
    
    def retrain_all_needed(self) -> Dict[str, bool]:
        """
        Retrain all symbols that need it.
        
        Returns:
            Dict mapping symbol to success status
        """
        results = {}
        symbols_to_retrain = self.get_symbols_needing_retrain()
        
        if not symbols_to_retrain:
            logger.info("No symbols need retraining")
            return results
        
        logger.info(f"Retraining {len(symbols_to_retrain)} symbols...")
        
        for i, symbol in enumerate(symbols_to_retrain):
            logger.info(f"Progress: {i+1}/{len(symbols_to_retrain)}")
            results[symbol] = self.retrain_symbol(symbol)
        
        return results
    
    def initial_optimization(self, overwrite: bool = False) -> Dict[str, PipelineResult]:
        """
        Run initial optimization for all symbols with skip/resume support.
        
        This is the main entry point for optimization that:
        - Skips symbols with valid configs (unless overwrite=True)
        - Saves progress after each symbol (incremental persistence)
        - Never loses existing configs on failure
        
        Args:
            overwrite: If True, re-optimize all symbols ignoring validity
            
        Returns:
            Dict of pipeline results (only for symbols that were optimized)
        """
        results = {}
        
        total_start = time.time()
        
        # Determine which symbols to optimize
        current_regime_versions: Dict[str, Dict[str, Any]] = {}
        if not overwrite:
            for sym in self.symbols:
                try:
                    current_regime_versions[sym] = self.pipeline.build_regime_detection_version(sym)
                except Exception:
                    current_regime_versions[sym] = {}
        to_optimize, to_skip = self.ledger.get_symbols_to_optimize(
            self.symbols,
            overwrite=overwrite,
            current_regime_versions=current_regime_versions,
        )
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"PORTFOLIO MANAGER - STATEFUL OPTIMIZATION")
        logger.info(f"Total symbols: {len(self.symbols)}")
        logger.info(f"To optimize: {len(to_optimize)}")
        logger.info(f"Skipping: {len(to_skip)} (valid configs)")
        if overwrite:
            logger.info(f"OVERWRITE MODE: ignoring validity checks")
        logger.info(f"Strategies: {StrategyRegistry.count()}")
        logger.info(f"Timeframes: {self.config.timeframes}")
        logger.info(f"{'#'*60}")
        
        if not to_optimize:
            logger.info("All symbols have valid configs. Nothing to optimize.")
            return results
        
        max_workers = int(getattr(self.config, "optimization_max_workers", 1) or 1)

        if max_workers > 1 and len(to_optimize) > 1:
            cfg_dict = _pipeline_config_to_dict(self.config)
            logger.info(f"Parallel optimization enabled: workers={max_workers}")

            completed = 0
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_optimize_symbol_worker, cfg_dict, symbol): symbol
                    for symbol in to_optimize
                }

                for future in as_completed(future_map):
                    symbol = future_map[future]
                    completed += 1
                    logger.info(f"\nProgress: {completed}/{len(to_optimize)}")

                    try:
                        sym, result = future.result()
                        results[sym] = result
                        symbol_for_save = sym
                    except Exception as e:
                        logger.exception(f"[{symbol}] Optimization error: {e}")
                        results[symbol] = PipelineResult(symbol=symbol, success=False, error_message=str(e))
                        symbol_for_save = symbol

                    # Incremental persistence: save immediately after each symbol
                    result = results.get(symbol_for_save)
                    if result and result.success and result.config:
                        self._save_symbol_config(symbol_for_save, result.config)
                        logger.info(f"SAVED {symbol_for_save} to {self.config_file} (atomic)")
                    else:
                        logger.warning(f"FAILED {symbol_for_save}: keeping existing config (if any)")

            # Clear any in-process cache after parallel run
            self.pipeline.data_loader.clear_cache()
            FeatureComputer.clear_cache()
        else:
            # Process symbols needing optimization sequentially
            for i, symbol in enumerate(to_optimize):
                logger.info(f"\nProgress: {i+1}/{len(to_optimize)}")
                
                try:
                    result = self.pipeline.run_for_symbol(symbol)
                    results[symbol] = result
                    
                    # Incremental persistence: save immediately after each symbol
                    if result.success and result.config:
                        self._save_symbol_config(symbol, result.config)
                        logger.info(f"SAVED {symbol} to {self.config_file} (atomic)")
                    else:
                        # Keep existing config on failure (if any)
                        logger.warning(f"FAILED {symbol}: keeping existing config (if any)")
                    
                except Exception as e:
                    logger.exception(f"[{symbol}] Optimization error: {e}")
                    # Create a failure result but don't overwrite existing config
                    from pm_pipeline import PipelineResult
                    results[symbol] = PipelineResult(symbol=symbol, error_message=str(e))
                
                # Clear data cache after each symbol to prevent memory leaks
                self.pipeline.data_loader.clear_cache()
                FeatureComputer.clear_cache()
        
        # Summary
        total_time = time.time() - total_start
        successful = sum(1 for r in results.values() if r.success)
        validated = sum(1 for r in results.values() if r.config and r.config.is_validated)
        
        # Include skipped symbols in final counts
        total_validated = len(self.get_validated_configs())
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"OPTIMIZATION COMPLETE")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Optimized: {successful}/{len(to_optimize)}")
        logger.info(f"Skipped: {len(to_skip)} (already valid)")
        logger.info(f"Total validated: {total_validated}/{len(self.symbols)}")
        logger.info(f"{'#'*60}")
        
        return results
    
    def get_validated_configs(self) -> Dict[str, SymbolConfig]:
        """Get only validated configurations."""
        # Refresh from ledger to ensure consistency
        self.symbol_configs = self.ledger.get_all_configs()
        return {s: c for s, c in self.symbol_configs.items() if c.is_validated}
    
    def get_active_strategy(self, symbol: str) -> Optional[BaseStrategy]:
        """
        Get active strategy instance for a symbol.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Strategy instance or None
        """
        if symbol not in self.symbol_configs:
            return None
        
        config = self.symbol_configs[symbol]
        
        if not config.is_validated:
            return None
        
        return StrategyRegistry.get(config.strategy_name, **config.parameters)
    
    def get_ledger_stats(self) -> Dict[str, Any]:
        """Get ledger statistics for monitoring."""
        return self.ledger.get_stats()
    
    def print_status(self):
        """Print current portfolio status."""
        stats = self.ledger.get_stats()
        
        print("\n" + "=" * 70)
        print("PORTFOLIO MANAGER STATUS")
        print("=" * 70)
        print(f"Config file: {stats['filepath']}")
        print(f"Symbols managed: {len(self.symbols)}")
        print(f"Configured: {stats['total']}")
        print(f"Valid: {stats['valid']}")
        print(f"Expired: {stats['expired']}")
        print(f"Invalid: {stats['invalid']}")
        print(f"Need optimization: {len(self.get_symbols_needing_retrain())}")
        print()
        
        now = datetime.now()
        for symbol in self.symbols:
            if symbol in self.symbol_configs:
                c = self.symbol_configs[symbol]
                
                # Determine status indicator
                if not c.is_validated:
                    status = "NO"
                elif c.valid_until and now >= c.valid_until:
                    status = "EX"  # Expired
                else:
                    status = "OK"
                
                # Calculate days remaining/expired
                if c.valid_until:
                    days = (c.valid_until - now).days
                    if days >= 0:
                        expire_str = f"{c.valid_until.strftime('%Y-%m-%d')} ({days}d)"
                    else:
                        expire_str = f"EXPIRED {-days}d ago"
                else:
                    expire_str = "N/A"
                
                print(f"  {status} {symbol:8} | {c.strategy_name:25} | {c.timeframe:3} | "
                      f"Score: {c.composite_score:5.1f} | Expires: {expire_str}")
            else:
                print(f"  -- {symbol:8} | Not configured")
        
        print("=" * 70)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    'SymbolConfig',
    'RegimeConfig',
    'PipelineResult',
    # Config persistence
    'ConfigLedger',
    # Optimization stages
    'StrategySelector',
    'HyperparameterOptimizer',
    'RetrainPeriodSelector',
    'Validator',
    'RegimeOptimizer',
    # Main pipeline
    'OptimizationPipeline',
    'PortfolioManager',
    # Optuna utilities (re-exported)
    'OPTUNA_AVAILABLE',
    'is_optuna_available',
    'get_optimization_method',
]


if __name__ == "__main__":
    # Test pipeline
    logging.basicConfig(level=logging.INFO)
    
    # Log optimization method
    logger.info(f"Optimization method: {get_optimization_method()}")
    logger.info(f"Optuna available: {is_optuna_available()}")
    
    config = PipelineConfig(
        data_dir=Path("./data"),
        timeframes=['H1', 'H4'],
        retrain_periods=[14, 30, 60]
    )
    
    pm = PortfolioManager(
        config=config,
        symbols=['EURUSD', 'GBPUSD']
    )
    
    pm.print_status()
    print("\npm_pipeline.py loaded successfully!")

