"""
FX Portfolio Manager - Trading Strategies
==========================================

Contains all trading strategy implementations.
Includes 50 strategies across categories:
- Trend Following (18 strategies)
- Mean Reversion (18 strategies)
- Breakout/Momentum (14 strategies)

Each strategy provides:
- Signal generation (returns -1, 0, 1)
- Stop loss/take profit calculation
- Parameter grid for optimization
- Default parameters
- Feature requirements (for lazy loading optimization)

All strategies share a standardized SL/TP ATR multiplier grid defined in
_GLOBAL_SL_GRID and _GLOBAL_TP_GRID.

Version: 4.1 (Portfolio Manager — efficiency optimizations)
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Set

from pm_core import StrategyCategory, get_instrument_spec, InstrumentSpec, FeatureComputer


# =============================================================================
# GLOBAL SL/TP PARAM GRID — single source of truth for all strategies
# =============================================================================

_GLOBAL_SL_GRID: List[float] = list(np.arange(1.5, 3.5, 0.5))   # [1.5, 2.0, 2.5, 3.0]
_GLOBAL_TP_GRID: List[float] = list(np.arange(1.0, 6.5, 0.5))   # [1.0 .. 6.0]


# =============================================================================
# FEATURE LOOKUP HELPERS (efficiency optimization)
# =============================================================================

def _memoize_series(features: pd.DataFrame, col: str, result: pd.Series) -> None:
    """Memoize computed helper series unless ``features`` is a slice/view."""
    if getattr(features, "_is_copy", None) is not None:
        return
    features.loc[:, col] = result


def _get_ema(features: pd.DataFrame, period: int) -> pd.Series:
    """Get EMA from precomputed features or compute if missing (C3: memoize)."""
    col = f'EMA_{period}'
    if col in features.columns:
        return features[col]
    result = features['Close'].ewm(span=period, adjust=False).mean()
    _memoize_series(features, col, result)
    return result


def _get_sma(features: pd.DataFrame, period: int) -> pd.Series:
    """Get SMA from precomputed features or compute if missing (C3: memoize)."""
    col = f'SMA_{period}'
    if col in features.columns:
        return features[col]
    result = features['Close'].rolling(period).mean()
    _memoize_series(features, col, result)
    return result


def _get_tr(features: pd.DataFrame) -> pd.Series:
    """Get True Range (C2: consolidated helper). Memoized into features."""
    col = '_TR'
    if col in features.columns:
        return features[col]
    high_low = features['High'] - features['Low']
    high_close = (features['High'] - features['Close'].shift(1)).abs()
    low_close = (features['Low'] - features['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    _memoize_series(features, col, tr)
    return tr


def _get_atr(features: pd.DataFrame, period: int) -> pd.Series:
    """Get ATR from precomputed features or compute if missing (C3: memoize)."""
    col = f'ATR_{period}'
    if col in features.columns:
        return features[col]
    tr = _get_tr(features)
    result = tr.rolling(period).mean()
    _memoize_series(features, col, result)
    return result


def _get_rsi(features: pd.DataFrame, period: int) -> pd.Series:
    """Get RSI from precomputed features or compute if missing (C3: memoize)."""
    col = f'RSI_{period}'
    if col in features.columns:
        return features[col]
    result = FeatureComputer.rsi(features['Close'], period)
    _memoize_series(features, col, result)
    return result


def _get_keltner(features: pd.DataFrame, ema_period: int = 20,
                 atr_period: int = 14, mult: float = 2.0
                 ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Get Keltner Channel (mid, upper, lower). C2 consolidation."""
    mid = _get_ema(features, ema_period)
    atr = _get_atr(features, atr_period)
    upper = mid + mult * atr
    lower = mid - mult * atr
    return mid, upper, lower


def _detect_swing_points(series: pd.Series, order: int = 5
                         ) -> Tuple[pd.Series, pd.Series]:
    """Detect swing highs and lows using rolling window comparison (D0 helper).

    Returns boolean Series pair (swing_highs, swing_lows).
    A swing high at bar i means series[i] >= all values in [i-order, i+order].
    Shifted by `order` to avoid lookahead.
    """
    swing_highs = pd.Series(False, index=series.index)
    swing_lows = pd.Series(False, index=series.index)
    vals = series.values
    n = len(vals)
    for i in range(order, n - order):
        window = vals[i - order: i + order + 1]
        if np.all(np.isnan(window)):
            continue
        if vals[i] == np.nanmax(window):
            swing_highs.iloc[i + order] = True  # delayed by order bars
        if vals[i] == np.nanmin(window):
            swing_lows.iloc[i + order] = True
    return swing_highs, swing_lows


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank (0-100) of current value within its window (D0 helper)."""
    def _pct_rank(arr):
        if len(arr) < 2:
            return 50.0
        current = arr[-1]
        return (np.sum(arr[:-1] < current) / (len(arr) - 1)) * 100.0
    return series.rolling(window, min_periods=window).apply(_pct_rank, raw=True)


def _get_adx_di(features: pd.DataFrame, period: int = 14
                ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Get ADX, +DI, -DI from precomputed features or compute if missing (memoized).

    Uses Wilder-style EMA (alpha=1/period) and reuses _get_tr() for True Range
    to avoid redundant TR computation across strategies.
    """
    adx_col = f'_ADX_{period}'
    pdi_col = f'_PLUS_DI_{period}'
    mdi_col = f'_MINUS_DI_{period}'

    # Return cached if available
    if adx_col in features.columns:
        return features[adx_col], features[pdi_col], features[mdi_col]

    # Also accept standard column names for period=14
    if period == 14 and {'ADX', 'PLUS_DI', 'MINUS_DI'}.issubset(features.columns):
        return features['ADX'], features['PLUS_DI'], features['MINUS_DI']

    high = features['High']
    low = features['Low']

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _get_tr(features)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / (atr + 1e-10))

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    _memoize_series(features, adx_col, adx)
    _memoize_series(features, pdi_col, plus_di)
    _memoize_series(features, mdi_col, minus_di)
    return adx, plus_di, minus_di


def _get_hull_ma(features: pd.DataFrame, period: int) -> pd.Series:
    """Get Hull MA from precomputed features or compute if missing (C3: memoize)."""
    col = f'HULL_MA_{period}' if period != 20 else 'HULL_MA'
    if col in features.columns:
        return features[col]
    result = FeatureComputer.hull_ma(features['Close'], period)
    _memoize_series(features, col, result)
    return result


def _get_vortex(features: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series]:
    """Compute Vortex Indicator VI+ and VI- (memoized).

    VI+ = sum(|High_i - Low_{i-1}|, period) / sum(TR, period)
    VI- = sum(|Low_i - High_{i-1}|, period) / sum(TR, period)
    """
    vip_col = f'_VI_PLUS_{period}'
    vim_col = f'_VI_MINUS_{period}'
    if vip_col in features.columns:
        return features[vip_col], features[vim_col]
    high = features['High']
    low = features['Low']
    tr = _get_tr(features)
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    tr_sum = tr.rolling(period, min_periods=period).sum()
    vi_plus = vm_plus.rolling(period, min_periods=period).sum() / (tr_sum + 1e-10)
    vi_minus = vm_minus.rolling(period, min_periods=period).sum() / (tr_sum + 1e-10)
    _memoize_series(features, vip_col, vi_plus)
    _memoize_series(features, vim_col, vi_minus)
    return vi_plus, vi_minus


def _get_swing_state(features: pd.DataFrame, order: int
                     ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Get confirmed swing marks and forward-filled swing price levels (memoized).

    Detects swing highs on High series and swing lows on Low series separately,
    then forward-fills the last confirmed swing price for use at each bar.

    Returns: (swing_high_marks, swing_low_marks, swing_high_prices, swing_low_prices)
    """
    sh_col = f'_SWS_H_{order}'
    sl_col = f'_SWS_L_{order}'
    shp_col = f'_SWS_HP_{order}'
    slp_col = f'_SWS_LP_{order}'
    if sh_col in features.columns:
        return features[sh_col], features[sl_col], features[shp_col], features[slp_col]

    sh_marks, _ = _detect_swing_points(features['High'], order)
    _, sl_marks = _detect_swing_points(features['Low'], order)
    # Swing marks are delayed by `order` bars for confirmation. Map each mark to
    # the originating pivot price via shift(order), then forward-fill last pivot.
    sh_prices = features['High'].shift(order).where(sh_marks).ffill()
    sl_prices = features['Low'].shift(order).where(sl_marks).ffill()

    _memoize_series(features, sh_col, sh_marks)
    _memoize_series(features, sl_col, sl_marks)
    _memoize_series(features, shp_col, sh_prices)
    _memoize_series(features, slp_col, sl_prices)
    return sh_marks, sl_marks, sh_prices, sl_prices


def _get_trix(features: pd.DataFrame, period: int, signal_period: int
              ) -> Tuple[pd.Series, pd.Series]:
    """Compute TRIX line and signal line (memoized).

    TRIX = 100 * pct_change(EMA(EMA(EMA(close, period), period), period))
    Signal = EMA(TRIX, signal_period)
    """
    trix_col = f'_TRIX_{period}'
    sig_col = f'_TRIX_SIG_{period}_{signal_period}'
    if trix_col in features.columns and sig_col in features.columns:
        return features[trix_col], features[sig_col]
    close = features['Close']
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = ema3.pct_change() * 100.0
    signal_line = trix.ewm(span=signal_period, adjust=False).mean()
    _memoize_series(features, trix_col, trix)
    _memoize_series(features, sig_col, signal_line)
    return trix, signal_line


def _get_elder_power(features: pd.DataFrame, ema_period: int, smooth_period: int = 1
                     ) -> Tuple[pd.Series, pd.Series]:
    """Compute Elder Ray Bull Power and Bear Power (memoized).

    Bull Power = High - EMA(close, ema_period)
    Bear Power = Low - EMA(close, ema_period)
    Optional smoothing via EMA of power values.
    """
    bp_col = f'_BULL_PWR_{ema_period}_{smooth_period}'
    brp_col = f'_BEAR_PWR_{ema_period}_{smooth_period}'
    if bp_col in features.columns:
        return features[bp_col], features[brp_col]
    ema = _get_ema(features, ema_period)
    bull_power = features['High'] - ema
    bear_power = features['Low'] - ema
    if smooth_period > 1:
        bull_power = bull_power.ewm(span=smooth_period, adjust=False).mean()
        bear_power = bear_power.ewm(span=smooth_period, adjust=False).mean()
    _memoize_series(features, bp_col, bull_power)
    _memoize_series(features, brp_col, bear_power)
    return bull_power, bear_power


def _get_bb(features: pd.DataFrame, period: int, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Get Bollinger Bands from precomputed features or compute if missing."""
    mid_col = f'BB_MID_{period}'
    if mid_col in features.columns:
        return features[mid_col], features[f'BB_UPPER_{period}'], features[f'BB_LOWER_{period}']
    return FeatureComputer.bollinger_bands(features['Close'], period, std)


def _get_stochastic(features: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Get Stochastic from precomputed features or compute if missing."""
    if 'STOCH_K' in features.columns and k_period == 14:
        return features['STOCH_K'], features['STOCH_D']
    return FeatureComputer.stochastic(features, k_period, d_period)


def _get_macd(features: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Get MACD from precomputed features or compute if missing."""
    if 'MACD' in features.columns and fast == 12 and slow == 26 and signal == 9:
        return features['MACD'], features['MACD_SIGNAL'], features['MACD_HIST']
    return FeatureComputer.macd(features['Close'], fast, slow, signal)


# =============================================================================
# BASE STRATEGY CLASS
# =============================================================================

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement:
    - name: Strategy identifier
    - category: Strategy category (trend, mean_reversion, breakout)
    - get_default_params(): Default parameter values
    - generate_signals(): Signal generation logic
    
    Optional:
    - get_required_features(): List of feature columns needed (for lazy loading)
    """
    
    def __init__(self, **params):
        """
        Initialize strategy with parameters.
        
        Args:
            **params: Strategy parameters (overrides defaults)
        """
        self.params = {}
        self._default_params = self.get_default_params()
        
        # Start with defaults
        for key, value in self._default_params.items():
            self.params[key] = value
        
        # Override with provided params
        for key, value in params.items():
            self.params[key] = value
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name/identifier."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> StrategyCategory:
        """Strategy category."""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameter values."""
        pass
    
    @abstractmethod
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        """
        Generate trading signals.
        
        Args:
            features: DataFrame with OHLCV and indicators
            symbol: Symbol being traded
            
        Returns:
            Series of signals: -1 (short), 0 (flat), 1 (long)
        """
        pass
    
    MIN_BARS: int = 50

    @staticmethod
    def _zero_warmup(signals: pd.Series, warmup: int) -> pd.Series:
        """Zero out the first ``warmup`` bars to prevent NaN-derived signals."""
        if warmup > 0 and len(signals) > warmup:
            signals.iloc[:warmup] = 0
        return signals

    def get_required_features(self) -> Set[str]:
        """
        Get set of feature columns this strategy requires.

        Override in subclasses for lazy feature loading optimization.
        Returns empty set by default (compute all features).

        Returns:
            Set of column names required by this strategy
        """
        return set()
    
    def calculate_stops(self, features: pd.DataFrame,
                        signal: int, symbol: str,
                        spec: Optional[InstrumentSpec] = None,
                        bar_index: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit in pips.
        
        Default implementation uses ATR-based stops.
        
        Args:
            features: DataFrame with OHLCV and indicators (full or sliced)
            signal: Direction (1 for long, -1 for short)
            symbol: Symbol being traded
            spec: Optional InstrumentSpec
            bar_index: Optional bar index (if provided, features is the full DataFrame
                       and we read values at bar_index using .iat for O(1) access)
        
        Returns:
            Tuple of (stop_loss_pips, take_profit_pips)
        """
        # Get ATR (use bar_index with .iat for O(1) access in backtests)
        atr_col = 'ATR_14'
        if atr_col in features.columns:
            if bar_index is not None:
                atr = float(features[atr_col].iat[bar_index])
            else:
                atr = float(features[atr_col].iloc[-1])
        else:
            # Fallback (should be rare if FeatureComputer is producing ATR_14)
            if bar_index is None:
                _df = features
            else:
                _df = features.iloc[:bar_index + 1]
            high_low = _df['High'] - _df['Low']
            high_close = abs(_df['High'] - _df['Close'].shift(1))
            low_close = abs(_df['Low'] - _df['Close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])

        if spec is None:
            spec = get_instrument_spec(symbol)
        atr_pips = spec.price_to_pips(atr)
        sl_mult = self.params.get('sl_atr_mult', 2.0)
        tp_mult = self.params.get('tp_atr_mult', 3.0)
        
        sl_pips = max(5.0, atr_pips * sl_mult)
        tp_pips = max(10.0, atr_pips * tp_mult)
        
        return sl_pips, tp_pips
    
    def get_param_grid(self) -> Dict[str, List]:
        """Get parameter grid for optimization."""
        return {}
    
    @staticmethod
    def _base_sl_tp_grid() -> Dict[str, List[float]]:
        """Return the global SL/TP grid entries for use in get_param_grid()."""
        return {
            'sl_atr_mult': _GLOBAL_SL_GRID,
            'tp_atr_mult': _GLOBAL_TP_GRID,
        }
    
    def set_params(self, **params):
        """Update strategy parameters."""
        for key, value in params.items():
            self.params[key] = value
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self.params.copy()
    
    def __repr__(self) -> str:
        return f"{self.name}({self.params})"


# =============================================================================
# TREND FOLLOWING STRATEGIES
# =============================================================================

class EMACrossoverStrategy(BaseStrategy):
    """EMA Crossover Strategy - Fast/Slow EMA crossovers."""
    
    @property
    def name(self) -> str:
        return "EMACrossoverStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'fast_period': 10,
            'slow_period': 20,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0
        }
    
    def get_required_features(self) -> Set[str]:
        """EMA crossover needs EMA columns for fast and slow periods."""
        fast = self.params.get('fast_period', 10)
        slow = self.params.get('slow_period', 20)
        return {f'EMA_{fast}', f'EMA_{slow}'}
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        fast = self.params.get('fast_period', 10)
        slow = self.params.get('slow_period', 20)
        
        # Use precomputed EMAs if available (efficiency optimization)
        fast_ema = _get_ema(features, fast)
        slow_ema = _get_ema(features, slow)
        
        signals = pd.Series(0, index=features.index)
        cross_above = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        cross_below = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        
        signals[cross_above] = 1
        signals[cross_below] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'fast_period': [5, 8, 10, 13, 15],
            'slow_period': [20, 26, 30, 40, 50],
            **self._base_sl_tp_grid(),
        }


class SupertrendStrategy(BaseStrategy):
    """Supertrend Strategy - ATR-based trend following with band continuation.
    
    Signals fire on direction change (flip) only, not continuously.
    Implements proper recursive band continuation rules.
    """
    
    @property
    def name(self) -> str:
        return "SupertrendStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 10,
            'multiplier': 3.0,
            'sl_atr_mult': 1.5,
            'tp_atr_mult': 3.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('atr_period', 10)
        mult = self.params.get('multiplier', 3.0)
        
        # Use pre-computed ATR if available (efficiency optimization)
        atr_col = f'ATR_{period}'
        if atr_col in features.columns:
            atr = features[atr_col]
        elif 'ATR_14' in features.columns and period == 14:
            atr = features['ATR_14']
        else:
            # Calculate ATR
            high_low = features['High'] - features['Low']
            high_close = abs(features['High'] - features['Close'].shift(1))
            low_close = abs(features['Low'] - features['Close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
        
        # Calculate bands
        hl2 = (features['High'] + features['Low']) / 2
        upperband = (hl2 + (mult * atr)).values
        lowerband = (hl2 - (mult * atr)).values
        close = features['Close'].values
        n = len(features)
        
        # Track direction with continuation rules - OPTIMIZED with NumPy
        direction = np.zeros(n, dtype=np.int32)
        final_upper = upperband.copy()
        final_lower = lowerband.copy()
        
        for i in range(1, n):
            # Band continuation rules (must remain sequential due to dependency)
            if lowerband[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = lowerband[i]
            else:
                final_lower[i] = final_lower[i-1]
            
            if upperband[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = upperband[i]
            else:
                final_upper[i] = final_upper[i-1]
            
            # Determine direction
            if close[i] > final_upper[i-1]:
                direction[i] = 1
            elif close[i] < final_lower[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
        
        # Generate signals on direction change - vectorized
        signals = np.zeros(n, dtype=np.int32)
        dir_change = np.diff(direction, prepend=0)
        signals[dir_change == 2] = 1   # -1 to 1
        signals[dir_change == -2] = -1  # 1 to -1
        
        return pd.Series(signals, index=features.index)
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'atr_period': [7, 10, 14, 20, 25],
            'multiplier': [1.5, 2.0, 2.5, 3.0, 4.0],
            **self._base_sl_tp_grid(),
        }


class MACDTrendStrategy(BaseStrategy):
    """MACD Trend Strategy - MACD/Signal line crossovers."""
    
    @property
    def name(self) -> str:
        return "MACDTrendStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0
        }
    
    def get_required_features(self) -> Set[str]:
        """MACD strategy uses precomputed MACD columns for default params."""
        fast = self.params.get('fast_period', 12)
        slow = self.params.get('slow_period', 26)
        sig = self.params.get('signal_period', 9)
        if fast == 12 and slow == 26 and sig == 9:
            return {'MACD', 'MACD_SIGNAL'}
        return set()
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        fast = self.params.get('fast_period', 12)
        slow = self.params.get('slow_period', 26)
        sig = self.params.get('signal_period', 9)
        
        # Use precomputed MACD if available (efficiency optimization)
        macd, signal_line, _ = _get_macd(features, fast, slow, sig)
        
        signals = pd.Series(0, index=features.index)
        cross_above = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        cross_below = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        
        signals[cross_above] = 1
        signals[cross_below] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'fast_period': [8, 10, 12, 16],
            'slow_period': [21, 26, 30, 34],
            'signal_period': [7, 9, 12],
            **self._base_sl_tp_grid(),
        }


class ADXTrendStrategy(BaseStrategy):
    """ADX Trend Strategy - DI crossovers with ADX strength filter.
    
    Entry-event based: fires on +DI/-DI cross when ADX confirms
    trend strength above threshold.
    """
    
    @property
    def name(self) -> str:
        return "ADXTrendStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'adx_threshold': 25,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('adx_period', 14)
        threshold = self.params.get('adx_threshold', 25)
        
        # Calculate +DM and -DM
        plus_dm = features['High'].diff()
        minus_dm = -features['Low'].diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Calculate ATR
        high_low = features['High'] - features['Low']
        high_close = abs(features['High'] - features['Close'].shift(1))
        low_close = abs(features['Low'] - features['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        signals = pd.Series(0, index=features.index)
        buy_signal = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1)) & (adx > threshold)
        sell_signal = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1)) & (adx > threshold)
        
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'adx_period': [7, 10, 14, 20, 25],
            'adx_threshold': [15, 20, 25, 30, 35],
            **self._base_sl_tp_grid(),
        }


class IchimokuStrategy(BaseStrategy):
    """Ichimoku Cloud Strategy - Tenkan/Kijun crossovers with cloud filter."""
    
    @property
    def name(self) -> str:
        return "IchimokuStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        tenkan_p = self.params.get('tenkan_period', 9)
        kijun_p = self.params.get('kijun_period', 26)
        senkou_b_p = self.params.get('senkou_b_period', 52)
        
        tenkan = (features['High'].rolling(tenkan_p).max() + 
                  features['Low'].rolling(tenkan_p).min()) / 2
        kijun = (features['High'].rolling(kijun_p).max() + 
                 features['Low'].rolling(kijun_p).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_p)
        senkou_b = ((features['High'].rolling(senkou_b_p).max() + 
                     features['Low'].rolling(senkou_b_p).min()) / 2).shift(kijun_p)
        
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        
        signals = pd.Series(0, index=features.index)
        above_cloud = features['Close'] > cloud_top
        below_cloud = features['Close'] < cloud_bottom
        tenkan_cross_up = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
        tenkan_cross_down = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
        
        signals[above_cloud & tenkan_cross_up] = 1
        signals[below_cloud & tenkan_cross_down] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'tenkan_period': [7, 9, 12, 15, 18],
            'kijun_period': [18, 22, 26, 30, 34],
            **self._base_sl_tp_grid(),
        }


class HullMATrendStrategy(BaseStrategy):
    """Hull MA Trend Strategy - Hull MA direction changes.
    
    Uses the optimized cached Hull MA implementation from FeatureComputer.
    """
    
    @property
    def name(self) -> str:
        return "HullMATrendStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 20,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0
        }
    
    def get_required_features(self) -> Set[str]:
        """Hull MA strategy uses precomputed HULL_MA for period 20."""
        period = self.params.get('period', 20)
        if period == 20:
            return {'HULL_MA'}
        return set()
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('period', 20)
        
        # Use precomputed Hull MA if available (efficiency optimization)
        hma = _get_hull_ma(features, period)
        
        signals = pd.Series(0, index=features.index)
        hma_diff = hma.diff()
        turn_up = (hma_diff > 0) & (hma_diff.shift(1) <= 0)
        turn_down = (hma_diff < 0) & (hma_diff.shift(1) >= 0)
        
        signals[turn_up] = 1
        signals[turn_down] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'period': [9, 14, 20, 30, 50],
            **self._base_sl_tp_grid(),
        }


class AroonTrendStrategy(BaseStrategy):
    """Aroon Trend Timing Strategy — "Time since highs/lows" trend detection.

    Aroon Up   = 100 * (period - bars_since_highest) / period
    Aroon Down = 100 * (period - bars_since_lowest)  / period

    Long entry:  Aroon Up crosses above Aroon Down AND Aroon Up > strength_level
    Short entry: Aroon Down crosses above Aroon Up AND Aroon Down > strength_level

    Signals fire on the crossover event only (not continuously).
    """

    @property
    def name(self) -> str:
        return "AroonTrendStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 25,
            'strength_level': 70,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
        }

    @staticmethod
    def _aroon(high: pd.Series, low: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
        """Compute Aroon Up and Aroon Down without lookahead.

        Uses a rolling window of ``period + 1`` bars (current bar inclusive).
        ``bars_since_high`` = number of bars since the highest high in the window.
        Aroon Up = 100 * (period - bars_since_high) / period.
        """
        aroon_up = pd.Series(np.nan, index=high.index)
        aroon_down = pd.Series(np.nan, index=low.index)

        high_vals = high.to_numpy(dtype=float)
        low_vals = low.to_numpy(dtype=float)
        n = len(high_vals)

        for i in range(period, n):
            window_h = high_vals[i - period: i + 1]
            window_l = low_vals[i - period: i + 1]
            # argmax/argmin give index from window start; bars_since = period - idx
            bars_since_high = period - int(np.argmax(window_h))
            bars_since_low = period - int(np.argmin(window_l))
            aroon_up.iat[i] = 100.0 * (period - bars_since_high) / period
            aroon_down.iat[i] = 100.0 * (period - bars_since_low) / period

        return aroon_up, aroon_down

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = int(self.params.get('period', 25))
        strength = float(self.params.get('strength_level', 70))

        aroon_up, aroon_down = self._aroon(features['High'], features['Low'], period)

        cross_up = (aroon_up > aroon_down) & (aroon_up.shift(1) <= aroon_down.shift(1))
        cross_down = (aroon_down > aroon_up) & (aroon_down.shift(1) <= aroon_up.shift(1))

        signals = pd.Series(0, index=features.index)
        signals[cross_up & (aroon_up > strength)] = 1
        signals[cross_down & (aroon_down > strength)] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'period': [14, 20, 25, 30, 50],
            'strength_level': [50, 60, 70, 80, 90],
            **self._base_sl_tp_grid(),
        }


class ADXDIStrengthStrategy(BaseStrategy):
    """ADX + DI Directional Strength Trend Entry.

    A regime-confirmed trend entry that triggers on DI dominance confirmed
    by strong ADX.  Distinguished from ADXTrendStrategy by requiring
    sustained DI dominance (DI spread > di_spread_min) rather than just
    the cross event, and by using a rising-ADX filter.

    Long:  +DI crosses above -DI, ADX > adx_threshold, +DI - -DI > di_spread_min
    Short: -DI crosses above +DI, ADX > adx_threshold, -DI - +DI > di_spread_min

    Entry-event based: fires on the cross bar only.
    """

    @property
    def name(self) -> str:
        return "ADXDIStrengthStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'adx_period': 14,
            'adx_threshold': 20,
            'di_spread_min': 5.0,
            'require_adx_rising': True,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
        }

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = int(self.params.get('adx_period', 14))
        adx_th = float(self.params.get('adx_threshold', 22))
        spread_min = float(self.params.get('di_spread_min', 5.0))
        adx_rising = bool(self.params.get('require_adx_rising', True))

        adx, plus_di, minus_di = _get_adx_di(features, period)

        # DI crosses
        cross_up = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
        cross_down = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))

        # Strength filters
        strong_adx = adx > adx_th
        long_spread = (plus_di - minus_di) > spread_min
        short_spread = (minus_di - plus_di) > spread_min

        if adx_rising:
            adx_up = adx > adx.shift(1)
            strong_adx = strong_adx & adx_up

        signals = pd.Series(0, index=features.index)
        signals[cross_up & strong_adx & long_spread] = 1
        signals[cross_down & strong_adx & short_spread] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'adx_period': [7, 10, 14, 20],
            'adx_threshold': [15, 20, 25, 30],
            'di_spread_min': [0.0, 5.0, 10.0, 15.0],
            'require_adx_rising': [True, False],
            **self._base_sl_tp_grid(),
        }


class KeltnerPullbackStrategy(BaseStrategy):
    """Keltner Pullback Continuation Strategy.

    Trend continuation that enters on a pullback rather than a breakout.

    Logic:
    1. Trend definition: EMA slope (EMA > EMA shifted = uptrend) AND
       price above EMA for longs (below for shorts).
    2. Pullback: prior bar's low dipped to the EMA/Keltner midline or
       lower band (upper band for shorts).
    3. Confirmation: current bar closes back in trend direction.

    Signal-only strategy; ATR SL/TP stays centralized via calculate_stops.
    """

    @property
    def name(self) -> str:
        return "KeltnerPullbackStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'kc_period': 20,
            'kc_mult': 2.0,
            'ema_slope_bars': 5,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
        }

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        kc_period = int(self.params.get('kc_period', 20))
        kc_mult = float(self.params.get('kc_mult', 2.0))
        slope_bars = int(self.params.get('ema_slope_bars', 5))

        close = features['Close']

        mid, upper, lower = _get_keltner(features, kc_period, kc_period, kc_mult)

        # Trend filter: EMA rising/falling and price on correct side
        ema_rising = mid > mid.shift(slope_bars)
        ema_falling = mid < mid.shift(slope_bars)
        above_ema = close > mid
        below_ema = close < mid

        # Pullback detection: prior bar touched the midline or lower/upper band
        long_pullback = features['Low'].shift(1) <= mid.shift(1)
        short_pullback = features['High'].shift(1) >= mid.shift(1)

        # Confirmation: current bar closes in trend direction
        bullish_close = close > close.shift(1)
        bearish_close = close < close.shift(1)

        signals = pd.Series(0, index=features.index)
        signals[ema_rising & above_ema & long_pullback & bullish_close] = 1
        signals[ema_falling & below_ema & short_pullback & bearish_close] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'kc_period': [10, 15, 20, 25, 30],
            'kc_mult': [1.0, 1.5, 2.0, 2.5],
            'ema_slope_bars': [3, 5, 8, 10],
            **self._base_sl_tp_grid(),
        }


# =============================================================================
# MEAN REVERSION STRATEGIES
# =============================================================================

class RSIExtremesStrategy(BaseStrategy):
    """RSI Extremes Strategy - Oversold/overbought reversals."""
    
    @property
    def name(self) -> str:
        return "RSIExtremesStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 2.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('rsi_period', 14)
        oversold = self.params.get('oversold', 30)
        overbought = self.params.get('overbought', 70)

        rsi = _get_rsi(features, period)
        
        signals = pd.Series(0, index=features.index)
        buy_signal = (rsi > oversold) & (rsi.shift(1) <= oversold)
        sell_signal = (rsi < overbought) & (rsi.shift(1) >= overbought)
        
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'rsi_period': [5, 7, 10, 14, 21],
            'oversold': [15, 20, 25, 30],
            'overbought': [70, 75, 80, 85],
            **self._base_sl_tp_grid(),
        }


class BollingerBounceStrategy(BaseStrategy):
    """Bollinger Bands Bounce Strategy - Mean reversion from band touches."""
    
    @property
    def name(self) -> str:
        return "BollingerBounceStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 20,
            'std_dev': 2.0,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 2.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('period', 20)
        std_dev = self.params.get('std_dev', 2.0)
        
        mid = features['Close'].rolling(period).mean()
        std = features['Close'].rolling(period).std()
        upper = mid + (std_dev * std)
        lower = mid - (std_dev * std)
        
        signals = pd.Series(0, index=features.index)
        touch_lower = features['Low'] <= lower
        bounce_up = features['Close'] > features['Close'].shift(1)
        touch_upper = features['High'] >= upper
        bounce_down = features['Close'] < features['Close'].shift(1)
        
        signals[touch_lower & bounce_up] = 1
        signals[touch_upper & bounce_down] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'period': [10, 15, 20, 25, 30],
            'std_dev': [1.5, 1.75, 2.0, 2.25, 2.5],
            **self._base_sl_tp_grid(),
        }


class ZScoreMRStrategy(BaseStrategy):
    """Z-Score Mean Reversion Strategy - Statistical mean reversion."""
    
    @property
    def name(self) -> str:
        return "ZScoreMRStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 20,
            'entry_z': 2.0,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 2.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('period', 20)
        entry_z = self.params.get('entry_z', 2.0)
        
        mean = features['Close'].rolling(period).mean()
        std = features['Close'].rolling(period).std()
        z_score = (features['Close'] - mean) / (std + 1e-10)
        
        signals = pd.Series(0, index=features.index)
        buy_signal = (z_score > -entry_z) & (z_score.shift(1) <= -entry_z)
        sell_signal = (z_score < entry_z) & (z_score.shift(1) >= entry_z)
        
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'period': [10, 15, 20, 30, 40],
            'entry_z': [1.5, 1.75, 2.0, 2.25, 2.5],
            **self._base_sl_tp_grid(),
        }


class StochasticReversalStrategy(BaseStrategy):
    """Stochastic Reversal Strategy - %K/%D crossovers in extreme zones."""
    
    @property
    def name(self) -> str:
        return "StochasticReversalStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'k_period': 14,
            'd_period': 3,
            'oversold': 20,
            'overbought': 80,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 2.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        k_period = self.params.get('k_period', 14)
        d_period = self.params.get('d_period', 3)
        oversold = self.params.get('oversold', 20)
        overbought = self.params.get('overbought', 80)
        
        low_min = features['Low'].rolling(k_period).min()
        high_max = features['High'].rolling(k_period).max()
        denom = high_max - low_min
        stoch_k = pd.Series(50.0, index=features.index)
        valid = denom > 1e-8
        stoch_k[valid] = 100 * (features['Close'][valid] - low_min[valid]) / denom[valid]
        stoch_d = stoch_k.rolling(d_period).mean()
        
        signals = pd.Series(0, index=features.index)
        # %K crosses above %D in oversold zone
        buy_signal = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)) & (stoch_k < oversold + 10)
        # %K crosses below %D in overbought zone
        sell_signal = (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)) & (stoch_k > overbought - 10)
        
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'k_period': [5, 9, 14, 21],
            'd_period': [3, 5, 7],
            'oversold': [10, 15, 20, 25],
            'overbought': [75, 80, 85, 90],
            **self._base_sl_tp_grid(),
        }


class CCIReversalStrategy(BaseStrategy):
    """CCI Reversal Strategy - Commodity Channel Index reversals."""
    
    @property
    def name(self) -> str:
        return "CCIReversalStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'cci_period': 20,
            'oversold': -100,
            'overbought': 100,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 2.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('cci_period', 20)
        oversold = self.params.get('oversold', -100)
        overbought = self.params.get('overbought', 100)
        
        tp = (features['High'] + features['Low'] + features['Close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad + 1e-10)
        
        signals = pd.Series(0, index=features.index)
        buy_signal = (cci > oversold) & (cci.shift(1) <= oversold)
        sell_signal = (cci < overbought) & (cci.shift(1) >= overbought)
        
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'cci_period': [10, 14, 20, 30],
            'oversold': [-150, -125, -100, -80],
            'overbought': [80, 100, 125, 150],
            **self._base_sl_tp_grid(),
        }


class WilliamsRStrategy(BaseStrategy):
    """Williams %R Strategy - Williams percent range reversals."""
    
    @property
    def name(self) -> str:
        return "WilliamsRStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 14,
            'oversold': -80,
            'overbought': -20,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 2.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('period', 14)
        oversold = self.params.get('oversold', -80)
        overbought = self.params.get('overbought', -20)
        
        high_max = features['High'].rolling(period).max()
        low_min = features['Low'].rolling(period).min()
        denom = high_max - low_min
        willr = pd.Series(-50.0, index=features.index)
        valid = denom > 1e-8
        willr[valid] = -100 * (high_max[valid] - features['Close'][valid]) / denom[valid]
        
        signals = pd.Series(0, index=features.index)
        buy_signal = (willr > oversold) & (willr.shift(1) <= oversold)
        sell_signal = (willr < overbought) & (willr.shift(1) >= overbought)
        
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'period': [7, 10, 14, 21, 28],
            'oversold': [-95, -90, -85, -80],
            'overbought': [-20, -15, -10, -5],
            **self._base_sl_tp_grid(),
        }


class FisherTransformMRStrategy(BaseStrategy):
    """Fisher Transform Mean Reversion Strategy.

    Computes the proper clamped/recursive Fisher Transform of a normalised
    price oscillator, then triggers entries on the Fisher crossing back
    inside an extreme threshold.

    The Fisher Transform maps bounded values to an approximately Gaussian
    distribution, producing sharper turning-point signals than RSI/Stoch.

    Long:  Fisher crosses up through -threshold (was <= -threshold, now > -threshold)
    Short: Fisher crosses down through +threshold (was >= +threshold, now < +threshold)

    Best suited for mean-reversion / non-trending regimes.
    """

    @property
    def name(self) -> str:
        return "FisherTransformMRStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 10,
            'threshold': 1.5,
            'signal_period': 1,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 2.0,
        }

    @staticmethod
    def _fisher_transform(high: pd.Series, low: pd.Series, close: pd.Series,
                          period: int) -> Tuple[pd.Series, pd.Series]:
        """Compute recursive Fisher Transform.

        Steps:
        1. Normalise price to [-1, +1] using rolling highest/lowest over *period*.
        2. Apply EMA smoothing (alpha 0.5) to the normalised value.
        3. Clamp to (-0.999, +0.999) to prevent log blow-ups.
        4. Apply the Fisher Transform: fisher = 0.5 * ln((1+x)/(1-x))
           using recursive smoothing: fisher[i] = 0.5*transform + 0.5*fisher[i-1]
        5. Signal line = fisher shifted by 1 bar.
        """
        hl2 = (high + low) / 2.0
        highest = high.rolling(period, min_periods=period).max()
        lowest = low.rolling(period, min_periods=period).min()
        raw = 2.0 * ((hl2 - lowest) / (highest - lowest + 1e-10)) - 1.0

        # EMA smooth the normalised value (alpha=0.5 gives quick response)
        smooth = raw.ewm(alpha=0.5, min_periods=1, adjust=False).mean()

        # Clamp to avoid log blow-ups
        clamped = smooth.clip(-0.999, 0.999)

        # Recursive Fisher Transform
        fisher_vals = np.zeros(len(clamped))
        clamped_np = clamped.to_numpy(dtype=float)
        for i in range(len(clamped_np)):
            if np.isnan(clamped_np[i]):
                fisher_vals[i] = 0.0
            else:
                transform = 0.5 * np.log((1.0 + clamped_np[i]) / (1.0 - clamped_np[i]))
                fisher_vals[i] = 0.5 * transform + 0.5 * (fisher_vals[i - 1] if i > 0 else 0.0)

        fisher = pd.Series(fisher_vals, index=high.index)
        signal = fisher.shift(1)
        return fisher, signal

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = int(self.params.get('period', 10))
        threshold = float(self.params.get('threshold', 1.5))
        sig_period = int(self.params.get('signal_period', 1))

        fisher, fisher_signal = self._fisher_transform(
            features['High'], features['Low'], features['Close'], period
        )

        if sig_period > 1:
            fisher_signal = fisher.rolling(sig_period).mean().shift(1)

        # Cross back inside threshold (mean-reversion entry)
        long_cross = (fisher > -threshold) & (fisher.shift(1) <= -threshold)
        short_cross = (fisher < threshold) & (fisher.shift(1) >= threshold)

        signals = pd.Series(0, index=features.index)
        signals[long_cross] = 1
        signals[short_cross] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'period': [5, 8, 10, 14, 20],
            'threshold': [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
            'signal_period': [1, 2, 3],
            **self._base_sl_tp_grid(),
        }


class ZScoreVWAPReversionStrategy(BaseStrategy):
    """Z-Score / VWAP Deviation Mean Reversion Strategy.

    "Fair value reversion" using rolling VWAP (tick-volume weighted if
    available) and a z-score of the deviation.

    Entry on a cross back inside the z-threshold (instead of catching the
    knife at max deviation).  Optional ADX filter to prefer ranging regimes
    (ADX below threshold).

    Long:  z crosses up through -entry_z
    Short: z crosses down through +entry_z

    Uses precomputed VWAP column if present, else computes from Volume.
    """

    @property
    def name(self) -> str:
        return "ZScoreVWAPReversionStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'vwap_window': 50,
            'z_window': 50,
            'entry_z': 2.0,
            'use_adx_filter': True,
            'adx_threshold': 25,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 2.0,
        }

    @staticmethod
    def _rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
        """Compute rolling VWAP.  Prefer precomputed column if available."""
        col = f'VWAP_{window}'
        if col in df.columns:
            return df[col]

        vol = pd.to_numeric(df.get('Volume', pd.Series(1.0, index=df.index)),
                            errors='coerce').fillna(1.0)
        tp = (df['High'] + df['Low'] + df['Close']) / 3.0
        pv = tp * vol
        vol_sum = vol.rolling(window=window, min_periods=window).sum()
        pv_sum = pv.rolling(window=window, min_periods=window).sum()
        return pv_sum / (vol_sum + 1e-10)

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        vwap_w = int(self.params.get('vwap_window', 50))
        z_w = int(self.params.get('z_window', 50))
        entry_z = float(self.params.get('entry_z', 2.0))

        vwap = self._rolling_vwap(features, vwap_w)
        dev = features['Close'] - vwap

        mu = dev.rolling(window=z_w, min_periods=z_w).mean()
        sd = dev.rolling(window=z_w, min_periods=z_w).std()
        z = (dev - mu) / (sd + 1e-10)

        # Optional ADX range filter
        if bool(self.params.get('use_adx_filter', True)) and 'ADX' in features.columns:
            allow = features['ADX'] < float(self.params.get('adx_threshold', 25))
        else:
            allow = pd.Series(True, index=features.index)

        long_entry = allow & (z > -entry_z) & (z.shift(1) <= -entry_z)
        short_entry = allow & (z < entry_z) & (z.shift(1) >= entry_z)

        signals = pd.Series(0, index=features.index, dtype=int)
        signals[long_entry] = 1
        signals[short_entry] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'vwap_window': [20, 50, 100, 150, 200],
            'z_window': [20, 30, 50, 100],
            'entry_z': [1.5, 1.75, 2.0, 2.25, 2.5],
            'use_adx_filter': [True, False],
            'adx_threshold': [15, 20, 25, 30],
            **self._base_sl_tp_grid(),
        }


# =============================================================================
# BREAKOUT / MOMENTUM STRATEGIES
# =============================================================================

class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian Breakout Strategy - Pure structure breakout.
    
    Clean breakout above prior rolling high / below prior rolling low.
    Uses shift(1) to avoid lookahead.  Exits handled entirely by the
    shared ATR SL/TP via calculate_stops.
    """
    
    @property
    def name(self) -> str:
        return "DonchianBreakoutStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 20,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('period', 20)
        
        high_max = features['High'].rolling(period).max()
        low_min = features['Low'].rolling(period).min()
        
        signals = pd.Series(0, index=features.index)
        # Breakout above channel (shift(1) avoids lookahead)
        breakout_up = features['Close'] > high_max.shift(1)
        # Breakout below channel
        breakout_down = features['Close'] < low_min.shift(1)
        
        signals[breakout_up] = 1
        signals[breakout_down] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'period': [10, 15, 20, 30, 50],
            **self._base_sl_tp_grid(),
        }


class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility Breakout Strategy - ATR-based breakouts."""
    
    @property
    def name(self) -> str:
        return "VolatilityBreakoutStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'atr_period': 20,
            'breakout_mult': 1.0,
            'sl_atr_mult': 1.5,
            'tp_atr_mult': 3.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('atr_period', 20)
        mult = self.params.get('breakout_mult', 1.0)
        
        # Calculate ATR
        high_low = features['High'] - features['Low']
        high_close = abs(features['High'] - features['Close'].shift(1))
        low_close = abs(features['Low'] - features['Close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Price change vs ATR
        price_change = features['Close'].diff()
        threshold = atr * mult
        
        signals = pd.Series(0, index=features.index)
        signals[price_change > threshold] = 1
        signals[price_change < -threshold] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'atr_period': [10, 14, 20, 30],
            'breakout_mult': [0.5, 0.75, 1.0, 1.25, 1.5],
            **self._base_sl_tp_grid(),
        }


class MomentumBurstStrategy(BaseStrategy):
    """Momentum Burst Strategy - Rapid momentum changes."""
    
    @property
    def name(self) -> str:
        return "MomentumBurstStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'momentum_period': 10,
            'threshold_pct': 1.0,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('momentum_period', 10)
        threshold = self.params.get('threshold_pct', 1.0)
        
        momentum = (features['Close'] / features['Close'].shift(period) - 1) * 100
        
        signals = pd.Series(0, index=features.index)
        signals[momentum > threshold] = 1
        signals[momentum < -threshold] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'momentum_period': [3, 5, 10, 15, 20],
            'threshold_pct': [0.3, 0.5, 1.0, 1.5, 2.0],
            **self._base_sl_tp_grid(),
        }


class SqueezeBreakoutStrategy(BaseStrategy):
    """Bollinger Squeeze -> Expansion Breakout Strategy.
    
    Detects squeeze (BB inside KC or low BB bandwidth), then takes the
    first expansion move.  "Squeeze released" is based on prior-bar
    squeeze state to avoid repainting.
    """
    
    @property
    def name(self) -> str:
        return "SqueezeBreakoutStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'bb_period': 20,
            'bb_std': 2.0,
            'kc_period': 20,
            'kc_mult': 1.5,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        bb_period = self.params.get('bb_period', 20)
        bb_std = self.params.get('bb_std', 2.0)
        kc_period = self.params.get('kc_period', 20)
        kc_mult = self.params.get('kc_mult', 1.5)
        
        # Bollinger Bands (shared helper)
        bb_mid, bb_upper, bb_lower = _get_bb(features, bb_period, bb_std)

        # Keltner Channels (shared helper)
        kc_mid, kc_upper, kc_lower = _get_keltner(features, kc_period, kc_period, kc_mult)
        
        # Squeeze detection: BB inside KC
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        squeeze_off = ~squeeze_on
        
        # Momentum
        momentum = features['Close'] - features['Close'].rolling(bb_period).mean()
        
        signals = pd.Series(0, index=features.index)
        # Squeeze releases with positive momentum (prior-bar squeeze state)
        buy_signal = squeeze_off & squeeze_on.shift(1) & (momentum > 0)
        sell_signal = squeeze_off & squeeze_on.shift(1) & (momentum < 0)
        
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        
        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'bb_period': [10, 15, 20, 25],
            'bb_std': [1.5, 2.0, 2.5, 3.0],
            'kc_mult': [1.0, 1.25, 1.5, 2.0],
            **self._base_sl_tp_grid(),
        }


class KeltnerBreakoutStrategy(BaseStrategy):
    """Keltner Channel Breakout Strategy."""
    
    @property
    def name(self) -> str:
        return "KeltnerBreakoutStrategy"
    
    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'period': 20,
            'atr_mult': 2.0,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0
        }
    
    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        period = self.params.get('period', 20)
        mult = self.params.get('atr_mult', 2.0)

        mid, upper, lower = _get_keltner(features, period, period, mult)

        signals = pd.Series(0, index=features.index)
        signals[features['Close'] > upper] = 1
        signals[features['Close'] < lower] = -1

        return signals
    
    def get_param_grid(self) -> Dict[str, List]:
        return {
            'period': [10, 15, 20, 25, 30],
            'atr_mult': [1.25, 1.5, 1.75, 2.0, 2.5],
            **self._base_sl_tp_grid(),
        }


class PivotBreakoutStrategy(BaseStrategy):
    """Pivot Break + Retest Confirmation Strategy.

    Two-stage logic to reduce false breakouts:
      Stage 1 — Breakout: Close breaks through a prior swing high/low
                (rolling pivot level, shifted to avoid lookahead).
      Stage 2 — Retest:   Within a configurable confirmation window after
                the breakout, price must pull back near the broken level
                (within retest_tolerance * ATR) and then close back in
                the trend direction.

    The strategy emits a single entry signal at confirmation time only.
    SL/TP is handled centrally by calculate_stops.
    """

    @property
    def name(self) -> str:
        return "PivotBreakoutStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'lookback': 10,
            'confirm_window': 5,
            'retest_tolerance': 0.5,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
        }

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        lookback = int(self.params.get('lookback', 10))
        confirm_window = int(self.params.get('confirm_window', 5))
        retest_tol = float(self.params.get('retest_tolerance', 0.5))

        high = features['High'].to_numpy(dtype=float)
        low = features['Low'].to_numpy(dtype=float)
        close = features['Close'].to_numpy(dtype=float)
        n = len(close)

        # ATR for tolerance scaling
        atr_col = 'ATR_14'
        if atr_col in features.columns:
            atr = features[atr_col].to_numpy(dtype=float)
        else:
            hl = features['High'] - features['Low']
            hc = (features['High'] - features['Close'].shift(1)).abs()
            lc = (features['Low'] - features['Close'].shift(1)).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().to_numpy(dtype=float)

        # Compute rolling pivot levels (shifted by 1 to avoid lookahead)
        pivot_high = features['High'].rolling(lookback).max().shift(1).to_numpy(dtype=float)
        pivot_low = features['Low'].rolling(lookback).min().shift(1).to_numpy(dtype=float)

        signals = np.zeros(n, dtype=int)

        # State tracking for pending breakouts
        pending_long_bar = -1
        pending_long_level = np.nan
        pending_short_bar = -1
        pending_short_level = np.nan

        for i in range(lookback + 1, n):
            cur_atr = atr[i] if not np.isnan(atr[i]) else 0.0

            # ----------------------------------------------------------
            # Stage 1: Detect new breakouts
            # ----------------------------------------------------------
            if not np.isnan(pivot_high[i]) and close[i] > pivot_high[i]:
                # New long breakout — start or reset pending
                pending_long_bar = i
                pending_long_level = pivot_high[i]

            if not np.isnan(pivot_low[i]) and close[i] < pivot_low[i]:
                # New short breakout
                pending_short_bar = i
                pending_short_level = pivot_low[i]

            # ----------------------------------------------------------
            # Stage 2: Check for retest confirmation (long)
            # ----------------------------------------------------------
            if pending_long_bar > 0 and i > pending_long_bar:
                bars_since = i - pending_long_bar
                if bars_since <= confirm_window:
                    tolerance = retest_tol * cur_atr
                    # Pullback near the broken level
                    retested = low[i] <= pending_long_level + tolerance
                    # Confirmation: close back above the level
                    confirmed = close[i] > pending_long_level
                    if retested and confirmed:
                        signals[i] = 1
                        pending_long_bar = -1
                        pending_long_level = np.nan
                else:
                    # Window expired — cancel pending
                    pending_long_bar = -1
                    pending_long_level = np.nan

            # ----------------------------------------------------------
            # Stage 2: Check for retest confirmation (short)
            # ----------------------------------------------------------
            if pending_short_bar > 0 and i > pending_short_bar:
                bars_since = i - pending_short_bar
                if bars_since <= confirm_window:
                    tolerance = retest_tol * cur_atr
                    retested = high[i] >= pending_short_level - tolerance
                    confirmed = close[i] < pending_short_level
                    if retested and confirmed:
                        signals[i] = -1
                        pending_short_bar = -1
                        pending_short_level = np.nan
                else:
                    pending_short_bar = -1
                    pending_short_level = np.nan

        return pd.Series(signals, index=features.index)

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'lookback': [5, 7, 10, 15, 20],
            'confirm_window': [3, 5, 7, 10],
            'retest_tolerance': [0.25, 0.5, 0.75, 1.0],
            **self._base_sl_tp_grid(),
        }


# =============================================================================
# ADDITIONAL ADVANCED STRATEGIES (Indicator-Based, PM-Compatible)
# =============================================================================

class EMARibbonADXStrategy(BaseStrategy):
    """EMA Ribbon + ADX Strategy - Trend following with regime confirmation.

    Long condition:
      - EMA_fast > EMA_mid > EMA_slow
      - ADX > adx_threshold
      - Optional: +DI > -DI

    Short condition:
      - EMA_fast < EMA_mid < EMA_slow
      - ADX > adx_threshold
      - Optional: -DI > +DI

    Signals fire only on transitions into a valid state (entry-style signals).
    """

    @property
    def name(self) -> str:
        return "EMARibbonADXStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "ema_fast": 8,
            "ema_mid": 21,
            "ema_slow": 50,
            "adx_period": 14,
            "adx_threshold": 20,
            "use_di_confirmation": True,
            "sl_atr_mult": 2.0,
            "tp_atr_mult": 3.0,
        }

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        ema_fast_p = int(self.params.get("ema_fast", 8))
        ema_mid_p = int(self.params.get("ema_mid", 21))
        ema_slow_p = int(self.params.get("ema_slow", 50))
        adx_period = int(self.params.get("adx_period", 14))
        adx_th = float(self.params.get("adx_threshold", 20))
        use_di = bool(self.params.get("use_di_confirmation", True))

        ema_fast = _get_ema(features, ema_fast_p)
        ema_mid = _get_ema(features, ema_mid_p)
        ema_slow = _get_ema(features, ema_slow_p)
        adx, plus_di, minus_di = _get_adx_di(features, adx_period)

        long_state = (ema_fast > ema_mid) & (ema_mid > ema_slow) & (adx > adx_th)
        short_state = (ema_fast < ema_mid) & (ema_mid < ema_slow) & (adx > adx_th)

        if use_di:
            long_state = long_state & (plus_di > minus_di)
            short_state = short_state & (minus_di > plus_di)

        # Convert to boolean numpy arrays (avoids pandas object downcasting warnings and is faster)
        long_np = long_state.to_numpy(dtype=bool)
        short_np = short_state.to_numpy(dtype=bool)
        prev_long = np.empty_like(long_np)
        prev_short = np.empty_like(short_np)
        if len(long_np) > 0:
            prev_long[0] = False
            prev_short[0] = False
            if len(long_np) > 1:
                prev_long[1:] = long_np[:-1]
                prev_short[1:] = short_np[:-1]
        enter_long = long_np & (~prev_long)
        enter_short = short_np & (~prev_short)
        sig = np.zeros(len(long_np), dtype=np.int8)
        sig[enter_long] = 1
        sig[enter_short] = -1
        return pd.Series(sig.astype(int), index=features.index, dtype=int)

    def get_param_grid(self) -> Dict[str, List]:
        return {
            "ema_fast": [5, 8, 10, 13, 15],
            "ema_mid": [20, 25, 30, 35],
            "ema_slow": [50, 100, 150, 200],
            "adx_threshold": [15, 20, 25, 30],
            "adx_period": [10, 14, 20],
            "use_di_confirmation": [True, False],
            **self._base_sl_tp_grid(),
        }


class RSITrendFilteredMRStrategy(BaseStrategy):
    """RSI mean reversion with a trend/regime filter.

    Long:
      - RSI crosses up through oversold
      - Price above EMA(trend)

    Short:
      - RSI crosses down through overbought
      - Price below EMA(trend)
    """

    @property
    def name(self) -> str:
        return "RSITrendFilteredMRStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "ema_trend_period": 200,
            "sl_atr_mult": 2.0,
            "tp_atr_mult": 2.0,
        }

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        rsi_p = int(self.params.get("rsi_period", 14))
        oversold = float(self.params.get("oversold", 30))
        overbought = float(self.params.get("overbought", 70))
        ema_p = int(self.params.get("ema_trend_period", 200))

        close = features["Close"]
        rsi = _get_rsi(features, rsi_p)
        ema = _get_ema(features, ema_p)

        trend_up = close > ema
        trend_down = close < ema

        signals = pd.Series(0, index=features.index, dtype=int)
        signals[trend_up & (rsi > oversold) & (rsi.shift(1) <= oversold)] = 1
        signals[trend_down & (rsi < overbought) & (rsi.shift(1) >= overbought)] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {
            "rsi_period": [5, 7, 10, 14, 21],
            "oversold": [20, 25, 30, 35],
            "overbought": [65, 70, 75, 80],
            "ema_trend_period": [50, 100, 150, 200],
            **self._base_sl_tp_grid(),
        }


class MACDHistogramMomentumStrategy(BaseStrategy):
    """MACD Histogram Momentum Strategy - Histogram zero-line crosses.

    Long when MACD histogram crosses above 0.
    Short when MACD histogram crosses below 0.

    Optional filters:
      - EMA filter: only long above EMA, short below EMA
      - ADX filter: only trade when ADX > threshold
    """

    @property
    def name(self) -> str:
        return "MACDHistogramMomentumStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "use_ema_filter": True,
            "ema_filter_period": 50,
            "use_adx_filter": False,
            "adx_threshold": 20,
            "sl_atr_mult": 2.0,
            "tp_atr_mult": 3.0,
        }

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        fast = int(self.params.get("macd_fast", 12))
        slow = int(self.params.get("macd_slow", 26))
        sig = int(self.params.get("macd_signal", 9))

        close = features["Close"]

        _, _, hist = _get_macd(features, fast, slow, sig)

        cross_up = (hist > 0) & (hist.shift(1) <= 0)
        cross_down = (hist < 0) & (hist.shift(1) >= 0)

        # Filters
        if bool(self.params.get("use_ema_filter", True)):
            ema_p = int(self.params.get("ema_filter_period", 50))
            ema = _get_ema(features, ema_p)
            cross_up = cross_up & (close > ema)
            cross_down = cross_down & (close < ema)

        if bool(self.params.get("use_adx_filter", False)) and "ADX" in features.columns:
            adx_th = float(self.params.get("adx_threshold", 20))
            cross_up = cross_up & (features["ADX"] > adx_th)
            cross_down = cross_down & (features["ADX"] > adx_th)

        signals = pd.Series(0, index=features.index, dtype=int)
        signals[cross_up] = 1
        signals[cross_down] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {
            "macd_fast": [8, 10, 12, 16],
            "macd_slow": [21, 26, 30, 34],
            "macd_signal": [7, 9, 12],
            "use_ema_filter": [True, False],
            "ema_filter_period": [20, 50, 100],
            "use_adx_filter": [False, True],
            "adx_threshold": [15, 20, 25, 30],
            **self._base_sl_tp_grid(),
        }


class StochRSITrendGateStrategy(BaseStrategy):
    """StochRSI reversal entries gated by a slow EMA trend filter.

    Long:
      - %K crosses above %D AND both are below lower_band
      - Close above EMA(trend)

    Short:
      - %K crosses below %D AND both are above upper_band
      - Close below EMA(trend)
    """

    @property
    def name(self) -> str:
        return "StochRSITrendGateStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "rsi_period": 14,
            "stoch_period": 14,
            "smooth_k": 3,
            "smooth_d": 3,
            "lower_band": 20,
            "upper_band": 80,
            "ema_trend_period": 200,
            "sl_atr_mult": 2.0,
            "tp_atr_mult": 2.0,
        }

    @staticmethod
    def _stoch_rsi_from_rsi(rsi: pd.Series, stoch_period: int,
                            smooth_k: int, smooth_d: int) -> Tuple[pd.Series, pd.Series]:
        rsi_min = rsi.rolling(stoch_period).min()
        rsi_max = rsi.rolling(stoch_period).max()
        rsi_denom = rsi_max - rsi_min
        stoch = pd.Series(50.0, index=rsi.index)
        rsi_valid = rsi_denom > 1e-8
        stoch[rsi_valid] = 100 * (rsi[rsi_valid] - rsi_min[rsi_valid]) / rsi_denom[rsi_valid]

        k = stoch.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()
        return k, d

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        rsi_p = int(self.params.get("rsi_period", 14))
        stoch_p = int(self.params.get("stoch_period", 14))
        smooth_k = int(self.params.get("smooth_k", 3))
        smooth_d = int(self.params.get("smooth_d", 3))
        lower = float(self.params.get("lower_band", 20))
        upper = float(self.params.get("upper_band", 80))
        ema_p = int(self.params.get("ema_trend_period", 200))

        close = features["Close"]
        ema = _get_ema(features, ema_p)

        # Prefer precomputed if available
        if "STOCH_RSI_K" in features.columns and "STOCH_RSI_D" in features.columns:
            k = features["STOCH_RSI_K"]
            d = features["STOCH_RSI_D"]
        else:
            rsi = _get_rsi(features, rsi_p)
            k, d = self._stoch_rsi_from_rsi(rsi, stoch_p, smooth_k, smooth_d)

        cross_up = (k > d) & (k.shift(1) <= d.shift(1)) & (k < lower) & (d < lower)
        cross_down = (k < d) & (k.shift(1) >= d.shift(1)) & (k > upper) & (d > upper)

        signals = pd.Series(0, index=features.index, dtype=int)
        signals[(close > ema) & cross_up] = 1
        signals[(close < ema) & cross_down] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {
            "rsi_period": [5, 7, 10, 14, 21],
            "stoch_period": [10, 14, 20],
            "smooth_k": [3, 5],
            "smooth_d": [3, 5],
            "lower_band": [10, 15, 20, 30],
            "upper_band": [70, 80, 85, 90],
            "ema_trend_period": [50, 100, 150, 200],
            **self._base_sl_tp_grid(),
        }


# VWAPDeviationReversionStrategy retired — near-duplicate of ZScoreVWAPReversionStrategy.
# Existing configs are migrated to ZScoreVWAPReversionStrategy on load.


# =============================================================================
# NEW STRATEGIES (D1-D15)
# =============================================================================


class InsideBarBreakoutStrategy(BaseStrategy):
    """Inside bar breakout: mother bar engulfs child bar(s), breakout of mother range."""

    @property
    def name(self) -> str:
        return "InsideBarBreakoutStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM

    def get_default_params(self) -> Dict[str, Any]:
        return {'min_inside_bars': 1, 'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        min_ib = int(self.params.get('min_inside_bars', 1))
        h, l, c = features['High'], features['Low'], features['Close']
        # Inside bar: current H < prev H AND current L > prev L
        inside = (h < h.shift(1)) & (l > l.shift(1))
        # Count consecutive inside bars
        consec = inside.astype(int)
        for i in range(1, min_ib):
            consec = consec + inside.shift(i, fill_value=False).astype(int)
        qualified = consec >= min_ib
        # Mother bar is the bar before the inside bar sequence
        mother_h = h.shift(min_ib)
        mother_l = l.shift(min_ib)
        signals = pd.Series(0, index=features.index, dtype=int)
        signals[qualified & (c > mother_h)] = 1
        signals[qualified & (c < mother_l)] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'min_inside_bars': [1, 2, 3], **self._base_sl_tp_grid()}


class NarrowRangeBreakoutStrategy(BaseStrategy):
    """Narrow range breakout: bar range is narrowest in N bars, trade breakout."""

    @property
    def name(self) -> str:
        return "NarrowRangeBreakoutStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM

    def get_default_params(self) -> Dict[str, Any]:
        return {'nr_lookback': 7, 'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        lookback = int(self.params.get('nr_lookback', 7))
        bar_range = features['High'] - features['Low']
        min_range = bar_range.rolling(lookback).min()
        is_nr = (bar_range == min_range) & (bar_range > 0)
        # Breakout on next bar
        nr_h = features['High'].shift(1)
        nr_l = features['Low'].shift(1)
        nr_flag = is_nr.shift(1, fill_value=False)
        signals = pd.Series(0, index=features.index, dtype=int)
        signals[nr_flag & (features['Close'] > nr_h)] = 1
        signals[nr_flag & (features['Close'] < nr_l)] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'nr_lookback': [4, 7, 10], **self._base_sl_tp_grid()}


class TurtleSoupReversalStrategy(BaseStrategy):
    """Turtle soup: fade failed Donchian breakouts that close back inside the channel."""

    @property
    def name(self) -> str:
        return "TurtleSoupReversalStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {'channel_period': 20, 'reclaim_window': 2,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 2.0}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        ch = int(self.params.get('channel_period', 20))
        rw = int(self.params.get('reclaim_window', 2))
        h, l, c = features['High'], features['Low'], features['Close']
        don_h = h.rolling(ch).max().shift(1)
        don_l = l.rolling(ch).min().shift(1)
        # Broke above channel high within last rw bars
        broke_high = pd.Series(False, index=features.index)
        broke_low = pd.Series(False, index=features.index)
        for lag in range(1, rw + 1):
            broke_high = broke_high | (h.shift(lag) > don_h.shift(lag))
            broke_low = broke_low | (l.shift(lag) < don_l.shift(lag))
        # Reclaimed (closed back inside)
        signals = pd.Series(0, index=features.index, dtype=int)
        signals[broke_high & (c < don_h)] = -1  # Failed upside breakout → short
        signals[broke_low & (c > don_l)] = 1    # Failed downside breakout → long
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'channel_period': [10, 15, 20, 25],
                'reclaim_window': [1, 2, 3], **self._base_sl_tp_grid()}


class PinBarReversalStrategy(BaseStrategy):
    """Pin bar reversal: wick/body ratio detection with location filter."""

    @property
    def name(self) -> str:
        return "PinBarReversalStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {'wick_ratio': 2.5, 'proximity_atr': 1.0,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 2.0}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14', 'BB_LOWER_20', 'BB_UPPER_20'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        wick_r = float(self.params.get('wick_ratio', 2.5))
        prox = float(self.params.get('proximity_atr', 1.0))
        o, h, l, c = features['Open'], features['High'], features['Low'], features['Close']
        body = (c - o).abs()
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
        body_safe = body.clip(lower=1e-10)
        atr = _get_atr(features, 14)
        _, bb_upper, bb_lower = _get_bb(features, 20)
        # Bullish pin: long lower wick near BB lower band
        bull_pin = (lower_wick / body_safe > wick_r) & (l <= bb_lower + prox * atr)
        # Bearish pin: long upper wick near BB upper band
        bear_pin = (upper_wick / body_safe > wick_r) & (h >= bb_upper - prox * atr)
        signals = pd.Series(0, index=features.index, dtype=int)
        signals[bull_pin.shift(1, fill_value=False)] = 1
        signals[bear_pin.shift(1, fill_value=False)] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'wick_ratio': [2.0, 2.5, 3.0],
                'proximity_atr': [0.5, 1.0, 1.5], **self._base_sl_tp_grid()}


class EngulfingPatternStrategy(BaseStrategy):
    """Engulfing pattern with location filter (BB/Donchian proximity)."""

    @property
    def name(self) -> str:
        return "EngulfingPatternStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {'lookback_level': 20, 'use_adx_filter': True,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 2.5}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14', 'BB_LOWER_20', 'BB_UPPER_20', 'ADX'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        o, c = features['Open'], features['Close']
        o1, c1 = o.shift(1), c.shift(1)
        body = (c - o)
        body1 = (c1 - o1)
        # Bullish engulfing: prior bar bearish, current bar bullish, body engulfs
        bull_eng = (body1 < 0) & (body > 0) & (o <= c1) & (c >= o1)
        # Bearish engulfing
        bear_eng = (body1 > 0) & (body < 0) & (o >= c1) & (c <= o1)
        _, bb_upper, bb_lower = _get_bb(features, 20)
        atr = _get_atr(features, 14)
        near_lower = features['Low'] <= bb_lower + atr
        near_upper = features['High'] >= bb_upper - atr
        # ADX filter (prefer range)
        if bool(self.params.get('use_adx_filter', True)) and 'ADX' in features.columns:
            allow = features['ADX'] < 30
        else:
            allow = pd.Series(True, index=features.index)
        signals = pd.Series(0, index=features.index, dtype=int)
        signals[bull_eng & near_lower & allow] = 1
        signals[bear_eng & near_upper & allow] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'lookback_level': [10, 20, 30],
                'use_adx_filter': [True, False], **self._base_sl_tp_grid()}


class VolumeSpikeMomentumStrategy(BaseStrategy):
    """Volume spike + directional close: breakout on unusual volume."""

    @property
    def name(self) -> str:
        return "VolumeSpikeMomentumStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM

    def get_default_params(self) -> Dict[str, Any]:
        return {'vol_mult': 2.0, 'vol_lookback': 20,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        vm = float(self.params.get('vol_mult', 2.0))
        vl = int(self.params.get('vol_lookback', 20))
        vol = pd.to_numeric(features.get('Volume', pd.Series(0, index=features.index)),
                            errors='coerce').fillna(0)
        avg_vol = vol.rolling(vl, min_periods=vl).mean()
        spike = vol > vm * avg_vol
        bar_range = features['High'] - features['Low']
        close_pos = (features['Close'] - features['Low']) / (bar_range + 1e-10)
        signals = pd.Series(0, index=features.index, dtype=int)
        signals[spike & (close_pos > 0.7)] = 1   # Close in upper 30%
        signals[spike & (close_pos < 0.3)] = -1  # Close in lower 30%
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'vol_mult': [1.5, 2.0, 2.5, 3.0],
                'vol_lookback': [10, 20, 30], **self._base_sl_tp_grid()}


class RSIDivergenceStrategy(BaseStrategy):
    """RSI divergence: swing point divergence between price and RSI."""

    @property
    def name(self) -> str:
        return "RSIDivergenceStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {'rsi_period': 14, 'swing_order': 5,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 2.5}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        rsi_p = int(self.params.get('rsi_period', 14))
        order = int(self.params.get('swing_order', 5))
        rsi = _get_rsi(features, rsi_p)
        close = features['Close']
        _, price_lows = _detect_swing_points(close, order)
        price_highs, _ = _detect_swing_points(close, order)
        _, rsi_lows = _detect_swing_points(rsi, order)
        rsi_highs, _ = _detect_swing_points(rsi, order)
        signals = pd.Series(0, index=features.index, dtype=int)
        # Bullish divergence: price lower low, RSI higher low
        for i in range(2 * order + 1, len(features)):
            if price_lows.iloc[i]:
                # Find previous swing low
                prev_lows = price_lows.iloc[max(0, i - 5 * order):i]
                prev_idx = prev_lows[prev_lows].index
                if len(prev_idx) > 0:
                    j = features.index.get_loc(prev_idx[-1])
                    if close.iloc[i] < close.iloc[j] and rsi.iloc[i] > rsi.iloc[j]:
                        signals.iloc[i] = 1
            if price_highs.iloc[i]:
                prev_highs = price_highs.iloc[max(0, i - 5 * order):i]
                prev_idx = prev_highs[prev_highs].index
                if len(prev_idx) > 0:
                    j = features.index.get_loc(prev_idx[-1])
                    if close.iloc[i] > close.iloc[j] and rsi.iloc[i] < rsi.iloc[j]:
                        signals.iloc[i] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'rsi_period': [7, 14, 21], 'swing_order': [3, 5, 7],
                **self._base_sl_tp_grid()}


class MACDDivergenceStrategy(BaseStrategy):
    """MACD histogram divergence with price."""

    @property
    def name(self) -> str:
        return "MACDDivergenceStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {'swing_order': 5, 'macd_fast': 12, 'macd_slow': 26,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 2.5}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14', 'MACD_HIST'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        order = int(self.params.get('swing_order', 5))
        fast = int(self.params.get('macd_fast', 12))
        slow = int(self.params.get('macd_slow', 26))
        _, _, hist = _get_macd(features, fast, slow)
        close = features['Close']
        _, price_lows = _detect_swing_points(close, order)
        price_highs, _ = _detect_swing_points(close, order)
        _, hist_lows = _detect_swing_points(hist, order)
        hist_highs, _ = _detect_swing_points(hist, order)
        signals = pd.Series(0, index=features.index, dtype=int)
        for i in range(2 * order + 1, len(features)):
            if price_lows.iloc[i]:
                prev_lows = price_lows.iloc[max(0, i - 5 * order):i]
                prev_idx = prev_lows[prev_lows].index
                if len(prev_idx) > 0:
                    j = features.index.get_loc(prev_idx[-1])
                    if close.iloc[i] < close.iloc[j] and hist.iloc[i] > hist.iloc[j]:
                        signals.iloc[i] = 1
            if price_highs.iloc[i]:
                prev_highs = price_highs.iloc[max(0, i - 5 * order):i]
                prev_idx = prev_highs[prev_highs].index
                if len(prev_idx) > 0:
                    j = features.index.get_loc(prev_idx[-1])
                    if close.iloc[i] > close.iloc[j] and hist.iloc[i] < hist.iloc[j]:
                        signals.iloc[i] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'swing_order': [3, 5, 7], 'macd_fast': [8, 12],
                'macd_slow': [21, 26], **self._base_sl_tp_grid()}


class OBVDivergenceStrategy(BaseStrategy):
    """On-Balance Volume divergence with price."""

    @property
    def name(self) -> str:
        return "OBVDivergenceStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {'swing_order': 5, 'obv_smooth': 5,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 2.5}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        order = int(self.params.get('swing_order', 5))
        smooth = int(self.params.get('obv_smooth', 5))
        vol = pd.to_numeric(features.get('Volume', pd.Series(0, index=features.index)),
                            errors='coerce').fillna(0)
        direction = features['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (vol * direction).cumsum()
        if smooth > 0:
            obv = obv.rolling(smooth, min_periods=1).mean()
        close = features['Close']
        _, price_lows = _detect_swing_points(close, order)
        price_highs, _ = _detect_swing_points(close, order)
        signals = pd.Series(0, index=features.index, dtype=int)
        for i in range(2 * order + 1, len(features)):
            if price_lows.iloc[i]:
                prev_lows = price_lows.iloc[max(0, i - 5 * order):i]
                prev_idx = prev_lows[prev_lows].index
                if len(prev_idx) > 0:
                    j = features.index.get_loc(prev_idx[-1])
                    if close.iloc[i] < close.iloc[j] and obv.iloc[i] > obv.iloc[j]:
                        signals.iloc[i] = 1
            if price_highs.iloc[i]:
                prev_highs = price_highs.iloc[max(0, i - 5 * order):i]
                prev_idx = prev_highs[prev_highs].index
                if len(prev_idx) > 0:
                    j = features.index.get_loc(prev_idx[-1])
                    if close.iloc[i] > close.iloc[j] and obv.iloc[i] < obv.iloc[j]:
                        signals.iloc[i] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'swing_order': [3, 5, 7], 'obv_smooth': [0, 5, 10],
                **self._base_sl_tp_grid()}


class KeltnerFadeStrategy(BaseStrategy):
    """Fade outer Keltner Channel band back to midline."""

    @property
    def name(self) -> str:
        return "KeltnerFadeStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {'kc_ema': 20, 'kc_mult': 2.0, 'adx_threshold': 25,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 2.0}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14', 'EMA_20', 'ADX'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        ema_p = int(self.params.get('kc_ema', 20))
        mult = float(self.params.get('kc_mult', 2.0))
        adx_t = float(self.params.get('adx_threshold', 25))
        mid, upper, lower = _get_keltner(features, ema_p, 14, mult)
        c = features['Close']
        # Touch upper KC then next bar closes inside → short
        touched_upper = (features['High'].shift(1) >= upper.shift(1)) & (c < upper)
        touched_lower = (features['Low'].shift(1) <= lower.shift(1)) & (c > lower)
        allow = pd.Series(True, index=features.index)
        if 'ADX' in features.columns:
            allow = features['ADX'] < adx_t
        signals = pd.Series(0, index=features.index, dtype=int)
        signals[touched_lower & allow] = 1
        signals[touched_upper & allow] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'kc_ema': [15, 20, 25], 'kc_mult': [1.5, 2.0, 2.5],
                'adx_threshold': [20, 25, 30], **self._base_sl_tp_grid()}


class ROCExhaustionReversalStrategy(BaseStrategy):
    """Rate of Change exhaustion: ROC reaches extreme percentile then reverses."""

    @property
    def name(self) -> str:
        return "ROCExhaustionReversalStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {'roc_period': 10, 'pct_lookback': 100, 'extreme_pct': 10,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 2.0}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        roc_p = int(self.params.get('roc_period', 10))
        pct_lb = int(self.params.get('pct_lookback', 100))
        ext_pct = float(self.params.get('extreme_pct', 10))
        roc = features['Close'].pct_change(roc_p) * 100
        pct_rank = _rolling_percentile_rank(roc, pct_lb)
        # Extreme high ROC crossing back below threshold
        was_extreme_high = (pct_rank.shift(1) >= (100 - ext_pct))
        now_normal_high = (pct_rank < (100 - ext_pct))
        was_extreme_low = (pct_rank.shift(1) <= ext_pct)
        now_normal_low = (pct_rank > ext_pct)
        signals = pd.Series(0, index=features.index, dtype=int)
        signals[was_extreme_low & now_normal_low] = 1    # Oversold bounce
        signals[was_extreme_high & now_normal_high] = -1  # Overbought fade
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'roc_period': [5, 10, 14, 20], 'pct_lookback': [50, 100, 200],
                'extreme_pct': [5, 10, 15], **self._base_sl_tp_grid()}


class EMAPullbackContinuationStrategy(BaseStrategy):
    """EMA crossover confirms trend, enter on pullback to fast EMA."""

    @property
    def name(self) -> str:
        return "EMAPullbackContinuationStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {'fast_period': 10, 'slow_period': 30, 'touch_atr': 0.5,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        fp = int(self.params.get('fast_period', 10))
        sp = int(self.params.get('slow_period', 30))
        touch = float(self.params.get('touch_atr', 0.5))
        fast_ema = _get_ema(features, fp)
        slow_ema = _get_ema(features, sp)
        atr = _get_atr(features, 14)
        uptrend = fast_ema > slow_ema
        downtrend = fast_ema < slow_ema
        # Pullback to fast EMA (within touch_atr * ATR)
        near_fast = (features['Low'] <= fast_ema + touch * atr) & (features['Close'] > fast_ema)
        near_fast_short = (features['High'] >= fast_ema - touch * atr) & (features['Close'] < fast_ema)
        signals = pd.Series(0, index=features.index, dtype=int)
        signals[uptrend & near_fast] = 1
        signals[downtrend & near_fast_short] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'fast_period': [8, 10, 13], 'slow_period': [20, 30, 50],
                'touch_atr': [0.3, 0.5, 0.7], **self._base_sl_tp_grid()}


class ParabolicSARTrendStrategy(BaseStrategy):
    """Parabolic SAR trend-following with optional ADX filter."""

    @property
    def name(self) -> str:
        return "ParabolicSARTrendStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {'af_start': 0.02, 'af_max': 0.20, 'adx_threshold': 20,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14', 'ADX'}

    @staticmethod
    def _compute_psar(high: np.ndarray, low: np.ndarray,
                      af_start: float, af_max: float) -> np.ndarray:
        """Compute Parabolic SAR."""
        n = len(high)
        psar = np.zeros(n)
        bull = True
        af = af_start
        ep = low[0]
        psar[0] = high[0]
        for i in range(1, n):
            if bull:
                psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
                psar[i] = min(psar[i], low[i - 1], low[max(0, i - 2)])
                if low[i] < psar[i]:
                    bull = False
                    psar[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_start, af_max)
            else:
                psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
                psar[i] = max(psar[i], high[i - 1], high[max(0, i - 2)])
                if high[i] > psar[i]:
                    bull = True
                    psar[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_start, af_max)
        return psar

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        af_s = float(self.params.get('af_start', 0.02))
        af_m = float(self.params.get('af_max', 0.20))
        adx_t = float(self.params.get('adx_threshold', 20))
        psar = self._compute_psar(features['High'].values, features['Low'].values,
                                  af_s, af_m)
        psar_s = pd.Series(psar, index=features.index)
        c = features['Close']
        # PSAR flip signals
        above = c > psar_s
        flip_long = above & ~above.shift(1, fill_value=False)
        flip_short = ~above & above.shift(1, fill_value=True)
        allow = pd.Series(True, index=features.index)
        if 'ADX' in features.columns:
            allow = features['ADX'] > adx_t
        signals = pd.Series(0, index=features.index, dtype=int)
        signals[flip_long & allow] = 1
        signals[flip_short & allow] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'af_start': [0.01, 0.02, 0.03], 'af_max': [0.15, 0.20, 0.25],
                'adx_threshold': [15, 20, 25], **self._base_sl_tp_grid()}


class ATRPercentileBreakoutStrategy(BaseStrategy):
    """ATR compression→expansion breakout: low ATR percentile followed by rise."""

    @property
    def name(self) -> str:
        return "ATRPercentileBreakoutStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM

    def get_default_params(self) -> Dict[str, Any]:
        return {'pct_lookback': 100, 'compress_pct': 20, 'expand_pct': 70,
                'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        pct_lb = int(self.params.get('pct_lookback', 100))
        comp = float(self.params.get('compress_pct', 20))
        exp = float(self.params.get('expand_pct', 70))
        atr = _get_atr(features, 14)
        atr_pct = _rolling_percentile_rank(atr, pct_lb)
        # Was compressed, now expanding
        was_low = atr_pct.shift(1) <= comp
        now_high = atr_pct >= exp
        trigger = was_low & now_high
        bar_range = features['High'] - features['Low']
        close_pos = (features['Close'] - features['Low']) / (bar_range + 1e-10)
        signals = pd.Series(0, index=features.index, dtype=int)
        signals[trigger & (close_pos > 0.6)] = 1
        signals[trigger & (close_pos < 0.4)] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'pct_lookback': [50, 100, 200], 'compress_pct': [10, 20, 30],
                'expand_pct': [60, 70, 80], **self._base_sl_tp_grid()}


class KaufmanAMATrendStrategy(BaseStrategy):
    """Kaufman Adaptive Moving Average: efficiency ratio adjusts smoothing speed."""

    @property
    def name(self) -> str:
        return "KaufmanAMATrendStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {'er_period': 10, 'fast_period': 2, 'slow_period': 30,
                'signal_mode': 'direction', 'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    @staticmethod
    def _kaufman_ama(close: np.ndarray, er_period: int,
                     fast_period: int, slow_period: int) -> np.ndarray:
        """Compute Kaufman Adaptive Moving Average."""
        n = len(close)
        ama = np.full(n, np.nan)
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        if er_period >= n:
            return ama
        ama[er_period] = close[er_period]
        for i in range(er_period + 1, n):
            direction = abs(close[i] - close[i - er_period])
            volatility = 0.0
            for j in range(i - er_period + 1, i + 1):
                volatility += abs(close[j] - close[j - 1])
            er = direction / volatility if volatility > 0 else 0.0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            ama[i] = ama[i - 1] + sc * (close[i] - ama[i - 1])
        return ama

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        er_p = int(self.params.get('er_period', 10))
        fast_p = int(self.params.get('fast_period', 2))
        slow_p = int(self.params.get('slow_period', 30))
        mode = str(self.params.get('signal_mode', 'direction'))
        close = features['Close'].values
        ama = self._kaufman_ama(close, er_p, fast_p, slow_p)
        ama_s = pd.Series(ama, index=features.index)
        signals = pd.Series(0, index=features.index, dtype=int)
        if mode == 'crossover':
            # Price crosses above/below AMA
            above = features['Close'] > ama_s
            prev_above = above.shift(1, fill_value=False)
            signals[above & ~prev_above] = 1
            signals[~above & prev_above] = -1
        else:
            # AMA direction change
            ama_diff = ama_s.diff()
            signals[(ama_diff > 0) & (ama_diff.shift(1) <= 0)] = 1
            signals[(ama_diff < 0) & (ama_diff.shift(1) >= 0)] = -1
        return signals

    def get_param_grid(self) -> Dict[str, List]:
        return {'er_period': [5, 10, 20], 'fast_period': [2, 3],
                'slow_period': [20, 30, 50],
                'signal_mode': ['direction', 'crossover'],
                **self._base_sl_tp_grid()}


# =============================================================================
# NEW STRATEGIES (42 → 50)
# =============================================================================


class VortexTrendStrategy(BaseStrategy):
    """Vortex Indicator trend rotation/birth capture via VI+/VI- cross with spread quality."""

    @property
    def name(self) -> str:
        return "VortexTrendStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'vortex_period': 14,
            'min_vi_spread': 0.03,
            'use_ema_filter': False,
            'ema_filter_period': 100,
            'use_adx_filter': False,
            'adx_threshold': 22,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
        }

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        vp = int(self.params.get('vortex_period', 14))
        min_spread = float(self.params.get('min_vi_spread', 0.03))
        use_ema = bool(self.params.get('use_ema_filter', False))
        ema_p = int(self.params.get('ema_filter_period', 100))
        use_adx = bool(self.params.get('use_adx_filter', False))
        adx_th = float(self.params.get('adx_threshold', 22))

        vi_plus, vi_minus = _get_vortex(features, vp)
        spread = vi_plus - vi_minus

        cross_bull = (spread > min_spread) & (spread.shift(1) <= min_spread)
        cross_bear = (spread < -min_spread) & (spread.shift(1) >= -min_spread)

        if use_ema:
            ema = _get_ema(features, ema_p)
            cross_bull = cross_bull & (features['Close'] > ema)
            cross_bear = cross_bear & (features['Close'] < ema)

        if use_adx:
            adx, _, _ = _get_adx_di(features, 14)
            cross_bull = cross_bull & (adx > adx_th)
            cross_bear = cross_bear & (adx > adx_th)

        signals = pd.Series(0, index=features.index, dtype=int)
        signals[cross_bull] = 1
        signals[cross_bear] = -1
        return self._zero_warmup(signals, vp + 5)

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'vortex_period': [10, 14, 21, 28, 34],
            'min_vi_spread': [0.0, 0.03, 0.06, 0.09],
            'use_ema_filter': [False, True],
            'ema_filter_period': [50, 100, 200],
            'use_adx_filter': [False, True],
            'adx_threshold': [18, 22, 28],
            **self._base_sl_tp_grid(),
        }


class ElderRayBullBearStrategy(BaseStrategy):
    """Elder Ray Bull/Bear Power trend continuation via power zero/threshold crossings."""

    @property
    def name(self) -> str:
        return "ElderRayBullBearStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'ema_period': 13,
            'entry_mode': 'zero_cross',
            'threshold_atr_norm': 0.2,
            'require_ema_trend': True,
            'trend_ema_period': 100,
            'power_smooth': 1,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
        }

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        ema_p = int(self.params.get('ema_period', 13))
        mode = str(self.params.get('entry_mode', 'zero_cross'))
        th_norm = float(self.params.get('threshold_atr_norm', 0.2))
        req_trend = bool(self.params.get('require_ema_trend', True))
        trend_p = int(self.params.get('trend_ema_period', 100))
        smooth = int(self.params.get('power_smooth', 1))

        bull_power, bear_power = _get_elder_power(features, ema_p, smooth)
        atr = _get_atr(features, 14)
        threshold = th_norm * atr

        if mode == 'threshold_cross':
            long_trigger = (bear_power > -threshold) & (bear_power.shift(1) <= -threshold.shift(1))
            short_trigger = (bull_power < threshold) & (bull_power.shift(1) >= threshold.shift(1))
        else:  # zero_cross
            long_trigger = (bear_power > 0) & (bear_power.shift(1) <= 0)
            short_trigger = (bull_power < 0) & (bull_power.shift(1) >= 0)

        if req_trend:
            trend_ema = _get_ema(features, trend_p)
            ema_rising = trend_ema > trend_ema.shift(1)
            ema_falling = trend_ema < trend_ema.shift(1)
            long_trigger = long_trigger & ema_rising
            short_trigger = short_trigger & ema_falling

        signals = pd.Series(0, index=features.index, dtype=int)
        signals[long_trigger] = 1
        signals[short_trigger] = -1
        warmup = max(ema_p, trend_p if req_trend else 0) + 5
        return self._zero_warmup(signals, warmup)

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'ema_period': [10, 13, 20, 26, 34],
            'entry_mode': ['zero_cross', 'threshold_cross'],
            'threshold_atr_norm': [0.0, 0.2, 0.4, 0.6],
            'require_ema_trend': [False, True],
            'trend_ema_period': [50, 100, 200],
            'power_smooth': [1, 3],
            **self._base_sl_tp_grid(),
        }


class TRIXMomentumStrategy(BaseStrategy):
    """TRIX triple-smoothed momentum oscillator with signal/zero cross entries."""

    @property
    def name(self) -> str:
        return "TRIXMomentumStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'trix_period': 12,
            'signal_period': 9,
            'entry_mode': 'signal_cross',
            'use_trend_filter': False,
            'trend_ema_period': 100,
            'min_abs_trix': 0.0,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
        }

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        tp = int(self.params.get('trix_period', 12))
        sp = int(self.params.get('signal_period', 9))
        mode = str(self.params.get('entry_mode', 'signal_cross'))
        use_tf = bool(self.params.get('use_trend_filter', False))
        trend_p = int(self.params.get('trend_ema_period', 100))
        min_abs = float(self.params.get('min_abs_trix', 0.0))

        trix, signal_line = _get_trix(features, tp, sp)

        if mode == 'zero_cross':
            long_trigger = (trix > 0) & (trix.shift(1) <= 0)
            short_trigger = (trix < 0) & (trix.shift(1) >= 0)
        elif mode == 'both_confirm':
            sig_cross_bull = (trix > signal_line) & (trix.shift(1) <= signal_line.shift(1))
            sig_cross_bear = (trix < signal_line) & (trix.shift(1) >= signal_line.shift(1))
            long_trigger = sig_cross_bull & (trix > 0)
            short_trigger = sig_cross_bear & (trix < 0)
        else:  # signal_cross
            long_trigger = (trix > signal_line) & (trix.shift(1) <= signal_line.shift(1))
            short_trigger = (trix < signal_line) & (trix.shift(1) >= signal_line.shift(1))

        if min_abs > 0:
            long_trigger = long_trigger & (trix.abs() > min_abs)
            short_trigger = short_trigger & (trix.abs() > min_abs)

        if use_tf:
            ema = _get_ema(features, trend_p)
            long_trigger = long_trigger & (features['Close'] > ema)
            short_trigger = short_trigger & (features['Close'] < ema)

        signals = pd.Series(0, index=features.index, dtype=int)
        signals[long_trigger] = 1
        signals[short_trigger] = -1
        return self._zero_warmup(signals, tp * 3 + sp + 5)

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'trix_period': [8, 12, 15, 20, 30],
            'signal_period': [5, 9, 12],
            'entry_mode': ['signal_cross', 'zero_cross', 'both_confirm'],
            'use_trend_filter': [False, True],
            'trend_ema_period': [50, 100],
            'min_abs_trix': [0.0, 0.03],
            **self._base_sl_tp_grid(),
        }


class MarketStructureBOSPullbackStrategy(BaseStrategy):
    """Break of Structure (BOS) trend continuation with pullback entry.

    Tracks swing-sequence structure (HH/HL for bullish, LH/LL for bearish),
    detects BOS events when close breaks prior confirmed swing, then enters
    on pullback to the broken level with a confirmation close.
    """

    @property
    def name(self) -> str:
        return "MarketStructureBOSPullbackStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'swing_order': 5,
            'bos_buffer_atr': 0.1,
            'require_displacement': True,
            'displacement_atr': 1.2,
            'pullback_tolerance_atr': 0.4,
            'pullback_window': 4,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
        }

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        order = int(self.params.get('swing_order', 5))
        bos_buf = float(self.params.get('bos_buffer_atr', 0.1))
        req_disp = bool(self.params.get('require_displacement', True))
        disp_atr = float(self.params.get('displacement_atr', 1.2))
        pb_tol = float(self.params.get('pullback_tolerance_atr', 0.4))
        pb_win = int(self.params.get('pullback_window', 4))

        sh_marks, sl_marks, sh_prices, sl_prices = _get_swing_state(features, order)
        high = features['High'].values
        low = features['Low'].values
        close = features['Close'].values
        opn = features['Open'].values
        atr_col = 'ATR_14'
        atr = features[atr_col].values if atr_col in features.columns else _get_atr(features, 14).values
        sh_marks_v = sh_marks.values
        sl_marks_v = sl_marks.values
        sh_prices_v = sh_prices.values
        sl_prices_v = sl_prices.values
        n = len(close)

        signals = np.zeros(n, dtype=int)

        last_sh = np.nan
        last_sl = np.nan
        pending_dir = 0
        pending_bar = -1
        pending_level = np.nan

        warmup = order * 4 + pb_win + 10
        for i in range(warmup, n):
            cur_atr = atr[i] if not np.isnan(atr[i]) else 0.0

            if sh_marks_v[i] and not np.isnan(sh_prices_v[i]):
                last_sh = sh_prices_v[i]
            if sl_marks_v[i] and not np.isnan(sl_prices_v[i]):
                last_sl = sl_prices_v[i]

            # BOS detection: bullish
            if not np.isnan(last_sh) and close[i] > last_sh + bos_buf * cur_atr:
                disp_ok = True
                if req_disp:
                    disp_ok = (close[i] - opn[i]) > disp_atr * cur_atr
                if disp_ok:
                    pending_dir = 1
                    pending_bar = i
                    pending_level = last_sh

            # BOS detection: bearish
            if not np.isnan(last_sl) and close[i] < last_sl - bos_buf * cur_atr:
                disp_ok = True
                if req_disp:
                    disp_ok = (opn[i] - close[i]) > disp_atr * cur_atr
                if disp_ok:
                    pending_dir = -1
                    pending_bar = i
                    pending_level = last_sl

            # Pullback entry: long
            if pending_dir == 1 and 0 < (i - pending_bar) <= pb_win:
                if low[i] <= pending_level + pb_tol * cur_atr:
                    if close[i] > pending_level:
                        signals[i] = 1
                        pending_dir = 0

            # Pullback entry: short
            if pending_dir == -1 and 0 < (i - pending_bar) <= pb_win:
                if high[i] >= pending_level - pb_tol * cur_atr:
                    if close[i] < pending_level:
                        signals[i] = -1
                        pending_dir = 0

            # Expire stale pending
            if pending_dir != 0 and (i - pending_bar) > pb_win:
                pending_dir = 0

        return pd.Series(signals, index=features.index, dtype=int)

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'swing_order': [3, 5, 7],
            'bos_buffer_atr': [0.05, 0.1, 0.2],
            'require_displacement': [False, True],
            'displacement_atr': [0.8, 1.2],
            'pullback_tolerance_atr': [0.2, 0.4, 0.6],
            'pullback_window': [2, 4, 6],
            **self._base_sl_tp_grid(),
        }


class LiquiditySweepReversalStrategy(BaseStrategy):
    """Anti-breakout reversal on sweep and reclaim of multi-swing liquidity pools.

    Builds clustered swing pools, detects wick sweep through pool boundary,
    then enters reversal when price reclaims back inside.
    """

    @property
    def name(self) -> str:
        return "LiquiditySweepReversalStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.MEAN_REVERSION

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'swing_order': 5,
            'pool_lookback_bars': 40,
            'min_pool_points': 2,
            'sweep_buffer_atr': 0.1,
            'reclaim_window': 2,
            'min_reversal_body_atr': 0.0,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 2.0,
        }

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        order = int(self.params.get('swing_order', 5))
        lb = int(self.params.get('pool_lookback_bars', 40))
        min_pts = int(self.params.get('min_pool_points', 2))
        sweep_buf = float(self.params.get('sweep_buffer_atr', 0.1))
        reclaim_win = int(self.params.get('reclaim_window', 2))
        min_body = float(self.params.get('min_reversal_body_atr', 0.0))

        sh_marks, sl_marks, sh_prices, sl_prices = _get_swing_state(features, order)
        high = features['High'].values
        low = features['Low'].values
        close = features['Close'].values
        opn = features['Open'].values
        atr_col = 'ATR_14'
        atr = features[atr_col].values if atr_col in features.columns else _get_atr(features, 14).values
        sh_v = sh_marks.values
        sl_v = sl_marks.values
        sh_prices_v = sh_prices.values
        sl_prices_v = sl_prices.values
        n = len(close)

        signals = np.zeros(n, dtype=int)

        # Rolling swing pools (bounded lists of (price, bar_index))
        swing_highs_pool: List[Tuple[float, int]] = []
        swing_lows_pool: List[Tuple[float, int]] = []

        pending_low_sweep_bar = -1
        pending_low_sweep_level = np.nan
        pending_high_sweep_bar = -1
        pending_high_sweep_level = np.nan

        warmup = lb + order * 2 + 10
        for i in range(warmup, n):
            cur_atr = atr[i] if not np.isnan(atr[i]) else 0.0
            body = abs(close[i] - opn[i])

            # Update pools
            if sh_v[i] and not np.isnan(sh_prices_v[i]):
                swing_highs_pool.append((float(sh_prices_v[i]), i))
            if sl_v[i] and not np.isnan(sl_prices_v[i]):
                swing_lows_pool.append((float(sl_prices_v[i]), i))

            # Evict old entries
            swing_highs_pool = [(p, b) for p, b in swing_highs_pool if i - b <= lb]
            swing_lows_pool = [(p, b) for p, b in swing_lows_pool if i - b <= lb]

            # Find high pool level (cluster of swing highs)
            high_pool = np.nan
            if len(swing_highs_pool) >= min_pts:
                prices = sorted([p for p, _ in swing_highs_pool], reverse=True)
                cluster_width = cur_atr * sweep_buf * 8 if cur_atr > 0 else 0.001
                for k in range(len(prices) - min_pts + 1):
                    window = prices[k:k + min_pts]
                    if window[0] - window[-1] <= cluster_width:
                        high_pool = float(np.mean(window))
                        break

            # Find low pool level
            low_pool = np.nan
            if len(swing_lows_pool) >= min_pts:
                prices = sorted([p for p, _ in swing_lows_pool])
                cluster_width = cur_atr * sweep_buf * 8 if cur_atr > 0 else 0.001
                for k in range(len(prices) - min_pts + 1):
                    window = prices[k:k + min_pts]
                    if window[-1] - window[0] <= cluster_width:
                        low_pool = float(np.mean(window))
                        break

            # Downside sweep detection (bullish setup)
            if not np.isnan(low_pool):
                sweep_level = low_pool - sweep_buf * cur_atr
                if low[i] < sweep_level:
                    if close[i] > low_pool and body >= min_body * cur_atr:
                        signals[i] = 1
                    else:
                        pending_low_sweep_bar = i
                        pending_low_sweep_level = low_pool

            # Reclaim check for pending low sweep
            if pending_low_sweep_bar >= 0 and 0 < (i - pending_low_sweep_bar) <= reclaim_win:
                if close[i] > pending_low_sweep_level and body >= min_body * cur_atr:
                    signals[i] = 1
                    pending_low_sweep_bar = -1
            elif pending_low_sweep_bar >= 0 and (i - pending_low_sweep_bar) > reclaim_win:
                pending_low_sweep_bar = -1

            # Upside sweep detection (bearish setup)
            if not np.isnan(high_pool):
                sweep_level = high_pool + sweep_buf * cur_atr
                if high[i] > sweep_level:
                    if close[i] < high_pool and body >= min_body * cur_atr:
                        signals[i] = -1
                    else:
                        pending_high_sweep_bar = i
                        pending_high_sweep_level = high_pool

            # Reclaim check for pending high sweep
            if pending_high_sweep_bar >= 0 and 0 < (i - pending_high_sweep_bar) <= reclaim_win:
                if close[i] < pending_high_sweep_level and body >= min_body * cur_atr:
                    signals[i] = -1
                    pending_high_sweep_bar = -1
            elif pending_high_sweep_bar >= 0 and (i - pending_high_sweep_bar) > reclaim_win:
                pending_high_sweep_bar = -1

        return pd.Series(signals, index=features.index, dtype=int)

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'swing_order': [3, 5, 7],
            'pool_lookback_bars': [20, 40, 80, 120],
            'min_pool_points': [2, 3],
            'sweep_buffer_atr': [0.05, 0.1, 0.15, 0.2],
            'reclaim_window': [1, 2, 3],
            'min_reversal_body_atr': [0.0, 0.4],
            **self._base_sl_tp_grid(),
        }


class FibonacciRetracementPullbackStrategy(BaseStrategy):
    """Measured pullback entries to fixed fib levels on confirmed impulse legs.

    Detects impulse legs from confirmed swings, freezes 38.2/50/61.8/78.6 fib
    levels, enters on pullback into selected fib band with confirmation close
    and strict invalidation.
    """

    @property
    def name(self) -> str:
        return "FibonacciRetracementPullbackStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.TREND_FOLLOWING

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'swing_order': 5,
            'min_impulse_atr': 1.5,
            'fib_entry_set': '50_618',
            'confirmation_mode': 'close_back',
            'max_pullback_bars': 5,
            'invalidation_mode': 'below_786',
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
        }

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    @staticmethod
    def _parse_fib_set(fib_entry_set: str) -> Tuple[float, float]:
        """Parse fib_entry_set string to (upper_ratio, lower_ratio)."""
        mapping = {
            '38_50': (0.382, 0.500),
            '50_618': (0.500, 0.618),
            '382_618': (0.382, 0.618),
            '50_only': (0.480, 0.520),
        }
        return mapping.get(fib_entry_set, (0.500, 0.618))

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        order = int(self.params.get('swing_order', 5))
        min_imp = float(self.params.get('min_impulse_atr', 1.5))
        fib_set = str(self.params.get('fib_entry_set', '50_618'))
        conf_mode = str(self.params.get('confirmation_mode', 'close_back'))
        max_pb = int(self.params.get('max_pullback_bars', 5))
        inv_mode = str(self.params.get('invalidation_mode', 'below_786'))

        fib_upper_ratio, fib_lower_ratio = self._parse_fib_set(fib_set)

        sh_marks, sl_marks, sh_prices, sl_prices = _get_swing_state(features, order)
        high = features['High'].values
        low = features['Low'].values
        close = features['Close'].values
        opn = features['Open'].values
        atr_col = 'ATR_14'
        atr = features[atr_col].values if atr_col in features.columns else _get_atr(features, 14).values
        sh_v = sh_marks.values
        sl_v = sl_marks.values
        sh_prices_v = sh_prices.values
        sl_prices_v = sl_prices.values
        n = len(close)

        signals = np.zeros(n, dtype=int)

        # Leg tracking state
        last_sh_price = np.nan
        last_sl_price = np.nan
        leg_dir = 0  # 0=none, 1=bullish (low→high), -1=bearish (high→low)
        leg_start = np.nan
        leg_end = np.nan
        fib_upper_level = np.nan
        fib_lower_level = np.nan
        invalidation_level = np.nan
        pb_start_bar = -1

        warmup = order * 4 + max_pb + 20
        for i in range(warmup, n):
            cur_atr = atr[i] if not np.isnan(atr[i]) else 0.0

            # Update swing tracking
            new_leg = False
            if sh_v[i] and not np.isnan(sh_prices_v[i]):
                new_high = float(sh_prices_v[i])
                # Check for bullish impulse leg (prior swing low → this swing high)
                if not np.isnan(last_sl_price) and cur_atr > 0:
                    leg_size = new_high - last_sl_price
                    if leg_size > min_imp * cur_atr:
                        leg_dir = 1
                        leg_start = last_sl_price
                        leg_end = new_high
                        # Fib levels for bullish: retracement down from the high
                        fib_upper_level = leg_end - fib_upper_ratio * (leg_end - leg_start)
                        fib_lower_level = leg_end - fib_lower_ratio * (leg_end - leg_start)
                        if inv_mode == 'below_786':
                            invalidation_level = leg_end - 0.786 * (leg_end - leg_start)
                        else:
                            invalidation_level = leg_start
                        pb_start_bar = -1
                        new_leg = True
                last_sh_price = new_high

            if sl_v[i] and not np.isnan(sl_prices_v[i]):
                new_low = float(sl_prices_v[i])
                # Check for bearish impulse leg (prior swing high → this swing low)
                if not np.isnan(last_sh_price) and cur_atr > 0:
                    leg_size = last_sh_price - new_low
                    if leg_size > min_imp * cur_atr:
                        leg_dir = -1
                        leg_start = last_sh_price
                        leg_end = new_low
                        # Fib levels for bearish: retracement up from the low
                        fib_upper_level = leg_end + fib_upper_ratio * (leg_start - leg_end)
                        fib_lower_level = leg_end + fib_lower_ratio * (leg_start - leg_end)
                        if inv_mode == 'below_786':
                            invalidation_level = leg_end + 0.786 * (leg_start - leg_end)
                        else:
                            invalidation_level = leg_start
                        pb_start_bar = -1
                        new_leg = True
                last_sl_price = new_low

            if new_leg:
                continue

            # Bullish leg pullback entry
            if leg_dir == 1:
                # Invalidation
                if close[i] < invalidation_level:
                    leg_dir = 0
                    pb_start_bar = -1
                    continue
                # Check if price is in fib band (pulled back enough)
                in_band = low[i] <= fib_upper_level and close[i] >= fib_lower_level
                if in_band:
                    if pb_start_bar < 0:
                        pb_start_bar = i
                    if (i - pb_start_bar) <= max_pb:
                        confirmed = False
                        if conf_mode == 'engulfing_like':
                            confirmed = close[i] > opn[i] and close[i] > high[i - 1]
                        else:  # close_back
                            confirmed = close[i] > fib_upper_level
                        if confirmed:
                            signals[i] = 1
                            leg_dir = 0
                            pb_start_bar = -1
                    elif (i - pb_start_bar) > max_pb:
                        pb_start_bar = -1
                else:
                    if pb_start_bar >= 0 and close[i] > fib_upper_level:
                        pb_start_bar = -1

            # Bearish leg pullback entry
            elif leg_dir == -1:
                if close[i] > invalidation_level:
                    leg_dir = 0
                    pb_start_bar = -1
                    continue
                in_band = high[i] >= fib_upper_level and close[i] <= fib_lower_level
                if in_band:
                    if pb_start_bar < 0:
                        pb_start_bar = i
                    if (i - pb_start_bar) <= max_pb:
                        confirmed = False
                        if conf_mode == 'engulfing_like':
                            confirmed = close[i] < opn[i] and close[i] < low[i - 1]
                        else:  # close_back
                            confirmed = close[i] < fib_upper_level
                        if confirmed:
                            signals[i] = -1
                            leg_dir = 0
                            pb_start_bar = -1
                    elif (i - pb_start_bar) > max_pb:
                        pb_start_bar = -1
                else:
                    if pb_start_bar >= 0 and close[i] < fib_upper_level:
                        pb_start_bar = -1

        return pd.Series(signals, index=features.index, dtype=int)

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'swing_order': [3, 5, 7],
            'min_impulse_atr': [1.0, 1.5, 2.0],
            'fib_entry_set': ['38_50', '50_618', '382_618', '50_only'],
            'confirmation_mode': ['close_back', 'engulfing_like'],
            'max_pullback_bars': [3, 5, 8],
            'invalidation_mode': ['below_786', 'below_swing_start'],
            **self._base_sl_tp_grid(),
        }


class FractalSRZoneBreakRetestStrategy(BaseStrategy):
    """Zone-based break/retest continuation using confirmed fractal swing zones.

    Builds S/R zones from clustered confirmed fractal swings, scores zones by
    touch count and rejection quality, enters on breakout + retest + confirmation.
    """

    @property
    def name(self) -> str:
        return "FractalSRZoneBreakRetestStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'fractal_order': 5,
            'zone_width_atr': 0.4,
            'min_zone_touches': 2,
            'break_buffer_atr': 0.1,
            'retest_window': 4,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
        }

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        order = int(self.params.get('fractal_order', 5))
        zw = float(self.params.get('zone_width_atr', 0.4))
        min_touches = int(self.params.get('min_zone_touches', 2))
        break_buf = float(self.params.get('break_buffer_atr', 0.1))
        retest_win = int(self.params.get('retest_window', 4))

        sh_marks, sl_marks, sh_prices, sl_prices = _get_swing_state(features, order)
        high = features['High'].values
        low = features['Low'].values
        close = features['Close'].values
        atr_col = 'ATR_14'
        atr = features[atr_col].values if atr_col in features.columns else _get_atr(features, 14).values
        sh_v = sh_marks.values
        sl_v = sl_marks.values
        sh_prices_v = sh_prices.values
        sl_prices_v = sl_prices.values
        n = len(close)

        signals = np.zeros(n, dtype=int)

        # Zone state: list of [center, half_width, touch_count, wick_quality_sum, zone_type, creation_bar]
        # zone_type: 1=resistance, -1=support
        MAX_ZONES = 25
        zones: List[List] = []  # [center, half_width, touches, wick_quality_sum, zone_type, created_bar]

        # Pending breaks: (bar, level, direction)
        pending_break_bar = -1
        pending_break_level = np.nan
        pending_break_dir = 0
        pending_break_zone_idx = -1

        warmup = order * 2 + retest_win + 20
        for i in range(warmup, n):
            cur_atr = atr[i] if not np.isnan(atr[i]) else 0.0
            half_w = zw * cur_atr * 0.5

            # Add new swing highs as resistance zone candidates
            if sh_v[i] and cur_atr > 0 and not np.isnan(sh_prices_v[i]):
                swing_price = float(sh_prices_v[i])
                merged = False
                for z in zones:
                    if abs(swing_price - z[0]) <= zw * cur_atr and z[4] == 1:
                        # Merge into existing zone
                        z[0] = (z[0] * z[2] + swing_price) / (z[2] + 1)
                        z[2] += 1
                        # Wick quality: upper wick / range as rejection strength
                        bar_range = high[i] - low[i]
                        wick = high[i] - max(close[i], features['Open'].iat[i])
                        z[3] += (wick / (bar_range + 1e-10))
                        merged = True
                        break
                if not merged:
                    bar_range = high[i] - low[i]
                    wick = high[i] - max(close[i], features['Open'].iat[i])
                    zones.append([swing_price, half_w, 1, wick / (bar_range + 1e-10), 1, i])
                    if len(zones) > MAX_ZONES:
                        zones.pop(0)

            # Add new swing lows as support zone candidates
            if sl_v[i] and cur_atr > 0 and not np.isnan(sl_prices_v[i]):
                swing_price = float(sl_prices_v[i])
                merged = False
                for z in zones:
                    if abs(swing_price - z[0]) <= zw * cur_atr and z[4] == -1:
                        z[0] = (z[0] * z[2] + swing_price) / (z[2] + 1)
                        z[2] += 1
                        bar_range = high[i] - low[i]
                        wick = min(close[i], features['Open'].iat[i]) - low[i]
                        z[3] += (wick / (bar_range + 1e-10))
                        merged = True
                        break
                if not merged:
                    bar_range = high[i] - low[i]
                    wick = min(close[i], features['Open'].iat[i]) - low[i]
                    zones.append([swing_price, half_w, 1, wick / (bar_range + 1e-10), -1, i])
                    if len(zones) > MAX_ZONES:
                        zones.pop(0)

            # Evict stale zones (>200 bars old)
            zones = [z for z in zones if i - z[5] <= 200]

            # Break detection on qualified zones
            for zi, z in enumerate(zones):
                if z[2] < min_touches:
                    continue
                zone_upper = z[0] + z[1]
                zone_lower = z[0] - z[1]

                # Resistance breakout (bullish)
                if z[4] == 1 and close[i] > zone_upper + break_buf * cur_atr:
                    pending_break_bar = i
                    pending_break_level = zone_upper
                    pending_break_dir = 1
                    pending_break_zone_idx = zi

                # Support breakdown (bearish)
                if z[4] == -1 and close[i] < zone_lower - break_buf * cur_atr:
                    pending_break_bar = i
                    pending_break_level = zone_lower
                    pending_break_dir = -1
                    pending_break_zone_idx = zi

            # Retest confirmation
            if pending_break_dir == 1 and 0 < (i - pending_break_bar) <= retest_win:
                if low[i] <= pending_break_level + zw * cur_atr and close[i] > pending_break_level:
                    signals[i] = 1
                    # Remove broken zone
                    if 0 <= pending_break_zone_idx < len(zones):
                        zones.pop(pending_break_zone_idx)
                    pending_break_dir = 0
                    pending_break_zone_idx = -1

            if pending_break_dir == -1 and 0 < (i - pending_break_bar) <= retest_win:
                if high[i] >= pending_break_level - zw * cur_atr and close[i] < pending_break_level:
                    signals[i] = -1
                    if 0 <= pending_break_zone_idx < len(zones):
                        zones.pop(pending_break_zone_idx)
                    pending_break_dir = 0
                    pending_break_zone_idx = -1

            # Expire stale pending break
            if pending_break_dir != 0 and (i - pending_break_bar) > retest_win:
                pending_break_dir = 0
                pending_break_zone_idx = -1

        return pd.Series(signals, index=features.index, dtype=int)

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'fractal_order': [3, 5, 7, 9],
            'zone_width_atr': [0.25, 0.4, 0.6, 0.9],
            'min_zone_touches': [2, 3, 4],
            'break_buffer_atr': [0.05, 0.1, 0.2],
            'retest_window': [2, 4, 6, 8],
            **self._base_sl_tp_grid(),
        }


class SupplyDemandImpulseRetestStrategy(BaseStrategy):
    """Fresh supply/demand zones from displacement origin with early retest entries.

    Detects displacement impulse candles, builds origin zones from pre-impulse
    consolidation, enters on fresh retests with directional confirmation.
    """

    @property
    def name(self) -> str:
        return "SupplyDemandImpulseRetestStrategy"

    @property
    def category(self) -> StrategyCategory:
        return StrategyCategory.BREAKOUT_MOMENTUM

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'impulse_body_atr': 1.5,
            'base_lookback': 5,
            'zone_width_atr': 0.4,
            'max_zone_touches': 2,
            'confirmation_close_pos': 0.65,
            'sl_atr_mult': 2.0,
            'tp_atr_mult': 3.0,
        }

    def get_required_features(self) -> Set[str]:
        return {'ATR_14'}

    def generate_signals(self, features: pd.DataFrame, symbol: str) -> pd.Series:
        imp_body = float(self.params.get('impulse_body_atr', 1.5))
        base_lb = int(self.params.get('base_lookback', 5))
        zw = float(self.params.get('zone_width_atr', 0.4))
        max_touches = int(self.params.get('max_zone_touches', 2))
        conf_pos = float(self.params.get('confirmation_close_pos', 0.65))

        high = features['High'].values
        low = features['Low'].values
        close = features['Close'].values
        opn = features['Open'].values
        atr_col = 'ATR_14'
        atr = features[atr_col].values if atr_col in features.columns else _get_atr(features, 14).values
        n = len(close)

        signals = np.zeros(n, dtype=int)

        # Active zones: [zone_low, zone_high, direction(1=demand,-1=supply), touch_count, creation_bar]
        MAX_ZONES = 15
        zones: List[List] = []

        warmup = base_lb + 30
        for i in range(warmup, n):
            cur_atr = atr[i] if not np.isnan(atr[i]) else 0.0
            if cur_atr <= 0:
                continue

            body = abs(close[i] - opn[i])
            bar_range = high[i] - low[i]

            # Impulse detection
            is_bull_impulse = close[i] > opn[i] and body > imp_body * cur_atr
            is_bear_impulse = close[i] < opn[i] and body > imp_body * cur_atr

            if is_bull_impulse:
                # Demand zone = pre-impulse base
                base_start = max(0, i - base_lb)
                base_low = float(np.nanmin(low[base_start:i]))
                base_high = float(np.nanmax(high[base_start:i]))
                # Validate: base should be a consolidation, not too wide
                if (base_high - base_low) <= zw * cur_atr * 3:
                    zones.append([base_low, base_high, 1, 0, i])
                    if len(zones) > MAX_ZONES:
                        zones.pop(0)

            if is_bear_impulse:
                base_start = max(0, i - base_lb)
                base_low = float(np.nanmin(low[base_start:i]))
                base_high = float(np.nanmax(high[base_start:i]))
                if (base_high - base_low) <= zw * cur_atr * 3:
                    zones.append([base_low, base_high, -1, 0, i])
                    if len(zones) > MAX_ZONES:
                        zones.pop(0)

            # Evict stale and expired zones
            zones = [z for z in zones if i - z[4] <= 200 and z[3] < max_touches]

            # Retest detection
            close_pos = (close[i] - low[i]) / (bar_range + 1e-10)

            for z in zones:
                z_low, z_high, z_dir, z_touches, z_bar = z

                if z_dir == 1:  # Demand zone — bullish retest
                    if low[i] <= z_high and close[i] > z_low:
                        z[3] += 1
                        if close_pos >= conf_pos:
                            signals[i] = 1
                            z[3] = max_touches  # Expire after entry

                elif z_dir == -1:  # Supply zone — bearish retest
                    if high[i] >= z_low and close[i] < z_high:
                        z[3] += 1
                        inv_close_pos = (high[i] - close[i]) / (bar_range + 1e-10)
                        if inv_close_pos >= conf_pos:
                            signals[i] = -1
                            z[3] = max_touches

                # Invalidation: price closes through zone in wrong direction
                if z_dir == 1 and close[i] < z_low - zw * cur_atr:
                    z[3] = max_touches  # force expire
                if z_dir == -1 and close[i] > z_high + zw * cur_atr:
                    z[3] = max_touches

        return pd.Series(signals, index=features.index, dtype=int)

    def get_param_grid(self) -> Dict[str, List]:
        return {
            'impulse_body_atr': [1.0, 1.2, 1.5, 1.8, 2.2],
            'base_lookback': [3, 5, 8, 12],
            'zone_width_atr': [0.25, 0.4, 0.6, 0.8],
            'max_zone_touches': [1, 2, 3],
            'confirmation_close_pos': [0.55, 0.65, 0.75],
            **self._base_sl_tp_grid(),
        }


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

class StrategyRegistry:
    """
    Central registry for all available strategies.
    
    Provides:
    - Strategy discovery and listing
    - Strategy instantiation by name
    - Category-based filtering
    """
    
    # All available strategies
    _strategies: Dict[str, type] = {
        # Trend Following
        'EMACrossoverStrategy': EMACrossoverStrategy,
        'SupertrendStrategy': SupertrendStrategy,
        'MACDTrendStrategy': MACDTrendStrategy,
        'ADXTrendStrategy': ADXTrendStrategy,
        'IchimokuStrategy': IchimokuStrategy,
        'HullMATrendStrategy': HullMATrendStrategy,
        'EMARibbonADXStrategy': EMARibbonADXStrategy,
        'AroonTrendStrategy': AroonTrendStrategy,
        'ADXDIStrengthStrategy': ADXDIStrengthStrategy,
        'KeltnerPullbackStrategy': KeltnerPullbackStrategy,
        
        # Mean Reversion
        'RSIExtremesStrategy': RSIExtremesStrategy,
        'BollingerBounceStrategy': BollingerBounceStrategy,
        'ZScoreMRStrategy': ZScoreMRStrategy,
        'StochasticReversalStrategy': StochasticReversalStrategy,
        'CCIReversalStrategy': CCIReversalStrategy,
        'WilliamsRStrategy': WilliamsRStrategy,
        'RSITrendFilteredMRStrategy': RSITrendFilteredMRStrategy,
        'StochRSITrendGateStrategy': StochRSITrendGateStrategy,
        'FisherTransformMRStrategy': FisherTransformMRStrategy,
        'ZScoreVWAPReversionStrategy': ZScoreVWAPReversionStrategy,
        
        # Breakout/Momentum
        'DonchianBreakoutStrategy': DonchianBreakoutStrategy,
        'VolatilityBreakoutStrategy': VolatilityBreakoutStrategy,
        'MomentumBurstStrategy': MomentumBurstStrategy,
        'SqueezeBreakoutStrategy': SqueezeBreakoutStrategy,
        'KeltnerBreakoutStrategy': KeltnerBreakoutStrategy,
        'PivotBreakoutStrategy': PivotBreakoutStrategy,
        'MACDHistogramMomentumStrategy': MACDHistogramMomentumStrategy,

        # New strategies (D1-D15)
        # Breakout
        'InsideBarBreakoutStrategy': InsideBarBreakoutStrategy,
        'NarrowRangeBreakoutStrategy': NarrowRangeBreakoutStrategy,
        'VolumeSpikeMomentumStrategy': VolumeSpikeMomentumStrategy,
        'ATRPercentileBreakoutStrategy': ATRPercentileBreakoutStrategy,
        # Mean Reversion
        'TurtleSoupReversalStrategy': TurtleSoupReversalStrategy,
        'PinBarReversalStrategy': PinBarReversalStrategy,
        'EngulfingPatternStrategy': EngulfingPatternStrategy,
        'RSIDivergenceStrategy': RSIDivergenceStrategy,
        'MACDDivergenceStrategy': MACDDivergenceStrategy,
        'KeltnerFadeStrategy': KeltnerFadeStrategy,
        'ROCExhaustionReversalStrategy': ROCExhaustionReversalStrategy,
        # Trend Following
        'OBVDivergenceStrategy': OBVDivergenceStrategy,
        'EMAPullbackContinuationStrategy': EMAPullbackContinuationStrategy,
        'ParabolicSARTrendStrategy': ParabolicSARTrendStrategy,
        'KaufmanAMATrendStrategy': KaufmanAMATrendStrategy,

        # New strategies (42 → 50)
        # Trend Following
        'VortexTrendStrategy': VortexTrendStrategy,
        'ElderRayBullBearStrategy': ElderRayBullBearStrategy,
        'MarketStructureBOSPullbackStrategy': MarketStructureBOSPullbackStrategy,
        'FibonacciRetracementPullbackStrategy': FibonacciRetracementPullbackStrategy,
        # Mean Reversion
        'LiquiditySweepReversalStrategy': LiquiditySweepReversalStrategy,
        # Breakout/Momentum
        'TRIXMomentumStrategy': TRIXMomentumStrategy,
        'FractalSRZoneBreakRetestStrategy': FractalSRZoneBreakRetestStrategy,
        'SupplyDemandImpulseRetestStrategy': SupplyDemandImpulseRetestStrategy,
    }
    
    @classmethod
    def get(cls, name: str, **params) -> BaseStrategy:
        """Get strategy instance by name."""
        resolved_name = name
        if resolved_name not in cls._strategies:
            resolved_name = _STRATEGY_MIGRATION.get(resolved_name, resolved_name)
        if resolved_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        return cls._strategies[resolved_name](**params)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """List all available strategy names."""
        return list(cls._strategies.keys())
    
    @classmethod
    def list_by_category(cls, category: StrategyCategory) -> List[str]:
        """List strategies in a specific category."""
        result = []
        for name, strategy_cls in cls._strategies.items():
            instance = strategy_cls()
            if instance.category == category:
                result.append(name)
        return result
    
    @classmethod
    def register(cls, strategy_cls: type):
        """Register a new strategy class."""
        instance = strategy_cls()
        cls._strategies[instance.name] = strategy_cls
    
    @classmethod
    def get_all_instances(cls) -> List[BaseStrategy]:
        """Get instances of all strategies with default params."""
        return [strategy_cls() for strategy_cls in cls._strategies.values()]
    
    @classmethod
    def count(cls) -> int:
        """Get total number of registered strategies."""
        return len(cls._strategies)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BaseStrategy',
    'StrategyRegistry',
    'StrategyCategory',
    # Global grid constants
    '_GLOBAL_SL_GRID',
    '_GLOBAL_TP_GRID',
    # Trend Following
    'EMACrossoverStrategy',
    'SupertrendStrategy',
    'MACDTrendStrategy',
    'ADXTrendStrategy',
    'IchimokuStrategy',
    'HullMATrendStrategy',
    'EMARibbonADXStrategy',
    'AroonTrendStrategy',
    'ADXDIStrengthStrategy',
    'KeltnerPullbackStrategy',
    # Mean Reversion
    'RSIExtremesStrategy',
    'BollingerBounceStrategy',
    'ZScoreMRStrategy',
    'StochasticReversalStrategy',
    'CCIReversalStrategy',
    'WilliamsRStrategy',
    'RSITrendFilteredMRStrategy',
    'StochRSITrendGateStrategy',
    'FisherTransformMRStrategy',
    'ZScoreVWAPReversionStrategy',
    # Breakout/Momentum
    'DonchianBreakoutStrategy',
    'VolatilityBreakoutStrategy',
    'MomentumBurstStrategy',
    'SqueezeBreakoutStrategy',
    'KeltnerBreakoutStrategy',
    'PivotBreakoutStrategy',
    'MACDHistogramMomentumStrategy',
    # New strategies (D1-D15)
    'InsideBarBreakoutStrategy',
    'NarrowRangeBreakoutStrategy',
    'TurtleSoupReversalStrategy',
    'PinBarReversalStrategy',
    'EngulfingPatternStrategy',
    'VolumeSpikeMomentumStrategy',
    'RSIDivergenceStrategy',
    'MACDDivergenceStrategy',
    'OBVDivergenceStrategy',
    'KeltnerFadeStrategy',
    'ROCExhaustionReversalStrategy',
    'EMAPullbackContinuationStrategy',
    'ParabolicSARTrendStrategy',
    'ATRPercentileBreakoutStrategy',
    'KaufmanAMATrendStrategy',
    # New strategies (42 → 50)
    'VortexTrendStrategy',
    'ElderRayBullBearStrategy',
    'MarketStructureBOSPullbackStrategy',
    'FibonacciRetracementPullbackStrategy',
    'LiquiditySweepReversalStrategy',
    'TRIXMomentumStrategy',
    'FractalSRZoneBreakRetestStrategy',
    'SupplyDemandImpulseRetestStrategy',
]

# Migration map for retired strategy names → current names
_STRATEGY_MIGRATION = {
    'VWAPDeviationReversionStrategy': 'ZScoreVWAPReversionStrategy',
    'VWAPDeviationReversalStrategy': 'ZScoreVWAPReversionStrategy',
}


if __name__ == "__main__":
    # Test strategy registry
    print(f"Total strategies: {StrategyRegistry.count()}")
    print(f"\nAll strategies: {StrategyRegistry.list_all()}")
    
    print(f"\nTrend Following: {StrategyRegistry.list_by_category(StrategyCategory.TREND_FOLLOWING)}")
    print(f"Mean Reversion: {StrategyRegistry.list_by_category(StrategyCategory.MEAN_REVERSION)}")
    print(f"Breakout: {StrategyRegistry.list_by_category(StrategyCategory.BREAKOUT_MOMENTUM)}")
    
    # Test instantiation
    st = StrategyRegistry.get('SupertrendStrategy', atr_period=7, multiplier=2.0)
    print(f"\nInstantiated: {st}")
    print(f"Param grid: {st.get_param_grid()}")

    # Verify all strategies have standardized SL/TP grids
    print("\n--- SL/TP Grid Verification ---")
    expected_sl = _GLOBAL_SL_GRID
    expected_tp = _GLOBAL_TP_GRID
    all_ok = True
    for name in StrategyRegistry.list_all():
        instance = StrategyRegistry.get(name)
        grid = instance.get_param_grid()
        sl = grid.get('sl_atr_mult', [])
        tp = grid.get('tp_atr_mult', [])
        if sl != expected_sl or tp != expected_tp:
            print(f"  MISMATCH: {name} - sl={sl}, tp={tp}")
            all_ok = False
    if all_ok:
        print("  All strategies have standardized SL/TP grids ✓")
