"""
FX Portfolio Manager - Position Management Module
==================================================

Handles all position management operations including:
- Position entry/exit logic
- Stop loss and take profit management
- Position sizing based on risk
- Multi-position tracking
- Exit signal processing

This module is separate to allow for future enhancements:
- Trailing stops
- Partial position closing
- Break-even stops
- Scale in/out logic

Version: 3.0 (Portfolio Manager)
"""

import logging
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import math

import pandas as pd
import numpy as np

from pm_core import get_instrument_spec, InstrumentSpec, SignalType, TradeStatus

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# TRADE TAG UTILITIES (FIX #2: D1 + Lower-TF Support)
# =============================================================================

class TradeTagEncoder:
    """
    Encodes and decodes trade metadata in MT5 magic numbers and comments.

    Magic number encoding:
        Deterministic CRC32 of symbol|timeframe|regime for unique identification.

    Comment encoding goals:
        - Must fit within MT5's practical comment length limits.
        - Must be machine-parseable and backwards compatible with the older "PM:" format.
        - Must support extracting at least: symbol, timeframe, direction, and optional risk metadata.

    Formats
    --------
    v1 (legacy):
        "PM:{symbol}:{timeframe}:{strategy}:{direction}"

    v2 (legacy extended):
        "PM2:{symbol}:{timeframe}:{scode}:{dir}:{tier}:{risk_tenths}"

        where:
            scode        = short strategy code (base36 CRC32, 4-5 chars)
            dir          = 'L' or 'S'
            tier         = 1..4
            risk_tenths  = int(round(risk_pct * 10))

    v3 (preferred, non-tiered):
        "PM3:{symbol}:{timeframe}:{scode}:{dir}:{risk_tenths}"

    v2/v3 short (backward compatibility):
        "PM2:{symbol}:{timeframe}:{scode}"
        "PM3:{symbol}:{timeframe}:{scode}"

        Older live versions used these compact forms (no direction/tier/risk).
    """

    COMMENT_PREFIX_V1 = "PM:"
    COMMENT_PREFIX_V2 = "PM2:"
    COMMENT_PREFIX_V3 = "PM3:"

    @staticmethod
    def _base36(n: int) -> str:
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        if n == 0:
            return "0"
        out = []
        while n > 0:
            n, r = divmod(n, 36)
            out.append(chars[r])
        return "".join(reversed(out))

    @staticmethod
    def encode_magic(symbol: str, timeframe: str, regime: str) -> int:
        """Generate deterministic magic number for a (symbol, timeframe, regime) tuple."""
        key = f"{symbol}|{timeframe}|{regime}"
        return zlib.crc32(key.encode('utf-8')) & 0x7FFFFFFF

    @staticmethod
    def _strategy_code(strategy_name: str, max_len: int = 5) -> str:
        """Short stable code for a strategy name (CRC32 base36)."""
        try:
            crc = zlib.crc32(str(strategy_name).encode("utf-8")) & 0xFFFFFFFF
            code = TradeTagEncoder._base36(crc)
            return code[-max_len:].rjust(max_len, "0")
        except Exception:
            return "00000"[-max_len:]

    @staticmethod
    def encode_comment(symbol: str,
                       timeframe: str,
                       strategy_name: str,
                       direction: str,
                       risk_pct: float = None,
                       tier: int = None) -> str:
        """
        Encode trade metadata into MT5 comment string.

        If risk_pct is provided, emits compact v3 (PM3:...) without tier metadata.
        If both risk_pct and tier are provided, emits legacy v2 for backward compatibility.
        Otherwise falls back to legacy v1.
        """
        # Normalize direction for compactness
        dir_norm = str(direction).upper()
        dir_short = "L" if dir_norm.startswith("L") else "S" if dir_norm.startswith("S") else dir_norm[:1]

        if risk_pct is not None and tier is not None:
            scode = TradeTagEncoder._strategy_code(strategy_name, max_len=5)
            risk_tenths = int(round(float(risk_pct) * 10.0))
            tier_i = int(tier)
            return f"{TradeTagEncoder.COMMENT_PREFIX_V2}{symbol}:{timeframe}:{scode}:{dir_short}:{tier_i}:{risk_tenths}"

        if risk_pct is not None:
            scode = TradeTagEncoder._strategy_code(strategy_name, max_len=5)
            risk_tenths = int(round(float(risk_pct) * 10.0))
            return f"{TradeTagEncoder.COMMENT_PREFIX_V3}{symbol}:{timeframe}:{scode}:{dir_short}:{risk_tenths}"

        # Legacy v1
        return f"{TradeTagEncoder.COMMENT_PREFIX_V1}{symbol}:{timeframe}:{strategy_name}:{str(direction).upper()}"

    @staticmethod
    def _normalize_direction(direction_raw: Any) -> Optional[str]:
        """Normalize compact direction tokens into LONG/SHORT."""
        if direction_raw is None:
            return None
        d = str(direction_raw).strip().upper()
        if d in {"L", "LONG", "BUY", "1"}:
            return "LONG"
        if d in {"S", "SHORT", "SELL", "-1"}:
            return "SHORT"
        return d or None

    @staticmethod
    def decode_comment(comment: str) -> Optional[Dict[str, Any]]:
        """
        Decode trade metadata from MT5 comment string.

        Returns dict with keys:
            symbol, timeframe, direction, (optional) strategy_code, tier, risk_pct
        """
        if not comment:
            return None

        try:
            # PM2/PM3 compact and extended formats:
            # - PMx:symbol:tf:scode
            # - PM2:symbol:tf:scode:dir[:tier[:risk_tenths]]   (legacy)
            # - PM3:symbol:tf:scode:dir[:risk_tenths]          (current)
            for prefix in (TradeTagEncoder.COMMENT_PREFIX_V2, TradeTagEncoder.COMMENT_PREFIX_V3):
                if comment.startswith(prefix):
                    parts = comment[len(prefix):].split(':')
                    if len(parts) >= 3:
                        symbol, tf, scode = parts[:3]
                        decoded: Dict[str, Any] = {
                            "symbol": symbol,
                            "timeframe": tf,
                            "strategy_code": scode,
                        }

                        if len(parts) >= 4:
                            direction = TradeTagEncoder._normalize_direction(parts[3])
                            if direction:
                                decoded["direction"] = direction

                        if len(parts) >= 6:
                            # Legacy v2: ...:dir:tier:risk
                            try:
                                decoded["tier"] = int(parts[4])
                            except Exception:
                                pass
                            # Default interpretation for compact tags is tenths, but accept float too.
                            risk_raw = parts[5]
                            try:
                                if "." in risk_raw:
                                    decoded["risk_pct"] = float(risk_raw)
                                else:
                                    decoded["risk_pct"] = float(int(risk_raw)) / 10.0
                            except Exception:
                                pass
                        elif len(parts) >= 5:
                            # PM3 extended: ...:dir:risk
                            if prefix == TradeTagEncoder.COMMENT_PREFIX_V3:
                                risk_raw = parts[4]
                                try:
                                    if "." in risk_raw:
                                        decoded["risk_pct"] = float(risk_raw)
                                    else:
                                        decoded["risk_pct"] = float(int(risk_raw)) / 10.0
                                except Exception:
                                    pass
                            else:
                                # PM2 partial legacy: ...:dir:tier
                                try:
                                    decoded["tier"] = int(parts[4])
                                except Exception:
                                    pass

                        return decoded

            if comment.startswith(TradeTagEncoder.COMMENT_PREFIX_V1):
                parts = comment[len(TradeTagEncoder.COMMENT_PREFIX_V1):].split(':')
                # PM:symbol:tf:strategy:direction
                if len(parts) >= 4:
                    direction = TradeTagEncoder._normalize_direction(parts[3])
                    return {
                        'symbol': parts[0],
                        'timeframe': parts[1],
                        'strategy_name': parts[2],
                        'direction': direction or parts[3],
                    }

            # Very old shorthand comments carried strategy tag only (no timeframe).
            if comment.startswith("PM_"):
                strategy_tag = comment[3:].strip()
                if strategy_tag:
                    return {
                        "strategy_tag": strategy_tag,
                    }
        except Exception:
            return None

        return None

    @staticmethod
    def is_d1_trade(comment: str) -> bool:
        decoded = TradeTagEncoder.decode_comment(comment)
        return decoded is not None and decoded.get('timeframe') == 'D1'

    @staticmethod
    def get_timeframe_from_comment(comment: str) -> Optional[str]:
        decoded = TradeTagEncoder.decode_comment(comment)
        return decoded.get('timeframe') if decoded else None

    @staticmethod
    def get_risk_pct_from_comment(comment: str) -> Optional[float]:
        decoded = TradeTagEncoder.decode_comment(comment)
        if not decoded:
            return None
        return decoded.get('risk_pct')

    @staticmethod
    def get_tier_from_comment(comment: str) -> Optional[int]:
        decoded = TradeTagEncoder.decode_comment(comment)
        if not decoded:
            return None
        t = decoded.get('tier')
        return int(t) if t is not None else None


# =============================================================================
# DATA CLASSES

# =============================================================================

@dataclass
class PositionConfig:
    """Configuration for position management."""
    # Risk management
    risk_per_trade_pct: float = 1.0
    max_position_size: float = 0.0  # 0 = no limit (use broker limit)
    min_position_size: float = 0.01
    

    risk_basis: str = "balance"  # "balance" or "equity"
    max_risk_pct: float = 2.0  # hard safety cap (skip if exceeded)
    risk_tolerance_pct: float = 2.0  # allowed deviation vs target before logging/adjustment
    auto_widen_sl: bool = True  # widen SL only to satisfy broker min stop distance / constraints

    # Stop management
    use_trailing_stop: bool = False
    trailing_stop_pips: float = 0.0
    trailing_activation_pips: float = 0.0
    
    use_breakeven_stop: bool = False
    breakeven_trigger_pips: float = 0.0
    breakeven_offset_pips: float = 1.0
    
    # Scaling
    allow_scaling: bool = False
    max_scale_ins: int = 3
    scale_in_pct: float = 50.0
    
    # Time-based exits
    max_trade_duration_bars: int = 0  # 0 = no limit
    
    # Spread/slippage
    use_spread: bool = True
    use_slippage: bool = True
    slippage_pips: float = 0.5


@dataclass
class OpenPosition:
    """
    Represents an open trading position.
    
    Tracks all relevant position information for management.
    """
    # Identification
    position_id: str
    symbol: str
    magic: int
    
    # Direction and size
    direction: int  # 1 for long, -1 for short
    volume: float
    
    # Prices
    entry_price: float
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Original stops (for reference)
    initial_stop_loss: float = 0.0
    initial_take_profit: float = 0.0
    
    # Timing
    entry_time: datetime = None
    entry_bar: int = 0
    
    # P&L tracking
    unrealized_pnl_pips: float = 0.0
    unrealized_pnl_dollars: float = 0.0
    highest_pnl_pips: float = 0.0  # For trailing stop
    
    # Status
    is_open: bool = True
    exit_price: float = 0.0
    exit_time: datetime = None
    exit_bar: int = 0
    exit_reason: str = ""
    realized_pnl_pips: float = 0.0
    realized_pnl_dollars: float = 0.0
    
    # Scale-in tracking
    scale_count: int = 0
    avg_entry_price: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.direction == 1
    
    @property
    def is_short(self) -> bool:
        return self.direction == -1


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    volume: float
    risk_amount: float
    sl_pips: float
    pip_value: float
    equity: float
    details: str


@dataclass
class ExitCheckResult:
    """Result of checking if position should exit."""
    should_exit: bool
    exit_reason: str = ""
    exit_price: float = 0.0


# =============================================================================
# POSITION CALCULATOR
# =============================================================================

class PositionCalculator:
    """
    Calculates position sizes and prices.
    
    Handles:
    - Risk-based position sizing
    - Entry/exit price calculation with spread
    - Stop loss and take profit price calculation
    """
    
    def __init__(self, config: PositionConfig):
        """
        Initialize calculator.
        
        Args:
            config: Position configuration
        """
        self.config = config
    
    def calculate_position_size(self,
                                 equity: float,
                                 sl_pips: float,
                                 spec: InstrumentSpec) -> PositionSizeResult:
        """
        Calculate position size based on risk percentage.
        
        Args:
            equity: Current account equity
            sl_pips: Stop loss distance in pips
            spec: Instrument specification
            
        Returns:
            PositionSizeResult with calculated volume and details
        """
        # Calculate risk amount
        risk_amount = equity * (self.config.risk_per_trade_pct / 100.0)
        
        # Get pip value
        pip_value = spec.pip_value
        
        # Calculate lot size using tick-based math if available, else pip-based
        if spec.tick_value > 0 and spec.tick_size > 0 and sl_pips > 0:
            # Tick-based sizing: more accurate for CFDs/indices
            sl_price_dist = spec.pips_to_price(sl_pips)
            ticks = sl_price_dist / spec.tick_size
            loss_per_lot = ticks * spec.tick_value
            if loss_per_lot > 0:
                volume = risk_amount / loss_per_lot
            else:
                volume = self.config.min_position_size
        elif pip_value > 0 and sl_pips > 0:
            # Fallback to pip-based sizing
            volume = risk_amount / (sl_pips * pip_value)
        else:
            volume = self.config.min_position_size

        # If risk-based volume is below min_lot, the trade is not viable.
        # Clamping up to min_lot could take materially more risk than intended.
        if volume < spec.min_lot:
            logger.warning(
                f"Risk-based volume {volume:.8f} below min_lot {spec.min_lot} "
                f"(equity={equity:.2f}, sl_pips={sl_pips:.1f}); skipping trade"
            )
            return PositionSizeResult(
                volume=0.0,
                risk_amount=risk_amount,
                sl_pips=sl_pips,
                pip_value=pip_value,
                equity=equity,
                details=(
                    f"Equity: {equity:.2f}, Risk: {risk_amount:.2f}, "
                    f"SL: {sl_pips:.1f} pips - volume below min_lot, skipped"
                ),
            )
        
        # Apply limits
        volume = max(spec.min_lot, volume)
        volume = min(spec.max_lot, volume)
        
        if self.config.max_position_size > 0:
            volume = min(self.config.max_position_size, volume)
        
        volume = max(self.config.min_position_size, volume)
        
        # Round down to broker volume_step (not just 0.01) to avoid accidental risk increases
        # This is essential for CFDs/indices which may have different volume_step
        volume_step = spec.volume_step if spec.volume_step > 0 else 0.01
        volume = math.floor(volume / volume_step) * volume_step
        volume = round(volume, 8)

        if volume < spec.min_lot:
            logger.warning(
                f"Volume {volume:.8f} below min_lot {spec.min_lot} after rounding "
                f"(equity={equity:.2f}, sl_pips={sl_pips:.1f}); skipping trade"
            )
            volume = 0.0
        
        details = (f"Equity: {equity:.2f}, Risk: {risk_amount:.2f} "
                  f"({self.config.risk_per_trade_pct}%), SL: {sl_pips:.1f} pips, "
                  f"PipValue: {pip_value:.4f}, Volume: {volume:.4f}, "
                  f"VolumeStep: {volume_step}")
        
        return PositionSizeResult(
            volume=volume,
            risk_amount=risk_amount,
            sl_pips=sl_pips,
            pip_value=pip_value,
            equity=equity,
            details=details
        )
    
    def calculate_entry_price(self,
                               mid_price: float,
                               is_long: bool,
                               spec: InstrumentSpec) -> float:
        """
        Calculate entry price including spread and slippage.
        
        Args:
            mid_price: Mid/open price
            is_long: True for long position
            spec: Instrument specification
            
        Returns:
            Adjusted entry price
        """
        entry_price = mid_price
        
        # Apply spread
        if self.config.use_spread:
            entry_price = spec.get_entry_price(mid_price, is_long)
        
        # Apply slippage
        if self.config.use_slippage:
            slippage_price = spec.pips_to_price(self.config.slippage_pips)
            if is_long:
                entry_price += slippage_price
            else:
                entry_price -= slippage_price
        
        return entry_price
    
    def calculate_exit_price(self,
                              mid_price: float,
                              is_long: bool,
                              spec: InstrumentSpec) -> float:
        """
        Calculate exit price including spread.
        
        Args:
            mid_price: Mid/close price
            is_long: True for long position
            spec: Instrument specification
            
        Returns:
            Adjusted exit price
        """
        if self.config.use_spread:
            return spec.get_exit_price(mid_price, is_long)
        return mid_price
    
    def calculate_stop_prices(self,
                               entry_price: float,
                               sl_pips: float,
                               tp_pips: float,
                               is_long: bool,
                               spec: InstrumentSpec) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit prices.
        
        Args:
            entry_price: Position entry price
            sl_pips: Stop loss in pips
            tp_pips: Take profit in pips
            is_long: True for long position
            spec: Instrument specification
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        sl_distance = spec.pips_to_price(sl_pips)
        tp_distance = spec.pips_to_price(tp_pips)
        
        if is_long:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit


# =============================================================================
# POSITION MANAGER
# =============================================================================

class PositionManager:
    """
    Manages open positions and exit logic.
    
    Handles:
    - Position tracking
    - Exit signal detection (SL, TP, signal-based)
    - Position updates
    - P&L calculation
    
    Matches backtester logic exactly for consistency.
    """
    
    def __init__(self, config: PositionConfig):
        """
        Initialize position manager.
        
        Args:
            config: Position configuration
        """
        self.config = config
        self.calculator = PositionCalculator(config)
        
        # Active positions by ID
        self.positions: Dict[str, OpenPosition] = {}
        
        # Closed positions for history
        self.closed_positions: List[OpenPosition] = []
        
        # Position ID counter
        self._next_id = 1
    
    def _generate_position_id(self, symbol: str, magic: int) -> str:
        """Generate unique position ID."""
        pos_id = f"{symbol}_{magic}_{self._next_id}"
        self._next_id += 1
        return pos_id
    
    def open_position(self,
                      symbol: str,
                      magic: int,
                      direction: int,
                      entry_price: float,
                      volume: float,
                      stop_loss: float,
                      take_profit: float,
                      entry_time: datetime = None,
                      entry_bar: int = 0) -> OpenPosition:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            magic: Magic number for identification
            direction: 1 for long, -1 for short
            entry_price: Entry price
            volume: Position size in lots
            stop_loss: Stop loss price
            take_profit: Take profit price
            entry_time: Entry timestamp
            entry_bar: Entry bar index
            
        Returns:
            Created OpenPosition object
        """
        pos_id = self._generate_position_id(symbol, magic)
        
        position = OpenPosition(
            position_id=pos_id,
            symbol=symbol,
            magic=magic,
            direction=direction,
            volume=volume,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            initial_stop_loss=stop_loss,
            initial_take_profit=take_profit,
            entry_time=entry_time or datetime.now(),
            entry_bar=entry_bar,
            avg_entry_price=entry_price
        )
        
        self.positions[pos_id] = position
        
        logger.debug(f"Opened position {pos_id}: {'LONG' if direction == 1 else 'SHORT'} "
                    f"{symbol} @ {entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
        
        return position
    
    def close_position(self,
                       position: OpenPosition,
                       exit_price: float,
                       exit_reason: str,
                       exit_time: datetime = None,
                       exit_bar: int = 0,
                       spec: InstrumentSpec = None) -> OpenPosition:
        """
        Close an open position.
        
        Args:
            position: Position to close
            exit_price: Exit price
            exit_reason: Reason for exit
            exit_time: Exit timestamp
            exit_bar: Exit bar index
            spec: Instrument specification for P&L calculation
            
        Returns:
            Updated (closed) position
        """
        if not position.is_open:
            logger.warning(f"Position {position.position_id} already closed")
            return position
        
        # Calculate P&L
        if spec is None:
            spec = get_instrument_spec(position.symbol)
        
        if position.is_long:
            pnl_pips = spec.price_to_pips(exit_price - position.entry_price)
        else:
            pnl_pips = spec.price_to_pips(position.entry_price - exit_price)
        
        pnl_dollars = pnl_pips * position.volume * spec.pip_value
        
        # Update position
        position.is_open = False
        position.exit_price = exit_price
        position.exit_time = exit_time or datetime.now()
        position.exit_bar = exit_bar
        position.exit_reason = exit_reason
        position.realized_pnl_pips = round(pnl_pips, 2)
        position.realized_pnl_dollars = round(pnl_dollars, 2)
        
        # Move to closed positions
        if position.position_id in self.positions:
            del self.positions[position.position_id]
        self.closed_positions.append(position)
        
        logger.debug(f"Closed position {position.position_id}: {exit_reason}, "
                    f"PnL: {pnl_pips:.1f} pips (${pnl_dollars:.2f})")
        
        return position
    
    def update_position_price(self,
                               position: OpenPosition,
                               current_price: float,
                               spec: InstrumentSpec = None):
        """
        Update position with current price and calculate unrealized P&L.
        
        Args:
            position: Position to update
            current_price: Current market price
            spec: Instrument specification
        """
        if spec is None:
            spec = get_instrument_spec(position.symbol)
        
        position.current_price = current_price
        
        # Calculate unrealized P&L
        if position.is_long:
            pnl_pips = spec.price_to_pips(current_price - position.entry_price)
        else:
            pnl_pips = spec.price_to_pips(position.entry_price - current_price)
        
        pnl_dollars = pnl_pips * position.volume * spec.pip_value
        
        position.unrealized_pnl_pips = round(pnl_pips, 2)
        position.unrealized_pnl_dollars = round(pnl_dollars, 2)
        
        # Track highest P&L for trailing stop
        if pnl_pips > position.highest_pnl_pips:
            position.highest_pnl_pips = pnl_pips
    
    def check_exit_conditions(self,
                               position: OpenPosition,
                               high_price: float,
                               low_price: float,
                               current_bar: int,
                               spec: InstrumentSpec = None) -> ExitCheckResult:
        """
        Check if position should exit based on SL/TP.
        
        Matches backtester logic exactly:
        - Longs: SL hit if bid_low <= SL, TP hit if bid_high >= TP
        - Shorts: SL hit if ask_high >= SL, TP hit if ask_low <= TP
        
        Args:
            position: Position to check
            high_price: Bar high price
            low_price: Bar low price
            current_bar: Current bar index
            spec: Instrument specification
            
        Returns:
            ExitCheckResult indicating if exit should occur
        """
        if spec is None:
            spec = get_instrument_spec(position.symbol)
        
        # Calculate bid/ask from high/low
        if self.config.use_spread:
            bid_high = high_price - spec.get_half_spread_price()
            bid_low = low_price - spec.get_half_spread_price()
            ask_high = high_price + spec.get_half_spread_price()
            ask_low = low_price + spec.get_half_spread_price()
        else:
            # When spread disabled, bid and ask are the same as mid prices
            bid_high = high_price
            bid_low = low_price
            ask_high = high_price
            ask_low = low_price
        
        exit_price = None
        exit_reason = ""
        
        if position.is_long:
            # Long position exits at BID
            if bid_low <= position.stop_loss:
                exit_price = position.stop_loss
                exit_reason = TradeStatus.CLOSED_SL.value
            elif bid_high >= position.take_profit:
                exit_price = position.take_profit
                exit_reason = TradeStatus.CLOSED_TP.value
        else:
            # Short position exits at ASK
            if ask_high >= position.stop_loss:
                exit_price = position.stop_loss
                exit_reason = TradeStatus.CLOSED_SL.value
            elif ask_low <= position.take_profit:
                exit_price = position.take_profit
                exit_reason = TradeStatus.CLOSED_TP.value
        
        # Check time-based exit
        if self.config.max_trade_duration_bars > 0:
            bars_held = current_bar - position.entry_bar
            if bars_held >= self.config.max_trade_duration_bars:
                if exit_price is None:
                    # Use current close as exit
                    exit_price = (high_price + low_price) / 2
                    exit_reason = TradeStatus.CLOSED_TIME.value
        
        if exit_price is not None:
            return ExitCheckResult(
                should_exit=True,
                exit_reason=exit_reason,
                exit_price=exit_price
            )
        
        return ExitCheckResult(should_exit=False)
    
    def apply_trailing_stop(self,
                            position: OpenPosition,
                            spec: InstrumentSpec = None):
        """
        Apply trailing stop if configured and conditions met.
        
        Args:
            position: Position to update
            spec: Instrument specification
        """
        if not self.config.use_trailing_stop:
            return
        
        if spec is None:
            spec = get_instrument_spec(position.symbol)
        
        # Check if activation threshold met
        if position.unrealized_pnl_pips < self.config.trailing_activation_pips:
            return
        
        # Calculate new stop
        trail_distance = spec.pips_to_price(self.config.trailing_stop_pips)
        
        if position.is_long:
            new_stop = position.current_price - trail_distance
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                logger.debug(f"Trailing stop updated for {position.position_id}: {new_stop:.5f}")
        else:
            new_stop = position.current_price + trail_distance
            if new_stop < position.stop_loss:
                position.stop_loss = new_stop
                logger.debug(f"Trailing stop updated for {position.position_id}: {new_stop:.5f}")
    
    def apply_breakeven_stop(self,
                              position: OpenPosition,
                              spec: InstrumentSpec = None):
        """
        Move stop to breakeven if configured and conditions met.
        
        Args:
            position: Position to update
            spec: Instrument specification
        """
        if not self.config.use_breakeven_stop:
            return
        
        if spec is None:
            spec = get_instrument_spec(position.symbol)
        
        # Check if trigger threshold met
        if position.unrealized_pnl_pips < self.config.breakeven_trigger_pips:
            return
        
        # Calculate breakeven stop
        offset = spec.pips_to_price(self.config.breakeven_offset_pips)
        
        if position.is_long:
            be_stop = position.entry_price + offset
            if be_stop > position.stop_loss:
                position.stop_loss = be_stop
                logger.debug(f"Breakeven stop set for {position.position_id}: {be_stop:.5f}")
        else:
            be_stop = position.entry_price - offset
            if be_stop < position.stop_loss:
                position.stop_loss = be_stop
                logger.debug(f"Breakeven stop set for {position.position_id}: {be_stop:.5f}")
    
    def get_position(self, position_id: str) -> Optional[OpenPosition]:
        """Get position by ID."""
        return self.positions.get(position_id)
    
    def get_position_by_symbol_magic(self, symbol: str, magic: int) -> Optional[OpenPosition]:
        """Get position by symbol and magic number."""
        for pos in self.positions.values():
            if pos.symbol == symbol and pos.magic == magic:
                return pos
        return None
    
    def get_all_positions(self) -> List[OpenPosition]:
        """Get all open positions."""
        return list(self.positions.values())
    
    def get_positions_by_symbol(self, symbol: str) -> List[OpenPosition]:
        """Get all positions for a symbol."""
        return [p for p in self.positions.values() if p.symbol == symbol]
    
    def has_position(self, symbol: str, magic: int) -> bool:
        """Check if position exists for symbol and magic."""
        return self.get_position_by_symbol_magic(symbol, magic) is not None
    
    def count_positions(self, symbol: str = None) -> int:
        """Count open positions, optionally filtered by symbol."""
        if symbol:
            return len(self.get_positions_by_symbol(symbol))
        return len(self.positions)
    
    def get_closed_positions(self) -> List[OpenPosition]:
        """Get all closed positions."""
        return self.closed_positions.copy()
    
    def get_total_unrealized_pnl(self) -> Tuple[float, float]:
        """Get total unrealized P&L across all positions."""
        total_pips = sum(p.unrealized_pnl_pips for p in self.positions.values())
        total_dollars = sum(p.unrealized_pnl_dollars for p in self.positions.values())
        return total_pips, total_dollars
    
    def get_total_realized_pnl(self) -> Tuple[float, float]:
        """Get total realized P&L from closed positions."""
        total_pips = sum(p.realized_pnl_pips for p in self.closed_positions)
        total_dollars = sum(p.realized_pnl_dollars for p in self.closed_positions)
        return total_pips, total_dollars
    
    def to_trades_list(self) -> List[Dict[str, Any]]:
        """
        Convert closed positions to trades list format.
        
        Matches backtester output format.
        """
        trades = []
        for pos in self.closed_positions:
            trades.append({
                'entry_bar': pos.entry_bar,
                'exit_bar': pos.exit_bar,
                'entry_time': pos.entry_time,
                'exit_time': pos.exit_time,
                'direction': 'LONG' if pos.is_long else 'SHORT',
                'entry_price': pos.entry_price,
                'exit_price': pos.exit_price,
                'stop_loss': pos.initial_stop_loss,
                'take_profit': pos.initial_take_profit,
                'position_size': pos.volume,
                'pnl_pips': pos.realized_pnl_pips,
                'pnl_dollars': pos.realized_pnl_dollars,
                'exit_reason': pos.exit_reason,
            })
        return trades
    
    def reset(self):
        """Reset all positions."""
        self.positions.clear()
        self.closed_positions.clear()
        self._next_id = 1


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PositionConfig',
    'OpenPosition',
    'PositionSizeResult',
    'ExitCheckResult',
    'PositionCalculator',
    'PositionManager',
]


if __name__ == "__main__":
    # Basic tests
    logging.basicConfig(level=logging.DEBUG)
    
    config = PositionConfig(risk_per_trade_pct=1.0)
    manager = PositionManager(config)
    
    # Test position sizing
    spec = get_instrument_spec('EURUSD')
    size_result = manager.calculator.calculate_position_size(
        equity=10000,
        sl_pips=20,
        spec=spec
    )
    print(f"Position size result: {size_result}")
    
    # Test opening a position
    pos = manager.open_position(
        symbol='EURUSD',
        magic=12345,
        direction=1,
        entry_price=1.1000,
        volume=0.1,
        stop_loss=1.0980,
        take_profit=1.1030,
        entry_bar=0
    )
    print(f"\nOpened position: {pos}")
    
    # Test exit check
    exit_result = manager.check_exit_conditions(
        position=pos,
        high_price=1.1035,
        low_price=1.0995,
        current_bar=5,
        spec=spec
    )
    print(f"\nExit check: {exit_result}")
    
    # Test closing
    if exit_result.should_exit:
        manager.close_position(
            position=pos,
            exit_price=exit_result.exit_price,
            exit_reason=exit_result.exit_reason,
            exit_bar=5,
            spec=spec
        )
        print(f"\nClosed position: PnL = {pos.realized_pnl_pips} pips")
    
    print("\npm_position.py loaded successfully!")
