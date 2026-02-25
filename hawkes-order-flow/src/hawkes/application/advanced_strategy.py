"""Advanced trading strategy with proper signal filtering and risk controls.

Implements a production-ready strategy based on:
1. Intensity momentum (rate of change)
2. Activity burst detection (spike in event rate)
3. Mean reversion of extreme intensities
4. Proper transaction cost modeling
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from collections import deque
from enum import Enum


class SignalQuality(Enum):
    """Signal quality rating."""
    STRONG = 3
    MODERATE = 2
    WEAK = 1
    INVALID = 0


@dataclass
class StrategyConfig:
    """Strategy configuration with optimized parameters."""
    # Entry thresholds
    intensity_momentum_threshold: float = 0.15  # 15% change in intensity
    burst_threshold: float = 2.0  # Intensity > 2x baseline
    mean_reversion_threshold: float = 3.0  # Z-score > 3
    
    # Risk management
    stop_loss_pct: float = 0.002  # 20 bps
    take_profit_pct: float = 0.006  # 60 bps
    max_holding_seconds: float = 30.0
    
    # Transaction costs
    commission_per_trade: float = 0.10  # $0.10 per trade
    spread_cost: float = 0.0001  # 1 bp
    
    # Position sizing
    base_position_size: int = 1
    max_position: int = 3
    
    # Filters
    min_signal_confidence: float = 0.6
    cooldown_seconds: float = 5.0


class AdvancedHawkesStrategy:
    """Advanced Hawkes-based trading strategy with multiple factors.
    
    Signal generation logic:
    
    1. INTENSITY MOMENTUM (Primary factor)
       - Buy when λ increases rapidly (momentum > threshold)
       - Sell when λ decreases rapidly
       
    2. ACTIVITY BURST (Secondary factor)
       - Buy when λ > burst_threshold * μ (unusual activity)
       - Indicates informed trading
       
    3. MEAN REVERSION (Exit factor)
       - Close long when λ > mean + 3*std (overbought)
       - Close short when λ < mean - 3*std (oversold)
    
    Risk Management:
    - Stop-loss at 20 bps
    - Take-profit at 60 bps (3:1 reward/risk)
    - Maximum holding time 30 seconds
    - Position sizing based on signal quality
    """
    
    def __init__(self, forecaster, config: Optional[StrategyConfig] = None):
        """Initialize strategy.
        
        Args:
            forecaster: IntensityForecaster instance
            config: Strategy configuration
        """
        self.forecaster = forecaster
        self.config = config or StrategyConfig()
        
        # State tracking
        self._intensity_history: deque = deque(maxlen=50)
        self._signal_history: deque = deque(maxlen=100)
        self._last_trade_time: float = -np.inf
        self._current_position: int = 0
        self._entry_price: Optional[float] = None
        self._entry_time: Optional[float] = None
        
        # Performance tracking
        self._total_commission: float = 0.0
        self._total_slippage: float = 0.0
    
    def generate_signal(
        self,
        timestamp: float,
        current_price: float,
        mid_price: float
    ) -> Dict:
        """Generate trading signal with quality rating.
        
        Args:
            timestamp: Current time
            current_price: Current market price
            mid_price: Mid price for spread calculation
            
        Returns:
            Dictionary with signal details
        """
        # Update forecaster
        self.forecaster._current_time = timestamp
        
        # Get current intensity
        forecast = self.forecaster.forecast(horizon=1.0, n_steps=5)
        current_intensity = np.mean(forecast.total_intensity)
        baseline = np.sum(self.forecaster.mu)
        
        # Calculate metrics
        metrics = self._calculate_metrics(current_intensity, baseline)
        
        # Default: no signal
        signal = {
            'timestamp': timestamp,
            'action': 'HOLD',
            'direction': 0,
            'quality': SignalQuality.INVALID,
            'confidence': 0.0,
            'size': 0,
            'metrics': metrics,
            'expected_cost': 0.0,
            'expected_profit': 0.0
        }
        
        # Check cooldown
        if timestamp - self._last_trade_time < self.config.cooldown_seconds:
            return signal
        
        # Check if we need to exit current position
        if self._current_position != 0 and self._entry_price is not None:
            exit_signal = self._check_exit_conditions(
                timestamp, current_price, current_intensity, baseline
            )
            if exit_signal:
                return exit_signal
        
        # Generate entry signal
        entry_signal = self._generate_entry_signal(
            timestamp, current_price, current_intensity, baseline, metrics
        )
        
        return entry_signal if entry_signal else signal
    
    def _calculate_metrics(self, intensity: float, baseline: float) -> Dict:
        """Calculate strategy metrics."""
        # Store history
        self._intensity_history.append({
            'timestamp': self.forecaster._current_time,
            'intensity': intensity,
            'baseline': baseline
        })
        
        metrics = {
            'current_intensity': intensity,
            'baseline': baseline,
            'ratio': intensity / baseline if baseline > 0 else 1.0,
            'momentum': 0.0,
            'z_score': 0.0
        }
        
        # Calculate momentum
        if len(self._intensity_history) >= 10:
            recent = [h['intensity'] for h in list(self._intensity_history)[-10:]]
            if len(recent) >= 2 and recent[0] > 0:
                metrics['momentum'] = (recent[-1] - recent[0]) / recent[0]
        
        # Calculate z-score
        if len(self._intensity_history) >= 20:
            intensities = [h['intensity'] for h in self._intensity_history]
            mean_int = np.mean(intensities)
            std_int = np.std(intensities)
            if std_int > 0:
                metrics['z_score'] = (intensity - mean_int) / std_int
        
        return metrics
    
    def _generate_entry_signal(
        self,
        timestamp: float,
        price: float,
        intensity: float,
        baseline: float,
        metrics: Dict
    ) -> Optional[Dict]:
        """Generate entry signal based on multiple factors."""
        
        # Factor 1: Intensity Momentum
        momentum_score = 0.0
        if metrics['momentum'] > self.config.intensity_momentum_threshold:
            momentum_score = min(1.0, metrics['momentum'] / 0.3)
        elif metrics['momentum'] < -self.config.intensity_momentum_threshold:
            momentum_score = -min(1.0, abs(metrics['momentum']) / 0.3)
        
        # Factor 2: Activity Burst
        burst_score = 0.0
        if metrics['ratio'] > self.config.burst_threshold:
            burst_score = min(1.0, (metrics['ratio'] - 1) / 2)
        elif metrics['ratio'] < 1 / self.config.burst_threshold:
            burst_score = -min(1.0, (1 / metrics['ratio'] - 1) / 2)
        
        # Combined score
        combined_score = 0.6 * momentum_score + 0.4 * burst_score
        
        # Determine direction and quality
        if combined_score > self.config.min_signal_confidence:
            direction = 1  # Buy
            quality = self._rate_quality(combined_score, metrics)
        elif combined_score < -self.config.min_signal_confidence:
            direction = -1  # Sell
            quality = self._rate_quality(abs(combined_score), metrics)
        else:
            return None
        
        # Check position limits
        new_position = self._current_position + direction * self.config.base_position_size
        if abs(new_position) > self.config.max_position:
            return None
        
        # Calculate costs
        position_size = self._calculate_position_size(quality)
        expected_cost = self._estimate_transaction_cost(price, position_size)
        expected_profit = self._estimate_expected_profit(direction, metrics)
        
        # Only trade if expected profit > 2x costs
        if expected_profit < 2 * expected_cost:
            return None
        
        action = 'BUY' if direction > 0 else 'SELL'
        
        return {
            'timestamp': timestamp,
            'action': action,
            'direction': direction,
            'quality': quality,
            'confidence': abs(combined_score),
            'size': position_size,
            'metrics': metrics,
            'expected_cost': expected_cost,
            'expected_profit': expected_profit,
            'entry_price': price
        }
    
    def _check_exit_conditions(
        self,
        timestamp: float,
        price: float,
        intensity: float,
        baseline: float
    ) -> Optional[Dict]:
        """Check if we should exit current position."""
        if self._entry_price is None or self._entry_time is None:
            return None
        
        holding_time = timestamp - self._entry_time
        pnl_pct = (price - self._entry_price) / self._entry_price * self._current_position
        
        # Stop loss
        if pnl_pct < -self.config.stop_loss_pct:
            return {
                'timestamp': timestamp,
                'action': 'STOP_LOSS',
                'direction': 0,
                'quality': SignalQuality.STRONG,
                'confidence': 1.0,
                'size': abs(self._current_position),
                'metrics': {'pnl_pct': pnl_pct, 'holding_time': holding_time},
                'expected_cost': self._estimate_transaction_cost(price, abs(self._current_position)),
                'expected_profit': 0.0
            }
        
        # Take profit
        if pnl_pct > self.config.take_profit_pct:
            return {
                'timestamp': timestamp,
                'action': 'TAKE_PROFIT',
                'direction': 0,
                'quality': SignalQuality.STRONG,
                'confidence': 1.0,
                'size': abs(self._current_position),
                'metrics': {'pnl_pct': pnl_pct, 'holding_time': holding_time},
                'expected_cost': self._estimate_transaction_cost(price, abs(self._current_position)),
                'expected_profit': pnl_pct * price * abs(self._current_position)
            }
        
        # Time-based exit
        if holding_time > self.config.max_holding_seconds:
            return {
                'timestamp': timestamp,
                'action': 'TIME_EXIT',
                'direction': 0,
                'quality': SignalQuality.MODERATE,
                'confidence': 0.5,
                'size': abs(self._current_position),
                'metrics': {'pnl_pct': pnl_pct, 'holding_time': holding_time},
                'expected_cost': self._estimate_transaction_cost(price, abs(self._current_position)),
                'expected_profit': pnl_pct * price * abs(self._current_position)
            }
        
        # Mean reversion exit
        metrics = self._calculate_metrics(intensity, baseline)
        if abs(metrics.get('z_score', 0)) > self.config.mean_reversion_threshold:
            # Exit if intensity is extreme
            if (self._current_position > 0 and metrics['z_score'] > 0) or \
               (self._current_position < 0 and metrics['z_score'] < 0):
                return {
                    'timestamp': timestamp,
                    'action': 'MEAN_REVERSION',
                    'direction': 0,
                    'quality': SignalQuality.MODERATE,
                    'confidence': 0.6,
                    'size': abs(self._current_position),
                    'metrics': metrics,
                    'expected_cost': self._estimate_transaction_cost(price, abs(self._current_position)),
                    'expected_profit': pnl_pct * price * abs(self._current_position)
                }
        
        return None
    
    def _rate_quality(self, score: float, metrics: Dict) -> SignalQuality:
        """Rate signal quality based on score and metrics."""
        if score > 0.85 and abs(metrics.get('momentum', 0)) > 0.2:
            return SignalQuality.STRONG
        elif score > 0.7:
            return SignalQuality.MODERATE
        elif score > 0.6:
            return SignalQuality.WEAK
        else:
            return SignalQuality.INVALID
    
    def _calculate_position_size(self, quality: SignalQuality) -> int:
        """Calculate position size based on signal quality."""
        size_map = {
            SignalQuality.STRONG: self.config.base_position_size * 2,
            SignalQuality.MODERATE: self.config.base_position_size,
            SignalQuality.WEAK: max(1, self.config.base_position_size // 2),
            SignalQuality.INVALID: 0
        }
        return size_map.get(quality, 0)
    
    def _estimate_transaction_cost(self, price: float, size: int) -> float:
        """Estimate total transaction cost."""
        commission = self.config.commission_per_trade * 2  # Entry + exit
        spread_cost = price * self.config.spread_cost * size
        return commission + spread_cost
    
    def _estimate_expected_profit(self, direction: int, metrics: Dict) -> float:
        """Estimate expected profit based on signal strength."""
        # Simple model: stronger signals have higher expected returns
        momentum = abs(metrics.get('momentum', 0))
        return momentum * self.config.take_profit_pct * 100  # Scaled
    
    def execute_signal(self, signal: Dict, current_price: float) -> Dict:
        """Execute signal and update state."""
        action = signal['action']
        direction = signal['direction']
        size = signal['size']
        
        result = {
            'executed': False,
            'action': action,
            'direction': direction,
            'size': size,
            'price': current_price,
            'cost': 0.0,
            'pnl': 0.0
        }
        
        if action in ['BUY', 'SELL']:
            # Entry
            self._current_position = direction * size
            self._entry_price = current_price
            self._entry_time = signal['timestamp']
            self._last_trade_time = signal['timestamp']
            
            # Track costs
            cost = self._estimate_transaction_cost(current_price, size)
            self._total_commission += self.config.commission_per_trade
            self._total_slippage += current_price * self.config.spread_cost * size
            
            result['executed'] = True
            result['cost'] = cost
            
        elif action in ['STOP_LOSS', 'TAKE_PROFIT', 'TIME_EXIT', 'MEAN_REVERSION']:
            # Exit
            if self._entry_price is not None:
                pnl = (current_price - self._entry_price) * self._current_position
                result['pnl'] = pnl
                result['cost'] = self._estimate_transaction_cost(current_price, abs(self._current_position))
            
            self._current_position = 0
            self._entry_price = None
            self._entry_time = None
            self._last_trade_time = signal['timestamp']
            result['executed'] = True
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get strategy statistics."""
        return {
            'total_commission': self._total_commission,
            'total_slippage': self._total_slippage,
            'current_position': self._current_position,
            'signals_generated': len(self._signal_history)
        }
