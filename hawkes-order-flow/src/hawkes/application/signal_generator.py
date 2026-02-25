"""Trading signal generator based on Hawkes intensity.

Generates actionable trading signals from Hawkes process predictions
for high-frequency trading strategies.
"""

import numpy as np
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque


class SignalType(Enum):
    """Types of trading signals."""
    NO_SIGNAL = 0
    BUY = 1
    SELL = 2
    BUY_AGGRESSIVE = 3
    SELL_AGGRESSIVE = 4
    HOLD = 5


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    timestamp: float
    signal_type: SignalType
    confidence: float
    intensity_buy: float
    intensity_sell: float
    imbalance: float
    expected_return: Optional[float] = None
    risk_score: Optional[float] = None
    metadata: Optional[dict] = None


class SignalGenerator:
    """Generate trading signals from Hawkes intensity predictions.
    
    Signal generation strategies:
    
    1. **Intensity Imbalance**: Buy when buy_intensity >> sell_intensity
    2. **Momentum**: Buy when intensity is increasing (self-excitation)
    3. **Mean Reversion**: Sell when intensity is too high (overbought)
    4. **Combination**: Weighted combination of multiple factors
    
    Risk management:
    - Maximum position limits
    - Signal confidence thresholds
    - Cooldown periods between trades
    """
    
    def __init__(
        self,
        forecaster,
        buy_threshold: float = 0.6,
        sell_threshold: float = -0.6,
        confidence_threshold: float = 0.7,
        cooldown_seconds: float = 5.0,
        max_position: int = 10
    ):
        """Initialize signal generator.
        
        Args:
            forecaster: IntensityForecaster instance
            buy_threshold: Imbalance threshold for buy signal (0-1)
            sell_threshold: Imbalance threshold for sell signal (-1 to 0)
            confidence_threshold: Minimum confidence for signal (0-1)
            cooldown_seconds: Minimum time between signals
            max_position: Maximum position size
        """
        self.forecaster = forecaster
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.confidence_threshold = confidence_threshold
        self.cooldown = cooldown_seconds
        self.max_position = max_position
        
        # State
        self._last_signal_time: float = -np.inf
        self._current_position: int = 0
        self._signal_history: deque = deque(maxlen=1000)
        self._intensity_history: deque = deque(maxlen=100)
    
    def generate_signal(
        self,
        timestamp: float,
        current_price: Optional[float] = None,
        strategy: str = 'imbalance'
    ) -> TradingSignal:
        """Generate trading signal.
        
        Args:
            timestamp: Current timestamp
            current_price: Current market price (optional)
            strategy: Signal strategy ('imbalance', 'momentum', 'mean_reversion', 'combined')
            
        Returns:
            TradingSignal
        """
        # Update forecaster time
        self.forecaster._current_time = timestamp
        
        # Get intensity forecast
        forecast = self.forecaster.forecast(horizon=5.0, n_steps=10)
        
        # Calculate metrics
        buy_intensity = forecast.total_intensity[0] + forecast.total_intensity[2]
        sell_intensity = forecast.total_intensity[1] + forecast.total_intensity[3]
        total = buy_intensity + sell_intensity
        
        imbalance = 0.0
        if total > 0:
            imbalance = (buy_intensity - sell_intensity) / total
        
        # Store history
        self._intensity_history.append({
            'timestamp': timestamp,
            'buy': buy_intensity,
            'sell': sell_intensity,
            'imbalance': imbalance
        })
        
        # Check cooldown
        if timestamp - self._last_signal_time < self.cooldown:
            signal_type = SignalType.HOLD
            confidence = 0.0
        else:
            # Generate signal based on strategy
            signal_type, confidence = self._apply_strategy(
                strategy, imbalance, buy_intensity, sell_intensity, timestamp
            )
        
        # Apply position limits
        signal_type = self._apply_position_limits(signal_type)
        
        # Calculate expected return (simplified model)
        expected_return = self._estimate_return(signal_type, imbalance)
        
        # Calculate risk score
        risk_score = self._calculate_risk(signal_type, forecast)
        
        signal = TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            confidence=confidence,
            intensity_buy=buy_intensity,
            intensity_sell=sell_intensity,
            imbalance=imbalance,
            expected_return=expected_return,
            risk_score=risk_score,
            metadata={
                'strategy': strategy,
                'position': self._current_position,
                'forecast_horizon': 5.0
            }
        )
        
        # Update state if signal generated
        if signal_type not in [SignalType.NO_SIGNAL, SignalType.HOLD]:
            self._last_signal_time = timestamp
            self._update_position(signal_type)
            self._signal_history.append(signal)
        
        return signal
    
    def _apply_strategy(
        self,
        strategy: str,
        imbalance: float,
        buy_intensity: float,
        sell_intensity: float,
        timestamp: float
    ) -> Tuple[SignalType, float]:
        """Apply signal generation strategy."""
        
        if strategy == 'imbalance':
            return self._imbalance_strategy(imbalance)
        
        elif strategy == 'momentum':
            return self._momentum_strategy(timestamp)
        
        elif strategy == 'mean_reversion':
            return self._mean_reversion_strategy(buy_intensity, sell_intensity)
        
        elif strategy == 'combined':
            return self._combined_strategy(imbalance, buy_intensity, sell_intensity, timestamp)
        
        else:
            return SignalType.NO_SIGNAL, 0.0
    
    def _imbalance_strategy(self, imbalance: float) -> Tuple[SignalType, float]:
        """Signal based on order flow imbalance."""
        if imbalance > self.buy_threshold:
            confidence = min(1.0, imbalance / 0.8)
            if imbalance > 0.8:
                return SignalType.BUY_AGGRESSIVE, confidence
            return SignalType.BUY, confidence
        
        elif imbalance < self.sell_threshold:
            confidence = min(1.0, abs(imbalance) / 0.8)
            if imbalance < -0.8:
                return SignalType.SELL_AGGRESSIVE, confidence
            return SignalType.SELL, confidence
        
        return SignalType.NO_SIGNAL, 0.0
    
    def _momentum_strategy(self, timestamp: float) -> Tuple[SignalType, float]:
        """Signal based on intensity momentum."""
        if len(self._intensity_history) < 10:
            return SignalType.NO_SIGNAL, 0.0
        
        # Calculate intensity trend
        recent = list(self._intensity_history)[-10:]
        buy_trend = np.polyfit([r['timestamp'] for r in recent], 
                               [r['buy'] for r in recent], 1)[0]
        sell_trend = np.polyfit([r['timestamp'] for r in recent],
                                [r['sell'] for r in recent], 1)[0]
        
        if buy_trend > 0.1 and buy_trend > sell_trend:
            confidence = min(1.0, buy_trend / 0.5)
            return SignalType.BUY, confidence
        elif sell_trend > 0.1 and sell_trend > buy_trend:
            confidence = min(1.0, sell_trend / 0.5)
            return SignalType.SELL, confidence
        
        return SignalType.NO_SIGNAL, 0.0
    
    def _mean_reversion_strategy(
        self,
        buy_intensity: float,
        sell_intensity: float
    ) -> Tuple[SignalType, float]:
        """Signal based on mean reversion of intensity."""
        if len(self._intensity_history) < 20:
            return SignalType.NO_SIGNAL, 0.0
        
        # Calculate historical average
        hist = list(self._intensity_history)
        avg_buy = np.mean([h['buy'] for h in hist])
        avg_sell = np.mean([h['sell'] for h in hist])
        
        std_buy = np.std([h['buy'] for h in hist])
        std_sell = np.std([h['sell'] for h in hist])
        
        # Z-scores
        z_buy = (buy_intensity - avg_buy) / std_buy if std_buy > 0 else 0
        z_sell = (sell_intensity - avg_sell) / std_sell if std_sell > 0 else 0
        
        if z_buy > 2.0:  # Overbought
            return SignalType.SELL, min(1.0, z_buy / 3.0)
        elif z_sell > 2.0:  # Oversold
            return SignalType.BUY, min(1.0, z_sell / 3.0)
        
        return SignalType.NO_SIGNAL, 0.0
    
    def _combined_strategy(
        self,
        imbalance: float,
        buy_intensity: float,
        sell_intensity: float,
        timestamp: float
    ) -> Tuple[SignalType, float]:
        """Combined strategy with weighted signals."""
        # Get individual signals
        imb_signal, imb_conf = self._imbalance_strategy(imbalance)
        mom_signal, mom_conf = self._momentum_strategy(timestamp)
        mr_signal, mr_conf = self._mean_reversion_strategy(buy_intensity, sell_intensity)
        
        # Weight by strategy reliability (imbalance most reliable)
        weights = {'imbalance': 0.5, 'momentum': 0.3, 'mean_reversion': 0.2}
        
        # Aggregate buy/sell scores
        buy_score = 0.0
        sell_score = 0.0
        
        for signal, conf, weight in [
            (imb_signal, imb_conf, weights['imbalance']),
            (mom_signal, mom_conf, weights['momentum']),
            (mr_signal, mr_conf, weights['mean_reversion'])
        ]:
            if 'BUY' in signal.name:
                buy_score += conf * weight
            elif 'SELL' in signal.name:
                sell_score += conf * weight
        
        # Generate final signal
        if buy_score > sell_score and buy_score > self.confidence_threshold:
            if buy_score > 0.8:
                return SignalType.BUY_AGGRESSIVE, buy_score
            return SignalType.BUY, buy_score
        elif sell_score > buy_score and sell_score > self.confidence_threshold:
            if sell_score > 0.8:
                return SignalType.SELL_AGGRESSIVE, sell_score
            return SignalType.SELL, sell_score
        
        return SignalType.NO_SIGNAL, max(buy_score, sell_score)
    
    def _apply_position_limits(self, signal: SignalType) -> SignalType:
        """Apply position limits to signal."""
        if 'BUY' in signal.name and self._current_position >= self.max_position:
            return SignalType.HOLD
        if 'SELL' in signal.name and self._current_position <= -self.max_position:
            return SignalType.HOLD
        return signal
    
    def _update_position(self, signal: SignalType):
        """Update current position based on signal."""
        if 'BUY' in signal.name:
            self._current_position += 1
        elif 'SELL' in signal.name:
            self._current_position -= 1
    
    def _estimate_return(
        self,
        signal: SignalType,
        imbalance: float
    ) -> Optional[float]:
        """Estimate expected return for signal."""
        if signal == SignalType.NO_SIGNAL or signal == SignalType.HOLD:
            return 0.0
        
        # Simple model: imbalance predicts short-term price movement
        # Higher imbalance -> higher expected return in that direction
        if 'BUY' in signal.name:
            return imbalance * 0.001  # 10 bps per unit imbalance
        elif 'SELL' in signal.name:
            return -imbalance * 0.001
        
        return 0.0
    
    def _calculate_risk(
        self,
        signal: SignalType,
        forecast
    ) -> float:
        """Calculate risk score for signal (0-1, higher = riskier)."""
        if signal == SignalType.NO_SIGNAL or signal == SignalType.HOLD:
            return 0.0
        
        # Risk factors
        risks = []
        
        # 1. Intensity uncertainty (coefficient of variation)
        total_intensity = np.sum(forecast.total_intensity)
        if total_intensity > 0:
            cv = np.std(forecast.total_intensity) / np.mean(forecast.total_intensity)
            risks.append(min(1.0, cv))
        
        # 2. Position risk
        position_risk = abs(self._current_position) / self.max_position
        risks.append(position_risk)
        
        # 3. Signal aggressiveness
        if 'AGGRESSIVE' in signal.name:
            risks.append(0.3)
        
        return np.mean(risks) if risks else 0.0
    
    def get_signal_statistics(self) -> dict:
        """Get statistics of generated signals."""
        if not self._signal_history:
            return {}
        
        signals = list(self._signal_history)
        
        buy_count = sum(1 for s in signals if 'BUY' in s.signal_type.name)
        sell_count = sum(1 for s in signals if 'SELL' in s.signal_type.name)
        
        return {
            'total_signals': len(signals),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'avg_confidence': np.mean([s.confidence for s in signals]),
            'avg_imbalance': np.mean([s.imbalance for s in signals]),
            'current_position': self._current_position
        }
    
    def reset(self):
        """Reset generator state."""
        self._last_signal_time = -np.inf
        self._current_position = 0
        self._signal_history.clear()
        self._intensity_history.clear()
