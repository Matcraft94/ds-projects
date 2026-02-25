"""Real-time intensity forecasting for Hawkes processes.

Provides short-term predictions of trading intensity for
order flow modeling and market making applications.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class ForecastResult:
    """Result from intensity forecast."""
    timestamp: float
    baseline_intensity: np.ndarray
    excited_intensity: np.ndarray
    total_intensity: np.ndarray
    expected_events: np.ndarray
    confidence_interval: Optional[Tuple[np.ndarray, np.ndarray]] = None


class IntensityForecaster:
    """Real-time intensity forecaster for Hawkes processes.
    
    Forecasts future trading intensity using the Hawkes conditional
    intensity function. Useful for:
    - Market making spread adjustment
    - Order arrival prediction
    - Liquidity forecasting
    
    The conditional intensity for dimension i at time t is:
        λ_i(t) = μ_i + Σ_j Σ_{t_j < t} α_ij * exp(-β_ij * (t - t_j))
    
    For forecasting horizon h:
        E[N_i(t + h) - N_i(t)] = ∫_t^{t+h} λ_i(s) ds
    """
    
    def __init__(
        self,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        history_window: float = 60.0,
        forecast_horizon: float = 5.0
    ):
        """Initialize forecaster.
        
        Args:
            mu: Baseline intensities (n_dims,)
            alpha: Excitation matrix (n_dims, n_dims)
            beta: Decay matrix (n_dims, n_dims)
            history_window: Seconds of history to maintain
            forecast_horizon: Default forecast horizon in seconds
        """
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.n_dims = len(mu)
        self.history_window = history_window
        self.forecast_horizon = forecast_horizon
        
        # Event history (timestamp, dimension)
        self._history: deque = deque()
        self._current_time: float = 0.0
    
    def update(self, timestamp: float, dimension: int):
        """Update with new event.
        
        Args:
            timestamp: Event timestamp
            dimension: Event dimension (0 to n_dims-1)
        """
        self._current_time = max(self._current_time, timestamp)
        self._history.append((timestamp, dimension))
        
        # Remove old events outside window
        cutoff = self._current_time - self.history_window
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
    
    def predict_intensity(
        self,
        target_time: float,
        return_components: bool = False
    ) -> np.ndarray:
        """Predict intensity at target time.
        
        Args:
            target_time: Time to predict intensity for
            return_components: If True, return baseline and excited separately
            
        Returns:
            Intensity vector (n_dims,) or tuple of (baseline, excited, total)
        """
        dt = target_time - self._current_time
        
        # Baseline intensity (constant)
        baseline = self.mu.copy()
        
        # Excited intensity from past events
        excited = np.zeros(self.n_dims)
        
        for event_time, event_dim in self._history:
            time_since = target_time - event_time
            if time_since > 0:
                # Excitation decays exponentially
                for i in range(self.n_dims):
                    excited[i] += (
                        self.alpha[i, event_dim] * 
                        np.exp(-self.beta[i, event_dim] * time_since)
                    )
        
        total = baseline + excited
        
        if return_components:
            return baseline, excited, total
        return total
    
    def forecast(
        self,
        horizon: Optional[float] = None,
        n_steps: int = 10
    ) -> ForecastResult:
        """Generate intensity forecast over horizon.
        
        Args:
            horizon: Forecast horizon in seconds (default: self.forecast_horizon)
            n_steps: Number of forecast steps
            
        Returns:
            ForecastResult with predictions
        """
        if horizon is None:
            horizon = self.forecast_horizon
        
        times = np.linspace(
            self._current_time,
            self._current_time + horizon,
            n_steps
        )
        
        intensities = np.zeros((self.n_dims, n_steps))
        baselines = np.zeros((self.n_dims, n_steps))
        excited = np.zeros((self.n_dims, n_steps))
        
        for i, t in enumerate(times):
            b, e, tot = self.predict_intensity(t, return_components=True)
            baselines[:, i] = b
            excited[:, i] = e
            intensities[:, i] = tot
        
        # Expected number of events = integral of intensity
        dt = times[1] - times[0]
        expected_events = np.trapezoid(intensities, dx=dt, axis=1)
        
        # Simple confidence interval (Poisson assumption)
        ci_lower = np.maximum(0, expected_events - 2 * np.sqrt(expected_events))
        ci_upper = expected_events + 2 * np.sqrt(expected_events)
        
        return ForecastResult(
            timestamp=self._current_time,
            baseline_intensity=baselines[:, -1],
            excited_intensity=excited[:, -1],
            total_intensity=intensities[:, -1],
            expected_events=expected_events,
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def predict_arrival_probability(
        self,
        dimension: int,
        time_window: float,
        n_events: int = 1
    ) -> float:
        """Predict probability of n events in time window.
        
        Uses Poisson approximation:
            P(N(t, t+Δt) = k) = (λΔt)^k * exp(-λΔt) / k!
        
        Args:
            dimension: Dimension to predict for
            time_window: Time window in seconds
            n_events: Number of events (default 1)
            
        Returns:
            Probability of observing n_events
        """
        # Average intensity over window
        t_mid = self._current_time + time_window / 2
        lambda_avg = self.predict_intensity(t_mid)[dimension]
        
        # Poisson probability
        lambda_t = lambda_avg * time_window
        from scipy.stats import poisson
        return poisson.pmf(n_events, lambda_t)
    
    def get_liquidity_forecast(self) -> dict:
        """Generate liquidity forecasting metrics.
        
        Returns:
            Dictionary with liquidity metrics
        """
        forecast = self.forecast(horizon=10.0, n_steps=20)
        
        # Buy/Sell intensity (assuming dims 0,2 = buy; 1,3 = sell)
        buy_intensity = forecast.total_intensity[0] + forecast.total_intensity[2]
        sell_intensity = forecast.total_intensity[1] + forecast.total_intensity[3]
        
        # Imbalance
        total = buy_intensity + sell_intensity
        imbalance = (buy_intensity - sell_intensity) / total if total > 0 else 0
        
        # Expected order flow
        expected_buy = forecast.expected_events[0] + forecast.expected_events[2]
        expected_sell = forecast.expected_events[1] + forecast.expected_events[3]
        
        return {
            'timestamp': forecast.timestamp,
            'buy_intensity': buy_intensity,
            'sell_intensity': sell_intensity,
            'total_intensity': total,
            'imbalance': imbalance,
            'expected_buy_orders': expected_buy,
            'expected_sell_orders': expected_sell,
            'confidence_interval': forecast.confidence_interval
        }
    
    def reset(self):
        """Clear history."""
        self._history.clear()
        self._current_time = 0.0
