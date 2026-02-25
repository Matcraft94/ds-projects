"""Risk management for Hawkes-based trading strategies.

Provides real-time risk monitoring and position sizing based on
Hawkes process volatility estimates.
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from collections import deque


@dataclass
class RiskMetrics:
    """Risk metrics for trading strategy."""
    timestamp: float
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    position_size_limit: int
    intensity_volatility: float
    self_excitation_risk: float


class RiskManager:
    """Risk manager for Hawkes-based trading.
    
    Key risk management features:
    
    1. **Position Sizing**: Based on intensity volatility
    2. **VaR Calculation**: Using Hawkes process properties
    3. **Drawdown Control**: Dynamic position reduction
    4. **Self-Excitation Risk**: Monitor for explosive behavior
    
    Risk formulas:
    - VaR_α = μ_α * σ * √Δt (for Gaussian approximation)
    - Position size ∝ 1 / (1 + ρ) where ρ is spectral radius
    - Self-excitation risk = 1 - exp(-λ_burst * t_cooldown)
    """
    
    def __init__(
        self,
        forecaster,
        max_var_pct: float = 0.02,  # 2% VaR limit
        max_drawdown_pct: float = 0.10,  # 10% max drawdown
        target_sharpe: float = 1.5,
        risk_free_rate: float = 0.02 / 252 / 6.5 / 3600  # Per second
    ):
        """Initialize risk manager.
        
        Args:
            forecaster: IntensityForecaster instance
            max_var_pct: Maximum VaR as percentage of capital
            max_drawdown_pct: Maximum allowed drawdown
            target_sharpe: Target Sharpe ratio
            risk_free_rate: Risk-free rate per second
        """
        self.forecaster = forecaster
        self.max_var_pct = max_var_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.target_sharpe = target_sharpe
        self.risk_free_rate = risk_free_rate
        
        # State
        self._equity_history: deque = deque(maxlen=10000)
        self._trade_returns: deque = deque(maxlen=1000)
        self._peak_equity: float = 0.0
        self._current_equity: float = 1.0  # Normalized
        
        # Risk parameters
        self._volatility_estimate: float = 0.001  # Initial guess
        self._last_update: float = 0.0
    
    def update_equity(self, timestamp: float, equity: float):
        """Update equity curve.
        
        Args:
            timestamp: Current timestamp
            equity: Current equity value
        """
        self._current_equity = equity
        self._equity_history.append((timestamp, equity))
        
        # Update peak
        if equity > self._peak_equity:
            self._peak_equity = equity
        
        # Calculate returns if we have history
        if len(self._equity_history) > 1:
            prev_equity = list(self._equity_history)[-2][1]
            ret = (equity - prev_equity) / prev_equity
            self._trade_returns.append(ret)
            
            # Update volatility estimate
            if len(self._trade_returns) > 10:
                self._volatility_estimate = np.std(list(self._trade_returns))
        
        self._last_update = timestamp
    
    def calculate_risk_metrics(self, timestamp: float) -> RiskMetrics:
        """Calculate current risk metrics.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            RiskMetrics object
        """
        # VaR calculation
        var_95, var_99 = self._calculate_var()
        
        # Expected shortfall (CVaR)
        es = self._calculate_expected_shortfall()
        
        # Drawdown
        max_dd, current_dd = self._calculate_drawdown()
        
        # Performance ratios
        sharpe = self._calculate_sharpe()
        calmar = self._calculate_calmar()
        
        # Position limit based on risk
        position_limit = self._calculate_position_limit()
        
        # Intensity volatility
        intensity_vol = self._estimate_intensity_volatility()
        
        # Self-excitation risk
        se_risk = self._calculate_self_excitation_risk()
        
        return RiskMetrics(
            timestamp=timestamp,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=es,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            position_size_limit=position_limit,
            intensity_volatility=intensity_vol,
            self_excitation_risk=se_risk
        )
    
    def _calculate_var(self) -> tuple:
        """Calculate Value at Risk.
        
        Uses parametric VaR with volatility estimate.
        """
        if len(self._trade_returns) < 10:
            return 0.0, 0.0
        
        returns = np.array(list(self._trade_returns))
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Parametric VaR (normal assumption)
        var_95 = -(mu - 1.645 * sigma)
        var_99 = -(mu - 2.326 * sigma)
        
        return var_95, var_99
    
    def _calculate_expected_shortfall(self) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        if len(self._trade_returns) < 20:
            return 0.0
        
        returns = np.array(list(self._trade_returns))
        var_95 = np.percentile(returns, 5)
        
        # ES is mean of returns below VaR
        es = np.mean(returns[returns <= var_95])
        return -es
    
    def _calculate_drawdown(self) -> tuple:
        """Calculate max and current drawdown."""
        if not self._equity_history:
            return 0.0, 0.0
        
        equity = np.array([e[1] for e in self._equity_history])
        
        # Running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Drawdowns
        drawdowns = (running_max - equity) / running_max
        
        max_dd = np.max(drawdowns)
        current_dd = drawdowns[-1]
        
        return max_dd, current_dd
    
    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self._trade_returns) < 20:
            return 0.0
        
        returns = np.array(list(self._trade_returns))
        excess_returns = returns - self.risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        # Annualized Sharpe (assuming 6.5 hours trading day)
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252 * 6.5 * 3600)
        return sharpe
    
    def _calculate_calmar(self) -> float:
        """Calculate Calmar ratio."""
        if len(self._equity_history) < 2:
            return 0.0
        
        # Annualized return
        equity = np.array([e[1] for e in self._equity_history])
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Approximate time in years
        if len(self._equity_history) > 1:
            times = np.array([e[0] for e in self._equity_history])
            years = (times[-1] - times[0]) / (365.25 * 24 * 3600)
            if years > 0:
                annual_return = (1 + total_return) ** (1 / years) - 1
            else:
                annual_return = 0.0
        else:
            annual_return = 0.0
        
        max_dd, _ = self._calculate_drawdown()
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / max_dd
    
    def _calculate_position_limit(self) -> int:
        """Calculate dynamic position size limit.
        
        Based on:
        1. Current drawdown (reduce size during drawdown)
        2. Volatility (reduce size in high vol)
        3. Spectral radius (reduce if near critical)
        """
        base_limit = 10  # Base maximum
        
        # Drawdown reduction
        _, current_dd = self._calculate_drawdown()
        if current_dd > self.max_drawdown_pct * 0.5:
            # Reduce linearly as drawdown increases
            dd_factor = 1 - (current_dd / self.max_drawdown_pct)
            dd_factor = max(0.0, min(1.0, dd_factor))
        else:
            dd_factor = 1.0
        
        # Volatility scaling (Kelly-like)
        if self._volatility_estimate > 0:
            vol_factor = 0.02 / self._volatility_estimate  # Target 2% vol
            vol_factor = min(2.0, max(0.5, vol_factor))
        else:
            vol_factor = 1.0
        
        # Spectral radius factor (handle NaN from zero off-diagonal elements)
        B = self.forecaster.alpha / self.forecaster.beta
        B = np.nan_to_num(B, nan=0.0)  # Replace NaN with 0
        try:
            rho = np.max(np.abs(np.linalg.eigvals(B)))
        except:
            rho = 0.5  # Default if eigvals fails
        rho_factor = max(0.0, 1 - rho)  # Reduce as ρ approaches 1
        
        # Combined limit
        limit = int(base_limit * dd_factor * vol_factor * rho_factor)
        return max(1, limit)
    
    def _estimate_intensity_volatility(self) -> float:
        """Estimate volatility from intensity fluctuations."""
        forecast = self.forecaster.forecast(horizon=10.0, n_steps=20)
        intensities = forecast.total_intensity
        
        if len(intensities) > 1:
            return np.std(intensities) / np.mean(intensities) if np.mean(intensities) > 0 else 0.0
        return 0.0
    
    def _calculate_self_excitation_risk(self) -> float:
        """Calculate risk from self-excitation.
        
        Risk increases when:
        - Spectral radius is high
        - Recent event rate is elevated
        """
        B = self.forecaster.alpha / self.forecaster.beta
        B = np.nan_to_num(B, nan=0.0)  # Handle NaN from zero off-diagonal
        try:
            rho = np.max(np.abs(np.linalg.eigvals(B)))
        except:
            rho = 0.5
        
        # Base risk from spectral radius
        base_risk = rho / (1 - rho) if rho < 1 else 10.0
        
        # Recent activity factor
        forecast = self.forecaster.forecast(horizon=1.0, n_steps=5)
        recent_intensity = np.mean(forecast.total_intensity)
        
        # Normalize by baseline
        baseline = np.sum(self.forecaster.mu)
        if baseline > 0:
            activity_factor = recent_intensity / baseline
        else:
            activity_factor = 1.0
        
        risk = base_risk * activity_factor
        return min(1.0, risk / (1 + risk))  # Sigmoid-like compression
    
    def check_risk_limits(self, timestamp: float) -> dict:
        """Check if any risk limits are breached.
        
        Returns:
            Dictionary with limit status
        """
        metrics = self.calculate_risk_metrics(timestamp)
        
        checks = {
            'var_95_breached': metrics.var_95 > self.max_var_pct,
            'drawdown_breached': metrics.current_drawdown > self.max_drawdown_pct,
            'sharpe_too_low': metrics.sharpe_ratio < self.target_sharpe * 0.5,
            'self_excitation_high': metrics.self_excitation_risk > 0.7,
            'position_limit': metrics.position_size_limit
        }
        
        checks['trading_allowed'] = not any([
            checks['var_95_breached'],
            checks['drawdown_breached'],
            checks['self_excitation_high']
        ])
        
        return checks
    
    def get_risk_report(self, timestamp: float) -> str:
        """Generate formatted risk report."""
        metrics = self.calculate_risk_metrics(timestamp)
        checks = self.check_risk_limits(timestamp)
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    RISK MANAGEMENT REPORT                     ║
╠══════════════════════════════════════════════════════════════╣
║ Time: {timestamp:12.2f}                                           ║
╠══════════════════════════════════════════════════════════════╣
║ Risk Metrics:                                                ║
║   VaR 95%:        {metrics.var_95:8.4%}  {'⚠ BREACH' if checks['var_95_breached'] else '✓ OK'}          ║
║   VaR 99%:        {metrics.var_99:8.4%}                         ║
║   Exp Shortfall:  {metrics.expected_shortfall:8.4%}                         ║
║   Max Drawdown:   {metrics.max_drawdown:8.4%}                         ║
║   Current DD:     {metrics.current_drawdown:8.4%}  {'⚠ BREACH' if checks['drawdown_breached'] else '✓ OK'}          ║
╠══════════════════════════════════════════════════════════════╣
║ Performance:                                                 ║
║   Sharpe Ratio:   {metrics.sharpe_ratio:8.2f}  {'⚠ LOW' if checks['sharpe_too_low'] else '✓ OK'}           ║
║   Calmar Ratio:   {metrics.calmar_ratio:8.2f}                         ║
╠══════════════════════════════════════════════════════════════╣
║ Hawkes-Specific:                                             ║
║   Intensity Vol:  {metrics.intensity_volatility:8.4f}                         ║
║   Self-Exc Risk:  {metrics.self_excitation_risk:8.4f}  {'⚠ HIGH' if checks['self_excitation_high'] else '✓ OK'}           ║
║   Position Limit: {metrics.position_size_limit:8d}                          ║
╠══════════════════════════════════════════════════════════════╣
║ Status: {'🟢 TRADING ALLOWED' if checks['trading_allowed'] else '🔴 TRADING HALTED'}                           ║
╚══════════════════════════════════════════════════════════════╝
"""
        return report
