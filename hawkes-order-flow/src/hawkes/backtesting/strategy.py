"""Backtesting strategies based on Hawkes process intensity.

Implements trading strategies that use the predicted conditional intensity
as a signal for order flow prediction.
"""

import numpy as np
from typing import Literal, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from ..estimation.mle import MultivariateHawkesMLE
from ..estimation.em import MultivariateHawkesEM


class SignalType(Enum):
    """Types of signals that can be generated from Hawkes process."""
    INTENSITY = "intensity"           # Raw intensity levels
    INTENSITY_RATIO = "intensity_ratio"  # Ratio of buy/sell intensity
    PREDICTION = "prediction"         # Predicted next event type
    BRANCHING_EXPECTATION = "branching"  # Expected offspring


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: float
    entry_price: float
    direction: int  # +1 for long, -1 for short
    exit_time: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    
    def close(self, time: float, price: float):
        """Close the trade."""
        self.exit_time = time
        self.exit_price = price
        self.pnl = self.direction * (price - self.entry_price)


@dataclass  
class BacktestResult:
    """Results from backtesting."""
    trades: list[Trade]
    equity_curve: pd.DataFrame
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    
    def summary(self) -> str:
        """Generate text summary."""
        return f"""
Backtest Results
================
Total Trades:     {self.num_trades}
Win Rate:         {self.win_rate:.2%}
Total Return:     {self.total_return:.2%}
Sharpe Ratio:     {self.sharpe_ratio:.2f}
Max Drawdown:     {self.max_drawdown:.2%}
Profit Factor:    {self.profit_factor:.2f}
"""


class IntensityStrategy:
    """Trading strategy based on Hawkes process intensity.
    
    Generates trading signals from the predicted conditional intensity
    of the multivariate Hawkes process.
    
    Two approaches:
    1. 'intensity': Signal based on imbalance between buy/sell intensities
    2. 'prediction': Signal based on predicted most likely next event
    
    Parameters:
        estimator: Fitted Hawkes estimator (MLE or EM)
        entry_threshold: Minimum signal strength to enter position
        exit_threshold: Signal strength to exit position
        approach: 'intensity' or 'prediction'
        max_position_duration: Maximum time to hold position
        transaction_costs: Proportional transaction costs
    """
    
    def __init__(
        self,
        estimator: MultivariateHawkesMLE | MultivariateHawkesEM,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5,
        approach: Literal['intensity', 'prediction'] = 'intensity',
        max_position_duration: float = 60.0,  # seconds
        transaction_costs: float = 0.001  # 10 bps
    ):
        self.estimator = estimator
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.approach = approach
        self.max_position_duration = max_position_duration
        self.transaction_costs = transaction_costs
        
        # Event type mapping (assuming standard order)
        self.MB = 0  # Market Buy
        self.MS = 1  # Market Sell
        self.LB = 2  # Limit Buy
        self.LS = 3  # Limit Sell
    
    def generate_signal(
        self,
        intensities: np.ndarray,
        approach: Optional[str] = None
    ) -> tuple[int, float]:
        """Generate trading signal from intensities.
        
        Args:
            intensities: Current intensity vector, shape (n_dims,)
            approach: Override default approach
            
        Returns:
            (direction, strength) tuple
            direction: -1 (short), 0 (neutral), +1 (long)
            strength: Signal magnitude
        """
        approach = approach or self.approach
        
        if approach == 'intensity':
            return self._signal_from_intensity(intensities)
        elif approach == 'prediction':
            return self._signal_from_prediction(intensities)
        else:
            raise ValueError(f"Unknown approach: {approach}")
    
    def _signal_from_intensity(
        self,
        intensities: np.ndarray
    ) -> tuple[int, float]:
        """Generate signal from buy/sell intensity imbalance.
        
        Buy pressure: MB + LB (aggressive + passive buying)
        Sell pressure: MS + LS (aggressive + passive selling)
        """
        buy_intensity = intensities[self.MB] + intensities[self.LB]
        sell_intensity = intensities[self.MS] + intensities[self.LS]
        
        if buy_intensity + sell_intensity == 0:
            return 0, 0.0
        
        # Imbalance ratio
        imbalance = (buy_intensity - sell_intensity) / (buy_intensity + sell_intensity)
        
        if imbalance > self.entry_threshold / 10:  # Scale threshold
            return 1, abs(imbalance)
        elif imbalance < -self.entry_threshold / 10:
            return -1, abs(imbalance)
        else:
            return 0, 0.0
    
    def _signal_from_prediction(
        self,
        intensities: np.ndarray
    ) -> tuple[int, float]:
        """Generate signal from predicted next event type.
        
        Predict the most likely next event and trade accordingly.
        """
        # Most likely event is the one with highest intensity
        predicted_event = np.argmax(intensities)
        total_intensity = np.sum(intensities)
        
        if total_intensity == 0:
            return 0, 0.0
        
        confidence = intensities[predicted_event] / total_intensity
        
        if confidence < self.entry_threshold / 10:
            return 0, 0.0
        
        # Map event to direction
        if predicted_event in [self.MB, self.LB]:  # Buy events
            return 1, confidence
        else:  # Sell events
            return -1, confidence
    
    def backtest(
        self,
        events: list[np.ndarray],
        prices: np.ndarray,
        times: np.ndarray,
        end_time: Optional[float] = None
    ) -> BacktestResult:
        """Run backtest on historical data.
        
        Args:
            events: List of event time arrays for each dimension
            prices: Price array corresponding to times
            times: Time grid for evaluation
            end_time: End of backtest period
            
        Returns:
            BacktestResult with performance metrics
        """
        if end_time is None:
            end_time = times[-1]
        
        trades = []
        current_position: Optional[Trade] = None
        equity = [1.0]  # Start with 1.0
        
        for i, t in enumerate(times):
            if t > end_time:
                break
            
            # Get current price
            price = prices[i]
            
            # Predict intensities up to current time
            # Use only events before current time
            past_events = [e[e < t] for e in events]
            intensities = self.estimator.predict_intensity(
                np.array([t]), past_events
            ).flatten()
            
            # Generate signal
            direction, strength = self.generate_signal(intensities)
            
            # Position management
            if current_position is None:
                # No position - check entry
                if direction != 0 and strength >= self.entry_threshold / 10:
                    current_position = Trade(
                        entry_time=t,
                        entry_price=price * (1 + self.transaction_costs * direction),
                        direction=direction
                    )
            else:
                # Have position - check exit
                time_held = t - current_position.entry_time
                opposite_signal = (direction != 0 and 
                                 direction != current_position.direction and
                                 strength >= self.exit_threshold / 10)
                time_exit = time_held >= self.max_position_duration
                
                if opposite_signal or time_exit:
                    # Close position
                    exit_price = price * (1 - self.transaction_costs * current_position.direction)
                    current_position.close(t, exit_price)
                    
                    # Update equity
                    trade_return = current_position.pnl / current_position.entry_price
                    equity.append(equity[-1] * (1 + trade_return))
                    trades.append(current_position)
                    current_position = None
            
            # Pad equity if no trade
            if len(equity) <= i:
                equity.append(equity[-1])
        
        # Close any open position at end
        if current_position is not None:
            current_position.close(times[-1], prices[-1])
            trade_return = current_position.pnl / current_position.entry_price
            equity.append(equity[-1] * (1 + trade_return))
            trades.append(current_position)
        
        # Compute metrics
        return self._compute_metrics(trades, equity, times[:len(equity)])
    
    def _compute_metrics(
        self,
        trades: list[Trade],
        equity: list[float],
        times: np.ndarray
    ) -> BacktestResult:
        """Compute performance metrics."""
        equity_array = np.array(equity)
        
        # Basic metrics
        num_trades = len(trades)
        if num_trades == 0:
            return BacktestResult(
                trades=[], equity_curve=pd.DataFrame(),
                sharpe_ratio=0, total_return=0, max_drawdown=0,
                win_rate=0, profit_factor=0, num_trades=0
            )
        
        # Returns
        total_return = (equity_array[-1] / equity_array[0]) - 1
        
        # Win rate
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        win_rate = len(winning_trades) / num_trades
        
        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl and t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Sharpe ratio (assuming equally spaced observations)
        returns = np.diff(equity_array) / equity_array[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        cummax = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - cummax) / cummax
        max_dd = np.min(drawdowns)
        
        # Equity curve DataFrame
        # Ensure times and equity_array have the same length
        n_points = min(len(times), len(equity_array))
        equity_df = pd.DataFrame({
            'time': times[:n_points],
            'equity': equity_array[:n_points]
        })
        
        return BacktestResult(
            trades=trades,
            equity_curve=equity_df,
            sharpe_ratio=sharpe,
            total_return=total_return,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades
        )
    
    def walk_forward_analysis(
        self,
        events: list[np.ndarray],
        prices: np.ndarray,
        times: np.ndarray,
        train_window: float,
        test_window: float,
        step: float
    ) -> list[BacktestResult]:
        """Perform walk-forward analysis.
        
        Args:
            events: Event data
            prices: Price data
            times: Time grid
            train_window: Training period duration
            test_window: Testing period duration
            step: Step size between windows
            
        Returns:
            List of backtest results for each window
        """
        results = []
        start_time = times[0]
        end_time = times[-1]
        
        current_time = start_time
        while current_time + train_window + test_window <= end_time:
            # Split data
            train_mask = (times >= current_time) & (times < current_time + train_window)
            test_mask = (times >= current_time + train_window) & (
                times < current_time + train_window + test_window
            )
            
            train_events = [e[(e >= current_time) & (e < current_time + train_window)] 
                          for e in events]
            
            # Refit model
            self.estimator.fit(
                train_events,
                end_time=current_time + train_window
            )
            
            # Backtest
            test_times = times[test_mask]
            test_prices = prices[test_mask]
            
            if len(test_times) > 0:
                result = self.backtest(
                    events, test_prices, test_times,
                    end_time=current_time + train_window + test_window
                )
                results.append(result)
            
            current_time += step
        
        return results
