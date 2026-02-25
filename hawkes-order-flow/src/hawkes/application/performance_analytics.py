"""Performance analytics for Hawkes-based trading strategies.

Provides comprehensive performance metrics and trade analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime


@dataclass
class Trade:
    """Single trade record."""
    entry_time: float
    exit_time: Optional[float] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    size: int = 0  # Positive = long, negative = short
    signal_type: str = ""
    pnl: Optional[float] = None
    return_pct: Optional[float] = None
    exit_reason: str = ""
    intensity: float = 0.0
    imbalance: float = 0.0
    
    @property
    def duration(self) -> Optional[float]:
        """Trade duration in seconds."""
        if self.exit_time is not None:
            return self.exit_time - self.entry_time
        return None
    
    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_time is not None and self.pnl is not None


@dataclass
class PerformanceSummary:
    """Comprehensive performance summary."""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade metrics
    avg_trade_duration: float = 0.0
    max_trade_duration: float = 0.0
    min_trade_duration: float = 0.0
    
    # Distribution
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Consecutive
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Hawkes-specific
    avg_intensity_at_entry: float = 0.0
    avg_imbalance_at_entry: float = 0.0
    signal_performance: Dict[str, Dict] = field(default_factory=dict)


class PerformanceAnalytics:
    """Performance analytics for trading strategies.
    
    Tracks and analyzes trade performance with specific focus on
    Hawkes process-based metrics.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """Initialize analytics.
        
        Args:
            initial_capital: Initial capital for return calculations
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Trade history
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        
        # Equity curve
        self.equity_curve: deque = deque(maxlen=100000)
        self.equity_curve.append((0.0, initial_capital))
        
        # Trade returns history
        self.returns: deque = deque(maxlen=10000)
        
        # Hawkes-specific tracking
        self._intensity_at_signals: deque = deque(maxlen=1000)
        self._imbalance_at_signals: deque = deque(maxlen=1000)
    
    def record_signal(
        self,
        timestamp: float,
        signal_type: str,
        price: float,
        intensity: float,
        imbalance: float,
        size: int
    ) -> Trade:
        """Record new trade entry.
        
        Args:
            timestamp: Entry time
            signal_type: Type of signal
            price: Entry price
            intensity: Intensity at entry
            imbalance: Imbalance at entry
            size: Position size
            
        Returns:
            Created Trade object
        """
        trade = Trade(
            entry_time=timestamp,
            entry_price=price,
            size=size,
            signal_type=signal_type,
            intensity=intensity,
            imbalance=imbalance
        )
        
        self.open_trades.append(trade)
        self._intensity_at_signals.append(intensity)
        self._imbalance_at_signals.append(imbalance)
        
        return trade
    
    def close_trade(
        self,
        trade: Trade,
        exit_time: float,
        exit_price: float,
        exit_reason: str = "signal"
    ):
        """Close an open trade.
        
        Args:
            trade: Trade to close
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        if trade not in self.open_trades:
            return
        
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Calculate P&L
        if trade.size > 0:  # Long
            trade.pnl = (exit_price - trade.entry_price) * trade.size
            trade.return_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:  # Short
            trade.pnl = (trade.entry_price - exit_price) * abs(trade.size)
            trade.return_pct = (trade.entry_price - exit_price) / trade.entry_price
        
        # Update capital
        self.current_capital += trade.pnl
        
        # Record return
        self.returns.append(trade.return_pct)
        
        # Move to closed trades
        self.open_trades.remove(trade)
        self.trades.append(trade)
        
        # Update equity curve
        self.equity_curve.append((exit_time, self.current_capital))
    
    def update_equity(self, timestamp: float, price: float):
        """Update equity curve with mark-to-market.
        
        Args:
            timestamp: Current time
            price: Current price
        """
        # Calculate unrealized P&L from open trades
        unrealized = 0.0
        for trade in self.open_trades:
            if trade.size > 0:  # Long
                unrealized += (price - trade.entry_price) * trade.size
            else:  # Short
                unrealized += (trade.entry_price - price) * abs(trade.size)
        
        total_equity = self.current_capital + unrealized
        self.equity_curve.append((timestamp, total_equity))
    
    def calculate_performance(self) -> PerformanceSummary:
        """Calculate comprehensive performance summary."""
        if not self.trades:
            return PerformanceSummary()
        
        summary = PerformanceSummary()
        
        # Basic counts
        summary.total_trades = len(self.trades)
        summary.winning_trades = sum(1 for t in self.trades if t.pnl and t.pnl > 0)
        summary.losing_trades = sum(1 for t in self.trades if t.pnl and t.pnl <= 0)
        summary.win_rate = summary.winning_trades / summary.total_trades if summary.total_trades > 0 else 0
        
        # P&L calculations
        profits = [t.pnl for t in self.trades if t.pnl and t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl and t.pnl <= 0]
        all_pnls = [t.pnl for t in self.trades if t.pnl is not None]
        
        summary.total_pnl = sum(all_pnls)
        summary.gross_profit = sum(profits) if profits else 0.0
        summary.gross_loss = sum(losses) if losses else 0.0
        summary.profit_factor = (
            abs(summary.gross_profit / summary.gross_loss) 
            if summary.gross_loss != 0 else float('inf')
        )
        
        summary.avg_trade = np.mean(all_pnls) if all_pnls else 0.0
        summary.avg_winner = np.mean(profits) if profits else 0.0
        summary.avg_loser = np.mean(losses) if losses else 0.0
        
        # Returns for risk metrics
        returns_array = np.array([t.return_pct for t in self.trades if t.return_pct is not None])
        
        if len(returns_array) > 1:
            # Sharpe ratio
            excess_returns = returns_array - 0.02 / 252  # Daily risk-free
            summary.sharpe_ratio = (
                np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                if np.std(excess_returns) > 0 else 0.0
            )
            
            # Sortino ratio
            downside_returns = returns_array[returns_array < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            summary.sortino_ratio = (
                np.mean(excess_returns) / downside_std * np.sqrt(252)
                if downside_std > 0 else 0.0
            )
            
            # Distribution
            summary.skewness = self._calculate_skewness(returns_array)
            summary.kurtosis = self._calculate_kurtosis(returns_array)
        
        # Drawdown
        summary.max_drawdown, summary.max_drawdown_pct = self._calculate_max_drawdown()
        
        # Calmar ratio
        if summary.max_drawdown_pct > 0 and len(returns_array) > 0:
            annual_return = np.mean(returns_array) * 252
            summary.calmar_ratio = annual_return / summary.max_drawdown_pct
        
        # Trade duration
        durations = [t.duration for t in self.trades if t.duration is not None]
        if durations:
            summary.avg_trade_duration = np.mean(durations)
            summary.max_trade_duration = np.max(durations)
            summary.min_trade_duration = np.min(durations)
        
        # Consecutive trades
        summary.max_consecutive_wins, summary.max_consecutive_losses = \
            self._calculate_consecutive()
        
        # Hawkes-specific metrics
        if self._intensity_at_signals:
            summary.avg_intensity_at_entry = np.mean(list(self._intensity_at_signals))
        if self._imbalance_at_signals:
            summary.avg_imbalance_at_entry = np.mean(list(self._imbalance_at_signals))
        
        # Signal type performance
        summary.signal_performance = self._analyze_by_signal_type()
        
        return summary
    
    def _calculate_max_drawdown(self) -> Tuple[float, float]:
        """Calculate maximum drawdown."""
        if len(self.equity_curve) < 2:
            return 0.0, 0.0
        
        equity = np.array([e[1] for e in self.equity_curve])
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        
        max_dd_idx = np.argmax(drawdowns)
        max_dd = running_max[max_dd_idx] - equity[max_dd_idx]
        max_dd_pct = drawdowns[max_dd_idx]
        
        return max_dd, max_dd_pct
    
    def _calculate_consecutive(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        if not self.trades:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.pnl and trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.pnl and trade.pnl <= 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        if len(data) < 3:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        return (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        if len(data) < 4:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        
        kurt = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * \
               np.sum(((data - mean) / std) ** 4) - \
               (3 * (n-1) ** 2 / ((n-2) * (n-3)))
        return kurt
    
    def _analyze_by_signal_type(self) -> Dict[str, Dict]:
        """Analyze performance by signal type."""
        signal_stats = {}
        
        for trade in self.trades:
            sig_type = trade.signal_type
            if sig_type not in signal_stats:
                signal_stats[sig_type] = {
                    'count': 0,
                    'wins': 0,
                    'total_pnl': 0.0,
                    'avg_return': 0.0
                }
            
            signal_stats[sig_type]['count'] += 1
            if trade.pnl and trade.pnl > 0:
                signal_stats[sig_type]['wins'] += 1
            if trade.pnl:
                signal_stats[sig_type]['total_pnl'] += trade.pnl
            if trade.return_pct:
                signal_stats[sig_type]['avg_return'] += trade.return_pct
        
        # Calculate averages
        for sig_type in signal_stats:
            stats = signal_stats[sig_type]
            if stats['count'] > 0:
                stats['win_rate'] = stats['wins'] / stats['count']
                stats['avg_return'] /= stats['count']
                stats['avg_pnl'] = stats['total_pnl'] / stats['count']
        
        return signal_stats
    
    def generate_report(self) -> str:
        """Generate formatted performance report."""
        perf = self.calculate_performance()
        
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║              PERFORMANCE ANALYTICS REPORT                       ║
╠════════════════════════════════════════════════════════════════╣
║ CAPITAL & RETURNS                                               ║
║   Initial Capital:  ${self.initial_capital:>15,.2f}              ║
║   Current Capital:  ${self.current_capital:>15,.2f}              ║
║   Total P&L:        ${perf.total_pnl:>15,.2f}  ({perf.total_pnl/self.initial_capital:+.2%})      ║
╠════════════════════════════════════════════════════════════════╣
║ TRADE STATISTICS                                                ║
║   Total Trades:     {perf.total_trades:>15,d}                  ║
║   Win Rate:         {perf.win_rate:>15.1%}                  ║
║   Profit Factor:    {perf.profit_factor:>15.2f}                  ║
║   Avg Trade:        ${perf.avg_trade:>15,.2f}                  ║
║   Avg Winner:       ${perf.avg_winner:>15,.2f}                  ║
║   Avg Loser:        ${perf.avg_loser:>15,.2f}                  ║
╠════════════════════════════════════════════════════════════════╣
║ RISK METRICS                                                    ║
║   Max Drawdown:     {perf.max_drawdown_pct:>15.2%}                  ║
║   Sharpe Ratio:     {perf.sharpe_ratio:>15.2f}                  ║
║   Sortino Ratio:    {perf.sortino_ratio:>15.2f}                  ║
║   Calmar Ratio:     {perf.calmar_ratio:>15.2f}                  ║
╠════════════════════════════════════════════════════════════════╣
║ HAWKES-SPECIFIC                                                 ║
║   Avg Intensity at Entry:  {perf.avg_intensity_at_entry:>10.2f}          ║
║   Avg Imbalance at Entry:  {perf.avg_imbalance_at_entry:>10.3f}          ║
╚════════════════════════════════════════════════════════════════╝
"""
        return report
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export trades to DataFrame."""
        data = []
        for trade in self.trades:
            data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'duration': trade.duration,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct,
                'signal_type': trade.signal_type,
                'exit_reason': trade.exit_reason
            })
        return pd.DataFrame(data)
