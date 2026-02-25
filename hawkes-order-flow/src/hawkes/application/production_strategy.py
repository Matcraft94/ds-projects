"""Production-ready trading strategy with robust signal generation.

Uses event rate differential as primary signal source.
"""

import numpy as np
from typing import Optional, Dict, List
from collections import deque


class ProductionHawkesStrategy:
    """Production trading strategy with robust parameters.
    
    Signal Logic:
    1. Buy when buy event rate > sell event rate (positive imbalance)
    2. Sell when sell event rate > buy event rate (negative imbalance)
    3. Exit on profit target, stop loss, or time limit
    
    Key Parameters:
    - Entry threshold: 0.02 (2% imbalance)
    - Stop loss: 15 bps
    - Take profit: 45 bps  
    - Max hold: 20 seconds
    """
    
    def __init__(
        self,
        entry_threshold: float = 0.02,
        stop_loss_pct: float = 0.0015,
        take_profit_pct: float = 0.0045,
        max_hold_seconds: float = 20.0,
        cooldown_seconds: float = 3.0,
        commission: float = 0.05,
        spread: float = 0.0001
    ):
        """Initialize strategy.
        
        Args:
            entry_threshold: Minimum imbalance to trigger signal (0.02 = 2%)
            stop_loss_pct: Stop loss percentage (0.0015 = 15 bps)
            take_profit_pct: Take profit percentage (0.0045 = 45 bps)
            max_hold_seconds: Maximum holding time
            cooldown_seconds: Time between trades
            commission: Commission per trade in $
            spread: Bid-ask spread as fraction of price
        """
        self.entry_threshold = entry_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_seconds = max_hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.commission = commission
        self.spread = spread
        
        # State
        self.position = 0
        self.entry_price = None
        self.entry_time = None
        self.last_trade_time = -np.inf
        
        # Event counting
        self.buy_events = deque(maxlen=100)
        self.sell_events = deque(maxlen=100)
        self._current_time = 0.0
        
        # Performance tracking
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.trades = []
    
    def update(self, timestamp: float, dimension: int):
        """Update with new market event.
        
        Args:
            timestamp: Event time
            dimension: 0=MB, 1=MS, 2=LB, 3=LS
        """
        self._current_time = timestamp
        
        # Count buy/sell events
        if dimension in [0, 2]:  # Buy side
            self.buy_events.append(timestamp)
        else:  # Sell side
            self.sell_events.append(timestamp)
        
        # Clean old events (older than 10 seconds)
        cutoff = timestamp - 10.0
        while self.buy_events and self.buy_events[0] < cutoff:
            self.buy_events.popleft()
        while self.sell_events and self.sell_events[0] < cutoff:
            self.sell_events.popleft()
    
    def check_signal(self, timestamp: float, price: float) -> Optional[Dict]:
        """Check for trading signal.
        
        Args:
            timestamp: Current time
            price: Current mid price
            
        Returns:
            Signal dict or None
        """
        # Check cooldown
        if timestamp - self.last_trade_time < self.cooldown_seconds:
            return None
        
        # Check exit conditions if in position
        if self.position != 0 and self.entry_price is not None:
            exit_signal = self._check_exit(timestamp, price)
            if exit_signal:
                return exit_signal
        
        # Calculate imbalance
        n_buy = len(self.buy_events)
        n_sell = len(self.sell_events)
        total = n_buy + n_sell
        
        if total < 5:  # Need minimum activity
            return None
        
        imbalance = (n_buy - n_sell) / total
        
        # Generate entry signal
        if imbalance > self.entry_threshold and self.position <= 0:
            return {
                'action': 'BUY',
                'direction': 1,
                'imbalance': imbalance,
                'n_buy': n_buy,
                'n_sell': n_sell,
                'expected_cost': self._calc_cost(price, 1)
            }
        elif imbalance < -self.entry_threshold and self.position >= 0:
            return {
                'action': 'SELL',
                'direction': -1,
                'imbalance': imbalance,
                'n_buy': n_buy,
                'n_sell': n_sell,
                'expected_cost': self._calc_cost(price, 1)
            }
        
        return None
    
    def _check_exit(self, timestamp: float, price: float) -> Optional[Dict]:
        """Check if should exit position."""
        if self.position == 0 or self.entry_price is None:
            return None
        
        holding_time = timestamp - self.entry_time
        pnl_pct = (price - self.entry_price) / self.entry_price * self.position
        
        # Stop loss
        if pnl_pct < -self.stop_loss_pct:
            return {
                'action': 'STOP_LOSS',
                'direction': 0,
                'pnl_pct': pnl_pct,
                'holding_time': holding_time
            }
        
        # Take profit
        if pnl_pct > self.take_profit_pct:
            return {
                'action': 'TAKE_PROFIT',
                'direction': 0,
                'pnl_pct': pnl_pct,
                'holding_time': holding_time
            }
        
        # Time exit
        if holding_time > self.max_hold_seconds:
            return {
                'action': 'TIME_EXIT',
                'direction': 0,
                'pnl_pct': pnl_pct,
                'holding_time': holding_time
            }
        
        return None
    
    def execute(self, signal: Dict, price: float) -> Dict:
        """Execute signal.
        
        Args:
            signal: Signal dict
            price: Execution price
            
        Returns:
            Execution result
        """
        action = signal['action']
        
        result = {
            'action': action,
            'price': price,
            'pnl': 0.0,
            'cost': 0.0,
            'position_after': self.position
        }
        
        if action in ['BUY', 'SELL']:
            # Entry
            direction = 1 if action == 'BUY' else -1
            self.position = direction
            self.entry_price = price
            self.entry_time = self._current_time
            self.last_trade_time = self._current_time
            
            cost = self._calc_cost(price, 1)
            self.total_commission += self.commission
            self.total_slippage += price * self.spread
            
            result['cost'] = cost
            result['position_after'] = self.position
            
        elif action in ['STOP_LOSS', 'TAKE_PROFIT', 'TIME_EXIT']:
            # Exit
            if self.position != 0 and self.entry_price is not None:
                pnl = (price - self.entry_price) * self.position
                result['pnl'] = pnl
                result['cost'] = self._calc_cost(price, 1)
                
                self.trades.append({
                    'entry_time': self.entry_time,
                    'exit_time': self._current_time,
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'pnl': pnl,
                    'exit_type': action
                })
            
            self.position = 0
            self.entry_price = None
            self.entry_time = None
            self.last_trade_time = self._current_time
            result['position_after'] = 0
        
        return result
    
    def _calc_cost(self, price: float, size: int) -> float:
        """Calculate transaction cost."""
        return self.commission * 2 + price * self.spread * size
    
    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        return {
            'total_trades': len(self.trades),
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'current_position': self.position,
            'winning_trades': sum(1 for t in self.trades if t['pnl'] > 0),
            'total_pnl': sum(t['pnl'] for t in self.trades)
        }
