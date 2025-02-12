# Creado por: Lucy
import pandas as pd
import numpy as np
from typing import List, Dict

class BacktestStrategy:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.positions = pd.Series(index=data.index, dtype=float)
        
    def generate_signals(self, threshold: float = 1.5) -> pd.Series:
        """Genera señales de trading basadas en volatilidad"""
        signals = pd.Series(0, index=self.data.index)
        signals[self.data['volatility'] > threshold * self.data['volatility'].mean()] = -1
        signals[self.data['volatility'] < 0.5 * self.data['volatility'].mean()] = 1
        return signals
    
    def calculate_returns(self, signals: pd.Series) -> pd.Series:
        """Calcula retornos de la estrategia"""
        position_changes = signals.diff()
        returns = self.data['returns'] * signals.shift(1)
        returns[position_changes != 0] -= 0.001  # Simular costos de transacción
        return returns
    
    def run_backtest(self) -> Dict:
        """Ejecuta backtest completo"""
        signals = self.generate_signals()
        returns = self.calculate_returns(signals)
        
        results = {
            'total_return': returns.sum(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min(),
            'win_rate': len(returns[returns > 0]) / len(returns[returns != 0])
        }
        
        return results