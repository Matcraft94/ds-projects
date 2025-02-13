# Creado por: Lucy
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple

class BacktestStrategy:
    def __init__(self, data: pd.DataFrame, model: torch.nn.Module = None, scaler = None):
        """
        Inicializa la estrategia de backtesting.
        
        Args:
            data: DataFrame con los datos de mercado
            model: Modelo LSTM entrenado (opcional)
            scaler: StandardScaler ajustado (opcional)
        """
        self.data = data
        self.model = model
        self.scaler = scaler
        self.positions = pd.Series(index=data.index, dtype=float)
        self.window_size = 60
        self.predictions = None
        
    def generate_predictions(self) -> np.ndarray:
        """Genera predicciones usando el modelo LSTM si está disponible."""
        if self.model is None or self.scaler is None:
            return None
            
        self.model.eval()
        predictions = []
        
        # Preparar datos para predicción
        scaled_data = pd.DataFrame(
            self.scaler.transform(self.data),
            columns=self.data.columns,
            index=self.data.index
        )
        
        with torch.no_grad():
            for i in range(self.window_size, len(scaled_data)):
                window = scaled_data.iloc[i-self.window_size:i].values
                window_tensor = torch.FloatTensor(window).unsqueeze(0)
                prediction = self.model(window_tensor)
                predictions.append(prediction.item())
        
        full_predictions = np.array([np.nan] * self.window_size + predictions)
        self.predictions = full_predictions
        return full_predictions
    
    def generate_volatility_signals(self, threshold: float = 1.5) -> pd.Series:
        """Genera señales de trading basadas en volatilidad"""
        signals = pd.Series(0, index=self.data.index)
        signals[self.data['volatility'] > threshold * self.data['volatility'].mean()] = -1
        signals[self.data['volatility'] < 0.5 * self.data['volatility'].mean()] = 1
        return signals
    
    def generate_model_signals(self) -> pd.Series:
        """Genera señales basadas en las predicciones del modelo"""
        if self.predictions is None and self.model is not None:
            self.generate_predictions()
            
        signals = pd.Series(0, index=self.data.index)
        
        if self.predictions is not None:
            for i in range(self.window_size, len(self.predictions)):
                if np.isnan(self.predictions[i]):
                    continue
                    
                if self.predictions[i] > self.data['close'].iloc[i] * 1.01:
                    signals.iloc[i] = 1
                elif self.predictions[i] < self.data['close'].iloc[i] * 0.99:
                    signals.iloc[i] = -1
                    
        return signals
    
    def generate_signals(self) -> pd.Series:
        """Combina señales de volatilidad y modelo"""
        vol_signals = self.generate_volatility_signals()
        
        if self.model is not None:
            model_signals = self.generate_model_signals()
            # Combinar señales: usar señal del modelo si está disponible, sino usar volatilidad
            signals = model_signals.copy()
            signals[signals == 0] = vol_signals[signals == 0]
        else:
            signals = vol_signals
            
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
            'win_rate': len(returns[returns > 0]) / len(returns[returns != 0]),
            'num_trades': (signals != 0).sum()
        }
        
        return results