# Creado por: Lucy
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple

class BacktestStrategy:
    def __init__(
        self,
        data: pd.DataFrame,
        model: torch.nn.Module = None,
        scaler = None,
        scaler_stats: dict = None,
        target_col: str = "close",
    ):
        """
        Inicializa la estrategia de backtesting.

        Args:
            data: DataFrame con los datos de mercado
            model: Modelo LSTM entrenado (opcional)
            scaler: no usado (mantenido por compatibilidad)
            scaler_stats: estadísticas de normalización ajustadas en el fold
                de entrenamiento (dict con 'mean', 'std', 'columns' — ver
                MarketDataset.fit_scaler). Las predicciones se
                des-normalizan a escala de precio antes de compararse con close.
            target_col: columna objetivo del modelo
        """
        self.data = data
        self.model = model
        self.scaler = scaler
        self.scaler_stats = scaler_stats
        self.target_col = target_col
        self.positions = pd.Series(index=data.index, dtype=float)
        self.window_size = 60
        self.predictions = None

    def generate_predictions(self) -> np.ndarray:
        """Genera predicciones usando el modelo LSTM si está disponible.

        Las predicciones se des-normalizan con las estadísticas del target
        para quedar en escala de precio y poder compararse con 'close'.
        """
        if self.model is None or self.scaler_stats is None:
            return None

        self.model.eval()
        predictions = []

        stats = self.scaler_stats
        cols = list(stats["columns"])
        scaled = (self.data[cols] - stats["mean"]) / stats["std"]

        target_mean = stats["mean"][self.target_col]
        target_std = stats["std"][self.target_col]

        with torch.no_grad():
            for i in range(self.window_size, len(scaled)):
                window = scaled.iloc[i - self.window_size:i].values
                window_tensor = torch.FloatTensor(window).unsqueeze(0)
                prediction = self.model(window_tensor).item()
                predictions.append(prediction * target_std + target_mean)

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

        if self.model is not None and self.scaler_stats is not None:
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

    def run_backtest(self, periods_per_year: int = 252) -> Dict:
        """Ejecuta backtest completo.

        Args:
            periods_per_year: barras por año para anualizar el Sharpe.
                DEBE coincidir con la frecuencia de las barras de datos
                (252 = diarias, 252*78 ≈ minutarias de mercado US,
                365*24*60 = minutarias 24/7). El default 252 asume barras
                diarias; con otra frecuencia el valor por defecto es un
                error, no una convención.
        """
        signals = self.generate_signals()
        returns = self.calculate_returns(signals)

        equity = (1 + returns).cumprod()
        running_max = equity.cummax()
        max_dd = ((equity - running_max) / running_max).min()

        results = {
            'total_return': equity.iloc[-1] - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0.0,
            'max_drawdown': max_dd,
            'win_rate': len(returns[returns > 0]) / len(returns[returns != 0]),
            'num_trades': (signals != 0).sum(),
            'periods_per_year': periods_per_year,
        }

        return results