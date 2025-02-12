# Creado por: Lucy
import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcula el Average True Range (ATR) para un DataFrame de datos financieros.
        
        Args:
            df: DataFrame con columnas 'high', 'low', 'close'
            period: Período para el cálculo del ATR (default: 14)
            
        Returns:
            pd.Series: Serie con los valores ATR calculados
        """
        # Validación de columnas requeridas
        required_columns = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame debe contener las columnas: {required_columns}")
            
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    @staticmethod
    def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera características técnicas para análisis financiero.
        
        Args:
            df: DataFrame con datos de precios y volumen
               Debe contener columnas: 'close', 'high', 'low', 'tick_volume'
               
        Returns:
            pd.DataFrame: DataFrame con las características técnicas añadidas
        """
        # Validación de columnas requeridas
        required_columns = ['close', 'high', 'low', 'tick_volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame debe contener las columnas: {required_columns}")
        
        df = df.copy()
        
        # Add features
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['ATR'] = FeatureEngineer.calculate_atr(df)
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
        
        df = df.dropna()
        
        print("Final shape, after dropna:", df.shape)
        
        return df