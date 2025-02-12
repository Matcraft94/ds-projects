# Creado por: Lucy
import pandas as pd
import numpy as np
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader

class MarketDataProcessor:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        
    def load_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Carga y preprocesa datos iniciales"""
        df = df.copy()
        print("Initial shape:", df.shape)
        
        df['returns'] = df['close'].pct_change()
        
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        df['rsi'] = self.calculate_rsi(df['close'])
        
        df = df.dropna()
        print("Final shape:", df.shape)
        
        return df
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula el RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = pd.Series(np.where(loss == 0, np.nan, gain / loss), index=prices.index)
        return 100 - (100 / (1 + rs))

class MarketDataset(Dataset):
    def __init__(self, data: pd.DataFrame, window_size: int):
        # Select only numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data_numeric = data[numeric_columns]
        
        # Normalize numeric data
        data_normalized = (data_numeric - data_numeric.mean()) / data_numeric.std()
        self.data = torch.FloatTensor(data_normalized.values)
        self.window_size = window_size
    
    def __len__(self) -> int:
        return len(self.data) - self.window_size
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size, 0]  # Predecir próximo close
        return x, y