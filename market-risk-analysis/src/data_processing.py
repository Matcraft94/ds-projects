# Creado por: Lucy
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
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
    """Sliding-window dataset with explicit target column and train-fitted
    normalization.

    Fixes two defects of the original version:
    - target was ``self.data[idx + window, 0]`` (first numeric column),
      which for the crash500 schema is ``open`` — the comment said close.
      The target is now an explicit ``target_col``.
    - normalization used the mean/std of whatever data the dataset was
      built from, leaking validation statistics into folds. Fit the scaler
      on the TRAIN fold (``fit_scaler``) and pass it to every dataset.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int,
        scaler_stats: Optional[dict] = None,
        target_col: str = "close",
    ):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data_numeric = data[numeric_columns]

        if scaler_stats is None:
            scaler_stats = self.fit_scaler(data)
        self.scaler_stats = scaler_stats
        self.feature_cols = list(scaler_stats["columns"])
        if target_col not in self.feature_cols:
            raise ValueError(f"target_col '{target_col}' not in numeric columns {self.feature_cols}")
        self.target_col = target_col

        data_numeric = data_numeric[self.feature_cols]
        data_normalized = (data_numeric - scaler_stats["mean"]) / scaler_stats["std"]
        self.data = torch.FloatTensor(data_normalized.values)
        self.target_idx = self.feature_cols.index(target_col)
        self.window_size = window_size

    @staticmethod
    def fit_scaler(data: pd.DataFrame) -> dict:
        numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
        return {
            "columns": numeric_columns,
            "mean": data[numeric_columns].mean(),
            "std": data[numeric_columns].std().replace(0, 1.0),
        }

    def __len__(self) -> int:
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size, self.target_idx]
        return x, y