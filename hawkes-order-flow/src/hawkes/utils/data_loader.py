"""Data loading utilities for Hawkes process analysis.

Supports multiple data sources:
- Binance: Cryptocurrency trades via API
- CSV files: Generic trade data format
- LOBSTER: NASDAQ order book samples
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
import requests
from datetime import datetime, timedelta
import time
import os


class BinanceDataLoader:
    """Load historical trade data from Binance API.
    
    Binance provides free access to historical trade data for all
    trading pairs. This loader downloads aggregate trade data.
    
    Reference: https://github.com/binance/binance-public-data/
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """Initialize loader.
        
        Args:
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
    
    def download_trades(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Download aggregate trades from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_time: Start datetime
            end_time: End datetime
            save_path: Optional path to save CSV
            
        Returns:
            DataFrame with trades
        """
        all_trades = []
        current_time = start_time
        
        print(f"Downloading {symbol} trades from {start_time} to {end_time}...")
        
        while current_time < end_time:
            # Get trades in chunks
            chunk_end = min(current_time + timedelta(hours=1), end_time)
            
            params = {
                'symbol': symbol,
                'startTime': int(current_time.timestamp() * 1000),
                'endTime': int(chunk_end.timestamp() * 1000),
                'limit': 1000
            }
            
            try:
                response = self.session.get(
                    f"{self.BASE_URL}/aggTrades",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                trades = response.json()
                
                if trades:
                    all_trades.extend(trades)
                    print(f"  Downloaded {len(trades)} trades for {current_time}")
                
                time.sleep(self.rate_limit_delay)
                
            except requests.RequestException as e:
                print(f"  Error downloading {current_time}: {e}")
                time.sleep(1)
            
            current_time = chunk_end
        
        if not all_trades:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_trades)
        df['time'] = pd.to_datetime(df['T'], unit='ms')
        df['price'] = df['p'].astype(float)
        df['quantity'] = df['q'].astype(float)
        df['is_buyer_maker'] = df['m'].astype(bool)
        
        # Determine trade type
        # is_buyer_maker=True: buyer was maker (limit buy = passive)
        # is_buyer_maker=False: buyer was taker (market buy = aggressive)
        df['side'] = df['is_buyer_maker'].apply(lambda x: 'SELL' if x else 'BUY')
        df['type'] = df.apply(
            lambda row: 'MARKET' if not row['is_buyer_maker'] else 'LIMIT',
            axis=1
        )
        
        df = df[['time', 'price', 'quantity', 'side', 'type', 'is_buyer_maker']]
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Saved {len(df)} trades to {save_path}")
        
        return df
    
    def download_klines(
        self,
        symbol: str,
        interval: str = '1m',
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Download OHLCV klines from Binance.
        
        Args:
            symbol: Trading pair
            interval: Time interval (1m, 5m, 1h, etc.)
            start_time: Start datetime
            end_time: End datetime
            limit: Maximum candles
            
        Returns:
            DataFrame with OHLCV data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        response = self.session.get(f"{self.BASE_URL}/klines", params=params)
        response.raise_for_status()
        data = response.json()
        
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ]
        
        df = pd.DataFrame(data, columns=columns)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        return df


def load_csv_trades(
    filepath: str,
    timestamp_col: str = 'timestamp',
    price_col: str = 'price',
    size_col: str = 'size',
    side_col: str = 'side',
    timestamp_unit: str = 'ms'
) -> pd.DataFrame:
    """Load trades from CSV file.
    
    Args:
        filepath: Path to CSV
        timestamp_col: Name of timestamp column
        price_col: Name of price column
        size_col: Name of size/quantity column
        side_col: Name of side column (BUY/SELL)
        timestamp_unit: Unit for timestamp conversion
        
    Returns:
        Standardized DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Standardize column names
    df = df.rename(columns={
        timestamp_col: 'time',
        price_col: 'price',
        size_col: 'quantity',
        side_col: 'side'
    })
    
    # Parse timestamps
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        if timestamp_unit:
            df['time'] = pd.to_datetime(df['time'], unit=timestamp_unit)
        else:
            df['time'] = pd.to_datetime(df['time'])
    
    # Standardize side
    df['side'] = df['side'].str.upper()
    
    return df


def trades_to_hawkes_events(
    df: pd.DataFrame,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    event_types: list[str] = ['MB', 'MS', 'LB', 'LS']
) -> tuple[list[np.ndarray], dict]:
    """Convert trade DataFrame to Hawkes event format.
    
    Categorizes trades into 4 event types:
    - MB: Market Buy (aggressive buyer)
    - MS: Market Sell (aggressive seller)
    - LB: Limit Buy (passive buyer)
    - LS: Limit Sell (passive seller)
    
    Args:
        df: Trade DataFrame with columns [time, price, quantity, side, type]
        start_time: Optional start filter
        end_time: Optional end filter
        event_types: List of event type names
        
    Returns:
        (events, metadata) where events is list of time arrays
    """
    # Filter by time
    if start_time:
        df = df[df['time'] >= start_time]
    if end_time:
        df = df[df['time'] <= end_time]
    
    if len(df) == 0:
        return [np.array([]) for _ in range(4)], {}
    
    # Convert to relative time (seconds from start)
    t0 = df['time'].min()
    df['t'] = (df['time'] - t0).dt.total_seconds()
    
    # Categorize events
    events = []
    
    # Market Buy: side=BUY and (type=MARKET or is_buyer_maker=False)
    mb_mask = (df['side'] == 'BUY') & (
        (df.get('type') == 'MARKET') | 
        (~df.get('is_buyer_maker', False))
    )
    events.append(df.loc[mb_mask, 't'].values)
    
    # Market Sell: side=SELL and (type=MARKET or is_buyer_maker=True)
    ms_mask = (df['side'] == 'SELL') & (
        (df.get('type') == 'MARKET') | 
        (df.get('is_buyer_maker', False))
    )
    events.append(df.loc[ms_mask, 't'].values)
    
    # Limit Buy: side=BUY and type=LIMIT
    lb_mask = (df['side'] == 'BUY') & (
        (df.get('type') == 'LIMIT') | 
        (df.get('is_buyer_maker', False))
    )
    # Exclude already classified as market
    lb_mask = lb_mask & ~mb_mask
    events.append(df.loc[lb_mask, 't'].values)
    
    # Limit Sell: side=SELL and type=LIMIT
    ls_mask = (df['side'] == 'SELL') & (
        (df.get('type') == 'LIMIT') | 
        (~df.get('is_buyer_maker', True))
    )
    ls_mask = ls_mask & ~ms_mask
    events.append(df.loc[ls_mask, 't'].values)
    
    # Metadata
    metadata = {
        'start_time': t0,
        'duration_seconds': df['t'].max() if len(df) > 0 else 0,
        'n_events': [len(e) for e in events],
        'event_labels': event_types,
        'total_trades': len(df)
    }
    
    return events, metadata


def simulate_hawkes_process(
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    T: float,
    seed: Optional[int] = None
) -> list[np.ndarray]:
    """Simulate multivariate Hawkes process using Ogata's thinning method.
    
    Args:
        mu: Baseline intensities, shape (n_dims,)
        alpha: Excitation matrix, shape (n_dims, n_dims)
        beta: Decay matrix, shape (n_dims, n_dims)
        T: End time
        seed: Random seed
        
    Returns:
        List of event time arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_dims = len(mu)
    events = [[] for _ in range(n_dims)]
    
    # Current time and intensities
    t = 0.0
    lambda_current = mu.copy()
    
    # For tracking decay of past events (Ogata's recursive method)
    R = np.zeros((n_dims, n_dims))
    
    while t < T:
        # Total intensity
        lambda_total = np.sum(lambda_current)
        
        if lambda_total <= 0:
            break
        
        # Generate next event time
        dt = np.random.exponential(1.0 / lambda_total)
        t = t + dt
        
        if t >= T:
            break
        
        # Decay since last event
        for i in range(n_dims):
            for j in range(n_dims):
                R[i, j] *= np.exp(-beta[i, j] * dt)
        
        # Update intensities
        lambda_current = mu + np.sum(R, axis=1)
        lambda_total = np.sum(lambda_current)
        
        # Determine which dimension triggered
        probs = lambda_current / lambda_total
        dim = np.random.choice(n_dims, p=probs)
        
        # Record event
        events[dim].append(t)
        
        # Update R for this event
        for i in range(n_dims):
            R[i, dim] += alpha[i, dim]
        
        lambda_current = mu + np.sum(R, axis=1)
    
    # Convert to numpy arrays
    return [np.array(e) for e in events]


def load_sample_data() -> tuple[list[np.ndarray], dict]:
    """Load or generate sample data for testing.
    
    Returns:
        (events, metadata)
    """
    # Generate synthetic data
    np.random.seed(42)
    
    mu = np.array([0.5, 0.5, 1.0, 1.0])
    alpha = np.array([
        [0.1, 0.05, 0.08, 0.02],
        [0.05, 0.1, 0.02, 0.08],
        [0.08, 0.02, 0.1, 0.05],
        [0.02, 0.08, 0.05, 0.1]
    ])
    beta = np.ones((4, 4)) * 0.5
    
    events = simulate_hawkes_process(mu, alpha, beta, T=1000.0, seed=42)
    
    metadata = {
        'start_time': pd.Timestamp.now(),
        'duration_seconds': 1000.0,
        'n_events': [len(e) for e in events],
        'event_labels': ['MB', 'MS', 'LB', 'LS'],
        'total_trades': sum(len(e) for e in events),
        'synthetic': True
    }
    
    return events, metadata
