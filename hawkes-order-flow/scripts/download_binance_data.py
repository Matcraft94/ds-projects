#!/usr/bin/env python3
"""Download historical trade data from Binance.

Usage:
    python download_binance_data.py --symbol BTCUSDT --start 2024-01-01 --end 2024-01-31
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hawkes.utils.data_loader import BinanceDataLoader, trades_to_hawkes_events


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance trade data"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading pair symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['trades', 'events'],
        default='trades',
        help="Output format: raw trades or Hawkes events"
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download data
    loader = BinanceDataLoader()
    
    if args.format == 'trades':
        output_file = output_dir / f"{args.symbol}_{args.start}_{args.end}_trades.csv"
        df = loader.download_trades(
            args.symbol,
            start_date,
            end_date,
            save_path=str(output_file)
        )
        print(f"Downloaded {len(df)} trades")
        
    elif args.format == 'events':
        # Download and convert to events
        df = loader.download_trades(args.symbol, start_date, end_date)
        events, metadata = trades_to_hawkes_events(df)
        
        # Save events
        import numpy as np
        output_file = output_dir / f"{args.symbol}_{args.start}_{args.end}_events.npz"
        np.savez(
            output_file,
            mb=events[0],
            ms=events[1],
            lb=events[2],
            ls=events[3],
            metadata=metadata
        )
        print(f"Saved events to {output_file}")
        print(f"Event counts: MB={len(events[0])}, MS={len(events[1])}, "
              f"LB={len(events[2])}, LS={len(events[3])}")


if __name__ == "__main__":
    main()
