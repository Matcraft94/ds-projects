#!/usr/bin/env python3
"""Benchmark performance of different estimation methods.

Usage:
    python benchmark_performance.py --n-events 1000 --method all
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hawkes.utils.data_loader import simulate_hawkes_process
from hawkes.kernels import ExponentialKernel
from hawkes.estimation import (
    MultivariateHawkesMLE,
    MultivariateHawkesEM,
    FastMultivariateHawkesMLE,
    ParallelMultivariateHawkesMLE
)


def generate_test_data(n_events_target=1000, n_dims=4, seed=42):
    """Generate test data with approximately n_events total."""
    np.random.seed(seed)
    
    # Adjust T to get approximately n_events
    mu = np.array([0.5, 0.5, 1.0, 1.0])
    alpha = np.array([
        [0.1, 0.05, 0.08, 0.02],
        [0.05, 0.1, 0.02, 0.08],
        [0.08, 0.02, 0.1, 0.05],
        [0.02, 0.08, 0.05, 0.1]
    ])
    beta = np.ones((4, 4)) * 0.5
    
    # Scale T to get desired number of events
    T = n_events_target / np.sum(mu) * 2  # Rough estimate
    
    events = simulate_hawkes_process(mu, alpha, beta, T, seed=seed)
    actual_events = sum(len(e) for e in events)
    
    return events, T, actual_events


def benchmark_method(method_name, estimator, events, T, n_dims=4):
    """Benchmark a single method."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {method_name}")
    print(f"{'='*60}")
    
    start = time.time()
    
    try:
        if hasattr(estimator, 'fit'):
            if 'Fast' in method_name or 'Parallel' in method_name:
                estimator.fit(events, end_time=T, verbose=False)
            else:
                estimator.fit(events, end_time=T, method='L-BFGS-B', verbose=False)
        
        elapsed = time.time() - start
        
        # Get results
        if hasattr(estimator, 'log_likelihood_'):
            ll = estimator.log_likelihood_
        else:
            ll = None
            
        if hasattr(estimator, 'compute_spectral_radius'):
            rho = estimator.compute_spectral_radius()
        else:
            rho = None
        
        print(f"✓ Completed in {elapsed:.2f} seconds")
        if ll:
            print(f"  Log-likelihood: {ll:.2f}")
        if rho:
            print(f"  Spectral radius: {rho:.4f}")
        
        return {
            'method': method_name,
            'time': elapsed,
            'log_likelihood': ll,
            'spectral_radius': rho,
            'success': True
        }
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Failed after {elapsed:.2f} seconds")
        print(f"  Error: {e}")
        return {
            'method': method_name,
            'time': elapsed,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hawkes estimation methods")
    parser.add_argument("--n-events", type=int, default=1000, 
                       help="Target number of events (default: 1000)")
    parser.add_argument("--method", type=str, default="all",
                       choices=["all", "mle", "em", "fast", "parallel"],
                       help="Method to benchmark")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Number of parallel jobs (default: -1 = all cores)")
    
    args = parser.parse_args()
    
    print("Hawkes Estimation Performance Benchmark")
    print("=" * 60)
    print(f"Target events: {args.n_events}")
    print(f"Method: {args.method}")
    
    # Generate data
    print("\nGenerating test data...")
    events, T, actual = generate_test_data(args.n_events)
    print(f"Generated {actual} events (T={T:.1f}s)")
    
    # Methods to benchmark
    methods = []
    
    if args.method in ["all", "fast"]:
        methods.append(("FastMultivariateHawkesMLE", 
                       FastMultivariateHawkesMLE(n_dims=4)))
    
    if args.method in ["all", "parallel"]:
        methods.append(("ParallelMultivariateHawkesMLE (5 init)", 
                       ParallelMultivariateHawkesMLE(n_dims=4, n_init=5, n_jobs=args.n_jobs)))
    
    if args.method in ["all", "mle"]:
        kernel = ExponentialKernel(n_dims=4)
        methods.append(("Standard MultivariateHawkesMLE", 
                       MultivariateHawkesMLE(kernel)))
    
    if args.method in ["all", "em"]:
        kernel = ExponentialKernel(n_dims=4)
        methods.append(("MultivariateHawkesEM (30 iter)", 
                       MultivariateHawkesEM(kernel, max_iter=30, verbose=False)))
    
    # Run benchmarks
    results = []
    for name, estimator in methods:
        result = benchmark_method(name, estimator, events, T)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"{status} {r['method']}: {r['time']:.2f}s", end="")
        if r['log_likelihood']:
            print(f" (LL: {r['log_likelihood']:.1f})", end="")
        print()
    
    # Find fastest successful
    successful = [r for r in results if r['success']]
    if successful:
        fastest = min(successful, key=lambda x: x['time'])
        print(f"\nFastest method: {fastest['method']} ({fastest['time']:.2f}s)")
        
        # Speedup compared to standard MLE
        standard = next((r for r in results if 'Standard' in r['method']), None)
        if standard and standard['success'] and fastest != standard:
            speedup = standard['time'] / fastest['time']
            print(f"Speedup vs standard MLE: {speedup:.1f}x")
    
    print("\nRecommendations:")
    print("  - Use FastMultivariateHawkesMLE for single runs (fastest)")
    print("  - Use ParallelMultivariateHawkesMLE for robustness (multiple init)")
    print("  - Avoid standard MLE for datasets > 5000 events")
    print("  - Use EM only for small datasets (< 2000 events)")


if __name__ == "__main__":
    main()
