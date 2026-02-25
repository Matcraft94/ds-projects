# Performance Optimization Guide

If estimation is taking too long, follow this guide.

## Quick Fixes (Try These First)

### 1. Use Fast Estimator

Replace this:

```python
from hawkes.estimation import MultivariateHawkesMLE
kernel = ExponentialKernel(n_dims=4)
estimator = MultivariateHawkesMLE(kernel)
```

With this:

```python
from hawkes.estimation import FastMultivariateHawkesMLE
estimator = FastMultivariateHawkesMLE(n_dims=4)
```

**Speedup: 10-100x for large datasets**

### 2. Enable Parallel Processing

```python
from hawkes.estimation import ParallelMultivariateHawkesMLE

estimator = ParallelMultivariateHawkesMLE(
    n_dims=4,
    n_init=5,      # Number of random starts
    n_jobs=-1      # Use all CPU cores
)
```

### 3. Subsample Large Datasets

```python
estimator.fit(
    events,
    end_time=T,
    max_events=5000  # Use only 5000 events per dimension
)
```

### 4. Skip EM Algorithm

EM is 5-10x slower than MLE. For quick results:

- Use MLE (L-BFGS-B) instead of EM
- If you need EM, reduce `max_iter` to 20-30

## Benchmark Results


| Method             | 1K events | 5K events | 10K events |
| ------------------ | --------- | --------- | ---------- |
| Standard MLE       | 30s       | 10min+    | Hours      |
| Fast MLE           | 2s        | 10s       | 30s        |
| Parallel (4 cores) | 5s        | 20s       | 60s        |
| EM (30 iter)       | 60s       | 30min+    | Hours      |

*Times are approximate on a 4-core machine*

## Detailed Configuration

### For Very Large Datasets (>10K events)

```python
CONFIG = {
    'use_fast_mle': True,
    'max_events': 5000,      # Aggressive subsampling
    'n_init': 3,             # Fewer random starts
    'n_jobs': -1,            # All cores
}
```

### For Maximum Accuracy

```python
CONFIG = {
    'use_fast_mle': True,
    'max_events': None,      # Use all data
    'n_init': 10,            # Many random starts
    'n_jobs': -1,
}
```

### For Quick Exploration

```python
CONFIG = {
    'use_fast_mle': True,
    'max_events': 2000,      # Heavy subsampling
    'n_init': 1,             # Single start
    'n_jobs': 1,             # No parallel overhead
}
```

## Compile Cython Extensions

For additional 2-3x speedup:

```bash
cd hawkes-order-flow
source venv/bin/activate
python setup.py build_ext --inplace
```

## Monitor Performance

Run the benchmark script:

```bash
python scripts/benchmark_performance.py --n-events 5000 --method all
```

## Troubleshooting

### Still Too Slow?

1. **Check data size**: `sum(len(e) for e in events)`

   - If > 10000, definitely use subsampling
2. **Check CPU usage**: Run `htop` during estimation

   - If using < 100%, parallel is working
   - If at 100% on one core, use parallel mode
3. **Profile the code**:

   ```python
   import cProfile
   cProfile.run('estimator.fit(events, end_time=T)')
   ```
4. **Use GPU** (advanced):

   - Not implemented yet, but could speed up likelihood computation
   - Would require rewriting in PyTorch/JAX

### Memory Issues?

For very large datasets:

```python
# Process in chunks
from hawkes.estimation.fast_mle import FastMultivariateHawkesMLE

estimator = FastMultivariateHawkesMLE(n_dims=4)
estimator.fit(events, end_time=T, max_events=5000)
```
