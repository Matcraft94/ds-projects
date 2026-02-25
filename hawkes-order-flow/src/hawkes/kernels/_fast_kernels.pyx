"""Cython-optimized kernel computations."""

import numpy as np
cimport numpy as np
from libc.math cimport exp, log

# DTYPE for numpy arrays
dt = np.float64
ctypedef np.float64_t DTYPE_t

def exp_kernel_eval(
    np.ndarray[DTYPE_t, ndim=1] t,
    DTYPE_t alpha,
    DTYPE_t beta
):
    """Evaluate exponential kernel: α * exp(-β * t) for t >= 0.
    
    Args:
        t: Time differences (1D array)
        alpha: Excitation magnitude
        beta: Decay rate
        
    Returns:
        Kernel values
    """
    cdef int n = t.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=dt)
    cdef int i
    cdef DTYPE_t val
    
    for i in range(n):
        if t[i] >= 0:
            val = alpha * exp(-beta * t[i])
            result[i] = val
    
    return result

def exp_kernel_integral(DTYPE_t alpha, DTYPE_t beta):
    """Compute integral of exponential kernel: ∫_0^∞ α*exp(-β*t) dt = α/β."""
    if beta <= 0:
        return np.inf
    return alpha / beta

def compute_intensity_recursive_exp(
    list events,
    np.ndarray[DTYPE_t, ndim=1] mu,
    np.ndarray[DTYPE_t, ndim=2] alpha,
    np.ndarray[DTYPE_t, ndim=2] beta,
    DTYPE_t end_time,
    DTYPE_t resolution
):
    """Compute conditional intensities using Ogata's recursive method.
    
    For exponential kernels, we can compute in O(N) instead of O(N²).
    
    Args:
        events: List of numpy arrays with event times for each dimension
        mu: Baseline intensities
        alpha: Excitation matrix
        beta: Decay matrix
        end_time: End of observation period
        resolution: Time resolution for output
        
    Returns:
        (times, intensities) tuple
    """
    cdef int n_dims = len(events)
    cdef int n_steps = int((end_time - 0.0) / resolution) + 1
    cdef np.ndarray[DTYPE_t, ndim=1] times = np.linspace(0, end_time, n_steps)
    cdef np.ndarray[DTYPE_t, ndim=2] intensities = np.zeros((n_dims, n_steps), dtype=dt)
    
    # R matrix: current excitation from each j to each i
    cdef np.ndarray[DTYPE_t, ndim=2] R = np.zeros((n_dims, n_dims), dtype=dt)
    
    cdef DTYPE_t current_time = 0.0
    cdef DTYPE_t t, dt
    cdef int t_idx, i, j, dim
    cdef DTYPE_t decay_factor
    
    # Create unified event list
    all_events = []
    for dim in range(n_dims):
        for t in events[dim]:
            all_events.append((t, dim))
    all_events.sort(key=lambda x: x[0])
    
    cdef int event_idx = 0
    cdef int n_events = len(all_events)
    
    for t_idx in range(n_steps):
        t = times[t_idx]
        
        # Decay R since last time point
        dt = t - current_time
        if dt > 0:
            for i in range(n_dims):
                for j in range(n_dims):
                    R[i, j] *= exp(-beta[i, j] * dt)
        
        current_time = t
        
        # Process all events at this time
        while event_idx < n_events and all_events[event_idx][0] <= t:
            _, dim = all_events[event_idx]
            for i in range(n_dims):
                R[i, dim] += alpha[i, dim]
            event_idx += 1
        
        # Compute intensity: μ + Σ_j R_ij
        for i in range(n_dims):
            intensities[i, t_idx] = mu[i]
            for j in range(n_dims):
                intensities[i, t_idx] += R[i, j]
    
    return times, intensities

def sum_exp_kernel_eval(
    np.ndarray[DTYPE_t, ndim=1] t,
    DTYPE_t alpha1, DTYPE_t beta1,
    DTYPE_t alpha2, DTYPE_t beta2
):
    """Evaluate sum of two exponentials kernel."""
    cdef int n = t.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=dt)
    cdef int i
    cdef DTYPE_t val
    
    for i in range(n):
        if t[i] >= 0:
            val = alpha1 * exp(-beta1 * t[i]) + alpha2 * exp(-beta2 * t[i])
            result[i] = val
    
    return result
