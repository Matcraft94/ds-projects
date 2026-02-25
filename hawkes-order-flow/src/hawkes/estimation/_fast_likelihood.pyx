"""Cython-optimized likelihood computations."""

import numpy as np
cimport numpy as np
from libc.math cimport exp, log

dt = np.float64
ctypedef np.float64_t DTYPE_t

def compute_log_likelihood_exp(
    list events,
    np.ndarray[DTYPE_t, ndim=1] mu,
    np.ndarray[DTYPE_t, ndim=2] alpha,
    np.ndarray[DTYPE_t, ndim=2] beta,
    DTYPE_t end_time
):
    """Compute log-likelihood for exponential kernel Hawkes process.
    
    This is O(N²) but optimized in Cython.
    
    Args:
        events: List of event time arrays
        mu: Baseline intensities
        alpha: Excitation matrix
        beta: Decay matrix
        end_time: End of observation period
        
    Returns:
        Log-likelihood value
    """
    cdef int n_dims = len(events)
    cdef DTYPE_t ll = 0.0
    cdef DTYPE_t lambda_i, dt, contrib
    cdef int i, j, n, m
    cdef DTYPE_t t_i, t_j
    
    for i in range(n_dims):
        # Sum of log intensities at event times
        for n in range(len(events[i])):
            t_i = events[i][n]
            lambda_i = mu[i]
            
            for j in range(n_dims):
                for m in range(len(events[j])):
                    t_j = events[j][m]
                    if t_j >= t_i:
                        break
                    dt = t_i - t_j
                    contrib = alpha[i, j] * exp(-beta[i, j] * dt)
                    lambda_i += contrib
            
            if lambda_i > 0:
                ll += log(lambda_i)
        
        # Subtract compensator (baseline contribution)
        ll -= mu[i] * end_time
        
        # Kernel contributions to compensator
        for j in range(n_dims):
            for m in range(len(events[j])):
                t_j = events[j][m]
                # ∫_0^{T-t_j} α*exp(-β*t) dt = (α/β) * (1 - exp(-β*(T-t_j)))
                dt = end_time - t_j
                ll -= (alpha[i, j] / beta[i, j]) * (1 - exp(-beta[i, j] * dt))
    
    return ll

def compute_compensator_exp(
    list events,
    np.ndarray[DTYPE_t, ndim=1] mu,
    np.ndarray[DTYPE_t, ndim=2] alpha,
    np.ndarray[DTYPE_t, ndim=2] beta,
    DTYPE_t end_time
):
    """Compute compensator Λ(t) = ∫_0^t λ(s) ds.
    
    The compensator is used for goodness-of-fit testing (residual analysis).
    
    Returns:
        Compensator values at event times for each dimension
    """
    cdef int n_dims = len(events)
    cdef list compensators = []
    cdef DTYPE_t t_i, t_j, dt, comp
    cdef int i, j, n, m
    
    for i in range(n_dims):
        comp_values = []
        for n in range(len(events[i])):
            t_i = events[i][n]
            
            # Baseline contribution
            comp = mu[i] * t_i
            
            # Kernel contributions
            for j in range(n_dims):
                for m in range(len(events[j])):
                    t_j = events[j][m]
                    if t_j >= t_i:
                        break
                    dt = t_i - t_j
                    # ∫_0^{t_i-t_j} α*exp(-β*t) dt
                    comp += (alpha[i, j] / beta[i, j]) * (1 - exp(-beta[i, j] * dt))
            
            comp_values.append(comp)
        
        compensators.append(np.array(comp_values))
    
    return compensators
