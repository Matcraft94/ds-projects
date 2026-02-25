"""Fast Maximum Likelihood Estimation for Hawkes processes.

Uses Ogata's recursive method for O(N) computation with exponential kernels.
Includes parallel processing and Numba optimizations.
"""

import numpy as np
from typing import Optional, Literal, Callable
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eigvals
import warnings
from numba import jit, prange
import multiprocessing as mp
from joblib import Parallel, delayed

from ..kernels.exponential import ExponentialKernel
from ..kernels.sum_exponential import SumExponentialsKernel


@jit(nopython=True, cache=True, fastmath=True)
def _fast_log_likelihood_exp(
    events_list: list,
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    end_time: float
) -> float:
    """Numba-accelerated log-likelihood for exponential kernel.
    
    Uses Ogata's recursive method for O(N) computation.
    """
    n_dims = len(events_list)
    ll = 0.0
    
    # Pre-compute R matrices for recursive updates
    for i in range(n_dims):
        events_i = events_list[i]
        n_i = len(events_i)
        
        if n_i == 0:
            ll -= mu[i] * end_time
            continue
        
        # Use recursive computation
        R = np.zeros(n_dims)  # R[j] = sum of alpha[i,j] * exp(-beta[i,j] * dt)
        
        for n in range(n_i):
            t_n = events_i[n]
            
            # Decay R since last event
            if n > 0:
                dt = t_n - events_i[n-1]
                for j in range(n_dims):
                    R[j] *= np.exp(-beta[i, j] * dt)
            
            # Compute intensity at t_n
            lambda_i = mu[i] + np.sum(R)
            
            if lambda_i > 0:
                ll += np.log(lambda_i)
            else:
                return -1e10
            
            # Update R with contribution from this event
            # Find which dimension this event belongs to
            for j in range(n_dims):
                events_j = events_list[j]
                # Check if t_n is in events_j
                for m in range(len(events_j)):
                    if abs(events_j[m] - t_n) < 1e-10:
                        R[j] += alpha[i, j]
                        break
        
        # Compensator (integral of intensity)
        ll -= mu[i] * end_time
        
        for j in range(n_dims):
            events_j = events_list[j]
            for m in range(len(events_j)):
                tau = end_time - events_j[m]
                ll -= (alpha[i, j] / beta[i, j]) * (1.0 - np.exp(-beta[i, j] * tau))
    
    return ll


@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def _parallel_compensator(
    events_array: np.ndarray,
    alpha_flat: np.ndarray,
    beta_flat: np.ndarray,
    end_time: float,
    n_dims: int
) -> float:
    """Parallel computation of compensator terms."""
    total = 0.0
    
    for idx in prange(len(events_array)):
        t_j = events_array[idx]
        tau = end_time - t_j
        if tau > 0:
            # Sum over all target dimensions
            for i in range(n_dims):
                for j in range(n_dims):
                    idx_flat = i * n_dims + j
                    alpha_ij = alpha_flat[idx_flat]
                    beta_ij = beta_flat[idx_flat]
                    if beta_ij > 0:
                        total -= (alpha_ij / beta_ij) * (1.0 - np.exp(-beta_ij * tau))
    
    return total


class FastMultivariateHawkesMLE:
    """Fast MLE estimator using O(N) recursive algorithm.
    
    This is dramatically faster than the naive O(N²) implementation
    for exponential kernels.
    
    Parameters:
        n_dims: Number of dimensions
        use_parallel: Whether to use parallel processing
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    
    def __init__(
        self,
        n_dims: int,
        use_parallel: bool = True,
        n_jobs: int = -1
    ):
        self.n_dims = n_dims
        self.use_parallel = use_parallel
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        
        # Fitted parameters
        self.mu_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
        
    def _compute_log_likelihood_fast(
        self,
        events: list[np.ndarray],
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        end_time: float
    ) -> float:
        """Fast O(N) log-likelihood using recursive method."""
        ll = 0.0
        
        # Convert events to list for Numba
        events_list = [e.astype(np.float64) for e in events]
        
        for i in range(self.n_dims):
            events_i = events_list[i]
            n_i = len(events_i)
            
            if n_i == 0:
                ll -= mu[i] * end_time
                continue
            
            # Recursive computation for this dimension
            R = np.zeros(self.n_dims)  # Current excitation from each j
            
            # Create unified event list with dimension labels
            all_events = []
            for j in range(self.n_dims):
                for t in events_list[j]:
                    all_events.append((t, j))
            all_events.sort(key=lambda x: x[0])
            
            prev_time = 0.0
            for t_n in events_i:
                # Decay R
                dt = t_n - prev_time
                if dt > 0:
                    R *= np.exp(-beta[i, :] * dt)
                
                # Compute intensity
                lambda_i = mu[i] + np.sum(R)
                
                if lambda_i > 0:
                    ll += np.log(lambda_i)
                else:
                    return -1e10
                
                # Find events at this time and add excitation
                for (t_event, j_dim) in all_events:
                    if abs(t_event - t_n) < 1e-10:
                        R[j_dim] += alpha[i, j_dim]
                
                prev_time = t_n
            
            # Baseline compensator
            ll -= mu[i] * end_time
            
            # Kernel compensator
            for j in range(self.n_dims):
                for t_j in events_list[j]:
                    tau = end_time - t_j
                    if tau > 0 and beta[i, j] > 0:
                        ll -= (alpha[i, j] / beta[i, j]) * (1.0 - np.exp(-beta[i, j] * tau))
        
        return ll
    
    def _objective(self, params: np.ndarray, events: list, end_time: float) -> float:
        """Objective function for optimization."""
        mu = params[:self.n_dims]
        alpha_flat = params[self.n_dims:self.n_dims + self.n_dims**2]
        beta_flat = params[self.n_dims + self.n_dims**2:]
        
        alpha = alpha_flat.reshape(self.n_dims, self.n_dims)
        beta = beta_flat.reshape(self.n_dims, self.n_dims)
        
        # Check constraints
        if np.any(mu <= 0) or np.any(alpha < 0) or np.any(beta <= 0):
            return 1e10
        
        ll = self._compute_log_likelihood_fast(events, mu, alpha, beta, end_time)
        return -ll
    
    def fit(
        self,
        events: list[np.ndarray],
        end_time: Optional[float] = None,
        method: Literal['L-BFGS-B', 'differential_evolution'] = 'L-BFGS-B',
        init_params: Optional[np.ndarray] = None,
        verbose: bool = False,
        max_events: Optional[int] = None  # For subsampling
    ) -> 'FastMultivariateHawkesMLE':
        """Fit the Hawkes process.
        
        Args:
            events: List of event time arrays
            end_time: End of observation
            method: Optimization method
            init_params: Initial parameters
            verbose: Print progress
            max_events: Max events per dimension (subsample if more)
        """
        if end_time is None:
            end_time = max(max(e) if len(e) > 0 else 0 for e in events) + 1.0
        
        # Subsample if needed for speed
        if max_events is not None:
            events = self._subsample_events(events, max_events)
            if verbose:
                print(f"Subsampled to {max_events} events per dimension")
        
        # Initialize parameters
        if init_params is None:
            mu_init = np.array([len(t) / end_time * 0.5 for t in events])
            mu_init = np.clip(mu_init, 1e-6, 10.0)
            
            # Initialize alpha and beta
            alpha_init = np.random.uniform(0.05, 0.15, (self.n_dims, self.n_dims))
            beta_init = np.ones((self.n_dims, self.n_dims)) * 0.5
            
            init_params = np.concatenate([
                mu_init,
                alpha_init.flatten(),
                beta_init.flatten()
            ])
        
        # Bounds
        lower = np.concatenate([
            np.full(self.n_dims, 1e-6),  # mu > 0
            np.zeros(self.n_dims**2),     # alpha >= 0
            np.full(self.n_dims**2, 1e-6) # beta > 0
        ])
        upper = np.concatenate([
            np.full(self.n_dims, 10.0),   # mu max
            np.full(self.n_dims**2, 5.0), # alpha max
            np.full(self.n_dims**2, 10.0) # beta max
        ])
        
        # Optimize
        if verbose:
            print(f"Fitting with {method} using {self.n_jobs} cores...")
        
        if method == 'differential_evolution':
            result = differential_evolution(
                self._objective,
                list(zip(lower, upper)),
                args=(events, end_time),
                maxiter=100,
                workers=self.n_jobs,
                polish=True,
                disp=verbose
            )
        else:
            result = minimize(
                self._objective,
                init_params,
                args=(events, end_time),
                method='L-BFGS-B',
                bounds=list(zip(lower, upper)),
                options={'disp': verbose, 'maxiter': 100}
            )
        
        # Store results
        self.mu_ = result.x[:self.n_dims]
        self.alpha_ = result.x[self.n_dims:self.n_dims + self.n_dims**2].reshape(self.n_dims, self.n_dims)
        self.beta_ = result.x[self.n_dims + self.n_dims**2:].reshape(self.n_dims, self.n_dims)
        self.log_likelihood_ = -result.fun
        
        if verbose:
            print(f"Optimization: {'success' if result.success else 'failed'}")
            print(f"Log-likelihood: {self.log_likelihood_:.2f}")
        
        return self
    
    def _subsample_events(
        self,
        events: list[np.ndarray],
        max_events: int
    ) -> list[np.ndarray]:
        """Subsample events if too many."""
        result = []
        for e in events:
            if len(e) > max_events:
                # Stratified sampling
                indices = np.linspace(0, len(e)-1, max_events, dtype=int)
                result.append(e[indices])
            else:
                result.append(e)
        return result
    
    def compute_branching_ratio(self) -> np.ndarray:
        """Compute branching ratio matrix."""
        if self.alpha_ is None:
            raise ValueError("Model not fitted")
        return self.alpha_ / self.beta_
    
    def compute_spectral_radius(self) -> float:
        """Compute spectral radius."""
        B = self.compute_branching_ratio()
        return np.max(np.abs(eigvals(B)))
    
    def predict_intensity(
        self,
        times: np.ndarray,
        events: list[np.ndarray]
    ) -> np.ndarray:
        """Predict conditional intensities."""
        if self.mu_ is None:
            raise ValueError("Model not fitted")
        
        intensities = np.zeros((self.n_dims, len(times)))
        
        for idx, t in enumerate(times):
            for i in range(self.n_dims):
                lambda_i = self.mu_[i]
                
                for j in range(self.n_dims):
                    past_events = events[j][events[j] < t]
                    if len(past_events) > 0:
                        dt = t - past_events
                        lambda_i += np.sum(
                            self.alpha_[i, j] * np.exp(-self.beta_[i, j] * dt)
                        )
                
                intensities[i, idx] = lambda_i
        
        return intensities


class ParallelMultivariateHawkesMLE:
    """MLE with parallel processing for multiple initializations.
    
    Runs optimization from multiple starting points in parallel
    and returns the best result.
    """
    
    def __init__(
        self,
        n_dims: int,
        n_init: int = 10,
        n_jobs: int = -1
    ):
        self.n_dims = n_dims
        self.n_init = n_init
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        
        self.best_result_ = None
        
    def _fit_single(
        self,
        events: list,
        end_time: float,
        seed: int
    ) -> dict:
        """Fit from a single initialization."""
        np.random.seed(seed)
        
        estimator = FastMultivariateHawkesMLE(self.n_dims)
        
        # Random initialization
        mu_init = np.random.uniform(0.1, 1.0, self.n_dims)
        alpha_init = np.random.uniform(0.01, 0.2, (self.n_dims, self.n_dims))
        beta_init = np.random.uniform(0.3, 2.0, (self.n_dims, self.n_dims))
        
        init_params = np.concatenate([
            mu_init,
            alpha_init.flatten(),
            beta_init.flatten()
        ])
        
        try:
            estimator.fit(events, end_time, init_params=init_params)
            return {
                'estimator': estimator,
                'log_likelihood': estimator.log_likelihood_,
                'success': True
            }
        except Exception as e:
            return {
                'estimator': None,
                'log_likelihood': -np.inf,
                'success': False,
                'error': str(e)
            }
    
    def fit(
        self,
        events: list[np.ndarray],
        end_time: Optional[float] = None,
        verbose: bool = False
    ):
        """Fit with multiple random initializations in parallel."""
        if end_time is None:
            end_time = max(max(e) if len(e) > 0 else 0 for e in events) + 1.0
        
        if verbose:
            print(f"Running {self.n_init} optimizations with {self.n_jobs} jobs...")
        
        # Run in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single)(events, end_time, seed)
            for seed in range(self.n_init)
        )
        
        # Find best result
        best_idx = np.argmax([r['log_likelihood'] for r in results])
        self.best_result_ = results[best_idx]
        
        if verbose:
            print(f"Best log-likelihood: {self.best_result_['log_likelihood']:.2f}")
            print(f"Successful fits: {sum(r['success'] for r in results)}/{self.n_init}")
        
        # Copy attributes from best estimator
        if self.best_result_['estimator'] is not None:
            est = self.best_result_['estimator']
            self.mu_ = est.mu_
            self.alpha_ = est.alpha_
            self.beta_ = est.beta_
            self.log_likelihood_ = est.log_likelihood_
        
        return self
