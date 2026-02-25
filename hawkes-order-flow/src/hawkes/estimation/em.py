"""Expectation-Maximization algorithm for multivariate Hawkes processes.

Reference:
    Lewis, E., & Mohler, G. (2011). A nonparametric EM algorithm for 
    multiscale Hawkes processes. Journal of Nonparametric Statistics.
"""

import numpy as np
from typing import Optional, Literal
from scipy.linalg import eigvals
import warnings

from ..kernels.base import BaseKernel
from ..kernels.exponential import ExponentialKernel


class MultivariateHawkesEM:
    r"""Expectation-Maximization estimator for multivariate Hawkes processes.
    
    The EM algorithm is more robust to initialization than MLE and provides
    a principled way to handle the latent variable structure (which past
    events caused each new event).
    
    For exponential kernels, the EM updates have closed form:
    
    E-step: Compute expected number of offspring and expected triggering times
    M-step: Update parameters using expected sufficient statistics
    
    Parameters:
        kernel: Kernel instance
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        verbose: Print progress
    
    Example:
        >>> from hawkes.kernels import ExponentialKernel
        >>> kernel = ExponentialKernel(n_dims=4)
        >>> estimator = MultivariateHawkesEM(kernel, max_iter=100)
        >>> estimator.fit(events)
    """
    
    def __init__(
        self,
        kernel: BaseKernel,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False
    ):
        self.kernel = kernel
        self.n_dims = kernel.n_dims
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        # Fitted parameters
        self.mu_: Optional[np.ndarray] = None
        self.kernel_params_: Optional[np.ndarray] = None
        self.log_likelihood_: Optional[float] = None
        self.history_: list[dict] = []
        
    def _compute_responsibilities(
        self,
        events: list[np.ndarray],
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        end_time: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """E-step: Compute responsibilities (posterior probabilities).
        
        For each event, compute the probability that it was triggered by:
        - Background (baseline intensity)
        - Each previous event of each dimension
        
        Returns:
            p_background: Probability each event is background, shape (n_dims, max_events)
            p_trigger: Probability each event triggers future events, shape (n_dims, n_dims, max_events, max_events)
            delta_t: Time differences for triggered events
        """
        max_events = max(len(t) for t in events)
        
        p_background = []
        p_trigger = np.zeros((self.n_dims, self.n_dims, max_events, max_events))
        
        for i in range(self.n_dims):
            p_bg_i = []
            for n, t_n in enumerate(events[i]):
                # Compute intensity at t_n
                lambda_i = mu[i]
                
                # Contribution from past events
                trigger_probs = []
                for j in range(self.n_dims):
                    for t_m in events[j]:
                        if t_m >= t_n:
                            break
                        dt = t_n - t_m
                        contrib = alpha[i, j] * np.exp(-beta[i, j] * dt)
                        lambda_i += contrib
                        trigger_probs.append((j, t_m, contrib))
                
                # Background probability
                p_bg = mu[i] / lambda_i if lambda_i > 0 else 1.0
                p_bg_i.append(p_bg)
                
                # Trigger probabilities
                for j, t_m, contrib in trigger_probs:
                    m_idx = np.where(events[j] == t_m)[0]
                    if len(m_idx) > 0:
                        m = m_idx[0]
                        if m < max_events and n < max_events:
                            p_trigger[i, j, n, m] = contrib / lambda_i
            
            p_background.append(np.array(p_bg_i))
        
        return p_background, p_trigger, None
    
    def _m_step_update(
        self,
        events: list[np.ndarray],
        p_background: list[np.ndarray],
        p_trigger: np.ndarray,
        end_time: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M-step: Update parameters given responsibilities.
        
        Returns:
            Updated (mu, alpha, beta)
        """
        # Update mu: expected number of background events / T
        mu_new = np.array([
            np.sum(p_bg) / end_time for p_bg in p_background
        ])
        
        # Clip to avoid zero
        mu_new = np.maximum(mu_new, 1e-6)
        
        # Update alpha and beta
        alpha_new = np.zeros((self.n_dims, self.n_dims))
        beta_new = np.zeros((self.n_dims, self.n_dims))
        
        for i in range(self.n_dims):
            for j in range(self.n_dims):
                # Expected number of triggered events
                expected_triggered = np.sum(p_trigger[i, j, :, :])
                
                if expected_triggered > 0:
                    # Update alpha: expected triggered / expected survival
                    # For exponential kernel, this simplifies to expected_triggered / R_sum
                    
                    # Compute expected time differences
                    weighted_dt_sum = 0.0
                    for n in range(len(events[i])):
                        for m in range(len(events[j])):
                            if events[j][m] < events[i][n]:
                                dt = events[i][n] - events[j][m]
                                weighted_dt_sum += p_trigger[i, j, n, m] * dt
                    
                    # Heuristic update (simplified)
                    alpha_new[i, j] = expected_triggered / len(events[j]) if len(events[j]) > 0 else 0.1
                    
                    # Beta update: expected_triggered / weighted_dt_sum
                    if weighted_dt_sum > 0:
                        beta_new[i, j] = expected_triggered / weighted_dt_sum
                    else:
                        beta_new[i, j] = 1.0
                else:
                    alpha_new[i, j] = 0.01
                    beta_new[i, j] = 1.0
        
        return mu_new, alpha_new, beta_new
    
    def _compute_log_likelihood(
        self,
        events: list[np.ndarray],
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        end_time: float
    ) -> float:
        """Compute log-likelihood for current parameters."""
        ll = 0.0
        
        for i in range(self.n_dims):
            # Sum of log intensities at event times
            for t_i in events[i]:
                lambda_i = mu[i]
                for j in range(self.n_dims):
                    for t_j in events[j]:
                        if t_j >= t_i:
                            break
                        dt = t_i - t_j
                        lambda_i += alpha[i, j] * np.exp(-beta[i, j] * dt)
                
                if lambda_i > 0:
                    ll += np.log(lambda_i)
            
            # Subtract compensator
            # Baseline
            ll -= mu[i] * end_time
            
            # Kernel contributions
            for j in range(self.n_dims):
                for t_j in events[j]:
                    tau = end_time - t_j
                    # ∫_0^τ α*exp(-β*t) dt = (α/β)*(1 - exp(-β*τ))
                    ll -= (alpha[i, j] / beta[i, j]) * (1 - np.exp(-beta[i, j] * tau))
        
        return ll
    
    def fit(
        self,
        events: list[np.ndarray],
        end_time: Optional[float] = None,
        init_params: Optional[tuple] = None
    ) -> 'MultivariateHawkesEM':
        """Fit the Hawkes process using EM algorithm.
        
        Args:
            events: List of event time arrays
            end_time: End of observation period
            init_params: Optional (mu, alpha, beta) tuple for initialization
            
        Returns:
            self
        """
        if len(events) != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} event arrays")
        
        if end_time is None:
            end_time = max(max(t) if len(t) > 0 else 0 for t in events) + 1.0
        
        # Initialize parameters
        if init_params is None:
            mu = np.array([len(t) / end_time * 0.5 for t in events])
            kernel_params = self.kernel.initialize_params(events)
            # Extract alpha and beta for exponential kernel
            if isinstance(self.kernel, ExponentialKernel):
                alpha = kernel_params[:self.n_dims**2].reshape(self.n_dims, self.n_dims)
                beta = kernel_params[self.n_dims**2:].reshape(self.n_dims, self.n_dims)
            else:
                # For other kernels, use simple initialization
                alpha = np.random.uniform(0.05, 0.1, (self.n_dims, self.n_dims))
                beta = np.random.uniform(0.5, 2.0, (self.n_dims, self.n_dims))
        else:
            mu, alpha, beta = init_params
        
        # EM iterations
        prev_ll = -np.inf
        self.history_ = []
        
        for iteration in range(self.max_iter):
            # E-step
            p_background, p_trigger, _ = self._compute_responsibilities(
                events, mu, alpha, beta, end_time
            )
            
            # M-step
            mu_new, alpha_new, beta_new = self._m_step_update(
                events, p_background, p_trigger, end_time
            )
            
            # Ensure positivity
            mu_new = np.maximum(mu_new, 1e-6)
            alpha_new = np.maximum(alpha_new, 1e-6)
            beta_new = np.maximum(beta_new, 1e-6)
            
            # Compute log-likelihood
            ll = self._compute_log_likelihood(events, mu_new, alpha_new, beta_new, end_time)
            
            # Store history
            self.history_.append({
                'iteration': iteration,
                'log_likelihood': ll,
                'mu': mu_new.copy(),
                'alpha': alpha_new.copy(),
                'beta': beta_new.copy()
            })
            
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: LL = {ll:.4f}")
            
            # Check convergence
            if abs(ll - prev_ll) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Update parameters
            mu, alpha, beta = mu_new, alpha_new, beta_new
            prev_ll = ll
        
        else:
            if self.verbose:
                print(f"Reached max iterations ({self.max_iter})")
        
        # Store final parameters
        self.mu_ = mu
        self.alpha_ = alpha
        self.beta_ = beta
        
        # Convert to kernel parameter format
        if isinstance(self.kernel, ExponentialKernel):
            self.kernel_params_ = np.concatenate([alpha.flatten(), beta.flatten()])
            self.kernel.set_params(self.kernel_params_)
        
        self.log_likelihood_ = prev_ll
        
        return self
    
    def compute_branching_ratio(self) -> np.ndarray:
        """Compute branching ratio matrix."""
        if not hasattr(self, 'alpha_') or not hasattr(self, 'beta_'):
            raise ValueError("Model not fitted yet")
        return self.alpha_ / self.beta_
    
    def compute_spectral_radius(self) -> float:
        """Compute spectral radius."""
        B = self.compute_branching_ratio()
        eigenvalues = eigvals(B)
        return np.max(np.abs(eigenvalues))
