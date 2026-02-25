"""Exponential kernel implementation for Hawkes processes."""

import numpy as np
from typing import Optional
from .base import BaseKernel, _exp_kernel_eval, _exp_kernel_integral


class ExponentialKernel(BaseKernel):
    r"""Exponential excitation kernel.
    
    The exponential kernel is defined as:
        φ_ij(t) = α_ij * exp(-β_ij * t) for t >= 0
    
    where:
        - α_ij >= 0 is the excitation magnitude (infectivity)
        - β_ij > 0 is the decay rate
    
    The branching ratio (total expected offspring) is:
        B_ij = α_ij / β_ij
    
    This kernel is computationally efficient because it allows O(N)
    recursive computation of conditional intensities using Ogata's method.
    
    Parameters:
        n_dims: Number of dimensions (event types)
    
    Example:
        >>> kernel = ExponentialKernel(n_dims=4)
        >>> params = np.array([0.5, 0.1, 1.0, ...])  # Flattened α and β
        >>> kernel.set_params(params)
        >>> values = kernel.evaluate(np.array([0.1, 0.2, 0.3]))
    """
    
    def __init__(self, n_dims: int):
        super().__init__(n_dims)
        self._alpha_slice = slice(0, n_dims * n_dims)
        self._beta_slice = slice(n_dims * n_dims, 2 * n_dims * n_dims)
    
    @property
    def n_params(self) -> int:
        """Total parameters: n_dims² for α + n_dims² for β."""
        return 2 * self.n_dims * self.n_dims
    
    @property
    def param_names(self) -> list[str]:
        """Generate parameter names."""
        names = []
        # Alpha parameters
        for i in range(self.n_dims):
            for j in range(self.n_dims):
                names.append(f"alpha_{i}{j}")
        # Beta parameters
        for i in range(self.n_dims):
            for j in range(self.n_dims):
                names.append(f"beta_{i}{j}")
        return names
    
    def _get_alpha_matrix(self, params: np.ndarray) -> np.ndarray:
        """Extract α matrix from parameters."""
        return params[self._alpha_slice].reshape(self.n_dims, self.n_dims)
    
    def _get_beta_matrix(self, params: np.ndarray) -> np.ndarray:
        """Extract β matrix from parameters."""
        return params[self._beta_slice].reshape(self.n_dims, self.n_dims)
    
    def evaluate(
        self, 
        t: np.ndarray, 
        params: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r"""Evaluate exponential kernel.
        
        Args:
            t: Time differences, shape (n,)
            params: Flattened parameters [α_flat, β_flat]. If None, uses fitted.
            
        Returns:
            Kernel values φ_ij(t) for all i,j, shape (n_dims, n_dims, n)
        """
        if params is None:
            params = self._params
        if params is None:
            raise ValueError("Parameters not set")
        
        alpha = self._get_alpha_matrix(params)
        beta = self._get_beta_matrix(params)
        
        n = len(t)
        result = np.zeros((self.n_dims, self.n_dims, n))
        
        for i in range(self.n_dims):
            for j in range(self.n_dims):
                result[i, j, :] = _exp_kernel_eval(t, alpha[i, j], beta[i, j])
        
        return result
    
    def integrate(self, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute branching ratio matrix B_ij = α_ij / β_ij."""
        if params is None:
            params = self._params
        if params is None:
            raise ValueError("Parameters not set")
        
        alpha = self._get_alpha_matrix(params)
        beta = self._get_beta_matrix(params)
        
        B = np.zeros((self.n_dims, self.n_dims))
        for i in range(self.n_dims):
            for j in range(self.n_dims):
                B[i, j] = _exp_kernel_integral(alpha[i, j], beta[i, j])
        
        return B
    
    def initialize_params(self, events: list[np.ndarray]) -> np.ndarray:
        """Initialize parameters from event data.
        
        Strategy:
        - α initialized to small random values (0.01-0.1)
        - β initialized based on typical inter-event times
        
        Args:
            events: List of event times arrays for each dimension
            
        Returns:
            Initial parameter vector
        """
        n_dims = len(events)
        
        # Initialize alpha small but positive
        alpha = np.random.uniform(0.01, 0.1, (n_dims, n_dims))
        
        # Initialize beta based on mean inter-event times
        beta = np.zeros((n_dims, n_dims))
        for j in range(n_dims):
            if len(events[j]) > 1:
                # Mean inter-event time for dimension j
                mean_dt = np.mean(np.diff(events[j]))
                # Set beta so that decay happens over ~5 mean intervals
                beta_val = 1.0 / (5.0 * mean_dt) if mean_dt > 0 else 1.0
            else:
                beta_val = 1.0
            beta[:, j] = beta_val
        
        return np.concatenate([alpha.flatten(), beta.flatten()])
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Parameter bounds for optimization.
        
        Returns:
            (lower_bounds, upper_bounds) tuple
        """
        n = self.n_params
        # α >= 0, β > 0
        lower = np.zeros(n)
        lower[self._beta_slice] = 1e-6  # β must be positive
        upper = np.full(n, np.inf)
        return lower, upper
    
    def compute_intensity_recursive(
        self,
        events: list[np.ndarray],
        mu: np.ndarray,
        params: np.ndarray,
        end_time: float,
        resolution: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Compute conditional intensities using recursive formula (Ogata 1981).
        
        For exponential kernels, we can compute intensity efficiently:
            λ_i(t) = μ_i + Σ_j R_ij(t)
        
        where R_ij decays exponentially between events:
            R_ij(t_{k+1}) = R_ij(t_k) * exp(-β_ij * (t_{k+1} - t_k))
        
        At events of type j:
            R_ij(t_k^+) = R_ij(t_k^-) + α_ij
        
        This is O(N) instead of O(N²).
        
        Args:
            events: List of event times for each dimension
            mu: Baseline intensities, shape (n_dims,)
            params: Kernel parameters
            end_time: End of observation period
            resolution: Time resolution for output
            
        Returns:
            (times, intensities) where intensities has shape (n_dims, n_times)
        """
        alpha = self._get_alpha_matrix(params)
        beta = self._get_beta_matrix(params)
        
        # Create unified timeline
        all_events = []
        for dim, times in enumerate(events):
            for t in times:
                all_events.append((t, dim))
        all_events.sort(key=lambda x: x[0])
        
        if not all_events:
            return np.array([0.0]), mu.reshape(-1, 1)
        
        # Initialize
        n_steps = int((end_time - all_events[0][0]) / resolution) + 1
        times = np.linspace(all_events[0][0], end_time, n_steps)
        intensities = np.zeros((self.n_dims, n_steps))
        
        # R matrix: current excitation from each j to each i
        R = np.zeros((self.n_dims, self.n_dims))
        current_time = times[0]
        event_idx = 0
        
        for t_idx, t in enumerate(times):
            # Decay R since last time point
            dt = t - current_time
            if dt > 0:
                R *= np.exp(-beta * dt)
            current_time = t
            
            # Process all events at this time
            while event_idx < len(all_events) and all_events[event_idx][0] <= t:
                _, j = all_events[event_idx]
                # Add excitation from event of type j to all i
                R[:, j] += alpha[:, j]
                event_idx += 1
            
            # Compute intensity: μ + Σ_j R_ij
            intensities[:, t_idx] = mu + R.sum(axis=1)
        
        return times, intensities
