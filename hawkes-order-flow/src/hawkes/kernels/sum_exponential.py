"""Sum of exponentials kernel for Hawkes processes."""

import numpy as np
from typing import Optional
from .base import BaseKernel, _sum_exp_kernel_eval, _sum_exp_kernel_integral


class SumExponentialsKernel(BaseKernel):
    r"""Sum of two exponential kernels for multi-timescale excitation.
    
    This kernel captures both short-term and long-term memory effects:
        φ_ij(t) = α_ij^(1) * exp(-β_ij^(1) * t) + α_ij^(2) * exp(-β_ij^(2) * t)
    
    where:
        - Component 1 (fast): High β, captures immediate excitation
        - Component 2 (slow): Low β, captures persistent effects
    
    Branching ratio:
        B_ij = α_ij^(1)/β_ij^(1) + α_ij^(2)/β_ij^(2)
    
    The recursive computation for intensity still applies, but we need
    to maintain two R matrices (one per exponential component).
    
    Parameters:
        n_dims: Number of dimensions (event types)
        n_components: Number of exponential components (default 2)
    
    Example:
        >>> kernel = SumExponentialsKernel(n_dims=4)
        >>> # Parameters: [α1_flat, α2_flat, β1_flat, β2_flat]
        >>> params = kernel.initialize_params(events)
        >>> kernel.set_params(params)
    """
    
    def __init__(self, n_dims: int, n_components: int = 2):
        super().__init__(n_dims)
        self.n_components = n_components
        
        # Parameter layout: [α1, α2, ..., αK, β1, β2, ..., βK]
        # Each αk and βk is n_dims × n_dims
        n_kernel_params = n_dims * n_dims
        self._slices = []
        for k in range(n_components):
            self._slices.append(slice(k * n_kernel_params, (k + 1) * n_kernel_params))
        for k in range(n_components):
            self._slices.append(
                slice(
                    (n_components + k) * n_kernel_params,
                    (n_components + k + 1) * n_kernel_params
                )
            )
    
    @property
    def n_params(self) -> int:
        """Total parameters: 2 * n_components * n_dims²."""
        return 2 * self.n_components * self.n_dims * self.n_dims
    
    @property
    def param_names(self) -> list[str]:
        """Generate parameter names."""
        names = []
        # Alpha parameters for each component
        for k in range(self.n_components):
            for i in range(self.n_dims):
                for j in range(self.n_dims):
                    names.append(f"alpha{k+1}_{i}{j}")
        # Beta parameters for each component
        for k in range(self.n_components):
            for i in range(self.n_dims):
                for j in range(self.n_dims):
                    names.append(f"beta{k+1}_{i}{j}")
        return names
    
    def _get_component_params(
        self, 
        params: np.ndarray, 
        component: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract α and β matrices for a specific component."""
        alpha_slice = self._slices[component]
        beta_slice = self._slices[self.n_components + component]
        alpha = params[alpha_slice].reshape(self.n_dims, self.n_dims)
        beta = params[beta_slice].reshape(self.n_dims, self.n_dims)
        return alpha, beta
    
    def evaluate(
        self, 
        t: np.ndarray, 
        params: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r"""Evaluate sum of exponentials kernel.
        
        Args:
            t: Time differences, shape (n,)
            params: Flattened parameters. If None, uses fitted.
            
        Returns:
            Kernel values, shape (n_dims, n_dims, n)
        """
        if params is None:
            params = self._params
        if params is None:
            raise ValueError("Parameters not set")
        
        n = len(t)
        result = np.zeros((self.n_dims, self.n_dims, n))
        
        # Sum contributions from each component
        for k in range(self.n_components):
            alpha, beta = self._get_component_params(params, k)
            for i in range(self.n_dims):
                for j in range(self.n_dims):
                    result[i, j, :] += _exp_kernel_eval(t, alpha[i, j], beta[i, j])
        
        return result
    
    def integrate(self, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute branching ratio matrix (sum of integrals)."""
        if params is None:
            params = self._params
        if params is None:
            raise ValueError("Parameters not set")
        
        B = np.zeros((self.n_dims, self.n_dims))
        
        for k in range(self.n_components):
            alpha, beta = self._get_component_params(params, k)
            for i in range(self.n_dims):
                for j in range(self.n_dims):
                    B[i, j] += _exp_kernel_integral(alpha[i, j], beta[i, j])
        
        return B
    
    def initialize_params(self, events: list[np.ndarray]) -> np.ndarray:
        """Initialize parameters with different timescales.
        
        Strategy:
        - Component 1 (fast): β high, α moderate
        - Component 2 (slow): β low, α small but persistent
        
        Args:
            events: List of event times for each dimension
            
        Returns:
            Initial parameter vector
        """
        n_dims = len(events)
        n_per_comp = n_dims * n_dims
        
        all_params = []
        
        # Compute mean inter-event times for scaling
        mean_dts = []
        for j in range(n_dims):
            if len(events[j]) > 1:
                mean_dts.append(np.mean(np.diff(events[j])))
            else:
                mean_dts.append(1.0)
        avg_dt = np.mean(mean_dts) if mean_dts else 1.0
        
        # Initialize alphas for each component
        for k in range(self.n_components):
            if k == 0:
                # Fast component: moderate alpha
                alpha = np.random.uniform(0.05, 0.15, (n_dims, n_dims))
            else:
                # Slow component: smaller alpha but more persistent
                alpha = np.random.uniform(0.02, 0.08, (n_dims, n_dims))
            all_params.append(alpha.flatten())
        
        # Initialize betas for each component
        for k in range(self.n_components):
            if k == 0:
                # Fast decay (high β)
                beta = np.full((n_dims, n_dims), 1.0 / avg_dt)
            else:
                # Slow decay (low β) - 10x slower
                beta = np.full((n_dims, n_dims), 0.1 / avg_dt)
            all_params.append(beta.flatten())
        
        return np.concatenate(all_params)
    
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Parameter bounds for optimization."""
        n = self.n_params
        # All α >= 0, all β > 0
        lower = np.zeros(n)
        # Beta parameters start after all alphas
        n_alpha = self.n_components * self.n_dims * self.n_dims
        lower[n_alpha:] = 1e-6
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
        r"""Compute intensities with multiple exponential components.
        
        Maintains separate R matrices for each component:
            λ_i(t) = μ_i + Σ_k Σ_j R_ij^k(t)
        
        Args:
            events: List of event times for each dimension
            mu: Baseline intensities
            params: Kernel parameters
            end_time: End of observation
            resolution: Time resolution
            
        Returns:
            (times, intensities)
        """
        # Get parameters for each component
        alphas = []
        betas = []
        for k in range(self.n_components):
            a, b = self._get_component_params(params, k)
            alphas.append(a)
            betas.append(b)
        
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
        
        # R matrices for each component: R[k][i, j]
        Rs = [np.zeros((self.n_dims, self.n_dims)) for _ in range(self.n_components)]
        current_time = times[0]
        event_idx = 0
        
        for t_idx, t in enumerate(times):
            dt = t - current_time
            if dt > 0:
                for k in range(self.n_components):
                    Rs[k] *= np.exp(-betas[k] * dt)
            current_time = t
            
            # Process events
            while event_idx < len(all_events) and all_events[event_idx][0] <= t:
                _, j = all_events[event_idx]
                for k in range(self.n_components):
                    Rs[k][:, j] += alphas[k][:, j]
                event_idx += 1
            
            # Sum over components
            R_total = sum(Rs)
            intensities[:, t_idx] = mu + R_total.sum(axis=1)
        
        return times, intensities
    
    def get_timescale_importance(
        self, 
        params: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute relative importance of each timescale component.
        
        Returns the proportion of total branching ratio contributed
        by each component: B_k / B_total
        
        Args:
            params: Kernel parameters
            
        Returns:
            Importance array, shape (n_components,)
        """
        if params is None:
            params = self._params
        
        B_by_component = []
        for k in range(self.n_components):
            alpha, beta = self._get_component_params(params, k)
            B_k = np.zeros((self.n_dims, self.n_dims))
            for i in range(self.n_dims):
                for j in range(self.n_dims):
                    B_k[i, j] = _exp_kernel_integral(alpha[i, j], beta[i, j])
            B_by_component.append(B_k.sum())
        
        B_total = sum(B_by_component)
        if B_total == 0:
            return np.ones(self.n_components) / self.n_components
        
        return np.array(B_by_component) / B_total
