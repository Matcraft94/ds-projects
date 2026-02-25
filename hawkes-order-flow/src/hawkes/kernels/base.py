"""Base kernel classes for Hawkes processes."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from numba import jit


class BaseKernel(ABC):
    """Abstract base class for Hawkes excitation kernels.
    
    A kernel defines the excitation function φ(t) that determines how
    past events influence the current conditional intensity.
    
    For a multivariate Hawkes process with d dimensions:
        λ_i(t) = μ_i + Σ_j ∫ φ_ij(t-s) dN_j(s)
    
    where φ_ij is the kernel from dimension j to dimension i.
    """
    
    def __init__(self, n_dims: int):
        """Initialize kernel.
        
        Args:
            n_dims: Number of dimensions (event types)
        """
        self.n_dims = n_dims
        self._params: Optional[np.ndarray] = None
        
    @property
    @abstractmethod
    def n_params(self) -> int:
        """Total number of parameters for this kernel."""
        pass
    
    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Human-readable parameter names."""
        pass
    
    @abstractmethod
    def evaluate(
        self, 
        t: np.ndarray, 
        params: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Evaluate kernel at times t.
        
        Args:
            t: Time differences (non-negative), shape (n,)
            params: Kernel parameters. If None, uses fitted params.
            
        Returns:
            Kernel values, shape (n_dims, n_dims, n)
        """
        pass
    
    @abstractmethod
    def integrate(self, params: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute integral of kernel from 0 to ∞ (branching ratio matrix).
        
        For kernel φ(t), computes B_ij = ∫_0^∞ φ_ij(t) dt
        
        Args:
            params: Kernel parameters. If None, uses fitted params.
            
        Returns:
            Branching ratio matrix, shape (n_dims, n_dims)
        """
        pass
    
    @abstractmethod
    def initialize_params(self, events: list[np.ndarray]) -> np.ndarray:
        """Initialize parameters from event data.
        
        Args:
            events: List of event times for each dimension
            
        Returns:
            Initial parameter vector
        """
        pass
    
    def set_params(self, params: np.ndarray) -> None:
        """Set kernel parameters."""
        if params.shape[0] != self.n_params:
            raise ValueError(f"Expected {self.n_params} params, got {params.shape[0]}")
        self._params = params.copy()
    
    def get_params(self) -> Optional[np.ndarray]:
        """Get current parameters."""
        return self._params


@jit(nopython=True, cache=True)
def _exp_kernel_eval(t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Numba-accelerated exponential kernel evaluation.
    
    φ(t) = α * exp(-β * t) for t >= 0, 0 otherwise
    """
    result = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] >= 0:
            result[i] = alpha * np.exp(-beta * t[i])
    return result


@jit(nopython=True, cache=True)
def _exp_kernel_integral(alpha: float, beta: float) -> float:
    """Integral of exponential kernel: ∫_0^∞ α*exp(-β*t) dt = α/β"""
    if beta <= 0:
        return np.inf
    return alpha / beta


@jit(nopython=True, cache=True)
def _sum_exp_kernel_eval(
    t: np.ndarray, 
    alpha1: float, beta1: float,
    alpha2: float, beta2: float
) -> np.ndarray:
    """Numba-accelerated sum of exponentials kernel."""
    result = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] >= 0:
            result[i] = (
                alpha1 * np.exp(-beta1 * t[i]) + 
                alpha2 * np.exp(-beta2 * t[i])
            )
    return result


@jit(nopython=True, cache=True)
def _sum_exp_kernel_integral(
    alpha1: float, beta1: float,
    alpha2: float, beta2: float
) -> float:
    """Integral of sum of exponentials kernel."""
    integral = 0.0
    if beta1 > 0:
        integral += alpha1 / beta1
    if beta2 > 0:
        integral += alpha2 / beta2
    return integral
