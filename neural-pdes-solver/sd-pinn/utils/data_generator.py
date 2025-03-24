"""
utils/data_generator.py
Path: utils/data_generator.py
Date: 23-Mar-2025
Author: Lucy
Description: Utilities for generating data points for Physics-Informed Neural Networks (PINNs).
             Includes functions to generate points in the domain, boundaries, and initial conditions.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
from torch.utils.data import Dataset, DataLoader


def generate_random_points(
    domain_bounds: Dict[str, Tuple[float, float]],
    n_points: int,
    sampling: str = "uniform",
    seed: Optional[int] = None,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Generate random points in the domain.
    
    Args:
        domain_bounds: Dictionary with domain bounds for each variable
        n_points: Number of points to generate
        sampling: Sampling method ('uniform', 'latin', 'sobol', 'halton', 'grid')
        seed: Random seed for reproducibility
        device: Device to store the points
        
    Returns:
        Dictionary with tensors for each variable
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Get number of dimensions
    n_dims = len(domain_bounds)
    dims = list(domain_bounds.keys())
    
    # Initialize points dictionary
    points = {}
    
    if sampling == "uniform":
        # Generate uniform random points
        for dim in dims:
            low, high = domain_bounds[dim]
            points[dim] = torch.FloatTensor(n_points, 1).uniform_(low, high).to(device)
    
    elif sampling == "latin":
        # Generate Latin hypercube samples
        try:
            from scipy.stats.qmc import LatinHypercube
            
            # Create sampler
            sampler = LatinHypercube(d=n_dims, seed=seed)
            
            # Generate samples in [0, 1]
            samples = sampler.random(n=n_points)
            
            # Transform to desired domain
            for i, dim in enumerate(dims):
                low, high = domain_bounds[dim]
                points[dim] = torch.FloatTensor(samples[:, i].reshape(-1, 1) * (high - low) + low).to(device)
        except ImportError:
            print("scipy not available, falling back to uniform sampling")
            for dim in dims:
                low, high = domain_bounds[dim]
                points[dim] = torch.FloatTensor(n_points, 1).uniform_(low, high).to(device)
    
    elif sampling == "sobol":
        # Generate Sobol sequence
        try:
            from scipy.stats.qmc import Sobol
            
            # Create sampler
            sampler = Sobol(d=n_dims, seed=seed)
            
            # Generate samples in [0, 1]
            samples = sampler.random(n=n_points)
            
            # Transform to desired domain
            for i, dim in enumerate(dims):
                low, high = domain_bounds[dim]
                points[dim] = torch.FloatTensor(samples[:, i].reshape(-1, 1) * (high - low) + low).to(device)
        except ImportError:
            print("scipy not available, falling back to uniform sampling")
            for dim in dims:
                low, high = domain_bounds[dim]
                points[dim] = torch.FloatTensor(n_points, 1).uniform_(low, high).to(device)
    
    elif sampling == "halton":
        # Generate Halton sequence
        try:
            from scipy.stats.qmc import Halton
            
            # Create sampler
            sampler = Halton(d=n_dims, seed=seed)
            
            # Generate samples in [0, 1]
            samples = sampler.random(n=n_points)
            
            # Transform to desired domain
            for i, dim in enumerate(dims):
                low, high = domain_bounds[dim]
                points[dim] = torch.FloatTensor(samples[:, i].reshape(-1, 1) * (high - low) + low).to(device)
        except ImportError:
            print("scipy not available, falling back to uniform sampling")
            for dim in dims:
                low, high = domain_bounds[dim]
                points[dim] = torch.FloatTensor(n_points, 1).uniform_(low, high).to(device)
    
    elif sampling == "grid":
        # Generate points on a grid
        # Calculate number of points per dimension
        n_per_dim = int(np.ceil(n_points**(1/n_dims)))
        
        # Generate grid points
        grid_points = []
        for dim in dims:
            low, high = domain_bounds[dim]
            grid_points.append(torch.linspace(low, high, n_per_dim))
        
        # Create meshgrid
        mesh = torch.meshgrid(*grid_points, indexing="ij")
        
        # Flatten and store in dictionary
        for i, dim in enumerate(dims):
            points[dim] = mesh[i].flatten()[:n_points].unsqueeze(1).to(device)
    
    else:
        raise ValueError(f"Sampling method {sampling} not supported")
    
    return points


def generate_boundary_points(
    domain_bounds: Dict[str, Tuple[float, float]],
    n_points_per_boundary: int,
    boundaries: List[Tuple[str, str]],
    sampling: str = "uniform",
    seed: Optional[int] = None,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Generate points on the domain boundaries.
    
    Args:
        domain_bounds: Dictionary with domain bounds for each variable
        n_points_per_boundary: Number of points per boundary
        boundaries: List of (dimension, bound) tuples, e.g., [("x", "min"), ("x", "max")]
        sampling: Sampling method ('uniform', 'latin', 'sobol', 'halton', 'grid')
        seed: Random seed for reproducibility
        device: Device to store the points
        
    Returns:
        Dictionary with tensors for each variable
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Get dimensions
    dims = list(domain_bounds.keys())
    
    # Initialize points dictionary
    all_points = {dim: [] for dim in dims}
    
    # Generate points for each boundary
    for dim, bound in boundaries:
        # Get bounds for boundary dimension
        low, high = domain_bounds[dim]
        bound_value = low if bound == "min" else high
        
        # Create dictionary for other dimensions
        other_dims = [d for d in dims if d != dim]
        other_bounds = {d: domain_bounds[d] for d in other_dims}
        
        # Generate random points for other dimensions
        if other_dims:
            other_points = generate_random_points(
                domain_bounds=other_bounds,
                n_points=n_points_per_boundary,
                sampling=sampling,
                seed=seed,
                device=device
            )
            
            # Add boundary dimension with fixed value
            for d in dims:
                if d == dim:
                    all_points[d].append(torch.ones(n_points_per_boundary, 1).to(device) * bound_value)
                else:
                    all_points[d].append(other_points[d])
        else:
            # Only one dimension (1D problem)
            all_points[dim].append(torch.ones(n_points_per_boundary, 1).to(device) * bound_value)
    
    # Concatenate points for all boundaries
    for dim in dims:
        if all_points[dim]:
            all_points[dim] = torch.cat(all_points[dim], dim=0)
        else:
            all_points[dim] = torch.empty(0, 1).to(device)
    
    return all_points


def generate_initial_points(
    domain_bounds: Dict[str, Tuple[float, float]],
    time_var: str,
    n_points: int,
    sampling: str = "uniform",
    seed: Optional[int] = None,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Generate points for initial conditions (t = 0).
    
    Args:
        domain_bounds: Dictionary with domain bounds for each variable
        time_var: Name of the time variable
        n_points: Number of points to generate
        sampling: Sampling method ('uniform', 'latin', 'sobol', 'halton', 'grid')
        seed: Random seed for reproducibility
        device: Device to store the points
        
    Returns:
        Dictionary with tensors for each variable
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Get spatial dimensions
    spatial_dims = [dim for dim in domain_bounds if dim != time_var]
    spatial_bounds = {dim: domain_bounds[dim] for dim in spatial_dims}
    
    # Generate random points in spatial domain
    spatial_points = generate_random_points(
        domain_bounds=spatial_bounds,
        n_points=n_points,
        sampling=sampling,
        seed=seed,
        device=device
    )
    
    # Get time bounds
    t_min, _ = domain_bounds[time_var]
    
    # Create points dictionary
    points = {}
    for dim in spatial_dims:
        points[dim] = spatial_points[dim]
    
    # Add time dimension with fixed value (t = t_min)
    points[time_var] = torch.ones(n_points, 1).to(device) * t_min
    
    return points


class PINNDataset(Dataset):
    """
    Dataset class for Physics-Informed Neural Networks.
    
    Handles batching of residual, boundary, and initial points for training.
    """
    
    def __init__(
        self,
        residual_points: Dict[str, torch.Tensor],
        boundary_points: Optional[Dict[str, torch.Tensor]] = None,
        initial_points: Optional[Dict[str, torch.Tensor]] = None,
        data_points: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            residual_points: Dictionary with tensors for residual points
            boundary_points: Dictionary with tensors for boundary points (optional)
            initial_points: Dictionary with tensors for initial points (optional)
            data_points: Dictionary with tensors for data points (optional)
            batch_size: Batch size for training (None for full-batch)
        """
        self.residual_points = residual_points
        self.boundary_points = boundary_points or {}
        self.initial_points = initial_points or {}
        self.data_points = data_points or {}
        
        # Get number of points for each type
        self.n_residual = len(next(iter(residual_points.values())))
        self.n_boundary = len(next(iter(boundary_points.values()))) if boundary_points else 0
        self.n_initial = len(next(iter(initial_points.values()))) if initial_points else 0
        self.n_data = len(next(iter(data_points.values()))) if data_points else 0
        
        # Set batch size
        if batch_size is None:
            # Use full batch
            self.batch_size = max(self.n_residual, self.n_boundary, self.n_initial, self.n_data)
        else:
            self.batch_size = batch_size
    
    def __len__(self):
        """Return the number of batches."""
        if self.batch_size is None or self.batch_size == 0:
            return 1
        else:
            return int(np.ceil(self.n_residual / self.batch_size))
    
    def __getitem__(self, idx):
        """Get a batch of points."""
        # Calculate batch indices
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_residual)
        
        # Create batch dictionary
        batch = {}
        
        # Add residual points
        if self.n_residual > 0:
            batch["residual"] = {k: v[start_idx:end_idx] for k, v in self.residual_points.items()}
        
        # Add boundary points (use all points or subsample to batch size)
        if self.n_boundary > 0:
            if self.n_boundary <= self.batch_size:
                batch["boundary"] = self.boundary_points
            else:
                boundary_indices = torch.randperm(self.n_boundary)[:self.batch_size]
                batch["boundary"] = {k: v[boundary_indices] for k, v in self.boundary_points.items()}
        
        # Add initial points (use all points or subsample to batch size)
        if self.n_initial > 0:
            if self.n_initial <= self.batch_size:
                batch["initial"] = self.initial_points
            else:
                initial_indices = torch.randperm(self.n_initial)[:self.batch_size]
                batch["initial"] = {k: v[initial_indices] for k, v in self.initial_points.items()}
        
        # Add data points (use all points or subsample to batch size)
        if self.n_data > 0:
            if self.n_data <= self.batch_size:
                batch["data"] = self.data_points
            else:
                data_indices = torch.randperm(self.n_data)[:self.batch_size]
                batch["data"] = {k: v[data_indices] for k, v in self.data_points.items()}
        
        return batch
    
    def get_dataloader(self, shuffle: bool = True):
        """
        Create a DataLoader for this dataset.
        
        Args:
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for this dataset
        """
        return DataLoader(
            dataset=self,
            batch_size=1,  # Each batch already contains multiple points
            shuffle=shuffle,
            collate_fn=lambda x: x[0]  # Return the batch directly
        )