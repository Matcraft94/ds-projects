"""
models/sa_pinn.py
Path: models/sa_pinn.py
Date: 23-Mar-2025
Author: Lucy
Description: Implementation of Self-Adaptive Physics-Informed Neural Networks (SA-PINNs)
             with soft attention mechanism based on the paper "Self-Adaptive Physics-Informed
             Neural Networks using a Soft Attention Mechanism".
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Union
from models.pinn_base import PINN


class SoftAttentionMask(nn.Module):
    """
    Implements various soft attention mask functions for SA-PINNs.
    
    These mask functions are used to weight individual training points
    during the training process, allowing the network to focus on 
    challenging regions of the solution domain.
    """
    
    def __init__(self, mask_type: str = "polynomial", params: Optional[Dict] = None):
        """
        Initialize the soft attention mask.
        
        Args:
            mask_type: Type of mask function ('polynomial', 'sigmoid', 'exponential')
            params: Dictionary of parameters specific to the chosen mask type
        """
        super(SoftAttentionMask, self).__init__()
        
        self.mask_type = mask_type
        self.params = params or {}
        
        # Set default parameters if not provided
        if mask_type == "polynomial":
            self.q = self.params.get("q", 2.0)
            self.c = self.params.get("c", 1.0)
            self.max_value = self.params.get("max_value", 1e5)
        elif mask_type == "sigmoid":
            self.scale = self.params.get("scale", 1.0)
            self.shift = self.params.get("shift", 0.0)
            self.min_value = self.params.get("min_value", 0.1)
            self.max_value = self.params.get("max_value", 10.0)
        elif mask_type == "exponential":
            self.scale = self.params.get("scale", 1.0)
            self.max_value = self.params.get("max_value", 1e5)
        else:
            raise ValueError(f"Mask type {mask_type} not supported")
    
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply the mask function to the adaptive weights.
        
        Args:
            weights: Tensor of adaptive weights
            
        Returns:
            Masked weights tensor of same shape
        """
        if self.mask_type == "polynomial":
            # m(λ) = c * λ^q, with upper limit to prevent overflow
            return torch.clamp(self.c * weights.pow(self.q), max=self.max_value)
        
        elif self.mask_type == "sigmoid":
            # Sigmoidal mask with adjustable sharpness and range
            sigmoid = torch.sigmoid(self.scale * (weights - self.shift))
            return self.min_value + (self.max_value - self.min_value) * sigmoid
        
        elif self.mask_type == "exponential":
            # Exponential mask: m(λ) = exp(scale * λ)
            return torch.clamp(torch.exp(self.scale * weights), max=self.max_value)
        
    def mask_gradient(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of the mask function with respect to weights.
        This is used in the self-adaptive weight update.
        
        Args:
            weights: Tensor of adaptive weights
            
        Returns:
            Gradient of mask with respect to weights
        """
        if self.mask_type == "polynomial":
            # d/dλ(c * λ^q) = c * q * λ^(q-1)
            return self.c * self.q * weights.pow(self.q - 1)
        
        elif self.mask_type == "sigmoid":
            # Derivative of sigmoid mask
            sigmoid = torch.sigmoid(self.scale * (weights - self.shift))
            return (self.max_value - self.min_value) * self.scale * sigmoid * (1 - sigmoid)
        
        elif self.mask_type == "exponential":
            # d/dλ(exp(scale * λ)) = scale * exp(scale * λ)
            return self.scale * torch.clamp(torch.exp(self.scale * weights), max=self.max_value)


class SAPINNWeights(nn.Module):
    """
    Manages the self-adaptive weights for different components of the loss function.
    
    This class handles the initialization, storage, and updates of weights for
    residual points, boundary conditions, and initial conditions.
    """
    
    def __init__(
        self,
        n_residual: int,
        n_boundary: int,
        n_initial: int,
        init_range: Dict[str, Tuple[float, float]] = None,
        weight_decay: Dict[str, float] = None,
        mask_type: str = "polynomial",
        mask_params: Optional[Dict] = None,
        trainable: Dict[str, bool] = None,
        device: str = "cpu"
    ):
        """
        Initialize adaptive weights for SA-PINN.
        
        Args:
            n_residual: Number of residual points
            n_boundary: Number of boundary condition points
            n_initial: Number of initial condition points
            init_range: Dictionary with ranges for initializing weights for each component
            weight_decay: Dictionary with weight decay factors for each component
            mask_type: Type of mask function to use
            mask_params: Parameters for the mask function
            trainable: Dictionary specifying which components have trainable weights
            device: Device to store the weights on ('cpu' or 'cuda')
        """
        super(SAPINNWeights, self).__init__()
        
        # Initialize default parameters if not provided
        self.init_range = init_range or {
            "residual": (0.0, 1.0),
            "boundary": (0.0, 1.0),
            "initial": (0.0, 1.0)
        }
        
        self.weight_decay = weight_decay or {
            "residual": 0.0,
            "boundary": 0.0,
            "initial": 0.0
        }
        
        self.trainable = trainable or {
            "residual": True,
            "boundary": True,
            "initial": True
        }
        
        # Create the soft attention mask
        self.mask = SoftAttentionMask(mask_type, mask_params)
        
        # Initialize weights
        self._init_weights(n_residual, n_boundary, n_initial, device)
    
    def _init_weights(self, n_residual: int, n_boundary: int, n_initial: int, device: str):
        """Initialize weight parameters with specified ranges."""
        # Residual weights
        if n_residual > 0:
            res_min, res_max = self.init_range["residual"]
            if self.trainable["residual"]:
                self.residual_weights = nn.Parameter(
                    torch.empty(n_residual, device=device).uniform_(res_min, res_max)
                )
            else:
                self.register_buffer(
                    "residual_weights",
                    torch.empty(n_residual, device=device).uniform_(res_min, res_max)
                )
        
        # Boundary weights
        if n_boundary > 0:
            bnd_min, bnd_max = self.init_range["boundary"]
            if self.trainable["boundary"]:
                self.boundary_weights = nn.Parameter(
                    torch.empty(n_boundary, device=device).uniform_(bnd_min, bnd_max)
                )
            else:
                self.register_buffer(
                    "boundary_weights",
                    torch.empty(n_boundary, device=device).uniform_(bnd_min, bnd_max)
                )
        
        # Initial condition weights
        if n_initial > 0:
            init_min, init_max = self.init_range["initial"]
            if self.trainable["initial"]:
                self.initial_weights = nn.Parameter(
                    torch.empty(n_initial, device=device).uniform_(init_min, init_max)
                )
            else:
                self.register_buffer(
                    "initial_weights",
                    torch.empty(n_initial, device=device).uniform_(init_min, init_max)
                )
    
    def get_masks(self) -> Dict[str, torch.Tensor]:
        """
        Compute mask values for all weights.
        
        Returns:
            Dictionary with masked weights for each component
        """
        masks = {}
        
        if hasattr(self, "residual_weights"):
            masks["residual"] = self.mask(self.residual_weights)
        
        if hasattr(self, "boundary_weights"):
            masks["boundary"] = self.mask(self.boundary_weights)
        
        if hasattr(self, "initial_weights"):
            masks["initial"] = self.mask(self.initial_weights)
        
        return masks
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Compute gradients of the mask function for all weights.
        
        Returns:
            Dictionary with mask gradients for each component
        """
        grads = {}
        
        if hasattr(self, "residual_weights"):
            grads["residual"] = self.mask.mask_gradient(self.residual_weights)
        
        if hasattr(self, "boundary_weights"):
            grads["boundary"] = self.mask.mask_gradient(self.boundary_weights)
        
        if hasattr(self, "initial_weights"):
            grads["initial"] = self.mask.mask_gradient(self.initial_weights)
        
        return grads


class SAPINN(PINN):
    """
    Self-Adaptive Physics-Informed Neural Network (SA-PINN) with soft attention mechanism.
    
    Extends the base PINN class with trainable weights for individual training points,
    allowing the network to adaptively focus on challenging regions of the solution.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        n_residual: int,
        n_boundary: int,
        n_initial: int,
        mask_type: str = "polynomial",
        mask_params: Optional[Dict] = None,
        init_range: Optional[Dict[str, Tuple[float, float]]] = None,
        trainable_weights: Optional[Dict[str, bool]] = None,
        activation: str = "tanh",
        initializer: str = "xavier_normal",
        dropout_rate: float = 0.0,
        device: str = "cpu"
    ):
        """
        Initialize the SA-PINN model.
        
        Args:
            input_dim: Number of input dimensions
            output_dim: Number of output dimensions
            hidden_layers: List of integers representing the number of neurons in each hidden layer
            n_residual: Number of residual points
            n_boundary: Number of boundary condition points
            n_initial: Number of initial condition points
            mask_type: Type of mask function ('polynomial', 'sigmoid', 'exponential')
            mask_params: Parameters for the mask function
            init_range: Dictionary with ranges for initializing weights for each component
            trainable_weights: Dictionary specifying which components have trainable weights
            activation: Activation function
            initializer: Weight initialization method
            dropout_rate: Dropout probability
            device: Device to store the model on
        """
        super(SAPINN, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            initializer=initializer,
            dropout_rate=dropout_rate
        )
        
        # Initialize self-adaptive weights
        self.adaptive_weights = SAPINNWeights(
            n_residual=n_residual,
            n_boundary=n_boundary,
            n_initial=n_initial,
            init_range=init_range,
            mask_type=mask_type,
            mask_params=mask_params,
            trainable=trainable_weights,
            device=device
        )
    
    def get_masked_losses(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply masks to individual loss components.
        
        Args:
            losses: Dictionary with unmasked losses for each point
            
        Returns:
            Dictionary with masked losses for each component
        """
        # Get masks for all components
        masks = self.adaptive_weights.get_masks()
        
        # Apply masks to individual losses
        masked_losses = {}
        
        for component in losses:
            if component in masks:
                masked_losses[component] = losses[component] * masks[component]
            else:
                masked_losses[component] = losses[component]
        
        return masked_losses
    
    def get_weight_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about the current adaptive weights.
        
        Returns:
            Dictionary with statistics for each component
        """
        stats = {}
        masks = self.adaptive_weights.get_masks()
        
        for component, mask in masks.items():
            if hasattr(self.adaptive_weights, f"{component}_weights"):
                weights = getattr(self.adaptive_weights, f"{component}_weights")
                stats[component] = {
                    "min_weight": weights.min().item(),
                    "max_weight": weights.max().item(),
                    "mean_weight": weights.mean().item(),
                    "min_mask": mask.min().item(),
                    "max_mask": mask.max().item(),
                    "mean_mask": mask.mean().item()
                }
        
        return stats