# Created by por: Lucy
# Date : 2025-02-22
# common/pinn_base.py

import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PINNConfig:
    """Configuration class for PINN hyperparameters"""
    hidden_layers: int = 5
    neurons_per_layer: int = 50
    activation: nn.Module = nn.Tanh()
    w_initial: float = 1.0  # Weight for initial condition loss
    w_boundary: float = 1.0  # Weight for boundary condition loss
    w_residual: float = 1.0  # Weight for residual loss

class BasePINN(nn.Module):
    """
    Base class for Physics Informed Neural Networks (PINN).
    Implements the core functionality described in section 2 of the paper.
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 config: PINNConfig):
        """
        Initialize the PINN.
        
        Args:
            input_dim: Number of input dimensions (spatial + temporal)
            output_dim: Number of output dimensions
            config: Configuration object containing network architecture details
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
        # Build neural network
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, config.neurons_per_layer))
        layers.append(config.activation)
        
        # Hidden layers
        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(config.neurons_per_layer, config.neurons_per_layer))
            layers.append(config.activation)
            
        # Output layer
        layers.append(nn.Linear(config.neurons_per_layer, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network"""
        return self.network(x)

    def compute_initial_loss(self, x_initial: torch.Tensor, 
                           initial_condition: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition loss according to equation 2.5
        
        Args:
            x_initial: Points where initial condition is enforced
            initial_condition: True initial condition values
        """
        pred_initial = self.forward(x_initial)
        return torch.mean((pred_initial - initial_condition)**2)

    def compute_boundary_loss(self, x_boundary: torch.Tensor,
                            boundary_condition: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss according to equation 2.6
        
        Args:
            x_boundary: Points on the boundary
            boundary_condition: True boundary values
        """
        pred_boundary = self.forward(x_boundary)
        return torch.mean((pred_boundary - boundary_condition)**2)

    # def compute_residual_loss(self, x_residual: torch.Tensor,
    #                         pde_operator: Callable) -> torch.Tensor:
    #     """
    #     Compute PDE residual loss according to equation 2.7
        
    #     Args:
    #         x_residual: Interior points for computing PDE residual
    #         pde_operator: Function that computes the PDE residual
    #     """
    #     # Enable gradient computation
    #     x_residual.requires_grad_(True)
        
    #     # Compute residual
    #     residual = pde_operator(x_residual, self)
        
    #     return torch.mean(residual**2)

    def compute_residual_loss(self, x_residual: torch.Tensor, 
                            pde_operator: Callable) -> torch.Tensor:
        """Compute PDE residual loss according to equation 2.7"""
        # Asegurarnos de que x_residual requiere gradiente
        if not x_residual.requires_grad:
            x_residual.requires_grad_(True)
        
        # Compute residual
        with torch.set_grad_enabled(True):
            residual = pde_operator(x_residual)
            return torch.mean(residual**2)

    def loss(self, x_initial, initial_condition, x_boundary, boundary_condition, 
            x_residual, pde_operator):
        """
        Compute total loss according to equation 2.4
        
        Returns:
            total_loss: Combined weighted loss
            loss_components: Dictionary containing individual loss components
        """
        with torch.set_grad_enabled(True):
            loss_initial = self.compute_initial_loss(x_initial, initial_condition)
            loss_boundary = self.compute_boundary_loss(x_boundary, boundary_condition)
            loss_residual = self.compute_residual_loss(x_residual, pde_operator)
            
            # Combine losses with weights
            total_loss = (self.config.w_initial * loss_initial + 
                        self.config.w_boundary * loss_boundary +
                        self.config.w_residual * loss_residual)
            
            # Store individual loss components
            loss_components = {
                'initial': loss_initial.item(),
                'boundary': loss_boundary.item(),
                'residual': loss_residual.item(),
                'total': total_loss.item()
            }
            
            return total_loss, loss_components

    def compute_gradients(self, x: torch.Tensor, 
                         order: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        Compute gradients of the network output with respect to input.
        Supports computing arbitrary order derivatives.
        
        Args:
            x: Input points
            order: Tuple specifying the order of derivatives for each input dimension
                  e.g. (1,0) means first derivative w.r.t first dimension
                  
        Returns:
            Tensor containing the computed derivatives
        """
        if order is None:
            return self.forward(x)
            
        x.requires_grad_(True)
        y = self.forward(x)
        
        # For each dimension, compute the specified order of derivative
        for dim, order_dim in enumerate(order):
            for _ in range(order_dim):
                grads = torch.autograd.grad(
                    y, x,
                    grad_outputs=torch.ones_like(y),
                    create_graph=True
                )[0]
                y = grads[..., dim:dim+1]
                
        return y

    def evaluate_error(self, x_test: torch.Tensor, 
                      y_true: torch.Tensor) -> Dict[str, float]:
        """
        Compute various error metrics on test data according to equations 2.9-2.11
        
        Returns dictionary containing:
            - L2 relative error
            - L1 mean absolute error
            - L∞ maximum absolute error
        """
        with torch.no_grad():
            y_pred = self.forward(x_test)
            
            # L2 relative error (eq 2.9)
            l2_rel = torch.sqrt(torch.sum((y_pred - y_true)**2)) / \
                    torch.sqrt(torch.sum(y_true**2))
            
            # L1 mean absolute error (eq 2.10)
            l1_abs = torch.mean(torch.abs(y_pred - y_true))
            
            # L∞ maximum absolute error (eq 2.11)
            linf_abs = torch.max(torch.abs(y_pred - y_true))
            
        return {
            'l2_relative': l2_rel.item(),
            'l1_absolute': l1_abs.item(),
            'linf_absolute': linf_abs.item()
        }