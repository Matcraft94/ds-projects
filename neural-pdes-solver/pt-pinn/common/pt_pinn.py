# Created by por: Lucy
# Date : 2025-02-22
# common/pt_pinn.py

import torch
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Callable
from copy import deepcopy
from dataclasses import dataclass
from .pinn_base import BasePINN, PINNConfig

@dataclass
class PreTrainingConfig:
    """Configuration for pre-training phases"""
    intervals: List[float]  # Pre-training time intervals [T1, T2, ..., T]
    n_supervised_points: int = 1000  # Number of supervised points to generate
    supervised_weight: float = 1.0  # Weight for supervised loss (w_sp)
    adam_iterations: int = 2000  # Number of Adam iterations
    lbfgs_iterations: int = 1000  # Number of L-BFGS iterations
    resampling_interval: int = 200  # K in Algorithm 1
    resampling_ratio: float = 0.6  # η in Algorithm 1
    resampling_termination: int = 4000  # F in Algorithm 1

class PT_PINN(BasePINN):
    """
    Pre-training Physics Informed Neural Network (PT-PINN) implementation
    following Algorithm 2 from the paper.
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 pinn_config: PINNConfig,
                 pt_config: PreTrainingConfig):
        """
        Initialize PT-PINN
        
        Args:
            input_dim: Number of input dimensions
            output_dim: Number of output dimensions
            pinn_config: Base PINN configuration
            pt_config: Pre-training specific configuration
        """
        super().__init__(input_dim, output_dim, pinn_config)
        self.pt_config = pt_config
        
        # Store pre-trained models for each interval
        self.pretrained_models: List[BasePINN] = []
        
    def generate_supervised_data(self, 
                               x_domain: torch.Tensor,
                               pretrained_model: BasePINN) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate supervised learning data from a pre-trained model
        following equation 3.4
        """
        # Randomly sample points from the domain
        indices = torch.randperm(len(x_domain))[:self.pt_config.n_supervised_points]
        x_supervised = x_domain[indices]
        
        # Generate predictions using pre-trained model
        with torch.no_grad():
            y_supervised = pretrained_model(x_supervised)
            
        return x_supervised, y_supervised

    def compute_supervised_loss(self,
                              x_supervised: torch.Tensor,
                              y_supervised: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised learning loss according to equation 3.3
        """
        pred_supervised = self.forward(x_supervised)
        return torch.mean((pred_supervised - y_supervised)**2)

    # def train_step(self,
    #               x_initial: torch.Tensor,
    #               initial_condition: torch.Tensor,
    #               x_boundary: torch.Tensor,
    #               boundary_condition: torch.Tensor,
    #               x_residual: torch.Tensor,
    #               pde_operator: Callable,
    #               optimizer: torch.optim.Optimizer,
    #               x_supervised: Optional[torch.Tensor] = None,
    #               y_supervised: Optional[torch.Tensor] = None) -> Dict[str, float]:
    #     """
    #     Perform one training step with optional supervised learning
    #     """
    #     optimizer.zero_grad()
        
    #     # Compute standard PINN losses
    #     total_loss, loss_components = self.loss(
    #         x_initial, initial_condition,
    #         x_boundary, boundary_condition,
    #         x_residual, pde_operator
    #     )
        
    #     # Add supervised loss if provided
    #     if x_supervised is not None and y_supervised is not None:
    #         supervised_loss = self.compute_supervised_loss(x_supervised, y_supervised)
    #         total_loss += self.pt_config.supervised_weight * supervised_loss
    #         loss_components['supervised'] = supervised_loss.item()
            
    #     total_loss.backward()
    #     optimizer.step()
        
    #     return loss_components

    def train_step(self,
                x_initial, initial_condition,
                x_boundary, boundary_condition,
                x_residual, pde_operator,
                optimizer, x_supervised=None, y_supervised=None):
        """Perform single training step"""
        optimizer.zero_grad()
        
        # Forward pass and loss computation
        with torch.set_grad_enabled(True):
            total_loss, loss_components = self.model.loss(
                x_initial, initial_condition,
                x_boundary, boundary_condition,
                x_residual, pde_operator
            )
            
            # Add supervised loss if provided
            if x_supervised is not None and y_supervised is not None:
                supervised_loss = self.model.compute_supervised_loss(x_supervised, y_supervised)
                total_loss += supervised_loss
                loss_components['supervised'] = supervised_loss.item()
            
            # Backward pass
            total_loss.backward(retain_graph=True)
            optimizer.step()
        
        return loss_components

    def resample_residual_points(self,
                               x_residual: torch.Tensor,
                               step: int) -> torch.Tensor:
        """
        Implement residual points resampling strategy from Algorithm 1
        """
        if (step % self.pt_config.resampling_interval == 0 and 
            step < self.pt_config.resampling_termination):
            
            n_points = len(x_residual)
            n_resample = int(self.pt_config.resampling_ratio * n_points)
            
            # Keep unchanged points
            keep_indices = torch.randperm(n_points)[:n_points - n_resample]
            kept_points = x_residual[keep_indices]
            
            # Generate new points
            # Note: This is a simplified version - actual implementation would
            # need domain-specific sampling logic
            new_points = torch.rand_like(x_residual[:n_resample])
            
            # Combine kept and new points
            x_residual = torch.cat([kept_points, new_points], dim=0)
            
        return x_residual

    def train_interval(self,
                      x_initial: torch.Tensor,
                      initial_condition: torch.Tensor,
                      x_boundary: torch.Tensor,
                      boundary_condition: torch.Tensor,
                      x_residual: torch.Tensor,
                      pde_operator: Callable,
                      x_supervised: Optional[torch.Tensor] = None,
                      y_supervised: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Train the model for one interval using Adam and L-BFGS optimizers
        """
        # First phase: Adam optimization with resampling
        optimizer_adam = optim.Adam(self.parameters())
        
        for step in range(self.pt_config.adam_iterations):
            # Resample residual points
            x_residual = self.resample_residual_points(x_residual, step)
            
            loss_components = self.train_step(
                x_initial, initial_condition,
                x_boundary, boundary_condition,
                x_residual, pde_operator,
                optimizer_adam,
                x_supervised, y_supervised
            )
            
        # Second phase: L-BFGS optimization
        optimizer_lbfgs = optim.LBFGS(self.parameters(),
                                     max_iter=self.pt_config.lbfgs_iterations)
        
        def closure():
            optimizer_lbfgs.zero_grad()
            loss, components = self.loss(
                x_initial, initial_condition,
                x_boundary, boundary_condition,
                x_residual, pde_operator
            )
            if x_supervised is not None and y_supervised is not None:
                supervised_loss = self.compute_supervised_loss(x_supervised, y_supervised)
                loss += self.pt_config.supervised_weight * supervised_loss
            loss.backward()
            return loss
            
        optimizer_lbfgs.step(closure)
        
        # Return final loss components
        with torch.no_grad():
            return self.train_step(
                x_initial, initial_condition,
                x_boundary, boundary_condition,
                x_residual, pde_operator,
                optimizer_lbfgs,
                x_supervised, y_supervised
            )

    def pretrain(self,
                x_initial: torch.Tensor,
                initial_condition: torch.Tensor,
                x_boundary: torch.Tensor,
                boundary_condition: torch.Tensor,
                x_residual: torch.Tensor,
                pde_operator: Callable) -> None:
        """
        Implement pre-training strategy following Algorithm 2
        """
        self.pretrained_models = []
        
        for interval_idx, T in enumerate(self.pt_config.intervals[:-1]):
            print(f"Pre-training interval {interval_idx + 1}: [0, {T}]")
            
            # For first interval, use standard training
            if interval_idx == 0:
                losses = self.train_interval(
                    x_initial, initial_condition,
                    x_boundary, boundary_condition,
                    x_residual, pde_operator
                )
                
            # For subsequent intervals, use supervised data from previous model
            else:
                previous_model = self.pretrained_models[-1]
                x_sup, y_sup = self.generate_supervised_data(x_residual, previous_model)
                
                losses = self.train_interval(
                    x_initial, initial_condition,
                    x_boundary, boundary_condition,
                    x_residual, pde_operator,
                    x_sup, y_sup
                )
                
            # Store pre-trained model
            self.pretrained_models.append(deepcopy(self))
            print(f"Interval {interval_idx + 1} losses:", losses)

    def train(self,
              x_initial: torch.Tensor,
              initial_condition: torch.Tensor,
              x_boundary: torch.Tensor,
              boundary_condition: torch.Tensor,
              x_residual: torch.Tensor,
              pde_operator: Callable) -> Dict[str, float]:
        """
        Complete training process following Algorithm 2:
        1. Pre-training on multiple intervals
        2. Final training on full domain
        """
        # Pre-training phase
        self.pretrain(
            x_initial, initial_condition,
            x_boundary, boundary_condition,
            x_residual, pde_operator
        )
        
        # Final training phase
        print(f"Final training on interval [0, {self.pt_config.intervals[-1]}]")
        
        # Generate supervised data from last pre-trained model
        x_sup, y_sup = self.generate_supervised_data(
            x_residual, 
            self.pretrained_models[-1]
        )
        
        # Perform final training
        final_losses = self.train_interval(
            x_initial, initial_condition,
            x_boundary, boundary_condition,
            x_residual, pde_operator,
            x_sup, y_sup
        )
        
        print("Final training losses:", final_losses)
        return final_losses