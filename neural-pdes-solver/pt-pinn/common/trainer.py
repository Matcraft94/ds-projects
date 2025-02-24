# Created by por: Lucy
# Date : 2025-02-22
# common/trainer.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, Callable, List
import numpy as np
from dataclasses import dataclass
from .pinn_base import BasePINN

@dataclass
class TrainerConfig:
    """Configuration for training parameters"""
    # Adam optimizer parameters
    adam_iterations: int = 5000
    initial_lr: float = 1e-3
    lr_decay_rate: float = 0.98
    lr_decay_steps: int = 50
    
    # L-BFGS parameters
    lbfgs_iterations: int = 1000
    lbfgs_tolerance: float = 1e-8
    
    # Resampling parameters (Algorithm 1)
    resampling_enabled: bool = True
    resampling_ratio: float = 0.6  # η in paper
    resampling_interval: int = 200  # K in paper
    resampling_termination: int = 4000  # F in paper
    
    # Evaluation parameters
    eval_frequency: int = 100
    batch_size: int = 1024

class PINNTrainer:
    """
    Trainer class implementing combined optimization strategy and resampling
    following sections 3.3 and Algorithm 1 of the paper.
    """
    
    def __init__(self, 
                 model: BasePINN,
                 config: TrainerConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
    def setup_optimizers(self) -> Tuple[optim.Optimizer, optim.LBFGS]:
        """Setup Adam and L-BFGS optimizers with learning rate scheduling"""
        # Adam optimizer with exponential learning rate decay
        optimizer_adam = optim.Adam(self.model.parameters(), 
                                  lr=self.config.initial_lr)
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer_adam,
            step_size=self.config.lr_decay_steps,
            gamma=self.config.lr_decay_rate
        )
        
        # L-BFGS optimizer
        optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            max_iter=self.config.lbfgs_iterations,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            max_eval=20,
            line_search_fn="strong_wolfe"
        )
        
        return optimizer_adam, scheduler, optimizer_lbfgs
    
    def resample_residual_points(self,
                               x_residual: torch.Tensor,
                               domain_bounds: torch.Tensor,
                               step: int) -> torch.Tensor:
        """
        Implement residual points resampling strategy from Algorithm 1
        
        Args:
            x_residual: Current residual points
            domain_bounds: Tensor of shape (n_dims, 2) with min/max bounds
            step: Current training step
        """
        if not self.config.resampling_enabled:
            return x_residual
            
        if (step % self.config.resampling_interval == 0 and 
            step < self.config.resampling_termination):
            
            n_points = len(x_residual)
            n_resample = int(self.config.resampling_ratio * n_points)
            
            # Keep (1-η) fraction of points
            keep_indices = torch.randperm(n_points)[:n_points - n_resample]
            kept_points = x_residual[keep_indices]
            
            # Generate η fraction of new points
            new_points = torch.zeros((n_resample, x_residual.shape[1]), 
                                  device=self.device)
            
            # Sample uniformly from domain bounds
            for dim in range(x_residual.shape[1]):
                new_points[:, dim] = torch.rand(n_resample, device=self.device) * \
                    (domain_bounds[dim, 1] - domain_bounds[dim, 0]) + \
                    domain_bounds[dim, 0]
            
            # Combine kept and new points
            x_residual = torch.cat([kept_points, new_points], dim=0)
            
        return x_residual
    
    def compute_errors(self, 
                      predictions: torch.Tensor,
                      targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute L1, L2 and L∞ errors according to equations 2.9-2.11
        """
        with torch.no_grad():
            # L2 relative error (eq 2.9)
            l2_rel = torch.sqrt(torch.sum((predictions - targets)**2)) / \
                    torch.sqrt(torch.sum(targets**2))
            
            # L1 mean absolute error (eq 2.10)
            l1_abs = torch.mean(torch.abs(predictions - targets))
            
            # L∞ maximum absolute error (eq 2.11)
            linf_abs = torch.max(torch.abs(predictions - targets))
            
        return {
            'l2_relative': l2_rel.item(),
            'l1_absolute': l1_abs.item(),
            'linf_absolute': linf_abs.item()
        }
    
    def evaluate(self,
                x_test: torch.Tensor,
                y_test: torch.Tensor) -> Dict[str, float]:
        """Evaluate model on test data"""
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x_test)
            metrics = self.compute_errors(y_pred, y_test)
        self.model.train()
        return metrics
    
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
    #     """Perform single training step"""
    #     optimizer.zero_grad()
        
    #     # Forward pass and loss computation
    #     total_loss, loss_components = self.model.loss(
    #         x_initial, initial_condition,
    #         x_boundary, boundary_condition,
    #         x_residual, pde_operator
    #     )
        
    #     # Add supervised loss if provided
    #     if x_supervised is not None and y_supervised is not None:
    #         supervised_loss = self.model.compute_supervised_loss(x_supervised, y_supervised)
    #         total_loss += supervised_loss
    #         loss_components['supervised'] = supervised_loss.item()
        
    #     # Backward pass
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

    def train(self, x_initial, initial_condition, x_boundary, boundary_condition, 
            x_residual, domain_bounds, pde_operator, x_test=None, y_test=None, 
            x_supervised=None, y_supervised=None):
        """
        Main training loop implementing combined optimization strategy
        from section 3.3 and Algorithm 1
        """
        # Setup optimizers
        optimizer_adam, scheduler, optimizer_lbfgs = self.setup_optimizers()
        
        # Training history
        history = {
            'total_loss': [],
            'initial_loss': [],
            'boundary_loss': [],
            'residual_loss': [],
            'l2_relative': [],
            'l1_absolute': [],
            'linf_absolute': []
        }
        
        # Phase 1: Adam optimization with resampling
        print("Starting Adam optimization phase...")
        for step in range(self.config.adam_iterations):
            # Resample residual points
            x_residual = self.resample_residual_points(x_residual, domain_bounds, step)
            
            # Training step
            loss_components = self.train_step(
                x_initial, initial_condition,
                x_boundary, boundary_condition,
                x_residual, pde_operator,
                optimizer_adam,
                x_supervised, y_supervised
            )
            
            scheduler.step()
            
            # Update history
            for key, value in loss_components.items():
                history[f'{key}_loss'].append(value)
                
            # Evaluate on test data
            if x_test is not None and y_test is not None and step % self.config.eval_frequency == 0:
                metrics = self.evaluate(x_test, y_test)
                for key, value in metrics.items():
                    history[key].append(value)
                    
            if step % self.config.eval_frequency == 0:
                print(f"Step {step}: Loss = {loss_components['total']:.6f}")
        
        # Phase 2: L-BFGS optimization
        print("Starting L-BFGS optimization phase...")
        
        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss, _ = self.model.loss(
                x_initial, initial_condition,
                x_boundary, boundary_condition,
                x_residual, pde_operator
            )
            if x_supervised is not None and y_supervised is not None:
                supervised_loss = self.model.compute_supervised_loss(x_supervised, y_supervised)
                total_loss += supervised_loss
            total_loss.backward(retain_graph=True)  # Añadido retain_graph=True aquí
            return total_loss
                
        optimizer_lbfgs.step(closure)
        
        # Final evaluation
        if x_test is not None and y_test is not None:
            final_metrics = self.evaluate(x_test, y_test)
            for key, value in final_metrics.items():
                history[key].append(value)
        
        print("Training completed!")
        return history