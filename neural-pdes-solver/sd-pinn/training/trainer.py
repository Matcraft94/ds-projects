"""
training/trainer.py
Path: training/trainer.py
Date: 23-Mar-2025
Author: Lucy
Description: Main training methods for Physics-Informed Neural Networks (PINNs)
             and Self-Adaptive PINNs (SA-PINNs). Includes implementation of
             training loops with various optimization strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy

from models.pinn_base import PINN
from models.sa_pinn import SAPINN
from training.loss_functions import PINNLoss


class PINNTrainer:
    """
    Trainer class for Physics-Informed Neural Networks (PINNs).
    
    Handles the training process for both standard PINNs and SA-PINNs,
    including optimization, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: Union[PINN, SAPINN],
        loss_fn: PINNLoss,
        optimizer_config: Dict[str, Any],
        scheduler_config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        checkpoint_dir: str = "./checkpoints",
        use_tqdm: bool = True
    ):
        """
        Initialize the PINN trainer.
        
        Args:
            model: The PINN or SA-PINN model to train
            loss_fn: The loss function for training
            optimizer_config: Configuration for the optimizer
                {
                    "network": {"type": "adam", "lr": 0.001, ...},
                    "weights": {"type": "sgd", "lr": 0.1, ...}  # Only for SA-PINN
                }
            scheduler_config: Configuration for learning rate schedulers (optional)
            device: Device to run training on ('cpu' or 'cuda')
            checkpoint_dir: Directory to save checkpoints
            use_tqdm: Whether to use tqdm progress bars
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_tqdm = use_tqdm
        
        # Move model to device
        self.model.to(device)
        
        # Create optimizers
        self._setup_optimizers(optimizer_config)
        
        # Create schedulers if provided
        self.schedulers = {}
        if scheduler_config is not None:
            self._setup_schedulers(scheduler_config)
        
        # Initialize logging
        self.logs = {
            "train_loss": [],
            "val_loss": [],
            "train_loss_components": [],
            "val_loss_components": [],
            "train_time": [],
            "l2_error": []
        }
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def _setup_optimizers(self, optimizer_config: Dict[str, Any]):
        """Setup optimizers based on configuration."""
        self.optimizers = {}
        
        # Network optimizer
        net_config = optimizer_config.get("network", {})
        opt_type = net_config.pop("type", "adam").lower()
        
        if opt_type == "adam":
            self.optimizers["network"] = optim.Adam(self.model.parameters(), **net_config)
        elif opt_type == "sgd":
            self.optimizers["network"] = optim.SGD(self.model.parameters(), **net_config)
        elif opt_type == "lbfgs":
            self.optimizers["network"] = optim.LBFGS(self.model.parameters(), **net_config)
        else:
            raise ValueError(f"Optimizer type {opt_type} not supported")
        
        # Weights optimizer (only for SA-PINN)
        if isinstance(self.model, SAPINN) and "weights" in optimizer_config:
            weights_config = optimizer_config.get("weights", {})
            opt_type = weights_config.pop("type", "adam").lower()
            
            # Filter only trainable adaptive weights parameters
            adaptive_params = [p for p in self.model.adaptive_weights.parameters() 
                              if p.requires_grad]
            
            if adaptive_params:
                if opt_type == "adam":
                    self.optimizers["weights"] = optim.Adam(adaptive_params, **weights_config)
                elif opt_type == "sgd":
                    self.optimizers["weights"] = optim.SGD(adaptive_params, **weights_config)
                else:
                    raise ValueError(f"Optimizer type {opt_type} not supported for weights")
    
    def _setup_schedulers(self, scheduler_config: Dict[str, Any]):
        """Setup learning rate schedulers based on configuration."""
        for opt_name, opt in self.optimizers.items():
            if opt_name in scheduler_config:
                config = scheduler_config[opt_name]
                sched_type = config.pop("type", "").lower()
                
                if sched_type == "step":
                    self.schedulers[opt_name] = optim.lr_scheduler.StepLR(opt, **config)
                elif sched_type == "plateau":
                    self.schedulers[opt_name] = optim.lr_scheduler.ReduceLROnPlateau(opt, **config)
                elif sched_type == "cosine":
                    self.schedulers[opt_name] = optim.lr_scheduler.CosineAnnealingLR(opt, **config)
                elif sched_type == "warmup":
                    self.schedulers[opt_name] = optim.lr_scheduler.LinearLR(
                        opt, start_factor=config.get("start_factor", 0.1),
                        end_factor=1.0, total_iters=config.get("warmup_steps", 1000)
                    )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary containing input tensors for different components
            
        Returns:
            Dictionary with loss values
        """
        # Zero gradients
        for opt in self.optimizers.values():
            opt.zero_grad()
        
        # Compute raw loss components
        loss_components = self.loss_fn(self.model, batch)
        
        if isinstance(self.model, SAPINN):
            # For SA-PINN: Apply masks to losses
            masked_components = self.model.get_masked_losses(loss_components)
            
            # Network update (minimize loss)
            total_loss = sum(masked_components.values())
            total_loss.backward(retain_graph=True)
            self.optimizers["network"].step()
            
            # Weights update (maximize loss)
            if "weights" in self.optimizers:
                # Zero gradients for weights optimizer
                self.optimizers["weights"].zero_grad()
                
                # For each component with trainable weights, compute gradient ascent
                weight_grads = self.model.adaptive_weights.get_gradients()
                weight_loss = 0
                
                for component, grad in weight_grads.items():
                    if component in loss_components:
                        # We want to maximize w.r.t weights, so negate the gradient
                        term = -0.5 * grad * (loss_components[component].detach()**2)
                        weight_loss = weight_loss + term.sum()
                
                # Backpropagate for weights (gradient ascent)
                if weight_loss != 0:
                    weight_loss.backward()
                    self.optimizers["weights"].step()
            
            return {
                "total": total_loss.item(),
                "components": {k: v.item() for k, v in loss_components.items()},
                "masked": {k: v.item() for k, v in masked_components.items()}
            }
        else:
            # For standard PINN: Simply sum all components
            total_loss = sum(loss_components.values())
            total_loss.backward()
            self.optimizers["network"].step()
            
            return {
                "total": total_loss.item(),
                "components": {k: v.item() for k, v in loss_components.items()}
            }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader providing batches
            
        Returns:
            Dictionary with average loss values
        """
        self.model.train()
        epoch_losses = []
        component_losses = {}
        
        iterator = dataloader
        if self.use_tqdm:
            iterator = tqdm(dataloader, desc="Training", leave=False)
        
        for batch in iterator:
            # Move batch to device - handle nested dictionaries
            processed_batch = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    processed_batch[k] = {sub_k: sub_v.to(self.device) for sub_k, sub_v in v.items()}
                else:
                    processed_batch[k] = v.to(self.device)
            batch = processed_batch
            
            # Perform training step
            step_loss = self.train_step(batch)
            epoch_losses.append(step_loss["total"])
            
            # Accumulate component losses
            for component, loss in step_loss["components"].items():
                if component not in component_losses:
                    component_losses[component] = []
                component_losses[component].append(loss)
        
        # Calculate average losses
        avg_total_loss = sum(epoch_losses) / len(epoch_losses)
        avg_component_losses = {
            k: sum(v) / len(v) for k, v in component_losses.items()
        }
        
        return {
            "total": avg_total_loss,
            "components": avg_component_losses
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: DataLoader providing validation batches
            
        Returns:
            Dictionary with average validation loss values
        """
        self.model.eval()
        val_losses = []
        component_losses = {}
        
        iterator = dataloader
        if self.use_tqdm:
            iterator = tqdm(dataloader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for batch in iterator:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Compute loss components
                loss_components = self.loss_fn(self.model, batch)
                
                if isinstance(self.model, SAPINN):
                    # For SA-PINN: Apply masks to losses
                    masked_components = self.model.get_masked_losses(loss_components)

                    # Network update (minimize loss)
                    total_loss = sum(masked_components.values())
                    # Ensure loss is scalar by taking mean if it's not already
                    if total_loss.dim() > 0:
                        total_loss = total_loss.mean()
                    total_loss.backward(retain_graph=True)
                else:
                    # For standard PINN
                    total_loss = sum(loss_components.values())
                
                val_losses.append(total_loss.item())
                
                # Accumulate component losses
                for component, loss in loss_components.items():
                    if component not in component_losses:
                        component_losses[component] = []
                    component_losses[component].append(loss.item())
        
        # Calculate average losses
        avg_total_loss = sum(val_losses) / len(val_losses)
        avg_component_losses = {
            k: sum(v) / len(v) for k, v in component_losses.items()
        }
        
        return {
            "total": avg_total_loss,
            "components": avg_component_losses
        }
    
    def compute_l2_error(self, x_test: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Compute relative L2 error between model predictions and ground truth.
        
        Args:
            x_test: Input tensor
            y_true: Ground truth tensor
            
        Returns:
            Relative L2 error
        """
        self.model.eval()
        x_test = x_test.to(self.device)
        y_true = y_true.to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(x_test)
        
        error = torch.sqrt(torch.sum((y_pred - y_true)**2)) / torch.sqrt(torch.sum(y_true**2))
        
        return error.item()
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        n_epochs: int = 1000,
        test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        eval_freq: int = 10,
        checkpoint_freq: int = 100,
        early_stopping: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the PINN model.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional)
            n_epochs: Number of epochs to train
            test_data: Tuple of (x_test, y_true) for computing L2 error (optional)
            eval_freq: Frequency of validation and error computation
            checkpoint_freq: Frequency of model checkpointing
            early_stopping: Early stopping configuration (optional)
                {
                    "patience": int,
                    "min_delta": float,
                    "monitor": str  # "train_loss", "val_loss", or "l2_error"
                }
                
        Returns:
            Training logs
        """
        # Initialize early stopping parameters
        best_value = float('inf')
        patience_counter = 0
        
        # Main training loop
        iterator = range(n_epochs)
        if self.use_tqdm:
            iterator = tqdm(iterator, desc="Epochs")
        
        for epoch in iterator:
            # Train for one epoch
            start_time = time.time()
            train_loss = self.train_epoch(train_dataloader)
            epoch_time = time.time() - start_time
            
            # Log training loss
            self.logs["train_loss"].append(train_loss["total"])
            self.logs["train_loss_components"].append(train_loss["components"])
            self.logs["train_time"].append(epoch_time)
            
            # Validate and compute error at specified frequency
            if epoch % eval_freq == 0:
                # Validation
                if val_dataloader is not None:
                    val_loss = self.validate(val_dataloader)
                    self.logs["val_loss"].append(val_loss["total"])
                    self.logs["val_loss_components"].append(val_loss["components"])
                
                # Compute L2 error if test data is provided
                if test_data is not None:
                    x_test, y_true = test_data
                    l2_error = self.compute_l2_error(x_test, y_true)
                    self.logs["l2_error"].append(l2_error)
                    
                    if self.use_tqdm:
                        train_str = f"Train Loss: {train_loss['total']:.6f}"
                        val_str = f", Val Loss: {val_loss['total']:.6f}" if val_dataloader else ""
                        err_str = f", L2 Error: {l2_error:.6f}"
                        iterator.set_postfix_str(train_str + val_str + err_str)
                
                # Update learning rate schedulers
                for scheduler_name, scheduler in self.schedulers.items():
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        # For ReduceLROnPlateau, we need to pass the monitored value
                        if early_stopping and early_stopping.get("monitor") == "l2_error" and test_data:
                            scheduler.step(l2_error)
                        elif val_dataloader:
                            scheduler.step(val_loss["total"])
                        else:
                            scheduler.step(train_loss["total"])
                    else:
                        # For other schedulers
                        scheduler.step()
            
            # Save checkpoint at specified frequency
            if epoch % checkpoint_freq == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
            # Check for early stopping
            if early_stopping:
                patience = early_stopping.get("patience", 10)
                min_delta = early_stopping.get("min_delta", 0.0)
                monitor = early_stopping.get("monitor", "train_loss")
                
                # Determine current value to monitor
                if monitor == "l2_error" and test_data:
                    current_value = self.logs["l2_error"][-1]
                elif monitor == "val_loss" and val_dataloader:
                    current_value = self.logs["val_loss"][-1]
                else:
                    current_value = self.logs["train_loss"][-1]
                
                # Check if improvement
                if current_value < best_value - min_delta:
                    best_value = current_value
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint("best_model.pt")
                else:
                    patience_counter += 1
                
                # Stop if no improvement for specified patience
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        return self.logs
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "schedulers": {name: sched.state_dict() for name, sched in self.schedulers.items()},
            "logs": self.logs
        }
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
    
    def load_checkpoint(self, filename: str, load_optimizers: bool = True):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer states if required
        if load_optimizers:
            for name, state_dict in checkpoint["optimizers"].items():
                if name in self.optimizers:
                    self.optimizers[name].load_state_dict(state_dict)
            
            # Load scheduler states
            for name, state_dict in checkpoint.get("schedulers", {}).items():
                if name in self.schedulers:
                    self.schedulers[name].load_state_dict(state_dict)
        
        # Load logs
        if "logs" in checkpoint:
            self.logs = checkpoint["logs"]