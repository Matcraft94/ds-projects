"""
training/loss_functions.py
Path: training/loss_functions.py
Date: 23-Mar-2025
Author: Claude
Description: Loss function implementations for Physics-Informed Neural Networks (PINNs)
             and Self-Adaptive PINNs (SA-PINNs). Includes functions to compute losses
             for PDE residuals, boundary conditions, and initial conditions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
from models.pinn_base import PINN
from models.sa_pinn import SAPINN


class PINNLoss:
    """
    Base class for PINN loss functions.
    
    Computes the various components of the loss function for a PINN model,
    including PDE residuals, boundary conditions, and initial conditions.
    """
    
    def __init__(
        self,
        pde_residual_fn: Callable,
        boundary_condition_fns: Optional[Dict[str, Callable]] = None,
        initial_condition_fns: Optional[Dict[str, Callable]] = None,
        data_loss_fn: Optional[Callable] = None,
        mse_reduction: str = "mean"
    ):
        """
        Initialize the PINN loss function.
        
        Args:
            pde_residual_fn: Function to compute the PDE residual
                Signature: (model, x, t, u, derivatives) -> residual tensor
            boundary_condition_fns: Dictionary of functions to compute boundary conditions
                Signature: (model, x, t, u, derivatives) -> boundary error tensor
            initial_condition_fns: Dictionary of functions to compute initial conditions
                Signature: (model, x, t, u, derivatives) -> initial error tensor
            data_loss_fn: Function to compute the data fitting loss (optional)
                Signature: (model, x, t, u_pred, u_true) -> data error tensor
            mse_reduction: Reduction method for MSE loss ('mean' or 'none')
        """
        self.pde_residual_fn = pde_residual_fn
        self.boundary_condition_fns = boundary_condition_fns or {}
        self.initial_condition_fns = initial_condition_fns or {}
        self.data_loss_fn = data_loss_fn
        self.mse_reduction = mse_reduction
        
        # Create MSE loss function
        self.mse = nn.MSELoss(reduction=mse_reduction)
    
    def compute_derivatives(
        self, 
        model: Union[PINN, SAPINN], 
        inputs: Dict[str, torch.Tensor],
        output_var: str = "u"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute model outputs and derivatives for the given inputs.
        
        Args:
            model: PINN model
            inputs: Dictionary of input tensors
            output_var: Name of the output variable (default: "u")
            
        Returns:
            Dictionary with model outputs and derivatives
        """
        # Prepare inputs
        x = inputs.get("x")
        t = inputs.get("t")
        
        if x is None:
            raise ValueError("Input 'x' not found in inputs")
        
        # Combine spatial and temporal inputs
        if t is not None:
            x_t = torch.cat([x, t], dim=1)
        else:
            x_t = x
        
        # Make sure inputs require gradient
        x_t.requires_grad_(True)
        
        # Forward pass
        u = model(x_t)
        
        # Compute first-order derivatives
        u_derivatives = {}
        
        # Compute grad of u with respect to x_t
        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=x_t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Spatial derivatives
        if t is not None:
            # For space-time problems
            u_x = grad_u[:, 0:x.shape[1]]
            u_t = grad_u[:, x.shape[1]:]
            
            u_derivatives["u_x"] = u_x
            u_derivatives["u_t"] = u_t
            
            # Compute second-order spatial derivatives
            for i in range(x.shape[1]):
                # Compute u_xx, u_yy, etc.
                grad_outputs = torch.ones_like(u_x[:, i])
                u_xx = torch.autograd.grad(
                    outputs=u_x[:, i],
                    inputs=x_t,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                u_derivatives[f"u_xx_{i}"] = u_xx[:, i]
        else:
            # For steady-state problems
            u_x = grad_u
            u_derivatives["u_x"] = u_x
            
            # Compute second-order spatial derivatives
            for i in range(x.shape[1]):
                grad_outputs = torch.ones_like(u_x[:, i])
                u_xx = torch.autograd.grad(
                    outputs=u_x[:, i],
                    inputs=x,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                u_derivatives[f"u_xx_{i}"] = u_xx[:, i]
        
        # Package everything
        return {output_var: u, **u_derivatives}
    
    def compute_pde_residual(
        self, 
        model: Union[PINN, SAPINN], 
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the PDE residual loss.
        
        Args:
            model: PINN model
            inputs: Dictionary with input tensors for residual points
            
        Returns:
            PDE residual loss
        """
        # Compute model outputs and derivatives
        outputs = self.compute_derivatives(model, inputs)
        
        # Compute PDE residual
        residual = self.pde_residual_fn(model, inputs, outputs)
        
        # Compute MSE loss
        if self.mse_reduction == "mean":
            return self.mse(residual, torch.zeros_like(residual))
        else:
            return (residual ** 2)
    
    def compute_boundary_losses(
        self, 
        model: Union[PINN, SAPINN], 
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute boundary condition losses.
        
        Args:
            model: PINN model
            inputs: Dictionary with input tensors for boundary points
            
        Returns:
            Dictionary of boundary condition losses
        """
        boundary_losses = {}
        
        # Compute model outputs and derivatives
        outputs = self.compute_derivatives(model, inputs)
        
        # Compute each boundary condition loss
        for bc_name, bc_fn in self.boundary_condition_fns.items():
            bc_error = bc_fn(model, inputs, outputs)
            
            if self.mse_reduction == "mean":
                bc_loss = self.mse(bc_error, torch.zeros_like(bc_error))
            else:
                bc_loss = (bc_error ** 2)
            
            boundary_losses[f"boundary_{bc_name}"] = bc_loss
        
        return boundary_losses
    
    def compute_initial_losses(
        self, 
        model: Union[PINN, SAPINN], 
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute initial condition losses.
        
        Args:
            model: PINN model
            inputs: Dictionary with input tensors for initial points
            
        Returns:
            Dictionary of initial condition losses
        """
        initial_losses = {}
        
        # Compute model outputs and derivatives
        outputs = self.compute_derivatives(model, inputs)
        
        # Compute each initial condition loss
        for ic_name, ic_fn in self.initial_condition_fns.items():
            ic_error = ic_fn(model, inputs, outputs)
            
            if self.mse_reduction == "mean":
                ic_loss = self.mse(ic_error, torch.zeros_like(ic_error))
            else:
                ic_loss = (ic_error ** 2)
            
            initial_losses[f"initial_{ic_name}"] = ic_loss
        
        return initial_losses
    
    def compute_data_loss(
        self, 
        model: Union[PINN, SAPINN], 
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute data fitting loss.
        
        Args:
            model: PINN model
            inputs: Dictionary with input tensors and target values
            
        Returns:
            Data fitting loss
        """
        if self.data_loss_fn is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        # Extract inputs and targets
        x = inputs.get("x")
        t = inputs.get("t")
        u_true = inputs.get("u_true")
        
        if u_true is None:
            raise ValueError("Target 'u_true' not found in inputs")
        
        # Combine spatial and temporal inputs
        if t is not None:
            x_t = torch.cat([x, t], dim=1)
        else:
            x_t = x
        
        # Forward pass
        u_pred = model(x_t)
        
        # Compute data loss
        data_error = self.data_loss_fn(model, x_t, u_pred, u_true)
        
        if self.mse_reduction == "mean":
            return self.mse(data_error, torch.zeros_like(data_error))
        else:
            return (data_error ** 2)
    
    def __call__(
        self, 
        model: Union[PINN, SAPINN], 
        batch: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components for the given batch.
        
        Args:
            model: PINN model
            batch: Dictionary with input tensors for different components
                {
                    "residual": {"x": tensor, "t": tensor, ...},
                    "boundary": {"x": tensor, "t": tensor, ...},
                    "initial": {"x": tensor, "t": tensor, ...},
                    "data": {"x": tensor, "t": tensor, "u_true": tensor, ...}
                }
            
        Returns:
            Dictionary with loss values for each component
        """
        loss_components = {}
        
        # Compute PDE residual loss
        if "residual" in batch:
            residual_loss = self.compute_pde_residual(model, batch["residual"])
            loss_components["residual"] = residual_loss
        
        # Compute boundary condition losses
        if "boundary" in batch and self.boundary_condition_fns:
            boundary_losses = self.compute_boundary_losses(model, batch["boundary"])
            loss_components.update(boundary_losses)
        
        # Compute initial condition losses
        if "initial" in batch and self.initial_condition_fns:
            initial_losses = self.compute_initial_losses(model, batch["initial"])
            loss_components.update(initial_losses)
        
        # Compute data fitting loss
        if "data" in batch and self.data_loss_fn:
            data_loss = self.compute_data_loss(model, batch["data"])
            loss_components["data"] = data_loss
        
        return loss_components


class AdvectionDiffusionLoss(PINNLoss):
    """
    Loss function for advection-diffusion equations.
    
    Implements loss components for equations of the form:
    u_t + v⋅∇u = D∇²u + s
    
    where:
    - u is the concentration
    - v is the velocity field
    - D is the diffusion coefficient
    - s is a source term
    """
    
    def __init__(
        self,
        diffusion_coefficient: float,
        velocity_field: Union[float, List[float], Callable] = 0.0,
        source_term: Union[float, Callable] = 0.0,
        initial_condition: Optional[Callable] = None,
        boundary_conditions: Optional[Dict[str, Callable]] = None,
        data_loss_fn: Optional[Callable] = None,
        mse_reduction: str = "mean"
    ):
        """
        Initialize the advection-diffusion loss function.
        
        Args:
            diffusion_coefficient: Diffusion coefficient (D)
            velocity_field: Velocity field (v) as a scalar, vector, or function
            source_term: Source term (s) as a scalar or function
            initial_condition: Function that computes the initial condition error
            boundary_conditions: Dictionary of functions for boundary conditions
            data_loss_fn: Function to compute the data fitting loss
            mse_reduction: Reduction method for MSE loss
        """
        # Define PDE residual function
        def advection_diffusion_residual(model, inputs, outputs):
            # Extract coordinates and time
            x = inputs.get("x")
            t = inputs.get("t")
            
            # Extract solution and derivatives
            u = outputs["u"]
            u_t = outputs.get("u_t")
            u_x = outputs["u_x"]
            
            # Get spatial dimension
            dim = x.shape[1]
            
            # Process velocity field
            if callable(velocity_field):
                # Velocity field is a function of x and t
                if t is not None:
                    v = velocity_field(x, t)
                else:
                    v = velocity_field(x)
            elif isinstance(velocity_field, (list, tuple, np.ndarray)):
                # Velocity field is a vector
                v = torch.tensor(velocity_field, device=x.device).expand(x.shape[0], -1)
            else:
                # Velocity field is a scalar
                v = torch.ones(x.shape[0], dim, device=x.device) * velocity_field
            
            # Process source term
            if callable(source_term):
                # Source term is a function of x and t
                if t is not None:
                    s = source_term(x, t)
                else:
                    s = source_term(x)
            else:
                # Source term is a scalar
                s = torch.ones(x.shape[0], 1, device=x.device) * source_term
            
            # Compute laplacian (∇²u)
            laplacian = torch.zeros_like(u)
            for i in range(dim):
                laplacian += outputs[f"u_xx_{i}"].unsqueeze(1)
            
            # Compute divergence (v⋅∇u)
            divergence = torch.zeros_like(u)
            for i in range(dim):
                divergence += v[:, i:i+1] * u_x[:, i:i+1]
            
            # Compute PDE residual: u_t + v⋅∇u - D∇²u - s = 0
            if u_t is not None:
                residual = u_t + divergence - diffusion_coefficient * laplacian - s
            else:
                # Steady-state case: v⋅∇u - D∇²u - s = 0
                residual = divergence - diffusion_coefficient * laplacian - s
            
            return residual
        
        # Initialize initial and boundary condition functions
        initial_conditions = {}
        if initial_condition is not None:
            initial_conditions["u0"] = initial_condition
        
        # Initialize parent class
        super().__init__(
            pde_residual_fn=advection_diffusion_residual,
            boundary_condition_fns=boundary_conditions,
            initial_condition_fns=initial_conditions,
            data_loss_fn=data_loss_fn,
            mse_reduction=mse_reduction
        )
        
        # Store parameters
        self.diffusion_coefficient = diffusion_coefficient
        self.velocity_field = velocity_field
        self.source_term = source_term


class BurgersLoss(PINNLoss):
    """
    Loss function for Burgers' equation.
    
    Implements loss components for the viscous Burgers equation of the form:
    u_t + u⋅u_x = ν⋅u_xx
    
    where:
    - u is the solution
    - ν is the viscosity coefficient
    """
    
    def __init__(
        self,
        viscosity: float,
        initial_condition: Optional[Callable] = None,
        boundary_conditions: Optional[Dict[str, Callable]] = None,
        data_loss_fn: Optional[Callable] = None,
        mse_reduction: str = "mean"
    ):
        """
        Initialize the Burgers equation loss function.
        
        Args:
            viscosity: Viscosity coefficient (ν)
            initial_condition: Function that computes the initial condition error
            boundary_conditions: Dictionary of functions for boundary conditions
            data_loss_fn: Function to compute the data fitting loss
            mse_reduction: Reduction method for MSE loss
        """
        # Define PDE residual function
        def burgers_residual(model, inputs, outputs):
            # Extract solution and derivatives
            u = outputs["u"]
            u_t = outputs.get("u_t")
            u_x = outputs["u_x"][:, 0:1]  # Assuming 1D spatial domain
            u_xx = outputs.get("u_xx_0", torch.zeros_like(u)).unsqueeze(1)
            
            # Compute PDE residual: u_t + u⋅u_x - ν⋅u_xx = 0
            if u_t is not None:
                residual = u_t + u * u_x - viscosity * u_xx
            else:
                # Steady-state case: u⋅u_x - ν⋅u_xx = 0
                residual = u * u_x - viscosity * u_xx
            
            return residual
        
        # Initialize initial and boundary condition functions
        initial_conditions = {}
        if initial_condition is not None:
            initial_conditions["u0"] = initial_condition
        
        # Initialize parent class
        super().__init__(
            pde_residual_fn=burgers_residual,
            boundary_condition_fns=boundary_conditions,
            initial_condition_fns=initial_conditions,
            data_loss_fn=data_loss_fn,
            mse_reduction=mse_reduction
        )
        
        # Store parameters
        self.viscosity = viscosity


class AllenCahnLoss(PINNLoss):
    """
    Loss function for the Allen-Cahn equation.
    
    Implements loss components for the Allen-Cahn equation of the form:
    u_t = ε⋅u_xx + u - u³
    
    where:
    - u is the solution (phase field)
    - ε is the interface width parameter
    """
    
    def __init__(
        self,
        epsilon: float,
        initial_condition: Optional[Callable] = None,
        boundary_conditions: Optional[Dict[str, Callable]] = None,
        data_loss_fn: Optional[Callable] = None,
        mse_reduction: str = "mean"
    ):
        """
        Initialize the Allen-Cahn equation loss function.
        
        Args:
            epsilon: Interface width parameter (ε)
            initial_condition: Function that computes the initial condition error
            boundary_conditions: Dictionary of functions for boundary conditions
            data_loss_fn: Function to compute the data fitting loss
            mse_reduction: Reduction method for MSE loss
        """
        # Define PDE residual function
        def allen_cahn_residual(model, inputs, outputs):
            # Extract solution and derivatives
            u = outputs["u"]
            u_t = outputs.get("u_t")
            
            # Compute Laplacian (only x-direction for 1D case)
            dim = inputs["x"].shape[1]
            u_xx = torch.zeros_like(u)
            for i in range(dim):
                u_xx += outputs[f"u_xx_{i}"].unsqueeze(1)
            
            # Compute cubic term
            u_cubed = u ** 3
            
            # Compute PDE residual: u_t - ε⋅u_xx - u + u³ = 0
            if u_t is not None:
                residual = u_t - epsilon * u_xx - u + u_cubed
            else:
                # Steady-state case: -ε⋅u_xx - u + u³ = 0
                residual = -epsilon * u_xx - u + u_cubed
            
            return residual
        
        # Initialize initial and boundary condition functions
        initial_conditions = {}
        if initial_condition is not None:
            initial_conditions["u0"] = initial_condition
        
        # Initialize parent class
        super().__init__(
            pde_residual_fn=allen_cahn_residual,
            boundary_condition_fns=boundary_conditions,
            initial_condition_fns=initial_conditions,
            data_loss_fn=data_loss_fn,
            mse_reduction=mse_reduction
        )
        
        # Store parameters
        self.epsilon = epsilon
        
class WaveLoss(PINNLoss):
    """
    Loss function for the wave equation.
    
    Implements loss components for the wave equation of the form:
    u_tt = c²⋅∇²u
    
    where:
    - u is the displacement
    - c is the wave speed
    """
    
    def __init__(
        self,
        wave_speed: float,
        initial_displacement: Optional[Callable] = None,
        initial_velocity: Optional[Callable] = None,
        boundary_conditions: Optional[Dict[str, Callable]] = None,
        data_loss_fn: Optional[Callable] = None,
        mse_reduction: str = "mean"
    ):
        """
        Initialize the wave equation loss function.
        """
        # Definición de la función residual y otras implementaciones...