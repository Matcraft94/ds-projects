"""
utils/equations.py
Path: utils/equations.py
Date: 23-Mar-2025
Author: Lucy
Description: Implementation of physical equations for Physics-Informed Neural Networks (PINNs).
             Includes analytical solutions and boundary condition functions for various PDEs.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union, Any


class AdvectionDiffusionEquation:
    """
    Advection-Diffusion equation solver with analytical solutions.
    
    u_t + v * u_x = D * u_xx + s
    
    where:
    - u is the concentration/temperature
    - v is the velocity field
    - D is the diffusion coefficient
    - s is a source term
    """
    
    def __init__(
        self,
        diffusion_coefficient: float,
        velocity: float,
        domain_length: float = 1.0,
        source_term: float = 0.0
    ):
        """
        Initialize the advection-diffusion equation.
        
        Args:
            diffusion_coefficient: Diffusion coefficient (D)
            velocity: Velocity field (v)
            domain_length: Length of the spatial domain
            source_term: Source term (s)
        """
        self.D = diffusion_coefficient
        self.v = velocity
        self.L = domain_length
        self.s = source_term
    
    def gaussian_pulse_solution(
        self,
        x: Union[torch.Tensor, np.ndarray],
        t: Union[float, torch.Tensor, np.ndarray],
        x0: float = 0.0,
        sigma0: float = 0.1
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Analytical solution for Gaussian pulse initial condition.
        
        u(x, 0) = exp(-(x - x0)^2 / (2 * sigma0^2))
        
        Args:
            x: Spatial coordinates
            t: Time
            x0: Initial center of the Gaussian pulse
            sigma0: Initial width of the Gaussian pulse
            
        Returns:
            Solution u(x, t)
        """
        # Check input types
        is_torch = isinstance(x, torch.Tensor)
        
        # Convert to numpy if torch tensors
        if is_torch:
            x_np = x.detach().cpu().numpy()
            t_np = t.detach().cpu().item() if isinstance(t, torch.Tensor) else t
        else:
            x_np = x
            t_np = t
        
        # Compute sigma at time t
        sigma_t = np.sqrt(sigma0**2 + 2 * self.D * t_np)
        
        # Compute center position at time t
        x0_t = x0 + self.v * t_np
        
        # Compute solution
        exponent = -((x_np - x0_t)**2) / (2 * sigma_t**2)
        amplitude = sigma0 / sigma_t
        solution = amplitude * np.exp(exponent)
        
        # Return torch tensor if input was torch tensor
        if is_torch:
            return torch.from_numpy(solution).to(x.device)
        else:
            return solution
    
    def steady_state_solution(
        self,
        x: Union[torch.Tensor, np.ndarray],
        bc_left: float = 1.0,
        bc_right: float = 0.0
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Analytical solution for steady-state case with Dirichlet boundary conditions.
        
        u(0) = bc_left, u(L) = bc_right
        
        Args:
            x: Spatial coordinates
            bc_left: Boundary condition at x = 0
            bc_right: Boundary condition at x = L
            
        Returns:
            Steady-state solution u(x)
        """
        # Check input types
        is_torch = isinstance(x, torch.Tensor)
        
        # Convert to numpy if torch tensors
        if is_torch:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        # Peclet number
        Pe = self.v * self.L / self.D
        
        if abs(Pe) < 1e-10:
            # Diffusion-dominated case (v ≈ 0)
            solution = bc_left + (bc_right - bc_left) * x_np / self.L
        else:
            # Advection-diffusion case
            exponent = self.v * x_np / self.D
            denominator = np.exp(Pe) - 1
            solution = bc_left + (bc_right - bc_left) * (np.exp(exponent) - 1) / denominator
        
        # Add source term contribution if present
        if abs(self.s) > 1e-10:
            if abs(Pe) < 1e-10:
                # Diffusion-dominated case (v ≈ 0)
                source_contrib = (self.s / (2 * self.D)) * x_np * (self.L - x_np)
            else:
                # Advection-diffusion case
                source_contrib = (self.s / self.v) * (x_np - self.D / self.v * (1 - np.exp(self.v * x_np / self.D)))
            
            solution += source_contrib
        
        # Return torch tensor if input was torch tensor
        if is_torch:
            return torch.from_numpy(solution).to(x.device).float()
        else:
            return solution
    
    def get_ic_function(self, ic_type: str = "gaussian", **kwargs) -> Callable:
        """
        Get initial condition function.
        
        Args:
            ic_type: Type of initial condition ("gaussian" or "step")
            **kwargs: Additional parameters for the initial condition
            
        Returns:
            Initial condition function
        """
        if ic_type == "gaussian":
            x0 = kwargs.get("x0", 0.0)
            sigma0 = kwargs.get("sigma0", 0.1)
            
            def gaussian_ic(x: torch.Tensor) -> torch.Tensor:
                return torch.exp(-((x - x0)**2) / (2 * sigma0**2))
            
            return gaussian_ic
        
        elif ic_type == "step":
            x0 = kwargs.get("x0", 0.5)
            
            def step_ic(x: torch.Tensor) -> torch.Tensor:
                return torch.heaviside(x - x0, torch.ones_like(x))
            
            return step_ic
        
        else:
            raise ValueError(f"Initial condition type {ic_type} not supported")
    
    def get_bc_function(self, bc_type: str = "dirichlet", **kwargs) -> Dict[str, Callable]:
        """
        Get boundary condition functions.
        
        Args:
            bc_type: Type of boundary condition ("dirichlet" or "neumann")
            **kwargs: Additional parameters for the boundary conditions
            
        Returns:
            Dictionary of boundary condition functions
        """
        bc_left = kwargs.get("bc_left", 1.0)
        bc_right = kwargs.get("bc_right", 0.0)
        
        if bc_type == "dirichlet":
            def left_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_left = (x[:, 0] < 1e-5)
                u = outputs["u"]
                return u[mask_left] - bc_left
            
            def right_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_right = (torch.abs(x[:, 0] - self.L) < 1e-5)
                u = outputs["u"]
                return u[mask_right] - bc_right
            
            return {"left": left_bc, "right": right_bc}
        
        elif bc_type == "neumann":
            def left_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_left = (x[:, 0] < 1e-5)
                u_x = outputs["u_x"]
                return u_x[mask_left, 0] - bc_left
            
            def right_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_right = (torch.abs(x[:, 0] - self.L) < 1e-5)
                u_x = outputs["u_x"]
                return u_x[mask_right, 0] - bc_right
            
            return {"left": left_bc, "right": right_bc}
        
        else:
            raise ValueError(f"Boundary condition type {bc_type} not supported")


class BurgersEquation:
    """
    Burgers' equation solver with analytical solutions.
    
    u_t + u * u_x = nu * u_xx
    
    where:
    - u is the velocity
    - nu is the viscosity coefficient
    """
    
    def __init__(self, viscosity: float = 0.01):
        """
        Initialize the Burgers equation.
        
        Args:
            viscosity: Viscosity coefficient (nu)
        """
        self.nu = viscosity
    
    def cole_hopf_solution(
        self,
        x: Union[torch.Tensor, np.ndarray],
        t: Union[float, torch.Tensor, np.ndarray],
        ic_function: Callable = None
    ) -> Union[torch.Tensor, np.ndarray]:
        # Check input types
        is_torch = isinstance(x, torch.Tensor)
        
        # Convert to numpy if torch tensors
        if is_torch:
            x_np = x.detach().cpu().numpy()
            t_np = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
        else:
            x_np = x
            t_np = t
        
        # Ensure time is positive to avoid division by zero
        t_np = np.maximum(t_np, 1e-6)
        
        # Default initial condition
        if ic_function is None:
            def ic_function(x):
                return -np.sin(np.pi * x)
        
        # Simplified approximation for sinusoidal initial condition
        # Create output array with same shape as input
        result = np.zeros_like(x_np)
        
        # Flattened arrays for processing
        x_flat = x_np.flatten()
        t_flat = t_np.flatten()
        
        # Calculate solution for each point
        for i in range(len(x_flat)):
            idx = i % len(t_flat)  # Handle case where t has different length
            decay = np.exp(-np.pi**2 * self.nu * t_flat[idx])
            result.flat[i] = -np.sin(np.pi * x_flat[i]) * decay
        
        # Return torch tensor if input was torch tensor
        if is_torch:
            return torch.from_numpy(result).to(x.device).float()
        else:
            return result
    
    def sinusoidal_solution(
        self,
        x: Union[torch.Tensor, np.ndarray],
        t: Union[float, torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Analytical solution for sinusoidal initial condition.
        
        u(x, 0) = -sin(pi * x)
        
        Args:
            x: Spatial coordinates
            t: Time
            
        Returns:
            Solution u(x, t)
        """
        # Use Cole-Hopf solution with sine initial condition
        def sine_ic(x):
            if isinstance(x, torch.Tensor):
                return -torch.sin(torch.pi * x)
            else:
                return -np.sin(np.pi * x)
        
        return self.cole_hopf_solution(x, t, sine_ic)
    
    def shock_solution(
        self,
        x: Union[torch.Tensor, np.ndarray],
        t: Union[float, torch.Tensor, np.ndarray],
        u_left: float = 1.0,
        u_right: float = -1.0,
        x0: float = 0.0
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Self-similar solution for Riemann problem (shock).
        
        u(x, 0) = u_left if x < x0, u_right if x > x0
        
        Args:
            x: Spatial coordinates
            t: Time
            u_left: Left state
            u_right: Right state
            x0: Initial discontinuity position
            
        Returns:
            Solution u(x, t)
        """
        # Check input types
        is_torch = isinstance(x, torch.Tensor)
        
        # Convert to numpy if torch tensors
        if is_torch:
            x_np = x.detach().cpu().numpy()
            t_np = t.detach().cpu().item() if isinstance(t, torch.Tensor) else t
        else:
            x_np = x
            t_np = t
        
        # Ensure time is positive
        t_np = max(t_np, 1e-10)
        
        # Shock speed
        s = 0.5 * (u_left + u_right)
        
        # Shock position
        xs = x0 + s * t_np
        
        # Solution
        solution = np.where(x_np < xs, u_left, u_right)
        
        # Add diffusion layer around shock
        width = 4 * np.sqrt(self.nu * t_np)
        if width > 0:
            # Smooth transition using tanh
            solution = u_left + 0.5 * (u_right - u_left) * (1 + np.tanh((x_np - xs) / width))
        
        # Return torch tensor if input was torch tensor
        if is_torch:
            return torch.from_numpy(solution).to(x.device).float()
        else:
            return solution
    
    def get_ic_function(self, ic_type: str = "sine", **kwargs) -> Callable:
        """
        Get initial condition function.
        
        Args:
            ic_type: Type of initial condition ("sine" or "shock")
            **kwargs: Additional parameters for the initial condition
            
        Returns:
            Initial condition function
        """
        if ic_type == "sine":
            def sine_ic(model, inputs, outputs):
                x = inputs["x"]
                u = outputs["u"]
                return u + torch.sin(torch.pi * x[:, 0:1])
            
            return sine_ic
        
        elif ic_type == "shock":
            u_left = kwargs.get("u_left", 1.0)
            u_right = kwargs.get("u_right", -1.0)
            x0 = kwargs.get("x0", 0.0)
            
            def shock_ic(model, inputs, outputs):
                x = inputs["x"]
                u = outputs["u"]
                u_expected = torch.where(x[:, 0:1] < x0, 
                                         torch.ones_like(x[:, 0:1]) * u_left,
                                         torch.ones_like(x[:, 0:1]) * u_right)
                return u - u_expected
            
            return shock_ic
        
        else:
            raise ValueError(f"Initial condition type {ic_type} not supported")
    
    def get_bc_function(self, bc_type: str = "dirichlet", **kwargs) -> Dict[str, Callable]:
        """
        Get boundary condition functions.
        
        Args:
            bc_type: Type of boundary condition ("dirichlet" or "periodic")
            **kwargs: Additional parameters for the boundary conditions
            
        Returns:
            Dictionary of boundary condition functions
        """
        domain_length = kwargs.get("domain_length", 2.0)
        
        if bc_type == "dirichlet":
            bc_left = kwargs.get("bc_left", 0.0)
            bc_right = kwargs.get("bc_right", 0.0)
            
            def left_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_left = (x[:, 0] < 1e-5)
                u = outputs["u"]
                return u[mask_left] - bc_left
            
            def right_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_right = (torch.abs(x[:, 0] - domain_length) < 1e-5)
                u = outputs["u"]
                return u[mask_right] - bc_right
            
            return {"left": left_bc, "right": right_bc}
        
        elif bc_type == "periodic":
            def periodic_bc(model, inputs, outputs):
                x = inputs["x"]
                # Find boundary points
                mask_left = (x[:, 0] < 1e-5)
                mask_right = (torch.abs(x[:, 0] - domain_length) < 1e-5)
                
                # Get values and derivatives at boundaries
                u = outputs["u"]
                u_x = outputs["u_x"]
                
                # Value continuity: u(0) = u(L)
                value_diff = u[mask_left] - u[mask_right]
                
                # Derivative continuity: u_x(0) = u_x(L)
                deriv_diff = u_x[mask_left, 0:1] - u_x[mask_right, 0:1]
                
                return torch.cat([value_diff, deriv_diff], dim=0)
            
            return {"periodic": periodic_bc}
        
        else:
            raise ValueError(f"Boundary condition type {bc_type} not supported")

class AllenCahnEquation:
    """
    Allen-Cahn equation solver.
    
    u_t = ε∇²u + u - u³
    
    where:
    - u is the solution (phase field)
    - ε is the interface width parameter
    """
    
    def __init__(self, epsilon: float = 0.0001):
        """
        Initialize the Allen-Cahn equation.
        
        Args:
            epsilon: Interface width parameter (ε)
        """
        self.epsilon = epsilon
    
    def get_bc_function(self, bc_type: str = "periodic", **kwargs) -> Dict[str, Callable]:
        """
        Get boundary condition functions.
        
        Args:
            bc_type: Type of boundary condition ("dirichlet" or "periodic")
            **kwargs: Additional parameters for the boundary conditions
            
        Returns:
            Dictionary of boundary condition functions
        """
        domain_length = kwargs.get("domain_length", 2.0)
        
        if bc_type == "dirichlet":
            bc_left = kwargs.get("bc_left", 0.0)
            bc_right = kwargs.get("bc_right", 0.0)
            
            def left_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_left = (x[:, 0] < 1e-5)
                u = outputs["u"]
                return u[mask_left] - bc_left
            
            def right_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_right = (torch.abs(x[:, 0] - domain_length) < 1e-5)
                u = outputs["u"]
                return u[mask_right] - bc_right
            
            return {"left": left_bc, "right": right_bc}
        
        elif bc_type == "periodic":
            def periodic_bc(model, inputs, outputs):
                x = inputs["x"]
                # Find boundary points
                mask_left = (x[:, 0] < 1e-5)
                mask_right = (torch.abs(x[:, 0] - domain_length) < 1e-5)
                
                if not torch.any(mask_left) or not torch.any(mask_right):
                    return torch.zeros(0, device=x.device)
                
                # Get values and derivatives at boundaries
                u = outputs["u"]
                u_x = outputs["u_x"]
                
                # Value continuity: u(0) = u(L)
                value_diff = u[mask_left] - u[mask_right]
                
                # Derivative continuity: u_x(0) = u_x(L)
                deriv_diff = u_x[mask_left, 0:1] - u_x[mask_right, 0:1]
                
                return torch.cat([value_diff, deriv_diff], dim=0)
            
            return {"periodic": periodic_bc}
        
        else:
            raise ValueError(f"Boundary condition type {bc_type} not supported")
            
    def tanh_solution(
        self,
        x: Union[torch.Tensor, np.ndarray],
        t: Union[float, torch.Tensor, np.ndarray],
        x0: float = 0.0,
        width: float = 0.5
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Approximate solution for the Allen-Cahn equation with a tanh profile.
        
        This is an approximate solution that works for sufficiently small epsilon.
        
        Args:
            x: Spatial coordinates
            t: Time
            x0: Initial interface position
            width: Interface width
            
        Returns:
            Solution u(x, t)
        """
        # Check input types
        is_torch = isinstance(x, torch.Tensor)
        
        # Convert to numpy if torch tensors
        if is_torch:
            x_np = x.detach().cpu().numpy()
            t_np = t.detach().cpu().item() if isinstance(t, torch.Tensor) else t
        else:
            x_np = x
            t_np = t
        
        # For small epsilon, the interface moves with velocity v = -κ
        # where κ is the curvature (0 for 1D case)
        # So the interface position is approximately constant
        
        # Solution is approximately tanh((x-x0)/width)
        solution = np.tanh((x_np - x0) / width)
        
        # Return torch tensor if input was torch tensor
        if is_torch:
            return torch.from_numpy(solution).to(x.device).float()
        else:
            return solution


class WaveEquation:
    """
    Wave equation solver with analytical solutions.
    
    u_tt = c²∇²u
    
    where:
    - u is the displacement
    - c is the wave speed
    """
    
    def __init__(self, wave_speed: float = 1.0):
        """
        Initialize the wave equation.
        
        Args:
            wave_speed: Wave speed (c)
        """
        self.c = wave_speed
    
    def standing_wave_solution(
        self,
        x: Union[torch.Tensor, np.ndarray],
        t: Union[float, torch.Tensor, np.ndarray],
        domain_length: float = 1.0,
        modes: List[Tuple[int, float]] = [(1, 1.0)]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Analytical solution for standing wave with multiple modes.
        
        u(x, t) = sum_n a_n * sin(n*pi*x/L) * cos(n*pi*c*t/L)
        
        Args:
            x: Spatial coordinates
            t: Time
            domain_length: Length of the domain
            modes: List of (mode_number, amplitude) tuples
            
        Returns:
            Solution u(x, t)
        """
        # Check input types
        is_torch = isinstance(x, torch.Tensor)
        
        # Convert to numpy if torch tensors
        if is_torch:
            x_np = x.detach().cpu().numpy()
            t_np = t.detach().cpu().item() if isinstance(t, torch.Tensor) else t
        else:
            x_np = x
            t_np = t
        
        # Compute solution
        solution = np.zeros_like(x_np)
        
        for mode, amplitude in modes:
            # Compute wave number
            k = mode * np.pi / domain_length
            
            # Compute angular frequency
            omega = self.c * k
            
            # Add contribution from this mode
            solution += amplitude * np.sin(k * x_np) * np.cos(omega * t_np)
        
        # Return torch tensor if input was torch tensor
        if is_torch:
            return torch.from_numpy(solution).to(x.device).float()
        else:
            return solution
    
    def traveling_wave_solution(
        self,
        x: Union[torch.Tensor, np.ndarray],
        t: Union[float, torch.Tensor, np.ndarray],
        wave_number: float = np.pi,
        amplitude: float = 1.0,
        direction: int = 1
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Analytical solution for traveling wave.
        
        u(x, t) = A * sin(k*x - omega*t) for direction = 1 (right-traveling)
        u(x, t) = A * sin(k*x + omega*t) for direction = -1 (left-traveling)
        
        Args:
            x: Spatial coordinates
            t: Time
            wave_number: Wave number (k)
            amplitude: Wave amplitude (A)
            direction: Wave direction (1 for right, -1 for left)
            
        Returns:
            Solution u(x, t)
        """
        # Check input types
        is_torch = isinstance(x, torch.Tensor)
        
        # Convert to numpy if torch tensors
        if is_torch:
            x_np = x.detach().cpu().numpy()
            t_np = t.detach().cpu().item() if isinstance(t, torch.Tensor) else t
        else:
            x_np = x
            t_np = t
        
        # Compute angular frequency
        omega = self.c * wave_number
        
        # Compute solution
        if direction > 0:
            # Right-traveling wave
            solution = amplitude * np.sin(wave_number * x_np - omega * t_np)
        else:
            # Left-traveling wave
            solution = amplitude * np.sin(wave_number * x_np + omega * t_np)
        
        # Return torch tensor if input was torch tensor
        if is_torch:
            return torch.from_numpy(solution).to(x.device).float()
        else:
            return solution
    
    def get_ic_function(self, ic_type: str = "sine", **kwargs) -> Dict[str, Callable]:
        """
        Get initial condition functions.
        
        Args:
            ic_type: Type of initial condition ("sine", "gaussian", or "pulse")
            **kwargs: Additional parameters for the initial condition
            
        Returns:
            Dictionary of initial condition functions for displacement and velocity
        """
        domain_length = kwargs.get("domain_length", 1.0)
        
        if ic_type == "sine":
            # Standing wave initial condition
            modes = kwargs.get("modes", [(1, 1.0)])
            
            def u0_func(model, inputs, outputs):
                x = inputs["x"]
                u = outputs["u"]
                
                # Compute expected value at t=0
                u_expected = torch.zeros_like(u)
                for mode, amplitude in modes:
                    k = mode * np.pi / domain_length
                    u_expected += amplitude * torch.sin(k * x[:, 0:1])
                
                return u - u_expected
            
            def ut0_func(model, inputs, outputs):
                x = inputs["x"]
                u_t = outputs["u_t"]
                
                # Derivative at t=0 is zero for standing wave
                return u_t
            
            return {"u0": u0_func, "ut0": ut0_func}
        
        elif ic_type == "gaussian":
            x0 = kwargs.get("x0", 0.5)
            sigma = kwargs.get("sigma", 0.1)
            
            def u0_func(model, inputs, outputs):
                x = inputs["x"]
                u = outputs["u"]
                
                # Gaussian pulse initial condition
                u_expected = torch.exp(-((x[:, 0:1] - x0) / sigma)**2)
                
                return u - u_expected
            
            def ut0_func(model, inputs, outputs):
                # Derivative at t=0 is zero
                return outputs["u_t"]
            
            return {"u0": u0_func, "ut0": ut0_func}
        
        elif ic_type == "pulse":
            # Traveling pulse initial condition
            x0 = kwargs.get("x0", 0.5)
            sigma = kwargs.get("sigma", 0.1)
            direction = kwargs.get("direction", 1)
            
            def u0_func(model, inputs, outputs):
                x = inputs["x"]
                u = outputs["u"]
                
                # Gaussian pulse initial condition
                u_expected = torch.exp(-((x[:, 0:1] - x0) / sigma)**2)
                
                return u - u_expected
            
            def ut0_func(model, inputs, outputs):
                x = inputs["x"]
                u_t = outputs["u_t"]
                
                # Initial velocity for traveling pulse
                v0 = direction * self.c
                
                # Derivative of Gaussian: -2*(x-x0)/sigma^2 * exp(-(x-x0)^2/sigma^2)
                ut_expected = -v0 * 2 * (x[:, 0:1] - x0) / sigma**2 * torch.exp(-((x[:, 0:1] - x0) / sigma)**2)
                
                return u_t - ut_expected
            
            return {"u0": u0_func, "ut0": ut0_func}
        
        else:
            raise ValueError(f"Initial condition type {ic_type} not supported")
    
    def get_bc_function(self, bc_type: str = "dirichlet", **kwargs) -> Dict[str, Callable]:
        """
        Get boundary condition functions.
        
        Args:
            bc_type: Type of boundary condition ("dirichlet", "neumann", or "periodic")
            **kwargs: Additional parameters for the boundary conditions
            
        Returns:
            Dictionary of boundary condition functions
        """
        domain_length = kwargs.get("domain_length", 1.0)
        
        if bc_type == "dirichlet":
            # Fixed displacement at boundaries
            def left_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_left = (x[:, 0] < 1e-5)
                u = outputs["u"]
                return u[mask_left]
            
            def right_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_right = (torch.abs(x[:, 0] - domain_length) < 1e-5)
                u = outputs["u"]
                return u[mask_right]
            
            return {"left": left_bc, "right": right_bc}
        
        elif bc_type == "neumann":
            # Zero derivative at boundaries
            def left_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_left = (x[:, 0] < 1e-5)
                u_x = outputs["u_x"]
                return u_x[mask_left, 0]
            
            def right_bc(model, inputs, outputs):
                x = inputs["x"]
                mask_right = (torch.abs(x[:, 0] - domain_length) < 1e-5)
                u_x = outputs["u_x"]
                return u_x[mask_right, 0]
            
            return {"left": left_bc, "right": right_bc}
        
        elif bc_type == "periodic":
            def periodic_bc(model, inputs, outputs):
                x = inputs["x"]
                
                # Find boundary points
                mask_left = (x[:, 0] < 1e-5)
                mask_right = (torch.abs(x[:, 0] - domain_length) < 1e-5)
                
                if not torch.any(mask_left) or not torch.any(mask_right):
                    return torch.zeros(0, device=x.device)
                
                # Get values and derivatives at boundaries
                u = outputs["u"]
                u_x = outputs["u_x"]
                
                # Value continuity: u(0) = u(L)
                value_diff = u[mask_left] - u[mask_right]
                
                # Derivative continuity: u_x(0) = u_x(L)
                deriv_diff = u_x[mask_left, 0:1] - u_x[mask_right, 0:1]
                
                return torch.cat([value_diff, deriv_diff], dim=0)
            
            return {"periodic": periodic_bc}
        
        else:
            raise ValueError(f"Boundary condition type {bc_type} not supported")