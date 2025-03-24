"""
main.py
Path: main.py
Date: 23-Mar-2025
Author: Lucy
Description: Main script for running Physics-Informed Neural Networks (PINNs)
             and Self-Adaptive PINNs (SA-PINNs) experiments. This script provides
             a command-line interface to configure and execute various PDE experiments.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

# Import models
from models.pinn_base import PINN
from models.sa_pinn import SAPINN

# Import training utilities
from training.trainer import PINNTrainer
from training.loss_functions import (
    PINNLoss,
    AdvectionDiffusionLoss,
    BurgersLoss,
    AllenCahnLoss,
    WaveLoss,
    HelmholtzLoss
)

# Import data utilities
from utils.data_generator import (
    generate_random_points,
    generate_boundary_points,
    generate_initial_points,
    PINNDataset
)

# Import visualization utilities
from utils.visualization import (
    plot_training_history,
    plot_loss_components,
    plot_weight_evolution,
    plot_1d_solution,
    plot_2d_solution,
    plot_solution_error,
    plot_adaptive_weights,
    create_solution_animation
)

# Import equation utilities
from utils.equations import (
    AdvectionDiffusionEquation,
    BurgersEquation,
    AllenCahnEquation,
    WaveEquation
)


def create_experiment_dir(base_dir="./experiments"):
    """Create a new directory for the experiment results."""
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a unique directory name based on timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    return experiment_dir, checkpoints_dir, plots_dir


def save_config(config, experiment_dir):
    """Save experiment configuration to a file."""
    config_path = os.path.join(experiment_dir, "config.json")
    
    # Convert non-serializable items to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool, list, dict, tuple)) or value is None:
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(config_path, "w") as f:
        json.dump(serializable_config, f, indent=4)


def setup_pde_experiment(args):
    """
    Set up a PDE experiment based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with experiment configuration
    """
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")
    
    # Create experiment directory
    experiment_dir, checkpoints_dir, plots_dir = create_experiment_dir(args.output_dir)
    print(f"Experiment directory: {experiment_dir}")
    
    # Create configuration dictionary
    config = {
        "experiment_dir": experiment_dir,
        "checkpoints_dir": checkpoints_dir,
        "plots_dir": plots_dir,
        "equation": args.equation,
        "model_type": args.model_type,
        "device": device,
        "seed": args.seed
    }
    
    # Set up model and equation-specific parameters
    if args.equation == "advection_diffusion":
        config.update(setup_advection_diffusion(args, device))
    elif args.equation == "burgers":
        config.update(setup_burgers(args, device))
    elif args.equation == "allen_cahn":
        config.update(setup_allen_cahn(args, device))
    elif args.equation == "wave":
        config.update(setup_wave(args, device))
    elif args.equation == "helmholtz":
        config.update(setup_helmholtz(args, device))
    else:
        raise ValueError(f"Equation {args.equation} not supported")
    
    # Save configuration
    save_config(config, experiment_dir)
    
    return config


def setup_advection_diffusion(args, device):
    """Set up advection-diffusion equation experiment."""
    # Domain bounds
    if args.domain_bounds:
        x_min, x_max = args.domain_bounds[0], args.domain_bounds[1]
        t_min, t_max = args.domain_bounds[2], args.domain_bounds[3] if len(args.domain_bounds) > 3 else (0.0, 1.0)
    else:
        x_min, x_max = 0.0, 1.0
        t_min, t_max = 0.0, 1.0
    
    domain_bounds = {"x": (x_min, x_max), "t": (t_min, t_max)}
    
    # Equation parameters
    diffusion_coefficient = args.diffusion_coef if args.diffusion_coef else 0.01
    velocity = args.velocity if args.velocity else 1.0
    
    # Create equation object
    equation = AdvectionDiffusionEquation(
        diffusion_coefficient=diffusion_coefficient,
        velocity=velocity,
        domain_length=x_max - x_min
    )
    
    # Generate data points
    residual_points = generate_random_points(
        domain_bounds=domain_bounds,
        n_points=args.n_residual,
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    boundary_points = generate_boundary_points(
        domain_bounds=domain_bounds,
        n_points_per_boundary=args.n_boundary // 4,  # Divide by number of boundaries
        boundaries=[("x", "min"), ("x", "max"), ("t", "min"), ("t", "max")],
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    initial_points = generate_initial_points(
        domain_bounds=domain_bounds,
        time_var="t",
        n_points=args.n_initial,
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    # Create initial and boundary condition functions
    ic_func = equation.get_ic_function(ic_type="gaussian", x0=0.5, sigma0=0.1)
    bc_funcs = equation.get_bc_function(bc_type="dirichlet", bc_left=0.0, bc_right=0.0)
    
    # Create loss function
    loss_fn = AdvectionDiffusionLoss(
        diffusion_coefficient=diffusion_coefficient,
        velocity_field=velocity,
        initial_condition=ic_func,
        boundary_conditions=bc_funcs
    )
    
    # Create dataset
    dataset = PINNDataset(
        residual_points=residual_points,
        boundary_points=boundary_points,
        initial_points=initial_points,
        batch_size=args.batch_size
    )
    
    # Create model
    if args.model_type == "pinn":
        model = PINN(
            input_dim=2,  # x, t
            output_dim=1,  # u
            hidden_layers=args.hidden_layers or [20, 20, 20, 20],
            activation=args.activation or "tanh"
        ).to(device)
    elif args.model_type == "sa-pinn":
        model = SAPINN(
            input_dim=2,  # x, t
            output_dim=1,  # u
            hidden_layers=args.hidden_layers or [20, 20, 20, 20],
            n_residual=len(residual_points["x"]),
            n_boundary=len(boundary_points["x"]),
            n_initial=len(initial_points["x"]),
            mask_type=args.mask_type or "polynomial",
            activation=args.activation or "tanh",
            device=device
        ).to(device)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")
    
    # Create exact solution function for evaluation
    def exact_solution_func(x, t):
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
            t_np = t.cpu().numpy() if isinstance(t, torch.Tensor) else t
        else:
            x_np, t_np = x, t
        
        return equation.gaussian_pulse_solution(x_np, t_np, x0=0.5, sigma0=0.1)
    
    # Test points for error evaluation
    test_x = torch.linspace(x_min, x_max, 100, device=device).unsqueeze(1)
    test_t = torch.linspace(t_min, t_max, 100, device=device).unsqueeze(1)
    X, T = torch.meshgrid(test_x.squeeze(), test_t.squeeze(), indexing="ij")
    X_flat = X.flatten().unsqueeze(1)
    T_flat = T.flatten().unsqueeze(1)
    
    test_input = torch.cat([X_flat, T_flat], dim=1)
    test_output = torch.tensor(exact_solution_func(X_flat, T_flat), device=device)
    
    return {
        "domain_bounds": domain_bounds,
        "equation_params": {
            "viscosity": viscosity
        },
        "dataset": dataset,
        "model": model,
        "loss_fn": loss_fn,
        "exact_solution": exact_solution_func,
        "test_data": (test_input, test_output)
    }


def setup_allen_cahn(args, device):
    """Set up Allen-Cahn equation experiment."""
    # Domain bounds
    if args.domain_bounds:
        x_min, x_max = args.domain_bounds[0], args.domain_bounds[1]
        t_min, t_max = args.domain_bounds[2], args.domain_bounds[3] if len(args.domain_bounds) > 3 else (0.0, 1.0)
    else:
        x_min, x_max = -1.0, 1.0
        t_min, t_max = 0.0, 1.0
    
    domain_bounds = {"x": (x_min, x_max), "t": (t_min, t_max)}
    
    # Equation parameters
    epsilon = args.epsilon if args.epsilon else 0.0001
    
    # Create equation object
    equation = AllenCahnEquation(epsilon=epsilon)
    
    # Generate data points
    residual_points = generate_random_points(
        domain_bounds=domain_bounds,
        n_points=args.n_residual,
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    # For Allen-Cahn, we use periodic boundary conditions
    boundary_points = generate_boundary_points(
        domain_bounds=domain_bounds,
        n_points_per_boundary=args.n_boundary // 2,  # Only need points at x = -1 and x = 1
        boundaries=[("x", "min"), ("x", "max")],
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    initial_points = generate_initial_points(
        domain_bounds=domain_bounds,
        time_var="t",
        n_points=args.n_initial,
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    # Create initial and boundary condition functions
    # Custom initial condition for Allen-Cahn: u(x, 0) = x² * cos(πx)
    def custom_ic(model, inputs, outputs):
        x = inputs["x"]
        u = outputs["u"]
        u_expected = x[:, 0:1]**2 * torch.cos(torch.pi * x[:, 0:1])
        return u - u_expected
    
    # Periodic boundary conditions
    bc_funcs = equation.get_bc_function(
        bc_type="periodic",
        domain_length=x_max - x_min
    )
    
    # Create loss function
    loss_fn = AllenCahnLoss(
        epsilon=epsilon,
        initial_condition=custom_ic,
        boundary_conditions=bc_funcs
    )
    
    # Create dataset
    dataset = PINNDataset(
        residual_points=residual_points,
        boundary_points=boundary_points,
        initial_points=initial_points,
        batch_size=args.batch_size
    )
    
    # Create model
    if args.model_type == "pinn":
        model = PINN(
            input_dim=2,  # x, t
            output_dim=1,  # u
            hidden_layers=args.hidden_layers or [128, 128, 128, 128],
            activation=args.activation or "tanh"
        ).to(device)
    elif args.model_type == "sa-pinn":
        # For SA-PINN, initialize weights with higher values for initial condition
        init_range = {
            "residual": (0.0, 1.0),
            "boundary": (0.0, 1.0),
            "initial": (0.0, 100.0)  # Higher weight for initial condition
        }
        
        model = SAPINN(
            input_dim=2,  # x, t
            output_dim=1,  # u
            hidden_layers=args.hidden_layers or [128, 128, 128, 128],
            n_residual=len(residual_points["x"]),
            n_boundary=len(boundary_points["x"]),
            n_initial=len(initial_points["x"]),
            mask_type=args.mask_type or "polynomial",
            init_range=init_range,
            activation=args.activation or "tanh",
            device=device
        ).to(device)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")
    
    # No exact solution for Allen-Cahn, but we can run a high-fidelity numerical solver
    # Here we return None to indicate no exact solution available
    exact_solution_func = None
    test_data = None
    
    return {
        "domain_bounds": domain_bounds,
        "equation_params": {
            "epsilon": epsilon
        },
        "dataset": dataset,
        "model": model,
        "loss_fn": loss_fn,
        "exact_solution": exact_solution_func,
        "test_data": test_data
    }


def setup_wave(args, device):
    """Set up wave equation experiment."""
    # Domain bounds
    if args.domain_bounds:
        x_min, x_max = args.domain_bounds[0], args.domain_bounds[1]
        t_min, t_max = args.domain_bounds[2], args.domain_bounds[3] if len(args.domain_bounds) > 3 else (0.0, 1.0)
    else:
        x_min, x_max = 0.0, 1.0
        t_min, t_max = 0.0, 1.0
    
    domain_bounds = {"x": (x_min, x_max), "t": (t_min, t_max)}
    
    # Equation parameters
    wave_speed = args.wave_speed if args.wave_speed else 1.0
    
    # Create equation object
    equation = WaveEquation(wave_speed=wave_speed)
    
    # Generate data points
    residual_points = generate_random_points(
        domain_bounds=domain_bounds,
        n_points=args.n_residual,
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    boundary_points = generate_boundary_points(
        domain_bounds=domain_bounds,
        n_points_per_boundary=args.n_boundary // 4,  # Divide by number of boundaries
        boundaries=[("x", "min"), ("x", "max"), ("t", "min"), ("t", "max")],
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    initial_points = generate_initial_points(
        domain_bounds=domain_bounds,
        time_var="t",
        n_points=args.n_initial,
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    # Create initial and boundary condition functions
    ic_funcs = equation.get_ic_function(
        ic_type="sine",
        domain_length=x_max - x_min,
        modes=[(1, 1.0), (4, 0.5)]  # Multiple modes as in paper
    )
    
    bc_funcs = equation.get_bc_function(
        bc_type="dirichlet",
        domain_length=x_max - x_min
    )
    
    # Create loss function
    loss_fn = WaveLoss(
        wave_speed=wave_speed,
        initial_displacement=ic_funcs["u0"],
        initial_velocity=ic_funcs["ut0"],
        boundary_conditions=bc_funcs
    )
    
    # Create dataset
    dataset = PINNDataset(
        residual_points=residual_points,
        boundary_points=boundary_points,
        initial_points=initial_points,
        batch_size=args.batch_size
    )
    
    # Create model
    if args.model_type == "pinn":
        model = PINN(
            input_dim=2,  # x, t
            output_dim=1,  # u
            hidden_layers=args.hidden_layers or [100, 100, 100, 100, 100],
            activation=args.activation or "tanh"
        ).to(device)
    elif args.model_type == "sa-pinn":
        model = SAPINN(
            input_dim=2,  # x, t
            output_dim=1,  # u
            hidden_layers=args.hidden_layers or [100, 100, 100, 100, 100],
            n_residual=len(residual_points["x"]),
            n_boundary=len(boundary_points["x"]),
            n_initial=len(initial_points["x"]),
            mask_type=args.mask_type or "polynomial",
            activation=args.activation or "tanh",
            device=device
        ).to(device)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")
    
    # Create exact solution function for evaluation
    def exact_solution_func(x, t):
        return equation.standing_wave_solution(
            x=x, 
            t=t, 
            domain_length=x_max - x_min,
            modes=[(1, 1.0), (4, 0.5)]
        )
    
    # Test points for error evaluation
    test_x = torch.linspace(x_min, x_max, 100, device=device).unsqueeze(1)
    test_t = torch.linspace(t_min, t_max, 100, device=device).unsqueeze(1)
    X, T = torch.meshgrid(test_x.squeeze(), test_t.squeeze(), indexing="ij")
    X_flat = X.flatten().unsqueeze(1)
    T_flat = T.flatten().unsqueeze(1)
    
    test_input = torch.cat([X_flat, T_flat], dim=1)
    test_output = torch.tensor(exact_solution_func(X_flat, T_flat), device=device)
    
    return {
        "domain_bounds": domain_bounds,
        "equation_params": {
            "wave_speed": wave_speed
        },
        "dataset": dataset,
        "model": model,
        "loss_fn": loss_fn,
        "exact_solution": exact_solution_func,
        "test_data": (test_input, test_output)
    }


def setup_helmholtz(args, device):
    """Set up Helmholtz equation experiment."""
    # Domain bounds (only spatial variables for Helmholtz)
    if args.domain_bounds:
        x_min, x_max = args.domain_bounds[0], args.domain_bounds[1]
        y_min, y_max = args.domain_bounds[2], args.domain_bounds[3] if len(args.domain_bounds) > 3 else (-1.0, 1.0)
    else:
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
    
    domain_bounds = {"x": (x_min, x_max), "y": (y_min, y_max)}
    
    # Equation parameters
    wave_number = args.wave_number if args.wave_number else 1.0
    
    # Create source term for manufactured solution
    # We use manufactured solution u(x,y) = sin(a₁πx)sin(a₂πy)
    a1 = 1
    a2 = 4
    
    def source_term(coords):
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        
        # q(x,y) = -(a₁π)² sin(a₁πx)sin(a₂πy) - (a₂π)² sin(a₁πx)sin(a₂πy) + k² sin(a₁πx)sin(a₂πy)
        return -(a1 * np.pi)**2 * torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * y) \
               -(a2 * np.pi)**2 * torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * y) \
               + wave_number**2 * torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * y)
    
    # Generate data points
    residual_points = generate_random_points(
        domain_bounds=domain_bounds,
        n_points=args.n_residual,
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    boundary_points = generate_boundary_points(
        domain_bounds=domain_bounds,
        n_points_per_boundary=args.n_boundary // 4,  # 4 boundaries for 2D
        boundaries=[("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")],
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    # Create boundary condition functions (homogeneous Dirichlet)
    def left_bc(model, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]
        mask_left = (torch.abs(x[:, 0] - x_min) < 1e-5)
        u = outputs["u"]
        return u[mask_left]
    
    def right_bc(model, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]
        mask_right = (torch.abs(x[:, 0] - x_max) < 1e-5)
        u = outputs["u"]
        return u[mask_right]
    
    def bottom_bc(model, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]
        mask_bottom = (torch.abs(y[:, 0] - y_min) < 1e-5)
        u = outputs["u"]
        return u[mask_bottom]
    
    def top_bc(model, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]
        mask_top = (torch.abs(y[:, 0] - y_max) < 1e-5)
        u = outputs["u"]
        return u[mask_top]
    
    bc_funcs = {
        "left": left_bc,
        "right": right_bc,
        "bottom": bottom_bc,
        "top": top_bc
    }
    
    # Create loss function
    loss_fn = HelmholtzLoss(
        wave_number=wave_number,
        source_term=source_term,
        boundary_conditions=bc_funcs
    )
    
    # Create dataset (no initial points for Helmholtz)
    dataset = PINNDataset(
        residual_points=residual_points,
        boundary_points=boundary_points,
        batch_size=args.batch_size
    )
    
    # Create model
    if args.model_type == "pinn":
        model = PINN(
            input_dim=2,  # x, y
            output_dim=1,  # u
            hidden_layers=args.hidden_layers or [50, 50, 50, 50],
            activation=args.activation or "tanh"
        ).to(device)
    elif args.model_type == "sa-pinn":
        model = SAPINN(
            input_dim=2,  # x, y
            output_dim=1,  # u
            hidden_layers=args.hidden_layers or [50, 50, 50, 50],
            n_residual=len(residual_points["x"]),
            n_boundary=len(boundary_points["x"]),
            n_initial=0,  # No initial condition for Helmholtz
            mask_type=args.mask_type or "polynomial",
            activation=args.activation or "tanh",
            device=device
        ).to(device)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")
    
    # Create exact solution function for evaluation
    def exact_solution_func(x, y):
        return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y)
    
    # Test points for error evaluation
    test_x = torch.linspace(x_min, x_max, 100, device=device)
    test_y = torch.linspace(y_min, y_max, 100, device=device)
    X, Y = torch.meshgrid(test_x, test_y, indexing="ij")
    X_flat = X.flatten().unsqueeze(1)
    Y_flat = Y.flatten().unsqueeze(1)
    
    test_input = torch.cat([X_flat, Y_flat], dim=1)
    test_output = torch.tensor(exact_solution_func(X_flat.cpu().numpy(), Y_flat.cpu().numpy()), 
                              device=device).float()
    
    return {
        "domain_bounds": domain_bounds,
        "equation_params": {
            "wave_number": wave_number,
            "a1": a1,
            "a2": a2
        },
        "dataset": dataset,
        "model": model,
        "loss_fn": loss_fn,
        "exact_solution": exact_solution_func,
        "test_data": (test_input, test_output)
    }


def run_experiment(config):
    """
    Run the PDE experiment with the specified configuration.
    
    Args:
        config: Dictionary with experiment configuration
    """
    # Extract configuration
    experiment_dir = config["experiment_dir"]
    checkpoints_dir = config["checkpoints_dir"]
    plots_dir = config["plots_dir"]
    device = config["device"]
    model = config["model"]
    dataset = config["dataset"]
    loss_fn = config["loss_fn"]
    exact_solution = config["exact_solution"]
    test_data = config["test_data"]
    
    # Create dataloader
    train_dataloader = dataset.get_dataloader(shuffle=True)
    
    # Configure optimizer
    if isinstance(model, SAPINN):
        optimizer_config = {
            "network": {"type": "adam", "lr": 1e-3},
            "weights": {"type": "adam", "lr": 1e-2}
        }
    else:
        optimizer_config = {
            "network": {"type": "adam", "lr": 1e-3}
        }
    
    # Configure learning rate scheduler
    scheduler_config = {
        "network": {"type": "step", "step_size": 1000, "gamma": 0.5}
    }
    
    # Create trainer
    trainer = PINNTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        device=device,
        checkpoint_dir=checkpoints_dir
    )
    
    # Setup early stopping
    early_stopping = {
        "patience": 100,
        "min_delta": 1e-5,
        "monitor": "l2_error" if test_data is not None else "train_loss"
    }
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    
    logs = trainer.train(
        train_dataloader=train_dataloader,
        n_epochs=10000,
        test_data=test_data,
        eval_freq=100,
        checkpoint_freq=1000,
        early_stopping=early_stopping
    )
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # Plot training history
    print("Generating plots...")
    plot_training_history(
        logs=logs,
        save_path=os.path.join(plots_dir, "training_history.png")
    )
    
    # Plot loss components
    plot_loss_components(
        logs=logs,
        save_path=os.path.join(plots_dir, "loss_components.png")
    )
    
    # Plot adaptive weights if using SA-PINN
    if isinstance(model, SAPINN):
        # Extract weight stats from logs
        if "weight_stats" in logs:
            plot_weight_evolution(
                weight_history=logs["weight_stats"],
                save_path=os.path.join(plots_dir, "weight_evolution.png")
            )
    
    # Plot solution and error if exact solution is available
    if exact_solution is not None:
        domain_bounds = config["domain_bounds"]
        
        if "y" in domain_bounds:  # 2D problem
            x_domain = domain_bounds["x"]
            y_domain = domain_bounds["y"]
            
            # Plot 2D solution
            plot_2d_solution(
                model=model,
                x_domain=x_domain,
                y_domain=y_domain,
                exact_solution=exact_solution,
                plot_type="both",
                save_path=os.path.join(plots_dir, "solution_2d.png")
            )
        else:  # 1D problem
            x_domain = domain_bounds["x"]
            
            if "t" in domain_bounds:  # Time-dependent
                t_domain = domain_bounds["t"]
                
                # Plot solution at different times
                for t in np.linspace(t_domain[0], t_domain[1], 5):
                    plot_1d_solution(
                        model=model,
                        x_domain=x_domain,
                        t=t,
                        exact_solution=exact_solution,
                        title=f"Solution at t = {t:.2f}",
                        save_path=os.path.join(plots_dir, f"solution_t{t:.2f}.png")
                    )
                
                # Create animation
                create_solution_animation(
                    model=model,
                    x_domain=x_domain,
                    t_domain=t_domain,
                    exact_solution=exact_solution,
                    save_path=os.path.join(plots_dir, "solution_animation.gif")
                )
                
                # Plot error
                plot_solution_error(
                    model=model,
                    x_domain=x_domain,
                    t_domain=t_domain,
                    exact_solution=exact_solution,
                    save_path=os.path.join(plots_dir, "solution_error.png")
                )
            else:  # Steady-state
                plot_1d_solution(
                    model=model,
                    x_domain=x_domain,
                    exact_solution=exact_solution,
                    save_path=os.path.join(plots_dir, "solution.png")
                )
    
    # Save final model
    trainer.save_checkpoint("final_model.pt")
    
    print(f"Experiment completed. Results saved to {experiment_dir}")


def main():
    """Main function to parse arguments and run experiment."""
    parser = argparse.ArgumentParser(description="Run Physics-Informed Neural Networks experiments")
    
    # General arguments
    parser.add_argument("--equation", type=str, default="burgers",
                        choices=["advection_diffusion", "burgers", "allen_cahn", "wave", "helmholtz"],
                        help="PDE to solve")
    parser.add_argument("--model_type", type=str, default="pinn",
                        choices=["pinn", "sa-pinn"],
                        help="Type of PINN model to use")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Directory to save experiment results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id to use (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Data generation arguments
    parser.add_argument("--n_residual", type=int, default=10000,
                        help="Number of residual points")
    parser.add_argument("--n_boundary", type=int, default=200,
                        help="Number of boundary points")
    parser.add_argument("--n_initial", type=int, default=100,
                        help="Number of initial points")
    parser.add_argument("--sampling", type=str, default="uniform",
                        choices=["uniform", "latin", "sobol", "halton", "grid"],
                        help="Sampling method for points")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training (None for full-batch)")
    parser.add_argument("--domain_bounds", type=float, nargs="+",
                        help="Domain bounds [x_min, x_max, t_min, t_max] or [x_min, x_max, y_min, y_max]")
    
    # Model arguments
    parser.add_argument("--hidden_layers", type=int, nargs="+",
                        help="Hidden layer sizes")
    parser.add_argument("--activation", type=str, default="tanh",
                        choices=["tanh", "relu", "sigmoid", "sin"],
                        help="Activation function")
    parser.add_argument("--mask_type", type=str, default="polynomial",
                        choices=["polynomial", "sigmoid", "exponential"],
                        help="Type of mask function for SA-PINN")
    
    # Equation-specific arguments
    parser.add_argument("--diffusion_coef", type=float,
                        help="Diffusion coefficient for advection-diffusion equation")
    parser.add_argument("--velocity", type=float,
                        help="Velocity for advection-diffusion equation")
    parser.add_argument("--viscosity", type=float,
                        help="Viscosity coefficient for Burgers equation")
    parser.add_argument("--epsilon", type=float,
                        help="Interface width parameter for Allen-Cahn equation")
    parser.add_argument("--wave_speed", type=float,
                        help="Wave speed for wave equation")
    parser.add_argument("--wave_number", type=float,
                        help="Wave number for Helmholtz equation")
    
    args = parser.parse_args()
    
    # Set up experiment
    config = setup_pde_experiment(args)
    
    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()



def setup_burgers(args, device):
    """Set up Burgers equation experiment."""
    # Domain bounds
    if args.domain_bounds:
        x_min, x_max = args.domain_bounds[0], args.domain_bounds[1]
        t_min, t_max = args.domain_bounds[2], args.domain_bounds[3] if len(args.domain_bounds) > 3 else (0.0, 1.0)
    else:
        x_min, x_max = -1.0, 1.0
        t_min, t_max = 0.0, 1.0
    
    domain_bounds = {"x": (x_min, x_max), "t": (t_min, t_max)}
    
    # Equation parameters
    viscosity = args.viscosity if args.viscosity else 0.01/np.pi
    
    # Create equation object
    equation = BurgersEquation(viscosity=viscosity)
    
    # Generate data points
    residual_points = generate_random_points(
        domain_bounds=domain_bounds,
        n_points=args.n_residual,
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    boundary_points = generate_boundary_points(
        domain_bounds=domain_bounds,
        n_points_per_boundary=args.n_boundary // 4,  # Divide by number of boundaries
        boundaries=[("x", "min"), ("x", "max"), ("t", "min"), ("t", "max")],
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    initial_points = generate_initial_points(
        domain_bounds=domain_bounds,
        time_var="t",
        n_points=args.n_initial,
        sampling=args.sampling,
        seed=args.seed,
        device=device
    )
    
    # Create initial and boundary condition functions
    ic_func = equation.get_ic_function(ic_type="sine")
    bc_funcs = equation.get_bc_function(
        bc_type="dirichlet", 
        domain_length=x_max - x_min,
        bc_left=0.0, 
        bc_right=0.0
    )
    
    # Create loss function
    loss_fn = BurgersLoss(
        viscosity=viscosity,
        initial_condition=ic_func,
        boundary_conditions=bc_funcs
    )
    
    # Create dataset
    dataset = PINNDataset(
        residual_points=residual_points,
        boundary_points=boundary_points,
        initial_points=initial_points,
        batch_size=args.batch_size
    )
    
    # Create model
    if args.model_type == "pinn":
        model = PINN(
            input_dim=2,  # x, t
            output_dim=1,  # u
            hidden_layers=args.hidden_layers or [20, 20, 20, 20, 20, 20, 20, 20],
            activation=args.activation or "tanh"
        ).to(device)
    elif args.model_type == "sa-pinn":
        model = SAPINN(
            input_dim=2,  # x, t
            output_dim=1,  # u
            hidden_layers=args.hidden_layers or [20, 20, 20, 20, 20, 20, 20, 20],
            n_residual=len(residual_points["x"]),
            n_boundary=len(boundary_points["x"]),
            n_initial=len(initial_points["x"]),
            mask_type=args.mask_type or "polynomial",
            activation=args.activation or "tanh",
            device=device
        ).to(device)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")
    
    # Create exact solution function for evaluation (using Cole-Hopf transform)
    def exact_solution_func(x, t):
        # Define sine initial condition
        def sine_ic(x):
            return -np.sin(np.pi * x)
        
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
            t_np = t.cpu().numpy() if isinstance(t, torch.Tensor) else t
        else:
            x_np, t_np = x, t
        
        return equation.cole_hopf_solution(x_np, t_np, sine_ic)
    
    # Test points for error evaluation
    test_x = torch.linspace(x_min, x_max, 100, device=device).unsqueeze(1)
    test_t = torch.linspace(t_min, t_max, 100, device=device).unsqueeze(1)
    X, T = torch.meshgrid(test_x.squeeze(), test_t.squeeze(), indexing="ij")
    X_flat = X.flatten().unsqueeze(1)
    T_flat = T.flatten().unsqueeze(1)
    
    test_input = torch.cat([X_flat, T_flat], dim=1)
    test