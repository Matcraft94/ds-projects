"""
utils/visualization.py
Path: utils/visualization.py
Date: 23-Mar-2025
Author: Lucy
Description: Visualization tools for Physics-Informed Neural Networks (PINNs)
             and Self-Adaptive PINNs (SA-PINNs). Includes functions to plot
             solution fields, error distributions, convergence, and adaptive weights.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
import os
from mpl_toolkits.mplot3d import Axes3D


def plot_training_history(
    logs: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    log_scale: bool = False,
    save_path: Optional[str] = None
):
    """
    Plot training history including losses and errors.
    
    Args:
        logs: Dictionary with training logs
        figsize: Figure size
        log_scale: Whether to use logarithmic scale for y-axis
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot total losses
    ax = axes[0]
    epochs = np.arange(1, len(logs["train_loss"]) + 1)
    
    ax.plot(epochs, logs["train_loss"], label="Training Loss", color="blue")
    
    if "val_loss" in logs and logs["val_loss"]:
        # Get indices where validation occurred
        val_epochs = np.arange(1, len(logs["val_loss"]) + 1) * (len(logs["train_loss"]) // len(logs["val_loss"]))
        ax.plot(val_epochs, logs["val_loss"], label="Validation Loss", color="red", marker="o")
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot L2 error if available
    ax = axes[1]
    
    if "l2_error" in logs and logs["l2_error"]:
        # Get indices where L2 error was computed
        error_epochs = np.arange(1, len(logs["l2_error"]) + 1) * (len(logs["train_loss"]) // len(logs["l2_error"]))
        ax.plot(error_epochs, logs["l2_error"], label="L2 Error", color="green", marker="o")
        
        if log_scale:
            ax.set_yscale("log")
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Relative L2 Error")
        ax.set_title("Model Error")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # If no L2 error, plot loss components instead
        if "train_loss_components" in logs and logs["train_loss_components"]:
            components = logs["train_loss_components"][0].keys()
            for component in components:
                component_values = [log[component] for log in logs["train_loss_components"]]
                ax.plot(epochs, component_values, label=f"{component}")
            
            if log_scale:
                ax.set_yscale("log")
            
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss Components")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_loss_components(
    logs: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 6),
    log_scale: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot individual loss components during training.
    
    Args:
        logs: Dictionary with training logs
        figsize: Figure size
        log_scale: Whether to use logarithmic scale for y-axis
        save_path: Path to save the figure (optional)
    """
    if "train_loss_components" not in logs or not logs["train_loss_components"]:
        print("No loss components found in logs")
        return
    
    # Get all unique components
    components = set()
    for log in logs["train_loss_components"]:
        components.update(log.keys())
    components = sorted(list(components))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each component
    epochs = np.arange(1, len(logs["train_loss_components"]) + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
    
    for i, component in enumerate(components):
        component_values = []
        for log in logs["train_loss_components"]:
            component_values.append(log.get(component, np.nan))
        
        # Convert to numpy array and handle NaN values
        component_values = np.array(component_values)
        mask = ~np.isnan(component_values)
        
        if np.any(mask):
            ax.plot(epochs[mask], component_values[mask], label=component, color=colors[i])
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_weight_evolution(
    weight_history: List[Dict[str, Dict[str, float]]],
    figsize: Tuple[int, int] = (12, 8),
    log_scale: bool = False,
    save_path: Optional[str] = None
):
    """
    Plot the evolution of self-adaptive weights during training.
    
    Args:
        weight_history: List of weight statistics dictionaries
        figsize: Figure size
        log_scale: Whether to use logarithmic scale for y-axis
        save_path: Path to save the figure (optional)
    """
    if not weight_history:
        print("No weight history provided")
        return
    
    # Get all unique components
    components = set()
    for stats in weight_history:
        components.update(stats.keys())
    components = sorted(list(components))
    
    # Get all metrics (min, max, mean) for weights and masks
    metrics = set()
    for stats in weight_history:
        for component in stats:
            metrics.update(stats[component].keys())
    
    # Group metrics by type (weight or mask)
    weight_metrics = [m for m in metrics if "weight" in m]
    mask_metrics = [m for m in metrics if "mask" in m]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot epochs
    epochs = np.arange(1, len(weight_history) + 1)
    
    # Plot weight metrics
    ax = axes[0]
    for component in components:
        for metric in weight_metrics:
            values = []
            for stats in weight_history:
                if component in stats and metric in stats[component]:
                    values.append(stats[component][metric])
                else:
                    values.append(np.nan)
            
            # Convert to numpy array and handle NaN values
            values = np.array(values)
            mask = ~np.isnan(values)
            
            if np.any(mask):
                label = f"{component} - {metric}"
                ax.plot(epochs[mask], values[mask], label=label)
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight Value")
    ax.set_title("Self-Adaptive Weight Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot mask metrics
    ax = axes[1]
    for component in components:
        for metric in mask_metrics:
            values = []
            for stats in weight_history:
                if component in stats and metric in stats[component]:
                    values.append(stats[component][metric])
                else:
                    values.append(np.nan)
            
            # Convert to numpy array and handle NaN values
            values = np.array(values)
            mask = ~np.isnan(values)
            
            if np.any(mask):
                label = f"{component} - {metric}"
                ax.plot(epochs[mask], values[mask], label=label)
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mask Value")
    ax.set_title("Self-Adaptive Mask Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_1d_solution(
    model: torch.nn.Module,
    x_domain: Tuple[float, float],
    t: Optional[float] = None,
    n_points: int = 1000,
    exact_solution: Optional[Callable] = None,
    title: str = "Solution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot the solution of a 1D problem at a specific time.
    
    Args:
        model: Trained PINN model
        x_domain: Domain bounds (x_min, x_max)
        t: Time at which to plot the solution (if None, assumes steady-state)
        n_points: Number of points to evaluate
        exact_solution: Exact solution function (optional)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Generate evaluation points
    x_min, x_max = x_domain
    x = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(1)
    
    with torch.no_grad():
        if t is not None:
            # Time-dependent problem
            t_tensor = torch.ones_like(x) * t
            x_t = torch.cat([x, t_tensor], dim=1)
            u_pred = model(x_t).cpu().numpy()
        else:
            # Steady-state problem
            u_pred = model(x).cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot predicted solution
    x_np = x.cpu().numpy()
    ax.plot(x_np, u_pred, label="PINN Solution", linewidth=2)
    
    # Plot exact solution if provided
    if exact_solution is not None:
        if t is not None:
            u_exact = exact_solution(x_np, t)
        else:
            u_exact = exact_solution(x_np)
        
        ax.plot(x_np, u_exact, label="Exact Solution", linestyle="--", linewidth=2)
        
        # Compute and display L2 error
        l2_error = np.sqrt(np.mean((u_pred - u_exact)**2)) / np.sqrt(np.mean(u_exact**2))
        ax.text(0.05, 0.95, f"L2 Error: {l2_error:.6f}", transform=ax.transAxes,
                fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.1))
    
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    
    if t is not None:
        ax.set_title(f"{title} at t = {t:.4f}")
    else:
        ax.set_title(title)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_2d_solution(
    model: torch.nn.Module,
    x_domain: Tuple[float, float],
    y_domain: Tuple[float, float],
    t: Optional[float] = None,
    resolution: int = 100,
    exact_solution: Optional[Callable] = None,
    plot_type: str = "surface",
    cmap: str = "viridis",
    title: str = "Solution",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot the solution of a 2D problem at a specific time.
    
    Args:
        model: Trained PINN model
        x_domain: x-domain bounds (x_min, x_max)
        y_domain: y-domain bounds (y_min, y_max)
        t: Time at which to plot the solution (if None, assumes steady-state)
        resolution: Number of points in each dimension
        exact_solution: Exact solution function (optional)
        plot_type: Type of plot ('surface', 'contour', or 'both')
        cmap: Colormap to use
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Generate mesh grid
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    x = torch.linspace(x_min, x_max, resolution, device=device)
    y = torch.linspace(y_min, y_max, resolution, device=device)
    
    X, Y = torch.meshgrid(x, y, indexing="ij")
    X_flat = X.flatten().unsqueeze(1)
    Y_flat = Y.flatten().unsqueeze(1)
    
    with torch.no_grad():
        if t is not None:
            # Time-dependent problem
            t_tensor = torch.ones_like(X_flat) * t
            xy_t = torch.cat([X_flat, Y_flat, t_tensor], dim=1)
            u_pred = model(xy_t).reshape(resolution, resolution).cpu().numpy()
        else:
            # Steady-state problem
            xy = torch.cat([X_flat, Y_flat], dim=1)
            u_pred = model(xy).reshape(resolution, resolution).cpu().numpy()
    
    # Compute exact solution if provided
    if exact_solution is not None:
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        
        if t is not None:
            u_exact = exact_solution(X_np, Y_np, t)
        else:
            u_exact = exact_solution(X_np, Y_np)
        
        # Compute L2 error
        l2_error = np.sqrt(np.mean((u_pred - u_exact)**2)) / np.sqrt(np.mean(u_exact**2))
    
    # Create figure
    if plot_type == "both":
        fig = plt.figure(figsize=(figsize[0] * 2, figsize[1]))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122)
    elif plot_type == "surface":
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111, projection="3d")
        ax2 = None
    else:  # contour
        fig = plt.figure(figsize=figsize)
        ax1 = None
        ax2 = fig.add_subplot(111)
    
    # Plot surface
    if ax1 is not None:
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        surf = ax1.plot_surface(X_np, Y_np, u_pred, cmap=cmap, edgecolor="none", alpha=0.8)
        
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("u(x,y)")
        
        if t is not None:
            ax1.set_title(f"{title} at t = {t:.4f}")
        else:
            ax1.set_title(title)
        
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Plot contour
    if ax2 is not None:
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        contour = ax2.contourf(X_np, Y_np, u_pred, cmap=cmap, levels=50)
        
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        
        if t is not None:
            ax2.set_title(f"{title} at t = {t:.4f}")
        else:
            ax2.set_title(title)
        
        fig.colorbar(contour, ax=ax2)
    
    # Add error information if exact solution provided
    if exact_solution is not None:
        if ax1 is not None:
            ax1.text2D(0.05, 0.95, f"L2 Error: {l2_error:.6f}", transform=ax1.transAxes,
                    fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.1))
        elif ax2 is not None:
            ax2.text(0.05, 0.95, f"L2 Error: {l2_error:.6f}", transform=ax2.transAxes,
                    fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_solution_error(
    model: torch.nn.Module,
    x_domain: Tuple[float, float],
    t_domain: Optional[Tuple[float, float]] = None,
    exact_solution: Callable,

    resolution: int = 100,
    cmap: str = "coolwarm",
    log_scale: bool = False,
    title: str = "Solution Error",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot the absolute error between the PINN solution and the exact solution.
    
    Args:
        model: Trained PINN model
        x_domain: x-domain bounds (x_min, x_max)
        t_domain: t-domain bounds (t_min, t_max) if time-dependent, None otherwise
        exact_solution: Exact solution function
        resolution: Number of points in each dimension
        cmap: Colormap to use
        log_scale: Whether to use logarithmic scale for error
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Generate mesh grid
    x_min, x_max = x_domain
    x = torch.linspace(x_min, x_max, resolution, device=device)
    
    if t_domain is not None:
        # Time-dependent problem
        t_min, t_max = t_domain
        t = torch.linspace(t_min, t_max, resolution, device=device)
        
        X, T = torch.meshgrid(x, t, indexing="ij")
        X_flat = X.flatten().unsqueeze(1)
        T_flat = T.flatten().unsqueeze(1)
        
        with torch.no_grad():
            x_t = torch.cat([X_flat, T_flat], dim=1)
            u_pred = model(x_t).reshape(resolution, resolution).cpu().numpy()
        
        # Compute exact solution
        X_np = X.cpu().numpy()
        T_np = T.cpu().numpy()
        u_exact = exact_solution(X_np, T_np)
        
        # Compute absolute error
        error = np.abs(u_pred - u_exact)
        
        # Plot error
        fig, ax = plt.subplots(figsize=figsize)
        
        if log_scale and np.min(error) > 0:
            contour = ax.contourf(X_np, T_np, error, cmap=cmap, levels=50, norm=plt.matplotlib.colors.LogNorm())
        else:
            contour = ax.contourf(X_np, T_np, error, cmap=cmap, levels=50)
        
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        
        fig.colorbar(contour, ax=ax, label="Absolute Error")
    else:
        # Steady-state problem
        with torch.no_grad():
            x_tensor = x.unsqueeze(1)
            u_pred = model(x_tensor).cpu().numpy()
        
        # Compute exact solution
        x_np = x.cpu().numpy()
        u_exact = exact_solution(x_np)
        
        # Compute absolute error
        error = np.abs(u_pred - u_exact.reshape(-1, 1))
        
        # Plot error
        fig, ax = plt.subplots(figsize=figsize)
        
        if log_scale and np.min(error) > 0:
            ax.semilogy(x_np, error, linewidth=2)
        else:
            ax.plot(x_np, error, linewidth=2)
        
        ax.set_xlabel("x")
        ax.set_ylabel("Absolute Error")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    # Add L2 error
    l2_error = np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(u_exact**2))
    ax.text(0.05, 0.95, f"L2 Error: {l2_error:.6f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", alpha=0.1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
    
    return l2_error


def plot_adaptive_weights(
    x: torch.Tensor,
    t: Optional[torch.Tensor],
    weights: torch.Tensor,
    weight_type: str = "residual",
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "viridis",
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot the self-adaptive weights distribution.
    
    Args:
        x: Spatial coordinates
        t: Temporal coordinates (optional)
        weights: Adaptive weights
        weight_type: Type of weights ('residual', 'boundary', or 'initial')
        figsize: Figure size
        cmap: Colormap to use
        title: Plot title (optional)
        save_path: Path to save the figure (optional)
    """
    # Convert to numpy arrays
    x_np = x.detach().cpu().numpy()
    weights_np = weights.detach().cpu().numpy()
    
    if t is not None:
        # Time-dependent problem
        t_np = t.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot with color representing weight magnitude
        scatter = ax.scatter(x_np, t_np, c=weights_np, cmap=cmap, alpha=0.7, s=30)
        
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        
        if title is None:
            title = f"Self-Adaptive {weight_type.capitalize()} Weights"
        ax.set_title(title)
        
        fig.colorbar(scatter, ax=ax, label="Weight Magnitude")
    else:
        # 1D spatial problem
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot with y-value and color representing weight magnitude
        scatter = ax.scatter(x_np, weights_np, c=weights_np, cmap=cmap, alpha=0.7)
        
        # Also draw a line to better visualize the trend
        ax.plot(x_np, weights_np, alpha=0.5, color='gray')
        
        ax.set_xlabel("x")
        ax.set_ylabel("Weight Magnitude")
        
        if title is None:
            title = f"Self-Adaptive {weight_type.capitalize()} Weights"
        ax.set_title(title)
        
        fig.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def create_solution_animation(
    model: torch.nn.Module,
    x_domain: Tuple[float, float],
    t_domain: Tuple[float, float],
    n_frames: int = 50,
    n_points: int = 200,
    exact_solution: Optional[Callable] = None,
    title: str = "Solution Evolution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    fps: int = 10
):
    """
    Create an animation of the solution evolution over time.
    
    Args:
        model: Trained PINN model
        x_domain: Domain bounds (x_min, x_max)
        t_domain: Time domain bounds (t_min, t_max)
        n_frames: Number of frames in the animation
        n_points: Number of spatial points
        exact_solution: Exact solution function (optional)
        title: Animation title
        figsize: Figure size
        save_path: Path to save the animation (optional)
        fps: Frames per second
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Generate spatial points
    x_min, x_max = x_domain
    x = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(1)
    
    # Generate time points
    t_min, t_max = t_domain
    t_values = torch.linspace(t_min, t_max, n_frames, device=device)
    
    # Compute solutions for each time point
    solutions = []
    exact_solutions = []
    
    with torch.no_grad():
        for t_val in t_values:
            # Create input tensor
            t_tensor = torch.ones_like(x) * t_val
            x_t = torch.cat([x, t_tensor], dim=1)
            
            # Compute model prediction
            u_pred = model(x_t).cpu().numpy()
            solutions.append(u_pred)
            
            # Compute exact solution if provided
            if exact_solution is not None:
                x_np = x.cpu().numpy()
                t_np = t_val.cpu().item()
                u_exact = exact_solution(x_np, t_np)
                exact_solutions.append(u_exact)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    
    # Find global min and max for consistent y-axis
    all_solutions = np.concatenate(solutions)
    y_min = np.min(all_solutions)
    y_max = np.max(all_solutions)
    
    if exact_solution is not None:
        all_exact = np.concatenate(exact_solutions)
        y_min = min(y_min, np.min(all_exact))
        y_max = max(y_max, np.max(all_exact))
    
    # Add some margin
    margin = 0.1 * (y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    # Create line objects
    x_np = x.cpu().numpy()
    line_pred, = ax.plot([], [], 'b-', label='PINN Solution', linewidth=2)
    
    if exact_solution is not None:
        line_exact, = ax.plot([], [], 'r--', label='Exact Solution', linewidth=2)
        error_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', alpha=0.1))
    
    # Add time text
    time_text = ax.text(0.05, 0.05, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', alpha=0.1))
    
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Initialization function
    def init():
        line_pred.set_data([], [])
        if exact_solution is not None:
            line_exact.set_data([], [])
            return line_pred, line_exact, time_text, error_text
        return line_pred, time_text
    
    # Animation function
    def animate(i):
        line_pred.set_data(x_np, solutions[i])
        time_text.set_text(f't = {t_values[i].item():.4f}')
        
        if exact_solution is not None:
            line_exact.set_data(x_np, exact_solutions[i])
            
            # Compute L2 error
            error = np.abs(solutions[i] - exact_solutions[i])
            l2_error = np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(exact_solutions[i]**2))
            error_text.set_text(f'L2 Error: {l2_error:.6f}')
            
            return line_pred, line_exact, time_text, error_text
        
        return line_pred, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   init_func=init, blit=True)
    
    plt.tight_layout()
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
    
    plt.close()
    
    return anim


def plot_pde_residual(
    model: torch.nn.Module,
    pde_residual_fn: Callable,
    x_domain: Tuple[float, float],
    t_domain: Optional[Tuple[float, float]] = None,
    resolution: int = 100,
    log_scale: bool = True,
    cmap: str = 'viridis',
    title: str = 'PDE Residual',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot the PDE residual across the domain.
    
    Args:
        model: Trained PINN model
        pde_residual_fn: Function to compute the PDE residual
        x_domain: x-domain bounds (x_min, x_max)
        t_domain: t-domain bounds (t_min, t_max) if time-dependent, None otherwise
        resolution: Number of points in each dimension
        log_scale: Whether to use logarithmic scale for residual
        cmap: Colormap to use
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (optional)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Generate mesh grid
    x_min, x_max = x_domain
    x = torch.linspace(x_min, x_max, resolution, device=device)
    
    if t_domain is not None:
        # Time-dependent problem
        t_min, t_max = t_domain
        t = torch.linspace(t_min, t_max, resolution, device=device)
        
        X, T = torch.meshgrid(x, t, indexing='ij')
        X_flat = X.flatten().unsqueeze(1)
        T_flat = T.flatten().unsqueeze(1)
        
        with torch.no_grad():
            # Prepare inputs
            x_t = torch.cat([X_flat, T_flat], dim=1)
            
            # Make sure inputs require gradient
            x_t.requires_grad_(True)
            
            # Forward pass
            u = model(x_t)
            
            # Compute derivatives
            u_x = torch.autograd.grad(
                outputs=u,
                inputs=x_t,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Extract spatial and temporal derivatives
            u_t = u_x[:, 1:2]
            u_x = u_x[:, 0:1]
            
            # Compute second derivatives
            u_xx = torch.autograd.grad(
                outputs=u_x,
                inputs=x_t,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True,
                retain_graph=True
            )[0][:, 0:1]
            
            # Prepare outputs for residual computation
            outputs = {
                'u': u,
                'u_x': u_x,
                'u_t': u_t,
                'u_xx_0': u_xx.squeeze(1)
            }
            
            # Compute residual
            inputs_dict = {'x': X_flat, 't': T_flat}
            residual = pde_residual_fn(model, inputs_dict, outputs)
            
            # Reshape residual
            residual = residual.reshape(resolution, resolution).cpu().numpy()
        
        # Plot residual
        fig, ax = plt.subplots(figsize=figsize)
        
        X_np = X.cpu().numpy()
        T_np = T.cpu().numpy()
        
        if log_scale:
            # Use absolute value for log scale
            abs_residual = np.abs(residual)
            vmin = np.min(abs_residual[abs_residual > 0])  # Exclude zeros
            norm = plt.matplotlib.colors.LogNorm(vmin=vmin)
            contour = ax.contourf(X_np, T_np, abs_residual, cmap=cmap, levels=50, norm=norm)
            cb_label = 'Absolute Residual (log scale)'
        else:
            contour = ax.contourf(X_np, T_np, residual, cmap=cmap, levels=50)
            cb_label = 'Residual'
        
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(title)
        
        fig.colorbar(contour, ax=ax, label=cb_label)
    else:
        # Steady-state problem
        x_tensor = x.unsqueeze(1)
        x_tensor.requires_grad_(True)
        
        with torch.no_grad():
            # Forward pass
            u = model(x_tensor)
            
            # Compute derivatives
            u_x = torch.autograd.grad(
                outputs=u,
                inputs=x_tensor,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Compute second derivatives
            u_xx = torch.autograd.grad(
                outputs=u_x,
                inputs=x_tensor,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Prepare outputs for residual computation
            outputs = {
                'u': u,
                'u_x': u_x,
                'u_xx_0': u_xx.squeeze(1)
            }
            
            # Compute residual
            inputs_dict = {'x': x_tensor}
            residual = pde_residual_fn(model, inputs_dict, outputs)
            
            # Convert to numpy array
            residual = residual.cpu().numpy()
        
        # Plot residual
        fig, ax = plt.subplots(figsize=figsize)
        
        x_np = x.cpu().numpy()
        
        if log_scale:
            # Use absolute value for log scale
            abs_residual = np.abs(residual)
            ax.semilogy(x_np, abs_residual, linewidth=2)
            ax.set_ylabel('Absolute Residual (log scale)')
        else:
            ax.plot(x_np, residual, linewidth=2)
            ax.set_ylabel('Residual')
        
        ax.set_xlabel('x')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()