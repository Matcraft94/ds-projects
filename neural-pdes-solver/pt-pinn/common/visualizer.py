# Created by por: Lucy
# Date : 2025-02-22
# common/visualizer.py

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple, Union
import torch

class PINNVisualizer:
    """
    Visualization tools for PINN and PT-PINN results
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize
        # Set style - using built-in style instead of seaborn
        plt.style.use('default')
        # Set color palette manually
        self.colors = ['#FF7F50', '#4682B4', '#98FB98', '#DDA0DD', '#F0E68C']
        
        # Configure default plot settings
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.color': '#CCCCCC',
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
        
    def plot_training_history(self, 
                            history: Dict[str, List[float]],
                            save_path: Optional[str] = None) -> None:
        """
        Plot training losses and error metrics evolution
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot losses
        ax1.semilogy(history['total_loss'], label='Total Loss')
        ax1.semilogy(history['initial_loss'], label='Initial Loss')
        ax1.semilogy(history['boundary_loss'], label='Boundary Loss')
        ax1.semilogy(history['residual_loss'], label='Residual Loss')
        if 'supervised_loss' in history:
            ax1.semilogy(history['supervised_loss'], label='Supervised Loss')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss (log scale)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot error metrics
        ax2.semilogy(history['l2_relative'], label='L2 Relative')
        ax2.semilogy(history['l1_absolute'], label='L1 Absolute')
        ax2.semilogy(history['linf_absolute'], label='L∞ Absolute')
        
        ax2.set_xlabel('Evaluation Step')
        ax2.set_ylabel('Error (log scale)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_1d_solution(self,
                        t: torch.Tensor,
                        x: torch.Tensor,
                        u_pred: torch.Tensor,
                        u_true: Optional[torch.Tensor] = None,
                        title: str = "",
                        save_path: Optional[str] = None) -> None:
        """
        Plot 1D solution evolution over time (similar to Fig 4.3)
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        T, X = torch.meshgrid(t, x)
        
        # Plot predicted solution
        surf = ax.plot_surface(T.cpu().numpy(), 
                             X.cpu().numpy(), 
                             u_pred.cpu().numpy(),
                             cmap='viridis',
                             alpha=0.8)
        
        # Plot true solution if provided
        if u_true is not None:
            ax.plot_surface(T.cpu().numpy(),
                          X.cpu().numpy(),
                          u_true.cpu().numpy(),
                          cmap='winter',
                          alpha=0.3)
        
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('u(x,t)')
        ax.set_title(title)
        
        fig.colorbar(surf)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_2d_solution(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        u: torch.Tensor,
                        t: float,
                        title: str = "",
                        save_path: Optional[str] = None) -> None:
        """
        Plot 2D solution at a specific time
        """
        fig = plt.figure(figsize=self.figsize)
        
        plt.pcolormesh(x.cpu().numpy(),
                      y.cpu().numpy(),
                      u.cpu().numpy(),
                      shading='auto',
                      cmap='viridis')
        
        plt.colorbar(label='u(x,y,t)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{title} at t={t:.2f}')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_error_heatmap(self,
                          error: torch.Tensor,
                          x: torch.Tensor,
                          t: torch.Tensor,
                          title: str = "",
                          save_path: Optional[str] = None) -> None:
        """
        Plot point-wise absolute error heatmap (similar to Fig 4.4)
        """
        plt.figure(figsize=self.figsize)
        
        plt.pcolormesh(t.cpu().numpy(),
                      x.cpu().numpy(),
                      error.cpu().numpy(),
                      shading='auto',
                      cmap='hot')
        
        plt.colorbar(label='|u_pred - u_true|')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_comparison(self,
                       x: torch.Tensor,
                       t: Union[float, torch.Tensor],
                       u_true: torch.Tensor,
                       u_pinn: torch.Tensor,
                       u_ptpinn: torch.Tensor,
                       title: str = "",
                       save_path: Optional[str] = None) -> None:
        """
        Compare solutions from PINN and PT-PINN (similar to Fig 4.8)
        """
        plt.figure(figsize=self.figsize)
        
        plt.plot(x.cpu().numpy(), u_true.cpu().numpy(), 'k-', label='Reference solution')
        plt.plot(x.cpu().numpy(), u_pinn.cpu().numpy(), '--', label='PINN')
        plt.plot(x.cpu().numpy(), u_ptpinn.cpu().numpy(), '-.', label='PT-PINN')
        
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title(f'{title} at t={t:.2f}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def create_animation(self,
                        x: torch.Tensor,
                        t: torch.Tensor,
                        u: torch.Tensor,
                        title: str = "",
                        save_path: Optional[str] = None) -> None:
        """
        Create animation of solution evolution over time
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        line, = ax.plot([], [])
        ax.set_xlim(x.min().item(), x.max().item())
        ax.set_ylim(u.min().item(), u.max().item())
        
        def init():
            line.set_data([], [])
            return line,
            
        def animate(i):
            line.set_data(x.cpu().numpy(), u[:, i].cpu().numpy())
            ax.set_title(f'{title} at t={t[i]:.2f}')
            return line,
            
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=len(t), interval=100,
                                     blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow')
        plt.show()
        
    def plot_multiple_times(self,
                          x: torch.Tensor,
                          t: torch.Tensor,
                          u: torch.Tensor,
                          times: List[float],
                          title: str = "",
                          save_path: Optional[str] = None) -> None:
        """
        Plot solution at multiple time points
        """
        plt.figure(figsize=self.figsize)
        
        for time in times:
            idx = (torch.abs(t - time)).argmin()
            plt.plot(x.cpu().numpy(), 
                    u[:, idx].cpu().numpy(),
                    label=f't={time:.2f}')
            
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_residual_points(self,
                           x_residual: torch.Tensor,
                           domain_bounds: torch.Tensor,
                           title: str = "",
                           save_path: Optional[str] = None) -> None:
        """
        Visualize distribution of residual points
        """
        plt.figure(figsize=self.figsize)
        
        if x_residual.shape[1] == 2:  # 1D PDE (x,t)
            plt.scatter(x_residual[:, 0].cpu().numpy(),
                       x_residual[:, 1].cpu().numpy(),
                       alpha=0.5)
            plt.xlabel('x')
            plt.ylabel('t')
        else:  # 2D PDE (x,y,t)
            ax = plt.gca(projection='3d')
            ax.scatter(x_residual[:, 0].cpu().numpy(),
                      x_residual[:, 1].cpu().numpy(),
                      x_residual[:, 2].cpu().numpy(),
                      alpha=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('t')
            
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_3d_slice(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     z: torch.Tensor,
                     u: torch.Tensor,
                     slice_dim: int,
                     slice_value: float,
                     title: str = "",
                     save_path: Optional[str] = None) -> None:
        """
        Plot 2D slice of 3D solution
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        
        # Find nearest slice index
        if slice_dim == 0:
            idx = (torch.abs(x - slice_value)).argmin()
            X, Y = torch.meshgrid(y, z)
            Z = u[idx, :, :]
            xlabel, ylabel = 'y', 'z'
        elif slice_dim == 1:
            idx = (torch.abs(y - slice_value)).argmin()
            X, Y = torch.meshgrid(x, z)
            Z = u[:, idx, :]
            xlabel, ylabel = 'x', 'z'
        else:
            idx = (torch.abs(z - slice_value)).argmin()
            X, Y = torch.meshgrid(x, y)
            Z = u[:, :, idx]
            xlabel, ylabel = 'x', 'y'
            
        im = ax.pcolormesh(X.cpu().numpy(),
                          Y.cpu().numpy(),
                          Z.cpu().numpy(),
                          shading='auto',
                          cmap='viridis')
        
        plt.colorbar(im)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} at {["x","y","z"][slice_dim]}={slice_value:.2f}')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()