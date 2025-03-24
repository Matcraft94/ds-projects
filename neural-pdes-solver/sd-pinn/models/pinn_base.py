"""
models/pinn_base.py
Path: models/pinn_base.py
Date: 23-Mar-2025
Author: Lucy
Description: Base implementation of Physics-Informed Neural Networks (PINNs)
             using PyTorch. Provides core architecture and forward methods.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Union


class PINN(nn.Module):
    """
    Base class for Physics-Informed Neural Networks (PINNs).
    
    Implements a fully connected neural network with customizable architecture
    and activation functions. Includes methods for computing derivatives using
    automatic differentiation for solving PDEs.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str = "tanh",
        initializer: str = "xavier_normal",
        dropout_rate: float = 0.0
    ):
        """
        Initialize the PINN model.
        
        Args:
            input_dim: Number of input dimensions (e.g., 2 for x and t in 1D+time problems)
            output_dim: Number of output dimensions
            hidden_layers: List of integers representing the number of neurons in each hidden layer
            activation: Activation function to use ('tanh', 'relu', 'sigmoid', or 'sin')
            initializer: Weight initialization method ('xavier_normal', 'xavier_uniform', 'he_normal')
            dropout_rate: Dropout probability (0.0 means no dropout)
        """
        super(PINN, self).__init__()
        
        # Save configuration
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Set activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "sin":
            self.activation = lambda x: torch.sin(x)
        else:
            raise ValueError(f"Activation function {activation} not supported")
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers)-1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights(initializer)
        
    def _initialize_weights(self, initializer: str):
        """Initialize the weights of the network according to the specified initializer."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if initializer == "xavier_normal":
                    nn.init.xavier_normal_(layer.weight)
                elif initializer == "xavier_uniform":
                    nn.init.xavier_uniform_(layer.weight)
                elif initializer == "he_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                else:
                    raise ValueError(f"Initializer {initializer} not supported")
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor with shape (batch_size, input_dim)
            
        Returns:
            Network output tensor with shape (batch_size, output_dim)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            if self.dropout_rate > 0:
                x = self.dropout(x)
        
        # No activation on the output layer
        x = self.layers[-1](x)
        
        return x
    
    def compute_gradients(self, outputs: torch.Tensor, 
                          inputs: torch.Tensor, 
                          order: int = 1) -> List[torch.Tensor]:
        """
        Compute gradients of outputs with respect to inputs using automatic differentiation.
        
        Args:
            outputs: Output tensor from the network
            inputs: Input tensor that requires gradient
            order: Order of the derivative (1 for first derivatives, 2 for second derivatives)
            
        Returns:
            List of gradients of each output with respect to each input
        """
        gradients = []
        
        for i in range(outputs.shape[1]):
            y = outputs[:, i].sum()
            
            # First-order derivatives
            grad_y = torch.autograd.grad(
                y, inputs, create_graph=True, retain_graph=True
            )[0]
            
            # For first-order, return the gradients directly
            if order == 1:
                gradients.append(grad_y)
            
            # For second-order, compute gradients of the first derivatives
            elif order == 2:
                for j in range(inputs.shape[1]):
                    gradij = torch.autograd.grad(
                        grad_y[:, j].sum(), inputs, create_graph=True, retain_graph=True
                    )[0]
                    gradients.append(gradij)
            
            else:
                raise ValueError(f"Order {order} not supported. Use 1 or 2.")
        
        return gradients
    
    def get_training_stats(self) -> Dict[str, int]:
        """Get statistics about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers
        }