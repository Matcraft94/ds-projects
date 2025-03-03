# PT-PINN: Pretrained Physics-Informed Neural Networks for Solving PDEs

This document summarizes the PT-PINN methodology, implementation, experiments, and results from the project.

## 1. Methodology

### 1.1 Standard PINN (Physics-Informed Neural Networks)

- PINNs are neural networks trained to solve partial differential equations by incorporating physical laws as soft constraints in the loss function
- The standard PINN approach uses a combined loss function with components for:
  - Initial condition loss
  - Boundary condition loss
  - PDE residual loss (enforces the differential equation in the domain)
- Loss function (from `pinn_base.py`):
  ```
  total_loss = w_initial * loss_initial + w_boundary * loss_boundary + w_residual * loss_residual
  ```
- Training typically uses a combination of Adam and L-BFGS optimizers

### 1.2 PT-PINN (Pretrained Physics-Informed Neural Networks)

- PT-PINN extends the standard PINN approach by implementing a progressive training strategy
- The method divides the time domain into intervals `[0, T1], [0, T2], ..., [0, T]` and trains sequentially
- Key concepts:
  - For the first interval `[0, T1]`, train a standard PINN
  - For subsequent intervals, use the knowledge from the previous interval by generating supervised data
  - Progressive training helps address the "multi-scale" nature of PDEs by gradually building solutions
  - Adds a supervised loss component to the standard PINN loss function for intervals after the first
  
- Implementation in `pt_pinn.py` follows:
  1. Pretraining phase with multiple intervals (Algorithm 2 in the paper)
  2. Final training on the full domain using knowledge from pretraining
  3. Resampling strategy for collocation points (Algorithm 1 in the paper)

## 2. Experiments

The project implements and tests both standard PINN and PT-PINN approaches on several PDEs:

### 2.1 Basic Diffusion Equation (01_basic_pinn.ipynb)
- Simple heat equation: ∂u/∂t - ∂²u/∂x² = 0
- Used as a baseline demonstration of standard PINN functionality

### 2.2 Reaction System (03_reaction_system.ipynb)
- Implements Fisher's equation: ∂u/∂t - ρu(1-u) = 0
- Tests with different values of ρ parameter (2, 5, 10, 15, 20)
- Compares standard PINN and PT-PINN performance

### 2.3 Heat Equation (04_heat_equation.ipynb)
- Enhanced implementation with ResNet architecture
- Uses input scaling to handle multi-scale nature of the problem

### 2.4 Allen-Cahn Equation (05_allen_cahn.ipynb)
- Non-linear PDE: ∂u/∂t - D∂²u/∂x² - ku(1-u²) = 0
- Parameters: D = 0.0001 (diffusion coefficient), k = 5.0 (reaction coefficient)
- Tests standard PINN and PT-PINN with different numbers of training intervals (1-interval and 2-interval)

## 3. Results

### 3.1 Reaction System Results

- Error comparison between standard PINN and PT-PINN shows that:
  - For low values of ρ (2, 5), PT-PINN significantly outperforms standard PINN
  - For ρ = 5, standard PINN shows dramatic error increase while PT-PINN maintains low error
  - For higher ρ values (10, 15, 20), both methods struggle with similar performance
  - The mean absolute error is shown on a log scale, with PT-PINN achieving errors as low as 10⁻⁴ for ρ = 2

### 3.2 Allen-Cahn Equation Results

- Metrics from `metricas.txt`:

| Model | Error L2 Relative | Error L1 Absolute | Error Linf Absolute |
|-------|------------------|-------------------|---------------------|
| PINN Estándar | 0.405721 | 0.043306 | 0.252884 |
| PT-PINN (1-int) | 0.651485 | 0.079208 | 0.439600 |
| PT-PINN (2-int) | 2.000576 | 0.219676 | 1.173166 |

- For the Allen-Cahn equation, standard PINN outperformed PT-PINN in terms of error metrics
- This contradicts the results from the reaction system, suggesting that the benefits of PT-PINN may be problem-dependent

### 3.3 Training Dynamics

- Training history plots (standard_history.png and pt_history.png) show:
  - Standard PINN achieves more significant residual loss reduction
  - PT-PINN shows more stable training behavior with fewer spikes in the loss
  - Standard PINN residual loss drops to ~10¹ while PT-PINN stays around ~10³

## 4. Implementation Details

### 4.1 Architecture

- Both PINN and PT-PINN use similar network architectures:
  - Default: Fully connected network with tanh activation
  - Enhanced: ResNet blocks with scaled residual connections (in heat equation notebook)
  
- Configuration options include:
  - Number of hidden layers
  - Neurons per layer
  - Loss weighting factors
  - Optimizer settings

### 4.2 Optimization Strategy

- Two-phase optimization:
  1. Adam optimizer with learning rate scheduling and resampling (Algorithm 1)
  2. L-BFGS for fine-tuning (with built-in line search)

### 4.3 Resampling Strategy

- Periodically replaces a fraction of collocation points during training
- Implementation in `trainer.py`:
  - Keeps (1-η) fraction of existing points
  - Generates η fraction of new points uniformly from the domain
  - Resampling stops after a termination step F

## 5. Conclusions

- PT-PINN shows significant benefits for some PDEs, particularly at lower parameter values in reaction systems
- For more complex equations like Allen-Cahn, standard PINN performed better in the tested configurations
- The progressive training strategy helps with stability but doesn't always improve accuracy
- The benefit of PT-PINN seems to be problem-dependent and may require tuning for specific equations

## 6. Future Work

- Further testing on different types of PDEs
- Optimization of pretraining intervals selection
- Improved resampling strategies based on error distribution
- Hybrid approaches combining the stability of PT-PINN with the final accuracy of standard PINN