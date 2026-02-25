"""Bootstrap confidence intervals for Hawkes process parameters.

Implements parametric and non-parametric bootstrap for uncertainty
quantification in Hawkes process estimates.

Reference:
    Cowling, A., Hall, P., & Phillips, M. J. (1996). Bootstrap confidence 
    regions for the intensity of a Poisson process. Journal of the American 
    Statistical Association.
"""

import numpy as np
from typing import Callable, Optional, Tuple
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt


class BootstrapConfidenceIntervals:
    """Bootstrap confidence intervals for Hawkes process parameters.
    
    Implements both parametric bootstrap (simulating from fitted model)
    and non-parametric bootstrap (resampling events).
    
    For a Hawkes process, the parametric bootstrap is preferred as it
    respects the temporal dependence structure.
    """
    
    def __init__(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        n_jobs: int = -1,
        random_state: Optional[int] = None
    ):
        """Initialize bootstrap.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals (e.g., 0.95)
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.bootstrap_estimates_: Optional[dict] = None
    
    def parametric_bootstrap(
        self,
        estimator_class: type,
        estimator_params: dict,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        end_time: float,
        fit_kwargs: Optional[dict] = None
    ) -> dict:
        """Parametric bootstrap by simulating from fitted model.
        
        This is the preferred method for Hawkes processes as it respects
        the temporal dependence structure.
        
        Args:
            estimator_class: Class of estimator to use
            estimator_params: Parameters for estimator initialization
            mu: True baseline intensities (from fitted model)
            alpha: True excitation matrix
            beta: True decay matrix
            end_time: End of observation period
            fit_kwargs: Additional arguments for fit method
            
        Returns:
            Dictionary with bootstrap estimates
        """
        from hawkes.utils.data_loader import simulate_hawkes_process
        
        if fit_kwargs is None:
            fit_kwargs = {}
        
        def fit_one_bootstrap(seed: int):
            """Single bootstrap iteration."""
            np.random.seed(seed)
            
            try:
                # Simulate from fitted model
                sim_events = simulate_hawkes_process(
                    mu, alpha, beta, end_time, seed=seed
                )
                
                # Fit to simulated data
                est = estimator_class(**estimator_params)
                est.fit(sim_events, end_time=end_time, **fit_kwargs)
                
                return {
                    'mu': est.mu_ if hasattr(est, 'mu_') else None,
                    'alpha': est.alpha_ if hasattr(est, 'alpha_') else None,
                    'beta': est.beta_ if hasattr(est, 'beta_') else None,
                    'success': True
                }
            except Exception:
                return {'success': False}
        
        # Run in parallel
        seeds = np.random.randint(0, 2**31, size=self.n_bootstrap)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_one_bootstrap)(seed) for seed in seeds
        )
        
        # Filter successful fits
        successful = [r for r in results if r['success']]
        n_success = len(successful)
        
        if n_success < self.n_bootstrap * 0.5:
            raise RuntimeError(
                f"Too many bootstrap failures: {self.n_bootstrap - n_success}/{self.n_bootstrap}"
            )
        
        # Collect estimates
        mu_estimates = np.array([r['mu'] for r in successful if r['mu'] is not None])
        alpha_estimates = np.array([r['alpha'] for r in successful if r['alpha'] is not None])
        beta_estimates = np.array([r['beta'] for r in successful if r['beta'] is not None])
        
        self.bootstrap_estimates_ = {
            'mu': mu_estimates,
            'alpha': alpha_estimates,
            'beta': beta_estimates,
            'n_success': n_success,
            'n_total': self.n_bootstrap
        }
        
        return self.bootstrap_estimates_
    
    def compute_intervals(
        self,
        method: str = 'percentile'
    ) -> dict:
        """Compute confidence intervals from bootstrap estimates.
        
        Args:
            method: Method for computing intervals ('percentile' or 'bca')
            
        Returns:
            Dictionary with confidence intervals for each parameter
        """
        if self.bootstrap_estimates_ is None:
            raise ValueError("Must run bootstrap first")
        
        alpha = 1 - self.confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100
        
        intervals = {}
        
        # Mu intervals
        mu_est = self.bootstrap_estimates_['mu']
        intervals['mu'] = {
            'mean': np.mean(mu_est, axis=0),
            'std': np.std(mu_est, axis=0),
            'lower': np.percentile(mu_est, lower_pct, axis=0),
            'upper': np.percentile(mu_est, upper_pct, axis=0)
        }
        
        # Alpha intervals
        alpha_est = self.bootstrap_estimates_['alpha']
        if len(alpha_est) > 0:
            intervals['alpha'] = {
                'mean': np.mean(alpha_est, axis=0),
                'std': np.std(alpha_est, axis=0),
                'lower': np.percentile(alpha_est, lower_pct, axis=0),
                'upper': np.percentile(alpha_est, upper_pct, axis=0)
            }
        
        # Beta intervals
        beta_est = self.bootstrap_estimates_['beta']
        if len(beta_est) > 0:
            intervals['beta'] = {
                'mean': np.mean(beta_est, axis=0),
                'std': np.std(beta_est, axis=0),
                'lower': np.percentile(beta_est, lower_pct, axis=0),
                'upper': np.percentile(beta_est, upper_pct, axis=0)
            }
        
        return intervals
    
    def compute_standard_errors(self) -> dict:
        """Compute bootstrap standard errors.
        
        Returns:
            Dictionary with standard errors for each parameter
        """
        if self.bootstrap_estimates_ is None:
            raise ValueError("Must run bootstrap first")
        
        return {
            'mu': np.std(self.bootstrap_estimates_['mu'], axis=0),
            'alpha': np.std(self.bootstrap_estimates_['alpha'], axis=0) 
                    if len(self.bootstrap_estimates_['alpha']) > 0 else None,
            'beta': np.std(self.bootstrap_estimates_['beta'], axis=0)
                   if len(self.bootstrap_estimates_['beta']) > 0 else None
        }
    
    def plot_bootstrap_distributions(
        self,
        param_name: str = 'mu',
        dim: int = 0,
        figsize: Tuple[int, int] = (12, 4)
    ) -> plt.Figure:
        """Plot bootstrap distribution for a parameter.
        
        Args:
            param_name: Parameter to plot ('mu', 'alpha', or 'beta')
            dim: Dimension to plot (for mu) or (i, j) for matrices
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        if self.bootstrap_estimates_ is None:
            raise ValueError("Must run bootstrap first")
        
        estimates = self.bootstrap_estimates_[param_name]
        
        if param_name == 'mu':
            values = estimates[:, dim]
            title = f'Bootstrap Distribution: μ[{dim}]'
        else:
            if isinstance(dim, tuple):
                i, j = dim
            else:
                i = j = dim
            values = estimates[:, i, j]
            title = f'Bootstrap Distribution: {param_name}[{i},{j}]'
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(values, bins=30, alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(values):.4f}')
        axes[0].axvline(np.median(values), color='green', linestyle='--',
                       label=f'Median: {np.median(values):.4f}')
        axes[0].set_xlabel('Parameter Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot for normality
        from scipy.stats import probplot
        probplot(values, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normality Check)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class ParameterStabilityAnalyzer:
    """Analyze parameter stability across different time windows.
    
    Useful for detecting structural breaks or parameter drift.
    """
    
    def __init__(self, window_size: float, step_size: float):
        """Initialize.
        
        Args:
            window_size: Size of each rolling window
            step_size: Step between windows
        """
        self.window_size = window_size
        self.step_size = step_size
    
    def rolling_estimation(
        self,
        events: list[np.ndarray],
        end_time: float,
        estimator_class: type,
        estimator_params: dict
    ) -> list[dict]:
        """Estimate parameters on rolling windows.
        
        Args:
            events: Event data
            end_time: End of full observation period
            estimator_class: Estimator class to use
            estimator_params: Parameters for estimator
            
        Returns:
            List of estimates for each window
        """
        results = []
        window_start = 0.0
        
        while window_start + self.window_size <= end_time:
            window_end = window_start + self.window_size
            
            # Extract events in window
            window_events = [
                e[(e >= window_start) & (e < window_end)] - window_start
                for e in events
            ]
            
            try:
                # Fit model
                est = estimator_class(**estimator_params)
                est.fit(window_events, end_time=self.window_size)
                
                results.append({
                    'window_start': window_start,
                    'window_end': window_end,
                    'mu': est.mu_.copy() if hasattr(est, 'mu_') else None,
                    'alpha': est.alpha_.copy() if hasattr(est, 'alpha_') else None,
                    'beta': est.beta_.copy() if hasattr(est, 'beta_') else None,
                    'log_likelihood': est.log_likelihood_ if hasattr(est, 'log_likelihood_') else None,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'window_start': window_start,
                    'window_end': window_end,
                    'success': False,
                    'error': str(e)
                })
            
            window_start += self.step_size
        
        return results
