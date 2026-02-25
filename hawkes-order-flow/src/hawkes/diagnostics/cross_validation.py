"""Time-series cross-validation for Hawkes processes.

Implements rolling-window and expanding-window cross-validation
specifically designed for temporal point process data.

Reference:
    Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures 
    for model selection. Statistics Surveys.
"""

import numpy as np
from typing import Callable, Optional, List, Dict
from dataclasses import dataclass
import pandas as pd


@dataclass
class CVResult:
    """Result from a single CV fold."""
    train_start: float
    train_end: float
    test_start: float
    test_end: float
    log_likelihood_train: float
    log_likelihood_test: float
    aic_train: float
    bic_train: float
    n_params: int
    n_events_train: int
    n_events_test: int


class TimeSeriesCrossValidator:
    """Time-series cross-validation for Hawkes processes.
    
    Unlike standard CV, we respect temporal ordering:
    - Train on past, test on future
    - No shuffling to preserve temporal dependence
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        min_train_size: float = 100.0,
        test_size: Optional[float] = None,
        gap: float = 0.0
    ):
        """Initialize CV.
        
        Args:
            n_folds: Number of CV folds
            min_train_size: Minimum training period size
            test_size: Test period size (if None, uses equal splits)
            gap: Gap between train and test (purged CV)
        """
        self.n_folds = n_folds
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap = gap
    
    def create_folds(
        self,
        events: list[np.ndarray],
        end_time: float
    ) -> list[dict]:
        """Create CV folds respecting temporal structure.
        
        Args:
            events: Event data
            end_time: End of observation
            
        Returns:
            List of fold specifications
        """
        total_duration = end_time
        
        if self.test_size is None:
            # Calculate test size to use all data
            remaining = total_duration - self.min_train_size
            self.test_size = remaining / self.n_folds
        
        folds = []
        for i in range(self.n_folds):
            train_start = 0.0
            train_end = self.min_train_size + i * self.test_size
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            if test_end > total_duration:
                test_end = total_duration
            
            folds.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'fold': i + 1
            })
        
        return folds
    
    def cross_validate(
        self,
        events: list[np.ndarray],
        end_time: float,
        estimator_class: type,
        estimator_params: dict,
        fit_kwargs: Optional[dict] = None
    ) -> list[CVResult]:
        """Run cross-validation.
        
        Args:
            events: Event data
            end_time: End of observation
            estimator_class: Estimator class
            estimator_params: Estimator initialization parameters
            fit_kwargs: Additional fit arguments
            
        Returns:
            List of CV results
        """
        if fit_kwargs is None:
            fit_kwargs = {}
        
        folds = self.create_folds(events, end_time)
        results = []
        
        for fold in folds:
            # Split events
            train_events = [
                e[(e >= fold['train_start']) & (e < fold['train_end'])] - fold['train_start']
                for e in events
            ]
            
            test_events = [
                e[(e >= fold['test_start']) & (e < fold['test_end'])] - fold['test_start']
                for e in events
            ]
            
            # Fit on training data
            try:
                estimator = estimator_class(**estimator_params)
                estimator.fit(
                    train_events, 
                    end_time=fold['train_end'] - fold['train_start'],
                    **fit_kwargs
                )
                
                # Compute metrics
                train_ll = estimator.log_likelihood_
                n_params = (estimator.n_dims + 
                           estimator.n_dims**2 * 2)  # mu + alpha + beta
                
                # Compute test likelihood
                test_ll = self._compute_test_likelihood(
                    estimator, test_events, 
                    fold['test_end'] - fold['test_start']
                )
                
                # AIC/BIC on training data
                n_train = sum(len(e) for e in train_events)
                aic = -2 * train_ll + 2 * n_params
                bic = -2 * train_ll + n_params * np.log(n_train)
                
                # Add fold number to result for tracking
                result = CVResult(
                    train_start=fold['train_start'],
                    train_end=fold['train_end'],
                    test_start=fold['test_start'],
                    test_end=fold['test_end'],
                    log_likelihood_train=train_ll,
                    log_likelihood_test=test_ll,
                    aic_train=aic,
                    bic_train=bic,
                    n_params=n_params,
                    n_events_train=n_train,
                    n_events_test=sum(len(e) for e in test_events)
                )
                result.fold = fold['fold']  # Add fold number
                results.append(result)
                
            except Exception as e:
                print(f"Warning: Fold {fold['fold']} failed: {e}")
                continue
        
        return results
    
    def _compute_test_likelihood(
        self,
        estimator,
        test_events: list[np.ndarray],
        test_end: float
    ) -> float:
        """Compute log-likelihood on test data.
        
        Args:
            estimator: Fitted estimator
            test_events: Test event data
            test_end: End of test period
            
        Returns:
            Log-likelihood value
        """
        # This is a simplified version - in practice would need
        # to extract parameters and compute properly
        try:
            # Try to compute using estimator's methods
            # This may need customization based on estimator type
            mu = estimator.mu_
            alpha = estimator.alpha_ if hasattr(estimator, 'alpha_') else None
            beta = estimator.beta_ if hasattr(estimator, 'beta_') else None
            
            if alpha is None or beta is None:
                # Extract from kernel_params if available
                if hasattr(estimator, 'kernel_params_'):
                    n = estimator.n_dims
                    alpha = estimator.kernel_params_[:n*n].reshape(n, n)
                    beta = estimator.kernel_params_[n*n:].reshape(n, n)
            
            # Simple log-likelihood computation
            ll = 0.0
            for i in range(estimator.n_dims):
                # Baseline contribution
                ll += len(test_events[i]) * np.log(mu[i] + 1e-10)
                ll -= mu[i] * test_end
                
                # Self-excitation (simplified)
                if alpha is not None and beta is not None:
                    for t_i in test_events[i]:
                        past = test_events[i][test_events[i] < t_i]
                        if len(past) > 0:
                            dt = t_i - past
                            ll += np.sum(
                                np.log(mu[i] + alpha[i,i] * np.exp(-beta[i,i] * dt))
                            )
            
            return ll
        except:
            return np.nan
    
    def summarize_results(self, results: list[CVResult]) -> pd.DataFrame:
        """Create summary table of CV results.
        
        Args:
            results: List of CVResult objects
            
        Returns:
            Summary DataFrame
        """
        data = []
        for r in results:
            data.append({
                'train_end': r.train_end,
                'test_start': r.test_start,
                'train_ll': r.log_likelihood_train,
                'test_ll': r.log_likelihood_test,
                'aic': r.aic_train,
                'bic': r.bic_train,
                'n_train': r.n_events_train,
                'n_test': r.n_events_test
            })
        
        df = pd.DataFrame(data)
        
        # Add aggregate statistics
        df['test_ll_per_event'] = df['test_ll'] / df['n_test']
        
        return df


class ModelComparisonCV:
    """Compare multiple models using cross-validation.
    """
    
    def __init__(self, cv: TimeSeriesCrossValidator):
        """Initialize with CV strategy.
        
        Args:
            cv: Cross-validator instance
        """
        self.cv = cv
    
    def compare_models(
        self,
        events: list[np.ndarray],
        end_time: float,
        models: dict[str, tuple[type, dict]]
    ) -> pd.DataFrame:
        """Compare multiple models using CV.
        
        Args:
            events: Event data
            end_time: End of observation
            models: Dictionary of model_name -> (estimator_class, params)
            
        Returns:
            Comparison DataFrame
        """
        comparison = []
        
        for name, (est_class, params) in models.items():
            print(f"Running CV for {name}...")
            
            results = self.cv.cross_validate(
                events, end_time, est_class, params
            )
            
            if results:
                summary = self.cv.summarize_results(results)
                comparison.append({
                    'model': name,
                    'mean_test_ll': summary['test_ll'].mean(),
                    'std_test_ll': summary['test_ll'].std(),
                    'mean_test_ll_per_event': summary['test_ll_per_event'].mean(),
                    'mean_aic': summary['aic'].mean(),
                    'mean_bic': summary['bic'].mean(),
                    'n_folds': len(results)
                })
        
        return pd.DataFrame(comparison)
