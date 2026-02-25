"""Diagnostics and validation tools for Hawkes processes.

This module provides comprehensive diagnostic tools for assessing
Hawkes process model fit, including:

- Stability analysis (spectral radius, branching ratios)
- Residual diagnostics (compensator analysis, Q-Q plots)
- Goodness-of-fit tests (Kolmogorov-Smirnov, Ljung-Box)
- Bootstrap confidence intervals
- Time-series cross-validation
- Baseline model comparison

Example:
    >>> from hawkes.diagnostics import (
    ...     StabilityDiagnostics,
    ...     ResidualDiagnostics,
    ...     BootstrapConfidenceIntervals,
    ...     ModelComparison
    ... )
    >>> 
    >>> # Stability analysis
    >>> B = estimator.compute_branching_ratio()
    >>> diag = StabilityDiagnostics(B)
    >>> print(f"Spectral radius: {diag.spectral_radius:.4f}")
    >>> 
    >>> # Residual analysis
    >>> res_diag = ResidualDiagnostics(events, end_time)
    >>> report = res_diag.generate_report(mu, alpha, beta)
    >>> print(f"Model valid: {report['model_valid']}")
    >>> 
    >>> # Bootstrap confidence intervals
    >>> boot = BootstrapConfidenceIntervals(n_bootstrap=100)
    >>> boot.parametric_bootstrap(...)
    >>> ci = boot.compute_intervals()
    >>> 
    >>> # Model comparison
    >>> comp = ModelComparison(events, end_time)
    >>> comp.fit_all(estimator)
    >>> comp.print_summary()
"""

from .stability import StabilityDiagnostics
from .residuals import ResidualDiagnostics
from .bootstrap import BootstrapConfidenceIntervals, ParameterStabilityAnalyzer
from .cross_validation import TimeSeriesCrossValidator, ModelComparisonCV
from .baseline import (
    HomogeneousPoisson,
    PiecewisePoisson,
    ModelComparison
)

__all__ = [
    # Stability
    "StabilityDiagnostics",
    
    # Residuals and goodness-of-fit
    "ResidualDiagnostics",
    
    # Uncertainty quantification
    "BootstrapConfidenceIntervals",
    "ParameterStabilityAnalyzer",
    
    # Validation
    "TimeSeriesCrossValidator",
    "ModelComparisonCV",
    
    # Baseline comparison
    "HomogeneousPoisson",
    "PiecewisePoisson",
    "ModelComparison",
]
