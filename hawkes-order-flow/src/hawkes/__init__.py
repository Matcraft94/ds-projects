"""Hawkes Process for Order Flow Alpha.

Multivariate Hawkes process implementation for extracting predictive 
signals from high-frequency order flow data.
"""

__version__ = "0.1.0"

from . import kernels
from . import estimation
from . import diagnostics
from . import backtesting
from . import utils

__all__ = [
    "kernels",
    "estimation",
    "diagnostics",
    "backtesting",
    "utils",
]
