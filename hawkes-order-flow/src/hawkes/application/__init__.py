"""Real-world applications of Hawkes processes for trading.

This module provides production-ready applications:
- Real-time intensity forecasting
- Trading signal generation
- Risk management
- Performance analytics
"""

from .intensity_forecaster import IntensityForecaster
from .signal_generator import SignalGenerator, SignalType
from .risk_manager import RiskManager
from .performance_analytics import PerformanceAnalytics

__all__ = [
    'IntensityForecaster',
    'SignalGenerator',
    'SignalType',
    'RiskManager',
    'PerformanceAnalytics'
]
