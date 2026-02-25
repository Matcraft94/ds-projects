"""Tests for diagnostics module."""

import numpy as np
import pytest

from hawkes.diagnostics import StabilityDiagnostics


class TestStabilityDiagnostics:
    """Test suite for stability diagnostics."""
    
    def test_stable_process(self):
        """Test with a stable branching matrix."""
        # Matrix with spectral radius < 1
        B = np.array([
            [0.2, 0.1],
            [0.1, 0.2]
        ])
        diag = StabilityDiagnostics(B)
        
        assert diag.is_stable()
        assert diag.get_criticality() < 1.0
    
    def test_unstable_process(self):
        """Test with an unstable branching matrix."""
        # Matrix with spectral radius > 1
        B = np.array([
            [1.5, 0.1],
            [0.1, 0.5]
        ])
        diag = StabilityDiagnostics(B)
        
        assert not diag.is_stable()
        assert diag.get_criticality() > 1.0
    
    def test_critical_process(self):
        """Test with near-critical process."""
        B = np.array([
            [0.5, 0.4],
            [0.4, 0.5]
        ])
        diag = StabilityDiagnostics(B)
        
        # Should be stable but close to critical
        assert diag.is_stable()
        assert 0.8 < diag.get_criticality() < 1.0
    
    def test_branching_ratios(self):
        B = np.array([
            [0.3, 0.2],
            [0.1, 0.4]
        ])
        diag = StabilityDiagnostics(B)
        
        total = diag.get_total_branching_ratio()
        assert np.isclose(total, np.sum(B))
        
        dim_ratios = diag.get_dimension_branching_ratios()
        assert len(dim_ratios) == 2
        assert np.isclose(dim_ratios[0], 0.3 + 0.2)
        assert np.isclose(dim_ratios[1], 0.1 + 0.4)
    
    def test_causal_influence(self):
        B = np.array([
            [0.3, 0.2],
            [0.1, 0.4]
        ])
        diag = StabilityDiagnostics(B)
        
        influence = diag.get_causal_influence()
        
        # Each row should sum to 1 (normalized)
        assert np.allclose(np.sum(influence, axis=1), 1.0)
        
        # Check specific values
        assert np.isclose(influence[0, 0], 0.3 / 0.5)
        assert np.isclose(influence[0, 1], 0.2 / 0.5)
    
    def test_endogenous_ratio(self):
        B = np.array([
            [0.3, 0.2],
            [0.1, 0.4]
        ])
        diag = StabilityDiagnostics(B)
        
        ratio = diag.get_endogenous_ratio()
        total_b = diag.get_total_branching_ratio()
        expected = total_b / (1 + total_b)
        
        assert np.isclose(ratio, expected)
    
    def test_generate_report(self):
        B = np.array([
            [0.3, 0.2],
            [0.1, 0.4]
        ])
        diag = StabilityDiagnostics(B)
        
        report = diag.generate_report()
        
        assert 'spectral_radius' in report
        assert 'is_stable' in report
        assert 'branching_matrix' in report
        assert isinstance(report['is_stable'], bool)
