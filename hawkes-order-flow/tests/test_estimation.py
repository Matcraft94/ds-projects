"""Tests for estimation algorithms."""

import numpy as np
import pytest

from hawkes.estimation import MultivariateHawkesMLE, MultivariateHawkesEM
from hawkes.kernels import ExponentialKernel
from hawkes.utils.data_loader import simulate_hawkes_process


class TestMultivariateHawkesMLE:
    """Test suite for MLE estimator."""
    
    @pytest.fixture
    def sample_events(self):
        """Generate synthetic events for testing."""
        np.random.seed(42)
        mu = np.array([0.5, 0.5])
        alpha = np.array([[0.1, 0.05], [0.05, 0.1]])
        beta = np.ones((2, 2)) * 0.5
        events = simulate_hawkes_process(mu, alpha, beta, T=100.0, seed=42)
        return events
    
    def test_initialization(self):
        kernel = ExponentialKernel(n_dims=2)
        estimator = MultivariateHawkesMLE(kernel)
        assert estimator.n_dims == 2
    
    def test_fit(self, sample_events):
        kernel = ExponentialKernel(n_dims=2)
        estimator = MultivariateHawkesMLE(kernel)
        
        estimator.fit(sample_events, end_time=100.0, method='L-BFGS-B')
        
        assert estimator.mu_ is not None
        assert estimator.kernel_params_ is not None
        assert estimator.log_likelihood_ is not None
        assert len(estimator.mu_) == 2
    
    def test_branching_ratio(self, sample_events):
        kernel = ExponentialKernel(n_dims=2)
        estimator = MultivariateHawkesMLE(kernel)
        estimator.fit(sample_events, end_time=100.0, method='L-BFGS-B')
        
        B = estimator.compute_branching_ratio()
        assert B.shape == (2, 2)
        assert np.all(B >= 0)
    
    def test_spectral_radius(self, sample_events):
        kernel = ExponentialKernel(n_dims=2)
        estimator = MultivariateHawkesMLE(kernel)
        estimator.fit(sample_events, end_time=100.0, method='L-BFGS-B')
        
        rho = estimator.compute_spectral_radius()
        assert rho >= 0
    
    def test_predict_intensity(self, sample_events):
        kernel = ExponentialKernel(n_dims=2)
        estimator = MultivariateHawkesMLE(kernel)
        estimator.fit(sample_events, end_time=100.0, method='L-BFGS-B')
        
        times = np.array([50.0, 75.0, 100.0])
        intensities = estimator.predict_intensity(times, sample_events)
        
        assert intensities.shape == (2, 3)
        assert np.all(intensities >= 0)


class TestMultivariateHawkesEM:
    """Test suite for EM estimator."""
    
    @pytest.fixture
    def sample_events(self):
        """Generate synthetic events for testing."""
        np.random.seed(42)
        mu = np.array([0.5, 0.5])
        alpha = np.array([[0.1, 0.05], [0.05, 0.1]])
        beta = np.ones((2, 2)) * 0.5
        events = simulate_hawkes_process(mu, alpha, beta, T=100.0, seed=42)
        return events
    
    def test_initialization(self):
        kernel = ExponentialKernel(n_dims=2)
        estimator = MultivariateHawkesEM(kernel, max_iter=10)
        assert estimator.n_dims == 2
        assert estimator.max_iter == 10
    
    def test_fit(self, sample_events):
        kernel = ExponentialKernel(n_dims=2)
        estimator = MultivariateHawkesEM(kernel, max_iter=20, verbose=False)
        
        estimator.fit(sample_events, end_time=100.0)
        
        assert estimator.mu_ is not None
        assert estimator.log_likelihood_ is not None
        assert len(estimator.mu_) == 2


def test_mle_vs_em_consistency():
    """Test that MLE and EM give similar results on simple data."""
    np.random.seed(42)
    mu = np.array([0.5, 0.5])
    alpha = np.array([[0.1, 0.05], [0.05, 0.1]])
    beta = np.ones((2, 2)) * 0.5
    events = simulate_hawkes_process(mu, alpha, beta, T=500.0, seed=42)
    
    kernel_mle = ExponentialKernel(n_dims=2)
    mle = MultivariateHawkesMLE(kernel_mle)
    mle.fit(events, end_time=500.0, method='L-BFGS-B')
    
    kernel_em = ExponentialKernel(n_dims=2)
    em = MultivariateHawkesEM(kernel_em, max_iter=50)
    em.fit(events, end_time=500.0)
    
    # Check that both estimate similar mu
    assert np.allclose(mle.mu_, em.mu_, rtol=0.5)
    
    # Check that both give stable results
    assert mle.compute_spectral_radius() < 2.0
    assert em.compute_spectral_radius() < 2.0
