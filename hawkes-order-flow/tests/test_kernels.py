"""Tests for kernel implementations."""

import numpy as np
import pytest

from hawkes.kernels import ExponentialKernel, SumExponentialsKernel


class TestExponentialKernel:
    """Test suite for ExponentialKernel."""
    
    def test_initialization(self):
        kernel = ExponentialKernel(n_dims=4)
        assert kernel.n_dims == 4
        assert kernel.n_params == 32  # 16 alpha + 16 beta
    
    def test_param_names(self):
        kernel = ExponentialKernel(n_dims=2)
        names = kernel.param_names
        assert len(names) == 8
        assert 'alpha_00' in names
        assert 'beta_11' in names
    
    def test_evaluate(self):
        kernel = ExponentialKernel(n_dims=2)
        params = np.array([
            0.5, 0.3, 0.2, 0.4,  # alpha matrix
            1.0, 1.0, 1.0, 1.0   # beta matrix
        ])
        kernel.set_params(params)
        
        t = np.array([0.0, 0.5, 1.0, 2.0])
        result = kernel.evaluate(t)
        
        assert result.shape == (2, 2, 4)
        # At t=0, kernel = alpha
        assert np.isclose(result[0, 0, 0], 0.5)
        # At t>0, kernel decays
        assert result[0, 0, 1] < result[0, 0, 0]
    
    def test_integrate(self):
        kernel = ExponentialKernel(n_dims=2)
        alpha = np.array([[0.5, 0.3], [0.2, 0.4]])
        beta = np.array([[1.0, 1.0], [1.0, 1.0]])
        params = np.concatenate([alpha.flatten(), beta.flatten()])
        kernel.set_params(params)
        
        B = kernel.integrate()
        expected = alpha / beta
        
        assert np.allclose(B, expected)
    
    def test_initialize_params(self):
        kernel = ExponentialKernel(n_dims=2)
        events = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.5, 2.5])
        ]
        
        params = kernel.initialize_params(events)
        
        assert len(params) == 8
        assert np.all(params[:4] >= 0)  # alphas non-negative
        assert np.all(params[4:] > 0)   # betas positive
    
    def test_get_bounds(self):
        kernel = ExponentialKernel(n_dims=2)
        lower, upper = kernel.get_bounds()
        
        assert len(lower) == 8
        assert len(upper) == 8
        assert np.all(lower[:4] == 0)  # alphas >= 0
        assert np.all(lower[4:] > 0)   # betas > 0


class TestSumExponentialsKernel:
    """Test suite for SumExponentialsKernel."""
    
    def test_initialization(self):
        kernel = SumExponentialsKernel(n_dims=4, n_components=2)
        assert kernel.n_dims == 4
        assert kernel.n_components == 2
        assert kernel.n_params == 64  # 2 * 2 * 16
    
    def test_evaluate(self):
        kernel = SumExponentialsKernel(n_dims=2, n_components=2)
        # params: [alpha1, alpha2, beta1, beta2]
        params = np.array([
            0.3, 0.1, 0.2, 0.05,  # alpha1
            0.2, 0.1, 0.1, 0.05,  # alpha2
            2.0, 2.0, 2.0, 2.0,   # beta1
            0.5, 0.5, 0.5, 0.5    # beta2
        ])
        kernel.set_params(params)
        
        t = np.array([0.0, 1.0, 5.0])
        result = kernel.evaluate(t)
        
        assert result.shape == (2, 2, 3)
        # At t=0, should be sum of alphas
        expected_00 = 0.3 + 0.2  # alpha1_00 + alpha2_00
        assert np.isclose(result[0, 0, 0], expected_00)
    
    def test_integrate(self):
        kernel = SumExponentialsKernel(n_dims=2, n_components=2)
        alpha1 = np.array([[0.5, 0.3], [0.2, 0.4]])
        alpha2 = np.array([[0.2, 0.1], [0.1, 0.2]])
        beta1 = np.ones((2, 2))
        beta2 = np.ones((2, 2)) * 0.5
        
        params = np.concatenate([
            alpha1.flatten(), alpha2.flatten(),
            beta1.flatten(), beta2.flatten()
        ])
        kernel.set_params(params)
        
        B = kernel.integrate()
        expected = alpha1/beta1 + alpha2/beta2
        
        assert np.allclose(B, expected)
    
    def test_timescale_importance(self):
        kernel = SumExponentialsKernel(n_dims=2, n_components=2)
        # Make first component more important
        params = np.array([
            0.8, 0.4, 0.4, 0.2,  # alpha1 (larger)
            0.1, 0.05, 0.05, 0.05,  # alpha2 (smaller)
            1.0, 1.0, 1.0, 1.0,   # beta1
            1.0, 1.0, 1.0, 1.0    # beta2
        ])
        kernel.set_params(params)
        
        importance = kernel.get_timescale_importance()
        
        assert len(importance) == 2
        assert importance[0] > importance[1]  # First component more important
        assert np.isclose(np.sum(importance), 1.0)


def test_kernel_consistency():
    """Test that sum of exponentials with single component equals exponential."""
    from hawkes.kernels.exponential import ExponentialKernel
    from hawkes.kernels.sum_exponential import SumExponentialsKernel
    
    t = np.array([0.0, 0.5, 1.0, 2.0])
    
    # Exponential kernel
    exp_kernel = ExponentialKernel(n_dims=2)
    exp_params = np.array([
        0.5, 0.3, 0.2, 0.4,  # alpha
        1.0, 1.0, 1.0, 1.0   # beta
    ])
    exp_kernel.set_params(exp_params)
    exp_result = exp_kernel.evaluate(t)
    
    # SumExponentials with n_components=1
    sum_kernel = SumExponentialsKernel(n_dims=2, n_components=1)
    sum_params = np.array([
        0.5, 0.3, 0.2, 0.4,  # alpha1
        1.0, 1.0, 1.0, 1.0   # beta1
    ])
    sum_kernel.set_params(sum_params)
    sum_result = sum_kernel.evaluate(t)
    
    assert np.allclose(exp_result, sum_result)
