import pytest
import numpy as np
from fastpt import FASTPT
from fastpt.JAXPT import JAXPT
import os
import jax
from jax import numpy as jnp
from fastpt.jax_utils import jax_k_extend

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]
P_window = jnp.array([0.2, 0.2])
C_window = 0.75

if __name__ == "__main__":
    from time import time
    fapt = FASTPT(d[:, 0])
    jpt = JAXPT(jnp.array(d[:, 0]))
    t0 = time()
    fapt.J_k_tensor(P, fapt.X_IA_A, P_window=np.array([0.2, 0.2]), C_window=C_window)
    t1 = time()
    jpt.jJ_k_tensor(P, jpt.X_IA_A, jpt.k_extrap, jpt.k_final, jpt.k_size, jpt.n_pad, jpt.id_pad, 
                    jpt.l, jpt.m, jpt.N, P_window=P_window, C_window=C_window)
    t2 = time()
    print(f"FASTPT time: {t1 - t0:.4f} seconds")
    print(f"JAXPT time: {t2 - t1:.4f} seconds")


@pytest.fixture
def jpt(): 
    k = jnp.array(d[:, 0])
    return JAXPT(k)

@pytest.fixture
def fpt():
    d = np.loadtxt(data_path)
    k = np.array(d[:, 0])
    return FASTPT(k)

############## Equality Tests ##############
def test_P_window(jpt, fpt):
    # Test that the P_window method returns the same result for JAXPT and FASTPT
    from fastpt.jax_utils import p_window as jp_window
    from fastpt.fastpt_extr import p_window as fp_window
    jax = jp_window(jpt.k_original, P_window[0], P_window[1])
    fast = fp_window(fpt.k_original, P_window[0], P_window[1])
    assert np.allclose(jax, fast), "P_window results are not equal"



############## k_extend Tests ##############
def test_k_extend_initialization(jpt, fpt):
    """Test that jax_k_extend initializes with the same k values as k_extend"""
    from fastpt.P_extend import k_extend
    
    # Test with no extensions
    jk_ext = jax_k_extend(jpt.k_original)
    k_ext = k_extend(fpt.k_original)
    
    assert isinstance(jk_ext.k, jnp.ndarray), "jax_k_extend.k is not a JAX array"
    
    # Test with low extension
    low_ext = -4.0  # Extend to 10^-4
    jk_ext_low = jax_k_extend(jpt.k_original, low=low_ext)
    k_ext_low = k_extend(fpt.k_original, low=low_ext)
    
    assert isinstance(jk_ext_low.k, jnp.ndarray), "jax_k_extend.k with low extension is not a JAX array"
    assert np.allclose(np.array(jk_ext_low.k), k_ext_low.k), "k arrays don't match with low extension"
    
    # Test with high extension
    high_ext = 3.0  # Extend to 10^3
    jk_ext_high = jax_k_extend(jpt.k_original, high=high_ext)
    k_ext_high = k_extend(fpt.k_original, high=high_ext)
    
    assert isinstance(jk_ext_high.k, jnp.ndarray), "jax_k_extend.k with high extension is not a JAX array"
    assert np.allclose(np.array(jk_ext_high.k), k_ext_high.k), "k arrays don't match with high extension"
    
    # Test with both extensions
    jk_ext_both = jax_k_extend(jpt.k_original, low=low_ext, high=high_ext)
    k_ext_both = k_extend(fpt.k_original, low=low_ext, high=high_ext)
    
    assert isinstance(jk_ext_both.k, jnp.ndarray), "jax_k_extend.k with both extensions is not a JAX array"
    assert np.allclose(np.array(jk_ext_both.k), k_ext_both.k), "k arrays don't match with both extensions"
    
    # Test that id_extrap is correctly converted to JAX array
    assert isinstance(jk_ext_both.id_extrap, jnp.ndarray), "jax_k_extend.id_extrap is not a JAX array"
    assert np.array_equal(np.array(jk_ext_both.id_extrap), k_ext_both.id_extrap), "id_extrap arrays don't match"

def test_extrap_k(jpt, fpt):
    """Test that extrap_k returns the same values in both implementations"""
    from fastpt.P_extend import k_extend
    
    # Initialize with both extensions
    low_ext, high_ext = -4.0, 3.0
    jk_ext = jax_k_extend(jpt.k_original, low=low_ext, high=high_ext)
    k_ext = k_extend(fpt.k_original, low=low_ext, high=high_ext)
    
    # Test extrap_k
    jk = jk_ext.extrap_k()
    k = k_ext.extrap_k()
    
    assert isinstance(jk, jnp.ndarray), "jax_k_extend.extrap_k result is not a JAX array"
    assert np.allclose(np.array(jk), k), "extrap_k results don't match"

def test_extrap_P_low(jpt, fpt):
    """Test that extrap_P_low returns the same values in both implementations"""
    from fastpt.P_extend import k_extend
    
    # Initialize with low extension only
    low_ext = -4.0
    jk_ext = jax_k_extend(jpt.k_original, low=low_ext)
    k_ext = k_extend(fpt.k_original, low=low_ext)
    
    # Copy of power spectrum for testing
    P_jax = jnp.array(P)
    P_np = np.array(P)
    
    # Test extrap_P_low
    jP_low = jk_ext.extrap_P_low(P_jax)
    P_low = k_ext.extrap_P_low(P_np)
    
    assert isinstance(jP_low, jnp.ndarray), "jax_k_extend.extrap_P_low result is not a JAX array"
    assert np.allclose(np.array(jP_low), P_low), "extrap_P_low results don't match"
    
    # Test that the returned array has the expected shape
    assert jP_low.shape[0] == jk_ext.k.shape[0], "extrap_P_low output size mismatch with k array"

def test_extrap_P_high(jpt, fpt):
    """Test that extrap_P_high returns the same values in both implementations"""
    from fastpt.P_extend import k_extend
    
    # Initialize with high extension only
    high_ext = 3.0
    jk_ext = jax_k_extend(jpt.k_original, high=high_ext)
    k_ext = k_extend(fpt.k_original, high=high_ext)
    
    # Copy of power spectrum for testing
    P_jax = jnp.array(P)
    P_np = np.array(P)
    
    # Test extrap_P_high
    jP_high = jk_ext.extrap_P_high(P_jax)
    P_high = k_ext.extrap_P_high(P_np)
    
    assert isinstance(jP_high, jnp.ndarray), "jax_k_extend.extrap_P_high result is not a JAX array"
    assert np.allclose(np.array(jP_high), P_high), "extrap_P_high results don't match"
    
    # Test that the returned array has the expected shape
    assert jP_high.shape[0] == jk_ext.k.shape[0], "extrap_P_high output size mismatch with k array"

def test_PK_original(jpt, fpt):
    """Test that PK_original returns the same values in both implementations"""
    from fastpt.P_extend import k_extend
    
    # Initialize with both extensions
    low_ext, high_ext = -4.0, 3.0
    jk_ext = jax_k_extend(jpt.k_original, low=low_ext, high=high_ext)
    k_ext = k_extend(fpt.k_original, low=low_ext, high=high_ext)
    
    # Create extended P
    P_jax = jnp.ones_like(jk_ext.k)  # Dummy power spectrum matching k size
    P_np = np.ones_like(k_ext.k)
    
    # Test PK_original
    jk_orig, jP_orig = jk_ext.PK_original(P_jax)
    k_orig, P_orig = k_ext.PK_original(P_np)
    
    assert isinstance(jk_orig, jnp.ndarray), "jax_k_extend.PK_original k result is not a JAX array"
    assert isinstance(jP_orig, jnp.ndarray), "jax_k_extend.PK_original P result is not a JAX array"
    assert np.allclose(np.array(jk_orig), k_orig), "PK_original k results don't match"
    assert np.allclose(np.array(jP_orig), P_orig), "PK_original P results don't match"

def test_full_extrapolation_workflow(jpt, fpt):
    """Test the complete extrapolation workflow with both implementations"""
    from fastpt.P_extend import k_extend
    
    # Initialize with both extensions
    low_ext, high_ext = -4.0, 3.0
    jk_ext = jax_k_extend(jpt.k_original, low=low_ext, high=high_ext)
    k_ext = k_extend(fpt.k_original, low=low_ext, high=high_ext)
    
    # Copy of power spectrum for testing
    P_jax = jnp.array(P)
    P_np = np.array(P)
    
    # 1. Extend P to lower k
    jP_low = jk_ext.extrap_P_low(P_jax)
    P_low = k_ext.extrap_P_low(P_np)
    
    # 2. Extend P to higher k
    jP_both = jk_ext.extrap_P_high(jP_low)
    P_both = k_ext.extrap_P_high(P_low)
    
    # Check full extended array
    assert isinstance(jP_both, jnp.ndarray), "Final extended P is not a JAX array"
    assert np.allclose(np.array(jP_both), P_both), "Full extrapolation workflow results don't match"
    
    # 3. Extract original k range
    jk_orig, jP_orig = jk_ext.PK_original(jP_both)
    k_orig, P_orig = k_ext.PK_original(P_both)
    
    # Check that we get back the original P within numerical tolerance
    assert np.allclose(np.array(jP_orig), P_orig), "Retrieved original P doesn't match"
    assert np.allclose(np.array(jP_orig), P_np), "Retrieved original P doesn't match input"

def test_jax_differentiability():
    """Test that jax_k_extend functions are differentiable with JAX"""    
    # Load test data
    d = np.loadtxt(data_path)
    k = jnp.array(d[:, 0])
    P_jax = jnp.array(d[:, 1])
    
    # Set up extension
    low_ext, high_ext = -4.0, 3.0
    jk_ext = jax_k_extend(k, low=low_ext, high=high_ext)
    
    # Define a function that uses the extrapolation
    def process_power_spectrum(P):
        # Apply extrapolations
        P_extended = jk_ext.extrap_P_low(P)
        P_extended = jk_ext.extrap_P_high(P_extended)
        
        # Compute a scalar result
        return jnp.mean(P_extended)
    
    # Test that we can compute gradients
    try:
        grad_func = jax.grad(process_power_spectrum)
        gradient = grad_func(P_jax)
        
        # Check gradient properties
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        assert gradient.shape == P_jax.shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"
        
        # Print summary info
        print(f"Gradient mean: {jnp.mean(gradient)}, min: {jnp.min(gradient)}, max: {jnp.max(gradient)}")
        
        # For the test to pass, we just need to confirm we can compute the gradient
        assert True, "JAX differentiation test passed"
    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")