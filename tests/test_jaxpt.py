import pytest
import numpy as np
from fastpt import FASTPT
from fastpt.JAXPT import JAXPT
import os
import jax
from jax import grad, jit, jacfwd
from jax import numpy as jnp
from fastpt.jax_utils import jax_k_extend
from fastpt.jax_utils import c_window as jc_window, p_window as jp_window
from fastpt.fastpt_extr import c_window as fc_window, p_window as fp_window
from time import time
from fastpt.JAXPT import fourier_coefficients, convolution

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
d = np.loadtxt(data_path)
P = d[:, 1]
P_window = jnp.array([0.2, 0.2])
C_window = 0.75

if __name__ == "__main__":
    fpt = FASTPT(d[:, 0])
    jpt = JAXPT(jnp.array(d[:, 0]))
    # For differentiating with respect to P
    def compute_term_wrt_P(P, jaxpt_instance, X, operation=None, P_window=None, C_window=None):
        return jaxpt_instance.compute_term(X, operation, P, P_window, C_window)

    # Create the gradient function
    grad_P = jax.jacfwd(compute_term_wrt_P, argnums=0)

    # Call the gradient function
    t0 = time()
    result = grad_P(P, jpt, jpt.X_IA_E, lambda x: x**2)
    t1 = time()
    print(f"Time taken: {t1 - t0} seconds")
    print(result)
    
    

@pytest.fixture
def jpt(): 
    k = jnp.array(d[:, 0])
    return JAXPT(k, low_extrap=-5, high_extrap=3)

@pytest.fixture
def fpt():
    d = np.loadtxt(data_path)
    k = np.array(d[:, 0])
    return FASTPT(k)

############## Equality Tests ##############
def test_P_window(jpt, fpt):
    # Test that the P_window method returns the same result for JAXPT and FASTPT
    jax = jp_window(jpt.k_original, P_window[0], P_window[1])
    fast = fp_window(fpt.k_original, P_window[0], P_window[1])
    assert np.allclose(jax, fast), "P_window results are not equal"

def test_C_window(jpt, fpt):
    jax = jc_window(jpt.m, int(C_window * jpt.N / 2.))
    fast = fc_window(fpt.m, int(C_window * fpt.N / 2.))
    assert np.allclose(jax, fast), "C_window results are not equal"

def test_fourier_coefficients(jpt, fpt):
    pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
    P_b1 = P * jpt.k_extrap ** (-nu1[1])
    W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
    P_b1 = P_b1 * W
    P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
    jax = fourier_coefficients(P_b1, jpt.m, jpt.N, C_window)
    fast = fpt._cache_fourier_coefficients(P_b1, C_window=C_window)
    assert np.allclose(jax, fast)

def test_convolution(jpt, fpt):
    #Tensor Case
    pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
    P_b1 = P * jpt.k_extrap ** (-nu1[1])
    P_b2 = P * jpt.k_extrap ** (-nu2[1])
    W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
    P_b1 = P_b1 * W
    P_b2 = P_b2 * W
    P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
    P_b2 = np.pad(P_b2, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)    
    c_m = fourier_coefficients(P_b1, jpt.m, jpt.N, C_window)
    c_n = fourier_coefficients(P_b2, jpt.m, jpt.N, C_window)
    jax = convolution(c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
    fast = fpt._cache_convolution(np.asarray(c_m), np.asarray(c_n), np.asarray(g_m[1,:]), np.asarray(g_n[1,:]), np.asarray(h_l[1,:]))
    assert np.allclose(jax, fast), "Convolution results are not equal"
    #Scalar Case
    pf, p, g_m, g_n, two_part_l, h_l = jpt.X_spt
    P_b = P * jpt.k_extrap ** (2)
    P_b = np.pad(P_b, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
    c_m = fourier_coefficients(P_b, jpt.m, jpt.N, C_window)
    jax = convolution(c_m, c_m, g_m[1,:], g_n[1,:], h_l[1,:], two_part_l[1])
    fast = fpt._cache_convolution(np.asarray(c_m), np.asarray(c_m), np.asarray(g_m[1,:]), np.asarray(g_n[1,:]), np.asarray(h_l[1,:]), np.asarray(two_part_l[1]))
    assert np.allclose(jax, fast), "Convolution results are not equal"

def test_j_k_scalar(jpt, fpt):
    jax = jpt.J_k_scalar(P, jpt.X_spt, -2, jpt.m, jpt.N, jpt.n_pad, jpt.id_pad,
                         jpt.k_extrap, jpt.k_final, jpt.k_size, jpt.l, C_window=C_window)
    fast = fpt.J_k_scalar(P, fpt.X_spt, -2, C_window=C_window)
    
    # Have to compare this way due to inhomogeneous shapes
    jax_0 = np.array(jax[0])
    fast_0 = fast[0]
    assert np.allclose(jax_0, fast_0), "First element of J_k_scalar differs"
    
    jax_1 = np.array(jax[1])
    fast_1 = fast[1]
    assert np.allclose(jax_1, fast_1), "Second element of J_k_scalar differs"

def test_j_k_tensor(jpt, fpt):
    jax = jpt.J_k_tensor(P, jpt.X_IA_A, jpt.k_extrap, jpt.k_final, jpt.k_size, 
                         jpt.n_pad, jpt.id_pad, jpt.l, jpt.m, jpt.N, P_window=P_window, C_window=C_window)
    fast = fpt.J_k_tensor(P, fpt.X_IA_A, P_window=np.array([0.2, 0.2]), C_window=C_window)
    
    jax_0 = np.array(jax[0])
    fast_0 = fast[0]
    assert np.allclose(jax_0, fast_0), "First element of J_k_tensor differs"
    jax_1 = np.array(jax[1])
    fast_1 = fast[1]
    assert np.allclose(jax_1, fast_1), "Second element of J_k_tensor differs"


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

############# JIT Compilation Tests ###########
def test_jit_fourier(jpt):
    """Test that the fourier_coefficients function can be JIT compiled"""
    try:
        pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
        P_b1 = P * jpt.k_extrap ** (-nu1[1])
        W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
        P_b1 = P_b1 * W
        P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
        
        # JIT compile the function
        jit_func = jit(fourier_coefficients)
        result = jit_func(P_b1, jpt.m, jpt.N, C_window)
        
        assert isinstance(result, jnp.ndarray), "JIT result is not a JAX array"
        assert result.shape == ((P_b1.shape[0] + 1),), "JIT result shape doesn't match input shape"
        
    except Exception as e:
        pytest.fail(f"JIT compilation failed with error: {str(e)}")

def test_jit_convolution(jpt):
    try: 
        pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
        P_b1 = P * jpt.k_extrap ** (-nu1[1])
        P_b2 = P * jpt.k_extrap ** (-nu2[1])
        W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
        P_b1 = P_b1 * W
        P_b2 = P_b2 * W
        P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
        P_b2 = np.pad(P_b2, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)    
        
        c_m = fourier_coefficients(P_b1, jpt.m, jpt.N, C_window)
        c_n = fourier_coefficients(P_b2, jpt.m, jpt.N, C_window)
        
        # JIT compile the convolution function
        jit_func = jit(convolution)
        result = jit_func(c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
        
        assert isinstance(result, jnp.ndarray), "JIT result is not a JAX array"
        assert result.shape == (c_m.shape[0] + c_n.shape[0] - 1,), "JIT result shape doesn't match expected shape"
    except Exception as e:
        pytest.fail(f"JIT compilation failed with error: {str(e)}")



############ Differentiability Tests ###########
def test_jax_extend_differentiability():
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

def test_P_window_differentiability(jpt):
    """Test that P_window is differentiable with JAX"""
    try:
        gradient = jacfwd(jp_window)(jpt.k_original, P_window[0], P_window[1])
        
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        expected_shape = (jpt.k_original.shape[0], jpt.k_original.shape[0])
        assert gradient.shape == expected_shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"

    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")

def test_C_window_differentiability(jpt):
    """Test that C_window is differentiable with JAX"""
    try:
        gradient = jacfwd(jc_window)(jnp.float64(jpt.m), jnp.float64(C_window * jpt.N / 2.))
        
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        expected_shape = (jpt.m.shape[0], jpt.m.shape[0])
        assert gradient.shape == expected_shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"

    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")

def test_fourier_coefficients_differentiability(jpt):
    """Test that fourier_coefficients is differentiable with JAX"""
    try:
        pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
        P_b1 = P * jpt.k_extrap ** (-nu1[1])
        W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
        P_b1 = P_b1 * W
        P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
        
        gradient = jacfwd(fourier_coefficients)(P_b1, jpt.m, jpt.N, C_window)
        
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        expected_shape = (P_b1.shape[0] + 1, P_b1.shape[0]) #<<<<<<<<< why is it 6001, 6000?
        assert gradient.shape == expected_shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"

    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")

def test_convolution_differentiability(jpt):
    """Test that convolution is differentiable with JAX"""
    try:
        pf, p, nu1, nu2, g_m, g_n, h_l = jpt.X_IA_A
        P_b1 = P * jpt.k_extrap ** (-nu1[1])
        P_b2 = P * jpt.k_extrap ** (-nu2[1])
        W = jp_window(jpt.k_extrap, P_window[0], P_window[1])
        P_b1 = P_b1 * W
        P_b2 = P_b2 * W
        P_b1 = np.pad(P_b1, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)
        P_b2 = np.pad(P_b2, pad_width=(jpt.n_pad, jpt.n_pad), mode='constant', constant_values=0)    
        
        c_m = fourier_coefficients(P_b1, jpt.m, jpt.N, C_window)
        c_n = fourier_coefficients(P_b2, jpt.m, jpt.N, C_window)
        
        gradient = jacfwd(convolution, holomorphic=True)(c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
        
        assert isinstance(gradient, jnp.ndarray), "Gradient is not a JAX array"
        expected_shape = (12001, 6001) #<<<<<<<<< why is it 6001, 6000?
        assert gradient.shape == expected_shape, "Gradient shape doesn't match input shape"
        assert not jnp.any(jnp.isnan(gradient)), "Gradient contains NaN values"

    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")

def test_j_k_scalar_differentiability(jpt):
    """Test that J_k_scalar is differentiable with JAX"""
    try:
        gradient = jacfwd(jpt.J_k_scalar)(P, jpt.X_spt, -2, jpt.m, jpt.N, jpt.n_pad,
                                         jpt.id_pad, jpt.k_extrap, jpt.k_final,
                                         jpt.k_size, jpt.l, C_window=C_window)
        
        assert isinstance(gradient, tuple), "Gradient should be a tuple"
        for i, grad_element in enumerate(gradient):
            assert isinstance(grad_element, jnp.ndarray), f"Gradient element {i} is not a JAX array"
            if i == 0:
                expected_shape = (P.shape[0], P.shape[0])
                assert grad_element.shape == expected_shape, f"Gradient element {i} shape mismatch"
            
            assert not jnp.any(jnp.isnan(grad_element)), f"Gradient element {i} contains NaN values"
        
    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")

def test_j_k_tensor_differentiability(jpt):
    """Test that J_k_tensor is differentiable with JAX"""
    try:
        gradient = jacfwd(jpt.J_k_tensor)(P, jpt.X_IA_A, jpt.k_extrap, jpt.k_final,
                                          jpt.k_size, jpt.n_pad, jpt.id_pad,
                                          jpt.l, jpt.m, jpt.N, P_window=P_window,
                                          C_window=C_window)
        
        assert isinstance(gradient, tuple), "Gradient should be a tuple"
        for i, grad_element in enumerate(gradient):
            assert isinstance(grad_element, jnp.ndarray), f"Gradient element {i} is not a JAX array"
            if i == 0:
                expected_shape = (P.shape[0], P.shape[0])
                assert grad_element.shape == expected_shape, f"Gradient element {i} shape mismatch"
            
            assert not jnp.any(jnp.isnan(grad_element)), f"Gradient element {i} contains NaN values"
        
    except Exception as e:
        pytest.fail(f"JAX differentiation failed with error: {str(e)}")