import numpy as np
cimport numpy as np
from scipy.signal import fftconvolve
from numpy.fft import rfft

# Import CacheManager for type declaration
from ..CacheManager import CacheManager

ctypedef np.complex128_t COMPLEX_t
ctypedef np.float64_t FLOAT_t
ctypedef np.int64_t INT_t

# Need to declare numpy is being used
np.import_array()

cpdef np.ndarray compute_convolution(
    object cache_manager,  # Use 'object' instead of 'CacheManager'
    object cache_key,
    np.ndarray c1,  # Remove buffer type declarations from function parameters
    np.ndarray c2,
    np.ndarray g_m,
    np.ndarray g_n,
    np.ndarray h_l,
    object two_part_l=None):  # Change to object
    """
    Cache and compute convolution of Fourier coefficients
    """
    # Check cache first
    cached_result = cache_manager.get("convolution", cache_key)
    if cached_result is not None:
        return cached_result
    
    # Declare C types for local variables
    cdef np.ndarray[COMPLEX_t, ndim=1] C_l
    
    # Calculate convolution
    C_l = fftconvolve(c1 * g_m, c2 * g_n)
    
    # Apply additional terms
    if two_part_l is not None:
        C_l = C_l * h_l * two_part_l
    else:
        C_l = C_l * h_l
    
    # Store in cache
    cache_manager.set(C_l, "convolution", cache_key)
    
    return C_l


cpdef np.ndarray compute_fourier_coefficients(
    object cache_manager,  # Use 'object' instead of 'CacheManager'
    object cache_key,
    np.ndarray P_b,  # Remove buffer type declarations from parameters
    np.ndarray m,
    double N,  # Use double instead of float
    object c_window_func=None,
    object c_window_param=None,  # Change to object to allow None
    bint verbose=False):
    """
    Cache and compute Fourier coefficients for a given biased power spectrum
    """
    # Check cache first
    cached_result = cache_manager.get("fourier_coefficients", cache_key)
    if cached_result is not None:
        return cached_result
    
    # Declare typed variables for local use
    cdef np.ndarray[COMPLEX_t, ndim=1] c_m_positive, c_m_negative, c_m
    cdef int window_param
    
    # Compute positive frequency coefficients
    c_m_positive = rfft(P_b)
    c_m_positive[c_m_positive.shape[0]-1] = c_m_positive[c_m_positive.shape[0]-1] / 2.0
    
    # Compute negative frequency coefficients (complex conjugate)
    c_m_negative = np.conjugate(c_m_positive[1:])
    
    # Combine and normalize
    c_m = np.hstack((c_m_negative[::-1], c_m_positive)) / N
    
    # Apply window function if provided
    if c_window_func is not None and c_window_param is not None:
        if verbose:
            print('windowing the Fourier coefficients')
        window_param = int(c_window_param * N / 2.0)
        c_m = c_m * c_window_func(m, window_param)
    
    # Store in cache
    cache_manager.set(c_m, "fourier_coefficients", cache_key)
    
    return c_m