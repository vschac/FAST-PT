import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t, uint64_t
from libc.math cimport sin, log10, pi
from numpy import exp 
from math import exp as math_exp  # Alternative import from math
from scipy import fft as scipy_fft
from scipy.signal import fftconvolve
from numpy.fft import rfft as np_rfft
from scipy.fft import next_fast_len
import importlib.util


# Import scalar_stuff and tensor_stuff explicitly
from ..initialize_params import scalar_stuff, tensor_stuff


def p_window(np.ndarray[double, ndim=1] k, double log_k_left, double log_k_right):
    cdef int i, k_size = k.shape[0]
    cdef np.ndarray[double, ndim=1] log_k = np.empty(k_size, dtype=np.float64)
    cdef double max_val, min_val
    cdef np.ndarray[double, ndim=1] W = np.ones(k_size, dtype=np.float64)
    cdef int left_count = 0
    cdef double x_left
    cdef np.ndarray[double, ndim=1] left
    cdef int right_count = 0
    cdef double x_right
    cdef np.ndarray[double, ndim=1] right
    cdef int idx
    
    # Calculate log10 of k
    for i in range(k_size):
        log_k[i] = log10(k[i])
    
    max_val = log_k.max()
    min_val = log_k.min()
    
    # Calculate window boundaries
    log_k_left = min_val + log_k_left
    log_k_right = max_val - log_k_right
    
    # Process left boundary
    for i in range(k_size):
        if log_k[i] <= log_k_left:
            left_count += 1
    
    if left_count > 0:
        left = np.empty(left_count, dtype=np.float64)
        idx = 0
        for i in range(k_size):
            if log_k[i] <= log_k_left:
                left[idx] = log_k[i]
                idx += 1
        
        for i in range(left_count):
            x_left = (min_val - left[i]) / (min_val - left[left_count-1])
            W[i] = x_left - 1.0/(2.0*pi)*sin(2.0*pi*x_left)
    
    # Process right boundary
    for i in range(k_size):
        if log_k[i] >= log_k_right:
            right_count += 1
    
    if right_count > 0:
        right = np.empty(right_count, dtype=np.float64)
        idx = 0
        for i in range(k_size):
            if log_k[i] >= log_k_right:
                right[idx] = log_k[i]
                idx += 1
        
        for i in range(k_size - right_count, k_size):
            x_right = (right[i - (k_size - right_count)] - right[right_count-1]) / (right[0] - max_val)
            W[i] = x_right - 1.0/(2.0*pi)*sin(2.0*pi*x_right)
    
    return W


def c_window(np.ndarray n, int n_cut):
    """Window function for smoothing in coefficient space.
    
    Creates a smooth tapering at the edges of the coefficient array.
    Always returns an array with exactly the same shape as the input.
    """
    cdef int n_size = n.shape[0]
    cdef np.ndarray[double, ndim=1] W = np.ones(n_size, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] n_array
    cdef int i
    cdef double n_right, n_left, theta_left, theta_right
    
    # Convert input array to float64 if needed
    if n.dtype != np.float64:
        n_array = n.astype(np.float64)
    else:
        n_array = n
    
    # Calculate boundary positions
    n_right = n_array[n_size-1] - n_cut
    n_left = n_array[0] + n_cut
    
    # Process left and right boundaries
    for i in range(n_size):
        if n_array[i] < n_left:
            theta_left = (n_array[i] - n_array[0]) / float(n_left - n_array[0] - 1)
            W[i] = theta_left - 1.0/(2.0*pi)*sin(2.0*pi*theta_left)
        elif n_array[i] > n_right:
            theta_right = (n_array[n_size-1] - n_array[i]) / float(n_array[n_size-1] - n_right - 1)
            W[i] = theta_right - 1.0/(2.0*pi)*sin(2.0*pi*theta_right)
    
    return W

ctypedef np.complex128_t COMPLEX_t
ctypedef np.float64_t FLOAT_t
ctypedef np.int64_t INT_t

# Need to declare numpy is being used
np.import_array()

# Pre-allocate global workspace arrays for reuse
cdef:
    # FFT convolution workspaces
    np.ndarray _c1g_workspace = None
    np.ndarray _c2g_workspace = None
    np.ndarray _conv_result_workspace = None
    
    # Fourier coefficient workspaces
    np.ndarray _fft_input_workspace = None
    np.ndarray _fft_pos_workspace = None
    np.ndarray _fft_neg_workspace = None
    np.ndarray _fft_combined_workspace = None
    
    # J_k workspaces
    np.ndarray _A_out_workspace = None
    np.ndarray _B_out_workspace = None
    
    # Track last shapes to detect when we need to resize
    tuple _last_conv_shapes = None
    tuple _last_fft_shape = None
    tuple _last_jk_shape = None


cpdef object apply_extrapolation_cy(
    object arr_or_arrays,
    bint extrap,
    object k_original,
    object k_extrap,
    object EK=None):
    """
    Optimized Cython implementation for applying extrapolation to arrays.
    
    Parameters
    ----------
    arr_or_arrays : ndarray or sequence of ndarrays
        The array(s) to extrapolate
    extrap : bool
        Whether to apply extrapolation
    k_original : ndarray
        Original k values
    k_extrap : ndarray
        Extended k values
    EK : object, optional
        Extrapolation kernel object with PK_original method
    """
    if not extrap:
        return arr_or_arrays
    
    cdef list results
    
    # Handle multiple arrays case
    if isinstance(arr_or_arrays, (list, tuple)):
        results = []
        for arr in arr_or_arrays:
            # Apply PK_original to map from extended k-space back to original k-space
            if EK is not None:
                results.append(EK.PK_original(arr)[1])
            else:
                # Fallback interpolation if no EK object is provided
                results.append(_interpolate_to_original_grid(arr, k_extrap, k_original))
        return results
    
    # Single array case
    if EK is not None:
        return EK.PK_original(arr_or_arrays)[1]
    else:
        return _interpolate_to_original_grid(arr_or_arrays, k_extrap, k_original)

cdef object _interpolate_to_original_grid(object arr, object k_extrap, object k_original):
    """
    Helper function to interpolate from extended grid to original grid
    when EK object is not available.
    """
    from scipy.interpolate import interp1d
    interp_func = interp1d(k_extrap, arr, bounds_error=False, fill_value="extrapolate")
    return interp_func(k_original)


cdef int64_t _hash_arrays_internal(object arrays):
    """
    Internal helper function for hash_arrays_cy that handles the recursion.
    """
    cdef Py_ssize_t i
    cdef uint64_t hash_key_hash = 0
    cdef uint64_t item_hash
    
    if arrays is None:
        return hash(None)
        
    if arrays is None:
        return hash(None) & 0x7FFFFFFF
        
    if isinstance(arrays, (tuple, list)):
        for i in range(len(arrays)):
            item = arrays[i]
            if isinstance(item, np.ndarray):
                item_hash = abs(hash(item.tobytes())) & 0x7FFFFFFF
            elif isinstance(item, (tuple, list)):
                item_hash = _hash_arrays_internal(item) & 0x7FFFFFFF
            else:
                item_hash = abs(hash(item)) & 0x7FFFFFFF
            hash_key_hash = hash_key_hash ^ ((item_hash + 0x9e3779b9) + (hash_key_hash << 6) + (hash_key_hash >> 2))
            hash_key_hash = hash_key_hash & 0x7FFFFFFF
        return <int64_t>hash_key_hash

    if isinstance(arrays, np.ndarray):
        return abs(hash(arrays.tobytes())) & 0x7FFFFFFF
    return abs(hash(arrays)) & 0x7FFFFFFF


cpdef int64_t hash_arrays_cy(object arrays):
    """
    Helper function to create a hash from multiple numpy arrays or scalars.
    Cython implementation of FASTPT._hash_arrays for faster hash computation.
    """
    return _hash_arrays_internal(arrays) & 0x7FFFFFFF


cdef int64_t _create_hash_key_internal(list hash_list):
    """
    Internal function to combine hashes into a single key.
    """
    cdef uint64_t hash_key = 0
    cdef uint64_t h
    cdef Py_ssize_t i
    
    for i in range(len(hash_list)):
        if hash_list[i] is not None:
            h = abs(hash(hash_list[i])) & 0x7FFFFFFF
            hash_key = hash_key ^ ((h + 0x9e3779b9) + (hash_key << 6) + (hash_key >> 2))
            hash_key = hash_key & 0x7FFFFFFF
    
    return <int64_t>hash_key

cpdef int64_t create_hash_key_cy(str term, str X, object P, object P_window, object C_window):
    """
    Create a hash key from the term and input parameters.
    Cython implementation of FASTPT._create_hash_key for faster hash computation.
    """
    cdef int64_t P_hash = hash_arrays_cy(P)
    cdef int64_t P_win_hash = hash_arrays_cy(P_window)
    cdef int64_t X_hash
    cdef int64_t term_hash
    cdef list hash_list
    
    # Get term hash
    term_hash = hash(term)
    X_hash = hash(X)
    
    # Create hash list
    hash_list = [term_hash, X_hash, P_hash, P_win_hash, hash(C_window)]
    
    # Use internal function to combine hashes
    return _create_hash_key_internal(hash_list)


cpdef object compute_term_cy(
    object cache_manager,
    str term,
    str X_name,
    object X,
    object m,
    object operation,
    np.ndarray P,
    object P_window=None,
    object C_window=None,
    object k_original=None,
    object k_extrap=None,
    bint extrap=False,
    object EK=None,
    bint verbose=False):
    """
    Optimized Cython implementation of compute_term for FASTPT.
    
    This function computes a Fast-PT term with caching support for a single X parameter.
    Multiple X parameters should be handled at the Python level by the caller.
    """
    cdef long cache_key = create_hash_key_cy(term, X_name, P, P_window, C_window)
    cdef object result = cache_manager.get(term, cache_key)
    cdef object res, final_result
    
    # Return cached result if available
    if result is not None:
        return result
    
    # Process a single X parameter
    res, _ = J_k_tensor_cy(
        cache_manager, 
        X_name,
        k_extrap,
        P,
        X,
        m,
        k_final=k_extrap,
        eta_m=None,
        l=None,
        tau_l=None,
        n_pad=0,
        id_pad=None,
        P_window=P_window,
        C_window=C_window,
        extrap=extrap,
        EK=EK,
        verbose=verbose
    )
    
    # Apply extrapolation if necessary
    if extrap and k_original is not None:
        result = apply_extrapolation_cy(res, True, k_original, k_extrap, EK)
    else:
        result = res
    
    # Apply operation if provided
    if operation is not None:
        final_result = operation(result)
        cache_manager.set(final_result, term, cache_key)
        return final_result
    
    # Cache and return result
    cache_manager.set(result, term, cache_key)
    return result




cdef np.ndarray[COMPLEX_t, ndim=1] optimized_fft_convolution(
    np.ndarray[COMPLEX_t, ndim=1] a,
    np.ndarray[COMPLEX_t, ndim=1] b):
    """
    Memory-efficient FFT convolution implementation for complex inputs.
    Uses scipy.fft.fft for complex arrays rather than rfft which is for real inputs.
    """
    cdef:
        int n_a = a.shape[0]
        int n_b = b.shape[0]
        int n_out = n_a + n_b - 1
        int n_fft = next_fast_len(n_out)  # Optimal FFT size for performance
        np.ndarray[COMPLEX_t, ndim=1] result
    
    # Use fft for complex inputs, not rfft which requires real inputs
    a_fft = scipy_fft.fft(a, n=n_fft)
    b_fft = scipy_fft.fft(b, n=n_fft)
    
    # Multiply in-place to avoid temporary array
    a_fft *= b_fft
    
    # Inverse FFT (truncate to correct length)
    result = scipy_fft.ifft(a_fft, n=n_fft)[:n_out]
    
    return result


cpdef np.ndarray compute_convolution(
    object cache_manager,
    np.ndarray c1,
    np.ndarray c2,
    np.ndarray g_m,
    np.ndarray g_n,
    np.ndarray h_l,
    object two_part_l=None):
    """
    Cache and compute convolution of Fourier coefficients with array reuse.
    
    This function optimizes memory usage by:
    1. Reusing pre-allocated workspaces
    2. Using scipy's optimized FFT implementation
    3. Performing operations in-place wherever possible
    4. Utilizing Cython typing for performance
    """
    
    # Create cache key using hash_arrays_cy like the Python version does
    cdef object c1_hash = hash_arrays_cy(c1)
    cdef object c2_hash = hash_arrays_cy(c2)
    cdef object g_m_hash = hash_arrays_cy(g_m)
    cdef object g_n_hash = hash_arrays_cy(g_n)
    cdef object h_l_hash = hash_arrays_cy(h_l)
    cdef object two_part_l_hash = hash_arrays_cy(two_part_l)
    
    # Combine hashes to form a single cache key
    cdef list hash_list = [c1_hash, c2_hash, g_m_hash, g_n_hash, h_l_hash, two_part_l_hash]
    cdef long hash_key = _create_hash_key_internal(hash_list)
    
    cdef object cached_result = cache_manager.get("convolution", hash_key)
    if cached_result is not None:
        return cached_result
    
    # Declare C types for local variables
    cdef:
        int c1_len = len(c1)
        int c2_len = len(c2)
        int output_len = c1_len + c2_len - 1
        tuple expected_shapes = (c1_len, c2_len, output_len)
        np.ndarray[COMPLEX_t, ndim=1] C_l
    
    # Initialize or resize workspaces if needed
    global _c1g_workspace, _c2g_workspace, _conv_result_workspace, _last_conv_shapes
    if (_c1g_workspace is None or _c2g_workspace is None or _conv_result_workspace is None or 
            _last_conv_shapes != expected_shapes):
        _c1g_workspace = np.zeros(c1_len, dtype=np.complex128)
        _c2g_workspace = np.zeros(c2_len, dtype=np.complex128)
        _conv_result_workspace = np.zeros(output_len, dtype=np.complex128)
        _last_conv_shapes = expected_shapes
    
    # Prepare input arrays using workspaces (in-place operations)
    _c1g_workspace[:] = c1 
    _c1g_workspace *= g_m  # In-place multiplication
    
    _c2g_workspace[:] = c2
    _c2g_workspace *= g_n  # In-place multiplication
    
    # Use our optimized FFT convolution
    _conv_result_workspace[:] = optimized_fft_convolution(_c1g_workspace, _c2g_workspace)
    
    # Apply additional terms in-place
    if two_part_l is not None:
        _conv_result_workspace *= h_l * two_part_l
    else:
        _conv_result_workspace *= h_l
    
    # Create a copy for caching to prevent mutation of cached values
    C_l = np.array(_conv_result_workspace, copy=True)
    
    # Store in cache
    cache_manager.set(C_l, "convolution", hash_key)
    
    return C_l


cpdef np.ndarray compute_fourier_coefficients(
    object cache_manager,
    np.ndarray P_b,
    np.ndarray m,
    double N,
    object c_window_func=None,
    object c_window_param=None,
    bint verbose=False):
    """
    Cache and compute Fourier coefficients for a given biased power spectrum
    with array reuse.
    
    This version minimizes memory allocations and uses scipy's FFT directly.
    """
    cdef long cache_key = create_hash_key_cy("fourier_coefficients", None, P_b, None, c_window_param)
    cdef object cached_result = cache_manager.get("fourier_coefficients", cache_key)
    if cached_result is not None:
        return cached_result
    
    # Declare C types for local variables
    cdef:
        int input_size = P_b.shape[0]
        int rfft_output_size = input_size//2 + 1
        int final_size = 2 * rfft_output_size - 1  # Size of combined array
        bint need_resize = False
        np.ndarray window_values
        int workspace_size
        np.ndarray temp_m
        int midpoint, i
        
    # Initialize or resize workspaces if needed
    global _fft_input_workspace, _fft_pos_workspace, _fft_neg_workspace, _fft_combined_workspace, _last_fft_shape
    if (_fft_input_workspace is None or _fft_pos_workspace is None or 
            _fft_neg_workspace is None or _fft_combined_workspace is None or
            _last_fft_shape != (input_size, final_size)):
        need_resize = True
    
    if need_resize:
        # P_b should be real for this function
        if np.iscomplexobj(P_b):
            P_b = np.real(P_b)
        
        _fft_input_workspace = np.zeros(input_size, dtype=np.float64)
        _fft_pos_workspace = np.zeros(rfft_output_size, dtype=np.complex128)
        _fft_neg_workspace = np.zeros(rfft_output_size-1, dtype=np.complex128)
        _fft_combined_workspace = np.zeros(final_size, dtype=np.complex128)
        _last_fft_shape = (input_size, final_size)
    
    # Make sure P_b is real
    if np.iscomplexobj(P_b):
        _fft_input_workspace[:] = np.real(P_b)
    else:
        _fft_input_workspace[:] = P_b
    
    # For rfft, input must be real
    temp_fft = scipy_fft.rfft(_fft_input_workspace)
    _fft_pos_workspace[:] = temp_fft
    
    # Adjust last coefficient in-place
    _fft_pos_workspace[rfft_output_size-1] = _fft_pos_workspace[rfft_output_size-1] / 2.0
    
    # Compute negative frequency coefficients in-place
    _fft_neg_workspace[:] = np.conjugate(_fft_pos_workspace[1:])
    
    # Combine coefficients in-place (avoid new array allocations)
    _fft_combined_workspace[:rfft_output_size-1] = _fft_neg_workspace[::-1]  # Reverse for correct order
    _fft_combined_workspace[rfft_output_size-1:] = _fft_pos_workspace
    
    # Normalize in-place
    _fft_combined_workspace /= N
    
    # Apply window function if provided
    if c_window_func is not None and c_window_param is not None:
        if verbose:
            print('windowing the Fourier coefficients')
        window_param = int(c_window_param * N / 2.0)
        
        # Create a numpy array of integers directly
        workspace_size = _fft_combined_workspace.shape[0]
        temp_m = np.zeros(workspace_size, dtype=np.float64)  # Use float64 to match expected type
        midpoint = workspace_size // 2
        
        # Fill the array with integers centered around zero
        for i in range(workspace_size):
            temp_m[i] = i - midpoint
        
        # Apply window function and ensure we get back the right shape
        try:
            window_values = c_window_func(temp_m, window_param)
            
            # Safety check - if window size doesn't match workspace size, create a new one
            if window_values.shape[0] != workspace_size:
                print(f"Warning: Window size {window_values.shape[0]} doesn't match workspace size {workspace_size}.")
                window_values = np.ones(workspace_size, dtype=np.float64)
                
                # Try to copy values from the original window where possible
                copy_size = min(window_values.shape[0], workspace_size)
                window_values[:copy_size] = window_values[:copy_size]
        
            _fft_combined_workspace *= window_values
        except Exception as e:
            print(f"Error applying window function: {e}")
            # Continue without windowing if there's an error
    
    # Create a copy for caching to prevent mutation of cached values
    cdef np.ndarray[COMPLEX_t, ndim=1] c_m = np.array(_fft_combined_workspace, copy=True)
    
    # Store in cache
    cache_manager.set(c_m, "fourier_coefficients", cache_key)
    
    return c_m


cpdef tuple J_k_scalar_cy(
    object cache_manager,
    str X_name,
    np.ndarray k,
    np.ndarray P,
    object X,
    double nu,
    double Taylor_order,
    np.ndarray m,
    np.ndarray eta_m,
    np.ndarray l,
    np.ndarray tau_l,
    object P_window=None,
    object C_window=None,
    bint extrap=False,
    object EK=None,
    bint verbose=False):
    """
    Cythonized implementation of J_k_scalar for improved performance and memory efficiency.
    
    This function computes convolution integrals using FFT techniques with optimized
    memory usage through workspace reuse. It also handles extrapolation internally.
    """
    cdef long cache_key = create_hash_key_cy("J_k_scalar", X_name, P, P_window, C_window)
    cdef object cached_result = cache_manager.get("J_k_scalar", cache_key)
    if cached_result is not None:
        return cached_result

    # Declare C types for local variables
    cdef:
        int k_size = k.size
        np.ndarray param_mat = X[0]  # Extract param_mat from X
        int param_rows = param_mat.shape[0]
        int i
        np.ndarray A_out, B_out, P_b, c_m
        np.ndarray pf, p, g_m, g_n, two_part_l, h_l
        double bias_nu
    
    # Handle extrapolation for input power spectrum if needed
    if extrap and EK is not None:
        # Apply low and high extrapolation directly
        if hasattr(EK, 'extrap_P_low'):
            P = EK.extrap_P_low(P)
        if hasattr(EK, 'extrap_P_high'):
            P = EK.extrap_P_high(P)
    
    # Initialize output arrays or reuse existing workspaces
    global _A_out_workspace, _B_out_workspace, _last_jk_shape
    if (_A_out_workspace is None or _B_out_workspace is None or 
            _last_jk_shape != (k_size, param_rows)):
        _A_out_workspace = np.zeros((k_size, param_rows), dtype=np.complex128)
        _B_out_workspace = np.zeros(k_size, dtype=np.float64)
        _last_jk_shape = (k_size, param_rows)
    else:
        _A_out_workspace.fill(0)
        _B_out_workspace.fill(0)
    
    A_out = _A_out_workspace
    B_out = _B_out_workspace
    
    # Apply window function to power spectrum if provided
    if P_window is not None:
        P_b = P * p_window(k, k[-1], 1.)
    else:
        P_b = P
    
    # Apply power bias for Fourier coefficients
    bias_nu = -nu - Taylor_order/2.
    P_modified = P_b * k**(bias_nu)
    
    # Compute Fourier coefficients
    c_m = compute_fourier_coefficients(
        cache_manager,
        P_modified,
        m,
        k.size,
        c_window_func=c_window if C_window is not None else None,
        c_window_param=C_window,
        verbose=verbose
    )
    
    # Either unpack X or call scalar_stuff based on what's available
    if len(X) >= 6:
        # If X contains all the needed parameters, unpack them
        pf, p, g_m, g_n, two_part_l, h_l = X
    else:
        # Otherwise, compute them from param_mat
        pf, p, g_m, g_n, two_part_l, h_l = scalar_stuff(param_mat, nu, k.size, m, eta_m, l, tau_l, cache_manager)
    
    # Compute convolution term for each row in param_mat
    for i in range(param_rows):
        # Compute the convolution
        C_l = compute_convolution(
            cache_manager,
            c_m,
            c_m,
            g_m[i,:],
            g_n[i,:],
            h_l[i,:],
            two_part_l[i,:]
        )
        
        # Apply results to output matrices
        A_out[:,i] = k**(p[i]) * C_l[0:k.size] * pf[i]
    
    # Sum A_out over rows to get B_out
    for i in range(param_rows):
        B_out += 2. * np.real(A_out[:,i])
    
    # Apply post-processing extrapolation if needed (map back to original k-grid)
    if extrap:
        result = apply_extrapolation_cy((B_out, A_out), True, None, k, EK)
        B_out = result[0]
        A_out = result[1]
    
    # Cache the result
    cache_manager.set((B_out, A_out), "J_k_scalar", cache_key)
    
    # Return the results as a tuple
    return B_out, A_out


cpdef tuple J_k_tensor_cy(
    object cache_manager,
    str X_name,  # Changed from object hash_key to str X_name for consistency
    np.ndarray k,
    np.ndarray P, 
    object X,
    np.ndarray m,
    object k_final=None,
    np.ndarray eta_m=None,  # Added eta_m parameter
    np.ndarray l=None,
    np.ndarray tau_l=None,  # Added tau_l parameter
    Py_ssize_t n_pad=0,
    object id_pad=None,
    object P_window=None,
    object C_window=None,
    bint extrap=False,
    object EK=None,
    bint verbose=False):
    """
    Cythonized implementation of J_k_tensor for improved performance and memory efficiency.
    
    This function computes tensor convolution integrals using FFT techniques with optimized
    memory usage through workspace reuse. It also handles extrapolation internally.
    """
    # Create a cache key from parameters
    cdef long cache_key = create_hash_key_cy("J_k_tensor", X_name, P, P_window, C_window)
    cdef object cached_result = cache_manager.get("J_k_tensor", cache_key)
    if cached_result is not None:
        return cached_result
        
    from numpy.fft import ifft
    
    cdef:
        np.ndarray pf, p, nu1, nu2, g_m, g_n, h_l
        Py_ssize_t i
        int k_size = k.shape[0]
        np.ndarray P_extrap = P.copy()  # Create a copy to avoid modifying the input
        np.ndarray A_out, P_fin, P_b1, P_b2
        np.ndarray c_m, c_n, C_l, c_plus, c_minus, A_k
    
    # Extract parameters from X or compute them if needed
    if len(X) >= 7:
        # If X contains all needed parameters, unpack them
        pf, p, nu1, nu2, g_m, g_n, h_l = X
    else:
        # Otherwise compute them from X (this would need tensor_stuff function)
        # pf, p, nu1, nu2, g_m, g_n, h_l = tensor_stuff(X, k_size, m, eta_m, l, tau_l, cache_manager)
        raise ValueError("X must contain all tensor parameters: pf, p, nu1, nu2, g_m, g_n, h_l")
    
    # Apply extrapolation to input power spectrum if needed
    if extrap and EK is not None:
        # Apply low and high extrapolation directly
        if hasattr(EK, 'extrap_P_low'):
            P_extrap = EK.extrap_P_low(P_extrap)
        if hasattr(EK, 'extrap_P_high'):
            P_extrap = EK.extrap_P_high(P_extrap)
    
    # Initialize output arrays
    A_out = np.zeros((pf.size, k_size), dtype=np.float64)
    P_fin = np.zeros(k_size, dtype=np.float64)
    
    # Compute for each component in the tensor
    for i in range(pf.size):
        # Apply power biasing
        P_b1 = P_extrap * k**(-nu1[i])
        P_b2 = P_extrap * k**(-nu2[i])
        
        # Apply window if provided
        if P_window is not None:
            if verbose:
                print('windowing biased power spectrum')
            W = p_window(k, P_window[0], P_window[1])
            P_b1 = P_b1 * W
            P_b2 = P_b2 * W
        
        # Apply padding if requested
        if n_pad > 0:
            P_b1 = np.pad(P_b1, pad_width=(n_pad, n_pad), mode='constant', constant_values=0)
            P_b2 = np.pad(P_b2, pad_width=(n_pad, n_pad), mode='constant', constant_values=0)
        
        # Compute Fourier coefficients
        c_m = compute_fourier_coefficients(
            cache_manager,
            P_b1, 
            m,
            P_b1.size,
            c_window_func=c_window if C_window is not None else None,
            c_window_param=C_window,
            verbose=verbose
        )
        
        c_n = compute_fourier_coefficients(
            cache_manager,
            P_b2, 
            m,
            P_b2.size,
            c_window_func=c_window if C_window is not None else None,
            c_window_param=C_window,
            verbose=verbose
        )
        
        # Compute convolution
        C_l = compute_convolution(
            cache_manager,
            c_m,
            c_n,
            g_m[i,:],
            g_n[i,:],
            h_l[i,:],
            None  # No two_part_l for tensor case
        )
        
        # Set up for inverse FFT
        c_plus = C_l[l >= 0]
        c_minus = C_l[l < 0]
        C_l = np.hstack((c_plus[:-1], c_minus))
        
        # Inverse FFT to get back to real space
        A_k = ifft(C_l) * C_l.size  # Multiply by size to remove normalization
        
        # Store result for this component
        if k_final is not None:
            A_out[i, :] = np.real(A_k[::2]) * pf[i] * k_final**(p[i])
        else:
            A_out[i, :] = np.real(A_k[::2]) * pf[i] * k**(p[i])
        
        # Accumulate total result
        P_fin += A_out[i, :]
    
    # Remove padding if applied
    if n_pad > 0 and id_pad is not None:
        A_out = A_out[:, id_pad]
        P_fin = P_fin[id_pad]
    
    # Apply post-processing extrapolation if needed (map back to original k-grid)
    if extrap and EK is not None:
        result = apply_extrapolation_cy((P_fin, A_out), True, None, k, EK)
        P_fin = result[0]
        A_out = result[1]
    
    # Cache and return results
    cache_manager.set((P_fin, A_out), "J_k_tensor", cache_key)
    return P_fin, A_out


# For testing memory usage
def clear_workspaces():
    """
    Clear all pre-allocated workspaces to free memory.
    Call this function when you want to release memory.
    """
    global _c1g_workspace, _c2g_workspace, _conv_result_workspace
    global _fft_input_workspace, _fft_pos_workspace, _fft_neg_workspace, _fft_combined_workspace
    global _A_out_workspace, _B_out_workspace
    global _last_conv_shapes, _last_fft_shape, _last_jk_shape
    
    _c1g_workspace = None
    _c2g_workspace = None
    _conv_result_workspace = None
    _fft_input_workspace = None
    _fft_pos_workspace = None
    _fft_neg_workspace = None
    _fft_combined_workspace = None
    _A_out_workspace = None
    _B_out_workspace = None
    _last_conv_shapes = None
    _last_fft_shape = None
    _last_jk_shape = None
    
    # Force garbage collection
    import gc
    gc.collect()