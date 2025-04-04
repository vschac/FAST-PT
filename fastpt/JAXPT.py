from fastpt.jax_utils import p_window, c_window, jax_k_extend
from fastpt.P_extend import k_extend
import jax.numpy as jnp
from jax import grad
from jax import jit
from time import time
import numpy as np
from jax import jacfwd, jacrev, vjp
from jax import config
import jax
from fastpt import FASTPT as FPT
config.update("jax_enable_x64", True)
import functools

def process_x_term(X):
    """Process X term for JAX compatibility, preserving complex values"""
    processed_X = []
    
    for term in X:
        if isinstance(term, np.ndarray):
            # If it's an object dtype, convert to appropriate numeric type
            if term.dtype == np.dtype('O'):
                # Check if the array contains complex values
                try:
                    # Sample the first element to see if it's complex
                    sample = term.flat[0]
                    if isinstance(sample, complex) or (hasattr(sample, 'imag') and sample.imag != 0):
                        # Convert to complex
                        term = term.astype(np.complex128)
                        term = jnp.asarray(term)
                    else:
                        # Convert to float
                        term = term.astype(np.float64)
                        term = jnp.asarray(term)
                except (IndexError, TypeError):
                    # If sampling fails, try float64 as a fallback
                    try:
                        term = term.astype(np.float64)
                        term = jnp.asarray(term)
                    except:
                        # If that fails too, try complex
                        try:
                            term = term.astype(np.complex128)
                            term = jnp.asarray(term)
                        except:
                            print(f"Warning: Could not convert array of dtype {term.dtype}")
            else:
                # Regular numeric array, convert to JAX
                term = jnp.asarray(term)
        # Non-array types just pass through
        processed_X.append(term)
    
    # Return tuple to match original format
    return tuple(processed_X)

def jax_cached_property(method):
    prop_name = '_' + method.__name__

    @functools.wraps(method)
    def wrapper(self):
        if not hasattr(self, prop_name):
            result = method(self)
            # Process X terms for JAX compatibility
            if isinstance(result, tuple):
                converted = process_x_term(result)
            else:
                converted = result
            setattr(self, prop_name, converted)
        return getattr(self, prop_name)

    return property(wrapper)

class JAXPT: 
    def __init__(self, k, low_extrap=None, high_extrap=None, n_pad=None):
        
        if (k is None or len(k) == 0):
            raise ValueError('You must provide an input k array.')        

        #self.cache = CacheManager()

        self.X_registry = {} #Stores the names of X terms to be used as an efficient unique identifier in hash keys
        self.__k_original = k
        self.temp_fpt = FPT(k.copy(), low_extrap=low_extrap, high_extrap=high_extrap, n_pad=n_pad)
        self.extrap = False
        if (low_extrap is not None or high_extrap is not None):
            if (high_extrap < low_extrap):
                raise ValueError('high_extrap must be greater than low_extrap')
            self.EK = jax_k_extend(k, low_extrap, high_extrap)
            k = self.EK.extrap_k()
            self.extrap = True

        self.low_extrap = low_extrap
        self.high_extrap = high_extrap
        self.__k_extrap = k #K extrapolation not padded

        dk = np.diff(np.log(k))
        delta_L = (jnp.log(k[-1]) - jnp.log(k[0])) / (k.size - 1)
        dk_test = np.ones_like(dk) * delta_L

        log_sample_test = 'ERROR! FASTPT will not work if your in put (k,Pk) values are not sampled evenly in log space!'
        np.testing.assert_array_almost_equal(dk, dk_test, decimal=4, err_msg=log_sample_test, verbose=False)

        if (k.size % 2 != 0):
            raise ValueError('Input array must contain an even number of elements.')

        if n_pad is None:
            n_pad = int(0.5 * len(k))
        self.n_pad = n_pad
        if (n_pad > 0):
            if not isinstance(n_pad, int):
                n_pad = int(n_pad)
            self.n_pad = n_pad
            self.id_pad = np.arange(k.size) + n_pad
            d_logk = delta_L
            k_pad = np.log(k[0]) - np.arange(1, n_pad + 1) * d_logk
            k_pad = np.exp(k_pad)
            k_left = k_pad[::-1]

            k_pad = np.log(k[-1]) + np.arange(1, n_pad + 1) * d_logk
            k_right = np.exp(k_pad)
            k = np.hstack((k_left, k, k_right))
            n_pad_check = int(np.log(2) / delta_L) + 1
            if (n_pad < n_pad_check):
                print('*** Warning ***')
                print(f'You should consider increasing your zero padding to at least {n_pad_check}')
                print('to ensure that the minimum k_output is > 2k_min in the FASTPT universe.')
                print(f'k_min in the FASTPT universe is {k[0]} while k_min_input is {self.k_extrap[0]}')

        self.__k_final = k #log spaced k, with padding and extrap
        self.k_size = k.size
        # self.scalar_nu=-2
        self.N = k.size

        # define eta_m and eta_n=eta_m
        omega = 2 * jnp.pi / (float(self.N) * delta_L)
        self.m = np.arange(-self.N // 2, self.N // 2 + 1)
        self.eta_m = omega * self.m

        # define l and tau_l
        self.n_l = self.m.size + self.m.size - 1
        self.l = np.arange(-self.n_l // 2 + 1, self.n_l // 2 + 1)
        self.tau_l = omega * self.l

        #JIT Compile functions
        try:
            self.J_k_scalar = jit(self.J_k_scalar, static_argnames=["n_pad", "k_size", "EK"])
        except:
            print("J_k_scalar JIT compilation failed. Using default python implementation.")
        try:
            self._J_k_tensor_core = jit(self._J_k_tensor_core, static_argnames=["n_pad", "k_size", "EK"])
        except:
            print("J_k_tensor JIT compilation failed. Using default python implementation.")
        try:
            self.fourier_coefficients = jit(self.fourier_coefficients)
        except:
            print("fourier_coefficients JIT compilation failed. Using default python implementation.")
        try:
            self.convolution = jit(self.convolution)
        except:
            print("convolution JIT compilation failed. Using default python implementation.")
    

    @jax_cached_property
    def X_spt(self):
        return self.temp_fpt.X_spt
    @jax_cached_property
    def X_lpt(self):
        return self.temp_fpt.X_lpt  
    @jax_cached_property
    def X_sptG(self):
        return self.temp_fpt.X_sptG
    @jax_cached_property
    def X_cleft(self):
        return self.temp_fpt.X_cleft
    @jax_cached_property
    def X_IA_A(self):
        return self.temp_fpt.X_IA_A
    @jax_cached_property
    def X_IA_B(self):
        return self.temp_fpt.X_IA_B
    @jax_cached_property
    def X_IA_E(self):
        return self.temp_fpt.X_IA_E
    @jax_cached_property
    def X_IA_DEE(self):
        return self.temp_fpt.X_IA_DEE
    @jax_cached_property
    def X_IA_DBB(self):
        return self.temp_fpt.X_IA_DBB
    @jax_cached_property
    def X_IA_deltaE1(self):
        return self.temp_fpt.X_IA_deltaE1
    @jax_cached_property
    def X_IA_0E0E(self):
        return self.temp_fpt.X_IA_0E0E
    @jax_cached_property
    def X_IA_0B0B(self):
        return self.temp_fpt.X_IA_0B0B
    @jax_cached_property
    def X_IA_gb2_fe(self):
        return self.temp_fpt.X_IA_gb2_fe
    @jax_cached_property
    def X_IA_gb2_he(self):
        return self.temp_fpt.X_IA_gb2_he
    @jax_cached_property
    def X_IA_tij_feG2(self):
        return self.temp_fpt.X_IA_tij_feG2
    @jax_cached_property
    def X_IA_tij_heG2(self):
        return self.temp_fpt.X_IA_tij_heG2
    @jax_cached_property
    def X_IA_tij_F2F2(self):
        return self.temp_fpt.X_IA_tij_F2F2
    @jax_cached_property
    def X_IA_tij_G2G2(self):
        return self.temp_fpt.X_IA_tij_G2G2
    @jax_cached_property
    def X_IA_tij_F2G2(self):
        return self.temp_fpt.X_IA_tij_F2G2
    @jax_cached_property
    def X_IA_tij_F2G2reg(self):
        return self.temp_fpt.X_IA_tij_F2G2reg
    @jax_cached_property
    def X_IA_gb2_F2(self):
        return self.temp_fpt.X_IA_gb2_F2
    @jax_cached_property
    def X_IA_gb2_G2(self):
        return self.temp_fpt.X_IA_gb2_G2
    @jax_cached_property
    def X_IA_gb2_S2F2(self):
        return self.temp_fpt.X_IA_gb2_S2F2
    @jax_cached_property
    def X_IA_gb2_S2fe(self):
        return self.temp_fpt.X_IA_gb2_S2fe
    @jax_cached_property
    def X_IA_gb2_S2he(self):
        return self.temp_fpt.X_IA_gb2_S2he
    @jax_cached_property
    def X_IA_gb2_S2G2(self):
        return self.temp_fpt.X_IA_gb2_S2G2
    @jax_cached_property
    def X_OV(self):
        return self.temp_fpt.X_OV
    @jax_cached_property
    def X_kP1(self):
        return self.temp_fpt.X_kP1
    @jax_cached_property
    def X_kP2(self):
        return self.temp_fpt.X_kP2
    @jax_cached_property
    def X_kP3(self):
        return self.temp_fpt.X_kP3
    @jax_cached_property
    def X_RSDA(self):
        return self.temp_fpt.X_RSDA
    @jax_cached_property
    def X_RSDB(self):
        return self.temp_fpt.X_RSDB


        
    @property
    def k_original(self):
        return self.__k_original
    
    @property
    def k_extrap(self):
        return self.__k_extrap
    
    @property
    def k_final(self):
        return self.__k_final

    def _apply_extrapolation(self, *args):
        """ Applies extrapolation to multiple variables at once """
        if not self.extrap:
            return args if len(args) > 1 else args[0]
        return [self.EK.PK_original(var)[1] for var in args] if len(args) > 1 else self.EK.PK_original(args[0])[1]

    def compute_term(self, X, operation=None, P=None, P_window=None, C_window=None):        
        result, _ = self.J_k_tensor(P, X, self.k_extrap, self.k_final, self.k_size,
                                    self.n_pad, self.id_pad, self.l, self.m, self.N, P_window=P_window, C_window=C_window)
        result = self._apply_extrapolation(result)

        if operation:
            final_result = operation(result)
            return final_result
        return result



    def J_k_scalar(self, P, X, nu, m, N, n_pad, id_pad, k_extrap, k_final, k_size, l, C_window=None, P_window=None, low_extrap=None, high_extrap=None, EK=None):
        from jax.numpy.fft import ifft, irfft
        
        pf, p, g_m, g_n, two_part_l, h_l = X

        if (low_extrap is not None):
            P = EK.extrap_P_low(P)
        
        if (high_extrap is not None):
            P = EK.extrap_P_high(P)
        
        P_b = P * k_extrap ** (-nu)
        
        if (n_pad > 0):
            P_b = jnp.pad(P_b, pad_width=(n_pad, n_pad), mode='constant', constant_values=0)
        
        c_m = self.fourier_coefficients(P_b, m, N, C_window)
        
        A_out = jnp.zeros((pf.shape[0], k_size))
        
        def process_single_row(i):
            C_l = self.convolution(c_m, c_m, g_m[i], g_n[i], h_l[i], None if two_part_l is None else two_part_l[i])
            
            # Instead of boolean indexing, we'll use a different approach:
            # 1. Create arrays for positive and negative indices
            l_size = l.shape[0]
            l_midpoint = l_size // 2  # Assuming l is centered around 0
            
            # 2. Extract positive and negative parts using slicing
            # This assumes l is arranged from negative to positive
            c_plus = C_l[l_midpoint:]  # Positive part (including 0)
            c_minus = C_l[:l_midpoint]  # Negative part
            
            # 3. Combine them, dropping the last element of c_plus
            C_l_combined = jnp.concatenate([c_plus[:-1], c_minus])
            
            # 4. FFT operations
            A_k = ifft(C_l_combined) * C_l_combined.size
            
            # 5. Downsample and compute final value
            # For downsampling, we'll use strided slicing
            stride = max(1, A_k.shape[0] // k_size)
            
            # Get the real part and apply scaling
            return jnp.real(A_k[::stride][:k_size]) * pf[i] * k_final ** (-p[i] - 2)
        
        rows = jnp.arange(pf.shape[0])
        A_out = jax.vmap(process_single_row)(rows)
        
        m_midpoint = (m.shape[0] + 1) // 2  # Position of 0 in m
        c_m_positive = c_m[m_midpoint-1:]  # Select m >= 0
        
        P_out = irfft(c_m_positive) * k_final ** nu * N
        
        if (n_pad > 0):
            P_out = P_out[id_pad]
            A_out = A_out[:, id_pad]
        
        return P_out, A_out



    def J_k_tensor(self, P, X, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, C_window=None, P_window=None, low_extrap=None, high_extrap=None, EK=None):
        # Create window outside of JIT if needed
        window = None
        if P_window is not None:
            window = p_window(jnp.array(k_extrap), P_window[0], P_window[1])
            
        return self._J_k_tensor_core(P, X, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                                    C_window, window, low_extrap, high_extrap, EK)

    def _J_k_tensor_core(self, P, X, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, 
                        C_window=None, window=None, low_extrap=None, high_extrap=None, EK=None):
        
        pf, p, nu1, nu2, g_m, g_n, h_l = X

        if (low_extrap is not None):
            P = EK.extrap_P_low(P)
        
        if (high_extrap is not None):
            P = EK.extrap_P_high(P)
        
        l_midpoint = l.shape[0] // 2

        def process_single_index(i):
            nu1_i = nu1[i]
            nu2_i = nu2[i]
            pf_i = pf[i]
            p_i = p[i]
            g_m_i = g_m[i]
            g_n_i = g_n[i]
            h_l_i = h_l[i]
            
            P_b1 = P * k_extrap ** (-nu1_i)
            P_b2 = P * k_extrap ** (-nu2_i)
            
            if window is not None:
                P_b1 = P_b1 * window
                P_b2 = P_b2 * window
                
            if n_pad > 0:
                P_b1 = jnp.pad(P_b1, pad_width=(n_pad, n_pad), mode='constant', constant_values=0)
                P_b2 = jnp.pad(P_b2, pad_width=(n_pad, n_pad), mode='constant', constant_values=0)
                
            c_m = self.fourier_coefficients(P_b1, m, N, C_window)
            c_n = self.fourier_coefficients(P_b2, m, N, C_window)
            
            C_l = self.convolution(c_m, c_n, g_m_i, g_n_i, h_l_i)
            
            c_plus = C_l[l_midpoint:]
            c_minus = C_l[:l_midpoint]
            C_l_combined = jnp.concatenate([c_plus[:-1], c_minus])

            A_k = jnp.fft.ifft(C_l_combined) * C_l_combined.size
            return jnp.real(A_k[::2]) * pf_i * k_final ** p_i
        
        indices = jnp.arange(pf.size)
        A_out = jax.vmap(process_single_index)(indices)
        
        P_fin = jnp.sum(A_out, axis=0)
        
        if n_pad > 0:
            P_fin = P_fin[id_pad]
            A_out = A_out[:, id_pad]
        
        return P_fin, A_out



    def fourier_coefficients(self, P_b, m, N, C_window=None):
        from jax.numpy.fft import rfft

        c_m_positive = rfft(P_b)
        c_m_positive = c_m_positive.at[-1].set(c_m_positive[-1] / 2.0)
        c_m_negative = jnp.conjugate(c_m_positive[1:])
        c_m = jnp.hstack((c_m_negative[::-1], c_m_positive)) / jnp.float64(N)
        
        if C_window is not None:
            window_size = jnp.array(C_window * N / 2.0, dtype=int)
            c_m = c_m * c_window(m, window_size)
            
        return c_m


    def convolution(self, c1, c2, g_m, g_n, h_l, two_part_l=None):
        from jax.scipy.signal import fftconvolve

        C_l = fftconvolve(c1 * g_m, c2 * g_n)

        if two_part_l is not None:
            C_l = C_l * h_l * two_part_l
        else:
            C_l = C_l * h_l

        return C_l


if __name__ == "__main__":
    k = np.logspace(1e-4, 1, 1000)
    P = np.logspace(1, 2, 1000)
    jpt = JAXPT(k, low_extrap=-5, high_extrap=3)
    #jpt.J_k_scalar(P, jpt.X_spt, -2, jpt.m, jpt.N, jpt.n_pad, jpt.id_pad, jpt.k_extrap, jpt.k_final, jpt.k_size, jpt.l, C_window=0.75, low_extrap=-5, high_extrap=5, EK=jpt.EK)
    #jpt.J_k_tensor(P, jpt.X_IA_A, jpt.k_extrap, jpt.k_final, jpt.k_size, jpt.n_pad, jpt.id_pad, jpt.l, jpt.m, jpt.N, C_window=0.75, P_window=jnp.array([0.2, 0.2]), low_extrap=-5, high_extrap=5, EK=jpt.EK)
    def j_k_tensor_wrapper(P_input):
        result = jpt.J_k_tensor(P_input, jpt.X_IA_A, jpt.k_extrap, jpt.k_final,
                            jpt.k_size, jpt.n_pad, jpt.id_pad,
                            jpt.l, jpt.m, jpt.N, P_window=jnp.array([0.2, 0.2]),
                            C_window=0.75, low_extrap=-5, high_extrap=3, 
                            EK=jpt.EK)[0]
        
        # Use the existing method to get back original k range
        return jpt._apply_extrapolation(result)
    
    def simple_model(P_input, degree=1):
        # Start with a simple polynomial transformation
        P_mod = P_input**degree
        result = jpt.J_k_tensor(P_mod, jpt.X_IA_A, jpt.k_extrap, jpt.k_final,
                        jpt.k_size, jpt.n_pad, jpt.id_pad,
                        jpt.l, jpt.m, jpt.N, P_window=jnp.array([0.2, 0.2]),
                        C_window=0.75, low_extrap=-5, high_extrap=3, 
                        EK=jpt.EK)[0]
        return jpt._apply_extrapolation(result)

    output, vjp_fn = vjp(simple_model, P)
    v = jnp.ones_like(output)
    gradient, = vjp_fn(v)
    import matplotlib.pyplot as plt


    # Also compute a finite difference approximation
    delta = 1e-5
    fd_gradient = np.zeros_like(P)
    for i in range(len(P)):
        P_plus = P.copy()
        P_plus = np.array(P_plus, dtype=np.float64)  # Copy to avoid JAX tracer issues
        P_plus[i] += delta
        output_plus = j_k_tensor_wrapper(P_plus)
        
        # Approximate derivative
        fd_gradient[i] = (output_plus[i] - output[i]) / delta
    

    # Print diagnostics about the data
    print(f"Output range: [{np.min(output)}, {np.max(output)}], shape: {output.shape}")
    print(f"Any zeros in output: {np.any(output == 0)}")
    print(f"Any negative values in output: {np.any(output < 0)}")
    print(f"Any NaNs in output: {np.any(np.isnan(output))}")

    print(f"Gradient range: [{np.min(gradient)}, {np.max(gradient)}], shape: {gradient.shape}")
    print(f"FD gradient range: [{np.min(fd_gradient)}, {np.max(fd_gradient)}], shape: {fd_gradient.shape}")

    import matplotlib.pyplot as plt
    
    # Plot on log scale to better see patterns
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Original function
    ax1.plot(k, output, label='Output')
    ax1.set_ylabel('Output Value')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)
    
    # Automatic gradient
    ax2.plot(k, np.abs(gradient), label='|Gradient|', color='orange')
    ax2.set_ylabel('Gradient Magnitude')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    
    # Finite difference gradient for comparison
    ax3.plot(k, np.abs(fd_gradient), label='|Finite Diff|', color='green')
    ax3.set_xlabel('k')
    ax3.set_ylabel('FD Gradient Magnitude')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()