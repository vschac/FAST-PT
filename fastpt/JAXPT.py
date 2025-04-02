from fastpt.jax_utils import p_window, c_window, jax_k_extend
from fastpt.P_extend import k_extend
import jax.numpy as jnp
from jax import grad
from jax import jit
from time import time
import numpy as np
from jax import jacfwd, jacrev
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


    def J_k_scalar(self, P, X, nu, m, N, n_pad, id_pad, k_extrap, k_final, k_size, l, C_window=None, P_window=None, low_extrap=None, high_extrap=None, EK=None):
        from jax.numpy.fft import ifft, irfft
        
        P = jnp.asarray(P)
        m = jnp.asarray(m)
        id_pad = jnp.asarray(id_pad)
        k_extrap = jnp.asarray(k_extrap)
        k_final = jnp.asarray(k_final)
        l = jnp.asarray(l)
        
        pf, p, g_m, g_n, two_part_l, h_l = X
        pf = jnp.asarray(pf)
        p = jnp.asarray(p)
        g_m = jnp.asarray(g_m)
        g_n = jnp.asarray(g_n)
        if two_part_l is not None:
            two_part_l = jnp.asarray(two_part_l)
        h_l = jnp.asarray(h_l)

        # Extrapolation handling would need JAX versions of these functions
        # if (low_extrap is not None):
        #     P = jextrap_P_low(P)  # This would need to be implemented in JAX
        
        # if (high_extrap is not None):
        #     P = jextrap_P_high(P) # This would need to be implemented in JAX
        
        P_b = P * k_extrap ** (-nu)
        
        if (n_pad > 0):
            P_b = jnp.pad(P_b, pad_width=(n_pad, n_pad), mode='constant', constant_values=0)
        
        c_m = fourier_coefficients(P_b, m, N, C_window)
        
        A_out = jnp.zeros((pf.shape[0], k_size))
        
        def process_single_row(i):
            # Convolution
            C_l = convolution(c_m, c_m, g_m[i], g_n[i], h_l[i], None if two_part_l is None else two_part_l[i])
            
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
        
        P_out = irfft(c_m_positive) * k_final ** nu * float(N)
        
        if (n_pad > 0):
            P_out = P_out[id_pad]
            A_out = A_out[:, id_pad]
        
        return P_out, A_out



    def J_k_tensor(self, P, X, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, C_window=None, P_window=None):
        P = jnp.asarray(P)
        id_pad = jnp.asarray(id_pad)
        k_extrap = jnp.asarray(k_extrap)
        k_final = jnp.asarray(k_final)
        l = jnp.asarray(l)
        
        pf, p, nu1, nu2, g_m, g_n, h_l = X
        pf = jnp.asarray(pf)
        p = jnp.asarray(p.astype(np.float64))
        nu1 = jnp.asarray(nu1.astype(np.float64))
        nu2 = jnp.asarray(nu2.astype(np.float64))
        g_m = jnp.asarray(g_m)
        g_n = jnp.asarray(g_n)
        h_l = jnp.asarray(h_l)

        # Extrapolation handling would need JAX versions of these functions
        # if (low_extrap is not None):
        #     P = jextrap_P_low(P)  # This would need to be implemented in JAX
        
        # if (high_extrap is not None):
        #     P = jextrap_P_high(P) # This would need to be implemented in JAX
        
        window = None
        if P_window is not None:
            window = p_window(k_extrap, P_window[0], P_window[1])

        A_out = jnp.zeros((pf.size, k_size))
        P_fin = jnp.zeros(k_size)

        l_midpoint = l.shape[0] // 2

        def process_element(i, carry):
            A_out, P_fin = carry
            
            nu1_i = nu1[i]
            nu2_i = nu2[i]

            P_b1 = P * k_extrap ** (-nu1_i)
            P_b2 = P * k_extrap ** (-nu2_i)
            
            if P_window is not None:
                P_b1 = P_b1 * window
                P_b2 = P_b2 * window
                
            if n_pad > 0:
                P_b1 = jnp.pad(P_b1, pad_width=(n_pad, n_pad), mode='constant', constant_values=0)
                P_b2 = jnp.pad(P_b2, pad_width=(n_pad, n_pad), mode='constant', constant_values=0)
                
            c_m = fourier_coefficients(P_b1, m, N, C_window)
            c_n = fourier_coefficients(P_b2, m, N, C_window)
            
            C_l = convolution(c_m, c_n, g_m[i,:], g_n[i,:], h_l[i,:])
            
            c_plus = C_l[l_midpoint:]
            c_minus = C_l[:l_midpoint]
            #C_l = jnp.hstack((c_plus[:-1], c_minus))
            C_l_combined = jnp.concatenate([c_plus[:-1], c_minus])

            A_k = jnp.fft.ifft(C_l_combined) * C_l_combined.size
            A_out_i = jnp.real(A_k[::2]) * pf[i] * k_final ** p[i]
            
            A_out = A_out.at[i].set(A_out_i)
            P_fin = P_fin + A_out_i
            
            return (A_out, P_fin)
        
        A_out, P_fin = jax.lax.fori_loop(0, pf.size, process_element, (A_out, P_fin))

        if n_pad > 0:
            P_fin = P_fin[id_pad]
            A_out = A_out[:, id_pad]
        
        return P_fin, A_out



@jit
def fourier_coefficients(P_b, m, N, C_window=None):
    from jax.numpy.fft import rfft

    c_m_positive = rfft(P_b)
    c_m_positive = c_m_positive.at[-1].set(c_m_positive[-1] / 2.0)
    c_m_negative = jnp.conjugate(c_m_positive[1:])
    c_m = jnp.hstack((c_m_negative[::-1], c_m_positive)) / jnp.float64(N)
    
    if C_window is not None:
        window_size = jnp.array(C_window * N / 2.0, dtype=int)
        c_m = c_m * c_window(m, window_size)
        
    return c_m

@jit
def convolution(c1, c2, g_m, g_n, h_l, two_part_l=None):
    from jax.scipy.signal import fftconvolve

    C_l = fftconvolve(c1 * g_m, c2 * g_n)

    if two_part_l is not None:
        C_l = C_l * h_l * two_part_l
    else:
        C_l = C_l * h_l

    return C_l