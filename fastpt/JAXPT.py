import jax.numpy as jnp
from jax import grad
from jax import jit
from time import time
import numpy as np
from jax import jacfwd, jacrev
from jax import config
config.update("jax_enable_x64", True)

class FastPT: #Simple FastPT class used for testing comparisons between numpy and jax
    def __init__(self, k, n_pad=None):
        self.k = k
        delta_L = (np.log(k[-1]) - np.log(k[0])) / (k.size - 1)
        if n_pad is None:
            n_pad = int(0.5 * len(k))
        self.n_pad = n_pad
        if (n_pad > 0):
            # Make sure n_pad is an integer
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

        self.N = k.size
        self.m = np.arange(-self.N // 2, self.N // 2 + 1)
        

    def fourier_coefficients(self, P_b, C_window=None):
        from numpy.fft import rfft

        c_m_positive = rfft(P_b)
        c_m_positive[-1] = c_m_positive[-1] / 2.
        c_m_negative = jnp.conjugate(c_m_positive[1:])
        c_m = jnp.hstack((c_m_negative[::-1], c_m_positive)) / float(self.N)

        if C_window is not None:
            c_m = c_m * c_window(self.m, int(C_window * self.N / 2.))
        return c_m
    
    def convolution(self, c1, c2, g_m, g_n, h_l, two_part_l=None):
        from scipy.signal import fftconvolve

        C_l = fftconvolve(c1 * g_m, c2 * g_n)

        if two_part_l is not None:
            C_l = C_l * h_l * two_part_l
        else:
            C_l = C_l * h_l

        return C_l
    


def jfourier_coefficients(P_b, m, N, C_window=None):
    from jax.numpy.fft import rfft

    c_m_positive = rfft(P_b)
    c_m_positive = c_m_positive.at[-1].set(c_m_positive[-1] / 2.0)
    c_m_negative = jnp.conjugate(c_m_positive[1:])
    c_m = jnp.hstack((c_m_negative[::-1], c_m_positive)) / jnp.float64(N)
    
    if C_window is not None:
        window_size = jnp.array(C_window * N / 2.0, dtype=int)
        c_m = c_m * jc_window(m, window_size)
        
    return c_m

def jconvolution(c1, c2, g_m, g_n, h_l, two_part_l=None):
    from jax.scipy.signal import fftconvolve

    C_l = fftconvolve(c1 * g_m, c2 * g_n)

    if two_part_l is not None:
        C_l = C_l * h_l * two_part_l
    else:
        C_l = C_l * h_l

    return C_l


def c_window(n,n_cut):
    import numpy as np
    from numpy import pi, sin

    n_right = n[-1] - n_cut
    n_left = n[0]+ n_cut 

    n_r=n[ n[:]  > n_right ] 
    n_l=n[ n[:]  <  n_left ] 

    theta_right=(n[-1]-n_r)/float(n[-1]-n_right-1) 
    theta_left=(n_l - n[0])/float(n_left-n[0]-1) 

    W=np.ones(n.size)
    W[n[:] > n_right]= theta_right - 1/(2*pi)*sin(2*pi*theta_right)
    W[n[:] < n_left]= theta_left - 1/(2*pi)*sin(2*pi*theta_left)

    return W

def jc_window(n, n_cut):
    from jax.numpy import pi, sin
    n_right = n[-1] - n_cut
    n_left = n[0] + n_cut
    
    # Create base window of ones
    W = jnp.ones_like(n)
    
    # Right side windowing
    right_mask = n > n_right
    theta_right = (n[-1] - n) / jnp.array(n[-1] - n_right - 1, dtype=float)
    right_window = theta_right - 1/(2*pi)*sin(2*pi*theta_right)
    
    # Left side windowing
    left_mask = n < n_left
    theta_left = (n - n[0]) / jnp.array(n_left - n[0] - 1, dtype=float)
    left_window = theta_left - 1/(2*pi)*sin(2*pi*theta_left)
    
    # Apply windows conditionally
    W = jnp.where(right_mask, right_window, W)
    W = jnp.where(left_mask, left_window, W)
    
    return W


def p_window(k,log_k_left,log_k_right):
	from numpy import sin, pi
	log_k=np.log10(k)
	
	max=np.max(log_k)
	min=np.min(log_k)
	
	log_k_left=min+log_k_left
	log_k_right=max-log_k_right
		
	left=log_k[log_k <= log_k_left]
	right=log_k[log_k >= log_k_right]
	x_right=(right- right[right.size-1])/(right[0]-max)
	x_left=(min-left)/(min-left[left.size-1])
	
	W=np.ones(k.size)
	W[log_k <= log_k_left] = (x_left - 1/(2*pi)*sin(2*pi*x_left))
	W[log_k  >= log_k_right] = (x_right-  1/(2*pi)*sin(2*pi*x_right))
	
	return W 

def jp_window(k, log_k_left, log_k_right):
    from jax.numpy import sin, pi
    log_k = jnp.log10(k)
    
    max_log_k = jnp.max(log_k)
    min_log_k = jnp.min(log_k)
    
    log_k_left = min_log_k + log_k_left
    log_k_right = max_log_k - log_k_right
    
    # Masks
    mask_left = log_k <= log_k_left
    mask_right = log_k >= log_k_right
    
    # Extract elements that satisfy the masks
    left = log_k[mask_left]
    right = log_k[mask_right]

    # Compute x_left and x_right with proper indexing
    if left.size > 0:
        x_left = (min_log_k - left) / (min_log_k - left[-1])
        W_left = x_left - (1 / (2 * jnp.pi)) * jnp.sin(2 * jnp.pi * x_left)
    else:
        W_left = jnp.array([])  # Avoid shape mismatch

    if right.size > 0:
        x_right = (right - right[-1]) / (right[0] - max_log_k)
        W_right = x_right - (1 / (2 * jnp.pi)) * jnp.sin(2 * jnp.pi * x_right)
    else:
        W_right = jnp.array([])  # Avoid shape mismatch

    # Initialize W
    W = jnp.ones_like(k)

    # Apply updates only if there are valid elements
    if W_left.size > 0:
        W = W.at[mask_left].set(W_left)
    if W_right.size > 0:
        W = W.at[mask_right].set(W_right)

    return W



def measure_differences(num1, num2):
    diff = np.abs(num1 - num2)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    relative_diff = diff / (np.abs(num1) + 1e-10)  # Add small epsilon to avoid division by zero
    max_rel_diff = np.max(relative_diff)
    avg_rel_diff = np.mean(relative_diff)
    print(f"Maximum absolute difference: {max_diff:.8e}")
    print(f"Average absolute difference: {avg_diff:.8e}")
    print(f"Maximum relative difference: {max_rel_diff:.8e}")
    print(f"Average relative difference: {avg_rel_diff:.8e}")

if __name__ == "__main__":
    from fastpt import FASTPT, FPTHandler
    k = np.logspace(1e-4, 1, 1000)
    fpt = FastPT(k)
    FPT = FASTPT(k)
    handler = FPTHandler(FPT)
    P = handler.generate_power_spectra()
    X_IA_A = FPT.X_IA_A
    X_spt = FPT.X_spt
    C_window = 0.75
    P_window = np.array([0.2, 0.2])

    # # RFFT and FFT Convolve
    print("RFFT and FFT Convolve")
    from numpy.fft import rfft
    old = rfft(P)
    from jax.numpy.fft import rfft as jax_rfft
    new = jax_rfft(P)
    print(np.allclose(old, new))
    try:
        jacfwd(jax_rfft)(P)
        print("jacfwd passed")
    except:
        print("jacfwd failed")

    print("=====================================================")

    from scipy.signal import fftconvolve
    old2 = fftconvolve(np.logspace(0, 1, 10), np.logspace(1, 2, 10), )
    from jax.scipy.signal import fftconvolve as jax_fftconvolve
    new2 = jax_fftconvolve(jnp.logspace(0, 1, 10), jnp.logspace(1, 2, 10))
    print(np.allclose(old2, new2))
    try:
        jacfwd(jax_fftconvolve)(jnp.logspace(0, 1, 10), jnp.logspace(1, 2, 10))
        print("jacfwd passed")
    except:
        print("jacfwd failed")


    print("\n", "="*100, "\n")


    # # P_window and C_window
    print("P_window and C_window")
    p1 = p_window(k, P_window[0], P_window[1])
    p2 = jp_window(k, P_window[0], P_window[1])
    print(np.allclose(p1, p2))
    try:
        jacfwd(jp_window)(k, P_window[0], P_window[1])
        print("jacfwd passed")
    except:
        print("jacfwd failed")

    print("=====================================================")
    m_float = jnp.asarray(FPT.m, dtype=jnp.float64)
    c1 = c_window(FPT.m, C_window)
    c2 = jc_window(m_float, jnp.float64(C_window))
    print(np.allclose(c1, c2))
    try:
        jacfwd(jc_window)(m_float, jnp.float64(C_window))
        print("jacfwd passed")
    except:
        print("jacfwd failed")
        


    print("\n", "="*100, "\n")


    # Fourier Coefficients
    print("Fourier Coefficients")
    pf, p, nu1, nu2, g_m, g_n, h_l = X_IA_A
    P_b1 = P * FPT.k_extrap ** (-nu1[1])
    W = p_window(FPT.k_extrap, P_window[0], P_window[1])
    P_b1 = P_b1 * W
    P_b1 = np.pad(P_b1, pad_width=(FPT.n_pad, FPT.n_pad), mode='constant', constant_values=0)
    c_m = fpt.fourier_coefficients(P_b1, C_window)
    jc_m = jfourier_coefficients(P_b1, FPT.m, FPT.N, C_window)
    print(np.allclose(c_m, jc_m))
    try:
        jacfwd(jfourier_coefficients)(P_b1, FPT.m, FPT.N, C_window)
        print("jacfwd passed")
    except:
        print("jacfwd failed")    


    # Convolution
    print("Convolution")
    # #Tensor case
    print("Tensor case")
    pf, p, nu1, nu2, g_m, g_n, h_l = X_IA_A
    P_b1 = P * FPT.k_extrap ** (-nu1[1])
    P_b2 = P * FPT.k_extrap ** (-nu2[1])
    W = p_window(FPT.k_extrap, P_window[0], P_window[1])
    P_b1 = P_b1 * W
    P_b2 = P_b2 * W
    P_b1 = np.pad(P_b1, pad_width=(FPT.n_pad, FPT.n_pad), mode='constant', constant_values=0)
    P_b2 = np.pad(P_b2, pad_width=(FPT.n_pad, FPT.n_pad), mode='constant', constant_values=0)    
    c_m = fpt.fourier_coefficients(P_b1, C_window)
    c_n = fpt.fourier_coefficients(P_b2, C_window)
    C_l = fpt.convolution(c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
    new_C_l = jconvolution(c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
    print(np.allclose(C_l, new_C_l))
    try:
        jacfwd(jconvolution, holomorphic=True)(c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
        print("jacfwd passed")
    except:
        print("jacfwd failed")

    # #Scalar case
    pf, p, g_m, g_n, two_part_l, h_l = X_spt
    P_b = P * FPT.k_extrap ** (2)
    P_b = np.pad(P_b, pad_width=(FPT.n_pad, FPT.n_pad), mode='constant', constant_values=0)
    c_m = fpt.fourier_coefficients(P_b, C_window)
    C_l = fpt.convolution(c_m, c_m, g_m[1,:], g_n[1,:], h_l[1,:], two_part_l[1])
    new_C_l = jconvolution(c_m, c_m, g_m[1,:], g_n[1,:], h_l[1,:], two_part_l[1])
    print(np.allclose(C_l, new_C_l))
    try:
        jacfwd(jconvolution, holomorphic=True)(c_m, c_m, g_m[1,:], g_n[1,:], h_l[1,:], two_part_l[1])
        print("jacfwd passed")
    except:
        print("jacfwd failed")
