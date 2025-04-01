import jax.numpy as jnp
from jax import grad
from jax import jit
from time import time
import numpy as np
from jax import jacfwd, jacrev
from jax import config
import jax
config.update("jax_enable_x64", True)

class JAXPT: 
    def __init__(self, k, n_pad=None):
        self.k = k
        self.k_extrap = k
        self.k_size = k.size
        delta_L = (np.log(k[-1]) - np.log(k[0])) / (k.size - 1)
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

        self.k_final = k
        self.N = k.size
        self.m = np.arange(-self.N // 2, self.N // 2 + 1)
        self.n_l = self.m.size + self.m.size - 1
        self.l = np.arange(-self.n_l // 2 + 1, self.n_l // 2 + 1)
        
        

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
    
    def J_k_scalar(self, P, X, nu, P_window=None, C_window=None):
        from numpy.fft import ifft, irfft

        pf, p, g_m, g_n, two_part_l, h_l = X

        # if (self.low_extrap is not None):
        #     P = self.EK.extrap_P_low(P)

        # if (self.high_extrap is not None):
        #     P = self.EK.extrap_P_high(P)

        P_b = P * self.k_extrap ** (-nu)

        if (self.n_pad > 0):
            P_b = np.pad(P_b, pad_width=(self.n_pad, self.n_pad), mode='constant', constant_values=0)

        c_m = self.fourier_coefficients(P_b, C_window)

        A_out = np.zeros((pf.shape[0], self.k_size))
        for i in range(pf.shape[0]):
            C_l = self.convolution(c_m, c_m, g_m[i,:], g_n[i,:], h_l[i,:], two_part_l[i])

            c_plus = C_l[self.l >= 0]
            c_minus = C_l[self.l < 0]

            C_l = np.hstack((c_plus[:-1], c_minus))
            A_k = ifft(C_l) * C_l.size 

            A_out[i, :] = np.real(A_k[::2]) * pf[i] * self.k_final ** (-p[i] - 2)

        P_out = irfft(c_m[self.m >= 0]) * self.k_final ** nu * float(self.N)
        if (self.n_pad > 0):
            P_out = P_out[self.id_pad]
            A_out = A_out[:, self.id_pad]

        return P_out, A_out
    





def jJ_k_scalar(P, X, nu, m, N, n_pad, id_pad, k_extrap, k_final, k_size, l, C_window=None, P_window=None, low_extrap=None, high_extrap=None, EK=None):
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
    
    c_m = jfourier_coefficients(P_b, m, N, C_window)
    
    A_out = jnp.zeros((pf.shape[0], k_size))
    
    def process_single_row(i):
        # Convolution
        C_l = jconvolution(c_m, c_m, g_m[i], g_n[i], h_l[i], None if two_part_l is None else two_part_l[i])
        
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



def jJ_k_tensor(P, X, k_extrap, k_final, k_size, n_pad, id_pad, l, m, N, C_window=None, P_window=None):
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
        window = jp_window(k_extrap, P_window[0], P_window[1])

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
            
        c_m = jfourier_coefficients(P_b1, m, N, C_window)
        c_n = jfourier_coefficients(P_b2, m, N, C_window)
        
        C_l = jconvolution(c_m, c_n, g_m[i,:], g_n[i,:], h_l[i,:])
        
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
    
    W = jnp.ones_like(n)
    
    right_mask = n > n_right
    theta_right = (n[-1] - n) / jnp.array(n[-1] - n_right - 1, dtype=float)
    right_window = theta_right - 1/(2*pi)*sin(2*pi*theta_right)
    
    left_mask = n < n_left
    theta_left = (n - n[0]) / jnp.array(n_left - n[0] - 1, dtype=float)
    left_window = theta_left - 1/(2*pi)*sin(2*pi*theta_left)
    
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
    
    mask_left = log_k <= log_k_left
    mask_right = log_k >= log_k_right
    
    # Extract elements that satisfy the masks
    left = log_k[mask_left]
    right = log_k[mask_right]

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

    W = jnp.ones_like(k)

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
    FPT = FASTPT(k)
    handler = FPTHandler(FPT)
    P = handler.generate_power_spectra()
    X_IA_A = FPT.X_IA_A
    X_spt = FPT.X_spt
    C_window = 0.75
    P_window = np.array([0.2, 0.2])

    # JK Scalar
    # print("JK Scalar")
    # old = FPT.J_k_scalar(P, X_spt, 2, C_window=C_window)
    # new = jJ_k_scalar(P, X_spt, 2, FPT.m, FPT.N, FPT.n_pad, FPT.id_pad, FPT.k_extrap, FPT.k_final, FPT.k_size, FPT.l, C_window=C_window)
    # print(np.allclose(old[0], new[0]) and np.allclose(old[1], new[1]))
    # try:
    #     jacfwd(jJ_k_scalar)(P, X_spt, 2, FPT.m, FPT.N, FPT.n_pad, FPT.id_pad, FPT.k_extrap, FPT.k_final, FPT.k_size, FPT.l, C_window=C_window)
    #     print("jacfwd passed")
    # except:
    #     print("jacfwd failed")

    # print("\n", "="*100, "\n")

    # # JK Tensor
    print("JK Tensor")
    print([type(FPT.X_IA_A[i]) for i in range(len(FPT.X_IA_A))])
    old = FPT.J_k_tensor(P, X_IA_A, P_window=P_window, C_window=C_window)
    new = jJ_k_tensor(P, X_IA_A, 
                      FPT.k_extrap, FPT.k_final, FPT.k_size, 
                      FPT.n_pad, FPT.id_pad, FPT.l, FPT.m, FPT.N,
                      P_window=P_window, C_window=C_window)
    print(np.allclose(old[0], new[0]) and np.allclose(old[1], new[1]))
    try:
        jacfwd(jJ_k_tensor)(P, X_IA_A, 
                            FPT.k_extrap, FPT.k_final, FPT.k_size, 
                            FPT.n_pad, FPT.id_pad, FPT.l, FPT.m, FPT.N,
                            P_window=P_window, C_window=C_window)
        print("jacfwd passed")
    except:
        print("jacfwd failed")

    # print("\n", "="*100, "\n")

    # # RFFT and FFT Convolve
    # print("RFFT and FFT Convolve")
    # from numpy.fft import rfft
    # old = rfft(P)
    # from jax.numpy.fft import rfft as jax_rfft
    # new = jax_rfft(P)
    # print(np.allclose(old, new))
    # try:
    #     jacfwd(jax_rfft)(P)
    #     print("jacfwd passed")
    # except:
    #     print("jacfwd failed")

    # print("=====================================================")

    # from scipy.signal import fftconvolve
    # old2 = fftconvolve(np.logspace(0, 1, 10), np.logspace(1, 2, 10), )
    # from jax.scipy.signal import fftconvolve as jax_fftconvolve
    # new2 = jax_fftconvolve(jnp.logspace(0, 1, 10), jnp.logspace(1, 2, 10))
    # print(np.allclose(old2, new2))
    # try:
    #     jacfwd(jax_fftconvolve)(jnp.logspace(0, 1, 10), jnp.logspace(1, 2, 10))
    #     print("jacfwd passed")
    # except:
    #     print("jacfwd failed")


    # print("\n", "="*100, "\n")


    # # P_window and C_window
    # print("P_window and C_window")
    # p1 = p_window(k, P_window[0], P_window[1])
    # p2 = jp_window(k, P_window[0], P_window[1])
    # print(np.allclose(p1, p2))
    # try:
    #     jacfwd(jp_window)(k, P_window[0], P_window[1])
    #     print("jacfwd passed")
    # except:
    #     print("jacfwd failed")

    # print("=====================================================")
    # m_float = jnp.asarray(FPT.m, dtype=jnp.float64)
    # c1 = c_window(FPT.m, C_window)
    # c2 = jc_window(m_float, jnp.float64(C_window))
    # print(np.allclose(c1, c2))
    # try:
    #     jacfwd(jc_window)(m_float, jnp.float64(C_window))
    #     print("jacfwd passed")
    # except:
    #     print("jacfwd failed")
        


    # print("\n", "="*100, "\n")


    # # Fourier Coefficients
    # print("Fourier Coefficients")
    # pf, p, nu1, nu2, g_m, g_n, h_l = X_IA_A
    # P_b1 = P * FPT.k_extrap ** (-nu1[1])
    # W = p_window(FPT.k_extrap, P_window[0], P_window[1])
    # P_b1 = P_b1 * W
    # P_b1 = np.pad(P_b1, pad_width=(FPT.n_pad, FPT.n_pad), mode='constant', constant_values=0)
    # c_m = fpt.fourier_coefficients(P_b1, C_window)
    # jc_m = jfourier_coefficients(P_b1, FPT.m, FPT.N, C_window)
    # print(np.allclose(c_m, jc_m))
    # try:
    #     jacfwd(jfourier_coefficients)(P_b1, FPT.m, FPT.N, C_window)
    #     print("jacfwd passed")
    # except:
    #     print("jacfwd failed")    


    # # Convolution
    # print("Convolution")
    # # #Tensor case
    # print("Tensor case")
    # pf, p, nu1, nu2, g_m, g_n, h_l = X_IA_A
    # P_b1 = P * FPT.k_extrap ** (-nu1[1])
    # P_b2 = P * FPT.k_extrap ** (-nu2[1])
    # W = p_window(FPT.k_extrap, P_window[0], P_window[1])
    # P_b1 = P_b1 * W
    # P_b2 = P_b2 * W
    # P_b1 = np.pad(P_b1, pad_width=(FPT.n_pad, FPT.n_pad), mode='constant', constant_values=0)
    # P_b2 = np.pad(P_b2, pad_width=(FPT.n_pad, FPT.n_pad), mode='constant', constant_values=0)    
    # c_m = fpt.fourier_coefficients(P_b1, C_window)
    # c_n = fpt.fourier_coefficients(P_b2, C_window)
    # C_l = fpt.convolution(c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
    # new_C_l = jconvolution(c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
    # print(np.allclose(C_l, new_C_l))
    # try:
    #     jacfwd(jconvolution, holomorphic=True)(c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
    #     print("jacfwd passed")
    # except:
    #     print("jacfwd failed")

    # # #Scalar case
    # pf, p, g_m, g_n, two_part_l, h_l = X_spt
    # P_b = P * FPT.k_extrap ** (2)
    # P_b = np.pad(P_b, pad_width=(FPT.n_pad, FPT.n_pad), mode='constant', constant_values=0)
    # c_m = fpt.fourier_coefficients(P_b, C_window)
    # C_l = fpt.convolution(c_m, c_m, g_m[1,:], g_n[1,:], h_l[1,:], two_part_l[1])
    # new_C_l = jconvolution(c_m, c_m, g_m[1,:], g_n[1,:], h_l[1,:], two_part_l[1])
    # print(np.allclose(C_l, new_C_l))
    # try:
    #     jacfwd(jconvolution, holomorphic=True)(c_m, c_m, g_m[1,:], g_n[1,:], h_l[1,:], two_part_l[1])
    #     print("jacfwd passed")
    # except:
    #     print("jacfwd failed")