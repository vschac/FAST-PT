from jax import numpy as jnp
from jax.numpy import pi, sin, log10, log, exp
from jax import lax

def c_window(n, n_cut):
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


def p_window(k, log_k_left, log_k_right):
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
        W_left = x_left - (1 / (2 * pi)) * sin(2 * pi * x_left)
    else:
        W_left = jnp.array([])  # Avoid shape mismatch

    if right.size > 0:
        x_right = (right - right[-1]) / (right[0] - max_log_k)
        W_right = x_right - (1 / (2 * pi)) * sin(2 * pi * x_right)
    else:
        W_right = jnp.array([])  # Avoid shape mismatch

    W = jnp.ones_like(k)

    if W_left.size > 0:
        W = W.at[mask_left].set(W_left)
    if W_right.size > 0:
        W = W.at[mask_right].set(W_right)

    return W


class jax_k_extend: 

    def __init__(self,k,low=None,high=None):
        # Initialize with original k 
        self.k = k.copy()  # Store original k explicitly
        self.DL = log(k[1])-log(k[0]) 
        
        if low is not None:
            if (low > log10(k[0])):
                low=log10(k[0])
                print('Warning, you selected a extrap_low that is greater than k_min. Therefore no extrapolation will be done.')
        
            low=10**low
            low=log(low)
            N=jnp.absolute(int((log(k[0])-low)/self.DL))
           
            if (N % 2 != 0 ):
                N=N+1 
            s=log(k[0]) -(jnp.arange(0,N)+1)*self.DL 
            s=s[::-1]
            self.k_min=k[0]
            self.k_low=exp(s) 
           
            self.k=jnp.append(self.k_low,k)
            self.id_extrap=jnp.where(self.k >=self.k_min)[0] 
        else:
            self.k_min = k[0]
            self.id_extrap = jnp.arange(len(k))  # Set default id_extrap
            

        if high is not None:
            if (high < log10(k[-1])):
                high=log10(k[-1])
                print('Warning, you selected a extrap_high that is less than k_max. Therefore no extrapolation will be done.')
                #raise ValueError('Error in P_extend.py. You can not request an extension to high k that is less than your input k_max.')
            
            high=10**high
            high=log(high)
            N=jnp.absolute(int((log(k[-1])-high)/self.DL))
            
            if (N % 2 != 0 ):
                N=N+1 
            s=log(k[-1]) + (jnp.arange(0,N)+1)*self.DL 
            self.k_max=k[-1]
            self.k_high=exp(s)
            self.k=jnp.append(self.k,self.k_high)
            self.id_extrap=jnp.where(self.k <= self.k_max)[0] 
        else:
            self.k_max = k[-1]
            # id_extrap is already set if neither high nor low is specified
            

        if (high is not None) & (low is not None):
            self.id_extrap=jnp.where((self.k <= self.k_max) & (self.k >=self.k_min))[0]
            
            
    def extrap_k(self):
        return self.k 
        
    def extrap_P_low(self,P):
        # If no low extension, return input
        if not hasattr(self, 'k_low') or self.k_low is None:
            return P
      
        ns=(log(P[1])-log(P[0]))/self.DL
        Amp=P[0]/self.k_min**ns
        P_low=self.k_low**ns*Amp
        return jnp.append(P_low,P) 

    def extrap_P_high(self,P):
        # If no high extension, return input
        if not hasattr(self, 'k_high') or self.k_high is None:
            return P
       
        ns=(log(P[-1])-log(P[-2]))/self.DL
        Amp=P[-1]/self.k_max**ns
        P_high=self.k_high**ns*Amp
        return jnp.append(P,P_high) 
    
    def PK_original(self,P): 
        return self.k[self.id_extrap], P[self.id_extrap]