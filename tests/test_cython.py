import pytest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from time import time

from fastpt import FASTPT, CYTHON_AVAILABLE
from fastpt.fastpt_extr import p_window as og_p_win, c_window as og_c_win

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
P = np.loadtxt(data_path)[:, 1]
C_window = 0.75
P_window = np.array([0.2, 0.2])

####### NEED TO TURN OFF CYTHON_AVAILABLE BEFORE RUNNING TESTS ###########
try:
    from fastpt.cython_pt.fastpt_core import (
        p_window,
        c_window,
        compute_term_cy,
        apply_extrapolation_cy,
        compute_convolution,
        compute_fourier_coefficients,
        J_k_scalar_cy,
        J_k_tensor_cy,
        clear_workspaces,
    )
    from fastpt.cython_pt.cython_CacheManager import CacheManager_cy
except ImportError as e:
    import traceback
    traceback.print_exc()

cache = CacheManager_cy()

@pytest.fixture
def fpt(): 
    d = np.loadtxt(data_path)
    k = d[:, 0]
    return FASTPT(k)

def test_P_window(fpt):
    t0 = time()
    r1 = og_p_win(fpt.k_extrap, P_window[0], P_window[1])
    t1 = time()
    og = t1 - t0
    t2 = time()
    r2 = p_window(fpt.k_extrap, P_window[0], P_window[1])
    t3 = time()
    cy = t3 - t2
    #assert (og > cy)
    assert np.allclose(r1, r2)
    
def test_C_window(fpt):
    assert np.allclose(og_c_win(fpt.m, int(C_window * fpt.N / 2.)), 
                       c_window(fpt.m, int(C_window * fpt.N / 2.)))
    
def test_compute_fourier(fpt):
    P_b = P * fpt.k_extrap ** (-2)
    P_b = np.pad(P_b, pad_width=(fpt.n_pad, fpt.n_pad), mode='constant', constant_values=0)
    t0 = time()
    r1 = fpt._cache_fourier_coefficients(P_b, C_window=C_window)
    t1 = time()
    og = t1 - t0
    fpt.cache.clear()
    t2 = time()
    r2 = compute_fourier_coefficients(cache, P_b, fpt.m, fpt.N,
                                      c_window_func=og_c_win, c_window_param=C_window)
    t3 = time()
    cy = t3 - t2
    assert (og > cy)
    assert np.allclose(r1, r2)


def test_compute_convolution(fpt):
    #Used i = 1 for these tests case
    #Scalar Case (C_m, C_m)
    X = fpt.X_spt
    P_b = P * fpt.k_extrap ** (-2)
    P_b = np.pad(P_b, pad_width=(fpt.n_pad, fpt.n_pad), mode='constant', constant_values=0)
    c_m = fpt._cache_fourier_coefficients(P_b, C_window)
    pf, p, g_m, g_n, two_part_l, h_l = X
    r1 = fpt._cache_convolution(c_m, c_m, g_m[1,:], g_n[1,:], h_l[1,:], two_part_l[1])
    fpt.cache.clear()
    r2 = compute_convolution(cache, c_m, c_m, g_m[1,:], g_n[1,:], h_l[1,:], two_part_l[1])
    assert np.allclose(r1, r2)

    #Tensor Case (C_m, C_n)
    X = fpt.X_IA_A
    P_b1 = P * fpt.k_extrap ** (-nu1[1])
    P_b1 = np.pad(P_b1, pad_width=(fpt.n_pad, fpt.n_pad), mode='constant', constant_values=0)
    P_b2 = P * fpt.k_extrap ** (-nu2[1])
    c_m = fpt._cache_fourier_coefficients(P_b1, C_window)
    c_n = fpt._cache_fourier_coefficients(P_b2, C_window)
    r1 = fpt._cache_convolution(c_m, c_m, g_m[1,:], g_n[1,:], h_l[1,:])
    fpt.cache.clear()
    r2 = compute_convolution(cache, c_m, c_n, g_m[1,:], g_n[1,:], h_l[1,:])
    assert np.allclose(r1, r2)

    