import pytest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from fastpt import FASTPT, FPTHandler
from fastpt.fastpt_extr import p_window as og_p_win, c_window as og_c_win

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
P = np.loadtxt(data_path)[:, 1]
C_window = 0.75
P_window = np.array([0.2, 0.2])

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

@pytest.fixture
def fpt(): 
    d = np.loadtxt(data_path)
    k = d[:, 0]
    return FASTPT(k, low_extrap=-5, high_extrap=3)

@pytest.fixture
def handler(fpt):
    return FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)

def test_P_window(fpt):
    assert np.allclose(og_p_win(fpt.k_extrap, P_window[0], P_window[1]), 
                       p_window(fpt.k_extrap, P_window[0], P_window[1]))
    
def test_C_window(fpt):
    assert np.allclose(og_c_win(fpt.m, int(C_window * fpt.N / 2.)), 
                       c_window(fpt.m, int(C_window * fpt.N / 2.)))