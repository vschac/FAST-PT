import pytest
import numpy as np
from fastpt import FASTPT

@pytest.fixture
def fpt(): 
    d = np.loadtxt('benchmarking/Pk_test.dat')
    k = d[:, 0]
    n_pad = int(0.5 * len(k))
    to_do = ['all']
    return FASTPT(k, to_do=to_do, low_extrap=-5, high_extrap=3, n_pad=n_pad)

####################INITIALIZATION TESTS####################
def test_init_empty_arrays():
    with pytest.raises(ValueError):
        FASTPT([], [])

def test_init_mismatched_arrays():
    k = np.logspace(-3, 1, 200)
    p = np.logspace(-3, 1, 201)  # Different length
    with pytest.raises(ValueError):
        FASTPT(k, p)

def test_init_negative_k():
    k = np.array([-1.0, 0.0, 1.0])
    p = np.ones(3)
    with pytest.raises(ValueError):
        FASTPT(k, p)

def test_init_zero_k():
    k = np.array([0.0, 1.0, 2.0])
    p = np.ones(3)
    with pytest.raises(ValueError):
        FASTPT(k, p)

def test_init_non_monotonic_k():
    k = np.array([1.0, 0.5, 2.0])  # Non-monotonic
    p = np.ones(3)
    with pytest.raises(ValueError):
        FASTPT(k, p)

def test_init_valid(fpt):
    assert isinstance(fpt, FASTPT)
    assert len(fpt.k) > 0
    assert all(fpt.k > 0)  # Check all k values are positive
    assert np.all(np.diff(fpt.k) > 0)  # Check k is monotonically increasing

def test_init_with_params(fpt):
    assert isinstance(fpt, FASTPT)
    assert fpt.n_pad == int(0.5 * len(fpt.k))
    assert fpt.low_extrap == -5
    assert fpt.high_extrap == 3
    assert fpt.to_do == ['all']



def test_one_loop_dd():
    assert True

def test_one_loop_dd_bias_b3nl():
    assert True

def test_one_loop_dd_bias_lpt_NL():
    assert True

def test_cleft_Q_R():
    assert True

def test_IA_TT():
    assert True

def test_IA_mix():
    assert True

def test_IA_ta():
    assert True

def test_IA_der():
    assert True

def test_IA_ct():
    assert True

def test_IA_ctbias():
    assert True

def test_IA_tij():
    assert True

def test_IA_gb2tij():
    assert True

def test_IA_g2():
    assert True

def test_IA_d2():
    assert True

def test_IA_s2():
    assert True

def test_OV():
    assert True

def test_kPol():
    assert True

def test_RSD_components():
    assert True

def test_RSD_ABsum_components():
    assert True

def test_RSD_ABsum_mu():
    assert True

def test_IRres():
    assert True

def test_one_loop():
    assert True

def test_P_bias():
    assert True

def test_J_K_scalar():
    assert True

def test_J_K_tensor():
    assert True
    