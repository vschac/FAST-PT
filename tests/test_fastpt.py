import pytest
import numpy as np
from fastpt import FASTPT

P = np.loadtxt('benchmarking/Pk_test.dat')[:, 1]
C_window = 0.75

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

def test_init_odd_length_k():
    """Test initialization with odd-length k array"""
    k = np.logspace(-3, 1, 201)  # Odd length
    with pytest.raises(ValueError):
        FASTPT(k)

def test_init_non_log_spaced():
    """Test initialization with non-log-spaced k array"""
    k = np.linspace(0.1, 10, 200)  # Linear spacing
    with pytest.raises(AssertionError):
        FASTPT(k)

def test_init_invalid_to_do():
    """Test initialization with invalid to_do parameter"""
    k = np.logspace(-3, 1, 200)
    with pytest.raises(ValueError):
        FASTPT(k, to_do=['invalid_option'])

def test_init_extrapolation_ranges():
    """Test initialization with various extrapolation ranges"""
    k = np.logspace(-3, 1, 200)
            
    # Test valid extrapolation
    fpt = FASTPT(k, low_extrap=-5, high_extrap=3)
    assert fpt.low_extrap == -5
    assert fpt.high_extrap == 3
            
    # Test invalid extrapolation
    with pytest.raises(ValueError):
        FASTPT(k, low_extrap=3, high_extrap=-5)  # Invalid range

def test_init_padding(fpt):
    """Test initialization with different padding values"""
    k = np.logspace(-3, 1, 200)
            
    # Test with no padding
    fpt1 = FASTPT(k, n_pad=None)
    assert not hasattr(fpt1, 'n_pad')
            
    # Test with padding
    assert hasattr(fpt, 'n_pad')



def test_one_loop_dd(fpt):
    assert True

def test_one_lood_dd_bias(fpt):
    assert True

def test_one_loop_dd_bias_b3nl(fpt):
    assert True

def test_one_loop_dd_bias_lpt_NL(fpt):
    assert True

#def test_cleft_Q_R(fpt):
#   assert True

def test_IA_TT(fpt):
    assert True

def test_IA_mix(fpt):
    assert True

def test_IA_ta(fpt):
    assert True

def test_IA_der(fpt):
    assert True

def test_IA_ct(fpt):
    assert True

def test_IA_ctbias(fpt):
    assert True

def test_IA_d2(fpt):
    assert True

def test_IA_s2(fpt):
    assert True

def test_OV(fpt):
    assert True

def test_kPol(fpt):
    assert True

def test_RSD_components(fpt):
    assert True

def test_RSD_ABsum_components(fpt):
    assert True

def test_RSD_ABsum_mu(fpt):
    assert True

def test_IRres(fpt):
    assert True

def test_one_loop(fpt):
    assert True

def test_P_bias(fpt):
    assert True

def test_J_K_scalar(fpt):
    assert True

def test_J_K_tensor(fpt):
    assert True
    