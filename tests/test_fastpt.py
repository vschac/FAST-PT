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
    bmark = np.transpose(fpt.one_loop_dd(P, C_window=C_window)[0])
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_dd_benchmark.txt'))

def test_one_lood_dd_bias(fpt):
    bmark = list(fpt.one_loop_dd_bias(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[7]
    bmark[7] = new_array
    assert np.allclose(np.transpose(bmark), np.loadtxt('benchmarking/P_bias_benchmark.txt'))

def test_one_loop_dd_bias_b3nl(fpt):
    bmark = list(fpt.one_loop_dd_bias_b3nl(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[7]
    bmark[7] = new_array
    assert np.allclose(np.transpose(bmark), np.loadtxt('benchmarking/P_bias_b3nl_benchmark.txt'))

def test_one_loop_dd_bias_lpt_NL(fpt):
    bmark = list(fpt.one_loop_dd_bias_lpt_NL(P, C_window=C_window))
    new_array = np.zeros(3000)
    new_array[0] = bmark[6]
    bmark[6] = new_array
    assert np.allclose(np.transpose(bmark), np.loadtxt('benchmarking/P_bias_lpt_NL_benchmark.txt'))

#def test_cleft_Q_R(fpt):
#    bmark = np.transpose(fpt.cleft_Q_R(P, C_window=C_window)
#    assert np.allclose(bmark, np.loadtxt('benchmarking/Q_R_benchmark.txt'))

def test_IA_TT(fpt):
    bmark = np.transpose(fpt.IA_tt(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/PIA_tt_benchmark.txt'))

def test_IA_mix(fpt):
    bmark = np.transpose(fpt.IA_mix(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_IA_mix_benchmark.txt'))

def test_IA_ta(fpt):
    bmark = np.transpose(fpt.IA_ta(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_IA_ta_benchmark.txt'))

def test_IA_der(fpt):
    bmark = np.transpose(fpt.IA_der(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_IA_der_benchmark.txt'))

def test_IA_ct(fpt):
    bmark = np.transpose(fpt.IA_ct(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_IA_ct_benchmark.txt'))

def test_IA_ctbias(fpt):
    bmark = np.transpose(fpt.IA_ctbias(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_IA_ctbias_benchmark.txt'))

def test_IA_d2(fpt):
    bmark = np.transpose(fpt.IA_d2(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_IA_d2_benchmark.txt'))

def test_IA_s2(fpt):
    bmark = np.transpose(fpt.IA_s2(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_IA_s2_benchmark.txt'))

def test_OV(fpt):
    bmark = np.transpose(fpt.OV(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_OV_benchmark.txt'))

def test_kPol(fpt):
    bmark = np.transpose(fpt.kPol(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_kPol_benchmark.txt'))

def test_RSD_components(fpt):
    bmark = np.transpose(fpt.RSD_components(P, 1.0, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_RSD_benchmark.txt'))

def test_RSD_ABsum_components(fpt):
    bmark = np.transpose(fpt.RSD_ABsum_components(P, 1.0, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_RSD_ABsum_components_benchmark.txt'))

def test_RSD_ABsum_mu(fpt):
    bmark = np.transpose(fpt.RSD_ABsum_mu(P, 1.0, 1.0, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_RSD_ABsum_mu_benchmark.txt'))

def test_IRres(fpt):
    bmark = np.transpose(fpt.IRres(P, C_window=C_window))
    assert np.allclose(bmark, np.loadtxt('benchmarking/P_IRres_benchmark.txt'))

def test_one_loop():
    assert True

def test_P_bias():
    assert True

def test_J_K_scalar():
    assert True

def test_J_K_tensor():
    assert True
    