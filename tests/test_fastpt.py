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


def test_validate_params_decorator(fpt):
    """Test the validate_params_decorator function with various inputs
        (Using one_loop_dd as a sample method though all decorating functions
        will follow the same validation behavior)"""
    P_window = np.array([0.2, 0.2])
    # Test 1: Valid cases
    assert fpt.one_loop_dd(P) is not None
    assert fpt.one_loop_dd(P, P_window=P_window, C_window=0.5) is not None
    
    # Test 2: Empty or None power spectrum
    with pytest.raises(ValueError, match=r'You must provide an input power spectrum array'):
        fpt.one_loop_dd(None)
    with pytest.raises(ValueError, match=r'You must provide an input power spectrum array'):
        fpt.one_loop_dd([])
        
    # Test 3: Zero power spectrum
    k = fpt.k
    P_zero = np.zeros_like(k)
    with pytest.raises(ValueError, match=r'Your input power spectrum array is all zeros'):
        fpt.one_loop_dd(P_zero)
    
    # Test 4: P_window validation
    max_window = (np.log(fpt.k[-1]) - np.log(fpt.k[0])) / 2
    Max_P_window = np.array([max_window, max_window])
    assert fpt.one_loop_dd(P, P_window=Max_P_window / 2) is not None
    
    with pytest.raises(ValueError, match=r'P_window must be a tuple of two values.'):
        fpt.one_loop_dd(P, P_window=Max_P_window[:-1])

    with pytest.raises(ValueError, match=r'P_window value is too large'):
        fpt.one_loop_dd(P, P_window=Max_P_window * 2)
        
    # Test 5: C_window validation
    # Test valid C_window values
    assert fpt.one_loop_dd(P, C_window=0.0) is not None
    assert fpt.one_loop_dd(P, C_window=0.5) is not None
    assert fpt.one_loop_dd(P, C_window=1.0) is not None
    
    # Test invalid C_window values
    with pytest.raises(ValueError, match=r'C_window must be between 0 and 1'):
        fpt.one_loop_dd(P, C_window=-0.1)
    with pytest.raises(ValueError, match=r'C_window must be between 0 and 1'):
        fpt.one_loop_dd(P, C_window=1.1)
        
    # Test 6: Combined parameter validation
    with pytest.raises(ValueError):
        fpt.one_loop_dd(None, P_window=P_window, C_window=0.5)
    with pytest.raises(ValueError):
        fpt.one_loop_dd(P, P_window=Max_P_window * 2, C_window=1.1)





####################EDGE CASE TESTS####################
def test_one_loop_dd(fpt):
    """Test the one_loop_dd function with various inputs"""
    # Test with standard input
    result = fpt.one_loop_dd(P)
    assert isinstance(result, tuple)

    # Test with window functions
    P_window = np.array([0.2, 0.2])
    result_window = fpt.one_loop_dd(P, P_window=P_window, C_window=C_window)
    assert isinstance(result_window, tuple)
    
    # Test shape consistency
    assert result[0].shape == P.shape

def test_one_lood_dd_bias(fpt):
    """Test the one_loop_dd_bias function including bias terms"""
        
    # Test standard calculation
    result = fpt.one_loop_dd_bias(P)
    assert isinstance(result, tuple)
        
    # Verify sigma4 calculation is positive
    assert result[-2] > 0  # sig4 should be positive
        
    # Test bias terms have correct shapes
    for term in result[1:-2]:  # Skip P_1loop and sig4
        assert term.shape == P.shape
            
    # Test with window functions
    result_window = fpt.one_loop_dd_bias(P, P_window=None, C_window=C_window)
    assert isinstance(result_window, tuple)

def test_one_loop_dd_bias_b3nl(fpt):
    """Test the one_loop_dd_bias_b3nl function including b3nl terms"""
        
    # Test standard calculation
    result = fpt.one_loop_dd_bias_b3nl(P)
    assert isinstance(result, tuple)
        
    # Test sig3nl term
    assert hasattr(result, 'sig3nl')
    assert result.sig3nl.shape == P.shape
        
    # Test with window functions
    result_window = fpt.one_loop_dd_bias_b3nl(P, P_window=None, C_window=C_window)
    assert isinstance(result_window, tuple)
        
    # Verify consistency between b3nl and standard bias results
    result_bias = fpt.one_loop_dd_bias(P)
    for i in range(min(len(result), len(result_bias))):
        assert np.allclose(result[i], result_bias[i])

def test_one_loop_dd_bias_lpt_NL(fpt):
    """Test the one_loop_dd_bias_lpt_NL function"""
        
    # Test standard calculation
    result = fpt.one_loop_dd_bias_lpt_NL(P)
    assert isinstance(result, tuple)
        
    # Test shapes of LPT bias terms
    expected_terms = ['Pb1L', 'Pb1L_2', 'Pb1L_b2L', 'Pb2L', 'Pb2L_2']
    for term, name in zip(result, expected_terms):
        assert term.shape == P.shape, f"{name} has incorrect shape"
        
    # Test with window functions
    result_window = fpt.one_loop_dd_bias_lpt_NL(P, P_window=None, C_window=0.75)
    assert isinstance(result_window, tuple)




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
    