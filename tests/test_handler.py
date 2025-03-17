import pytest
import numpy as np
from fastpt import FASTPT
from fastpt.FPTHandler import FPTHandler
import os

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
P = np.loadtxt(data_path)[:, 1]
k = np.loadtxt(data_path)[:, 0]
C_window = 0.75
P_window = np.array([0.2, 0.2])

if __name__ == "__main__":
    fpt = FASTPT(k, n_pad=int(0.5 * len(k)))
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window, f=0.6, mu_n=0.5, L=0.2, h=0.67, rsdrag=135)
    funcs = ['one_loop_dd', 'IA_tt', "IA_mix", "IA_ta", "IA_der", "IA_ct", "IA_ctbias", "IA_gb2", "IA_d2", "IA_s2", "OV", "kPol", "RSD_components", "RSD_ABsum_components", "RSD_ABsum_mu"]
    power_spectra = [P, P * 1.1, P * 1.2]
    results = handler.bulk_run(funcs, power_spectra, verbose=True, P_window=P_window, C_window=C_window)


@pytest.fixture
def fpt():
    k = np.loadtxt(data_path)[:, 0]
    n_pad=int(0.5*len(k))
    return FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=n_pad)

################# FUNCTIONALITY TESTS #################
def test_init_with_valid_params(fpt):
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    assert isinstance(handler.fastpt, FASTPT)
    assert handler.default_params['P'] is not None
    assert handler.default_params['P_window'].all() == P_window.all()
    assert handler.default_params['C_window'] == 0.75

def test_run_without_power_spectrum(fpt):
    handler = FPTHandler(fpt) #P is not required at handler init but must be passed at every function call
    with pytest.raises(ValueError, match="Missing required parameters for 'one_loop_dd': \\['P'\\]. Please recall with the missing parameters."):
        handler.run('one_loop_dd')

def test_init_with_zero_power_spectrum(fpt):
    P = np.zeros_like(fpt.k_original)
    with pytest.raises(ValueError, match='Your input power spectrum array is all zeros'):
        FPTHandler(fpt, P=P)

def test_init_with_mismatched_arrays(fpt):
    P = np.ones(10)  # Wrong size
    with pytest.raises(ValueError, match='Input k and P arrays must have the same size'):
        FPTHandler(fpt, P=P)

def test_invalid_c_window(fpt):
    with pytest.raises(ValueError, match='C_window must be between 0 and 1'):
        FPTHandler(fpt, P=P, C_window=1.5)
    with pytest.raises(ValueError, match='C_window must be between 0 and 1'):
        FPTHandler(fpt, P=P, C_window=-0.5)

def test_invalid_p_window(fpt):
    with pytest.raises(ValueError, match='P_window must be a tuple of two values'):
        FPTHandler(fpt, P=P, P_window=np.array([1.0]))

def test_cache_functionality(fpt):
    handler = FPTHandler(fpt, do_cache=True, P=P)
    handler.run('one_loop_dd')  # First run
    cache_size_before = len(handler.cache)
    handler.run('one_loop_dd')  # Should use cache
    assert len(handler.cache) == cache_size_before
    handler.clear_cache()
    assert len(handler.cache) == 0

def test_cache_with_other_params(fpt):
    """Test caching behavior with various parameter combinations"""
    handler = FPTHandler(fpt, do_cache=True)
    
    # Test parameters that should result in different cache entries
    param_combinations = [
        {'P': P, 'X': 0.5, 'nu': -2},  
        {'P': P, 'f': 0.5},
        {'P': P, 'f': 0.5, 'mu_n': 0.5},
        {'P': P, 'L': 0.2, 'h': 0.67, 'rsdrag': 135} 
    ]
    
    for params in param_combinations:
        func_name = 'one_loop_dd'
        
        # First run should compute and cache
        result1 = handler.run(func_name, **params)
        cache_size = len(handler.cache)
        
        # Second run should use cache
        result2 = handler.run(func_name, **params)
        assert len(handler.cache) == cache_size
        
        if isinstance(result1, (tuple, list)):
            assert len(result1) == len(result2)
            for r1, r2 in zip(result1, result2):
                if isinstance(r1, np.ndarray):
                    assert np.array_equal(r1, r2)
                else:
                    assert r1 == r2
        elif isinstance(result1, np.ndarray):
            assert np.array_equal(result1, result2)
        else:
            assert result1 == result2
    
    # Verify different parameter values create different cache entries
    handler.clear_cache()
    base_params = {'P': P}
    
    # Run with different parameter values
    handler.run('one_loop_dd', **base_params)
    cache_size = len(handler.cache)
    
    # Modify P slightly and verify new cache entry is created
    modified_P = P * 1.01
    handler.run('one_loop_dd', P=modified_P)
    assert len(handler.cache) > cache_size, "Different P values should create new cache entry"

def test_invalid_function_call(fpt):
    handler = FPTHandler(fpt, P=P)
    with pytest.raises(ValueError, match="Function 'nonexistent_function' not found"):
        handler.run('nonexistent_function')

def test_missing_required_params(fpt):
    handler = FPTHandler(fpt, P=P)
    with pytest.raises(ValueError, match="Missing required parameters"):
        handler.run('RSD_components')

def test_clear_specific_cache(fpt):
    handler = FPTHandler(fpt, do_cache=True, P=P)
    handler.run('one_loop_dd')
    handler.run('one_loop_dd_bias')
    handler.clear_cache('one_loop_dd')
    assert any('one_loop_dd_bias' in key[0] for key in handler.cache.keys())
    assert all(key[0] != 'one_loop_dd' for key in handler.cache.keys())

def test_all_fastpt_functions_with_handler_params(fpt):
    """Test FASTPT functions with parameters set during handler initialization"""
    # Initialize handler with all possible parameters
    default_params = {
        'P': P,
        'P_window': P_window,
        'C_window': C_window,
        'f': 0.5,
        'mu_n': 0.5,
        'nu': -2,
        'L': 0.2,
        'h': 0.67,
        'rsdrag': 135
    }
    handler = FPTHandler(fpt, **default_params)
    
    # Dictionary mapping functions to any additional required parameters
    func_names = (
        'one_loop_dd', 'one_loop_dd_bias', 'one_loop_dd_bias_b3nl',
        'one_loop_dd_bias_lpt_NL', 'IA_tt', 'IA_mix', 'IA_ta',
        'IA_der', 'IA_ct', 'IA_ctbias', 'IA_gb2', 'IA_d2',
        'IA_s2', 'OV', 'kPol', 'RSD_components', 'IRres',
        'RSD_ABsum_components', 'RSD_ABsum_mu',
    )
    
    for name in func_names:
        try:
            result = handler.run(name)
            assert result is not None, f"Function {name} returned None"
            if isinstance(result, tuple):
                assert all(r is not None for r in result), f"Function {name} returned None in tuple"
        except Exception as e:
            pytest.fail(f"Function {name} failed to run with error: {str(e)}")

def test_all_fastpt_functions_with_run_params(fpt):
    """Test FASTPT functions with parameters passed during run call"""
    handler = FPTHandler(fpt)
    
    # Dictionary mapping functions to their required run-time parameters
    function_params = {
        'one_loop_dd': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias_b3nl': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias_lpt_NL': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_tt': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_mix': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_ta': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_der': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_ct': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_ctbias': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_gb2': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_d2': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_s2': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'OV': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'kPol': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'RSD_components': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5},
        'RSD_ABsum_components': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5},
        'RSD_ABsum_mu': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5, 'mu_n': 0.5}
    }
    
    for func_name, params in function_params.items():
        try:
            result = handler.run(func_name, **params)
            assert result is not None, f"Function {func_name} returned None"
            if isinstance(result, tuple):
                assert all(r is not None for r in result), f"Function {func_name} returned None in tuple"
        except Exception as e:
            pytest.fail(f"Function {func_name} failed to run with error: {str(e)}")

def test_clear_params(fpt):
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    handler.clear_default_params()
    assert handler.default_params == {}

def test_override_params(fpt):
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    original_params = handler.default_params.copy()
    r1 = handler.run('one_loop_dd')
    new_params = {'P': P * 2, 'P_window': np.array([0.1, 0.1]), 'C_window': 0.5}
    # Run with overridden parameters (but this doesn't update default_params)
    r2 = handler.run('one_loop_dd', **new_params)
    assert not np.array_equal(r1, r2)
    # Assert that default_params weren't changed
    for key in original_params:
        if isinstance(original_params[key], np.ndarray):
            assert np.array_equal(handler.default_params[key], original_params[key])
        else:
            assert handler.default_params[key] == original_params[key]

def test_update_params(fpt):
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    new_params = {'P': P * 2, 'P_window': np.array([0.1, 0.1]), 'C_window': 0.5}
    handler.update_default_params(**new_params)
    assert handler.default_params == new_params

def test_update_fpt_instance(fpt):
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    handler.run('one_loop_dd')  # Run to initialize cache
    new_fpt = FASTPT(fpt.k_original)
    handler.update_fastpt_instance(new_fpt)
    assert handler.fastpt == new_fpt
    assert len(handler.cache) == 0

def test_max_cache_entries(fpt):
    handler = FPTHandler(fpt, max_cache_entries=5, P=P, P_window=P_window, C_window=C_window)
    for i in range(10):
        handler.run('one_loop_dd')
    assert len(handler.cache) <= 5

################# result_direct TESTS #################
def test_handler_function_equality(fpt):
    """Test that handler produces identical results to direct FASTPT function calls"""
    handler = FPTHandler(fpt)
    
    function_params = {
        'one_loop_dd': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias_b3nl': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'one_loop_dd_bias_lpt_NL': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_tt': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_mix': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_ta': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_der': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_ct': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_ctbias': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_gb2': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_d2': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'IA_s2': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'OV': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'kPol': {'P': P, 'P_window': P_window, 'C_window': C_window},
        'RSD_components': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5},
        'RSD_ABsum_components': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5},
        'RSD_ABsum_mu': {'P': P, 'P_window': P_window, 'C_window': C_window, 'f': 0.5, 'mu_n': 0.5}
    }
    
    for func_name, params in function_params.items():
        try:
            # Get direct FASTPT function
            fastpt_func = getattr(fpt, func_name)
            
            # Run both ways
            direct_result = fastpt_func(**params)
            handler_result = handler.run(func_name, **params)
            
            # Compare results
            if isinstance(direct_result, (tuple, list)):
                assert isinstance(handler_result, (tuple, list))
                assert len(direct_result) == len(handler_result)
                for dr, hr in zip(direct_result, handler_result):
                    if isinstance(dr, np.ndarray):
                        assert np.array_equal(dr, hr)
                    else:
                        assert dr == hr
            elif isinstance(direct_result, np.ndarray):
                assert np.array_equal(direct_result, handler_result)
            else:
                assert direct_result == handler_result
                
        except Exception as e:
            pytest.fail(f"Function {func_name} comparison failed with error: {str(e)}")


########### GET METHOD TESTING ############
def test_get_method_basics(fpt):
        """Test the basic functionality of the get method"""
        handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
        
        # Test single term retrieval
        p_deltaE1 = handler.get("P_deltaE1")
        p_deltaE1_direct = fpt.IA_ta(P=P, P_window=P_window, C_window=C_window)[0]
        assert np.array_equal(p_deltaE1, p_deltaE1_direct)
        
        # Test multiple terms retrieval
        terms = handler.get("P_deltaE1", "P_0E0E")
        assert "P_deltaE1" in terms
        assert "P_0E0E" in terms
        assert np.array_equal(terms["P_deltaE1"], p_deltaE1_direct)
        assert np.array_equal(terms["P_0E0E"], fpt.IA_ta(P=P, P_window=P_window, C_window=C_window)[2])

@pytest.mark.parametrize("term_name", ["P_1loop", "Ps", 
                                       "Pd1d2", "Pd2d2", "Pd1s2", "Pd2s2", "Ps2s2", "sig4",
                                       "sig3nl",
                                       "Pb1L", "Pb1L_2", "Pb1L_b2L", "Pb2L", "Pb2L_2",
                                       "P_E", "P_B",
                                       "P_A", "P_Btype2", "P_DEE", "P_DBB",
                                       "P_deltaE1", "P_deltaE2", "P_0E0E", "P_0B0B",
                                       "P_gb2sij", "P_gb2dsij", "P_gb2sij2",
                                       "P_der",
                                       "P_0tE", "P_0EtE", "P_E2tE", "P_tEtE",
                                       "P_d2tE", "P_s2tE",
                                       "P_s2E", "P_s20E", "P_s2E2",
                                       "P_d2E", "P_d20E", "P_d2E2",
                                       "P_OV",
                                       "P_kP1", "P_kP2", "P_kP3"])                   
def test_get_all_terms(fpt, term_name):
    term_sources = {
            "P_1loop": ("one_loop_dd", 0),
            "Ps": ("one_loop_dd", 1),
            "Pd1d2": ("one_loop_dd_bias", 2),  
            "Pd2d2": ("one_loop_dd_bias", 3),
            "Pd1s2": ("one_loop_dd_bias", 4),
            "Pd2s2": ("one_loop_dd_bias", 5),
            "Ps2s2": ("one_loop_dd_bias", 6),
            "sig4": ("one_loop_dd_bias", 7),
        
            "sig3nl": ("one_loop_dd_bias_b3nl", 8),
        
            "Pb1L": ("one_loop_dd_bias_lpt_NL", 1),
            "Pb1L_2": ("one_loop_dd_bias_lpt_NL", 2),
            "Pb1L_b2L": ("one_loop_dd_bias_lpt_NL", 3),
            "Pb2L": ("one_loop_dd_bias_lpt_NL", 4),
            "Pb2L_2": ("one_loop_dd_bias_lpt_NL", 5),
        
            "P_E": ("IA_tt", 0),
            "P_B": ("IA_tt", 1),
        
            "P_A": ("IA_mix", 0),
            "P_Btype2": ("IA_mix", 1),
            "P_DEE": ("IA_mix", 2),
            "P_DBB": ("IA_mix", 3),
        
            "P_deltaE1": ("IA_ta", 0),
            "P_deltaE2": ("IA_ta", 1),
            "P_0E0E": ("IA_ta", 2),
            "P_0B0B": ("IA_ta", 3),
        
            "P_gb2sij": ("IA_gb2", 0),
            "P_gb2dsij": ("IA_gb2", 1),
            "P_gb2sij2": ("IA_gb2", 2),

            "P_der": ("IA_der", 0),

            "P_0tE": ("IA_ct", 0),
            "P_0EtE": ("IA_ct", 1),
            "P_E2tE": ("IA_ct", 2),
            "P_tEtE": ("IA_ct", 3),
        
            "P_d2tE": ("IA_ctbias", 0),
            "P_s2tE": ("IA_ctbias", 1),
        
            "P_s2E": ("IA_s2", 0),
            "P_s20E": ("IA_s2", 1),
            "P_s2E2": ("IA_s2", 2),
        
            "P_d2E": ("IA_d2", 0),
            "P_d20E": ("IA_d2", 1),
            "P_d2E2": ("IA_d2", 2),
        
            "P_OV": ("OV", 0),
        
            "P_kP1": ("kPol", 0),
            "P_kP2": ("kPol", 1),
            "P_kP3": ("kPol", 2),
        
            # "A1": ("RSD_components", 0),
            # "A3": ("RSD_components", 1),
            # "A5": ("RSD_components", 2),
            # "B0": ("RSD_components", 3),
            # "B2": ("RSD_components", 4),
            # "B4": ("RSD_components", 5),
            # "B6": ("RSD_components", 6),
            # "P_Ap1": ("RSD_components", 7),
            # "P_Ap3": ("RSD_components", 8),
            # "P_Ap5": ("RSD_components", 9),
        
            # "ABsum_mu2": ("RSD_ABsum_components", 0),
            # "ABsum_mu4": ("RSD_ABsum_components", 1),
            # "ABsum_mu6": ("RSD_ABsum_components", 2),
            # "ABsum_mu8": ("RSD_ABsum_components", 3),
        
            # "ABsum": ("RSD_ABsum_components", 0),

            # "P_IRres": ("IRres", 0),
        }
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    term_source = term_sources[term_name]
    result = handler.get(term_name)
    result_direct = getattr(fpt, term_source[0])(P=P, P_window=P_window, C_window=C_window)
    if isinstance(result_direct, tuple):
        result_direct = result_direct[term_source[1]]
    assert np.allclose(result, result_direct)
    fpt.cache.clear()
    result_direct2 = getattr(fpt, term_source[0])(P=P, P_window=P_window, C_window=C_window)
    if isinstance(result_direct2, tuple):
        result_direct2 = result_direct2[term_source[1]]
    assert np.allclose(result, result_direct2)


def test_get_with_caching(fpt):
    """Test get method with caching enabled"""
    handler = FPTHandler(fpt, do_cache=True, P=P, P_window=P_window, C_window=C_window)
        
    # First call - should compute
    result1 = handler.get("P_deltaE1")
    
    # Check cache statistics before second call
    initial_hits = fpt.cache.hits
    initial_misses = fpt.cache.misses
    
    # Second call - should use cache
    result2 = handler.get("P_deltaE1")
    
    # Verify cache was used (hits increased)
    assert fpt.cache.hits > initial_hits
    assert fpt.cache.misses == initial_misses
    assert np.array_equal(result1, result2)
    
    # Clear the cache
    fpt.cache.clear()
    
    # After clearing, the next call should recompute (miss)
    pre_miss_count = fpt.cache.misses
    handler.get("P_deltaE1")
    assert fpt.cache.misses > pre_miss_count
    
    # Check that multiple different calculations create different cache entries
    pre_cache_count = len(fpt.cache.cache)
    handler.get("P_0E0E")  # Different calculation
    assert len(fpt.cache.cache) > pre_cache_count

def test_get_with_different_params(fpt):
    """Test get method with different parameter combinations"""
    handler = FPTHandler(fpt)
        
    # Test with parameters provided at runtime
    result1 = handler.get("P_deltaE1", P=P, P_window=P_window, C_window=C_window)
    fpt.cache.clear()
    # Test with default parameters
    handler2 = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    result2 = handler2.get("P_deltaE1")
        
    assert np.array_equal(result1, result2)
        
    # Test with override parameters
    handler3 = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    new_P = P * 5
    result3 = handler3.get("P_deltaE1", P=new_P)
        
    assert not np.array_equal(result1, result3)

def test_get_invalid_term(fpt):
    """Test get method with invalid term name"""
    handler = FPTHandler(fpt, P=P)
        
    with pytest.raises(ValueError, match="Term 'nonexistent_term' not found in FASTPT"):
        handler.get("nonexistent_term")

def test_get_missing_params(fpt):
        """Test get method with missing required parameters"""
        handler = FPTHandler(fpt)
        
        with pytest.raises(ValueError, match="Missing required parameters"):
            handler.get("P_deltaE1")  # P is required

def test_get_special_terms(fpt):
    """Test get method with special terms that have their own functions"""
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
        
    p_btype2 = handler.get("P_Btype2")
    fpt.cache.clear()
    p_btype2_direct = fpt._get_P_Btype2(P)
        
    assert np.array_equal(p_btype2, p_btype2_direct)
        
    p_deltaE2 = handler.get("P_deltaE2")
    fpt.cache.clear()
    p_deltaE2_direct = fpt._get_P_deltaE2(P)
        
    assert np.array_equal(p_deltaE2, p_deltaE2_direct)
    
    p_ov = handler.get("P_OV")
    fpt.cache.clear()
    p_ov_direct = fpt.OV(P, P_window=P_window, C_window=C_window)
    assert np.array_equal(p_ov, p_ov_direct), f"p_ov: {p_ov}, p_ov_direct: {p_ov_direct}"

    p_der = handler.get("P_der")
    fpt.cache.clear()
    p_der_direct = fpt.IA_der(P)
    assert np.array_equal(p_der, p_der_direct)


def test_get_edge_cases(fpt):
    """Test edge cases for the get method"""
    # Test with empty parameters
    handler = FPTHandler(fpt)
    
    # Test with empty term list
    with pytest.raises(ValueError, match="At least one term must be provided."):
        handler.get()
    
    # Test with required parameters passed directly at call
    result = handler.get("P_E", P=P, P_window=P_window, C_window=C_window)
    assert result is not None
    
    # Test with mixed valid and invalid terms
    with pytest.raises(ValueError, match="not found in FASTPT"):
        handler.get("P_E", "nonexistent_term", P=P, P_window=P_window, C_window=C_window)
    
    # Test parameter validation - P length must match k length
    with pytest.raises(ValueError):
        handler.get("P_E", P=np.ones(10))
    
    # Test parameter validation - C_window must be between 0 and 1
    with pytest.raises(ValueError):
        handler.get("P_E", P=P, C_window=1.5)
    
    # Test with overridden default parameters
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    new_P = P * 1.1
    result1 = handler.get("P_E")
    result2 = handler.get("P_E", P=new_P)
    assert not np.array_equal(result1, result2)

################# BULK RUN TESTS #################
def test_bulk_run_basic(fpt):
    """Test basic functionality of bulk_run method"""
    handler = FPTHandler(fpt)
    funcs = ['one_loop_dd', 'IA_tt']
    power_spectra = [P, P * 1.1, P * 1.2]
    
    results = handler.bulk_run(funcs, power_spectra, P_window=P_window, C_window=C_window)
    
    # Check that all expected results are present
    assert len(results) == len(funcs) * len(power_spectra)
    
    for func in funcs:
        for i in range(len(power_spectra)):
            assert (func, i) in results
            assert results[(func, i)] is not None

def test_bulk_run_results_correctness(fpt):
    """Test that bulk_run results match individual run calls"""
    handler = FPTHandler(fpt, P_window=P_window, C_window=C_window)
    funcs = ['one_loop_dd', 'IA_tt']
    power_spectra = [P, P * 1.5]
    
    bulk_results = handler.bulk_run(funcs, power_spectra)
    
    # Compare with individual runs
    for func in funcs:
        for i, spec in enumerate(power_spectra):
            individual_result = handler.run(func, P=spec)
            bulk_result = bulk_results[(func, i)]
            
            if isinstance(individual_result, tuple):
                assert isinstance(bulk_result, tuple)
                assert len(individual_result) == len(bulk_result)
                for ir, br in zip(individual_result, bulk_result):
                    assert np.array_equal(ir, br)
            else:
                assert np.array_equal(individual_result, bulk_result)

def test_bulk_run_with_overrides(fpt):
    """Test bulk_run with additional override parameters"""
    handler = FPTHandler(fpt)
    funcs = ['RSD_components']
    power_spectra = [P]
    
    # RSD_components requires 'f' parameter
    results = handler.bulk_run(funcs, power_spectra, 
                               P_window=P_window, C_window=C_window, f=0.5)
    
    assert (funcs[0], 0) in results
    assert results[(funcs[0], 0)] is not None

def test_bulk_run_empty_inputs(fpt):
    """Test bulk_run with empty function list or power spectra list"""
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    
    # Empty function list
    empty_results = handler.bulk_run([], [P])
    assert len(empty_results) == 0
    
    # Empty power spectra list
    empty_results = handler.bulk_run(['one_loop_dd'], [])
    assert len(empty_results) == 0

def test_bulk_run_with_caching(fpt):
    """Test that bulk_run properly uses caching"""
    handler = FPTHandler(fpt, do_cache=True)
    funcs = ['one_loop_dd']
    power_spectra = [P]
    
    # First run should compute
    handler.bulk_run(funcs, power_spectra, P_window=P_window, C_window=C_window)
    cache_size = len(handler.cache)
    
    # Second run should use cache
    handler.bulk_run(funcs, power_spectra, P_window=P_window, C_window=C_window)
    assert len(handler.cache) == cache_size
    
    # Different power spectrum should create new cache entry
    handler.bulk_run(funcs, [P * 1.1], P_window=P_window, C_window=C_window)
    assert len(handler.cache) > cache_size

def test_bulk_run_with_invalid_function(fpt):
    """Test bulk_run with invalid function name"""
    handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    
    with pytest.raises(ValueError, match="Function 'invalid_function' not found in FASTPT"):
        handler.bulk_run(['invalid_function'], [P])

def test_bulk_run_missing_params(fpt):
    """Test bulk_run with missing required parameters"""
    handler = FPTHandler(fpt)
    
    with pytest.raises(ValueError, match="Missing required parameters"):
        handler.bulk_run(['RSD_components'], [P], P_window=P_window, C_window=C_window)
        # Missing 'f' parameter for RSD_components

def test_bulk_run_large_input(fpt):
    """Test bulk_run with larger number of functions and spectra"""
    handler = FPTHandler(fpt)
    funcs = ['one_loop_dd', 'IA_tt', 'IA_mix', 'OV', 'kPol']
    power_spectra = [P, P * 1.1, P * 1.2, P * 1.3, P * 1.4]
    
    results = handler.bulk_run(funcs, power_spectra, P_window=P_window, C_window=C_window)
    
    # Check we got all expected combinations
    assert len(results) == len(funcs) * len(power_spectra)
    for func in funcs:
        for i in range(len(power_spectra)):
            assert (func, i) in results

def test_bulk_run_with_save_all(fpt):
    """Test bulk_run with save_all flag"""
    # Create a temporary outputs directory to avoid cluttering
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mock save_output that just records calls rather than saving files
        saved_outputs = []
        
        original_save_output = FPTHandler.save_output
        try:
            def mock_save_output(self, result, func_name, type="txt", output_dir=None):
                saved_outputs.append((result, func_name, type, output_dir))
                
            FPTHandler.save_output = mock_save_output
            
            handler = FPTHandler(fpt, save_all="txt", P_window=P_window, C_window=C_window, save_dir=temp_dir)
            funcs = ['one_loop_dd', 'IA_tt']
            power_spectra = [P, P * 1.1]
            
            handler.bulk_run(funcs, power_spectra)
            
            # Check that save_output was called for each function and power spectrum
            assert len(saved_outputs) == len(funcs) * len(power_spectra)
            
            # Verify the correct arguments were passed to save_output
            for output in saved_outputs:
                assert isinstance(output[0], tuple)  # Result
                assert output[1] in funcs  # Function name
                assert output[2] == "txt"  # File type
                assert output[3] == temp_dir  # Output directory
        finally:
            # Restore original method
            FPTHandler.save_output = original_save_output

################# OUTPUT FILE TESTS #################
def test_save_output_file_types(fpt):
    """Test saving output in different file formats"""
    import os
    import tempfile
    import json
    import csv
    import numpy as np
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
        
        # Test txt format
        txt_result = handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        txt_file = os.path.join(temp_dir, "one_loop_dd_output.txt")
        assert os.path.exists(txt_file)
        loaded_txt = np.loadtxt(txt_file)
        # Check that the saved data matches the result (accounting for transpose in saving)
        assert np.allclose(loaded_txt, np.transpose(txt_result))
        
        # Test csv format
        csv_result = handler.run('IA_tt', save_type="csv", save_dir=temp_dir)
        csv_file = os.path.join(temp_dir, "IA_tt_output.csv")
        assert os.path.exists(csv_file)
        # Read CSV file and check header and data
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            assert header == ['IA_tt_0', 'IA_tt_1']  # Header should contain function name with indices
            data = list(csv_reader)
            assert len(data) > 0
        
        # Test json format
        json_result = handler.run('one_loop_dd_bias', save_type="json", save_dir=temp_dir)
        json_file = os.path.join(temp_dir, "one_loop_dd_bias_output.json")
        assert os.path.exists(json_file)
        # Read JSON file and check structure
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            assert 'one_loop_dd_bias' in json_data
            # Check that the saved data has the same structure (number of arrays)
            if isinstance(json_result, tuple):
                assert len(json_data['one_loop_dd_bias']) == len(json_result)

def test_save_output_auto_naming(fpt):
    """Test automatic filename adjustment when files already exist"""
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
        
        # First run - should create base filename
        handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        base_file = os.path.join(temp_dir, "one_loop_dd_output.txt")
        assert os.path.exists(base_file)
        
        # Second run - should create filename with counter
        handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        counter_file = os.path.join(temp_dir, "one_loop_dd_1_output.txt")
        assert os.path.exists(counter_file)
        
        # Third run - should increment counter
        handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        counter_file2 = os.path.join(temp_dir, "one_loop_dd_2_output.txt")
        assert os.path.exists(counter_file2)
        
        # Different function should still use base filename
        handler.run('IA_tt', save_type="txt", save_dir=temp_dir)
        different_file = os.path.join(temp_dir, "IA_tt_output.txt")
        assert os.path.exists(different_file)

def test_output_directory_precedence(fpt):
    """Test precedence of different output directory parameters"""
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as tempdir1:
        with tempfile.TemporaryDirectory() as tempdir2:
            with tempfile.TemporaryDirectory() as tempdir3:
                # Initialize with default output directory
                handler = FPTHandler(fpt, save_dir=tempdir1, P=P, P_window=P_window, C_window=C_window)
                
                # 1. Test initializer-specified directory
                handler.run('one_loop_dd', save_type="txt")
                assert os.path.exists(os.path.join(tempdir1, "one_loop_dd_output.txt"))
                
                # 2. Test save_dir parameter overrides initializer
                handler.run('IA_tt', save_type="txt", save_dir=tempdir2)
                assert os.path.exists(os.path.join(tempdir2, "IA_tt_output.txt"))
                assert not os.path.exists(os.path.join(tempdir1, "IA_tt_output.txt"))

def test_invalid_file_type(fpt):
    """Test that invalid file types raise appropriate errors"""
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
        
        with pytest.raises(ValueError, match="Invalid file type"):
            handler.run('one_loop_dd', save_type="invalid", save_dir=temp_dir)

def test_save_output_direct_call(fpt):
    """Test direct calls to save_output method"""
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
        
        # Run function to get a result
        result = handler.run('one_loop_dd')
        
        # Call save_output directly with different file types
        handler.save_output(result, 'one_loop_dd', type="txt", output_dir=temp_dir)
        handler.save_output(result, 'one_loop_dd', type="csv", output_dir=temp_dir)
        handler.save_output(result, 'one_loop_dd', type="json", output_dir=temp_dir)
        
        # Check that all three files exist
        assert os.path.exists(os.path.join(temp_dir, "one_loop_dd_output.txt"))
        assert os.path.exists(os.path.join(temp_dir, "one_loop_dd_output.csv"))
        assert os.path.exists(os.path.join(temp_dir, "one_loop_dd_output.json"))

def test_bulk_run_with_save_options(fpt):
    """Test bulk_run with file saving options"""
    import os
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = FPTHandler(fpt, P_window=P_window, C_window=C_window)
        funcs = ['one_loop_dd', 'IA_tt']
        power_spectra = [P, P * 1.1]
        
        # Test bulk_run with save_type
        handler.bulk_run(funcs, power_spectra, save_type="txt", save_dir=temp_dir)
        
        # Check that files were created for each function and power spectrum
        for func in funcs:
            for i in range(len(power_spectra)):
                if i == 0:
                    base_path = os.path.join(temp_dir, f"{func}_output.txt")
                else:
                    base_path = os.path.join(temp_dir, f"{func}_{i}_output.txt")
                assert os.path.exists(base_path)

def test_load_method(fpt):
    """Test the load method for loading saved output files of different types"""
    import tempfile
    import numpy as np
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
        
        # Test txt format
        txt_result = handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        txt_loaded = handler.load('one_loop_dd_output.txt', load_dir=temp_dir)
        assert isinstance(txt_loaded, tuple)
        assert len(txt_loaded) == len(txt_result)
        for r1, r2 in zip(txt_result, txt_loaded):
            assert np.allclose(r1, r2)
        
        # Test csv format
        csv_result = handler.run('IA_tt', save_type="csv", save_dir=temp_dir)
        csv_loaded = handler.load('IA_tt_output.csv', load_dir=temp_dir)
        assert isinstance(csv_loaded, tuple)
        assert len(csv_loaded) == len(csv_result)
        for r1, r2 in zip(csv_result, csv_loaded):
            assert np.allclose(r1, r2)
        
        # Test json format
        json_result = handler.run('one_loop_dd_bias', save_type="json", save_dir=temp_dir)
        json_loaded = handler.load('one_loop_dd_bias_output.json', load_dir=temp_dir)
        assert isinstance(json_loaded, tuple)
        assert len(json_loaded) == len(json_result)
        for r1, r2 in zip(json_result, json_loaded):
            assert np.allclose(r1, r2), f"r1: {r1}, r2: {r2}"

def test_load_with_absolute_path(fpt):
    """Test loading files using absolute paths"""
    import os
    import tempfile
    import numpy as np
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
        
        # Save a result
        result = handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        file_path = os.path.join(temp_dir, "one_loop_dd_output.txt")
        
        # Load using absolute path - should ignore load_dir
        loaded_result = handler.load(file_path)
        assert isinstance(loaded_result, tuple)
        for r1, r2 in zip(result, loaded_result):
            assert np.allclose(r1, r2)
        
        # Load using absolute path but with a different load_dir (should ignore load_dir)
        with tempfile.TemporaryDirectory() as another_dir:
            loaded_result2 = handler.load(file_path, load_dir=another_dir)
            assert isinstance(loaded_result2, tuple)
            for r1, r2 in zip(result, loaded_result2):
                assert np.allclose(r1, r2)

def test_load_with_default_dir(fpt):
    """Test loading from the default output directory"""
    import tempfile
    import numpy as np
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize handler with custom output directory
        handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window, save_dir=temp_dir)
        
        # Save a result to the default directory
        result = handler.run('one_loop_dd', save_type="txt")
        
        # Load without specifying a directory (should use default)
        loaded_result = handler.load('one_loop_dd_output.txt')
        assert isinstance(loaded_result, tuple)
        for r1, r2 in zip(result, loaded_result):
            assert np.allclose(r1, r2)

def test_load_error_handling(fpt):
    """Test error handling in the load method"""
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
        
        # Test file not found
        with pytest.raises(FileNotFoundError):
            handler.load('nonexistent_file.txt', load_dir=temp_dir)
        
        # Test unsupported file type
        handler.run('one_loop_dd', save_type="txt", save_dir=temp_dir)
        # Create a file with unsupported extension by renaming
        import os
        import shutil
        src = os.path.join(temp_dir, "one_loop_dd_output.txt")
        dst = os.path.join(temp_dir, "one_loop_dd_output.xyz")
        shutil.copy(src, dst)
        
        with pytest.raises(FileNotFoundError):
            handler.load('one_loop_dd_output.xyz', load_dir=temp_dir)

def test_load_consistency_across_types(fpt):
    """Test that loading from different file types produces consistent results"""
    import tempfile
    import numpy as np
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
        
        # Save the same result in different formats
        result = handler.run('one_loop_dd')
        handler.save_output(result, 'one_loop_dd', type="txt", output_dir=temp_dir)
        handler.save_output(result, 'one_loop_dd', type="csv", output_dir=temp_dir)
        handler.save_output(result, 'one_loop_dd', type="json", output_dir=temp_dir)
        
        # Load from each format
        txt_loaded = handler.load('one_loop_dd_output.txt', load_dir=temp_dir)
        csv_loaded = handler.load('one_loop_dd_output.csv', load_dir=temp_dir)
        json_loaded = handler.load('one_loop_dd_output.json', load_dir=temp_dir)
        
        # Check that all loaded results match the original
        assert isinstance(txt_loaded, tuple) and isinstance(csv_loaded, tuple) and isinstance(json_loaded, tuple)
        assert len(txt_loaded) == len(csv_loaded) == len(json_loaded) == len(result)
        
        # Compare all formats against the original result
        for i in range(len(result)):
            assert np.allclose(result[i], txt_loaded[i])
            assert np.allclose(result[i], csv_loaded[i])
            assert np.allclose(result[i], json_loaded[i])

def test_load_with_relative_paths(fpt):
    """Test loading using relative paths"""
    import tempfile
    import os
    import numpy as np
    
    # Get current working directory
    orig_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Change to the temporary directory
            os.chdir(temp_dir)
            
            # Create a subdirectory for outputs
            os.makedirs('outputs', exist_ok=True)
            
            handler = FPTHandler(fpt, P=P, P_window=P_window, C_window=C_window)
            
            # Save a result
            result = handler.run('one_loop_dd', save_type="txt", save_dir='outputs')
            
            # Load using relative path
            loaded_result = handler.load(os.path.join('outputs', 'one_loop_dd_output.txt'))
            
            assert isinstance(loaded_result, tuple)
            for r1, r2 in zip(result, loaded_result):
                assert np.allclose(r1, r2)
        finally:
            # Make sure to return to original directory
            os.chdir(orig_cwd)