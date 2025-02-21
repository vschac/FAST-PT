import pytest
import numpy as np
from fastpt import FASTPT
from fastpt.FPTHandler import FunctionHandler
import os

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
P = np.loadtxt(data_path)[:, 1]
C_window = 0.75
P_window = np.array([0.2, 0.2])

@pytest.fixture
def fpt():
    k = np.loadtxt(data_path)[:, 0]
    n_pad=int(0.5*len(k))
    to_do = ['skip'] #Currently needed as no todo delegates to fastpt simple which does not init the necessary properties for tensor/scalar stuff
    return FASTPT(k, to_do=to_do, low_extrap=-5, high_extrap=3, n_pad=n_pad)

################# FUNCTIONALITY TESTS #################
def test_init_with_valid_params(fpt):
    handler = FunctionHandler(fpt, P=P, P_window=P_window, C_window=C_window)
    assert isinstance(handler.fastpt, FASTPT)
    assert handler.default_params['P'] is not None
    assert handler.default_params['P_window'].all() == P_window.all()
    assert handler.default_params['C_window'] == 0.75

def test_run_without_power_spectrum(fpt):
    handler = FunctionHandler(fpt) #P is not required at handler init but must be passed at every function call
    with pytest.raises(ValueError, match="Missing required parameters for 'one_loop_dd': \['P'\]. Please recall with the missing parameters."):
        handler.run('one_loop_dd')

def test_init_with_zero_power_spectrum(fpt):
    P = np.zeros_like(fpt.k_original)
    with pytest.raises(ValueError, match='Your input power spectrum array is all zeros'):
        FunctionHandler(fpt, P=P)

def test_init_with_mismatched_arrays(fpt):
    P = np.ones(10)  # Wrong size
    with pytest.raises(ValueError, match='Input k and P arrays must have the same size'):
        FunctionHandler(fpt, P=P)

def test_invalid_c_window(fpt):
    with pytest.raises(ValueError, match='C_window must be between 0 and 1'):
        FunctionHandler(fpt, P=P, C_window=1.5)
    with pytest.raises(ValueError, match='C_window must be between 0 and 1'):
        FunctionHandler(fpt, P=P, C_window=-0.5)

def test_invalid_p_window(fpt):
    with pytest.raises(ValueError, match='P_window must be a tuple of two values'):
        FunctionHandler(fpt, P=P, P_window=np.array([1.0]))

def test_cache_functionality(fpt):
    handler = FunctionHandler(fpt, P=P)
    handler.run('one_loop_dd')  # First run
    cache_size_before = len(handler.cache)
    handler.run('one_loop_dd')  # Should use cache
    assert len(handler.cache) == cache_size_before
    handler.clear_cache()
    assert len(handler.cache) == 0

def test_cache_with_other_params(fpt):
    """Test caching behavior with various parameter combinations"""
    handler = FunctionHandler(fpt)
    
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
    handler = FunctionHandler(fpt, P=P)
    with pytest.raises(ValueError, match="Function 'nonexistent_function' not found"):
        handler.run('nonexistent_function')

def test_missing_required_params(fpt):
    handler = FunctionHandler(fpt, P=P)
    with pytest.raises(ValueError, match="Missing required parameters"):
        handler.run('RSD_components')

def test_clear_specific_cache(fpt):
    handler = FunctionHandler(fpt, P=P)
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
    handler = FunctionHandler(fpt, **default_params)
    
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
    handler = FunctionHandler(fpt)
    
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


################# BENCHMARK TESTS #################
def test_handler_function_equality(fpt):
    """Test that handler produces identical results to direct FASTPT function calls"""
    handler = FunctionHandler(fpt)
    
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