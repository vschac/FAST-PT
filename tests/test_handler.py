import pytest
import numpy as np
from fastpt import FASTPT
from fastpt.FPTHandler import FunctionHandler

import pytest
import numpy as np
from fastpt import FASTPT
import os
from fastpt.FPTHandler import FunctionHandler

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
P = np.loadtxt(data_path)[:, 1]
C_window = 0.75
P_window = np.array([0.2, 0.2])

handler = FunctionHandler(FASTPT(np.loadtxt(data_path)[:, 0], to_do=['skip']), P=P, P_window=P_window, C_window=C_window)
print(handler._get_function_params(handler.fastpt.one_loop_dd))

@pytest.fixture
def setup_fastpt():
    k = np.loadtxt(data_path)[:, 0]
    to_do = ['skip'] #Currently needed as no todo delegates to fastpt simple which does not init the necessary properties for tensor/scalar stuff
    return FASTPT(k, to_do=to_do)

def test_init_with_valid_params(setup_fastpt):
    handler = FunctionHandler(setup_fastpt, P=P, P_window=P_window, C_window=C_window)
    assert isinstance(handler.fastpt, FASTPT)
    assert handler.default_params['P'] is not None
    assert handler.default_params['P_window'].all() == P_window.all()
    assert handler.default_params['C_window'] == 0.75

def test_run_without_power_spectrum(setup_fastpt):
    handler = FunctionHandler(setup_fastpt) #P is not required at handler init but must be passed at every function call
    with pytest.raises(ValueError, match="Missing required parameters for 'one_loop_dd': \['P'\]. Please recall with the missing parameters."):
        handler.run('one_loop_dd')

def test_init_with_zero_power_spectrum(setup_fastpt):
    P = np.zeros_like(setup_fastpt.k_original)
    with pytest.raises(ValueError, match='Your input power spectrum array is all zeros'):
        FunctionHandler(setup_fastpt, P=P)

def test_init_with_mismatched_arrays(setup_fastpt):
    P = np.ones(10)  # Wrong size
    with pytest.raises(ValueError, match='Input k and P arrays must have the same size'):
        FunctionHandler(setup_fastpt, P=P)

def test_invalid_c_window(setup_fastpt):
    with pytest.raises(ValueError, match='C_window must be between 0 and 1'):
        FunctionHandler(setup_fastpt, P=P, C_window=1.5)
    with pytest.raises(ValueError, match='C_window must be between 0 and 1'):
        FunctionHandler(setup_fastpt, P=P, C_window=-0.5)

def test_invalid_p_window(setup_fastpt):
    with pytest.raises(ValueError, match='P_window must be a tuple of two values'):
        FunctionHandler(setup_fastpt, P=P, P_window=np.array([1.0]))

def test_cache_functionality(setup_fastpt):
    handler = FunctionHandler(setup_fastpt, P=P)
    handler.run('one_loop_dd')  # First run
    cache_size_before = len(handler.cache)
    handler.run('one_loop_dd')  # Should use cache
    assert len(handler.cache) == cache_size_before
    handler.clear_cache()
    assert len(handler.cache) == 0

def test_invalid_function_call(setup_fastpt):
    handler = FunctionHandler(setup_fastpt, P=P)
    with pytest.raises(ValueError, match="Function 'nonexistent_function' not found"):
        handler.run('nonexistent_function')

def test_missing_required_params(setup_fastpt):
    handler = FunctionHandler(setup_fastpt, P=P)
    with pytest.raises(ValueError, match="Missing required parameters"):
        handler.run('RSD_components')

def test_clear_specific_cache(setup_fastpt):
    handler = FunctionHandler(setup_fastpt, P=P)
    handler.run('one_loop_dd')
    handler.run('one_loop_dd_bias')
    handler.clear_cache('one_loop_dd')
    assert any('one_loop_dd_bias' in key[0] for key in handler.cache.keys())
    assert all(key[0] != 'one_loop_dd' for key in handler.cache.keys())

def test_all_fastpt_functions_with_handler_params(setup_fastpt):
    """Test FASTPT functions with parameters set during handler initialization"""
    # Initialize handler with all possible parameters
    default_params = {
        'P': P,
        'P_window': P_window,
        'C_window': C_window,
        'f': 0.5,
        'mu_n': 0.5
    }
    handler = FunctionHandler(setup_fastpt, **default_params)
    
    # Dictionary mapping functions to any additional required parameters
    func_names = (
        'one_loop_dd', 'one_loop_dd_bias', 'one_loop_dd_bias_b3nl',
        'one_loop_dd_bias_lpt_NL', 'IA_tt', 'IA_mix', 'IA_ta',
        'IA_der', 'IA_ct', 'IA_ctbias', 'IA_gb2', 'IA_d2',
        'IA_s2', 'OV', 'kPol', 'RSD_components',
        'RSD_ABsum_components', 'RSD_ABsum_mu'
    )
    
    for name in func_names:
        try:
            result = handler.run(name)
            assert result is not None, f"Function {name} returned None"
            if isinstance(result, tuple):
                assert all(r is not None for r in result), f"Function {name} returned None in tuple"
        except Exception as e:
            pytest.fail(f"Function {name} failed to run with error: {str(e)}")

def test_all_fastpt_functions_with_run_params(setup_fastpt):
    """Test FASTPT functions with parameters passed during run call"""
    handler = FunctionHandler(setup_fastpt)
    
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