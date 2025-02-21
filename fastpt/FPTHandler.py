import numpy as np
from numpy import log
from pprint import pprint
import inspect
from fastpt import FASTPT

class FunctionHandler:
    def __init__(self, fastpt_instance: FASTPT, **params):
        self.fastpt = fastpt_instance
        if not params or params is None: print("Warning: P is a required parameter for all functions, it will need to be passed on the run call.")
        self.default_params = self._validate_params(**params) if params else {}
        self.cache = {}
 

    def _validate_params(self, **params):
        """" Same function as before """
        #Would need to add checks for every possible parameter (f, nu, X, etc)
        valid_params = ('P', 'P_window', 'C_window', 'f', 'X', 'nu', 'mu_n', 'L', 'h', 'rsdrag')
        for key in params.keys():
            if key not in valid_params:
                raise ValueError(f'Invalid parameter: {key}. Valid parameters are: {valid_params}')
        P = params.get('P', None)
        if (P is None or len(P) == 0):
            raise ValueError('You must provide an input power spectrum array.')
        if (len(P) != len(self.fastpt.k_original)):
            raise ValueError(f'Input k and P arrays must have the same size. P:{len(P)}, K:{len(self.fastpt.k_original)}')
            
        if (np.all(P == 0.0)):
            raise ValueError('Your input power spectrum array is all zeros.')

        P_window = params.get('P_window', np.array([]))
        C_window = params.get('C_window', None)

        if P_window is not None and P_window.size > 0:
            maxP = (log(self.fastpt.k_original[-1]) - log(self.fastpt.k_original[0])) / 2
            if len(P_window) != 2:
                raise ValueError(f'P_window must be a tuple of two values.')
            if P_window[0] > maxP or P_window[1] > maxP:
                raise ValueError(f'P_window value is too large. Decrease to less than {maxP} to avoid over tapering.')

        if C_window is not None:
            if C_window < 0 or C_window > 1:
                raise ValueError('C_window must be between 0 and 1.')

        return params


    def _get_function_params(self, func):
        """ Returns both required and optional parameter names for a given FASTPT function. """
        signature = inspect.signature(func)
        required_params = []
        optional_params = []
    
        for param_name, param in signature.parameters.items():
            # Skip self parameter and *args/**kwargs
            if (param_name == 'self' or 
                param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)):
                continue
            
            # Check if parameter is required or optional
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
            else:
                optional_params.append(param_name)
    
        return {
            'required': required_params,
            'optional': optional_params,
            'all': required_params + optional_params
        }


    def _cache_result(self, function_name, params, result):
        """ Stores results uniquely by function name and its specific parameters. """
        hashable_params = self._convert_to_hashable(params)
        self.cache[(function_name, hashable_params)] = result
    

    def _convert_to_hashable(self, params):
        """Convert parameters to hashable format, handling numpy arrays specially"""
        hashable_params = []
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                hashable_params.append((k, hash(v.tobytes())))
            else:
                hashable_params.append((k, v))
        return tuple(sorted(hashable_params))


    def run(self, function_name, **override_kwargs):
        """Runs the selected function from FASTPT with validated parameters."""
        if not hasattr(self.fastpt, function_name):
            raise ValueError(f"Function '{function_name}' not found in FASTPT.")

        func = getattr(self.fastpt, function_name)
        params_info = self._get_function_params(func)

        if (override_kwargs): 
            self._validate_params(**override_kwargs)
        merged_params = {**self.default_params, **override_kwargs}

        missing_params = [p for p in params_info['required'] if p not in merged_params]
        if missing_params:
            raise ValueError(f"Missing required parameters for '{function_name}': {missing_params}. "
                         f"Please recall with the missing parameters.")
    
        # Remove unneeded default params
        passing_params = {k: v for k, v in merged_params.items() if k in params_info['all']}

        # Convert parameters to hashable format for cache key
        cache_key = self._convert_to_hashable(passing_params)
        if cache_key in self.cache:
            print(f"Using cached result for {function_name}")
            return self.cache[cache_key]

        result = func(**passing_params)
        self._cache_result(function_name, passing_params, result)
        return result
    

    def clear_cache(self, function_name=None):
        """ Clears specific or all cached results. """
        if function_name:
            self.cache = {key: value for key, value in self.cache.items() if key[0] != function_name}
            print(f"Cache cleared for '{function_name}'.")
        else:
            self.cache.clear()
            print("Cache cleared for all functions.")

    def show_cache(self):
        """Display cache using pprint"""
        pprint(self.cache)


    def list_available_functions(self):
        """ Returns a list of valid FASTPT functions. """
        print([f for f in dir(self.fastpt) if callable(getattr(self.fastpt, f)) and not f.startswith("__")])

    
    def clear_default_params(self):
        self.default_params = {}
        print("Cache cleared for all functions.")

    def update_default_params(self, **params):
        self.default_params = self._validate_params(**params)
        print("Default parameters updated.")