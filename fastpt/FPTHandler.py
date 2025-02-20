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
        #Unless we want to make it so only P, P_window, C_window passable at init
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
        """ Returns the required parameter names for a given FASTPT function. """
        signature = inspect.signature(func)
        # Filter out self, *args and **kwargs from required parameters
        return [param for param in signature.parameters if signature.parameters[param].default == inspect.Parameter.empty]
        # return [
        #     param for param in signature.parameters 
        #     if (signature.parameters[param].default == inspect.Parameter.empty
        #         and signature.parameters[param].kind not in (
        #             inspect.Parameter.VAR_POSITIONAL,
        #             inspect.Parameter.VAR_KEYWORD
        #         )
        #         and param != 'self'  # Exclude the 'self' parameter
        #     )
        # ]


    def _cache_result(self, function_name, params, result):
        """ Stores results uniquely by function name and its specific parameters. """
        hashable_params = self._convert_to_hashable(params)
        self.cache[(function_name, hashable_params)] = result
    

    def _convert_to_hashable(self, params):
        hashable_params = []
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                hashable_params.append((k, tuple(v.flat)))
            else:
                hashable_params.append((k, v))
        return tuple(sorted(hashable_params))


    def run(self, function_name, *override_args, **override_kwargs):
        """ Runs the selected function from FASTPT with validated parameters. """
        if not hasattr(self.fastpt, function_name):
            raise ValueError(f"Function '{function_name}' not found in FASTPT.")

        func = getattr(self.fastpt, function_name)
        required_params = self._get_function_params(func)

        if (override_kwargs): self._validate_params(override_kwargs)
        if (override_args): 
            self._validate_params(override_args)
            #Convert positional args to a dict for merging (Python does this already with kwargs)
            args_dict = dict(zip(required_params[:len(override_args)], override_args))
            merged_params = {**self.default_params, **args_dict, **override_kwargs}
        else:
            merged_params = {**self.default_params, **override_kwargs}

        missing_params = [p for p in required_params if p not in merged_params]
        if missing_params:
            raise ValueError(f"Missing required parameters for '{function_name}': {missing_params}. "
                             f"Please recall with the missing parameters.")

        # Check cache first
        param_tuple = self._convert_to_hashable(merged_params)
        if (function_name, param_tuple) in self.cache:
            print(f"Using cached result for {function_name} with parameters {merged_params}.")
            return self.cache[(function_name, param_tuple)]

        result = func(**merged_params)
        self._cache_result(function_name, merged_params, result)
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