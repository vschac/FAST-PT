from functools import lru_cache
import numpy as np
import inspect

class FPT:
    def __init__(self, k):
        self.k = k
    """Regular init code here"""

    def one_loop_dd(self, P, P_window=None, C_window=None):
        """Method code here"""
        return P
    
    def IA_ct(self, P, P_window=None, C_window=None):
        """Method code here"""
        return P
    
    def RSD_components(self, P, f, P_window=None, C_window=None):
        """Method code here"""
        if f is None: raise ValueError("f must be provided.")
        return P
    




class FunctionHandler:
    def __init__(self, fastpt_instance, **params):
        self.fastpt = fastpt_instance
        self.default_params = self._validate_params(params)  # Validate once and store
        self.cache = {}  # Custom cache for results


    def _validate_params(self, params):
        """"Same function as before"""
        #Would need to add checks for every possible parameter (f, nu, X, etc)
        #Unless we want to make it so only P, P_window, C_window passable at init
        return params


    def _get_function_params(self, func):
        """
        Returns the required parameter names for a given FASTPT function.
        """
        signature = inspect.signature(func)
        return [param for param in signature.parameters if signature.parameters[param].default == inspect.Parameter.empty]


    def _cache_result(self, function_name, params, result):
        """
        Stores results uniquely by function name and its specific parameters.
        """
        # Convert any numpy arrays to tuples in the params
        hashable_params = []
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                hashable_params.append((k, tuple(v.flat)))
            else:
                hashable_params.append((k, v))
        param_tuple = tuple(sorted(hashable_params))
        self.cache[(function_name, param_tuple)] = result
        return param_tuple #Returns 


    def run(self, function_name, *args, **override_params):
        """
        Runs the selected function from FASTPT with validated parameters.

        :param function_name: Name of the FASTPT method to execute.
        :param args: Positional arguments (if applicable).
        :param override_params: Optional overrides for function-specific parameters.
        :return: The result of the method call.
        """
        if not hasattr(self.fastpt, function_name):
            raise ValueError(f"Function '{function_name}' not found in FASTPT.")

        func = getattr(self.fastpt, function_name)
        required_params = self._get_function_params(func)

        # Merge stored parameters with user overrides
        merged_params = {**self.default_params, **override_params}

        # Ensure all required parameters are provided
        missing_params = [p for p in required_params if p not in merged_params]
        if missing_params:
            raise ValueError(f"Missing required parameters for '{function_name}': {missing_params}. "
                             f"Please recall with the missing parameters.")

        # Ensure only relevant parameters are passed
        params_to_pass = {k: merged_params[k] for k in required_params}

        # Check cache first
        #####################   Review this part   #####################
        hashable_params = []
        for k, v in params_to_pass.items():
            if isinstance(v, np.ndarray):
                hashable_params.append((k, tuple(v.flat)))
            else:
                hashable_params.append((k, v))
        param_tuple = tuple(sorted(hashable_params))
    
        if (function_name, param_tuple) in self.cache:
            print(f"Using cached result for {function_name} with parameters {params_to_pass}.")
            return self.cache[(function_name, param_tuple)]

        # Compute the result and cache it
        result = func(*args, **params_to_pass)
        self._cache_result(function_name, params_to_pass, result)
        return result
    

    def clear_cache(self, function_name=None):
        """
        Clears specific or all cached results.
        :param function_name: If specified, clears cache for this function only.
        """
        if function_name:
            self.cache = {key: value for key, value in self.cache.items() if key[0] != function_name}
            print(f"Cache cleared for '{function_name}'.")
        else:
            self.cache.clear()
            print("Cache cleared for all functions.")


    def list_available_functions(self):
        """ Returns a list of valid FASTPT functions. """
        return [f for f in dir(self.fastpt) if callable(getattr(self.fastpt, f)) and not f.startswith("__")]


if __name__ == "__main__":
    """
        - Can pass in parametes at FuncionHandler definition to be pre validated
        - If parameters are unrecognized (or not passed during init), need to be revalidated
        - Function result with those specific parameters is cached
        - TODO: Can clear cache for specific functions or all functions
        - Can list available functions
        - Passing any parameters on the run call will override any stored parameters,
            not passing any required parameters will default to what is stored
    """

    k = np.logspace(-3, 1, 200)
    fpt = FPT(k)

    handler = FunctionHandler(fpt, P=np.array([1.0, 2.0, 3.0]), P_window=(0.1, 0.2), C_window=0.75)
    result = handler.run("one_loop_dd")
    print(result)
    r2 = handler.run("one_loop_dd")

    r3 = handler.run("IA_ct", P=(4, 5, 6))
    print(r3)

    try:
        r4 = handler.run("RSD_components", P=(1, 3, 6))
    except ValueError as e:
        print(e)

    r5 = handler.run("RSD_components", f=0.5)
    print(r5)

    print(handler.list_available_functions())