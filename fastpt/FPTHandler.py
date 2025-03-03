import numpy as np
import inspect
from fastpt import FASTPT
from numpy import pi, log

class FPTHandler:
    def __init__(self, fastpt_instance: FASTPT, do_cache=False, max_cache_entries=500, **params):
        self.fastpt = fastpt_instance
        if not params or params is None: print("Warning: P is a required parameter for all functions, it will need to be passed on the run call.")
        self.default_params = self._validate_params(**params) if params else {}
        self.cache = {}
        #Explain somewhere that caching is an option though not necessarily needed
        self.do_cache = do_cache
        self.max_cache_entries = max_cache_entries

        #Commented out terms have not been implemented yet
        self.term_sources = {
            # "P_1loop": ("one_loop_dd", 0),
            # "Ps": ("one_loop_dd", 1),
            # "Pd1d2": ("one_loop_dd_bias", 2),  
            # "Pd2d2": ("one_loop_dd_bias", 3),
            # "Pd1s2": ("one_loop_dd_bias", 4),
            # "Pd2s2": ("one_loop_dd_bias", 5),
            # "Ps2s2": ("one_loop_dd_bias", 6),
            # "sig4": ("one_loop_dd_bias", 7),
        
            # "sig3nl": ("one_loop_dd_bias_b3nl", 8),
        
            # "Pb1L": ("one_loop_dd_bias_lpt_NL", 1),
            # "Pb1L_2": ("one_loop_dd_bias_lpt_NL", 2),
            # "Pb1L_b2L": ("one_loop_dd_bias_lpt_NL", 3),
            # "Pb2L": ("one_loop_dd_bias_lpt_NL", 4),
            # "Pb2L_2": ("one_loop_dd_bias_lpt_NL", 5),
        
            "P_E": ("IA_tt", "X_IA_E", lambda x: 2 * x),
            "P_B": ("IA_tt", "X_IA_B", lambda x: 2 * x),
        
            "P_A": ("IA_mix", "X_IA_A", lambda x: 2 * x),
            "P_Btype2": ("IA_mix", None),
            "P_DEE": ("IA_mix", "X_IA_DEE", lambda x: 2 * x),
            "P_DBB": ("IA_mix", "X_IA_DBB", lambda x: 2 * x),
        
            "P_deltaE1": ("IA_ta", "X_IA_deltaE1", lambda x: 2 * x),
            "P_deltaE2": ("IA_ta", None),
            "P_0E0E": ("IA_ta", "X_IA_0E0E", None),
            "P_0B0B": ("IA_ta", "X_IA_0B0B", None),
        
            "P_gb2sij": ("IA_gb2", "X_IA_gb2_F2", lambda x: 2 * x),
            "P_gb2dsij": ("IA_gb2", "X_IA_gb2_fe", lambda x: 2 * x),
            "P_gb2sij2": ("IA_gb2", "X_IA_gb2_he", lambda x: 2 * x),

            "P_der": ("IA_der", None),

            # "P_0tE": ("IA_ct", 0),
            # "P_0EtE": ("IA_ct", 1),
            # "P_E2tE": ("IA_ct", 2),
            # "P_tEtE": ("IA_ct", 3),
        
            "P_d2tE": ("IA_ctbias", ("X_IA_gb2_F2", "X_IA_gb2_G2"), lambda results: 2 * (results[1] - results[0])),
            "P_s2tE": ("IA_ctbias", ("X_IA_gb2_S2F2", "X_IA_gb2_S2G2"), lambda results: 2 * (results[1] - results[0])),
        
            "P_s2E": ("IA_s2", "X_IA_gb2_S2F2", lambda x: 2 * x),
            "P_s20E": ("IA_s2", "X_IA_gb2_S2fe", lambda x: 2 * x),
            "P_s2E2": ("IA_s2", "X_IA_gb2_S2he", lambda x: 2 * x),
        
            "P_d2E": ("IA_d2", "X_IA_gb2_F2", lambda x: 2 * x),
            "P_d20E": ("IA_d2", "X_IA_gb2_he", lambda x: 2 * x),
            "P_d2E2": ("IA_d2", "X_IA_gb2_fe", lambda x: 2 * x),
        
            "P_OV": ("OV", None),
        
            "P_kP1": ("kPol", "X_kP1", lambda x: x / (80 * pi ** 2)),
            "P_kP2": ("kPol", "X_kP2", lambda x: x / (160 * pi ** 2)),
            "P_kP3": ("kPol", "X_kP3",lambda x: x / (80 * pi ** 2)),
        
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
        if len(self.cache) >= self.max_cache_entries:
            print("Max cache size reached. Removing oldest entry.")
            self.cache.pop(next(iter(self.cache)))
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
    
    def _prepare_function_params(self, func, override_kwargs):
        """Prepares and validates parameters for a FASTPT function."""
        if override_kwargs: 
            self._validate_params(**override_kwargs)
    
        merged_params = {**self.default_params, **override_kwargs}

        params_info = self._get_function_params(func)
        missing_params = [p for p in params_info['required'] if p not in merged_params]
    
        if missing_params:
            raise ValueError(f"Missing required parameters for '{func.__name__}': {missing_params}. "
                        f"Please recall with the missing parameters.")

        # Return only the params the function actually needs
        passing_params = {k: v for k, v in merged_params.items() if k in params_info['all']}
        return passing_params, params_info


    def run(self, function_name, **override_kwargs):
        """Runs the selected function from FASTPT with validated parameters."""
        if not hasattr(self.fastpt, function_name):
            raise ValueError(f"Function '{function_name}' not found in FASTPT.")

        func = getattr(self.fastpt, function_name)
        passing_params, _ = self._prepare_function_params(func, override_kwargs)
        
        if self.do_cache:
            cache_key = self._convert_to_hashable(passing_params)
            if cache_key in self.cache:
                print(f"Using cached result for {function_name}")
                return self.cache[cache_key]

        result = func(**passing_params)
        if self.do_cache: self._cache_result(function_name, passing_params, result)
        return result
    
    #Not saving any time by caching this function because the individual terms
    #are already cached in FASTPT
    def get(self, *terms, **override_kwargs):
        """Allows for quick access to a specific term or terms from a FASTPT function."""
        if not terms:
            raise ValueError("At least one term must be provided.")
        output = {}
        for term in terms:
            if term not in self.term_sources:
                raise ValueError(f"Term '{term}' not found in FASTPT.")
            if term in ("P_Btype2", "P_deltaE2", "P_der", "P_OV"): #Terms that have their own unique functions
                exceptions = {
                    "P_Btype2": "get_P_Btype2",
                    "P_deltaE2": "get_P_deltaE2",
                    "P_der": "IA_der",
                    "P_OV": "OV"
                }
                func_name = exceptions[term]
                func = getattr(self.fastpt, func_name)
            
                passing_params, _ = self._prepare_function_params(func, override_kwargs)
                result = func(**passing_params)
            else:
                func_name = self.term_sources[term][0]
                func = getattr(self.fastpt, func_name)
                passing_params, params_info = self._prepare_function_params(func, override_kwargs)

                compute_func = getattr(self.fastpt, "compute_term")

                X_source = self.term_sources[term][1]
                operation = self.term_sources[term][2]

                # Handle case where we need multiple X terms (like for ctbias)
                if isinstance(X_source, tuple):
                    X_names = X_source
                    X_terms = []
                    for name in X_names:
                        if name in dir(self.fastpt):
                            X_terms.append(getattr(self.fastpt, name))
                        else:
                            raise AttributeError(f"'{name}' not found in FASTPT")
                    result = compute_func(term, tuple(X_terms), operation=operation, **passing_params)
                else:
                    # Standard case with a single X tracer
                    X_term = getattr(self.fastpt, X_source)
                    result = compute_func(term, X_term, operation=operation, **passing_params)                
                
            output[term] = result

        # If only one term was requested, return just that value
        if len(output) == 1 and len(terms) == 1:
            return output[list(output.keys())[0]]
        return output

    def clear_cache(self, function_name=None):
        """ Clears specific or all cached results. """
        if function_name:
            self.cache = {key: value for key, value in self.cache.items() if key[0] != function_name}
            print(f"Cache cleared for '{function_name}'.")
        else:
            self.cache.clear()
            print("Cache cleared for all functions.")

    def show_cache_info(self):
        """Display cache information"""
        num_entries = len(self.cache)
        print({
            "num_entries": num_entries,
            "max_entries": self.max_cache_entries,
            "usage_percent": (num_entries / self.max_cache_entries) * 100 if self.max_cache_entries > 0 else 0
        })


    def list_available_functions(self):
        """ Returns a list of valid FASTPT functions. """
        print([f for f in dir(self.fastpt) if callable(getattr(self.fastpt, f)) and not f.startswith("__")])

    def list_available_terms(self):
        """List all available power spectrum terms that can be requested via get()"""
    
        # Organize by function
        organized = {}
        for term, (func, _) in self.term_sources.items():
            if func not in organized:
                organized[func] = []
            organized[func].append(term)
        
        # Print in a nice format
        print("Available terms by function:")
        for func, terms in organized.items():
            print(f"\n{func}:")
            terms_str = ", ".join(sorted(terms))
            print(f"  {terms_str}")
        
        # Special parameter requirements
        special_params = {
            "RSD_components": ["f"],
            "RSD_ABsum_components": ["f"],
            "RSD_ABsum_mu": ["f", "mu_n"],
            "IRres": ["L", "h", "rsdrag"]
        }
    
        print("\nSpecial parameter requirements:")
        for func, params in special_params.items():
            print(f"{func}: requires {', '.join(params)}")
        
        return organized
    
    def clear_default_params(self):
        self.default_params = {}
        print("Cache cleared for all functions.")

    def update_default_params(self, **params):
        self.default_params = self._validate_params(**params)
        print("Default parameters updated.")

    def update_fastpt_instance(self, fastpt_instance: FASTPT):
        self.fastpt = fastpt_instance
        self.clear_cache()
        print("FASTPT instance updated. Cached cleared.")