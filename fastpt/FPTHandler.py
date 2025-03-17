import numpy as np
import inspect
from fastpt import FASTPT
from numpy import pi, log
import os

class FPTHandler:
    def __init__(self, fastpt_instance: FASTPT, do_cache=False, save_all=None, save_dir=None, max_cache_entries=500, **params):
        self.__fastpt = fastpt_instance
        self.cache = {}
        self.do_cache = do_cache
        self.max_cache_entries = max_cache_entries
        self.save_all = save_all
        
        # Set default output directory if none specified
        if save_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        else:
            self.output_dir = save_dir
            
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.default_params = {}
        if params:
            try:
                self.default_params = self._validate_params(**params)
            except ValueError as e:
                if "You must provide an input power spectrum array" in str(e):
                    print("No power spectrum provided. You'll need to provide 'P' in each function call.")
                    self.default_params = params
                else:
                    raise e

        #Commented out terms have not been implemented yet
        self.term_sources = {
            "P_1loop": ("one_loop_dd", None),
            "Ps": ("one_loop_dd", None),
            "Pd1d2": ("one_loop_dd_bias", None),  
            "Pd2d2": ("one_loop_dd_bias", None),
            "Pd1s2": ("one_loop_dd_bias", None),
            "Pd2s2": ("one_loop_dd_bias", None),
            "Ps2s2": ("one_loop_dd_bias", None),
            "sig4": ("one_loop_dd_bias", None),
        
            "sig3nl": ("one_loop_dd_bias_b3nl", None),
        
            "Pb1L": ("one_loop_dd_bias_lpt_NL", None),
            "Pb1L_2": ("one_loop_dd_bias_lpt_NL", None),
            "Pb1L_b2L": ("one_loop_dd_bias_lpt_NL", None),
            "Pb2L": ("one_loop_dd_bias_lpt_NL", None),
            "Pb2L_2": ("one_loop_dd_bias_lpt_NL", None),
        
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

            "P_0tE": ("IA_ct", None),
            "P_0EtE": ("IA_ct", None),
            "P_E2tE": ("IA_ct", None),
            "P_tEtE": ("IA_ct", None),
        
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
 
    @property
    def fastpt(self):
        return self.__fastpt

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


    def run(self, function_name, save_type=None, save_dir=None, **override_kwargs):
        """Runs the selected function from FASTPT with validated parameters.
        
        Args:
            function_name (str): Name of the FASTPT function to run
            save_type (str, optional): Type of file to save results as ('txt', 'csv', or 'json'). Defaults to None.
            save_dir (str, optional): Directory to save results in. Defaults to the class's output_dir.
            **override_kwargs: Additional parameters to pass to the FASTPT function
            
        Returns:
            Result from the FASTPT function call
        """
        if not hasattr(self.fastpt, function_name):
            raise ValueError(f"Function '{function_name}' not found in FASTPT.")

        if 'save_type' in override_kwargs:
            save_param = override_kwargs.pop('save_type')
            if save_type is None:
                save_type = save_param
                
        if 'save_dir' in override_kwargs:
            save_dir = override_kwargs.pop('save_dir')

        output_directory = save_dir if save_dir is not None else self.output_dir

        func = getattr(self.fastpt, function_name)
        passing_params, _ = self._prepare_function_params(func, override_kwargs)
        
        if self.do_cache:
            cache_key = (function_name, self._convert_to_hashable(passing_params))
            if cache_key in self.cache:
                print(f"Using cached result for {function_name}")
                return self.cache[cache_key]

        result = func(**passing_params)
        if self.do_cache: 
            self._cache_result(function_name, passing_params, result)
        if save_type is not None: 
            self.save_output(result, function_name, type=save_type, output_dir=output_directory)
        elif self.save_all is not None: 
            self.save_output(result, function_name, type=self.save_all, output_dir=output_directory)
        return result
    
    def bulk_run(self, func_names, power_spectra, verbose=False, **override_kwargs):
        """
        Runs multiple functions with multiple power spectra.
        
        Args:
            func_names (list): List of FASTPT function names to call
            power_spectra (list): List of power spectra to use
            **override_kwargs: Additional parameters to pass to all function calls
        
        Returns:
            dict: Results keyed by (function_name, power_spectrum_index)
        """
        results = {}
        for func_name in func_names:
            for i, P in enumerate(power_spectra):
                # Combine override kwargs with the specific power spectrum
                params = {**self.default_params, **override_kwargs, 'P': P}
                if verbose: print(f"Running {func_name} with power spectrum {i}")
                results[(func_name, i)] = self.run(func_name, **params)
        return results
    
    def get(self, *terms, **override_kwargs):
        """Allows for quick access to a specific term or terms from a FASTPT function."""
        if not terms:
            raise ValueError("At least one term must be provided.")
        output = {}
        unique_funcs = {
                    "P_Btype2": "_get_P_Btype2",
                    "P_deltaE2": "_get_P_deltaE2",
                    "P_der": "IA_der",
                    "P_OV": "OV",
                    "P_0tE": "_get_P_0tE",
                    "P_0EtE": "_get_P_0EtE",
                    "P_E2tE": "_get_P_E2tE",
                    "P_tEtE": "_get_P_tEtE",
                    "P_1loop": "one_loop_dd",
                    "Ps": "one_loop_dd",
                    "Pd1d2": "_get_Pd1d2",
                    "Pd2d2": "_get_Pd2d2",
                    "Pd1s2": "_get_Pd1s2",
                    "Pd2s2": "_get_Pd2s2",
                    "Ps2s2": "_get_Ps2s2",
                    "sig4": "_get_sig4",
                    "sig3nl": "_get_sig3nl",
                    "Pb1L": "_get_Pb1L",
                    "Pb1L_2": "_get_Pb1L_2",
                    "Pb1L_b2L": "_get_Pb1L_b2L",
                    "Pb2L": "_get_Pb2L",
                    "Pb2L_2": "_get_Pb2L_2"
                }
        for term in terms:
            if term not in self.term_sources:
                raise ValueError(f"Term '{term}' not found in FASTPT.")
            if term in unique_funcs.keys(): #Terms that have their own unique functions
                func_name = unique_funcs[term]
                func = getattr(self.fastpt, func_name)
            
                passing_params, _ = self._prepare_function_params(func, override_kwargs)
                result = func(**passing_params)

                # Special handling for P_1loop and Ps terms which come from one_loop_dd, only func that returns a tuple
                if term == "P_1loop" and isinstance(result, tuple):
                    result = result[0]
                elif term == "Ps" and isinstance(result, tuple):
                    result = result[1]
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
        self.__fastpt = fastpt_instance
        self.clear_cache()
        print("FASTPT instance updated. Cached cleared.")

    def save_output(self, result, func_name, type="txt", output_dir=None):
        """ 
        Save the output to a file
        
        Args:
            result: The result to save
            func_name (str): Name of the function that produced the result
            type (str): File type ('txt', 'csv', or 'json')
            output_dir (str, optional): Directory to save the file in. Defaults to self.output_dir.
        """
        if type not in ("txt", "csv", "json"): 
            raise ValueError("Invalid file type. Must be 'txt', 'csv', or 'json'")
        
        import os
        save_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(save_dir, exist_ok=True)

        if func_name in ("one_loop_dd_bias_lpt_NL", "one_loop_dd_bias_b3nl", "one_loop_dd_bias"):
            for i, element in enumerate(result):
                if isinstance(element, float): # sig4 is of type float, converting it to np array
                    new_array = np.zeros(len(result[i-1]))
                    new_array[0] = element
                    result = list(result)
                    result[i] = new_array

        base_name = f"{func_name}_output.{type}"
        file_path = os.path.join(save_dir, base_name)
        
        counter = 1
        while os.path.exists(file_path):
            new_name = f"{func_name}_{counter}_output.{type}"
            file_path = os.path.join(save_dir, new_name)
            counter += 1
        
        try:
            if type == "txt":
                np.savetxt(file_path, np.transpose(result), header=f'{func_name}')
            elif type == "csv":
                import csv
                data_for_csv = []
                
                if isinstance(result, np.ndarray) and result.ndim == 1:
                    data_for_csv = [[x] for x in result]
                else:
                    # Try to handle as collection of arrays or values
                    transposed = np.transpose(result)
                    data_for_csv = transposed.tolist() if hasattr(transposed, 'tolist') else transposed
                
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if isinstance(result, tuple) or isinstance(result, list):
                        header = [f'{func_name}_{i}' for i in range(len(result))]
                    else:
                        header = [func_name]
                    writer.writerow(header)
                    writer.writerows(data_for_csv)
            elif type == "json":
                import json
                
                # Prepare data for JSON serialization (numpy arrays aren't directly JSON serializable)
                def numpy_to_python(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (tuple, list)):
                        return [numpy_to_python(item) for item in obj]
                    elif isinstance(obj, np.number):
                        return obj.item()
                    return obj
                
                json_data = {func_name: numpy_to_python(result)}
                with open(file_path, 'w') as jsonfile:
                    json.dump(json_data, jsonfile, indent=2)
            
            print(f"Output saved to {file_path}")
        except Exception as e:
            print(f"Error saving {func_name} output: {str(e)}")

    def load(self, file_path, load_dir=None):
        """ 
        Load a saved output file and return it in the same format as FASTPT outputs (tuple of numpy arrays)
        
        Args:
            file_path (str): Name or path of the file to load
            load_dir (str, optional): Directory to load file from. If None, uses default output directory
                                     If file_path already includes a directory, load_dir is ignored.
        
        Returns:
            tuple: A tuple of numpy arrays matching the original FASTPT function output format
        """
        import os
        import numpy as np
        import re
        
        # If file_path is an absolute path or already contains directory info, use it as is
        if os.path.isabs(file_path) or os.path.dirname(file_path):
            full_path = file_path
        else:
            # Otherwise, build path from load_dir or default output directory
            directory = load_dir if load_dir is not None else self.output_dir
            full_path = os.path.join(directory, file_path)
        
        # Get file extension
        _, ext = os.path.splitext(full_path)
        ext = ext.lower()
    
        # Check for valid extension before checking if file exists
        if ext not in (".txt", ".csv", ".json"):
            raise FileNotFoundError(f"Unsupported file extension: {ext}. Must be '.txt', '.csv', or '.json'")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File '{full_path}' not found.")

        
        # Extract function name from filename
        func_name = re.match(r'(.+?)(?:_\d+)?_output\.', os.path.basename(full_path))
        func_name = func_name.group(1) if func_name else None
        
        try:
            arrays = []
            
            if ext == ".txt":
                # Load and transpose to match original format
                loaded_data = np.loadtxt(full_path)
                # Split columns into separate arrays
                for i in range(loaded_data.shape[1]):
                    arrays.append(loaded_data[:, i])
                
            elif ext == ".csv":
                import csv
                with open(full_path, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    # Skip header row
                    next(reader)
                    # Read all rows
                    data_rows = list(reader)
                    
                    # Convert to numeric values
                    numeric_data = []
                    for row in data_rows:
                        numeric_data.append([float(val) for val in row])
                    
                    # Convert to numpy array
                    all_data = np.array(numeric_data)
                    
                    # Split columns into separate arrays (transposing to match original format)
                    for i in range(all_data.shape[1]):
                        arrays.append(all_data[:, i])
                    
            elif ext == ".json":
                import json
                with open(full_path, 'r') as jsonfile:
                    data = json.load(jsonfile)
                    # Get the function name from the data if available
                    if not func_name and len(data) == 1:
                        func_name = next(iter(data))
                    
                    # Get the data under the function name key
                    result_data = data[next(iter(data))]
                    
                    # Handle different possible structures in JSON
                    if isinstance(result_data, list):
                        # If result_data is a list of lists (multiple arrays)
                        if result_data and isinstance(result_data[0], list):
                            for arr in result_data:
                                arrays.append(np.array(arr))
                        else:
                            # Single array
                            arrays.append(np.array(result_data))
                    else:
                        # Single value or other structure
                        arrays.append(np.array([result_data]))
            else:
                raise ValueError(f"Unsupported file extension: {ext}. Must be '.txt', '.csv', or '.json'")
            
            # Handle special case for sig4 in bias functions - convert back to float
            # In one_loop_dd_bias and one_loop_dd_bias_b3nl, sig4 is at index 7
            # In one_loop_dd_bias_lpt_NL, sig4 is at index 6
            if func_name in ["one_loop_dd_bias", "one_loop_dd_bias_b3nl"] and len(arrays) > 7:
                # Check if the array is mostly zeros with one value
                if arrays[7].size > 1 and np.count_nonzero(arrays[7]) <= 1:
                    # Get the first non-zero value or the first value if all zeros
                    if np.any(arrays[7]):
                        sig4_value = arrays[7][np.nonzero(arrays[7])[0][0]]
                    else:
                        sig4_value = arrays[7][0]
                    arrays[7] = sig4_value
                    
            elif func_name == "one_loop_dd_bias_lpt_NL" and len(arrays) > 6:
                # Similar check for lpt_NL case
                if arrays[6].size > 1 and np.count_nonzero(arrays[6]) <= 1:
                    if np.any(arrays[6]):
                        sig4_value = arrays[6][np.nonzero(arrays[6])[0][0]]
                    else:
                        sig4_value = arrays[6][0]
                    arrays[6] = sig4_value
                    
            print(f"Output loaded from {full_path}")
            
            # Convert list of arrays to tuple to match FASTPT output format
            return tuple(arrays)
        
        except Exception as e:
            print(f"Error loading output from {full_path}: {str(e)}")
            return None