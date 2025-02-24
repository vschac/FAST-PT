import numpy as np
from FPTHandler import FPTHandler
from fastpt import FASTPT

# Example power spectrum data
k_values = np.logspace(-3, 1, 100)  # Example k values
default_P = np.abs(np.sin(k_values))  # Example P values

# Initialize FASTPT instance
fastpt_instance = FASTPT(k_values, to_do=['skip'])

# Create FPTHandler instance with default parameters, NOTE all parameters must be passed as keyword arguments
handler = FPTHandler(fastpt_instance, P=default_P, P_window=np.array([0.2, 0.2]), C_window=0.75)

print("\n--- Available FASTPT Functions ---")
handler.list_available_functions()

print("\n--- Running one_loop_dd ---")
result = handler.run('one_loop_dd')
#print("Result:", result)

print("\n--- Caching Demonstration ---")
handler.run('one_loop_dd')  # Should use cached result
handler.show_cache()

print("\n--- Updating Default Parameters ---")
handler.update_default_params(P=default_P * 1.1, P_window=np.array([0.1, 0.1]), C_window=0.5)

print("\n--- Clearing Cache ---")
handler.clear_cache()

print("\n--- Running Function with Updated Params ---")
result = handler.run('one_loop_dd')
#print("Result:", result)

print("\n--- Testing Error Handling ---")
try:
    handler.run('nonexistent_function')
except ValueError as e:
    print("Error:", e)

print("\n--- Demonstrating Function Parameter Listing ---")
function_params = handler._get_function_params(fastpt_instance.one_loop_dd)
print("Required Params:", function_params['required'])
print("Optional Params:", function_params['optional'])

print("\n--- Clearing Default Params ---")
handler.clear_default_params()

print("\n--- Demonstrate insufficient params ---")
try:
    handler.run('one_loop_dd')
except ValueError as e:
    print("Error:", e)
