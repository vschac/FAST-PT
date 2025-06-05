import numpy as np
from fastpt import FASTPT, FPTHandler

# Initialize with default parameters
k_values = np.loadtxt("Pk_test.dat")[:, 0]

fastpt_instance = FASTPT(k_values)
handler = FPTHandler(fastpt_instance, P_window=np.array([0.2, 0.2]), C_window=0.75)

# Generate and store a power spectrum
try: 
    P_camb = handler.generate_power_spectra(method='camb')
    P = np.loadtxt("Pk_test.dat")[:, 1]
except Exception as e:
    if ("class is not installed" in str(e)):
        print("Class is not installed to generate power spectra, using preloaded power spectrum instead.")
        P = np.loadtxt("Pk_test.dat")[:, 1]
    else:
        print("An error occurred while generating power spectra:", e)

handler.update_default_params(P=P)

# Get the 1-loop power spectrum, using the default parameters
result = handler.get("P_1loop")
result_camb = handler.get("P_1loop", P=P_camb)
# ^^ This is equivalent to: calling fastpt_instance.one_loop_dd(P, P_window=np.array([0.2,0.2]), C_window=0.75)[0]

#Plot the results
# handler.plot(data=result, title="P_1loop")

# # Save the results and your parameters
# handler.output_dir = "output"
# handler.save_output(result, "one_loop_dd")
# handler.save_params("params.npz")


from matplotlib import pyplot as plt
plt.plot(k_values, result, label='P_1loop')
plt.plot(k_values, result_camb, label='P_1loop camb', linestyle='--')
plt.xlabel('k [h/Mpc]')
plt.ylabel('P_1loop')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

rel_diff = np.abs(result - result_camb) / np.abs(result)
plt.plot(k_values, rel_diff, label='Relative difference')
plt.xlabel('k [h/Mpc]')
plt.ylabel('Relative difference')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()