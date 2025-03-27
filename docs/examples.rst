.. _examples:

Examples
=======

One-loop Matter Power Spectrum
----------------------------

.. code-block:: python

   from fastpt import FASTPT
   import numpy as np
   import matplotlib.pyplot as plt

   # Load data
   data = np.loadtxt('Pk_test.dat')
   k = data[:, 0]
   P = data[:, 1]

   # Initialize FASTPT
   fpt = FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5*len(k)))

   # Calculate corrections
   P_1loop, P_components = fpt.one_loop_dd(P, C_window=0.75)

   # Plot
   plt.figure(figsize=(10, 7))
   plt.loglog(k, P, label='Linear P(k)')
   plt.loglog(k, P_1loop, label='1-loop P(k)')
   plt.xlabel('k [h/Mpc]')
   plt.ylabel('P(k) [(Mpc/h)³]')
   plt.legend()
   plt.tight_layout()
   plt.show()

Using the FPTHandler
-----------------

.. code-block:: python

   import numpy as np
   from fastpt import FASTPT, FPTHandler

   # Initialize with default parameters
   k_values = np.logspace(-3, 1, 100)
   P_values = np.abs(np.sin(k_values))  # Example power spectrum

   fastpt_instance = FASTPT(k_values)
   handler = FPTHandler(fastpt_instance, P=P_values, P_window=np.array([0.2, 0.2]), C_window=0.75)

   # Run one_loop_dd calculation
   result = handler.run('one_loop_dd')
   print("Result:", result)

   # Show available FASTPT functions
   handler.list_available_functions()

   # Show cache information
   handler.show_cache_info()