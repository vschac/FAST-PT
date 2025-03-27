.. _quickstart:

Quick Start Guide
===============

Using FAST-PT is straightforward. Here's a simple example to get started:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from fastpt import FASTPT

   # Load a power spectrum
   data = np.loadtxt('Pk_test.dat')
   k = data[:, 0]
   P = data[:, 1]

   # Initialize FASTPT
   fpt = FASTPT(k, low_extrap=-5, high_extrap=3, n_pad=int(0.5*len(k)))

   # Calculate one-loop corrections
   P_1loop, P_components = fpt.one_loop_dd(P, C_window=0.75)

   # Plot the results
   plt.figure(figsize=(10, 7))
   plt.loglog(k, P, label='Linear P(k)')
   plt.loglog(k, P_1loop, label='One-loop P(k)')
   plt.xlabel('k [h/Mpc]')
   plt.ylabel('P(k) [(Mpc/h)³]')
   plt.legend()
   plt.show()