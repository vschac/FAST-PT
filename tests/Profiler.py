from fastpt import FASTPT
import numpy as np
import os
from line_profiler import LineProfiler
from fastpt import Wigner_symbols

def profile_fastpt():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'Pk_test.dat')
    d = np.loadtxt(data_path)
    k = d[:, 0]
    
    n_pad = int(0.5 * len(k))
    fpt = FASTPT(k, to_do=['all'], low_extrap=-5, high_extrap=3, n_pad=n_pad)


# Profile specific methods
lprofiler = LineProfiler()
lprofiler.add_module(Wigner_symbols)
'''
Longer runtime functions:
- initialize_params.g_m_vals (lines 38 and 42, from scalar stuff and tensor stuff)
- J_table.coeff_b <- factorial (lines 28 - 32)
'''

# Run profiling
lprofiler.run('profile_fastpt()')
lprofiler.print_stats()


