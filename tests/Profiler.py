from fastpt import FASTPT
import numpy as np
import os
from line_profiler import LineProfiler

def profile_fastpt():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'Pk_test.dat')
    d = np.loadtxt(data_path)
    k = d[:, 0]
    
    n_pad = int(0.5 * len(k))
    fpt = FASTPT(k, to_do=['one_loop_dd'], low_extrap=-5, high_extrap=3, n_pad=n_pad)


# Profile specific methods
lprofiler = LineProfiler()
lprofiler.add_function(FASTPT.__init__)

'''
Longer runtime functions:
- initialize_params.g_m_vals (lines 38 and 42)

'''

# Run profiling
lprofiler.run('profile_fastpt()')
lprofiler.print_stats()

