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
    fpt = FASTPT(k, to_do=['IA_tt'], low_extrap=-5, high_extrap=3, n_pad=n_pad)
    fpt.IA_tt(d[:,1])


# Profile specific methods
lprofiler = LineProfiler()
lprofiler.add_module(FASTPT)
'''
Longer runtime functions:
- initialize_params.g_m_vals (lines 38 and 42, from scalar stuff and tensor stuff)
- J_table.coeff_b <- factorial (lines 28 - 32)
'''

# Run profiling
#lprofiler.run('profile_fastpt()')
#lprofiler.print_stats()


from time import time

data_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'Pk_test.dat')
d = np.loadtxt(data_path)
k = d[:, 0]
    
n_pad = int(0.5 * len(k))
fpt = FASTPT(k, to_do=['IA_tt'], low_extrap=-5, high_extrap=3, n_pad=n_pad)
t0 = time()
fpt.IA_tt(d[:,1])
t1 = time()
print(f"Time taken for IA_tt with todo list: {t1-t0}")
fpt2 = FASTPT(k, to_do=['skip'], low_extrap=-5, high_extrap=3, n_pad=n_pad)
t3 = time()
fpt2.IA_tt(d[:,1])
t4 = time()
print(f"Time taken for IA_tt with skip list: {t4-t3}")
t4 = time()
fpt2.IA_tt(d[:,1])
t5 = time()
print(f"Time taken for IA_tt a second time (with skip list): {t5-t4}")