import cProfile
import pstats
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

profiler = cProfile.Profile()
profiler.run('profile_fastpt()')

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)


# Profile specific methods
profiler = LineProfiler()
profiler.add_function(FASTPT.__init__)

# Run profiling
profiler.run('profile_fastpt()')
profiler.print_stats()


