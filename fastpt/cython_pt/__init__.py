try:
    from .fastpt_core import *
    from .cython_CacheManager import CacheManager_cy
    CYTHON_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"(init.py Warning) FASTPT: Failed to import Cython modules: {e}. Falling back to Python implementation.")
    CYTHON_AVAILABLE = False