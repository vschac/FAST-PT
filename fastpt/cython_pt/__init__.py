try:
    from .fastpt_core import (
        hash_arrays_cy,
        create_hash_key_cy,
        compute_term_cy,
        apply_extrapolation_cy,
        compute_convolution,
        compute_fourier_coefficients,
        J_k_scalar_cy,
        J_k_tensor_cy,
        clear_workspaces
    )
    from .cython_CacheManager import CacheManager_cy
    CYTHON_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"(init.py Warning) FASTPT: Failed to import Cython modules: {e}. Falling back to Python implementation.")
    CYTHON_AVAILABLE = False