from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "fastpt_core", 
        ["fastpt_core.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "cython_CacheManager",
        ["cython_CacheManager.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    include_dirs=[np.get_include()],
)