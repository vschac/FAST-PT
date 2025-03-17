# docs/conf.py
import os
import sys
import datetime

# Add the project root directory to the Python path so Sphinx can find the package
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'FAST-PT'
copyright = f'2016-{datetime.datetime.now().year}, FAST-PT developers'
author = 'Joseph E. McEwen, Xiao Fang, Jonathan Blazek'

# The full version, including alpha/beta/rc tags
from fastpt.info import __version__
release = __version__

# Add any Sphinx extension module names here, as strings
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'numpydoc',
]

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ['_static']

# Configure autodoc to skip special methods
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,  # Don't include _private members
    'special-members': '__init__',  # Only include __init__ special method
    'exclude-members': 'J_k_tensor,J_k_scalar,X_IA_0B0B'  # List specific methods to exclude
}

# Configure numpydoc
numpydoc_show_class_members = False

# Configure intersphinx mapping for linking to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}