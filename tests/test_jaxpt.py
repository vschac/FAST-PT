import pytest
import numpy as np
from fastpt import FASTPT
from fastpt.JAXPT import JAXPT
import os
import jax
from jax import numpy as jnp

data_path = os.path.join(os.path.dirname(__file__), 'benchmarking', 'Pk_test.dat')
P = np.loadtxt(data_path)[:, 1]
P_window = jnp.array([0.2, 0.2])
C_window = 0.75

@pytest.fixture
def jpt(): 
    d = np.loadtxt(data_path)
    k = jnp.array(d[:, 0])
    return JAXPT(k)

@pytest.fixture
def fpt():
    d = np.loadtxt(data_path)
    k = np.array(d[:, 0])
    return FASTPT(k)

############## Equality Tests ##############
def test_P_window(jpt, fpt):
    # Test that the P_window method returns the same result for JAXPT and FASTPT
    jp_window = jpt.P_window(jpt.k, P_window[0], P_window[1])
    p_window = fpt.P_window(fpt.k, P_window[0], P_window[1])
    assert np.allclose(jp_window, p_window), "P_window results are not equal"