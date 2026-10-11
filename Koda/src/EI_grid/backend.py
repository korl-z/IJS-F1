"""Array backend: NumPy by default, JAX if EI_GRID_XP=jax is set before import."""

import os

import numpy as np

NAME = os.environ.get("EI_GRID_XP", "jax").lower()

if NAME == "jax":
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as xp

    jit = jax.jit
else:
    NAME = "numpy"
    xp = np

    def jit(fn):
        return fn


def sig(x):
    """Logistic function, accurate in both tails."""
    return xp.exp(-xp.logaddexp(0.0, -x))


def host(a):
    """Copy an array to the host as float64."""
    return np.asarray(a, dtype=np.float64)
