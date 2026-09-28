"""Steady states of the bath driven excitonic insulator on an energy shell grid."""

from .grid import Band, MF, Shells, auto_ns, make_shells, s_of_k
from .geom import Kern, Phonon, bath_kernel, from_cfg, kernels, omega, pair_kernel
from .kernel import cells, rates
from .eq import solve_eq
from .solve import Sol, pair_chi, solve_auto, solve_ness, sweep, t_eff, to_k


def solve_ness_jax(*args, **kwargs):
    from .solve_jax import solve_ness
    return solve_ness(*args, **kwargs)


def solve_auto_jax(*args, **kwargs):
    from .solve_jax import solve_auto
    return solve_auto(*args, **kwargs)


def sweep_jax(*args, **kwargs):
    from .solve_jax import sweep
    return sweep(*args, **kwargs)

__all__ = [
    "Band", "MF", "Shells", "auto_ns", "make_shells", "s_of_k",
    "Kern", "Phonon", "bath_kernel", "from_cfg", "kernels", "omega", "pair_kernel",
    "cells", "rates", "solve_eq",
    "Sol", "pair_chi", "solve_auto", "solve_ness", "sweep", "t_eff", "to_k",
    "solve_auto_jax", "solve_ness_jax", "sweep_jax",
]
