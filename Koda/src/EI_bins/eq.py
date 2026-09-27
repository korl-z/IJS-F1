"""Thermal equilibrium on the shell grid.

Uses exactly the representation of the kinetic solver: cell occupations
n = f(E_cell) for the filling, and occupations at the sub nodes for the mean
field sums (which the kinetic solver reproduces by its GGE interpolation). For
T1 = T2 the kinetic steady state must equal this solution to solver precision.
"""

from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq, root
from scipy.special import expit

from .kernel import cells, mf_terms


class EqSol(NamedTuple):
    d: float
    m: float
    mc: float  # chemical potential
    t: float
    mu: np.ndarray  # GGE multipliers (E - mc) / t, (2, ns)
    n: np.ndarray  # occupations, (2, ns)
    ok: bool
    err: float


def chem_pot(e, w, n0, t):
    """Chemical potential giving filling n0 for cell energies e (2, ns)."""
    def f(mc):
        return np.sum(w * expit(-(e - mc) / t)) - n0
    lo = e.min() - 50.0 * t - 1.0
    hi = e.max() + 50.0 * t + 1.0
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=500)


def _occ(sh, band, mf, t, d, m):
    cl = cells(sh, band, mf, d, m)
    mc = chem_pot(cl.e, sh.w, mf.n, t)
    return cl, mc, expit(-(cl.e - mc) / t)


def _res(x, sh, band, mf, t, normal):
    d = 0.0 if normal else x[0]
    m = x[-1]
    cl, mc, _ = _occ(sh, band, mf, t, d, m)
    # mean field sums with the exact thermal occupations at the sub nodes
    gap, _, mt = mf_terms(sh, cl, expit(-(cl.es - mc) / t))
    if normal:
        return [m - mt]
    return [1.0 - mf.v * gap, m - mt]


def solve_eq(sh, band, mf, t, d=1.0, m=0.0, normal=False, tol=1e-12):
    """Self consistent (d, m, mc) at temperature t on the shell grid.

    The divided gap equation excludes d = 0; normal=True fixes d = 0.
    ok is False if no ordered solution was found.
    """
    t = max(float(t), 1e-10)
    x0 = [m] if normal else [abs(d), m]
    sr = root(_res, x0, args=(sh, band, mf, t, normal), method="hybr", tol=tol)
    x = sr.x
    d = 0.0 if normal else abs(x[0])
    m = x[-1]
    err = float(np.max(np.abs(_res(x, sh, band, mf, t, normal))))
    ok = bool(sr.success and err < 1e-8 and (normal or d > 1e-8))
    cl, mc, n = _occ(sh, band, mf, t, d, m)
    return EqSol(d, m, mc, t, (cl.e - mc) / t, n, ok, err)
