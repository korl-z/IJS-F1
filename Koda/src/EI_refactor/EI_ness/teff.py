import numpy as np
from scipy.optimize import brentq

from . import model as M


def power(md, g, pk, t, rates, d0=1.0, m0=0.0):
    """Energy current into the electrons, sum_i w_i E_i [(1 - n_i) a_i - n_i b_i], thermal state at t."""
    eq = M.solve_eq(md, g, t, d=d0, m=m0)
    e = np.asarray(eq.st.e[:, g.wi]).reshape(-1)
    n = np.asarray(eq.n[:, g.wi]).reshape(-1)
    w = np.tile(g.w[g.wi], 2)
    r = np.asarray(rates(md, g, eq.st, pk))
    a, b = r @ (w * n), r.T @ (w * (1.0 - n))
    return float(np.sum(w * (e - eq.mu) * ((1.0 - n) * a - n * b)))


def t_eff(md, g, pk, rates, t1, t2, d0=1.0, m0=0.0, tol=1.0e-6):
    """Root of power(T) between the two bath temperatures (absolute units)."""
    lo, hi = min(t1, t2), max(t1, t2)
    if hi - lo <= tol * hi:
        return lo
    f = lambda t: power(md, g, pk, t, rates, d0, m0)
    return brentq(f, lo * (1 + 1e-9), hi * (1 - 1e-9), xtol=tol * hi)
