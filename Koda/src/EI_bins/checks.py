"""Validation checks for the shell grid solver."""

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq, root
from scipy.special import expit

from .eq import solve_eq
from .kernel import cells, rates
from .solve import solve_ness


def check_db(sh, band, mf, bath, A, t, amps=(1.0, 0.5), d=1.0, m=0.0):
    """Two baths at the same t must give exactly the shell equilibrium."""
    bs = [replace(bath, t=t, amp=a) for a in amps]
    s = solve_ness(sh, band, mf, bs, A=A, t0=t, d=d, m=m)
    e = solve_eq(sh, band, mf, t, d, m)
    return dict(ok=s.ok, dd=abs(s.d - e.d), dm=abs(s.m - e.m),
                dn=float(np.max(np.abs(s.n - e.n))))


def check_number(sh, band, mf, baths, A, d, m, seed=0):
    """Relative particle number rate sum_i W_i dn_i/dt for random occupations."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 3.0, 2 * sh.ns)
    K = rates(cells(sh, band, mf, d, m), baths, A)
    n, p = expit(-mu), expit(mu)
    gain, loss = p * (K @ n), n * (K.T @ p)
    return float(abs(np.sum(gain - loss)) / np.sum(gain + loss))


def eq_k(band, mf, t, nk=400, d=1.0, m=0.0):
    """Independent equilibrium on a plain nk x nk k grid, for comparison."""
    k = -np.pi + 2 * np.pi * (np.arange(nk) + 0.5) / nk
    s = (np.cos(k)[:, None] + np.cos(k)[None, :]).ravel()
    ea, eb = band.bare(s)

    def occ(x):
        dd, mm = abs(x[0]), x[1]
        xi = 0.5 * (ea - eb) - 0.5 * mf.h * mm
        et = 0.5 * (ea + eb) + 0.5 * mf.h * (mf.n - 1.0)
        ek = np.hypot(xi, dd)
        e = np.stack((et + ek, et - ek))
        mc = brentq(lambda c: np.mean(np.sum(expit(-(e - c) / t), 0)) - mf.n,
                    e.min() - 1.0, e.max() + 1.0, xtol=1e-14)
        return xi, ek, expit(-(e - mc) / t)

    def res(x):
        xi, ek, n = occ(x)
        return [1.0 - mf.v * np.mean((n[1] - n[0]) / (2.0 * ek)),
                x[1] - np.mean(xi / ek * (n[0] - n[1]))]

    x = root(res, [d, m], method="hybr", tol=1e-13).x
    return abs(x[0]), x[1]


def refine_eq(band, mf, t, ns_list, nk=1200, d=1.0, m=0.0):
    """Shell equilibrium against eq_k for a list of ns (expect O(h^2))."""
    from .grid import make_shells
    dk, mk = eq_k(band, mf, t, nk, d, m)
    out = []
    for ns in ns_list:
        e = solve_eq(make_shells(ns), band, mf, t, d, m)
        out.append((ns, e.d - dk, e.m - mk))
    return (dk, mk), out
