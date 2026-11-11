"""Two band EI on the C4v irreducible BZ: grid, mean field, equilibrium. Numpy only.

Units hbar = kB = a0 = 1. Energies are in the frame shifted by -h / 2 (h = Hartree U),
so E_F = 0 for ta = -tb and n = 1. State order everywhere: (alpha, beta) x IBZ points.
d is the magnitude of the order parameter, u v = d / (2 E).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


class Model(NamedTuple):
    gap: float
    ta: float
    tb: float
    v: float  # mean field V
    n: float = 1.0  # filling
    uh: float | None = None  # Hartree U, None: V

    @property
    def h(self):
        return self.v if self.uh is None else self.uh


class Grid(NamedTuple):
    nk: Any
    ki: Any  # IBZ points as integers (ni, 2), k = 2 pi ki / nk, 0 <= ky <= kx <= pi
    kg: Any  # the 8 C4v images of every IBZ point, integers mod nk, (ni, 8, 2)
    s: Any  # cos kx + cos ky, (ni,)
    hv: Any  # sin^2 kx + sin^2 ky, (ni,)
    w: Any  # orbit weights m_j / N, (ni,)
    wi: Any  # IBZ indices of the kinetic states (s window), (nw,)
    pos: Any  # IBZ index -> position in wi, nw for the frozen states, (ni,)
    sn: Any  # sin(2 pi i / nk), (nk,)
    s2: Any  # sin(pi i / nk), (2 nk,)


class State(NamedTuple):
    e: Any  # quasiparticle energies (2, ni)
    u: Any
    v: Any


class Eq(NamedTuple):
    d: float
    m: float
    mu: float
    n: Any
    st: State
    err: float
    it: int
    ok: bool


# ---------------- grid ----------------

def grid(nk, win=None):
    """C4v orbits of the Gamma centred nk x nk grid.

    win = (smin, smax): only states with smin <= s <= smax are kinetic, the rest
    stay frozen at n_alpha = 0, n_beta = 1.
    """
    h = nk // 2
    ix, iy = np.meshgrid(np.arange(h + 1), np.arange(h + 1), indexing="ij")
    x, y = ix[iy <= ix], iy[iy <= ix]
    im = ((x, y), (-x, y), (x, -y), (-x, -y), (y, x), (-y, x), (y, -x), (-y, -x))
    kg = np.stack([np.stack(p, -1) for p in im], 1) % nk
    mult = np.array([len({tuple(p) for p in q}) for q in kg])  # orbit sizes
    dk = 2.0 * np.pi / nk
    s = np.cos(dk * x) + np.cos(dk * y)
    ok = np.ones(s.size, bool) if win is None else (s >= win[0]) & (s <= win[1])
    wi = np.flatnonzero(ok)
    if wi.size == 0:
        raise ValueError("the s window holds no grid points")
    pos = np.full(s.size, wi.size)
    pos[wi] = np.arange(wi.size)
    return Grid(nk, np.stack((x, y), -1), kg, s, np.sin(dk * x) ** 2 + np.sin(dk * y) ** 2,
                mult / nk**2, wi, pos, np.sin(dk * np.arange(nk)),
                np.sin(0.5 * dk * np.arange(2 * nk)))


def sgrid(ns, nf=256):
    """Bins in s = cos kx + cos ky instead of k points: assumes n = n(s).

    Bin means and weights are counted on a fine nf x nf grid (stored as nk). The result
    is a Grid, so mf, targets, the equilibrium solvers and the NESS solvers work unchanged.
    """
    k = 2.0 * np.pi * np.arange(nf) / nf
    kx, ky = (q.ravel() for q in np.meshgrid(k, k, indexing="ij"))
    sv = np.cos(kx) + np.cos(ky)
    bi = np.clip(np.digitize(sv, np.linspace(-2.0, 2.0, ns + 1)) - 1, 0, ns - 1)
    cnt = np.bincount(bi, minlength=ns)
    if np.any(cnt == 0):
        raise ValueError("empty s bins, increase nf or decrease ns")
    hv = np.bincount(bi, weights=np.sin(kx) ** 2 + np.sin(ky) ** 2, minlength=ns) / cnt
    i = np.arange(ns)
    return Grid(nf, None, None, np.bincount(bi, weights=sv, minlength=ns) / cnt, hv, cnt / nf**2,
                i, i, None, None)


def unfold(a, g):
    """IBZ array (..., ni) -> full grid (..., nk, nk), k = 2 pi (i - nk // 2) / nk."""
    i = np.abs(np.arange(g.nk) - g.nk // 2)
    hi, lo = np.maximum(i[:, None], i[None, :]), np.minimum(i[:, None], i[None, :])
    return np.asarray(a)[..., hi * (hi + 1) // 2 + lo]


# ---------------- mean field ----------------

def mf(md, g, d, m):
    """Bogoliubov state (eq. 33, 44, 45), c_a = u alpha + v beta, c_b = -v alpha + u beta."""
    na, nb = 0.5 * (md.n + m), 0.5 * (md.n - m)
    eah = 0.5 * md.gap - 2.0 * md.ta * g.s + md.h * (nb - 0.5)
    ebh = -0.5 * md.gap - 2.0 * md.tb * g.s + md.h * (na - 0.5)
    xi, av = 0.5 * (eah - ebh), 0.5 * (eah + ebh)
    ek = np.hypot(xi, d)
    sf = np.where(ek > 0.0, ek, 1.0)
    big = np.sqrt((sf + np.abs(xi)) / (2.0 * sf))
    sml = np.abs(d) / (2.0 * sf * big)
    u = np.where(ek > 0.0, np.where(xi >= 0.0, big, sml), 1.0)
    v = np.where(ek > 0.0, np.copysign(np.where(xi >= 0.0, sml, big), d), 0.0)
    return State(np.stack((av + ek, av - ek)), u, v)


def targets(md, g, st, n):
    """Gap, imbalance and filling from the occupations (eq. 37, 42, 43)."""
    u2, v2 = st.u**2, st.v**2
    na = np.sum(g.w * (u2 * n[0] + v2 * n[1]))
    nb = np.sum(g.w * (v2 * n[0] + u2 * n[1]))
    d = md.v * np.sum(g.w * st.u * st.v * (n[1] - n[0]))
    return d, na - nb, na + nb


def gap_info(md, g, st):
    """Bare, Hartree and quasiparticle gaps at Gamma and over the zone."""
    ea, eb = 0.5 * md.gap - 2.0 * md.ta * g.s, -0.5 * md.gap - 2.0 * md.tb * g.s
    dd, av = st.e[0] - st.e[1], 0.5 * (st.e[0] + st.e[1])
    xi = 0.5 * dd * (st.u**2 - st.v**2)
    i0 = np.argmax(g.s)
    return dict(bare_g=ea[i0] - eb[i0], hartree_g=2.0 * xi[i0], diag_g=dd[i0],
                diag_min=dd.min(), diag_ind=st.e[0].min() - st.e[1].max(),
                bare_ind=ea.min() - eb.max(),
                hartree_ind=(av + xi).min() - (av - xi).max(),
                k_min=2.0 * np.pi * g.ki[np.argmin(dd)] / g.nk)


# ---------------- equilibrium ----------------

def fermi(st, w, pn, t, tol=1.0e-13, nmax=200):
    """Fermi occupations (2, ni) and mu at filling pn; t <= 0: T = 0 filling of the grid."""
    if t <= 0.0:
        return _fermi0(st, w, pn)
    es = max(np.ptp(st.e), t, 1.0)
    lo, hi = st.e.min() - 50.0 * es, st.e.max() + 50.0 * es
    for _ in range(nmax):
        mu = 0.5 * (lo + hi)
        lo, hi = (mu, hi) if np.sum(w * expit((mu - st.e) / t)) < pn else (lo, mu)
        if hi - lo <= tol * max(1.0, abs(mu)):
            break
    mu = 0.5 * (lo + hi)
    return expit((mu - st.e) / t), mu


def _fermi0(st, w, pn):
    y, ww = st.e.reshape(-1), np.tile(w, 2)
    ix = np.argsort(y)
    cw = np.cumsum(ww[ix])
    n = np.zeros_like(y)
    if pn <= 0.0:
        return n.reshape(st.e.shape), y.min() - 1.0
    if pn >= 2.0:
        return np.ones_like(st.e), y.max() + 1.0
    j = int(np.searchsorted(cw, pn, side="left"))
    n[ix[:j]] = 1.0
    n[ix[j]] = np.clip((pn - (cw[j - 1] if j else 0.0)) / ww[ix[j]], 0.0, 1.0)
    mu = y[ix[j]] if n[ix[j]] < 1.0 or j + 1 >= y.size else 0.5 * (y[ix[j]] + y[ix[j + 1]])
    return n.reshape(st.e.shape), mu


def solve_eq(md, g, t, d=0.2, m=0.0, mix=(0.2, 0.2), tol=1.0e-10, nmax=20000):
    """Damped iteration of eq. 46 - 48. Do not start at d = 0."""
    for it in range(1, nmax + 1):
        st = mf(md, g, d, m)
        n, mu = fermi(st, g.w, md.n, t)
        td, tm, tn = targets(md, g, st, n)
        er = max(abs(td - d), abs(tm - m), abs(tn - md.n))
        if er < tol:
            break
        d, m = d + mix[0] * (td - d), m + mix[1] * (tm - m)
    return Eq(abs(d), m, mu, n, mf(md, g, abs(d), m), er, it, er < tol)


def solve_normal(md, g, t, m=0.0, mix=0.3, tol=1.0e-11, nmax=10000):
    """Normal state: d = 0, self consistent m."""
    for it in range(1, nmax + 1):
        st = mf(md, g, 0.0, m)
        n, mu = fermi(st, g.w, md.n, t)
        _, tm, tn = targets(md, g, st, n)
        er = max(abs(tm - m), abs(tn - md.n))
        if er < tol:
            break
        m += mix * (tm - m)
    return Eq(0.0, m, mu, n, st, er, it, er < tol)


def pair_lambda(md, g, t, m=0.0, tol=1.0e-11):
    """lam = V sum w (n_beta - n_alpha) / (2 xi) of the normal state; lam > 1: unstable."""
    s = solve_normal(md, g, t, m=m, tol=tol)
    ek = 0.5 * (s.st.e[0] - s.st.e[1])
    ix = ek > 1.0e-12
    f0 = 0.5 * (s.n[0] + s.n[1])
    q = np.where(ix, (s.n[1] - s.n[0]) / (2.0 * np.where(ix, ek, 1.0)), f0 * (1.0 - f0) / t)
    return md.v * np.sum(g.w * q), s


def critical_temperature(md, g, tlo=1.0e-4, thi=5.0, tol=1.0e-8, nmax=80):
    """Largest T with lam = 1, by bisection."""
    ll, sl = pair_lambda(md, g, tlo)
    lh, sh = pair_lambda(md, g, thi)
    if ll <= 1.0:
        raise ValueError("no instability at tlo")
    if lh >= 1.0:
        raise ValueError("increase thi")
    ml, mh = sl.m, sh.m
    for _ in range(nmax):
        tm = 0.5 * (tlo + thi)
        lm, sm = pair_lambda(md, g, tm, m=0.5 * (ml + mh))
        if lm >= 1.0:
            tlo, ml = tm, sm.m
        else:
            thi, mh = tm, sm.m
        if thi - tlo <= tol * max(1.0, tm):
            return 0.5 * (tlo + thi)
    raise RuntimeError("T_c did not converge")


def bec_zero(md, g, d0, vhi=20.0):
    """BEC root at fixed d0 (ta = -tb, n = 1): V, delta0 = 1 + m0, xi0, E0. md.v is not used."""
    x = 0.5 * md.gap - (md.ta - md.tb) * g.s

    def parts(v):
        h, dl = (v if md.uh is None else md.uh), 0.0
        for _ in range(200):
            x0 = x + 0.5 * h * (1 - dl)
            e0 = np.hypot(x0, d0)
            new = np.sum(g.w * d0**2 / (e0 * (e0 + x0)))
            if abs(new - dl) <= 1e-15 * new:
                break
            dl = new
        return dl, x0, e0

    vlo = -x.min() * 2 + 1e-9 if md.uh is None else 1e-9  # no ring in the polarized state
    v = brentq(lambda v: v * np.sum(g.w * 0.5 / parts(v)[2]) - 1, vlo, vhi, xtol=1e-15)
    return (v, *parts(v))


def bec_lam1(md, g, d0, dl0, x0, e0, t):
    """lam(T) - 1 of the BEC normal state, margin and thermal part summed directly."""
    v, h = md.v, md.h
    xn, dl = 0.5 * md.gap - (md.ta - md.tb) * g.s + 0.5 * h, 0.0
    for _ in range(200):  # delta(T) = 1 + m = 2 sum w f(xi)
        new = 2 * np.sum(g.w * expit(-(xn - 0.5 * h * dl) / t))
        if new == 0 or abs(new - dl) <= 1e-14 * new:
            dl = new
            break
        dl = new
    xi = xn - 0.5 * h * dl
    if xi.min() <= 0:
        raise ValueError("the normal state has a ring at this T, lower thi")
    mg = v * np.sum(g.w * (d0**2 / (e0 + x0) - 0.5 * h * (dl0 - dl)) / (2 * xi * e0))
    return mg - v * np.sum(g.w * expit(-xi / t) / xi)
