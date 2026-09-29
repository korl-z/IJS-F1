"""Validation checks for the shell grid solver."""

from dataclasses import replace

import numpy as np
import scipy.fft as sfft
from scipy.optimize import brentq, root
from scipy.special import expit

from .eq import solve_eq
from .geom import Kern, _phi, _phis, auto_L, kernels
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
    K = rates(cells(sh, band, mf, d, m), baths, kernels(sh, band, baths, A))
    n, p = expit(-mu), expit(mu)
    gain, loss = p * (K @ n), n * (K.T @ p)
    return float(abs(np.sum(gain - loss)) / np.sum(gain + loss))


def check_iso(sh, band, mf, baths, sol, A=None, cache=None):
    """Within shell anisotropy left out by the closure n(k) = n(s(k)).

    One Jacobi step of the k resolved rate equations from the shell NESS sol:
    every fine grid state k keeps the momentum structure of |g(k - p)|^2 and
    the factors (coherence, line, Bose) of its cell pair, which gives the
    local multiplier mu(k) = log(G_out(k) / G_in(k)). Returns (2, ns) arrays:
    rel. spread of G_in + G_out ('gam'), spread of mu(k) - mu_I ('dmu') and
    mean n(k) - n_I ('dn'), NaN for cells with fewer than 4 fine points.
    Scattering inside one cell is left out.
    """
    baths = tuple(baths)
    kn = kernels(sh, band, baths, A, cache=cache)
    cl = cells(sh, band, mf, sol.d, sol.m)
    ns = sh.ns
    n, p = expit(-sol.mu).reshape(-1), expit(sol.mu).reshape(-1)
    L = max(auto_L(ns, b.lam) for b in baths)
    k = -np.pi + 2 * np.pi * (np.arange(L) + 0.5) / L
    lab = sh.label(np.cos(k)[:, None], np.cos(k)[None, :])
    gin, gout = np.zeros((2, L, L)), np.zeros((2, L, L))
    for b, kb in zip(baths, kn):
        if b.disp == "constant":
            phi = _phi(L, b.lam, b.qd, 4)[None] / b.w0
        else:
            phi = _phis(b, L, 4, kb.dw * (1.0 + 1e-9))[0]
            if phi.shape[0] != kb.om.size:
                raise ValueError("frequency bins differ from the kernel")
        for m in range(kb.om.size):
            # pair factor of bin m: cell pair rate over its kernel entry
            K = rates(cl, (b,), (Kern(kb.A[m:m + 1], kb.om[m:m + 1], kb.dw),))
            at = np.tile(kb.A[m], (2, 2))
            cf = np.where(at > 0, K / np.where(at > 0, at, 1.0), 0.0)
            fp = sfft.rfft2(phi[m], workers=-1)
            for j in range(ns):
                # r(k) = sum_{q in cell j} |g(k - q)|^2 / amp / L^2
                chi = sfft.rfft2((lab == j).astype(float), workers=-1)
                r = sfft.irfft2(fp * chi, s=(L, L), workers=-1) / L**2
                for nu in range(2):
                    jj = nu * ns + j
                    for mu in range(2):
                        ii = mu * ns + lab
                        gin[mu] += cf[ii, jj] * n[jj] * r
                        gout[mu] += cf[jj, ii] * p[jj] * r

    lf = lab.ravel()
    out = {key: np.full((2, ns), np.nan) for key in ("gam", "dmu", "dn")}
    for mu in range(2):
        gi, go = gin[mu].ravel(), gout[mu].ravel()
        ok = (gi > 0) & (go > 0)
        ml = np.log(np.where(ok, go, 1.0)) - np.log(np.where(ok, gi, 1.0))
        cnt = np.bincount(lf, weights=ok, minlength=ns)
        has = cnt >= 4
        c = np.where(has, cnt, 1.0)

        def spread(y):
            av = np.bincount(lf, weights=ok * y, minlength=ns) / c
            va = np.bincount(lf, weights=ok * (y - av[lf]) ** 2, minlength=ns) / c
            return av, np.sqrt(va)

        ag, sg = spread(gi + go)
        _, sm = spread(ml - sol.mu[mu][lf])
        an, _ = spread(expit(-ml))
        out["gam"][mu] = np.where(has, sg / np.where(ag > 0, ag, 1.0), np.nan)
        out["dmu"][mu] = np.where(has, sm, np.nan)
        out["dn"][mu] = np.where(has, an - sol.n[mu], np.nan)
    return out


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
