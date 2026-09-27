"""Cell energies, coherence factors and rates at a given (d, m).

States are flattened as (alpha cells, beta cells), alpha is the upper branch.
K[i, j] is the pair flux coefficient for j -> i, so that
W_i dn_i/dt = (1 - n_i) sum_j K[i, j] n_j - n_i sum_j K[j, i] (1 - n_j).
"""

from typing import NamedTuple

import numpy as np


class Cells(NamedTuple):
    e: np.ndarray  # cell energy centers, (2, ns)
    hw: np.ndarray  # cell energy widths, (2, ns)
    ma: np.ndarray  # alpha orbital second moments, (ns, 2, 2)
    mb: np.ndarray  # beta orbital second moments, (ns, 2, 2)
    es: np.ndarray  # sub node energies, (2, ns, nsub)
    uvs: np.ndarray  # u v at the sub nodes, (ns, nsub)
    xes: np.ndarray  # u^2 - v^2 at the sub nodes
    ies: np.ndarray  # 1 / (2 E) at the sub nodes


def _qp(s, band, mf, d, m):
    """xi, eta and sqrt(xi^2 + d^2) at shell values s."""
    ea, eb = band.bare(s)
    xi = 0.5 * (ea - eb) - 0.5 * mf.h * m
    et = 0.5 * (ea + eb) + 0.5 * mf.h * (mf.n - 1.0)
    return xi, et, np.hypot(xi, d)


def cells(sh, band, mf, d, m, hmin=1e-12):
    """Cell energies, coherence moments and sub node quantities at (d, m).

    Center and width are the DOS weighted mean and sqrt(12 var) of E over the
    cell, so both are smooth in (d, m). The factor nsub^2 / (nsub^2 - 1)
    removes the bias of the discrete variance of a linear E(s).
    """
    xi, et, ek = _qp(sh.ss, band, mf, d, m)

    def avg(f):
        return np.sum(sh.ws * f, axis=-1)

    ns2 = sh.ss.shape[1] ** 2
    cor = 12.0 * ns2 / max(ns2 - 1.0, 1.0)
    es = np.stack((et + ek, et - ek))
    e = avg(es)
    hw = np.sqrt(np.maximum(cor * avg((es - e[..., None]) ** 2), hmin**2))

    ok = ek > 0.0
    ep = np.where(ok, ek, 1.0)
    xr = np.where(ok, xi / ep, 1.0)
    uv = np.where(ok, d / (2.0 * ep), 0.0)
    a, b, c = avg(0.5 * (1.0 + xr)), avg(0.5 * (1.0 - xr)), avg(uv)
    ma = np.stack((np.stack((a, -c), -1), np.stack((-c, b), -1)), -2)
    mb = np.stack((np.stack((b, c), -1), np.stack((c, a), -1)), -2)
    ie = np.where(ok, 0.5 / ep, 0.0)
    return Cells(e, hw, ma, mb, es, uv, xr, ie)


def gge_coef(cl):
    """Linear interpolation of mu in energy inside each cell.

    mu_sub = c0 mu_I + cm mu_{I-1} + cp mu_{I+1} (per branch), with the slope
    dmu/dE fitted to the neighbouring cells. Exact for thermal states, where
    mu = (E - mc) / T is linear in E.
    """
    e = cl.e
    dp = np.zeros_like(e)
    dm = np.zeros_like(e)
    dp[:, :-1] = e[:, 1:] - e[:, :-1]
    dm[:, 1:] = e[:, :-1] - e[:, 1:]
    s2 = dp**2 + dm**2
    s2 = np.where(s2 > 0.0, s2, 1.0)
    de = cl.es - e[..., None]
    cp = de * (dp / s2)[..., None]
    cm = de * (dm / s2)[..., None]
    return 1.0 - cp - cm, cm, cp


def sub_mu(mu, coef):
    """Sub node multipliers (2, ns, nsub) from cell multipliers (2, ns)."""
    c0, cm, cp = coef
    mup = np.zeros_like(mu)
    mum = np.zeros_like(mu)
    mup[:, :-1] = mu[:, 1:]
    mum[:, 1:] = mu[:, :-1]
    return c0 * mu[..., None] + cm * mum[..., None] + cp * mup[..., None]


def sub_grad(g, coef):
    """Chain rule: d/dmu_cell from d/dmu_sub values g (2, ns, nsub)."""
    c0, cm, cp = coef
    out = np.sum(g * c0, axis=-1)
    out[:, :-1] += np.sum(g * cm, axis=-1)[:, 1:]
    out[:, 1:] += np.sum(g * cp, axis=-1)[:, :-1]
    return out


def mf_terms(sh, cl, nsb):
    """Mean field sums from sub node occupations nsb (2, ns, nsub).

    Returns gap = sum W <(n_b - n_a) / 2E> (gap equation 1 = V gap),
    dg = sum W <u v (n_b - n_a)> (d = V dg) and
    mt = sum W <(u^2 - v^2)(n_a - n_b)> (m = mt).
    """
    w = sh.w[:, None] * sh.ws
    dn = nsb[1] - nsb[0]
    return (np.sum(w * cl.ies * dn), np.sum(w * cl.uvs * dn),
            -np.sum(w * cl.xes * dn))


def coh2(mi, mj, c):
    """Cell pair average of |U_I^T c U_J|^2 for one branch pair."""
    x = np.einsum("ab,iac,cd->ibd", c, mi, c)
    return np.einsum("ibd,jbd->ij", x, mj)


def trap(y, a, b):
    """Density at y of the sum of two centered uniforms with widths a, b."""
    return np.clip(0.5 * (a + b) - np.abs(y), 0.0, np.minimum(a, b)) / (a * b)


def trap_cdf(y, a, b):
    """Cumulative distribution of the trapezoid, exactly 0 and 1 outside."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)

    def left(y):
        t = np.clip(y + 0.5 * (lo + hi), 0.0, 0.5 * (lo + hi))
        return np.where(t < lo, t * t / (2.0 * lo * hi), 0.5 * lo / hi + (t - lo) / hi)

    return np.where(y <= 0.0, left(y), 1.0 - left(-y))


def line(y, a, b, cmin=0.0):
    """Energy overlap kernel of a cell pair with a smoothing box of width c.

    Trapezoid convolved with a box of width c = max((a + b) / 2, cmin):
    normalized, even in y, C1 in (y, a, b) and exactly zero outside its
    support. cmin keeps rates smooth on the grid resolution scale, also for
    the narrow cells at the gap edge.
    """
    y, a, b = np.broadcast_arrays(y, a, b)
    c = np.maximum(0.5 * (a + b), cmin)
    out = np.zeros(y.shape)
    ix = np.abs(y) < 0.5 * (a + b + c)  # support, exactly zero outside
    y, a, b, c = y[ix], a[ix], b[ix], c[ix]
    out[ix] = (trap_cdf(y + 0.5 * c, a, b) - trap_cdf(y - 0.5 * c, a, b)) / c
    return out


def bose(w, t):
    return 1.0 / np.expm1(w / t) if t > 0 else 0.0


def rates(cl, baths, A):
    """Pair flux coefficients K (2 ns, 2 ns) summed over baths.

    The downhill direction (source above destination) uses the physical
    emission and absorption factors. The uphill direction follows from grid
    detailed balance, K[j, i] = K[i, j] exp(-(E_j - E_i) / T), so T1 = T2
    gives exactly Fermi-Dirac occupations on the cell centers.
    """
    e = cl.e.reshape(-1)
    hw = cl.hw.reshape(-1)
    x = e[None, :] - e[:, None]
    ha, hb = hw[:, None], hw[None, :]
    cf = hw.max()
    m2 = (cl.ma, cl.mb)
    at = np.tile(A, (2, 2))
    K = np.zeros_like(x)
    lines, cohs = {}, {}
    for bt in baths:
        if bt.c not in cohs:
            cm = bt.cm
            cohs[bt.c] = np.block([[coh2(m2[i], m2[j], cm) for j in range(2)]
                                   for i in range(2)]) * at
        g = (2.0 * np.pi * bt.amp / bt.w0) * cohs[bt.c]
        if bt.w0 not in lines:
            # emission kernel; absorption is its transpose since line is even
            lines[bt.w0] = line(x - bt.w0, ha, hb, cf)
        le = lines[bt.w0]
        nb = bose(bt.w0, bt.t)
        p = g * ((1.0 + nb) * le + nb * le.T)
        if bt.t > 0:
            up = p.T * np.exp(np.minimum(x, 0.0) / bt.t)
        else:
            up = np.zeros_like(p)
        K += np.where(x >= 0.0, p, up)
    np.fill_diagonal(K, 0.0)
    return K
