"""Flat surface mode (SPP / SPhP plateau) on the s grid, next to the Chord phonons. Numpy.

R[i, j] = amp [rho^2 K0 - 2 d rho sx K1 + d^2 sx^2 K2] Fbar(|de|) Omega N(Omega) / (w_i w_j)
amp = 2 pi alpha c ws^2 fr, fr = 1 / (einf + ed) - 1 / (e0 + ed)  (Drude metal: e0 -> inf).
"""
from types import SimpleNamespace

import numpy as np

from EI_refactor.EI_ness import model as M
from EI_refactor.EI_ness.cpu.rates import therm

ALPHA, CT = 1 / 137.036, 3946.54


def ktab(ns, nf, z0, nq=96, nphi=32):
    """K_n(I, J) = int d^2q / (2 pi)^2 q^(n - 1) e^(-2 q z0) <1[k in I] 1[k + q in J]>, n = 0, 1, 2.
 
    Radial q up to 2 q z0 = 8, not folded: umklapp is included since cos is periodic.
    """
    dq = 8.0 / (2.0 * z0) / nq
    k = 2.0 * np.pi * np.arange(nf) / nf
    kx, ky = (a.ravel() for a in np.meshgrid(k, k, indexing="ij"))
    sb = np.linspace(-2.0, 2.0, ns + 1)
    bi = np.clip(np.digitize(np.cos(kx) + np.cos(ky), sb) - 1, 0, ns - 1) * ns  # bin of k
    K = np.zeros((3, ns * ns))
    for q in dq * (np.arange(nq) + 0.5):
        wq = q * dq / (2.0 * np.pi) * np.exp(-2.0 * q * z0) * np.array([1.0 / q, 1.0, q])
        for ph in 2.0 * np.pi * (np.arange(nphi) + 0.5) / nphi:
            sj = np.cos(kx + q * np.cos(ph)) + np.cos(ky + q * np.sin(ph))  # s of k + q
            c = np.bincount(bi + np.clip(np.digitize(sj, sb) - 1, 0, ns - 1), minlength=ns * ns)
            K += wq[:, None] * c
    K = K.reshape(3, ns, ns) / (nphi * nf**2)
    K = 0.5 * (K + K.transpose(0, 2, 1))
    return K
 

def ewidth(md, d, m, ns, nsub=9):
    """Energy width of every s bin, (2, ns)."""
    sb = np.linspace(-2, 2, ns + 1)
    x = sb[:-1, None] + (sb[1] - sb[0]) * np.linspace(0, 1, nsub)[None, :]
    e = M.mf(md, SimpleNamespace(s=x), d, m).e
    return np.ptp(e, axis=-1)


def fbar(om, a, ws, gam):
    """Pair of Lorentzians F (eq. 271) averaged over |Omega| +- a."""
    om = np.maximum(np.abs(om), 1e-12)
    a = np.maximum(a, 1e-9 * gam)
    p = lambda x: 2 * np.arctan((x - ws) / gam) - 2 * np.arctan((x + ws) / gam)
    return (p(om + a) - p(om - a)) / (2 * a * om * ws)


def rates(md, g, st, t, K, ws, gam, fr, dip, ew):
    """R[dest, src] of the flat surface mode on the s grid, states (alpha, beta) x bins."""
    qa = np.concatenate((st.u, st.v))
    qb = np.concatenate((-st.v, st.u))
    rho = qa[:, None] * qa[None, :] + qb[:, None] * qb[None, :]
    sx = qa[:, None] * qb[None, :] + qb[:, None] * qa[None, :]
    k0, k1, k2 = (np.tile(x, (2, 2)) for x in K)
    e, w, wd = st.e.reshape(-1), np.tile(g.w, 2), ew.reshape(-1)
    de = e[:, None] - e[None, :]
    a = 0.5 * np.hypot(wd[:, None], wd[None, :])  # box with the variance of the two bins
    amp = 2 * np.pi * ALPHA * CT * ws**2 * fr
    r = amp * (rho**2 * k0 - 2 * dip * rho * sx * k1 + dip**2 * sx**2 * k2)
    r = r * fbar(de, a, ws, gam) * therm(de, t) / (w[:, None] * w[None, :])
    return r * (1 - np.eye(e.size))


def gout(r, w):
    """sum_k' Gamma out of every state, split (intra, inter): (2, 2 ns)."""
    n = w.size
    ww = np.tile(w, 2)[:, None]
    blk = np.arange(2 * n) // n
    same = blk[:, None] == blk[None, :]
    return np.array([np.sum(r * ww * same, 0), np.sum(r * ww * ~same, 0)])