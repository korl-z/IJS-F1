"""mu* model on the lattice with two pair channels: cavity photons (T1) and phonons (T2).

f = Fermi((E - mu*)/T2) (fast intraband phonons), mu* from the total pair balance
    sum_k J [N1 (1 - f)^2 - (1 + N1) f^2] + (1 - e^{2 mu*/T2}) sum_kk' W N2 (1 - f)(1 - f') = 0,
(Delta, m) from the exact lattice mean field equations on an s quadrature (no constant DOS).
W = interband phonon kernel from the cached P(s, s'; q) table of ness_fast (u free vertex).
Units t_a = a0 = hbar = kB = 1.
"""

import numpy as np
from scipy.special import expit
from scipy.optimize import brentq, root
import ness_fast as nf

# baths as in the yaml: photon (cavity) and phonon
B = dict(amp_ga=1.0, c=3945.0, w0=0.3, gam_ga=0.05,
         amp_ph=0.001432046318, tpa=-1.0e-3, tpb=1.0e-3, cs=0.026, gam_ph=0.1)


def kern(om, w0, gam):
    z4 = (om**2 + gam**2) ** 2
    return ((1 + w0**2 * (om**2 - gam**2) / z4) * (np.arctan((om - w0) / gam) + np.arctan((om + w0) / gam))
            + gam * om * w0**2 / z4 * np.log(((om - w0) ** 2 + gam**2) * ((om + w0) ** 2 + gam**2) / w0**4))


class Model:
    def __init__(self, g, v, b=B, geo=None, nfine=4000, ta=1.0, tb=-1.0):
        self.p = dict(gap=g, v=v, ta=ta, tb=tb)
        self.b = b
        self.s, self.w = nf.quad(nfine)  # fine quadrature, gap equations and photons
        sg, qg, pt, _ = nf.geometry(path="geo_ufree.npz") if geo is None else geo
        nk = sg.size
        kap = np.pi * (np.arange(nk) + 0.5) / nk
        self.sc, self.wc = sg, nf.dos(sg) * 2 * np.sin(kap) * np.pi / nk  # coarse nodes of the P table
        self.pt, self.qg = np.asarray(pt, float), qg
        self.wq = np.full(qg.size, qg[1])
        self.wq[[0, -1]] *= 0.5

    def bands(self, s, d, m):
        p = self.p
        return nf.mf(s, p["gap"], p["ta"], p["tb"], p["v"], d, m)

    def wph(self, d, m, t2, ecut=40.0):
        """Phonon pair kernel W N2 (per site, both weights) on active coarse nodes, E < E_min + ecut T2."""
        b = self.b
        _, e, u, vv = self.bands(self.sc, d, m)
        e = e[0]
        k = np.flatnonzero(e < e.min() + ecut * t2 + 0.2)
        om = e[k, None] + e[None, k]
        lz = lambda z: b["gam_ph"] / np.pi / (z**2 + b["gam_ph"] ** 2)
        wq = self.cs_q = b["cs"] * self.qg
        dg = (lz(om[..., None] - wq) - lz(om[..., None] + wq)) * self.wq / b["cs"]
        geo = np.einsum("ijl,ijl->ij", self.pt[np.ix_(k, k)], dg)
        m2 = (2 * (b["tpa"] * u[k, None] * vv[None, k] - b["tpb"] * vv[k, None] * u[None, k])) ** 2
        w2 = b["amp_ph"] * m2 * geo / np.expm1(om / t2) * self.wc[k, None] * self.wc[None, k]
        return e[k], w2

    def jph(self, e, u, vv, s):
        b, p = self.b, self.p
        return (b["amp_ga"] / (2 * np.pi * b["c"] ** 2) * (2 - 0.5 * s**2)
                * (2 * (p["ta"] - p["tb"]) * u * vv) ** 2 * kern(2 * e, b["w0"], b["gam_ga"])) * self.w

    def mustar(self, e, jj, ec, w2, t1, t2):
        n1 = 1 / np.expm1(2 * e / t1)

        def bal(mu):
            lf, lh = -np.logaddexp(0, (e - mu) / t2), -np.logaddexp(0, -(e - mu) / t2)
            gp = np.sum(jj * (n1 * np.exp(2 * lh) - (1 + n1) * np.exp(2 * lf)))  # photons
            hc = expit((ec - mu) / t2)  # 1 - f on the coarse nodes
            return gp - np.expm1(2 * mu / t2) * (hc @ w2 @ hc)  # phonons push mu* to 0

        return brentq(bal, e.min() - 100 * max(t1, t2), e.min() + 30 * t2)  # f(E_min) ~ 1 at the right end

    def gmap(self, t1, t2, d, m):
        """One pass of the mean field map (d, m) -> (d', m', mu*)."""
        p = self.p
        _, e, u, vv = self.bands(self.s, d, m)
        e = e[0]
        ec, w2 = self.wph(d, m, t2)
        mu = self.mustar(e, self.jph(e, u, vv, self.s), ec, w2, t1, t2)
        f = expit(-(e - mu) / t2)
        dn = p["v"] * np.sum(self.w * u * vv * (1 - 2 * f))
        mn = np.sum(self.w * (u**2 - vv**2) * (2 * f - 1))
        return dn, mn, mu

    def state(self, t1, t2, d, m, tol=1e-10, dmin=1e-6):
        """Self consistent (Delta, m, mu*) of the NESS from the start (d, m): hybrid root on (log d, m),
        damped fixed point if it fails. Delta below dmin is the normal state."""
        def res(z):
            dn, mn = self.gmap(t1, t2, np.exp(z[0]), z[1])[:2]
            return [np.log(max(dn, 1e-300)) - z[0], mn - z[1]]
        try:
            r = root(res, [np.log(max(d, dmin)), m], method="hybr", tol=tol)
            if r.success and np.exp(r.x[0]) > dmin:
                d, m = np.exp(r.x[0]), r.x[1]
                return d, m, self.gmap(t1, t2, d, m)[2]
        except ValueError:
            pass
        dl, ml = 10 * dmin, m  # normal state test: linear gain of the gap map at small d
        for _ in range(30):
            dn, ml, mu = self.gmap(t1, t2, dl, ml)
        if dn < dl:
            return 0.0, ml, mu
        for _ in range(3000):  # fallback, damped fixed point
            dn, mn, mu = self.gmap(t1, t2, d, m)
            if abs(dn - d) + abs(mn - m) < tol or dn < dmin:
                return (dn if dn >= dmin else 0.0), mn, mu
            d, m = d + 0.5 * (dn - d), m + 0.5 * (mn - m)
        return d, m, mu
