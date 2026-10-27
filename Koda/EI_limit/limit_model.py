"""Limit model of the bath driven excitonic insulator on the s grid.

States are (alpha s_0 .. s_N-1, beta s_0 .. s_N-1), s = cos kx + cos ky in [-2, 2],
n_gamma(k) = n_gamma(s(k)) (projection by the fast elastic phonon scattering).
Rate matrices use g[i, j] = rate from j to i and cell weights w (fraction of
k states, sum 1), so that dn_i/dt = sum_j w_j (g_ij n_j (1 - n_i) - g_ji n_i (1 - n_j)).

Photon bath (T1): vertical interband rates of the q -> 0 light cone, closed form
in the cavity width gam. Phonon bath (T2): intraband only, line 'fp' (gam -> 0
Fokker-Planck, c_s -> 0) or 'lorentz' (c_s -> 0 at finite gam). Cross branch
elastic terms (t'_a != -t'_b) are not included.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, root
from scipy.special import ellipe, ellipk, expit


@dataclass(frozen=True)
class Pars:
    gap: float
    ta: float
    tb: float
    v: float
    n: float = 1.0
    uh: float | None = None

    @property
    def h(self):
        """Hartree coupling, V unless uh is given."""
        return self.v if self.uh is None else self.uh


@dataclass(frozen=True)
class Grid:
    s: np.ndarray   # cell centres
    f: np.ndarray   # inner faces
    ds: float
    w: np.ndarray   # cell weights, int g ds over the cell, sum 1
    hb: np.ndarray  # cell average of |sin k|^2 (Phi / g)


def gs(s):
    """Density of states in s, g = K(kappa) / pi^2, kappa^2 = 1 - s^2 / 4 (eq. 211)."""
    with np.errstate(divide="ignore"):
        return ellipk(1.0 - 0.25 * np.square(s)) / np.pi**2


def phi(s):
    """int d2k / (2 pi)^2 |sin k|^2 delta(s - s_k) = 4 / pi^2 [E(kappa) - s^2 K(kappa) / 4]."""
    s = np.asarray(s, float)
    m = 1.0 - 0.25 * s * s
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(np.abs(s) < 1e-6, 0.0, 0.25 * s * s * ellipk(m))
    return 4.0 / np.pi**2 * (ellipe(m) - t)


def grid(ns):
    """Uniform cells in s with exact weights; ns odd keeps faces off the s = 0 van Hove point."""
    if ns % 2 == 0:
        raise ValueError("ns must be odd")
    e = np.linspace(-2.0, 2.0, ns + 1)
    w = np.array([quad(gs, a, b, points=[0.0] if a < 0 < b else None, limit=200)[0]
                  for a, b in zip(e[:-1], e[1:])])
    pw = np.array([quad(phi, a, b, limit=200)[0] for a, b in zip(e[:-1], e[1:])])
    return Grid(0.5 * (e[1:] + e[:-1]), e[1:-1], e[1] - e[0], w / w.sum(), pw / w)


def mf(p, s, d, m):
    """Hartree bands and rotation at s (eqs. 200 - 205), u v = d / 2E."""
    s = np.asarray(s, float)
    xi = 0.5 * (p.gap - p.h * m) - (p.ta - p.tb) * s
    e = np.maximum(np.hypot(xi, d), 1e-300)
    eta = 0.5 * p.h * p.n - (p.ta + p.tb) * s
    sg = 1.0 if d >= 0 else -1.0
    return SimpleNamespace(
        d=d, xi=xi, e=e, ea=eta + e, eb=eta - e,
        u=np.sqrt(0.5 * (1 + xi / e)), v=sg * np.sqrt(np.maximum(0.5 * (1 - xi / e), 0.0)),
        xi0=0.5 * p.gap - (p.ta - p.tb) * s,                      # bare, for the dipole
        da=-(p.ta + p.tb) - (p.ta - p.tb) * xi / e,               # dE_alpha / ds
        db=-(p.ta + p.tb) + (p.ta - p.tb) * xi / e)               # dE_beta / ds


def xe_cell(p, gr, d, m):
    """Cell average of xi / E (uniform in s inside a cell). Equal to the centre value up to
    O(ds^2) for |d| above the cell spread of xi, and continuous in m at d = 0, where the
    centre value is a sign function."""
    sl = p.ta - p.tb
    if abs(sl) < 1e-12:
        b = mf(p, gr.s, d, m)
        return b.xi / b.e
    xi = 0.5 * (p.gap - p.h * m) - sl * np.r_[-2.0, gr.f, 2.0]
    r = np.hypot(xi, d)
    return (r[:-1] - r[1:]) / (sl * gr.ds)


def fermi(x, t):
    """1 / (exp(x / t) + 1), step at t = 0."""
    x = np.asarray(x, float)
    if t <= 0:
        return (x < 0) + 0.5 * (x == 0)
    return expit(-x / t)


def bose(x, t):
    """N(x) for x > 0, zero at t = 0."""
    x = np.asarray(x, float)
    if t <= 0:
        return np.zeros_like(x)
    with np.errstate(over="ignore"):
        return 1.0 / np.expm1(x / t)


def xbose(x, t):
    """x N(x) for either sign of x, equal to t at x = 0 and to max(-x, 0) at t = 0."""
    x = np.asarray(x, float)
    if t <= 0:
        return np.maximum(-x, 0.0)
    y = x / t
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r = x / np.expm1(y)
    return np.where(np.abs(y) < 1e-12, t, r)


# photons ---------------------------------------------------------------------

def lineint(om, w0, gam):
    """I(om) = int_w0^inf dw (1 + w0^2 / w^2) [L(om - w) - L(om + w)], om > 0, closed form."""
    om = np.asarray(om, float)
    if gam <= 0:
        return 2 * np.pi * (1 + w0**2 / om**2) * (om > w0)
    th = (np.arctan((om - w0) / gam) + np.arctan((om + w0) / gam)) / np.pi
    if w0 <= 0:
        return 2 * np.pi * th

    def fz(a):
        return -np.log(1 - a / w0) / a**2 - 1 / (a * w0)

    return 2 * np.pi * th + 2 * w0**2 * np.imag(fz(om + 1j * gam) - fz(-om + 1j * gam))


def vertical(p, b, h2, pc, t):
    """Vertical photon rates per state, (up = beta -> alpha, down = alpha -> beta).

    b: mf at the points, h2: |sin k|^2 there (or its contour average hb),
    pc: photon parameters (amp, dip, c, w0, gam).
    """
    wk = (p.ta - p.tb)**2 * (b.d / b.e)**2 * h2 + 4 * pc["dip"]**2 * b.xi0**2
    om = 2 * b.e
    pre = pc["amp"] * wk * lineint(om, pc["w0"], pc["gam"]) / (4 * np.pi * pc["c"]**2)
    nb = bose(om, t)
    return pre * nb, pre * (1 + nb)


# phonons ---------------------------------------------------------------------

def ctil(qc, ui, vi, uj, vj):
    """C~ of the alpha alpha and beta beta channels (eqs. 239, 242 with t -> t')."""
    a, b = qc["ta_prime"], qc["tb_prime"]
    return 2 * (a * ui * uj + b * vi * vj), 2 * (a * vi * vj + b * ui * uj)


def fp_block(gr, p, d, m, qc, t):
    """Nearest neighbour rates of the gam -> 0 Fokker-Planck limit, one matrix per band.

    Flux J = -K [T dn/ds + E' n (1 - n)], K = 4 pi amp C~^2 g Phi / |E'|^3, discretised
    with face rates (K / ds) x N(x), x = energy gained, so FD at T is exact on the grid.
    """
    bf, bc = mf(p, gr.f, d, m), mf(p, gr.s, d, m)
    kf = 4 * np.pi * qc["amp"] * gs(gr.f) * phi(gr.f) / gr.ds
    i = np.arange(len(gr.f))
    ww = gr.w[i] * gr.w[i + 1]
    out = []
    for c, de, en in zip(ctil(qc, bf.u, bf.v, bf.u, bf.v), (bf.da, bf.db), (bc.ea, bc.eb)):
        r = kf * c**2 / np.maximum(np.abs(de), 1e-300)**3
        om = np.diff(en)
        g = np.zeros((len(gr.s),) * 2)
        g[i + 1, i] = r * xbose(om, t) / ww
        g[i, i + 1] = r * xbose(-om, t) / ww
        out.append(g)
    return out


def lorentz_block(gr, p, d, m, qc, t):
    """Pair rates of the c_s -> 0 kernel at finite gam, one matrix per band:
    g_ij = amp C~_ij^2 (hb_i + hb_j) x N(x) 8 gam / (x^2 + gam^2)^2, x = E_i - E_j."""
    b = mf(p, gr.s, d, m)
    hs = gr.hb[:, None] + gr.hb[None, :]
    out = []
    for c, en in zip(ctil(qc, b.u[:, None], b.v[:, None], b.u[None, :], b.v[None, :]), (b.ea, b.eb)):
        x = en[:, None] - en[None, :]
        g = qc["amp"] * c**2 * hs * xbose(x, t) * 8 * qc["gam"] / (x * x + qc["gam"]**2)**2
        np.fill_diagonal(g, 0.0)
        out.append(g)
    return out


def rates(gr, p, d, m, pc, qc, t1, t2):
    """Rate matrix g (2N x 2N, destination x source), weights (2N) and mf of both baths."""
    b = mf(p, gr.s, d, m)
    up, dn = vertical(p, b, gr.hb, pc, t1)
    ga, gb = (fp_block if qc["line"] == "fp" else lorentz_block)(gr, p, d, m, qc, t2)
    n = len(gr.s)
    g = np.zeros((2 * n, 2 * n))
    g[:n, :n], g[n:, n:] = ga, gb
    i = np.arange(n)
    g[i, n + i] = up / gr.w
    g[n + i, i] = dn / gr.w
    return g, np.tile(gr.w, 2), b


# k resolved versions, for comparisons --------------------------------------

def phonon_pair(p, d, m, qc, t, k, p0, cs=0.0):
    """Intraband alpha pair rate p0 -> k of the phonon kernel (amp W R), finite gam.
    cs = 0 gives the limit kernel, cs > 0 the full Lorentzian of eq. 186."""
    k, p0 = np.atleast_2d(k), np.asarray(p0, float)
    bk = mf(p, np.cos(k).sum(-1), d, m)
    bp = mf(p, np.cos(p0).sum(), d, m)
    c = ctil(qc, bk.u, bk.v, bp.u, bp.v)[0]
    w = np.sum((np.sin(k) - np.sin(p0))**2, -1) * c**2
    q = np.mod(k - p0 + np.pi, 2 * np.pi) - np.pi
    wq, x, g = cs * np.linalg.norm(q, axis=-1), bk.ea - bp.ea, qc["gam"]
    return qc["amp"] * w * xbose(x, t) * 8 * g / (((x - wq)**2 + g * g) * ((x + wq)**2 + g * g))


def phonon_moments(p, d, m, qc, t, p0, nkf=256):
    """sum_k w_k (E_k - E_p0)^j rate(p0 -> k), j = 1, 2, alpha band, limit kernel, fine grid."""
    k1 = -np.pi + 2 * np.pi * (np.arange(nkf) + 0.5) / nkf
    kk = np.stack(np.meshgrid(k1, k1, indexing="ij"), -1).reshape(-1, 2)
    r = phonon_pair(p, d, m, qc, t, kk, p0)
    x = mf(p, np.cos(kk).sum(-1), d, m).ea - mf(p, np.cos(p0).sum(), d, m).ea
    return np.mean(r * x), np.mean(r * x * x)


def fp_moments(p, d, m, qc, t, p0):
    """The same two moments in the gam -> 0, c_s -> 0 limit (same branch only):
    M1 = -2 pi X, M2 = 4 pi T X, X = amp C~^2 [Phi + g |sin p0|^2] / |E'|."""
    s0 = np.cos(p0).sum()
    b = mf(p, s0, d, m)
    c = ctil(qc, b.u, b.v, b.u, b.v)[0]
    x = qc["amp"] * c**2 * (phi(s0) + gs(s0) * np.sum(np.sin(p0)**2)) / abs(b.da)
    return -2 * np.pi * x, 4 * np.pi * t * x


# equilibrium -----------------------------------------------------------------

def mu_fill(gr, p, b, t):
    """Chemical potential of the filling p.n for the bands b at temperature t."""
    lo, hi = np.min(b.eb) - 1 - 40 * t, np.max(b.ea) + 1 + 40 * t
    return brentq(lambda mu: np.sum(gr.w * (fermi(b.ea - mu, t) + fermi(b.eb - mu, t))) - p.n,
                  lo, hi, xtol=1e-14)


def thermal(gr, p, t, d0=1.0, m0=0.0, fix_d=None):
    """Equilibrium (d, m, mu_c) at temperature t (eqs. 212 - 214); fix_d keeps d fixed."""
    def parts(d, m):
        b = mf(p, gr.s, d, m)
        mu = mu_fill(gr, p, b, t)
        return b, mu, fermi(b.ea - mu, t), fermi(b.eb - mu, t)

    def res(z):
        d, m = (fix_d, z[0]) if fix_d is not None else z
        b, _, fa, fb = parts(d, m)
        r = [m - np.sum(gr.w * xe_cell(p, gr, d, m) * (fa - fb))]
        return r if fix_d is not None else [d - p.v * np.sum(gr.w * d / (2 * b.e) * (fb - fa)), *r]

    r = root(res, [m0] if fix_d is not None else [d0, m0], method="hybr", options={"xtol": 1e-13})
    if not (r.success or np.max(np.abs(r.fun)) < 1e-11):
        raise RuntimeError(f"thermal state at t = {t}: {r.message}")
    d, m = (fix_d, r.x[0]) if fix_d is not None else r.x
    return d, m, parts(d, m)[1]


def tc(gr, p, tlo=1e-3):
    """Critical temperature from the linearised gap equation at d = 0+."""
    def lam(t):
        _, m, mu = thermal(gr, p, t, fix_d=0.0)
        b = mf(p, gr.s, 0.0, m)
        df = fermi(b.eb - mu, t) - fermi(b.ea - mu, t)
        return p.v * np.sum(gr.w * df / (2 * b.e)) - 1.0

    thi = tlo
    while lam(thi) > 0:
        thi *= 2
    return brentq(lam, 0.5 * thi, thi, xtol=1e-12)
