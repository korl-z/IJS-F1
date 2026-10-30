"""Bands, DOS and bath rates of the shell model (self contained, units t_a = 1 eV).

States are (alpha cells, beta cells). g[i, j] is the rate j -> i per state pair and w the
cell weights, so dn_i / dt = sum_j w_j (g_ij n_j (1 - n_i) - g_ji n_i (1 - n_j)).
Photon bath (T1): vertical interband rates of the q -> 0 light cone, closed form in gam.
Phonon bath (T2): intraband, c_s -> 0, Fokker-Planck ('fp') or finite gam kernel ('lorentz').
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.special import ellipe, ellipkm1, expit

KB = 1 / 11604.518          # k_B in t_a = 1 eV per Kelvin


def kel(x):
    """Energy in units of t_a to Kelvin."""
    return np.asarray(x) / KB


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
    s: np.ndarray    # cell centres
    f: np.ndarray    # inner faces
    ds: np.ndarray   # centre spacing across each inner face
    w: np.ndarray    # cell weights, int g ds over the cell
    hb: np.ndarray   # cell average of |sin k|^2
    xf: np.ndarray | None = None   # cell faces in xi


def g(s):
    """DOS in s = cos kx + cos ky, K(kappa) / pi^2, 1 - kappa^2 = s^2 / 4 (finite near s = 0)."""
    return ellipkm1(0.25 * np.square(s)) / np.pi**2


def phi(s):
    """int d2k / (2 pi)^2 |sin k|^2 delta(s - s_k) = 4 / pi^2 [E(kappa) - s^2 K(kappa) / 4]."""
    t = 0.25 * np.square(np.asarray(s, float))
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(t > 0, t * ellipkm1(t), 0.0)
    return 4 / np.pi**2 * (ellipe(1 - t) - k)


def mf(p, s, d, m):
    """Hartree bands and rotation at s, u v = d / 2E, dE / ds of both bands."""
    s = np.asarray(s, float)
    xi = 0.5 * (p.gap - p.h * m) - (p.ta - p.tb) * s
    e = np.maximum(np.hypot(xi, d), 1e-300)
    eta = 0.5 * p.h * p.n - (p.ta + p.tb) * s
    sg = 1.0 if d >= 0 else -1.0
    return SimpleNamespace(
        d=d, xi=xi, e=e, ea=eta + e, eb=eta - e,
        u=np.sqrt(0.5 * (1 + xi / e)), v=sg * np.sqrt(np.maximum(0.5 * (1 - xi / e), 0.0)),
        xi0=0.5 * p.gap - (p.ta - p.tb) * s,
        da=-(p.ta + p.tb) - (p.ta - p.tb) * xi / e,
        db=-(p.ta + p.tb) + (p.ta - p.tb) * xi / e)


def fd(x, t):
    """1 / (exp(x / t) + 1), overflow free."""
    return expit(-np.asarray(x) / t)


def bose(x, t):
    """N(x), x > 0."""
    with np.errstate(over="ignore"):
        return 1 / np.expm1(np.asarray(x) / t)


def xbose(x, t):
    """x N(x) for either sign of x, equal to t at x = 0."""
    x = np.asarray(x, float)
    y = x / t
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r = x / np.expm1(y)
    return np.where(np.abs(y) < 1e-12, t, r)


# photons --------------------------------------------------------------------------

def lineint(om, w0, gam):
    """int_w0^inf dw (1 + w0^2 / w^2) [L(om - w) - L(om + w)], om > 0, closed form."""
    om = np.asarray(om, float)
    th = (np.arctan((om - w0) / gam) + np.arctan((om + w0) / gam)) / np.pi
    if w0 <= 0:
        return 2 * np.pi * th

    def fz(a):
        return -np.log(1 - a / w0) / a**2 - 1 / (a * w0)

    return 2 * np.pi * th + 2 * w0**2 * np.imag(fz(om + 1j * gam) - fz(-om + 1j * gam))


def vertical(p, b, h2, pc, t):
    """Vertical photon rates per state (up = beta -> alpha, down = alpha -> beta)."""
    wk = (p.ta - p.tb)**2 * (b.d / b.e)**2 * h2 + 4 * pc["dip"]**2 * b.xi0**2
    om = 2 * b.e
    pre = pc["amp"] * wk * lineint(om, pc["w0"], pc["gam"]) / (4 * np.pi * pc["c"]**2)
    nb = bose(om, t)
    return pre * nb, pre * (1 + nb)


# phonons --------------------------------------------------------------------------

def ctil(qc, ui, vi, uj, vj):
    """C~ of the alpha alpha and beta beta channels."""
    a, b = qc["ta_prime"], qc["tb_prime"]
    return 2 * (a * ui * uj + b * vi * vj), 2 * (a * vi * vj + b * ui * uj)


def fp_block(gr, p, d, m, qc, t):
    """Nearest neighbour rates of the Fokker-Planck limit, one matrix per band. Face rates
    (K / ds) x N(x), K = 4 pi amp C~^2 g Phi / |E'|^3, so FD at t is exact on the grid."""
    bf, bc = mf(p, gr.f, d, m), mf(p, gr.s, d, m)
    kf = 4 * np.pi * qc["amp"] * g(gr.f) * phi(gr.f) / gr.ds
    i = np.arange(len(gr.f))
    ww = gr.w[i] * gr.w[i + 1]
    out = []
    for c, de, en in zip(ctil(qc, bf.u, bf.v, bf.u, bf.v), (bf.da, bf.db), (bc.ea, bc.eb)):
        r = kf * c**2 / np.maximum(np.abs(de), 1e-300)**3
        om = np.diff(en)
        gm = np.zeros((len(gr.s),) * 2)
        gm[i + 1, i] = r * xbose(om, t) / ww
        gm[i, i + 1] = r * xbose(-om, t) / ww
        out.append(gm)
    return out


def lorentz_block(gr, p, d, m, qc, t):
    """Pair rates of the c_s -> 0 kernel at finite gam, one matrix per band."""
    b = mf(p, gr.s, d, m)
    hs = gr.hb[:, None] + gr.hb[None, :]
    out = []
    for c, en in zip(ctil(qc, b.u[:, None], b.v[:, None], b.u[None, :], b.v[None, :]), (b.ea, b.eb)):
        x = en[:, None] - en[None, :]
        gm = qc["amp"] * c**2 * hs * xbose(x, t) * 8 * qc["gam"] / (x * x + qc["gam"]**2)**2
        np.fill_diagonal(gm, 0.0)
        out.append(gm)
    return out


def rates(gr, p, d, m, pc, qc, t1, t2):
    """Rate matrix (2N x 2N, destination x source) and weights (2N)."""
    b = mf(p, gr.s, d, m)
    up, dn = vertical(p, b, gr.hb, pc, t1)
    ga, gb = (fp_block if qc["line"] == "fp" else lorentz_block)(gr, p, d, m, qc, t2)
    n = len(gr.s)
    gm = np.zeros((2 * n, 2 * n))
    gm[:n, :n], gm[n:, n:] = ga, gb
    i = np.arange(n)
    gm[i, n + i] = up / gr.w
    gm[n + i, i] = dn / gr.w
    return gm, np.tile(gr.w, 2)
