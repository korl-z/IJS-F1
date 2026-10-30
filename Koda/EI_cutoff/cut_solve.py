"""Shell (cutoff) formulation at a fixed T = 0 gap d0: vacuum, equilibrium, Tc and NESS.

Excitations n_alpha and h_beta = 1 - n_beta live in the shell |xi| < lam; outside it they are
those of the T = 0 vacuum (zero, error ~ exp(-lam / T)). The full band enters only through
1D vacuum integrals over the DOS, and V is fixed once by d0 (zero):
  lvac(d, m) = int g [1 / 2E - 1 / 2E0] = sum_shell w <1 / 2E> (n_alpha + h_beta)
  m = mvac(d, m) + sum_shell w <xi / E> (n_alpha + h_beta),  mvac = -int g xi / E
  sum_shell w (n_alpha - h_beta) = n - 1 = 0
Shell cells are fixed in xi = 0.5 (gap - h m) - (ta - tb) s and move with m in s.
n = 1 / (1 + exp(mu)); n and 1 - n are both built from mu to keep the tails.
"""
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, root
from scipy.special import expit, log_expit, logsumexp, softmax

import cut_model as cm
from cut_model import fd, g

GL = np.polynomial.legendre.leggauss(8)


def xc(p, m):
    """xi at s = 0; the ring is at s* = xc / (ta - tb)."""
    return 0.5 * (p.gap - p.h * m)


def qx(sl, x0, fun, a, pts=()):
    """int_-2^2 g(s) fun(xi, s) ds with xi = x0 - sl s, in u = asinh(xi / a): structure of
    width a at the ring becomes O(1) wide. pts: extra xi break points."""
    lo, hi = np.arcsinh((x0 - 2 * sl) / a), np.arcsinh((x0 + 2 * sl) / a)

    def f(u):
        x = a * np.sinh(u)
        return g((x0 - x) / sl) * fun(x, (x0 - x) / sl) * a * np.cosh(u) / sl

    br = np.arcsinh(np.r_[x0, 0.0, pts] / a)          # van Hove, ring, extra
    br = np.unique(br[(br > lo) & (br < hi)])
    return quad(f, lo, hi, points=br, limit=500, epsabs=1e-13, epsrel=1e-11)[0]


# T = 0 vacuum ---------------------------------------------------------------------

def roots(gap, ta, tb, d0, uh=None, dx=0.1):
    """All T = 0 states with gap d0 (n = 1, no excitations), sorted by V: the roots of the
    Hartree condition x = h m0(x), bracketed on a scan of x. c.edge is None if the bands
    cross (ring), otherwise the band edge s = +-2 where |xi| is smallest (no ring)."""
    sl = ta - tb

    def parts(x):                                   # x = h m0, Hartree shift of the ring
        x0 = 0.5 * (gap - x)
        m0 = -qx(sl, x0, lambda y, s: y / np.hypot(y, d0), d0)
        return m0, 1 / qx(sl, x0, lambda y, s: 0.5 / np.hypot(y, d0), d0)

    def fx(x):
        m0, v = parts(x)
        return x - (v if uh is None else uh) * m0

    xm = 2 * (abs(gap) + 4 * abs(sl)) + 2 if uh is None else abs(uh) + 1
    xs = np.arange(-xm, xm + dx, dx) + 0.37 * dx           # offset: no scan point puts the ring on s = 0
    fs = np.array([fx(x) for x in xs])
    out = []
    for i in np.nonzero(fs[:-1] * fs[1:] <= 0)[0]:
        m0, v = parts(brentq(fx, xs[i], xs[i + 1], xtol=1e-14))
        p = cm.Pars(gap, ta, tb, v, 1.0, uh)
        b = cm.mf(p, np.linspace(-2, 2, 4001), d0, m0)
        x0 = xc(p, m0)
        out.append(SimpleNamespace(p=p, d0=d0, m0=m0, pockets=b.ea.min() <= b.eb.max(),
                                   edge=None if abs(x0) < 2 * abs(sl) else 2.0 * np.sign(x0 * sl)))
    return sorted(out, key=lambda c: c.p.v)


def zero(gap, ta, tb, d0, uh=None, branch="bcs"):
    """T = 0 state with gap d0 on the BCS (smallest V) or BEC (largest V) root."""
    cz = roots(gap, ta, tb, d0, uh)
    if not cz:
        raise ValueError(f"no T = 0 state with d0 = {d0}")
    c = cz[0] if branch == "bcs" else cz[-1]
    if c.pockets:
        raise ValueError("no indirect gap: the T = 0 state has Fermi pockets")
    c.branch = branch
    return c


def lvac(c, d, m):
    """int g [1 / 2E(d, m) - 1 / 2E(d0, m0)] over the whole band (d > 0)."""
    p, dx = c.p, 0.5 * c.p.h * (m - c.m0)             # xi0 - xi
    return qx(p.ta - p.tb, xc(p, m),
              lambda y, s: 0.5 / np.hypot(y, d) - 0.5 / np.hypot(y + dx, c.d0),
              min(abs(d), c.d0), pts=[-dx])


def mvac(c, d, m):
    """-int g xi / E over the whole band (the m equation of the vacuum)."""
    p = c.p
    return -qx(p.ta - p.tb, xc(p, m), lambda y, s: y / max(np.hypot(y, d), 1e-300),
               max(abs(d), 1e-3 * c.d0))


# grid free equilibrium and Tc ------------------------------------------------------

def exc(c, t, d, m, mc):
    """int g of X / 2E (d > 0 only), X xi / E and Y; X = n_alpha + h_beta, Y = n_alpha - h_beta."""
    p = c.p

    def nh(y, s):
        e = np.hypot(y, d)
        a = 0.5 * p.h * p.n - (p.ta + p.tb) * s - mc   # eta - mc
        return e, fd(a + e, t), fd(e - a, t)

    fs = [lambda y, e, na, hb: (na + hb) * y / max(e, 1e-300), lambda y, e, na, hb: na - hb]
    if d > 0:
        fs.insert(0, lambda y, e, na, hb: 0.5 * (na + hb) / e)
    a = min(abs(d), t) if d > 0 else t
    return [qx(p.ta - p.tb, xc(p, m), lambda y, s, f=f: f(y, *nh(y, s)), a) for f in fs]


def thermal(c, t, d=None, m=None):
    """Grid free equilibrium (d, m, mc) at t; d = 0 gives the normal state."""
    p = c.p
    z0 = [c.m0 if m is None else m, 0.5 * p.h * p.n]
    if d == 0:
        def res(z):
            i2, i3 = exc(c, t, 0.0, *z)
            return [z[0] - mvac(c, 0.0, z[0]) - i2, i3]
    else:
        z0 = [c.d0 if d is None else d, *z0]

        def res(z):
            i1, i2, i3 = exc(c, t, *z)
            return [p.v * (lvac(c, z[0], z[1]) - i1), z[1] - mvac(c, z[0], z[1]) - i2, i3]

    r = root(res, z0, method="hybr", options={"xtol": 1e-12})
    if np.max(np.abs(r.fun)) > 1e-9:
        raise RuntimeError(f"thermal at t = {t}: {r.message}")
    return (0.0, *r.x) if d == 0 else tuple(r.x)


def tc(c, lo=0.05, hi=2.0):
    """Tc from the subtracted Thouless equation (no grid):
    V int g [(n_beta - n_alpha) / 2|xi| (normal state at t) - 1 / 2E0] = 0."""
    p, sl, z = c.p, c.p.ta - c.p.tb, {"m": c.m0}

    def lam(t):
        _, m, mc = thermal(c, t, d=0, m=z["m"])
        z["m"] = m
        dx = 0.5 * p.h * (m - c.m0)

        def k(y, s):
            e, a = abs(y), 0.5 * p.h * p.n - (p.ta + p.tb) * s - mc
            q = (fd(a - e, t) - fd(a + e, t)) / (2 * e) if e > 1e-9 * t else fd(a, t) * fd(-a, t) / t
            return q - 0.5 / np.hypot(y + dx, c.d0)

        return p.v * qx(sl, xc(p, m), k, t, pts=[-dx])

    lo, hi = lo * c.d0, hi * c.d0
    while lam(hi) > 0:
        hi *= 2
    while lam(lo) < 0:
        lo /= 2
    z["m"] = c.m0
    return brentq(lam, lo, hi, xtol=1e-14 * c.d0, rtol=1e-12)


# shell grid --------------------------------------------------------------------------

def faces(c, lam, a, n=401):
    """Cell faces as xi offsets from the anchor, sinh spaced (~ a du near it, ~ offset du out).
    Ring (BCS): n odd cells on -lam..lam. Band edge (BEC): n cells on 0..lam, at most the band."""
    if c.edge is None:
        if n % 2 == 0:
            raise ValueError("n must be odd (no face on the ring)")
        u = np.arcsinh(lam / a)
        return a * np.sinh(np.linspace(-u, u, n + 1))
    lam = min(lam, 4 * abs(c.p.ta - c.p.tb))
    return a * np.sinh(np.linspace(0.0, np.arcsinh(lam / a), n + 1))


def cells(c, xf, m):
    """Cells at m: s centres, inner faces, centre spacing, weights, hb, xi faces.
    Ring: offsets from xi = 0, the cells move with the ring in s. Edge: offsets from the band
    edge, the cells stay fixed in s and their xi moves with m."""
    p = c.p
    sl, x0 = p.ta - p.tb, xc(p, m)
    sf = (x0 - xf) / sl if c.edge is None else c.edge - np.sign(c.edge) * xf / abs(sl)
    if sf.max() > 2 + 1e-12 or sf.min() < -2 - 1e-12:
        raise ValueError("the shell leaves the band, reduce lam")
    sf = np.clip(sf, -2.0, 2.0)
    lo, hi = np.minimum(sf[1:], sf[:-1]), np.maximum(sf[1:], sf[:-1])
    s = 0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * GL[0]
    w = 0.5 * (hi - lo) * (g(s) @ GL[1])
    hb = 0.5 * (hi - lo) * (cm.phi(s) @ GL[1]) / w
    for j in np.nonzero((lo < 0) & (hi > 0))[0]:         # cell on the van Hove point
        w[j] = quad(g, lo[j], hi[j], points=[0.0])[0]
        hb[j] = quad(cm.phi, lo[j], hi[j])[0] / w[j]
    sc = 0.5 * (sf[1:] + sf[:-1])
    return cm.Grid(sc, sf[1:-1], np.abs(np.diff(sc)), w, hb, x0 - sl * sf)


def avg(xf, d):
    """Cell averages <1 / 2E> and <xi / E> in xi (n constant in a cell)."""
    a, b = xf[:-1], xf[1:]
    return ((np.arcsinh(b / d) - np.arcsinh(a / d)) / (2 * (b - a)),
            (np.hypot(b, d) - np.hypot(a, d)) / (b - a))


def thermal_shell(c, xf, t, d, m):
    """Equilibrium on the shell grid, FD at the cell centres (the exact NESS at t1 = t2)."""
    p = c.p

    def state(d, m):
        gr = cells(c, xf, m)
        b = cm.mf(p, gr.s, d, m)
        lo, hi = b.eb.min() - 1, b.ea.max() + 1
        mc = brentq(lambda x: np.sum(gr.w * (fd(b.ea - x, t) - fd(x - b.eb, t))), lo, hi, xtol=1e-15)
        return gr, np.r_[b.ea - mc, b.eb - mc] / t

    def res(z):
        gr, mu = state(*z)
        n, nb = occ(mu)
        ns = len(gr.s)
        ie, xe = avg(gr.xf, z[0])
        x = n[:ns] + nb[ns:]
        return [p.v * (lvac(c, *z) - np.sum(gr.w * ie * x)), z[1] - mvac(c, *z) - np.sum(gr.w * xe * x)]

    r = root(res, [d, m], method="hybr", options={"xtol": 1e-12})
    if np.max(np.abs(r.fun)) > 1e-9:
        raise RuntimeError(f"shell thermal at t = {t}: {r.message}")
    return r.x[0], r.x[1], state(*r.x)[1]


# NESS ----------------------------------------------------------------------------------

def occ(mu):
    """n and 1 - n from mu."""
    return expit(-mu), expit(mu)


def balance(gm, w, n, nb):
    """Gain a and loss b of every state, dn / dt = (1 - n) a - n b."""
    return gm @ (w * n), gm.T @ (w * nb)


def target_mu(a, b, mu):
    """mu_t = ln b - ln a (n = a / (a + b)); states without rates keep mu."""
    mt = np.clip(np.log(np.maximum(b, 1e-300)) - np.log(np.maximum(a, 1e-300)), -700, 700)
    return np.where((a > 0) | (b > 0), mt, mu)


def jlog(gm, w, mu):
    """ln (sum of photon beta -> alpha terms / sum of alpha -> beta terms), tail safe."""
    ns, i = len(w) // 2, np.arange(len(w) // 2)
    lw = 2 * np.log(w[:ns])
    lu = lw + np.log(gm[i, ns + i]) + log_expit(-mu[ns:]) + log_expit(mu[:ns])
    ld = lw + np.log(gm[ns + i, i]) + log_expit(-mu[:ns]) + log_expit(mu[ns:])
    return logsumexp(lu) - logsumexp(ld), lu, ld


def jgrad(lu, ld, n, nb):
    """Gradient of jlog in mu. The row fixes the pair number N_alpha + H_beta, which the
    phonons conserve and the mu rows see only at photon / phonon ~ 1e-16 on fine cells."""
    ns = len(lu)
    pu, pd = softmax(lu), softmax(ld)
    return np.r_[pu * n[:ns] + pd * nb[:ns], -pu * nb[ns:] - pd * n[ns:]]


def resid(z, c, xf, pc, qc, t1, t2):
    """Rows: mu - mu_t (2N), filling, subtracted gap (times V), m equation, jlog."""
    k = len(z) - 2
    ns, mu, d, m = k // 2, z[:k], z[k], z[k + 1]
    gr = cells(c, xf, m)
    gm, w = cm.rates(gr, c.p, d, m, pc, qc, t1, t2)
    n, nb = occ(mu)
    a, bl = balance(gm, w, n, nb)
    ie, xe = avg(gr.xf, d)
    x = n[:ns] + nb[ns:]                                  # n_alpha + h_beta
    jl, lu, ld = jlog(gm, w, mu)
    r = np.r_[mu - target_mu(a, bl, mu),
              np.sum(gr.w * (n[:ns] - nb[ns:])),
              c.p.v * (lvac(c, d, m) - np.sum(gr.w * ie * x)),
              m - mvac(c, d, m) - np.sum(gr.w * xe * x),
              jl]
    return r, (gm, w, n, nb, a, bl, gr, ie, xe, lu, ld)


def newton(c, xf, pc, qc, t1, t2, mu, d, m, tol=1e-10, nmax=40, cap=10.0):
    """Newton on z = (mu, d, m): analytic mu columns, d and m columns by forward differences,
    least squares step capped at cap in mu, backtracking on the residual norm."""
    k, ns = len(mu), len(mu) // 2
    z = np.r_[mu, d, m]
    r, aux = resid(z, c, xf, pc, qc, t1, t2)
    for it in range(nmax):
        if np.max(np.abs(r)) < tol:
            break
        gm, w, n, nb, a, bl, gr, ie, xe, lu, ld = aux
        nn = n * nb
        act = ((a > 0) | (bl > 0))[:, None]
        jac = np.zeros((k + 4, k + 2))
        jac[:k, :k] = np.eye(k) - act * (gm.T / np.maximum(bl, 1e-300)[:, None]
                                         + gm / np.maximum(a, 1e-300)[:, None]) * (w * nn)[None, :]
        cg, cx = c.p.v * gr.w * ie, gr.w * xe
        jac[k, :k] = -w * nn
        jac[k + 1, :k] = np.r_[cg * nn[:ns], -cg * nn[ns:]]
        jac[k + 2, :k] = np.r_[cx * nn[:ns], -cx * nn[ns:]]
        jac[k + 3, :k] = jgrad(lu, ld, n, nb)
        for i in (k, k + 1):
            zp = z.copy()
            zp[i] += 1e-7 * c.d0
            jac[:, i] = (resid(zp, c, xf, pc, qc, t1, t2)[0] - r) / (1e-7 * c.d0)
        step = np.linalg.lstsq(jac, -r, rcond=None)[0]
        step *= min(1.0, cap / max(np.max(np.abs(step[:k])), 1e-300))
        lam, nr = 1.0, np.linalg.norm(r)
        while True:
            rt, at = resid(z + lam * step, c, xf, pc, qc, t1, t2)
            if np.linalg.norm(rt) < nr or lam < 1e-4:
                break
            lam *= 0.5
        z, r, aux = z + lam * step, rt, at
    return SimpleNamespace(mu=z[:k], n=aux[2], d=z[k], m=z[k + 1], err=np.max(np.abs(r)), it=it)


def ness(c, xf, pc, qc, t1, t2, d=None, m=None, tol=1e-10):
    """NESS by Newton, continued in 1 / T1 from the shell equilibrium at T2 (ordered branch).
    sol.t1 is the T1 reached (below t1 if the ordered branch ended)."""
    d, m, mu = thermal_shell(c, xf, t2, c.d0 if d is None else d, c.m0 if m is None else m)
    sol = SimpleNamespace(mu=mu, n=occ(mu)[0], d=d, m=m, err=0.0, it=0)
    s, ds, t1r = 0.0, 0.1, t2
    while s < 1 and ds > 1e-3:
        st = min(1.0, s + ds)
        t = 1 / ((1 - st) / t2 + st / t1)
        new = newton(c, xf, pc, qc, t, t2, sol.mu, sol.d, sol.m, tol=tol)
        if new.err < tol and new.d > 1e-6 * c.d0:
            sol, s, ds, t1r = new, st, min(2 * ds, 0.25), t
        else:
            ds *= 0.5
    sol.t1 = t1r
    return sol
