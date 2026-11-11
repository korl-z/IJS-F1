"""Fast NESS of the s (xi) reduced master equation, no k grid.

u free vertices: V = 4 - 2 (cos qx + cos qy) - (s - s')^2 / 2, photon h = 2 - s^2 / 2.
Geometry: P(s, s'; q) = int d^2k/(2 pi)^2 delta(s - s_k) <V delta(s' - s_k+q)>_phi from contour
quadrature (no grid), tabulated once on s = 2 cos(kappa) and cached to disk.
Kinetics: energy adapted s bins, bin averaged kernels (trapezoid of E_i - E_j), exact detailed
balance, inner Newton in logit(n), outer Broyden on (d, m) with a bracketing fallback in d.
Units t_a = a0 = hbar = kB = 1.

    geo = geometry()                      # once, cached in geo_ufree.npz (~8 s the first time)
    z, x, st, hist = solve(1.0, 0.05, P, geo, path="ness.png")
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, log_expit, ellipk
from scipy.ndimage import gaussian_filter1d

P = dict(gap=4.0, ta=1.0, tb=-1.0, v=4.0, amp_ph=1 / (2 * 598.0 * 200), tpa=-1.0e-3, tpb=1.0e-3,
         cs=0.026, gam_ph=0.0005, qd=np.sqrt(4 * np.pi), w0=0.1, gam_ga=0.01, c=3945.0,
         amp_ga=4 / 137.036 * 0.1)


def dos(s):
    """Shell DOS g(s) of the square lattice, int g ds = 1."""
    return ellipk(1 - s**2 / 4) / np.pi**2


def mf(s, gap, ta, tb, v, d, m, n=1.0):
    """Hartree shifted bands, xi, E (alpha, beta) and u, v at s."""
    na, nb = 0.5 * (n + m), 0.5 * (n - m)
    eah = 0.5 * gap - 2 * ta * s + v * nb - 0.5 * v
    ebh = -0.5 * gap - 2 * tb * s + v * na - 0.5 * v
    xi, av = 0.5 * (eah - ebh), 0.5 * (eah + ebh)
    ek = np.hypot(xi, d)
    u, vv = np.sqrt(0.5 * (1 + xi / ek)), np.sqrt(0.5 * (1 - xi / ek))
    return xi, np.stack((av + ek, av - ek)), u, vv


def bose(x, t):
    return 1 / np.expm1(x / t)


# ---------------- geometry (once, cached) ----------------

def contour(s, nt, eighth=False):
    """Contour nodes of shells s: sum w f = int d^2k/(2 pi)^2 delta(s - s_k) f.

    eighth: nodes with 0 <= ky <= kx, weights x 8, else one quadrant (signs added by caller).
    """
    s = np.atleast_1d(s)[:, None]
    if eighth:
        th, dth, mult = np.pi / 2 + 0.5 * np.pi * (np.arange(nt) + 0.5) / nt, 0.5 * np.pi / nt, 8.0
    else:
        th, dth, mult = np.pi * (np.arange(nt) + 0.5) / nt, np.pi / nt, 1.0
    rm = 1 - np.abs(s) / 2
    kx = np.arccos(np.clip(s / 2 + rm * np.cos(th), -1, 1))
    ky = np.arccos(np.clip(s / 2 - rm * np.cos(th), -1, 1))
    w = mult * rm * np.sin(th) * dth / (np.sin(kx) * np.sin(ky)) / (2 * np.pi) ** 2
    return kx, ky, w


def geometry(nk=256, nq=96, qd=np.sqrt(4 * np.pi), na=16, nb=64, sig=1.0, path="geo_ufree.npz"):
    """P[i, j, l] = P(s_i, s_j; q_l) on s_i = 2 cos(kappa_i), and moments int q^n P dq, n = 1, 3, 5.

    Pairs of contour nodes (a on 1/8 of shell i, b on all of shell j) are deposited on the q grid,
    P = 2 pi Pi / q, light Gaussian smoothing in q.
    """
    if path and os.path.exists(path):
        z = np.load(path)
        return z["sg"], z["qg"], z["P"], z["mom"]
    kap = np.pi * (np.arange(nk) + 0.5) / nk
    sg = 2 * np.cos(kap)
    qg = np.linspace(0, qd, nq)
    dq = qg[1]
    bx, by, bw = contour(sg, nb)
    sgn = ((1, 1), (-1, 1), (1, -1), (-1, -1))
    bx = np.concatenate([a * bx for a, _ in sgn], 1)
    by = np.concatenate([b * by for _, b in sgn], 1)
    bw = np.tile(bw, 4)
    pi = np.zeros((nk, nk, nq))
    for i in range(nk):  # shells j >= i, mirrored below
        ax, ay, aw = (z[0] for z in contour(sg[i], na, eighth=True))
        qx = np.mod(bx[i:][None] - ax[:, None, None] + np.pi, 2 * np.pi) - np.pi
        qy = np.mod(by[i:][None] - ay[:, None, None] + np.pi, 2 * np.pi) - np.pi
        q = np.hypot(qx, qy)
        vt = 4 - 2 * (np.cos(qx) + np.cos(qy)) - 0.5 * (sg[i:][None, :, None] - sg[i]) ** 2
        ww = aw[:, None, None] * bw[i:][None] * vt * (q <= qd)
        z = q / dq
        l0 = np.minimum(z.astype(int), nq - 2)
        fz = z - l0
        idx = np.arange(nk - i)[None, :, None] * nq + l0
        n = (nk - i) * nq
        dep = np.bincount(idx.ravel(), ((1 - fz) * ww).ravel(), n)
        dep += np.bincount((idx + 1).ravel(), (fz * ww).ravel(), n)
        pi[i, i:] = dep.reshape(nk - i, nq) / dq
        pi[i:, i] = pi[i, i:]
    pi = gaussian_filter1d(pi, sig, axis=2, mode="mirror")
    Pq = np.where(qg > 0, 2 * np.pi * pi / np.maximum(qg, 1e-30), 0.0)
    mom = np.stack([np.trapezoid(Pq * qg**n, qg, axis=2) for n in (1, 3, 5)])
    if path:
        np.savez(path, sg=sg, qg=qg, P=Pq.astype(np.float32), mom=mom)
    return sg, qg, Pq.astype(np.float32), mom


def lerp2(tab, sg, si, sj):
    """Bilinear interpolation of tab[i, j, ...] at shells (si, sj), tab on s = 2 cos(kappa)."""
    nk = sg.size
    fi = np.clip(np.arccos(np.clip(si / 2, -1, 1)) / (np.pi / nk) - 0.5, 0, nk - 1 - 1e-9)
    fj = np.clip(np.arccos(np.clip(sj / 2, -1, 1)) / (np.pi / nk) - 0.5, 0, nk - 1 - 1e-9)
    i0, j0 = fi.astype(int), fj.astype(int)
    a, b = (fi - i0)[..., None], (fj - j0)[..., None]
    if tab.ndim == 2:
        a, b = a[..., 0], b[..., 0]
    return ((1 - a) * (1 - b) * tab[i0, j0] + a * (1 - b) * tab[i0 + 1, j0]
            + (1 - a) * b * tab[i0, j0 + 1] + a * b * tab[i0 + 1, j0 + 1])


# ---------------- bins and per solve setup ----------------

def quad(n=1500):
    """Global s quadrature, s = 2 cos(kappa), weights g(s) ds."""
    kap = np.pi * (np.arange(n) + 0.5) / n
    s = 2 * np.cos(kap)
    w = dos(s) * np.sin(kap)
    return s, w / w.sum()


def occ(e, w, t):
    if np.allclose(e[0], -e[1]):  # particle hole symmetric bands, mu = 0
        return expit(-e / t)
    lo, hi = e.min() - 1, e.max() + 1
    for _ in range(55):
        mu = 0.5 * (lo + hi)
        lo, hi = (mu, hi) if np.sum(w * expit(-(e - mu) / t).sum(0)) < 1 else (lo, mu)
    return expit(-(e - mu) / t)


TC = {}


def tc(p, lo=1e-3, hi=3.0, eps=1e-7):
    """Mean field Tc from the linearized gap equation, on the global quadrature (cached)."""
    key = (p["gap"], p["ta"], p["tb"], p["v"])
    if key in TC:
        return TC[key]
    s, w = quad()
    m = 0.0
    for _ in range(30):
        tm = np.sqrt(lo * hi)
        for _ in range(300):
            _, e, u, vv = mf(s, p["gap"], p["ta"], p["tb"], p["v"], eps, m)
            nq = occ(e, w, tm)
            mt = np.sum(w * (u**2 - vv**2) * (nq[0] - nq[1]))
            if abs(mt - m) < 1e-12:
                break
            m = 0.5 * (m + mt)
        x = p["v"] * np.sum(w * u * vv * (nq[1] - nq[0])) / eps
        lo, hi = (tm, hi) if x > 1 else (lo, tm)
    TC[key] = np.sqrt(lo * hi)
    return TC[key]


def equilibrium(p, t, d=1.0, m=-0.9):
    """Mean field (d, m) at temperature t, only a start for the outer Newton (tolerance 1e-7)."""
    s, w = quad()
    for _ in range(400):
        _, e, u, vv = mf(s, p["gap"], p["ta"], p["tb"], p["v"], d, m)
        nq = occ(e, w, t)
        dt = p["v"] * np.sum(w * u * vv * (nq[1] - nq[0]))
        mt = np.sum(w * (u**2 - vv**2) * (nq[0] - nq[1]))
        if abs(dt - d) + abs(mt - m) < 1e-7:
            break
        d, m = d + 0.5 * (dt - d), m + 0.5 * (mt - m)
    return max(d, 1e-4), m


def edges(p, d0, m0, t1, t2, r=0.12, c1=12.0, c2=25.0):
    """s bin edges, designed for any gap <= d0: xi steps r |xi - xi_m|, at least min(T) / 2,
    at most omega_D / 4, xi_m = energy minimum (0 or band edge); cut at |xi - xi_m| = d0 + max(c1 T1, c2 T2) + 2 omega_D."""
    k, wd = p["ta"] - p["tb"], p["cs"] * p["qd"]
    x0 = 0.5 * p["gap"] - 0.5 * p["v"] * m0  # xi at s = 0
    xl, xh = x0 - 2 * k, x0 + 2 * k  # xi at s = 2, -2
    xm = min(max(0.0, xl), xh)
    ycut = d0 + max(c1 * t1, c2 * t2) + 2 * wd
    dmin, dmax = 0.5 * min(t1, t2), wd / 4
    xs = [xm]
    for sg, xe in ((1, xh), (-1, xl)):
        y, ye = 0.0, min(abs(xe - xm), ycut)
        while ye - y > 1e-12:
            dy = min(dmax, max(dmin, r * y))
            y = ye if ye - y < 1.3 * dy else y + dy
            xs.append(xm + sg * y)
    return np.sort((x0 - np.array(xs)) / k)


def setup(geo, p, t1, t2, d0, m0, ngl=8, **kw):
    """Bins, bin weights, exterior quadrature, near pairs with their P(q) and cumulative integrals."""
    sg, qg, Pt, mom = geo
    se = edges(p, d0, m0, t1, t2, **kw)
    lo, hi = se[:-1], se[1:]
    h, sc = hi - lo, 0.5 * (lo + hi)
    xg, wg = np.polynomial.legendre.leggauss(ngl)
    sk = sc[:, None] + 0.5 * h[:, None] * xg  # sub nodes for the gap equations
    gk = dos(sk) * 0.5 * h[:, None] * wg
    w = gk.sum(1)
    sq, wq = quad(20000)
    out = (sq < lo[0]) | (sq > hi[-1])  # exterior, alpha empty, beta full
    ns = sc.size

    # intraband pairs (i < j) reachable by one phonon for any gap up to ~d0, and interband
    # pairs close to the shell (|xi_i| + |xi_j| < 3 omega_D), both with P(q) on the q grid
    k = p["ta"] - p["tb"]
    xi = 0.5 * p["gap"] - k * sc - 0.5 * p["v"] * m0
    e = np.hypot(xi, 1.2 * d0 + 0.01)
    xe = 0.5 * p["gap"] - k * se - 0.5 * p["v"] * m0
    a = np.abs(np.diff(np.abs(xe)))
    wd = p["cs"] * p["qd"]
    I, J = np.meshgrid(np.arange(ns), np.arange(ns), indexing="ij")
    n1 = (np.abs(e[:, None] - e[None]) - a[:, None] - a[None] <= 1.2 * wd) & (I < J)
    n2 = np.abs(xi)[:, None] + np.abs(xi)[None] - a[:, None] - a[None] <= 3 * wd
    i1, j1, i2, j2 = I[n1], J[n1], I[n2], J[n2]
    nb = bose(np.maximum(p["cs"] * qg, 1e-300), t2)

    def tabs(i, j):
        pv = lerp2(Pt, sg, sc[i], sc[j]).astype(float)
        fn = pv * nb
        fn[:, 0] = pv[:, 1] / qg[1] * t2 / p["cs"]  # P N -> P'(0) T / cs at q = 0
        return pv, cums(fn, qg), cums(fn + pv, qg)

    pv1, fn1, fe1 = tabs(i1, j1)
    pv2, _, fe2 = tabs(i2, j2)
    st = dict(p=p, t1=t1, t2=t2, se=se, sc=sc, h=h, w=w, sk=sk, gk=gk, sq=sq[out], wq=wq[out], qg=qg,
              i1=i1, j1=j1, fn1=fn1, fe1=fe1, i2=i2, j2=j2, pv2=pv2, fe2=fe2,
              mom=lerp2(np.moveaxis(mom, 0, -1), sg, sc[:, None], sc[None]))
    return st


# ---------------- rates ----------------

def cint(c0, c1, f, q, dq):
    """Exact int_0^q F and int_0^q q' F for F piecewise linear on the q grid (c0, c1 at the nodes)."""
    nq = f.shape[1]
    l = np.clip((q / dq).astype(int), 0, nq - 2)
    ix = l + nq * np.arange(f.shape[0])[:, None]
    ql = l * dq
    tau = q - ql
    fl = f.ravel()[ix]
    sl = (f.ravel()[ix + 1] - fl) / dq
    g0 = c0.ravel()[ix] + fl * tau + 0.5 * sl * tau**2
    g1 = c1.ravel()[ix] + 0.5 * fl * (q**2 - ql**2) + sl * ((q**3 - ql**3) / 3 - 0.5 * ql * (q**2 - ql**2))
    return g0, g1


def cums(f, qg):
    """Node values of int_0^q F and int_0^q q F for F piecewise linear."""
    dq, ql, qr = qg[1], qg[:-1], qg[1:]
    fl, sl = f[:, :-1], (f[:, 1:] - f[:, :-1]) / dq
    z = np.zeros((f.shape[0], 1))
    c0 = np.concatenate((z, np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * dq, 1)), 1)
    c1 = np.concatenate((z, np.cumsum(0.5 * fl * (qr**2 - ql**2) + sl * ((qr**3 - ql**3) / 3 - 0.5 * ql * (qr**2 - ql**2)), 1)), 1)
    return c0, c1, f


def tz(mu, a, b, cf, cs, qd, dq):
    """int_0^qd dq F(q) T(cs q), T = trapezoid density of E_i - E_j (center mu, bin widths a, b),
    cf = cums(F)."""
    lo, hi, hh = 0.5 * (a + b), 0.5 * np.abs(a - b), 1 / np.maximum(a, b)
    p = np.stack((mu - lo, mu - hi, mu + hi, mu + lo), 1) / cs
    d1, d3 = p[:, 1] - p[:, 0], p[:, 3] - p[:, 2]
    al = np.stack((-hh * p[:, 0] / d1, hh, hh * p[:, 3] / d3), 1)
    be = np.stack((hh / d1, 0 * hh, -hh / d3), 1)
    q = np.clip(np.concatenate((p[:, :3], p[:, 1:]), 1), 0, qd)
    i0, i1 = cint(*cf, q, dq)
    return np.maximum(np.sum(al * (i0[:, 3:] - i0[:, :3]) + be * (i1[:, 3:] - i1[:, :3]), 1), 0.0)


def log_rates(st, d, m, photons=True):
    """log A[i, j], rate per state pair j -> i on 2 ns states (alpha bins, beta bins)."""
    p, t1, t2 = st["p"], st["t1"], st["t2"]
    cs, qd, gm, amp, wd = p["cs"], p["qd"], p["gam_ph"], p["amp_ph"], p["cs"] * p["qd"]
    qg, dq, sc, h, w = st["qg"], st["qg"][1], st["sc"], st["h"], st["w"]
    ns = sc.size
    xi, e, u, vv = mf(sc, p["gap"], p["ta"], p["tb"], p["v"], d, m)
    _, ee, _, _ = mf(st["se"], p["gap"], p["ta"], p["tb"], p["v"], d, m)
    wa = np.maximum(np.abs(np.diff(ee, axis=1)), 1e-9)  # energy widths of bins, (2, ns)
    a, b = np.stack((u, vv)), np.stack((-vv, u))  # orbital amplitudes of alpha, beta
    hw = (h / w)[:, None] * (h / w)[None]
    la = np.full((2 * ns, 2 * ns), -np.inf)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # intraband, sharp delta, bin averaged (trapezoid of E_i - E_j), downward value + detailed balance
        i, j = st["i1"], st["j1"]
        for lam in (0, 1):
            de = e[lam, i] - e[lam, j]
            x, y = wa[lam, i], wa[lam, j]
            ii = tz(-np.abs(de), x, y, st["fn1"], cs, qd, dq) + tz(np.abs(de), x, y, st["fe1"], cs, qd, dq)
            m2 = (2 * (p["tpa"] * a[lam, i] * a[lam, j] + p["tpb"] * b[lam, i] * b[lam, j])) ** 2
            val = np.log(amp * m2 * hw[i, j] * ii / cs)
            la[lam * ns + i, lam * ns + j] = val - np.maximum(de, 0) / t2
            la[lam * ns + j, lam * ns + i] = val - np.maximum(-de, 0) / t2
        # interband, all pairs: off shell Lorentzian tail by q moments (Om > 3 omega_D)
        om = e[0][:, None] - e[1][None]
        m1, m3, m5 = (st["mom"][..., n] for n in range(3))
        rf = 4 * gm / (np.pi * om**3) * (m1 + 2 * cs**2 * m3 / om**2 + 3 * cs**4 * m5 / om**4) * (1 + bose(om, t2))
        # interband near the shell: on shell (sharp, bin averaged) + direct off shell tail
        i, j = st["i2"], st["j2"]
        o2 = om[i, j]
        x, y = wa[0, i], wa[1, j]
        on = tz(o2, x, y, st["fe2"], cs, qd, dq) / cs
        sw = np.clip((o2 - 0.5 * (x + y) - wd) / (x + y), 0, 1)  # smooth switch to the tail
        lz = lambda z: gm / np.pi / (z**2 + gm**2)
        wq = np.full(qg.size, dq)
        wq[[0, -1]] *= 0.5
        tl = sw * np.sum(st["pv2"] * (lz(o2[:, None] - cs * qg) - lz(o2[:, None] + cs * qg)) * wq, 1) / cs
        rf[i, j] = on + tl * (1 + bose(o2, t2))
        m2 = (2 * (p["tpa"] * u[:, None] * vv[None] - p["tpb"] * vv[:, None] * u[None])) ** 2
        rf = np.log(amp * m2 * hw * rf)
        la[ns:, :ns] = rf.T  # beta_j <- alpha_i, emission
        la[:ns, ns:] = rf - om / t2  # alpha_i <- beta_j, absorption
        if photons:  # alpha_i <-> beta_i, h = 2 - s^2 / 2
            ok, g2, w0 = e[0] - e[1], p["gam_ga"], p["w0"]
            z4 = (ok**2 + g2**2) ** 2
            kk = ((1 + w0**2 * (ok**2 - g2**2) / z4) * (np.arctan((ok - w0) / g2) + np.arctan((ok + w0) / g2))
                  + g2 * ok * w0**2 / z4 * np.log(((ok - w0) ** 2 + g2**2) * ((ok + w0) ** 2 + g2**2) / w0**4))
            lj = np.log(p["amp_ga"] / (2 * np.pi * p["c"] ** 2) * (2 - 0.5 * sc**2)
                        * (2 * (p["ta"] - p["tb"]) * u * vv) ** 2 * kk / w)
            ln1 = -ok / t1 - np.log1p(-np.exp(-ok / t1))
            r = np.arange(ns)
            la[r, r + ns] = np.logaddexp(la[r, r + ns], lj + ln1)
            la[r + ns, r] = np.logaddexp(la[r + ns, r], lj + ln1 + ok / t1)
    return np.where(np.isnan(la), -np.inf, la)


def out_rates(st, d, m):
    """Bare out rates of alpha bins: (xi, Gamma_ph, Gamma_ph alpha -> beta, Gamma_gamma), for checks."""
    p, ns, ww = st["p"], st["sc"].size, np.tile(st["w"], 2)
    rp = np.exp(log_rates(st, d, m, photons=False)) * ww[:, None]  # state j -> all states of bin i
    ra = np.exp(log_rates(st, d, m)) * ww[:, None]
    r = np.arange(ns)
    xi = mf(st["sc"], p["gap"], p["ta"], p["tb"], p["v"], d, m)[0]
    return xi, rp[:, :ns].sum(0), rp[ns:, :ns].sum(0), ra[r + ns, r] - rp[r + ns, r]


# ---------------- solver ----------------

def lse(a, axis=None):
    """log sum exp, -inf safe."""
    mx = np.max(a, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return np.squeeze(mx, axis) + np.log(np.sum(np.exp(a - mx), axis=axis))


def ness(la, ww, x, tol=1e-11, nmax=300, cap=50.0, mc=50.0):
    """Inner Newton in x = logit(n), as ness_xi.ness."""
    li = la + np.log(ww)[None]
    lo = la.T + np.log(ww)[None]
    c, nn = 0.0, x.size
    ns, lw = nn // 2, np.log(ww[:nn // 2])
    for it in range(nmax):
        ln, lp = log_expit(x), log_expit(-x)
        lin, lout = lse(li + ln[None], 1), lse(lo + lp[None], 1)
        lpa, lhb = lse(lw + ln[:ns]), lse(lw + lp[ns:])
        g = np.append(x - lin + lout + c, lpa - lhb)
        deep = np.abs(x) > mc
        rel = np.append(~deep | (np.abs(x - g[:nn]) < mc), True)
        err = np.max(np.abs(g[rel]))
        if err < tol or not np.isfinite(err):
            break
        q = ln + lp
        jm = np.eye(nn + 1)
        jm[:nn, :nn] -= np.exp(li + q[None] - lin[:, None]) + np.exp(lo + q[None] - lout[:, None])
        jm[:nn, nn], jm[nn, nn] = 1.0, 0.0
        jm[nn, :nn] = np.exp(np.concatenate((lw + q[:ns] - lpa, lw + q[ns:] - lhb)))
        jm[np.flatnonzero(deep)] = np.eye(nn + 1)[deep.nonzero()[0]]
        if deep.all():
            jm[nn], g[nn] = np.eye(nn + 1)[nn], c
        try:
            dz = np.linalg.solve(jm, -g)
        except np.linalg.LinAlgError:
            dz = np.linalg.lstsq(jm, -g, rcond=None)[0]
        x = x + np.where(deep, dz[:nn], np.clip(dz[:nn], -cap, cap))
        c = c + dz[nn]
    return x, it, err


def gap_res(z, x, st):
    """Residual of the gap equations at (d, m), with the inner NESS n(d, m)."""
    p = st["p"]
    la = log_rates(st, z[0], z[1])
    ww = np.tile(st["w"], 2)
    x, nit, err = ness(la, ww, x)
    na, nb = np.split(expit(x), 2)
    _, _, uk, vk = mf(st["sk"], p["gap"], p["ta"], p["tb"], p["v"], z[0], z[1])
    _, _, uq, vq = mf(st["sq"], p["gap"], p["ta"], p["tb"], p["v"], z[0], z[1])
    dn = p["v"] * (np.sum(st["gk"] * uk * vk, 1) @ (nb - na) + np.sum(st["wq"] * uq * vq))
    mn = np.sum(st["gk"] * (uk**2 - vk**2), 1) @ (na - nb) - np.sum(st["wq"] * (uq**2 - vq**2))
    return np.array([dn - z[0], mn - z[1]]), x, np.sum(ww * expit(x)) - np.sum(st["w"]), nit, err


def bracket(z, r, x, st, dmin=1e-5, rel=1e-3):
    """Illinois search in d for r_d = 0 from d0 (first root reached by relaxation), m by its fixed
    point at each evaluation. Used when Newton stalls at kinks of the rates (2 d = omega_D)."""
    m, a, fa = z[1], z[0], r[0]

    def f(d, x):
        nonlocal m
        rr, x = gap_res(np.array([d, m]), x, st)[:2]
        m += rr[1]
        return rr[0], x

    b = a
    while True:  # march until r_d changes sign
        b *= 0.7 if fa < 0 else 1.3
        if b < dmin:  # normal state, d = 0
            for _ in range(10):
                f(0.0, x)
            return np.array([0.0, m]), x
        fb, x = f(b, x)
        if fb * fa < 0:
            break
        a, fa = b, fb
    for _ in range(40):
        c = b - fb * (b - a) / (fb - fa)
        fc, x = f(c, x)
        if fc * fb < 0:
            a, fa = b, fb
        else:
            fa *= 0.5
        b, fb = c, fc
        if abs(fb) < rel * abs(r[0]) or abs(b - a) < 1e-9:
            break
    return np.array([b, m]), x


def solve(t1=1.0, t2=0.05, p=P, geo=None, tol=1e-10, nmax=40, eps=1e-6, path=None, verbose=True, **kw):
    """NESS for (t1, t2) in units of Tc: outer Broyden on (d, m) from a FD Jacobian, with a
    bracketing fallback in d when the step fails."""
    geo = geometry(qd=p["qd"]) if geo is None else geo
    t0 = tc(p)
    t1, t2 = t1 * t0, t2 * t0
    z = np.array(equilibrium(p, t2))
    st = setup(geo, p, t1, t2, *z, **kw)
    _, e, _, _ = mf(st["sc"], p["gap"], p["ta"], p["tb"], p["v"], *z)
    x = -e.ravel() / t2
    r, x, fl, nit, err = gap_res(z, x, st)
    hist = [(z[0], z[1], fl)]
    if verbose:
        print(f"Tc = {t0:.5f} ({t0 * 0.1 / 8.617e-5:.1f} K), T1 = {t1:.5f}, T2 = {t2:.5f}, bins = {st['sc'].size}")
    jm, fresh = None, False
    for k in range(nmax):
        if verbose:
            print(f"step {k:2d}: d = {z[0]:.8f}, m = {z[1]:.8f}, n - 1 = {fl:.1e}, "
                  f"|r| = {np.max(np.abs(r)):.1e}, inner it = {nit}, inner err = {err:.1e}")
        if err > 1e-8:
            print("inner NESS not converged, stopping")
            break
        if np.max(np.abs(r)) < tol or z[0] == 0.0:
            break
        if jm is None:  # forward difference Jacobian
            jm, fresh = np.zeros((2, 2)), True
            for i in range(2):
                hh = eps * max(abs(z[i]), 1e-3)
                jm[:, i] = (gap_res(z + hh * np.eye(2)[i], x, st)[0] - r) / hh
        dz, t, ok = np.linalg.solve(jm, -r), 1.0, False
        for _ in range(4):  # backtracking
            zt = z + t * dz
            if zt[0] > 0:
                rt, xt, flt, nit, err = gap_res(zt, x, st)
                if np.max(np.abs(rt)) < np.max(np.abs(r)):
                    ok = True
                    break
            t *= 0.5
        if not ok:
            if fresh:  # Newton fails with an exact Jacobian: bracket in d
                z, x = bracket(z, r, x, st)
                r, x, fl, nit, err = gap_res(z, x, st)
            jm = None
            hist.append((z[0], z[1], fl))
            continue
        sz, yz = zt - z, rt - r
        jm, fresh = jm + np.outer(yz - jm @ sz, sz) / (sz @ sz), False  # Broyden
        z, r, x, fl = zt, rt, xt, flt
        hist.append((z[0], z[1], fl))
    hist = np.array(hist)
    if path is not None:
        plot(hist, path)
    return z, x, st, hist


def plot(hist, path):
    st = np.arange(len(hist))
    cl = plt.cm.Set1.colors
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(3.47412, 1.2 * 3.47412))
    ys = (hist[:, 0], np.maximum(np.abs(hist[:, 2]), 1e-17), hist[:, 1])
    lbs = (r"$\Delta$", r"$|n-1|$", r"$m$")
    for ax, y, lb, cc in zip(axs, ys, lbs, cl):
        ax.plot(st, y, "o-", ms=3, color=cc, label=lb)
        ax.set_ylabel(lb)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.tick_params(which="both", direction="in")
    axs[1].set_yscale("log")
    axs[-1].set_xlabel(r"step")
    axs[-1].set_xlim(st[0], max(st[-1], 1))
    plt.tight_layout()
    fig.savefig(path, dpi=300) if path else plt.show()


if __name__ == "__main__":
    solve(path="")
