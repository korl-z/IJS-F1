"""s binned NESS of the two band EI with analytic cavity photons and chord kernel phonons.

Self contained (numpy only). States: 2 bands x ns bins in s = cos kx + cos ky,
ordered (alpha bins, beta bins). n = n(s) is the only model assumption.
Units hbar = kB = a0 = 1, energies in t_a. Conventions follow ei_jax_2 / ei_phonon.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from scipy.special import expit
from scipy.sparse.csgraph import connected_components


# ---------------- parameters ----------------

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


class Phonon(NamedTuple):
    t: float  # temperature
    amp: float  # 1 / (2 M)
    tpa: float  # Peierls slope t'_a
    tpb: float  # Peierls slope t'_b
    cs: float
    qd: float = float(np.sqrt(4.0 * np.pi))


class Cavity(NamedTuple):
    t: float  # temperature
    w0: float
    gam: float
    c: float = 3945.0
    amp: float | None = None  # None: 4 alpha w0
    alpha: float = 1.0 / 137.036


class Grid(NamedTuple):
    sc: Any  # bin mean of s, (ns,)
    w: Any  # state fraction per bin, (ns,)
    h: Any  # bin mean of sin^2 kx + sin^2 ky, (ns,)
    dq: float  # q_l = l dq
    S: Any  # (ns, ns, nq), symmetric pair table


class State(NamedTuple):
    e: Any  # quasiparticle energies (2, ns)
    u: Any
    v: Any


class Sol(NamedTuple):
    n: Any
    mu: Any  # log((1 - n) / n), accurate where n is close to 0 or 1
    d: float
    m: float
    st: State
    err: float
    ok: bool
    hist: dict


# ---------------- geometry, computed once ----------------

def grid(ns, nq, qmax=float(np.sqrt(4.0 * np.pi)), nf=512, nphi=128):
    """Bins in s and S(s, s', q) = 1/N sum_{k in s} 1/M sum_m V 1[k + q_m in s']."""
    k = np.linspace(-np.pi, np.pi, nf, endpoint=False)
    kx, ky = (a.ravel() for a in np.meshgrid(k, k, indexing="ij"))
    sb = np.linspace(-2.0, 2.0, ns + 1)
    bi = np.clip(np.digitize(np.cos(kx) + np.cos(ky), sb) - 1, 0, ns - 1)
    cnt = np.bincount(bi, minlength=ns)
    if np.any(cnt == 0):
        raise ValueError("empty s bins, increase nf or decrease ns")
    sc = np.bincount(bi, weights=np.cos(kx) + np.cos(ky), minlength=ns) / cnt
    h = np.bincount(bi, weights=np.sin(kx) ** 2 + np.sin(ky) ** 2, minlength=ns) / cnt
    dq = qmax / (nq - 1)
    S = np.zeros((nq, ns * ns))
    for l in range(1, nq):
        for p in 2.0 * np.pi * (np.arange(nphi) + 0.5) / nphi:
            x, y = kx + l * dq * np.cos(p), ky + l * dq * np.sin(p)
            vv = (np.sin(x) - np.sin(kx)) ** 2 + (np.sin(y) - np.sin(ky)) ** 2
            bj = np.clip(np.digitize(np.cos(x) + np.cos(y), sb) - 1, 0, ns - 1)
            S[l] += np.bincount(bi * ns + bj, weights=vv, minlength=ns * ns)
    S = S.reshape(nq, ns, ns) / (nphi * kx.size)
    S = 0.5 * (S + S.transpose(0, 2, 1))
    return Grid(sc, cnt / kx.size, h, dq, np.moveaxis(S, 0, -1))


# ---------------- mean field ----------------

def mf(md, g, d, m):
    """Bogoliubov state at the bin means, same energy frame as ej._mf."""
    ea = 0.5 * md.gap - 2.0 * md.ta * g.sc
    eb = -0.5 * md.gap - 2.0 * md.tb * g.sc
    na, nb = 0.5 * (md.n + m), 0.5 * (md.n - m)
    eah = ea + md.h * nb - 0.5 * md.h
    ebh = eb + md.h * na - 0.5 * md.h
    xi, av = 0.5 * (eah - ebh), 0.5 * (eah + ebh)
    ek = np.hypot(xi, d)
    u = np.sqrt(0.5 * (1.0 + xi / np.maximum(ek, 1e-300)))
    v = np.copysign(np.sqrt(0.5 * (1.0 - xi / np.maximum(ek, 1e-300))), d)
    return State(np.stack((av + ek, av - ek)), u, v)


def targets(md, g, st, n):
    """Gap, imbalance and filling from the occupations."""
    u2, v2 = st.u ** 2, st.v ** 2
    na = np.sum(g.w * (u2 * n[0] + v2 * n[1]))
    nb = np.sum(g.w * (v2 * n[0] + u2 * n[1]))
    d = md.v * np.sum(g.w * st.u * st.v * (n[1] - n[0]))
    return d, na - nb, na + nb


# ---------------- rates ----------------

def bose(x, t):
    if t <= 0.0:
        return np.zeros_like(x)
    return 1.0 / np.expm1(np.maximum(x, 1e-300) / t)


def kern(om, w0, gam):
    """Photon bracket 4 gam Om int_w0^inf (w + w0^2 / w) / (d1 d2) dw."""
    z4 = (om ** 2 + gam ** 2) ** 2
    a = np.arctan((om - w0) / gam) + np.arctan((om + w0) / gam)
    l = np.log(((om - w0) ** 2 + gam ** 2) * ((om + w0) ** 2 + gam ** 2) / w0 ** 4)
    return (1.0 + w0 ** 2 * (om ** 2 - gam ** 2) / z4) * a + gam * om * w0 ** 2 / z4 * l


def rates(md, g, st, phs=(), cavs=()):
    """R[dest, src] on the 2 ns states; the solvers weight sources by w."""
    ns, nq = g.w.size, g.S.shape[-1]
    e = st.e.reshape(-1)
    a = np.concatenate((st.u, st.v))
    b = np.concatenate((-st.v, st.u))
    ib = np.tile(np.arange(ns), 2)
    bi, bj = ib[:, None], ib[None, :]
    ww = np.tile(g.w, 2)
    de = e[:, None] - e[None, :]
    r = np.zeros_like(de)
    for p in phs:  # chord kernel, q* = |dE| / cs
        z = np.minimum(np.abs(de) / (p.cs * g.dq), nq - 1.0)
        l0 = np.minimum(z.astype(int), nq - 2)
        fz = z - l0
        S = (1.0 - fz) * g.S[bi, bj, l0] + fz * g.S[bi, bj, l0 + 1]
        ok = (np.abs(de) <= p.cs * min(p.qd, (nq - 1) * g.dq)) & (de != 0.0)
        c1 = 2.0 * (p.tpa * a[:, None] * a[None, :] + p.tpb * b[:, None] * b[None, :])
        nb = bose(np.abs(de), p.t)
        bf = np.where(de < 0.0, 1.0 + nb, nb)  # emission if the source is higher
        r += np.where(ok, p.amp * c1 ** 2 * S * bf, 0.0) / (p.cs ** 2 * np.outer(ww, ww))
    np.fill_diagonal(r, 0.0)
    for c in cavs:  # vertical beta_s <-> alpha_s
        amp = 4.0 * c.alpha * c.w0 if c.amp is None else c.amp
        om = st.e[0] - st.e[1]
        j = amp / (2.0 * np.pi * c.c ** 2) * g.h * (2.0 * (md.ta - md.tb) * st.u * st.v) ** 2 \
            * kern(om, c.w0, c.gam)
        nb = bose(om, c.t)
        i = np.arange(ns)
        r[i, i + ns] += j * nb / g.w  # beta -> alpha, absorption
        r[i + ns, i] += j * (1.0 + nb) / g.w  # alpha -> beta, emission
    return r


def current(r, w, mu):
    """dn = (1 - n) a - n b, a = R (w n), b = R^T (w (1 - n)); n = 1 / (1 + e^mu).

    Returns dn, gain = (1 - n) a and loss = n b, all shaped like mu.
    """
    y, h, ww = expit(-np.ravel(mu)), expit(np.ravel(mu)), np.tile(w, 2)
    a, b = r @ (ww * y), r.T @ (ww * h)
    sh = np.shape(mu)
    return (h * a - y * b).reshape(sh), (h * a).reshape(sh), (y * b).reshape(sh)


# ---------------- solvers ----------------

def lsolve(A, rhs, it=20):
    """Solve A x = rhs after Ruiz equilibration (alternating row / column max norms).

    The mu columns scale as n (1 - n), which spans e^{+-E/T}; equilibration removes
    this so that only the physical conditioning (slow pair mode) remains.
    """
    dr, dc = np.ones(A.shape[0]), np.ones(A.shape[1])
    for _ in range(it):
        B = np.abs(dr[:, None] * A * dc[None, :])
        rm, cm = B.max(1), B.max(0)
        dr /= np.sqrt(np.where(rm > 0.0, rm, 1.0))
        dc /= np.sqrt(np.where(cm > 0.0, cm, 1.0))
    return dc * np.linalg.solve(dr[:, None] * A * dc[None, :], dr * rhs)


def frozen(r):
    """States outside the largest connected component of the rate graph.

    Each component conserves its own particle number, so only one can be solved
    for at fixed filling. Small components (bins at band extrema that no chord
    reaches within omega_D) are kept at their initial occupations.
    """
    nc, lab = connected_components((r > 0.0) | (r.T > 0.0), directed=False)
    return lab != np.argmax(np.bincount(lab))


def occ(r, w, pn, mu, tol=1e-12, nmax=60, cap=10.0):
    """Stationary state of the frozen rates R at filling pn, Newton in mu, n = 1 / (1 + e^mu).

    Unknowns: mu (2 ns), kap_alpha, kap_beta. Rows:
      F_i = -dn_i + kap_band(i) n_i (1 - n_i)       scaled by 1 / (gain_i + loss_i)
      filling: sum_i w_i n_i - pn                    absolute
      pair:    P = sum_{i in alpha} w_i dn_i^inter    scaled by 1 / sum (gain + loss)^inter
    Intraband rates conserve N_alpha and N_beta exactly, so each band has one redundant
    row; the kap absorb it and vanish at the solution, and the pair number is fixed by
    P, which uses only interband rates (no cancellation of large intraband terms).
    err = max over all scaled rows.
    """
    nn, ww = r.shape[0], np.tile(w, 2)
    bd = np.arange(nn) >= nn // 2  # beta states
    ri = np.where(bd[:, None] != bd[None, :], r, 0.0)  # interband part
    dead = frozen(r)  # states outside the main component stay put
    pa = ~bd & ~dead
    if not ri[np.ix_(~dead, ~dead)].any():
        raise ValueError("no interband rates: the pair number, hence the NESS, is not unique")
    mc = 600.0  # |mu| clip, keeps n (1 - n) >= e^-600 so no row or column underflows
    x = np.concatenate((np.clip(np.ravel(mu), -mc, mc), [0.0, 0.0]))

    def res(x):
        y, h = expit(-x[:nn]), expit(x[:nn])
        kap = np.where(bd, x[nn + 1], x[nn])
        a, b = r @ (ww * y), r.T @ (ww * h)
        ai, bi = ri @ (ww * y), ri.T @ (ww * h)
        F = np.where(dead, 0.0, y * b - h * a + kap * y * h)
        pr = np.sum((ww * (h * ai - y * bi))[pa])
        ps = np.sum((ww * (h * ai + y * bi))[pa])
        f = np.concatenate((F, [np.sum(ww * y) - pn, pr]))
        gl = y * b + h * a
        sc = np.concatenate((np.where(gl > 0.0, 1.0 / np.maximum(gl, 1e-300), 1.0),
                             [1.0, 1.0 / ps if ps > 0.0 else 1.0]))
        return f, sc, y, h, a, b, ai, bi, kap

    for _ in range(nmax):
        f, sc, y, h, a, b, ai, bi, kap = res(x)
        # pinned: at the clip and pushed further out (n -> 0 or 1 beyond e^-600), converged by the bound
        pin = (np.abs(x[:nn]) >= mc) & (np.sign(x[:nn]) * f[:nn] > 0.0)
        fix = np.append(dead | pin, [False, False])
        f[fix] = 0.0
        err = np.max(np.abs(sc * f))
        if err < tol:
            break
        s = ww * y * h
        jm = np.zeros((nn + 2, nn + 2))
        jm[:nn, :nn] = s[None, :] * (y[:, None] * r.T + h[:, None] * r)
        jm[np.arange(nn), np.arange(nn)] += y * h * (kap * (y - h) - a - b)
        jm[:nn, nn] = np.where(bd, 0.0, y * h)
        jm[:nn, nn + 1] = np.where(bd, y * h, 0.0)
        jm[nn, :nn] = -s
        jm[nn + 1, :nn] = (-s * (ri @ (ww * y * pa) + ri.T @ (ww * h * pa))
                           + pa * ww * y * h * (ai + bi))
        jm[np.flatnonzero(fix)] = 0.0
        jm[np.flatnonzero(fix), np.flatnonzero(fix)] = 1.0
        dx = lsolve(sc[:, None] * jm, -sc * f)
        dx[:nn] = np.clip(dx[:nn], -cap, cap)  # per component, so large steps do not stall the rest
        t, f0 = 1.0, np.sum((sc * f) ** 2)
        while t > 1e-8:  # backtracking on the scaled residual, same rows
            f1, sc1 = res(np.concatenate((np.clip((x + t * dx)[:nn], -mc, mc), (x + t * dx)[nn:])))[:2]
            f1[fix] = 0.0
            if np.isfinite(f1).all() and np.sum((sc1 * f1) ** 2) < (1 - 1e-4 * t) * f0:
                break
            t *= 0.5
        x = x + t * dx
        x[:nn] = np.clip(x[:nn], -mc, mc)
    return x[:nn].reshape(np.shape(mu)), err


def fd(st, w, pn, t):
    """mu_i = (E_i - mu_c) / t of the Fermi state at temperature t and filling pn."""
    lo, hi = st.e.min() - 50.0 * t - 1.0, st.e.max() + 50.0 * t + 1.0
    for _ in range(200):
        mc = 0.5 * (lo + hi)
        lo, hi = (mc, hi) if np.sum(w * expit(-(st.e - mc) / t).sum(0)) < pn else (lo, mc)
    return (st.e - mc) / t


def solve(md, g, phs, cavs, d=1.0, m=0.0, n=None, tol=1e-10, nmax=50, ntol=1e-12,
          dfl=1e-8, eps=1e-6):
    """Steady state: Newton in z = (log d, m), each evaluation an exact inner solve.

    F(z) = (d* / d - 1, m* - m), with (d*, m*) the targets of the stationary n of
    the rates at (d, m). The log and the ratio exclude the trivial root d = 0.
    Converged when max|F| < tol and the inner error e_n < 10 ntol.
    If d drops below dfl the state is normal: d = dfl is frozen, only m is solved.
    """
    if n is None:  # start: Fermi state at the coldest bath
        mu = fd(mf(md, g, d, m), g.w, md.n, max(min(x.t for x in (*phs, *cavs)), 1e-3))
    else:
        mu = np.log1p(-np.clip(n, 1e-300, 1 - 1e-16)) - np.log(np.clip(n, 1e-300, 1))

    def ev(z, mu):
        dd, mm = np.exp(z[0]), z[1]
        st = mf(md, g, dd, mm)
        mu, en = occ(rates(md, g, st, phs, cavs), g.w, md.n, mu, tol=ntol)
        dt, mt, _ = targets(md, g, st, expit(-mu))
        return np.array([dt / dd - 1.0, mt - mm]), mu, en

    z = np.array([np.log(max(abs(d), dfl)), m])
    f, mu, en = ev(z, mu)
    hs = {q: [] for q in ("d", "m", "en", "fd", "fm")}
    ok = False
    for it in range(nmax):
        for q, v in zip(hs, (np.exp(z[0]), z[1], en, f[0], f[1])):
            hs[q].append(v)
        nrm = z[0] < np.log(dfl)  # normal state: freeze d
        if nrm:
            z[0], f[0] = np.log(dfl), 0.0
        if np.max(np.abs(f)) < tol and en < 10.0 * ntol:
            ok = True
            break
        act = [1] if nrm else [0, 1]
        jm = np.zeros((2, 2))
        for j in act:  # forward difference columns
            jm[:, j] = (ev(z + eps * np.eye(2)[j], mu)[0] - f) / eps
        dz = np.zeros(2)
        dz[act] = np.linalg.solve(jm[np.ix_(act, act)], -f[act])
        dz *= min(1.0, 1.0 / max(np.max(np.abs(dz)), 1e-300))  # cap the step
        t = 1.0
        while t > 1e-4:  # backtracking on max|F|
            f1, mu1, en1 = ev(z + t * dz, mu)
            if np.max(np.abs(f1[act])) < (1.0 - 0.5 * t) * np.max(np.abs(f[act])):
                break
            t *= 0.5
        z, f, mu, en = z + t * dz, f1, mu1, en1
    n = expit(-mu)
    st = mf(md, g, np.exp(z[0]), z[1])
    return Sol(n, mu, float(np.exp(z[0])), float(z[1]), st, float(max(np.max(np.abs(f)), en)), ok,
               {q: np.asarray(v) for q, v in hs.items()})
