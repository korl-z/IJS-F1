"""NESS of the xi (shell) reduced master equation, phonons (T2) + cavity photons (T1).

Standalone, numpy only. Rates are the full shell averaged ones of rates_xi.xi_model,
on all s bins of the band (no window). Inner Newton in x = logit(n) for fixed (d, m),
outer Newton on the gap equations (d, m). Units t_a = a0 = hbar = kB = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, log_expit, logsumexp, zeta

P = dict(gap=4.0, ta=1.0, tb=-1.0, v=4.0, amp_ph=1 / (2 * 598.0 * 200), tpa=-1.0e-3, tpb=1.0e-3,
         cs=0.026, gam_ph=0.0005, qd=np.sqrt(4 * np.pi), w0=0.1, gam_ga=0.01, c=3945.0,
         amp_ga=4 / 137.036 * 0.1)


def mf(s, gap, ta, tb, v, d, m, n=1.0):
    """Hartree shifted bands, xi, E (alpha, beta) and u, v at s."""
    na, nb = 0.5 * (n + m), 0.5 * (n - m)
    eah = 0.5 * gap - 2 * ta * s + v * nb - 0.5 * v
    ebh = -0.5 * gap - 2 * tb * s + v * na - 0.5 * v
    xi, av = 0.5 * (eah - ebh), 0.5 * (eah + ebh)
    ek = np.hypot(xi, d)
    u, vv = np.sqrt(0.5 * (1 + xi / ek)), np.sqrt(0.5 * (1 - xi / ek))
    return xi, np.stack((av + ek, av - ek)), u, vv


def occ(e, w, t):
    """Fermi occupations of both bands at filling 1."""
    lo, hi = e.min() - 1, e.max() + 1
    for _ in range(100):
        mu = 0.5 * (lo + hi)
        lo, hi = (mu, hi) if np.sum(w * expit(-(e - mu) / t).sum(0)) < 1 else (lo, mu)
    return expit(-(e - mu) / t)


def tc(s, w, gap, ta, tb, v, lo=1e-3, hi=3.0, eps=1e-7):
    """Mean field Tc, chi(Tc) = 1 of the linearized gap equation."""
    m = 0.0
    for _ in range(30):
        tm = np.sqrt(lo * hi)
        for _ in range(500):  # normal state imbalance
            _, e, u, vv = mf(s, gap, ta, tb, v, eps, m)
            nq = occ(e, w, tm)
            mt = np.sum(w * (u**2 - vv**2) * (nq[0] - nq[1]))
            if abs(mt - m) < 1e-12:
                break
            m = 0.5 * (m + mt)
        x = v * np.sum(w * u * vv * (nq[1] - nq[0])) / eps
        lo, hi = (tm, hi) if x > 1 else (lo, tm)
    return np.sqrt(lo * hi)


def equilibrium(s, w, gap, ta, tb, v, t, d=1.0, m=-0.9):
    """Mean field (d, m) at temperature t, start of the outer Newton."""
    for _ in range(5000):
        _, e, u, vv = mf(s, gap, ta, tb, v, d, m)
        nq = occ(e, w, t)
        dt = v * np.sum(w * u * vv * (nq[1] - nq[0]))
        mt = np.sum(w * (u**2 - vv**2) * (nq[0] - nq[1]))
        if abs(dt - d) + abs(mt - m) < 1e-12:
            break
        d, m = d + 0.5 * (dt - d), m + 0.5 * (mt - m)
    return d, m


def table(ns=400, nq=48, qd=np.sqrt(4 * np.pi), nf=128, nphi=64):
    """s bins and pair table S[i, j, l] = 1/N sum_{k in i} <V 1[k + q_l in j]>_phi (as xi_model)."""
    k = np.linspace(-np.pi, np.pi, nf, endpoint=False)
    kx, ky = (a.ravel() for a in np.meshgrid(k, k, indexing="ij"))
    s = np.cos(kx) + np.cos(ky)
    bn = lambda z: np.clip(((z + 2) / 4 * ns).astype(int), 0, ns - 1)
    bi = bn(s)
    cnt = np.bincount(bi, minlength=ns)
    if np.any(cnt == 0):
        raise ValueError("empty s bins, increase nf or decrease ns")
    w = cnt / s.size
    sc = np.bincount(bi, weights=s, minlength=ns) / cnt
    hb = np.bincount(bi, weights=np.sin(kx)**2 + np.sin(ky)**2, minlength=ns) / cnt
    dq = qd / (nq - 1)
    S = np.zeros((nq, ns * ns))
    sx, sy = np.sin(kx), np.sin(ky)
    for l in range(1, nq):
        for p in 2 * np.pi * (np.arange(nphi) + 0.5) / nphi:
            qx, qy = l * dq * np.cos(p), l * dq * np.sin(p)
            if max(abs(qx), abs(qy)) > np.pi:  # min image q, as the grid
                continue
            x, y = kx + qx, ky + qy
            vt = (np.sin(x) - sx)**2 + (np.sin(y) - sy)**2
            S[l] += np.bincount(bi * ns + bn(np.cos(x) + np.cos(y)), weights=vt, minlength=ns * ns)
    S = np.moveaxis(S.reshape(nq, ns, ns), 0, -1) / (nphi * s.size)
    return sc, w, hb, dq, 0.5 * (S + S.transpose(1, 0, 2))


def lbose(x, t):
    """log N(x) and log(1 + N(x)) for x > 0."""
    z = x / t
    ln = -z - np.log1p(-np.exp(-z))
    return ln, ln + z


def log_rates(tab, d, m, t1, t2, p):
    """log A[i, j], rate per state pair j -> i on the 2 ns states (alpha bins, beta bins).

    The master equation is dn_i/dt = sum_j w_j [A_ij n_j (1 - n_i) - A_ji n_i (1 - n_j)].
    """
    sc, w, hb, dq, S = tab
    ns, nq = sc.size, S.shape[-1]
    xi, e, u, vv = mf(sc, p["gap"], p["ta"], p["tb"], p["v"], d, m)
    e2, a, b = e.ravel(), np.concatenate((u, vv)), np.concatenate((-vv, u))
    ib = np.tile(np.arange(ns), 2)
    ww = np.tile(w, 2)
    de = e2[:, None] - e2[None]  # E_dest - E_src
    same = (np.arange(2 * ns)[:, None] < ns) == (np.arange(2 * ns)[None] < ns)
    with np.errstate(divide="ignore", invalid="ignore"):
        lno, lne = lbose(np.abs(de), t2)
        lb = np.where(de > 0, lno, np.where(de < 0, lne, 0.0))  # absorption N, emission 1 + N, de = 0 has g = 0
        # intraband, sharp delta: S(q*) / cs^2
        z = np.abs(de) / (p["cs"] * dq)
        l0 = np.minimum(z.astype(int), nq - 2)
        fz = z - l0
        bi, bj = np.meshgrid(ib, ib, indexing="ij")
        g = ((1 - fz) * S[bi, bj, l0] + fz * S[bi, bj, l0 + 1]) / p["cs"]**2
        g = np.where(same & (z <= nq - 1) & (de != 0), g, 0.0)
        # interband, Lorentzian delta_gamma(Om, cs q) on the q grid
        om = (e[0][:, None] - e[1][None])[..., None]
        wq = p["cs"] * dq * np.arange(nq)
        lz = lambda x: p["gam_ph"] / np.pi / (x**2 + p["gam_ph"]**2)
        gi = np.sum(S * (lz(om - wq) - lz(om + wq)), -1) * dq / p["cs"]
        g[:ns, ns:], g[ns:, :ns] = gi, gi.T
        m2 = (2 * (p["tpa"] * a[:, None] * a[None] + p["tpb"] * b[:, None] * b[None]))**2
        la = np.log(p["amp_ph"] * m2 * g) + lb - np.log(ww)[:, None] - np.log(ww)[None]
        # photons, k diagonal alpha_i <-> beta_i
        ok, gm, w0 = e[0] - e[1], p["gam_ga"], p["w0"]
        z4 = (ok**2 + gm**2)**2
        kk = ((1 + w0**2 * (ok**2 - gm**2) / z4) * (np.arctan((ok - w0) / gm) + np.arctan((ok + w0) / gm))
              + gm * ok * w0**2 / z4 * np.log(((ok - w0)**2 + gm**2) * ((ok + w0)**2 + gm**2) / w0**4))
        lj = np.log(p["amp_ga"] / (2 * np.pi * p["c"]**2) * hb * (2 * (p["ta"] - p["tb"]) * u * vv)**2 * kk / w)
        gno, gne = lbose(ok, t1)
        r = np.arange(ns)
        la[r, r + ns] = np.logaddexp(la[r, r + ns], lj + gno)  # beta -> alpha, absorption
        la[r + ns, r] = np.logaddexp(la[r + ns, r], lj + gne)  # alpha -> beta, emission
    return la, xi, u, vv


def ness(la, ww, x, tol=1e-11, nmax=500, cap=50.0, mc=40.0):
    """Inner Newton in x = logit(n): G_i = x_i - log in_i + log out_i + c = 0, plus filling 1.

    Filling 1 is written as log N_alpha = log H_beta (alpha particles = beta holes), which stays
    well scaled when all occupations are exponentially close to 0 or 1. The border c removes
    the zero mode of number conservation, c = 0 at the solution.
    Deep states (|x| > mc, n < e^-40) get a fixed point update x_i = log in_i - log out_i - c
    instead of a Newton row: their levels are set by exponentially weak links, which makes
    their Newton block numerically singular, and they do not affect n of the other states.
    Convergence is tested on all states whose x or target lies inside |x| < mc.
    """
    li = la + np.log(ww)[None]  # log w_j A_ij
    lo = la.T + np.log(ww)[None]  # log w_j A_ji
    c, nn = 0.0, x.size
    ns, lw = nn // 2, np.log(ww[:nn // 2])
    for it in range(nmax):
        ln, lp = log_expit(x), log_expit(-x)
        lin, lout = logsumexp(li + ln[None], 1), logsumexp(lo + lp[None], 1)
        lpa, lhb = logsumexp(lw + ln[:ns]), logsumexp(lw + lp[ns:])
        g = np.append(x - lin + lout + c, lpa - lhb)
        deep = np.abs(x) > mc
        rel = np.append(~deep | (np.abs(x - g[:nn]) < mc), True)
        err = np.max(np.abs(g[rel]))
        if err < tol:
            break
        q = ln + lp  # log n (1 - n)
        jm = np.eye(nn + 1)
        jm[:nn, :nn] -= np.exp(li + q[None] - lin[:, None]) + np.exp(lo + q[None] - lout[:, None])
        jm[:nn, nn], jm[nn, nn] = 1.0, 0.0
        jm[nn, :nn] = np.exp(np.concatenate((lw + q[:ns] - lpa, lw + q[ns:] - lhb)))
        jm[np.flatnonzero(deep)] = np.eye(nn + 1)[deep.nonzero()[0]]  # fixed point rows
        if deep.all():  # no Newton rows, c cannot be fixed by the filling: hold c = 0
            jm[nn], g[nn] = np.eye(nn + 1)[nn], c
        dz = np.linalg.solve(jm, -g)
        x = x + np.where(deep, dz[:nn], np.clip(dz[:nn], -cap, cap))  # clip only Newton states
        c = c + dz[nn]
    return x, it, err


def gap_res(z, x, tab, t1, t2, p):
    """Residual of the gap equations at (d, m), with the inner NESS n(d, m)."""
    la, xi, u, vv = log_rates(tab, z[0], z[1], t1, t2, p)
    ww = np.tile(tab[1], 2)
    x, nit, err = ness(la, ww, x)
    na, nb = np.split(expit(x), 2)
    w = tab[1]
    dn = p["v"] * np.sum(w * u * vv * (nb - na))
    mn = np.sum(w * (u**2 - vv**2) * (na - nb))
    return np.array([dn - z[0], mn - z[1]]), x, np.sum(ww * expit(x)) - 1, nit, err


def solve(t1=1.0, t2=0.05, p=P, tab=None, tol=1e-4, nmax=40, eps=1e-6, path="ness_xi.png"):
    """NESS for (t1, t2) in units of Tc, outer Newton on (d, m) with FD Jacobian."""
    tab = table(qd=p["qd"]) if tab is None else tab
    sc, w = tab[0], tab[1]
    t0 = tc(sc, w, p["gap"], p["ta"], p["tb"], p["v"])
    t1, t2 = t1 * t0, t2 * t0
    z = np.array(equilibrium(sc, w, p["gap"], p["ta"], p["tb"], p["v"], t2))
    _, e, _, _ = mf(sc, p["gap"], p["ta"], p["tb"], p["v"], z[0], z[1])
    x = -e.ravel() / t2  # equilibrium start, mu = 0
    r, x, fl, nit, err = gap_res(z, x, tab, t1, t2, p)
    hist = [(z[0], z[1], fl)]
    print(f"Tc = {t0* 0.1 /(8.617 * 1e-5):.5f}, T1 = {t1* 0.1 /(8.617 * 1e-5):.5f}, T2 = {t2* 0.1 /(8.617 * 1e-5):.5f}")
    for k in range(nmax):
        print(f"step {k:2d}: d = {z[0]:.8f}, m = {z[1]:.8f}, n - 1 = {fl:.1e}, "
              f"|r| = {np.max(np.abs(r)):.1e}, inner it = {nit}, inner err = {err:.1e}")
        if err > 1e-8:
            print("inner NESS not converged, stopping")
            break
        if np.max(np.abs(r)) < tol:
            break
        jm = np.zeros((2, 2))
        for i in range(2):  # forward differences
            h = eps * max(abs(z[i]), 1e-3)
            jm[:, i] = (gap_res(z + h * np.eye(2)[i], x, tab, t1, t2, p)[0] - r) / h
        dz = np.linalg.solve(jm, -r)
        t = 1.0
        while True:  # backtracking, keep d > 0
            zt = z + t * dz
            if zt[0] > 0:
                rt, xt, flt, nit, err = gap_res(zt, x, tab, t1, t2, p)
                if np.max(np.abs(rt)) < np.max(np.abs(r)) or t < 1e-3:
                    break
            t *= 0.5
        z, r, x, fl = zt, rt, xt, flt
        hist.append((z[0], z[1], fl))
    hist = np.array(hist)
    plot(hist, path)
    return z, x, hist


def plot(hist, path):
    st = np.arange(len(hist))
    cl = plt.cm.Set1.colors
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(3.47412, 1.2 * 3.47412))
    ys = (hist[:, 0], np.maximum(np.abs(hist[:, 2]), 1e-17), hist[:, 1])  # floor for log axis
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
    # fig.savefig(path, dpi=300)
    plt.show()


if __name__ == "__main__":
    solve()
