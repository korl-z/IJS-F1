"""NESS solvers of the limit model (limit_model.py).

solve_fixed  : Picard iteration in mu, as ej.solve_fixed (mu based version).
solve_newton : Newton on (mu, d, m) with the filling row, least squares step.
solve_qf     : quasi-Fermi limit (phonons much faster than photons): each band is
               FD at T2 with its own mu_gamma; one interband balance fixes mu_a - mu_b.
t_eff        : equilibrium temperature with the same |d| as a NESS.
n = 1 / (1 + exp(mu)); n and 1 - n are both built from mu to keep the tails.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq, root
from scipy.special import expit, log_expit, logsumexp

import limit_model as lm


@dataclass
class Sol:
    mu: np.ndarray       # (2N,), alpha then beta
    n: np.ndarray
    d: float
    m: float
    err: float
    it: int
    hist: dict = field(default_factory=dict)


def occ(mu):
    """n and 1 - n from mu."""
    return expit(-mu), expit(mu)


def balance(g, w, n, nb):
    """Gain a and loss b of every state, dn/dt = (1 - n) a - n b."""
    return g @ (w * n), g.T @ (w * nb)


def targets(gr, p, b, n, m):
    """Right sides of the gap (eq. 37) and imbalance (eq. 46) equations for any n
    (xi / E as cell average, see lm.xe_cell)."""
    ns = len(gr.s)
    na, nbe = n[:ns], n[ns:]
    return (p.v * np.sum(gr.w * b.d / (2 * b.e) * (nbe - na)),
            np.sum(gr.w * lm.xe_cell(p, gr, b.d, m) * (na - nbe)))


def _target_mu(a, b, mu):
    """mu_t = ln b - ln a (n = a / (a + b)); states without rates keep mu."""
    mt = np.clip(np.log(np.maximum(b, 1e-300)) - np.log(np.maximum(a, 1e-300)), -700, 700)
    return np.where((a > 0) | (b > 0), mt, mu)


def _fix_fill(mu, w, n0):
    """Shift all mu so that sum w n = n0 (two Newton steps, each capped at 1)."""
    for _ in range(2):
        n, nb = occ(mu)
        mu = mu + np.clip((np.sum(w * n) - n0) / max(np.sum(w * n * nb), 1e-300), -1, 1)
    return mu


def jlog(g, w, mu):
    """ln (sum of photon beta -> alpha terms / sum of alpha -> beta terms), tail safe."""
    ns = len(w) // 2
    i = np.arange(ns)
    ln, lnb = log_expit(-mu), log_expit(mu)
    with np.errstate(divide="ignore"):
        lw = 2 * np.log(w[:ns])
        lu, ld = np.log(g[i, ns + i]), np.log(g[ns + i, i])
    r = logsumexp(lw + lu + ln[ns:] + lnb[:ns]) - logsumexp(lw + ld + ln[:ns] + lnb[ns:])
    return float(np.nan_to_num(r, nan=0.0, posinf=700.0, neginf=-700.0))


def _band_shift(mu, g, w, n0, h=1e-4):
    """One Newton step that moves mu_alpha against mu_beta (then refills) toward zero
    interband photon current. Phonons conserve both band numbers, so this is the slow
    GGE direction (eqs. 77 - 79); not part of ej.solve_fixed."""
    ns = len(w) // 2
    sgn = np.r_[np.ones(ns), -np.ones(ns)]
    j0 = jlog(g, w, mu)
    j1 = jlog(g, w, _fix_fill(mu + h * sgn, w, n0))
    c = -j0 * h / (j1 - j0) if j1 != j0 else 0.0
    return _fix_fill(mu + np.clip(c, -1, 1) * sgn, w, n0)


def solve_fixed(gr, p, pc, qc, t1, t2, mu, d, m, mix=(0.5, 0.5, 0.1), tol=1e-8,
                nmax=20000, chk=100, shift=True, prog=False):
    """Picard iteration of ej.solve_fixed: mu <- mu + xn (mu_t - mu), filling shift,
    d <- d + xd (d_t - d), m <- m + xm (m_t - m), checked every chk iterations.

    Changes against ej.solve_fixed (flagged): the error also contains |jlog|, the
    interband photon imbalance, since max |a / (a + b) - n| is blind to it when
    phonons are much faster; shift=True adds _band_shift after every update."""
    xn, xd, xm = mix
    mu = np.array(mu, float)
    hist = {"it": [], "err": [], "d": [], "m": [], "jlog": []}
    err = np.inf
    for it in range(1, nmax + 1):
        g, w, b = lm.rates(gr, p, d, m, pc, qc, t1, t2)
        n, nb = occ(mu)
        a, bl = balance(g, w, n, nb)
        dt, mt = targets(gr, p, b, n, m)
        if it == 1 or it % chk == 0:
            jl = jlog(g, w, mu)
            err = max(np.max(np.abs(a / np.maximum(a + bl, 1e-300) - n)), abs(dt - d), abs(mt - m), abs(jl))
            for k, x in zip(hist, (it, err, d, m, jl)):
                hist[k].append(x)
            if prog:
                print(f"{it:7d}  err {err:.2e}  d {d:.8f}  m {m:.8f}  jlog {jl:+.2e}")
            if err < tol:
                break
        mu = _fix_fill(mu + xn * (_target_mu(a, bl, mu) - mu), w, p.n)
        if shift:
            mu = _band_shift(mu, g, w, p.n)
        d, m = d + xd * (dt - d), m + xm * (mt - m)
    n, _ = occ(mu)
    return Sol(mu, n, d, m, err, it, hist)


def _residual(z, gr, p, pc, qc, t1, t2):
    k = 2 * len(gr.s)
    mu, d, m = z[:k], z[k], z[k + 1]
    g, w, b = lm.rates(gr, p, d, m, pc, qc, t1, t2)
    n, nb = occ(mu)
    a, bl = balance(g, w, n, nb)
    dt, mt = targets(gr, p, b, n, m)
    r = np.concatenate((mu - _target_mu(a, bl, mu), [np.sum(w * n) - p.n, d - dt, m - mt]))
    return r, (g, w, b, n, nb, a, bl)


def solve_newton(gr, p, pc, qc, t1, t2, mu, d, m, tol=1e-10, nmax=60, cap=10.0, prog=False):
    """Newton on z = (mu, d, m). Rows: mu - mu_t (2N), filling, gap, imbalance.
    mu block analytic, d and m columns by forward differences, least squares step
    (the filling row makes the system overdetermined by one), step capped at cap
    in mu and backtracked on the residual norm."""
    k = 2 * len(gr.s)
    z = np.concatenate((np.asarray(mu, float), [d, m]))
    r, (g, w, b, n, nb, a, bl) = _residual(z, gr, p, pc, qc, t1, t2)
    hist = {"it": [0], "err": [np.max(np.abs(r))]}
    for it in range(1, nmax + 1):
        if hist["err"][-1] < tol:
            break
        nn = n * nb
        act = ((a > 0) | (bl > 0))[:, None]
        jm = np.eye(k) - act * (g.T / np.maximum(bl, 1e-300)[:, None]
                                + g / np.maximum(a, 1e-300)[:, None]) * (w * nn)[None, :]
        ns = k // 2
        c = p.v * gr.w * b.d / (2 * b.e)
        x = gr.w * lm.xe_cell(p, gr, b.d, z[k + 1])
        rows = np.zeros((3, k))
        rows[0] = -w * nn
        rows[1, :ns], rows[1, ns:] = -c * nn[:ns], c * nn[ns:]
        rows[2, :ns], rows[2, ns:] = x * nn[:ns], -x * nn[ns:]
        jac = np.zeros((k + 3, k + 2))
        jac[:k, :k], jac[k:, :k] = jm, rows
        for i in (k, k + 1):
            h = 1e-6 * max(abs(z[i]), 0.1)
            zp = z.copy()
            zp[i] += h
            jac[:, i] = (_residual(zp, gr, p, pc, qc, t1, t2)[0] - r) / h
        step = np.linalg.lstsq(jac, -r, rcond=None)[0]
        step *= min(1.0, cap / max(np.max(np.abs(step[:k])), 1e-300))
        lam, nr = 1.0, np.linalg.norm(r)
        while lam > 1e-4:
            rt, aux = _residual(z + lam * step, gr, p, pc, qc, t1, t2)
            if np.linalg.norm(rt) < nr or lam <= 1.5e-4:
                break
            lam *= 0.5
        z, r, (g, w, b, n, nb, a, bl) = z + lam * step, rt, aux
        hist["it"].append(it)
        hist["err"].append(np.max(np.abs(r)))
        if prog:
            print(f"{it:3d}  err {hist['err'][-1]:.2e}  step {lam:.3f}  d {z[k]:.10f}  m {z[k + 1]:.10f}")
    return Sol(z[:k], n, z[k], z[k + 1], hist["err"][-1], it, hist)


def solve_qf(gr, p, pc, t1, t2, d, m):
    """Quasi-Fermi limit: n_gamma = FD((E_gamma - mu_gamma) / T2). Unknowns (mu_a - mu_b, d, m),
    the common mu from the filling; rows: ln of the ratio of the summed photon up and down
    terms, gap, imbalance. Continuation in 1 / T1 from T2. Returns (mu_a, mu_b, d, m) and a Sol."""
    ns = len(gr.s)
    w2 = np.tile(gr.w, 2)

    def state(z):
        """z = (mu_a - mu_b, d, m); the common mu follows from the filling."""
        dm, d, m = z
        b = lm.mf(p, gr.s, d, m)
        e = np.r_[b.ea - 0.5 * dm, b.eb + 0.5 * dm]
        lo, hi = np.min(e) - 1 - 40 * t2, np.max(e) + 1 + 40 * t2
        mc = brentq(lambda c: np.sum(w2 * occ((e - c) / t2)[0]) - p.n, lo, hi, xtol=1e-14)
        x = (e - mc) / t2
        return b, x, mc, *occ(x)

    def res(z):
        b, x, _, n, nb = state(z)
        up, dn = lm.vertical(p, b, gr.hb, pc, t1)
        ln, lnb = log_expit(-x), log_expit(x)     # ln n, ln (1 - n)
        with np.errstate(divide="ignore"):
            lw, lu, ld = np.log(gr.w), np.log(up), np.log(dn)
        # ln (sum of beta -> alpha terms) - ln (sum of alpha -> beta terms)
        jb = logsumexp(lw + lu + ln[ns:] + lnb[:ns]) - logsumexp(lw + ld + ln[:ns] + lnb[ns:])
        jb = np.nan_to_num(jb, nan=0.0, posinf=700.0, neginf=-700.0)
        dt, mt = targets(gr, p, b, n, z[2])
        return [jb, z[1] - dt, z[2] - mt]

    def res_normal(y):
        r = res([y[0], 0.0, y[1]])
        return [r[0], r[2]]

    def solve_normal(_):
        """d = 0: the balance is monotone in mu_a - mu_b, and m - m_t changes sign on [-1, 1];
        both are bracketed (nested brentq)."""
        def dmu(m):
            return brentq(lambda x: res([x, 0.0, m])[0], -60 * t2 - 20, 60 * t1 + 20, xtol=1e-14)

        m = brentq(lambda m: res([dmu(m), 0.0, m])[2], -p.n, p.n, xtol=1e-14)
        y = np.array([dmu(m), m])
        return SimpleNamespace(x=y, success=True, nfev=0, fun=res_normal(y))

    # continuation in 1 / T1 from T2, step halved on failure; when the ordered branch
    # ends (fold, first order loss of order) the ramp continues in the normal state d = 0
    z, s, ds, t1_target, nfev, normal = np.array([0.0, d, m], float), 0.0, 0.1, t1, 0, False
    hist = {"t1": [], "d": [], "fold": None}
    while s < 1:
        st = min(1.0, s + ds)
        t1 = 1 / ((1 - st) / t2 + st / t1_target)
        r = solve_normal(z[2]) if normal else root(res, z, method="hybr", options={"xtol": 1e-13})
        nfev += r.nfev
        ok = r.success or np.max(np.abs(r.fun)) < 1e-10
        if not normal and abs(r.x[1]) < 1e-6 < abs(z[1]):
            ok = False                     # fell onto the trivial d = 0 root, not a fold yet
        if ok:
            z = np.array([r.x[0], 0.0, r.x[1]]) if normal else r.x
            s, ds = st, min(2 * ds, 0.25)
            hist["t1"].append(t1)
            hist["d"].append(z[1])
        elif ds > 1e-4:
            ds *= 0.5
        elif not normal:
            normal, ds, hist["fold"] = True, 0.1, t1
        else:
            raise RuntimeError(f"quasi-Fermi solve at T1 = {t1}: {r.message}")
    b, x, mc, n, nb = state(z)
    err = np.max(np.abs(res_normal(z[[0, 2]]) if normal else res(z)))
    return (mc + 0.5 * z[0], mc - 0.5 * z[0], z[1], z[2]), Sol(x, n, z[1], z[2], err, nfev, hist)


def t_eff(gr, p, d_ness, t_c):
    """Equilibrium temperature whose gap equals |d| of the NESS (0 if |d| exceeds d(T = 0),
    None for a normal NESS, where the gap does not define it)."""
    if abs(d_ness) < 1e-8:
        return None
    ts = np.linspace(1e-3, 0.995, 100) * t_c
    tab, z = [], (1.0, 0.0)
    for t in ts:                                       # continuation from low T
        z = lm.thermal(gr, p, t, d0=z[0], m0=z[1])[:2]
        tab.append(z)
    dd = np.abs([x[0] for x in tab])
    if abs(d_ness) >= dd[0]:
        return 0.0
    if abs(d_ness) <= dd[-1]:
        return ts[-1]
    i = np.nonzero(dd < abs(d_ness))[0][0]           # bracket [ts[i - 1], ts[i]]
    d0, m0 = tab[i - 1]
    return brentq(lambda t: abs(lm.thermal(gr, p, t, d0=d0, m0=m0)[0]) - abs(d_ness),
                  ts[i - 1], ts[i], xtol=1e-12)
