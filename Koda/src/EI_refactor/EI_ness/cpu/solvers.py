"""NESS solvers, numpy. Unknowns are the GGE potentials mu, n = 1 / (1 + e^mu) (eq. 191).

solve:        root of the two mean field equations in (log d, m); every evaluation is an
              exact Newton solve for the stationary mu of the frozen rates.
solve_fixed:  Anderson accelerated fixed point mu = log(b / a), rates refreshed every chk steps.
Both report the same scaled residuals (resid), so "converged" means the same thing.
"""

from typing import Any, NamedTuple

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import expit

from ..model import fermi, mf, targets
from .rates import rates

MC = 600.0  # |mu| clip, keeps n (1 - n) >= e^-600
TINY = 1.0e-280  # rates below this count as zero


class Sol(NamedTuple):
    d: float
    m: float
    n: Any  # occupations (2, ni), frozen outside the window
    mu: Any  # GGE potentials (2, ni)
    st: Any  # mean field state at (d, m)
    err: float
    ok: bool
    hist: dict


def _loop(n, f, x):
    for i in range(n):
        x = f(i, x)
    return x


def _full(y, pos, fr=(0.0, 1.0)):
    """Kinetic vector (2 nw,) -> (2, ni); frozen states get fr = (alpha, beta)."""
    return np.concatenate((y.reshape(2, -1), np.array(fr)[:, None]), 1)[:, pos]


def cur(r, ww, mu):
    """dn = (1 - n) a - n b (eq. 192), with the gain (1 - n) a and the loss n b."""
    y, h = expit(-mu), expit(mu)
    ga, lo = h * (r @ (ww * y)), y * (r.T @ (ww * h))
    return ga - lo, ga, lo


def resid(md, g, r, mu, d, m):
    """Scaled residuals (en, ep, ed, em), all zero in the NESS.

    en = max |gain - loss| / (gain + loss) = max |tanh((mu - log(b / a)) / 2)|
    ep = net / gross interband flux of the alpha band (the slow pair number mode)
    ed = |d* - d| / max(d, 1e-8), em = |m* - m|
    """
    nw = mu.shape[0] // 2
    ww = np.concatenate((g.w[g.wi], g.w[g.wi]))
    dn, ga, lo = cur(r, ww, mu)
    gl = ga + lo
    en = np.max(np.where((gl > TINY) & (np.abs(mu) < MC), np.abs(dn) / np.where(gl > TINY, gl, 1.0), 0.0))
    y, h = expit(-mu), expit(mu)
    up = h[:nw] * (r[:nw, nw:] @ (ww * y)[nw:])  # beta -> alpha
    dw = y[:nw] * (r[nw:, :nw].T @ (ww * h)[nw:])  # alpha -> beta
    ps = np.sum(ww[:nw] * (up + dw))
    ep = np.abs(np.sum(ww[:nw] * (up - dw))) / np.where(ps > 0.0, ps, 1.0)
    dt, mt, _ = targets(md, g, mf(md, g, d, m), _full(y, g.pos))
    return np.stack((en, ep, np.abs(dt - d) / np.maximum(np.abs(d), 1.0e-8), np.abs(mt - m)))


# ---------------- inner problem: stationary mu of frozen rates ----------------

def frozen(r):
    """States outside the largest connected component of the rate graph; they keep their n."""
    _, lab = connected_components((r > 0.0) | (r.T > 0.0), directed=False)
    return lab != np.argmax(np.bincount(lab))


def _res(r, ri, ww, pn, x, dead):
    """Rows of occ and the pieces its Jacobian needs."""
    nn = ww.shape[0]
    mu = x[:nn]
    y, h = expit(-mu), expit(mu)
    bd = np.arange(nn) >= nn // 2  # beta states
    a, b = r @ (ww * y), r.T @ (ww * h)
    raw = np.log(np.maximum(b, TINY)) - np.log(np.maximum(a, TINY))
    idle = dead | ((a <= TINY) & (b <= TINY))  # no rates: keep mu
    ok = ~idle & (a > TINY) & (b > TINY) & (np.abs(raw) < MC)  # the rest sits at the clip
    pa = np.where(~bd & ~dead, ww, 0.0)
    ai, bi = ri @ (ww * y), ri.T @ (ww * h)
    up, dw = np.sum(pa * h * ai), np.sum(pa * y * bi)  # interband flux into and out of alpha
    pok = (up > TINY) & (dw > TINY)
    f = np.where(ok, mu - raw + np.where(bd, x[nn + 1], x[nn]),
                 np.where(idle, 0.0, mu - np.clip(raw, -MC, MC)))
    fp = np.where(pok, np.log(np.where(pok, up, 1.0)) - np.log(np.where(pok, dw, 1.0)), x[nn])
    return (np.concatenate((f, np.stack((np.sum(ww * y) - pn, fp)))),
            (y, h, bd, a, b, ai, bi, ok, pa, up, dw, pok))


def _step(r, ri, ww, x, f, aux, cap):
    """Newton step of occ with the analytic Jacobian; the whole step is scaled to |dmu| <= cap."""
    nn = ww.shape[0]
    y, h, bd, a, b, ai, bi, ok, pa, up, dw, pok = aux
    s = ww * y * h  # -w dn / dmu
    ia, ib = 1.0 / np.where(ok, a, 1.0), 1.0 / np.where(ok, b, 1.0)
    jmm = np.eye(nn) - np.where(ok[:, None], s[None, :] * (r.T * ib[:, None] + r * ia[:, None]), 0.0)
    ka = np.stack((np.where(ok & ~bd, 1.0, 0.0), np.where(ok & bd, 1.0, 0.0)), 1)
    du = pa * y * h * ai - s * (ri.T @ (pa * h))
    dd = -pa * y * h * bi + s * (ri @ (pa * y))
    rp = np.where(pok, np.concatenate((du / np.where(pok, up, 1.0) - dd / np.where(pok, dw, 1.0),
                                       np.zeros(2))),
                  np.concatenate((np.zeros(nn), np.array([1.0, 0.0]))))
    jm = np.concatenate((np.concatenate((jmm, ka), 1), np.concatenate((-s, np.zeros(2)))[None], rp[None]))
    dx = np.linalg.solve(jm, -f)
    return dx * np.minimum(1.0, cap / np.maximum(np.max(np.abs(dx[:nn])), 1.0e-300))


def _clip(x):
    return np.concatenate((np.clip(x[:-2], -MC, MC), x[-2:]))


def _ssq(r, ri, ww, pn, x, dead):
    return np.sum(_res(r, ri, ww, pn, x, dead)[0] ** 2)


def occ(r, ww, pn, mu, dead, tol=1.0e-11, nmax=80, cap=50.0):
    """Stationary state of the frozen rates R at filling pn, Newton in mu.

    Unknowns: mu (2 nw), kap_alpha, kap_beta. Rows, all in units of mu:
      F_i = mu_i - log(b_i / a_i) + kap_band(i)   gain_i e^kap = loss_i (eq. 192, 193)
      filling: sum_i w_i n_i - pn
      pair:    log of interband flux into alpha / out of alpha
    Intraband rates conserve N_alpha and N_beta, so the state rows of a band are
    redundant by one; the kap absorb this and vanish at the solution. The pair number
    is fixed by the pair row, which uses only interband rates (no cancellation of the
    large intraband terms). The log form does not saturate far from the solution and
    its Jacobian is 1 - (averages), so no row or column scaling is needed.
    States whose target lies beyond |mu| = MC, or with a = 0 or b = 0, sit at the clip.
    Returns mu and err = max |row|.
    """
    nn = ww.shape[0]
    bd = np.arange(nn) >= nn // 2
    ri = np.where(bd[:, None] != bd[None, :], r, 0.0)  # interband part
    x = np.concatenate((np.clip(mu, -MC, MC), np.zeros(2)))
    for _ in range(nmax):
        f, aux = _res(r, ri, ww, pn, x, dead)
        err = float(np.max(np.abs(f)))
        if err < tol:
            break
        dx = _step(r, ri, ww, x, f, aux, cap)
        t, f0 = 1.0, float(np.sum(f**2))
        while t > 1.0e-6:  # backtracking on |F|^2
            f1 = float(_ssq(r, ri, ww, pn, _clip(x + t * dx), dead))
            if np.isfinite(f1) and f1 < (1.0 - 1.0e-4 * t) * f0:
                break
            t *= 0.5
        # no decrease: one plain fixed point step mu -> log(b / a) - kap
        x = _clip(x + t * dx) if t > 1.0e-6 else _clip(np.concatenate((x[:nn] - f[:nn], x[nn:])))
    return x[:nn], err


# ---------------- NESS: Newton ----------------

def _start(md, g, pk, d, m, mu):
    ww = np.concatenate((g.w[g.wi], g.w[g.wi]))
    pn = md.n - 1.0 + np.sum(g.w[g.wi])  # filling left for the window
    if mu is None:  # Fermi state at the coldest bath, directly in potentials
        st, t = mf(md, g, d, m), max(pk.tmin, 1.0e-3)
        mu = (st.e - fermi(st, g.w, md.n, t)[1]) / t
    return ww, pn, np.clip(np.asarray(mu)[:, g.wi].reshape(-1), -MC, MC)


def _out(md, g, mu, d, m, err, ok, hs):
    return Sol(float(d), float(m), _full(expit(-mu), g.pos), _full(mu, g.pos, (MC, -MC)),
               mf(md, g, d, m), float(err), bool(ok), {q: np.asarray(v) for q, v in hs.items()})


def _root(fun, x, lo, hi, st, tol, c):
    """Root of fun next to x along the flow dx/dt = fun(x), for x in [lo, hi].

    x moves in the direction sign(fun) with doubling steps until fun changes sign,
    then the Illinois secant closes the bracket; this ends at a root where fun
    decreases (a stable one). The first step is 2 |fun / slope| if c holds the slope
    of an earlier call, else st. Returns the last evaluated x, fun(x) and
    end = True if the bound was reached without a sign change.
    """
    a, fa = x, fun(x)
    if abs(fa) < tol:
        return a, fa, False
    sg = np.sign(fa)
    if c.get("sl"):
        st = min(st, max(2.0 * abs(fa / c["sl"]), 2.0 * tol))
    while True:
        b = min(max(a + sg * st, lo), hi)
        fb = fun(b)
        if b != a and fb != fa:
            c["sl"] = (fb - fa) / (b - a)
        if abs(fb) < tol:
            return b, fb, False
        if fa * fb < 0.0:
            break
        if b in (lo, hi):
            return b, fb, True
        a, fa, st = b, fb, 2.0 * st
    while True:
        x = b - fb * (b - a) / (fb - fa)
        f = fun(x)
        if x != b and f != fb:
            c["sl"] = (f - fb) / (x - b)
        if abs(f) < tol or abs(x - b) < 1.0e-15:
            return x, f, False
        if f * fb < 0.0:
            a, fa = b, fb
        else:
            fa *= 0.5
        b, fb = x, f


def solve(md, g, pk, d=1.0, m=0.0, mu=None, tol=1.0e-9, ntol=1.0e-11, dfl=1.0e-8, eps=1.0e-6,
          nit=12, nmax=300, rows=None):
    """Steady state from F(log d, m) = (d* / d - 1, m* - m) = 0; every evaluation of F is
    an exact inner solve (occ) for the stationary n of the rates at (d, m), and (d*, m*)
    are its targets. The log and the ratio exclude the trivial root d = 0.

    1. Newton in (log d, m): forward difference Jacobian, Broyden updates, steps capped
       at 1, backtracking on max |F|. Accepted only if the root is stable under the
       relaxation flow dz / dt = F (trace < 0, det > 0 of a fresh Jacobian).
    2. If Newton gives up (two capped steps in a row, a failed step, a bound, nit
       iterations) or the root is unstable: nested root finding from the start,
         inner: m(d) from F_1 = 0 with _root, |m| <= min(n, 2 - n)
         outer: phi(log d) = F_0 at m(d) with _root, dfl <= d <= V / 2.
       If phi < 0 down to d = dfl the state is normal and d = 0 is returned.
       If the bracket closes on a jump of phi instead of a root, Newton is tried from there.
    Start: d, m (equilibrium or the previous point of a scan) and potentials mu (2, ni),
    e.g. sol.mu of the previous point; None: Fermi state at the coldest bath.
    nmax bounds the number of rate builds. hist: one row per build.
    """
    ww, pn, mu = _start(md, g, pk, d, m, mu)
    dead = [None]
    hs = {q: [] for q in ("d", "m", "fd", "fm", "en")}
    c = {"z": 0.0, "m": float(m), "mu": mu, "en": np.inf, "f0": np.inf, "f1": np.inf}  # last evaluation

    def ev(z0, mm):
        if len(hs["d"]) >= nmax:
            raise RuntimeError("solve: too many rate builds")
        st = mf(md, g, np.exp(z0), mm)
        r = rates(md, g, st, pk, rows)
        if dead[0] is None:
            dead[0] = frozen(np.asarray(r))
            nw = g.wi.shape[0]
            if not np.any(np.asarray(r)[:nw, nw:][~dead[0][:nw]][:, ~dead[0][nw:]] > 0.0):
                raise ValueError("no interband rates: the pair number, hence the NESS, is not unique")
        for _ in range(3):  # a cold start may need a second pass
            c["mu"], c["en"] = occ(r, ww, pn, c["mu"], dead[0], tol=ntol)
            if c["en"] < 10.0 * ntol:
                break
        else:
            raise RuntimeError("solve: the inner problem did not converge")
        dt, mt, _ = targets(md, g, st, _full(expit(-c["mu"]), g.pos))
        c.update(z=z0, m=mm, f0=float(dt) / np.exp(z0) - 1.0, f1=float(mt) - mm)
        for q, v in zip(hs, (np.exp(z0), mm, c["f0"], c["f1"], c["en"])):
            hs[q].append(v)
        return np.array([c["f0"], c["f1"]])

    mx = min(md.n, 2.0 - md.n)
    zn, zx = np.log(dfl), np.log(max(0.5 * abs(md.v), 2.0 * dfl))
    z0 = np.array([min(max(np.log(abs(d)) if d else zn, zn), zx), float(m)])

    def fdj(z, f):
        return np.stack([(ev(*(z + eps * e)) - f) / eps for e in np.eye(2)], 1)

    def newton(z, n):
        f, jm, big = ev(*z), None, 0
        for _ in range(n):
            if np.max(np.abs(f)) < tol:
                sv = dict(c)
                jm = fdj(z, f)  # stability of the root, then back to its state
                c.update(sv)
                return np.trace(jm) < 0.0 < np.linalg.det(jm)
            new = jm is None
            if new:
                jm = fdj(z, f)
            dz = np.linalg.lstsq(jm, -f, rcond=None)[0]
            cc = min(1.0, 1.0 / max(np.max(np.abs(dz)), 1.0e-300))  # cap the step
            dz, big = cc * dz, big + 1 if cc < 1.0 else 0
            if big > 1 or not zn < z[0] + dz[0] < zx or abs(z[1] + dz[1]) > mx:
                return False
            t, good = 1.0, False
            while t > 0.1 and not good:  # backtracking, the model decrease is cc t
                f1 = ev(*(z + t * dz))
                good = np.max(np.abs(f1)) < (1.0 - 0.25 * cc * t) * np.max(np.abs(f))
                t *= 1.0 if good else 0.5
            if not good:
                if new:
                    return False
                jm = None
                continue
            s = t * dz
            jm = jm + np.outer(f1 - f - jm @ s, s) / (s @ s)  # Broyden
            z, f = z + s, f1
        return False

    cm, cz, nrm = {}, {}, False

    def phi(z):
        _root(lambda mm: ev(z, mm)[1], c["m"], -mx, mx, 0.05, 0.1 * tol, cm)
        return c["f0"]

    try:
        if not newton(z0, nit):
            c["m"] = z0[1]
            z, f, end = _root(phi, z0[0], zn, zx, 0.5, tol, cz)
            nrm = end and z == zn
            if not nrm and abs(f) >= tol:  # phi jumps (m changes branch): Newton from there
                newton(np.array([z, c["m"]]), 4 * nit)
    except RuntimeError:
        pass
    er = max(abs(c["f1"]), c["en"], 0.0 if nrm else abs(c["f0"]))
    return _out(md, g, c["mu"], 0.0 if nrm else np.exp(c["z"]), c["m"], er,
                er < tol and c["en"] < 10.0 * ntol, hs)


# ---------------- NESS: Anderson accelerated fixed point ----------------

def _proj(mu, ww, pn):
    """Common shift of mu that restores the filling."""
    hi = np.max(np.abs(mu)) + 80.0

    def body(_, ab):
        s = 0.5 * (ab[0] + ab[1])
        big = np.sum(ww * expit(-mu - s)) > pn
        return np.where(big, s, ab[0]), np.where(big, ab[1], s)

    lo, hi = _loop(64, body, (-hi, hi))
    return mu + 0.5 * (lo + hi)


def _fmap(md, g, r, ww, pn, sc, x):
    """f = G(x) - x at frozen rates, x = (mu, d / sc_0, m / sc_1).

    G: mu -> log(b / a) (the n that balances gain and loss, eq. 192) shifted to the
    filling, d and m -> their targets.
    """
    nn = ww.shape[0]
    mu, d, m = x[:nn], x[nn] * sc[0], x[nn + 1] * sc[1]
    y, h = expit(-mu), expit(mu)
    a, b = r @ (ww * y), r.T @ (ww * h)
    mt = np.clip(np.log(np.maximum(b, 1.0e-300)) - np.log(np.maximum(a, 1.0e-300)), -MC, MC)
    mt = _proj(np.where(a + b > 0.0, mt, mu), ww, pn)
    dt, tm, _ = targets(md, g, mf(md, g, d, m), _full(y, g.pos))
    return np.concatenate((mt, np.stack((dt / sc[0], tm / sc[1])))) - x


def _chunk(md, g, r, ww, pn, sc, mix, reg, cap, x, ns, mem):
    """ns Anderson steps (memory mem, 0: plain mixing) at frozen rates."""
    nn = ww.shape[0]

    def fm(q):
        return _fmap(md, g, r, ww, pn, sc, q)

    def px(q):
        return np.concatenate((_proj(q[:nn], ww, pn), q[nn:]))

    def step(_, ca):
        x, xp, fp, dx, df, first = ca
        f = fm(x)
        xb = px(x + mix * f)
        if mem == 0:
            return xb, x, f, dx, df, first
        dx = np.concatenate((dx[1:], np.where(first, 0.0, x - xp)[None]))
        df = np.concatenate((df[1:], np.where(first, 0.0, f - fp)[None]))
        q = df @ df.T
        c = np.linalg.solve(q + reg * np.maximum(np.trace(q) / mem, 1.0e-30) * np.eye(mem), df @ f)
        dv = mix * f - c @ (dx + mix[None, :] * df)
        xa = px(x + np.minimum(1.0, cap / np.maximum(np.max(np.abs(dv[:nn])), 1.0e-30)) * dv)
        ea, eb = np.max(np.abs(fm(xa))), np.max(np.abs(fm(xb)))
        acc = ~first & np.isfinite(ea) & (ea <= eb) & (ea < np.max(np.abs(f)))
        keep = acc | first  # restart the history after a rejected extrapolation
        return (np.where(acc, xa, xb), x, f, np.where(keep, dx, 0.0), np.where(keep, df, 0.0),
                np.asarray(False))

    z = np.zeros((mem, x.shape[0]))
    return _loop(ns, step, (x, x, np.zeros_like(x), z, z, np.asarray(True)))[0]


def _rho(r, ww, pn, mu, dl):
    """log(interband flux into alpha / out of alpha) after the shift +-dl and the filling shift."""
    nw = ww.shape[0] // 2
    q = _proj(mu + dl * np.concatenate((np.ones(nw), -np.ones(nw))), ww, pn)
    y, h = expit(-q), expit(q)
    up = np.sum(ww[:nw] * h[:nw] * (r[:nw, nw:] @ (ww * y)[nw:]))
    dw = np.sum(ww[:nw] * y[:nw] * (r[nw:, :nw].T @ (ww * h)[nw:]))
    return np.log(up) - np.log(dw), q


def _pfix(r, ww, pn, mu, nit=8, h=1.0e-3):
    """Shift mu_alpha by +dl and mu_beta by -dl until the interband flux vanishes at frozen
    rates (Newton on _rho). This is the slow pair number mode that mixing barely moves."""
    dl, (p0, q) = 0.0, _rho(r, ww, pn, mu, 0.0)
    for _ in range(nit):
        if not np.isfinite(float(p0)) or abs(float(p0)) < 1.0e-10:
            break
        dl -= float(p0) * h / (float(_rho(r, ww, pn, mu, dl + h)[0]) - float(p0))
        p0, q = _rho(r, ww, pn, mu, dl)
    return q if np.isfinite(float(p0)) else mu


def solve_fixed(md, g, pk, d=1.0, m=0.0, mu=None, mix=(0.2, 0.2, 0.2), mem=4, chk=10, tol=1.0e-6,
                nmax=5000, reg=1.0e-8, cap=5.0, pc=True, rows=None):
    """Fixed point iteration in (mu, d, m) with Anderson acceleration (mem = 0: Picard).

    The rates are frozen for chk steps, then rebuilt at the new (d, m); the residuals
    of resid are evaluated with the rebuilt rates. Stops when their maximum is below tol.
    pc: after every chunk the pair number is corrected with _pfix.
    """
    ww, pn, mu = _start(md, g, pk, d, m, mu)
    nn = ww.shape[0]
    sc = np.array([max(abs(d), 0.1), max(abs(m), 0.1)])
    mx = np.concatenate((np.full(nn, mix[0]), np.asarray(mix[1:], dtype=float)))
    x = np.concatenate((_proj(mu, ww, pn), np.array([d, m]) / sc))
    r = rates(md, g, mf(md, g, d, m), pk, rows)
    hs = {q: [] for q in ("it", "d", "m", "en", "ep", "ed", "em")}
    ok, er = False, np.inf
    for i in range(0, int(nmax), int(chk)):
        ns = min(int(chk), int(nmax) - i)
        x = _chunk(md, g, r, ww, pn, sc, mx, reg, cap, x, ns, int(mem))
        if pc:
            x = np.concatenate((_pfix(r, ww, pn, x[:nn]), x[nn:]))
        d, m = float(x[nn] * sc[0]), float(x[nn + 1] * sc[1])
        r = rates(md, g, mf(md, g, d, m), pk, rows)
        e4 = np.asarray(resid(md, g, r, x[:nn], d, m))
        if not np.all(np.isfinite(e4)):
            raise RuntimeError("nonfinite state in the fixed point iteration")
        for q, v in zip(hs, (i + ns, d, m, *e4)):
            hs[q].append(v)
        er = float(e4.max())
        if er < tol:
            ok = True
            break
    return _out(md, g, x[:nn], abs(d), m, er, ok, hs)
