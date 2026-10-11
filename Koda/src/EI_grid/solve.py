"""Newton solver for the steady state of the bath driven EI on the k grid.

Unknowns: mu of every (band, k) with n = 1 / (1 + exp(mu)), the gap d, the
imbalance m and one multiplier kap. Equations:
  balance    sum_j w_j [g_ji (1 - n_j) / (1 - n_i) - g_ij n_j / n_i] + kap = 0
  gap        d - V sum_k w_k u_k v_k (n_beta - n_alpha) = 0        (eq. 37)
  imbalance  m - sum_k w_k (u_k^2 - v_k^2) (n_alpha - n_beta) = 0  (eq. 46)
  filling    sum_k w_k (n_alpha + n_beta) - n0 = 0
The balance row is the right hand side of eq. 193 summed over the baths; the
term with exp(Omega / T_b) is the reverse rate of eq. 187, taken as g_ji.
Particle number is conserved, so the balance rows are linearly dependent;
kap makes the system square and vanishes at the solution.
"""

from typing import Any, NamedTuple

import numpy as np

import EI.ei_unified as eu

from .backend import host, jit, sig, xp
from .bath import pack
from .rates import make_prob, mf, rate_mat, tgt


class Out(NamedTuple):
    mu: Any
    th: Any
    kap: Any
    st: str  # 'ok', 'stall' or 'maxit'
    err: float
    it: int


def _res(pk, bp, mu, th, kap):
    st = mf(pk, th[0], th[1])
    g = rate_mat(st, pk, bp)
    n, h = sig(-mu), sig(mu)
    ww = xp.concatenate((pk.w, pk.w))
    a = g @ (ww * n)  # in rate, sum_j g_ij n_j
    b = g.T @ (ww * h)  # out rate, sum_j g_ji (1 - n_j)
    r = b / h - a / n + kap
    d, m, nf = tgt(st, n.reshape(2, -1), pk)
    f = xp.stack((th[0] - d, th[1] - m, nf - pk.pn))
    return r, f, a, b, g


def _jmu(pk, mu, th, a, b, g):
    """Analytic derivatives of the balance and mean field rows with respect to mu."""
    st = mf(pk, th[0], th[1])
    n, h = sig(-mu), sig(mu)
    ww = xp.concatenate((pk.w, pk.w))
    s = ww * n * h  # -w dn/dmu
    jm = (g.T / h[:, None] + g / n[:, None]) * s[None, :]
    jm = jm - xp.diag(b * n / h + a * h / n)
    nk = pk.w.shape[0]
    sa, sb = s[:nk], s[nk:]
    uv = st.u * st.v
    x = st.u * st.u - st.v * st.v
    jd = pk.pv * xp.concatenate((-uv * sa, uv * sb))
    jq = xp.concatenate((x * sa, -x * sb))
    return jm, xp.stack((jd, jq, -s))


_res = jit(_res)
_jmu = jit(_jmu)


def residual(pk, bp, mu, th, kap=0.0):
    """Balance rows (2N,), eq. 193 plus kap, and the (gap, imbalance, filling) rows."""
    r, f, _, _, _ = _res(pk, bp, mu, xp.asarray(th, dtype=xp.float64), kap)
    return r, f


def jac(pk, bp, mu, th, kap, ed=1.0e-6):
    """Residual and Jacobian in the order (mu, d, m, kap).

    mu columns are analytic; d and m columns are central differences.
    """
    r, f, a, b, g = _res(pk, bp, mu, th, kap)
    jm, jf = _jmu(pk, mu, th, a, b, g)
    cols = []
    for j in range(2):
        hh = ed * max(abs(float(th[j])), 0.1)
        e = xp.asarray(np.eye(2)[j] * hh)
        rp, fp, _, _, _ = _res(pk, bp, mu, th + e, kap)
        rm, fm, _, _, _ = _res(pk, bp, mu, th - e, kap)
        cols.append(xp.concatenate((rp - rm, fp - fm)) / (2.0 * hh))
    jt = xp.stack(cols, axis=1)
    nm = mu.shape[0]
    top = xp.concatenate((jm, jt[:nm], xp.ones((nm, 1))), axis=1)
    bot = xp.concatenate((jf, jt[nm:], xp.zeros((3, 1))), axis=1)
    return r, f, xp.concatenate((top, bot)), a, b


def _scale(mu, a, b):
    """Row scale 1 / (a / n + b / (1 - n)): balance rows become relative."""
    return 1.0 / (a / sig(-mu) + b / sig(mu))


def _red(r, f, sc, nk, ph):
    """Scaled residual vector; with ph only the alpha, gap and imbalance rows."""
    if ph:
        return xp.concatenate((sc[:nk] * r[:nk], f[:2]))
    return xp.concatenate((sc * r, f))


def newton(pk, bp, mu, th, kap=0.0, tol=1.0e-10, nmax=30, cap=10.0, ph=False,
           hist=None):
    """Newton iteration with backtracking; returns an Out tuple.

    cap limits the change of mu per step. ph=True uses mu_beta = -mu_alpha
    (particle hole symmetric bands at n0 = 1): unknowns (mu_alpha, d, m),
    the filling row and kap drop out.
    """
    mu = xp.asarray(mu, dtype=xp.float64).reshape(-1)
    th = xp.asarray(th, dtype=xp.float64)
    nk = mu.shape[0] // 2
    if ph:
        mu, kap = xp.concatenate((mu[:nk], -mu[:nk])), 0.0
    hist = {"err": [], "step": []} if hist is None else hist
    st, it = "maxit", 0
    for it in range(nmax + 1):
        r, f, J, a, b = jac(pk, bp, mu, th, kap)
        sc = _scale(mu, a, b)
        F = _red(r, f, sc, nk, ph)
        err = float(xp.max(xp.abs(F)))
        hist["err"].append(err)
        if not np.isfinite(err):
            st = "stall"
            break
        if err < tol:
            st = "ok"
            break
        if it == nmax:
            break
        if ph:
            ix = np.r_[np.arange(nk), 2 * nk, 2 * nk + 1]
            A = J[ix]
            A = xp.concatenate((A[:, :nk] - A[:, nk:2 * nk], A[:, 2 * nk:2 * nk + 2]),
                               axis=1)
            sr = xp.concatenate((sc[:nk], xp.ones(2)))
        else:
            A = J
            sr = xp.concatenate((sc, xp.ones(3)))
        dz = xp.linalg.solve(A * sr[:, None], -F)

        def move(s):
            if ph:
                ma = mu[:nk] + s * dz[:nk]
                return xp.concatenate((ma, -ma)), th + s * dz[nk:nk + 2], 0.0
            return (mu + s * dz[:2 * nk], th + s * dz[2 * nk:2 * nk + 2],
                    kap + s * dz[-1])

        nmu = nk if ph else 2 * nk
        s = min(1.0, cap / max(float(xp.max(xp.abs(dz[:nmu]))), 1.0e-300))
        f0 = float(xp.sum(F * F))
        while s > 1.0e-6:
            z1 = move(s)
            r1, f1, _, _, _ = _res(pk, bp, z1[0], z1[1], z1[2])
            f1 = float(xp.sum(xp.square(_red(r1, f1, sc, nk, ph))))
            if np.isfinite(f1) and f1 < (1.0 - 1.0e-4 * s) * f0:
                break
            s *= 0.5
        hist["step"].append(s)
        if s <= 1.0e-6:
            st = "stall"
            break
        mu, th, kap = z1
    return Out(mu, th, kap, st, err, it)


def _ph_ok(bd, p):
    sc = max(float(np.ptp(bd.ea)), 1.0)
    return bool(np.ptp(bd.ea + bd.eb) < 1.0e-12 * sc and abs(p.n - 1.0) < 1.0e-14)


def _sol(bd, p, pk, bp, o, ok, hist, ph):
    """Pack the result as eu.OpenSol; eta holds mu, cur holds dn/dt."""
    r, f, a, b, _ = _res(pk, bp, o.mu, o.th, o.kap)
    nk = bd.size
    n, h = sig(-o.mu), sig(o.mu)
    err = float(xp.max(xp.abs(_red(r, f, _scale(o.mu, a, b), nk, ph))))
    d, m = float(o.th[0]), float(o.th[1])
    js = mf(pk, d, m)
    st = eu.MFState(bd, p, d, m, float(js.na), float(js.nb), host(js.eah),
                    host(js.ebh), host(js.e), host(js.u), host(js.v))
    hh = {q: np.asarray(z) for q, z in hist.items()}
    hh["kap"] = np.asarray(float(o.kap))
    it = len(hist["err"])
    return eu.OpenSol(d, m, host(n).reshape(2, nk), host(o.mu).reshape(2, nk), st,
                      host(h * a - n * b).reshape(2, nk), err, it, float(it),
                      bool(ok), hh)


def pair_chi(pk, mu, m):
    """V chi_0 at d = 0 for occupations mu: above 1 the normal state is unstable
    towards order (the gap equation, eq. 37, divided by d at d -> 0)."""
    st = mf(pk, 0.0, m)
    n = sig(-mu).reshape(2, -1)
    ek = xp.maximum(0.5 * (st.e[0] - st.e[1]), 1.0e-12)
    return float(pk.pv * xp.sum(pk.w * (n[1] - n[0]) / (2.0 * ek)))


def solve_ness(bd, p, geo, bs, mu=None, d=0.2, m=0.0, t0=None, tol=1.0e-10,
               nmax=30, cap=10.0, ph=False, dlmin=1.0 / 64, dmin=1.0e-7, eq=None):
    """Steady state for baths bs on the grid of bd; returns eu.OpenSol.

    mu given: one Newton solve from (mu, d, m), e.g. for continuation.
    Otherwise all baths start at the common temperature t0 (default: the
    coldest bath), where the thermal state of eu.solve_eq(bd, p, t0, d, m,
    **eq) solves all equations, and the temperatures are moved to their
    targets (linearly in 1/T) in steps that are halved when Newton fails.
    A step from an ordered state to |d| < dmin is also halved first; at the
    smallest step dlmin the order counts as lost (hist['lost']) and d = 0 is
    continued. ok is False if the targets were not reached; hist['lam'] is
    the path. For a normal result hist['chi'] is pair_chi.
    """
    bs = tuple(bs)
    pk = make_prob(bd, p, geo)
    if ph and not _ph_ok(bd, p):
        raise ValueError("ph=True needs ea + eb constant in k and n = 1")
    hist = {"err": [], "step": [], "lam": []}
    kw = dict(tol=tol, nmax=nmax, cap=cap, ph=ph, hist=hist)
    tt = np.asarray([b.t for b in bs], dtype=float)

    if mu is not None:
        o = newton(pk, pack(bs), mu, (d, m), **kw)
        ok = o.st == "ok"
        hist["lam"].append(1.0 if ok else 0.0)
    else:
        t0 = float(tt.min() if t0 is None else t0)
        se = eu.solve_eq(bd, p, t=t0, d=d, m=m, **{"prog": False, **(eq or {})})
        mu = ((se.st.e - se.mu) / t0).reshape(-1)
        d0 = se.d if abs(se.d) > dmin else 0.0
        o = Out(xp.asarray(mu), xp.asarray((d0, se.m)), 0.0, "ok", 0.0, 0)
        lam, dl, ok = 0.0, 1.0, False
        while True:
            lt = min(1.0, lam + dl)
            # path linear in the inverse temperatures
            tl = 1.0 / (1.0 / t0 + lt * (1.0 / tt - 1.0 / t0))
            o1 = newton(pk, pack(bs, tl), o.mu, o.th, o.kap, **kw)
            good = o1.st == "ok"
            lost = (good and abs(float(o.th[0])) > dmin
                    and abs(float(o1.th[0])) <= dmin)
            if good and not (lost and 0.5 * dl >= dlmin):
                if lost:
                    hist["lost"] = lt
                    o1 = o1._replace(th=xp.asarray((0.0, float(o1.th[1]))))
                o, lam = o1, lt
                hist["lam"].append(lam)
                if lam >= 1.0:
                    ok = True
                    break
                dl = min(1.0, 2.0 * dl)
                continue
            dl *= 0.5
            if dl < dlmin:
                break
    if abs(float(o.th[0])) <= dmin:
        hist["chi"] = pair_chi(pk, o.mu, float(o.th[1]))
    return _sol(bd, p, pk, pack(bs), o, ok, hist, ph)
