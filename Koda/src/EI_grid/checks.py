"""Basic checks of the rates and the solver. Run: python -m EI_grid.checks

Every function returns small numbers that should vanish (or a short dict).
All grids are tiny, the whole set takes a few seconds.
"""

import numpy as np

import EI.ei_unified as eu

from .backend import NAME, host, sig, xp
from .bath import Bath, bose, lor, nlor, pack, xbose
from .rates import cur, dense_rates, dissipator, make_prob, mf, parts, rate_mat
from .solve import _res, jac, solve_ness
from .vertex import make_geom

BP = dict(gap=2.0, ta=1.0, tb=-1.0)
MP = dict(v=4.4, n=1.0)


def setup(nk=6, dip=0.5, th=0.0, bp=BP, mp=MP):
    bd = eu.tb_2d(nk, **bp)
    p = eu.MFPars(**mp)
    return bd, p, make_geom(bd, bp["ta"], bp["tb"], dip, th)


def baths(t1=0.6, t2=0.3, w0=2.2, c=1.6, gam=0.3):
    return (Bath(t1, w0, c, gam, 1.0, "bath_1"), Bath(t2, 0.7 * w0, c, gam, 0.5, "bath_2"))


def scalar(seed=0):
    """Regular form of N(x) [A(x) - A(-x)] against the literal product."""
    rg = np.random.default_rng(seed)
    x = rg.uniform(-6.0, 6.0, 4000)
    om = rg.uniform(0.1, 4.0, 4000)
    gam, t = 0.3, 0.7
    lit = host(bose(x, t) * (lor(x, om, gam) - lor(-x, om, gam)))
    reg = host(nlor(x, om, gam, t))
    e1 = np.max(np.abs(lit - reg) / reg)
    e2 = np.max(np.abs(host(bose(-x, t)) + 1.0 + host(bose(x, t))))
    e3 = abs(float(xbose(xp.asarray(0.0), t)) - t)
    return float(e1), float(e2), e3


def m2_brute(bd, st, ta, tb, dip, th, b):
    """sum_pol |U_k^T M(k', q) U_k'|^2 from the 2x2 matrices of eq. 125."""
    k, nk = np.asarray(bd.k), bd.size
    out = np.zeros((2 * nk, 2 * nk))
    dv = dip * np.array([np.cos(th), np.sin(th)])
    for i in range(nk):
        ui = np.array([[st.u[i], st.v[i]], [-st.v[i], st.u[i]]])
        for j in range(nk):
            uj = np.array([[st.u[j], st.v[j]], [-st.v[j], st.u[j]]])
            q = (k[i] - k[j] + np.pi) % (2.0 * np.pi) - np.pi
            phi = np.arctan2(q[1], q[0])
            om = np.sqrt(b.w0**2 + b.c**2 * (q @ q))
            sn = np.sin(k[j] + 0.5 * q)
            te = np.array([-np.sin(phi), np.cos(phi)])
            tm = b.w0 / om * np.array([np.cos(phi), np.sin(phi)])
            for ep in (te, tm):
                mm = np.array([
                    [2.0 * ta * (ep @ sn), 1j * (ep @ dv) * (bd.ea[i] - bd.eb[j])],
                    [1j * (ep @ dv) * (bd.eb[i] - bd.ea[j]), 2.0 * tb * (ep @ sn)],
                ]) * np.sqrt(b.amp / om)
                mt = ui.T @ mm @ uj
                out[np.ix_([i, nk + i], [j, nk + j])] += np.abs(mt) ** 2
    return out


def vertex(nk=5, d=0.6, m=-0.5, th=0.3):
    """Fast |M~|^2 against the brute force of eq. 125 (odd nk: no zone edge)."""
    bd, p, geo = setup(nk, th=th, bp=dict(gap=2.0, ta=1.0, tb=-0.4))
    b = baths()[0]
    x = parts(bd, p, geo, b, d, m)
    st = mf(make_prob(bd, p, geo), d, m)
    st = type(st)(*(host(z) for z in st))
    ref = m2_brute(bd, st, geo.ta, geo.tb, geo.dip, geo.th, b)
    e1 = np.max(np.abs(x["m2"] - ref)) / np.max(ref)
    e2 = np.max(np.abs(x["m2v"] + x["m2d"] - x["m2"])) / np.max(ref)
    return float(e1), float(e2)


def herm(nk=6, d=0.6, m=-0.5):
    """Symmetry of |M~|^2 (eq. 177) and invariance under the sign of d."""
    bd, p, geo = setup(nk, bp=dict(gap=2.0, ta=1.0, tb=-0.4))
    b = baths()[0]
    a = parts(bd, p, geo, b, d, m)["m2"]
    c = parts(bd, p, geo, b, -d, m)["m2"]
    return (float(np.max(np.abs(a - a.T)) / np.max(a)),
            float(np.max(np.abs(a - c)) / np.max(a)))


def db(nk=6, d=0.6, m=-0.5):
    """Detailed balance, eq. 188, for one bath: g_ij - g_ji exp(-Omega / T)
    relative to max g, and log g_ij - log g_ji + Omega / T for rates above
    1e-20 max g (rates that vanish to rounding have no defined ratio)."""
    bd, p, geo = setup(nk)
    b = baths()[0]
    st, g = dense_rates(bd, p, geo, (b,), d, m)
    e = st.e.reshape(-1)
    x = e[:, None] - e[None, :]
    ok = g > 1.0e-20 * g.max()
    z = np.log(g[ok]) - np.log(g.T[ok]) + x[ok] / b.t
    return (float(np.max(np.abs(g - g.T * np.exp(-x / b.t))) / g.max()),
            float(np.max(np.abs(z))))


def number(nk=6, d=0.6, m=-0.5, seed=1):
    """Particle number rate sum_i w_i dn_i/dt for random occupations."""
    bd, p, geo = setup(nk)
    _, g = dense_rates(bd, p, geo, baths(), d, m)
    n = np.random.default_rng(seed).uniform(0.05, 0.95, (2, bd.size))
    dn = host(cur(xp.asarray(n), xp.asarray(bd.w), xp.asarray(g)))
    return float(np.sum(bd.w * dn.sum(0)) / np.max(np.abs(dn)))


def thermal(nk=6, t=0.4):
    """Residual of all equations at the thermal state for equal temperatures."""
    bd, p, geo = setup(nk)
    se = eu.solve_eq(bd, p, t=t, d=0.5, m=-0.5, tol=1.0e-13, prog=False)
    pk = make_prob(bd, p, geo)
    mu = xp.asarray(((se.st.e - se.mu) / t).reshape(-1))
    th = xp.asarray((se.d, se.m))
    r, f, a, b, _ = _res(pk, pack(baths(t, t)), mu, th, 0.0)
    sc = 1.0 / (a / sig(-mu) + b / sig(mu))
    return float(xp.max(xp.abs(sc * r))), float(xp.max(xp.abs(f)))


def eq193(nk=5, seed=2):
    """Balance rows against the literal eq. 193, summed over the baths."""
    bd, p, geo = setup(nk)
    bs = baths()
    pk = make_prob(bd, p, geo)
    rg = np.random.default_rng(seed)
    mu = rg.normal(0.0, 2.0, 2 * bd.size)
    th = np.array([0.6, -0.5])
    r = host(_res(pk, pack(bs), xp.asarray(mu), xp.asarray(th), 0.0)[0])
    st = mf(pk, th[0], th[1])
    e = host(st.e).reshape(-1)
    x = e[:, None] - e[None, :]
    ww = np.tile(bd.w, 2)
    fac = (1.0 + np.exp(mu))[:, None] / (1.0 + np.exp(mu))[None, :]
    ref = np.zeros_like(mu)
    for b in bs:
        g = host(rate_mat(st, pk, pack((b,))))
        ref += np.sum(ww[None, :] * g * fac
                      * np.expm1(x / b.t + mu[None, :] - mu[:, None]), axis=1)
    return float(np.max(np.abs(r - ref)) / np.max(np.abs(ref)))


def jacobian(nk=4, seed=3, h=1.0e-6):
    """Jacobian of solve.jac against central differences of the residual."""
    bd, p, geo = setup(nk)
    pk, bp = make_prob(bd, p, geo), pack(baths())
    rg = np.random.default_rng(seed)
    z = np.concatenate((rg.normal(0.0, 1.5, 2 * bd.size), [0.6, -0.5, 0.05]))
    nm = 2 * bd.size

    def fun(y):
        r, f, _, _, _ = _res(pk, bp, xp.asarray(y[:nm]), xp.asarray(y[nm:nm + 2]), y[-1])
        return np.concatenate((host(r), host(f)))

    J = host(jac(pk, bp, xp.asarray(z[:nm]), xp.asarray(z[nm:nm + 2]), z[-1])[2])
    ref = np.empty_like(J)
    for j in range(z.size):
        e = np.zeros(z.size)
        e[j] = h
        ref[:, j] = (fun(z + e) - fun(z - e)) / (2.0 * h)
    return float(np.max(np.abs(J - ref)) / np.max(np.abs(ref)))


def ness(nk=6, t1=0.6, t2=0.3):
    """Small steady state; general solver against the ph reduction."""
    bd, p, geo = setup(nk)
    bs = baths(t1, t2)
    s = solve_ness(bd, p, geo, bs)
    q = solve_ness(bd, p, geo, bs, ph=True)
    return dict(
        ok=(s.ok, q.ok), err=(s.err, q.err), it=(s.it, q.it), d=s.d, m=s.m,
        lam=s.hist["lam"].tolist(), kap=float(s.hist["kap"]),
        fill=float(np.sum(bd.w * s.n.sum(0))), cur=float(np.max(np.abs(s.cur))),
        ph_mu=float(np.max(np.abs(s.eta[0] + s.eta[1]))),
        ph_d=abs(s.d - q.d), ph_m=abs(s.m - q.m),
    )


def ness_asym(nk=6, t1=0.6, t2=0.3):
    """Steady state without particle hole symmetry (tb != -ta)."""
    bd, p, geo = setup(nk, bp=dict(gap=1.0, ta=1.0, tb=-0.3), mp=dict(v=5.0, n=1.0))
    s = solve_ness(bd, p, geo, baths(t1, t2))
    return dict(
        ok=s.ok, err=s.err, it=s.it, d=s.d, m=s.m, lam=s.hist["lam"].tolist(),
        kap=float(s.hist["kap"]), fill=float(np.sum(bd.w * s.n.sum(0))),
        cur=float(np.max(np.abs(s.cur))),
        ph_mu=float(np.max(np.abs(s.eta[0] + s.eta[1]))),
    )


def evolve(nk=6, t1=0.6, t2=0.3):
    """Newton steady state against the time stepping eu.solve_open, eq. 182."""
    bd, p, geo = setup(nk)
    bs = baths(t1, t2)
    s = solve_ness(bd, p, geo, bs)
    se = eu.solve_eq(bd, p, t=t2, d=0.5, m=-0.5, prog=False)
    o = eu.solve_open(bd, p, se.n, (dissipator(bd, p, geo, bs),), d=se.d, m=se.m,
                      dt=0.05, td=0.3, tm=0.3, tol=1.0e-9, nmax=40000, chk=50,
                      prog=False)
    return dict(ok=o.ok, it=o.it, d=abs(s.d - abs(o.d)), m=abs(s.m - o.m),
                n=float(np.max(np.abs(s.n - o.n))))


def symmetry(nk=6, d=0.6, m=-0.5, dip=0.5):
    """Rate matrix under k -> (-kx, ky) and under kx <-> ky, dipole along x."""
    bd, p, geo = setup(nk, dip=dip)
    _, g = dense_rates(bd, p, geo, baths(), d, m)
    ix = np.arange(nk * nk).reshape(nk, nk)
    mir = ix[(-np.arange(nk)) % nk, :].reshape(-1)
    swp = ix.T.reshape(-1)

    def dev(pm):
        pp = np.concatenate((pm, pm + nk * nk))
        return float(np.max(np.abs(g[np.ix_(pp, pp)] - g)) / np.max(g))

    return dev(mir), dev(swp)


def run():
    print("backend:", NAME)
    print("scalar   (literal, N(-x), xN(0)) :", scalar())
    print("vertex   (brute eq.125, v + d)   :", vertex())
    print("herm     nk=6 (sym, sign of d)   :", herm(6))
    print("herm     nk=5 (sym, sign of d)   :", herm(5))
    print("db       (eq.188: abs, log)      :", db())
    print("number   (relative)              :", number())
    print("thermal  (balance, mean field)   :", thermal())
    print("eq193    (literal)               :", eq193())
    print("jacobian (vs central diff)       :", jacobian())
    print("symmetry (mirror x, swap x y)    :", symmetry())
    print("symmetry dip = 0                 :", symmetry(dip=0.0))
    print("evolve   (vs eu.solve_open)      :", evolve())
    print("ness     ta = -tb :", ness())
    print("ness     tb = -0.3:", ness_asym())


if __name__ == "__main__":
    run()
