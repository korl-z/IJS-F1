import numpy as np
from tqdm.auto import tqdm, trange


def diag(ea, eb, d, m, uh=0.0, n0=1.0):
    """Diagonalize the EI Hamiltonian with cross-Hartree shifts."""
    na = 0.5 * (n0 + m)
    nb = 0.5 * (n0 - m)

    xa = np.asarray(ea, dtype=float) + uh * nb
    xb = np.asarray(eb, dtype=float) + uh * na

    av = 0.5 * (xa + xb)
    z = 0.5 * (xa - xb)
    rr = np.sqrt(z * z + d * d)
    rr = np.maximum(rr, 1.0e-14)

    c = np.clip(z / rr, -1.0, 1.0)
    u = np.sqrt(0.5 * (1.0 + c))
    v = np.copysign(np.sqrt(0.5 * (1.0 - c)), d)

    e = np.stack((av + rr, av - rr))
    return e, u, v


def fd(e, t, n0=1.0):
    """Make a Fermi-Dirac initial state at a fixed total filling."""
    e = np.asarray(e, dtype=float)
    if t <= 0.0:
        raise ValueError("t must be positive")

    lo = float(e.min() - 100.0 * t - 1.0)
    hi = float(e.max() + 100.0 * t + 1.0)

    for _ in range(200):
        mu = 0.5 * (lo + hi)
        y = np.clip((e - mu) / t, -700.0, 700.0)
        n = 1.0 / (np.exp(y) + 1.0)
        if np.mean(np.sum(n, axis=0)) < n0:
            lo = mu
        else:
            hi = mu

    mu = 0.5 * (lo + hi)
    y = np.clip((e - mu) / t, -700.0, 700.0)
    return 1.0 / (np.exp(y) + 1.0)


def sc(n, u, v, g):
    """Return the gap, imbalance, and total-filling targets."""
    d = g * np.mean(u * v * (n[1] - n[0]))
    m = np.mean((u * u - v * v) * (n[0] - n[1]))
    n0 = np.mean(np.sum(n, axis=0))
    return d, m, n0


def mk_gam(gs, st):
    """Add rate tensors returned by one or more gamma callbacks."""
    nk = st["e"].shape[1]
    sh = (2, nk, 2, nk)
    r = np.zeros(sh, dtype=float)

    for f in gs:
        q = np.asarray(f(st), dtype=float)
        if q.shape != sh:
            raise ValueError(f"gamma returned {q.shape}, expected {sh}")
        r += q

    if np.any(~np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("all scattering rates must be finite and nonnegative")

    rf = r.reshape(2 * nk, 2 * nk)
    np.fill_diagonal(rf, 0.0)
    return rf


def cur(n, r):
    """Evaluate the Pauli-blocked occupation current."""
    y = n.reshape(-1)
    gain = (1.0 - y) * (r @ y)
    loss = y * (r.T @ (1.0 - y))
    return (gain - loss).reshape(n.shape)


def lim_dt(n, dn, dt, fac=0.8):
    """Limit an Euler step so all occupations remain between zero and one."""
    z = []
    jp = dn > 0.0
    jm = dn < 0.0

    if np.any(jp):
        z.append(np.min((1.0 - n[jp]) / dn[jp]))
    if np.any(jm):
        z.append(np.min(-n[jm] / dn[jm]))

    if z:
        dt = min(dt, fac * max(0.0, min(z)))
    return dt


def solve(
    x,
    ea,
    eb,
    g,
    n,
    gs,
    d=0.1,
    m=0.0,
    uh=0.0,
    n0=1.0,
    dt=0.02,
    td=1.0,
    tm=1.0,
    nmax=200000,
    tol=1.0e-9,
    chk=100,
    par=None,
):
    """Relax occupations, gap, and imbalance to a coupled steady state.

    Each callback in gs receives st and returns r with shape
    (2, nk, 2, nk). The entry r[a, k, b, p] is the directed rate
    from mode (b, p) to mode (a, k). The branch order is alpha, beta.
    """
    x = np.asarray(x, dtype=float)
    ea = np.asarray(ea, dtype=float)
    eb = np.asarray(eb, dtype=float)
    n = np.asarray(n, dtype=float).copy()

    if ea.shape != eb.shape or ea.shape != x.shape:
        raise ValueError("x, ea, and eb must have the same shape")
    if n.shape != (2, x.size):
        raise ValueError("n must have shape (2, nk)")
    if np.any(n < 0.0) or np.any(n > 1.0):
        raise ValueError("initial occupations must lie between zero and one")
    if abs(np.mean(np.sum(n, axis=0)) - n0) > 1.0e-10:
        raise ValueError("initial occupations do not have the requested filling")
    if dt <= 0.0 or td <= 0.0 or tm <= 0.0:
        raise ValueError("dt, td, and tm must be positive")

    hs = {
        "it": [],
        "t": [],
        "err": [],
        "d": [],
        "m": [],
        "n0": [],
    }
    
    tt = 0.0
    ok = False
    
    bar = trange(1, nmax + 1, desc="EI relaxation")
    
    for it in bar:
        e, u, v = diag(ea, eb, d, m, uh=uh, n0=n0)
        st = {
            "x": x,
            "e": e,
            "u": u,
            "v": v,
            "n": n,
            "d": d,
            "m": m,
            "par": par,
        }
        r = mk_gam(gs, st)
        dn = cur(n, r)
        h = lim_dt(n, dn, dt)

        if h <= 0.0:
            raise RuntimeError("the occupation step collapsed to zero")

        n = n + h * dn
        n = np.clip(n, 0.0, 1.0)

        ds, ms, nt = sc(n, u, v, g)

        rd = ds - d
        rm = ms - m

        zd = -np.expm1(-h / td)
        zm = -np.expm1(-h / tm)

        d += zd * rd
        m += zm * rm

        tt += h
        
        if it == 1 or it % chk == 0:
            er = max(np.max(np.abs(dn)), abs(rd), abs(rm), abs(nt - n0))
            hs["it"].append(it)
            hs["t"].append(tt)
            hs["err"].append(er)
            hs["d"].append(d)
            hs["m"].append(m)
            hs["n0"].append(nt)


            if er < tol:
                ok = True
                break

    e, u, v = diag(ea, eb, d, m, uh=uh, n0=n0)
    st = {
        "x": x,
        "e": e,
        "u": u,
        "v": v,
        "n": n,
        "d": d,
        "m": m,
        "par": par,
    }
    r = mk_gam(gs, st)
    dn = cur(n, r)
    ds, ms, nt = sc(n, u, v, g)
    er = max(np.max(np.abs(dn)), abs(ds - d), abs(ms - m), abs(nt - n0))

    ep = 1.0e-14
    nc = np.clip(n, ep, 1.0 - ep)
    eta = np.log((1.0 - nc) / nc)

    return {
        "ok": ok or er < tol,
        "it": it,
        "err": er,
        "x": x,
        "e": e,
        "u": u,
        "v": v,
        "n": n,
        "eta": eta,
        "d": d,
        "m": m,
        "n0": nt,
        "i": dn,
        "hist": hs,
    }


def gam_db(t, c=1.0, wc=np.inf, band=None):
    if t <= 0.0 or c < 0.0 or wc <= 0.0:
        raise ValueError("t and wc must be positive, and c nonnegative")

    if band not in (None, "a", "b"):
        raise ValueError("band must be None, 'a', or 'b'")

    def gam(st):
        e = st["e"]
        nk = e.shape[1]

        de = e[:, :, None, None] - e[None, None, :, :]

        if np.isinf(wc):
            sp = 1.0
        else:
            sp = np.exp(-(de / wc) ** 2)

        z = -np.logaddexp(0.0, de / t)
        r = (c / nk) * sp * np.exp(z)

        if band is not None:
            u = st["u"]
            v = st["v"]

            if band == "a":
                q = np.stack((u, v))
            else:
                q = np.stack((-v, u))

            a = q[:, :, None, None] * q[None, None, :, :]
            r *= a * a

        return r

    return gam


def gam_scalar(f):
    """Adapt a scalar gamma function to the tensor callback interface.

    The scalar function is called as f(a, k, b, p, st).
    """

    def gam(st):
        nk = st["e"].shape[1]
        r = np.empty((2, nk, 2, nk), dtype=float)

        for a in range(2):
            for k in range(nk):
                for b in range(2):
                    for p in range(nk):
                        r[a, k, b, p] = f(a, k, b, p, st)
        return r

    return gam


if __name__ == "__main__":
    x = np.linspace(-1.0, 1.0, 101)
    ea = x
    eb = -x

    d0 = 0.2
    m0 = 0.0
    e0, _, _ = diag(ea, eb, d0, m0, uh=0.0, n0=1.0)
    n = fd(e0, t=0.08, n0=1.0)

    gs = [
        gam_db(t=0.08, c=0.03, wc=2.0, band="a"),
        gam_db(t=0.25, c=0.03, wc=2.0, band="b"),
    ]

    s = solve(
        x,
        ea,
        eb,
        g=2.5,
        n=n,
        gs=gs,
        d=d0,
        m=m0,
        uh=0.0,
        n0=1.0,
        dt=0.05,
        td=1.0,
        tm=1.0,
        tol=1.0e-8,
    )

    print(f"ok={s['ok']} it={s['it']} err={s['err']:.3e}")
    print(f"d={s['d']:.8f} m={s['m']:.8f} n0={s['n0']:.8f}")
